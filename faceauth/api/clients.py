"""Clients for the internal services: engine (§3), search (§4), KMS (§5) of docs/IMPLEMENTATION.md.

Any failure (connection, timeout, HTTP error, malformed answer) raises Unavailable: the API then answers 503
and never decides on partial data. Engine/search calls try the replicas in random order (same pure call,
so a retry on another replica is not a relaxed path).
"""
import asyncio
import math
import random
import uuid

import httpx
import msgpack
import numpy as np

MSGPACK = {"Content-Type": "application/msgpack"}


class Unavailable(Exception):
    """A dependency (DB, engine, search, KMS) failed or answered garbage -> 503, never a relaxed path."""


class PADModeMismatch(ValueError):
    """API and engine must agree on the explicitly configured pilot mode."""


def _decode(r: httpx.Response):
    r.raise_for_status()
    if "msgpack" in r.headers.get("content-type", ""):
        return msgpack.unpackb(r.content, raw=False)
    return r.json()


class Deps:
    def __init__(self, engine_urls: list, search_urls: list, kms_url: str, kms_token: str, admin_token: str,
                 *, pad_enabled: bool = True, pad_mode: str | None = None,
                 pad_profile_sha256: str = ""):
        self.engine_urls, self.search_urls, self.kms_url = engine_urls, search_urls, kms_url
        self.kms_auth = {"Authorization": f"Bearer {kms_token}"}
        self.admin_auth = {"Authorization": f"Bearer {admin_token}"}
        self.pad_enabled = pad_enabled
        self.pad_mode = pad_mode or ("enabled" if pad_enabled else "disabled_pilot")
        if (self.pad_mode not in ("enabled", "disabled_pilot", "femto_geometry_pilot", "femto_geometry_experimental")
                or pad_enabled is not (self.pad_mode != "disabled_pilot")):
            raise ValueError("inconsistent PAD mode configuration")
        self.pad_profile_sha256 = pad_profile_sha256
        # one pool for engine/search/KMS: the default 100 connections queued requests at ~50 sessions/s (E9)
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=2.0),
                                      limits=httpx.Limits(max_connections=1000, max_keepalive_connections=200))

    async def close(self):
        await self.http.aclose()

    async def _any(self, name: str, urls: list, call):
        last = None
        for url in random.sample(urls, len(urls)):
            try:
                return await call(url)
            except Exception as e:  # noqa: BLE001 - every failure is a dependency failure
                last = e
        raise Unavailable(f"{name}: {last!r}") from last

    async def analyze(self, mode: str, frames: list, calib_version: str, capture_purpose=None) -> dict:
        body = msgpack.packb({"mode": mode, "calib_version": calib_version, "capture_purpose": capture_purpose,
                             "pad_enabled": self.pad_enabled, "pad_mode": self.pad_mode,
                             "frames": [{m: f[m] for m in ("rgb", "depth", "ir")} for f in frames]})

        async def call(url):
            out = _decode(await self.http.post(f"{url}/v1/analyze", content=body, headers=MSGPACK, timeout=60.0))
            self._check_pad_mode(out, require_profile=not (mode == "enroll" or
                                  (mode == "guide" and capture_purpose == "enroll")))
            if not (isinstance(out["versions"]["recog"], str) and int(out["dim"]) > 0
                    and len(out["frames"]) == len(frames) and all(isinstance(f, dict) for f in out["frames"])
                    and out.get("calib_version") == calib_version
                    and (mode not in ("guide", "observe", "live") or out.get("capture_purpose") == capture_purpose)):
                raise ValueError("malformed engine answer")
            if mode == "live":
                for frame in out["frames"]:
                    faces, size = frame.get("faces"), frame.get("image_size")
                    if (not isinstance(faces, list) or len(faces) > 16
                            or not isinstance(size, list) or len(size) != 2
                            or not all(isinstance(v, int) and v > 0 for v in size)
                            or not isinstance(frame.get("n_faces"), int) or frame["n_faces"] < len(faces)
                            or not isinstance(frame.get("truncated"), bool)):
                        raise ValueError("malformed live engine answer")
                    for face in faces:
                        box = face.get("bbox") if isinstance(face, dict) else None
                        if (not isinstance(box, list) or len(box) != 4
                                or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in box)
                                or box[2] <= box[0] or box[3] <= box[1]):
                            raise ValueError("malformed live face box")
            return out
        return await self._any("engine", self.engine_urls, call)

    def _check_pad_mode(self, out, *, require_profile=True):
        if out.get("pad_enabled") is not self.pad_enabled or out.get("pad_mode") != self.pad_mode:
            raise PADModeMismatch("pad_mode_mismatch")
        profile_key = "pad_experiment" if self.pad_mode == "femto_geometry_experimental" else "pad_pilot_validation"
        if require_profile and self.pad_mode in ("femto_geometry_pilot", "femto_geometry_experimental") and self.pad_profile_sha256 and (
                (out.get(profile_key) or {}).get("profile_sha256") != self.pad_profile_sha256):
            raise PADModeMismatch("pad_profile_mismatch")

    async def readiness(self, calib_version: str, active_recog: str | None) -> dict:
        async def engine(url):
            out = _decode(await self.http.get(f"{url}/v1/readiness", params={"calib_version": calib_version}, timeout=3))
            # An unavailable PAD profile blocks authentication below, but must
            # not make a functioning recognition engine unavailable to enrollment.
            self._check_pad_mode(out, require_profile=False)
            if (not active_recog or out.get("capture_ready") is not True
                    or out.get("versions", {}).get("recog") != active_recog
                    or out.get("calib_version") != calib_version):
                raise ValueError("engine not ready or recognition model differs")
            return out

        try:
            out = await self._any("engine", self.engine_urls, engine)
        except Unavailable as exc:
            mismatch = isinstance(exc.__cause__, PADModeMismatch)
            return {"ready": False, "enrollment_ready": False, "capture_ready": False, "pad_validation": None,
                    "enrollment_pad_enabled": False,
                    "pad_enabled": self.pad_enabled, "pad_mode": self.pad_mode,
                    "blockers": [{"code": "pad_mode_mismatch" if mismatch else "engine_unavailable",
                                  "message": "API와 분석 엔진의 PAD 설정이 다릅니다." if mismatch
                                  else "GPU 분석 엔진이 준비되지 않았습니다."}]}
        status_key = ("pad_experiment" if self.pad_mode == "femto_geometry_experimental" else
                      "pad_pilot_validation" if self.pad_mode == "femto_geometry_pilot" else "pad_validation")
        validation = out.get(status_key) or {}
        failure_message = ("실험용 깊이 PAD 설정을 확인할 수 없습니다." if self.pad_mode == "femto_geometry_experimental"
                           else "이 센서의 위조 방지 검증이 완료되지 않았습니다.")
        blockers = [{"code": code, "message": failure_message}
                    for code in validation.get("reasons", [])] if self.pad_enabled else []
        if self.pad_enabled and validation.get("ready") is not True and not blockers:
            blockers.append({"code": "pad_validation_required", "message": "위조 방지 검증 상태를 확인할 수 없습니다."})
        if self.pad_mode == "femto_geometry_pilot" and (
                len(self.pad_profile_sha256) != 64 or validation.get("experimental") is not True
                or validation.get("scope") != ["print", "replay"]
                or validation.get("calib_version") != calib_version
                or validation.get("profile_sha256") != self.pad_profile_sha256):
            blockers.append({"code": "pad_pilot_profile_invalid", "message": "깊이 PAD 시험 설정을 확인할 수 없습니다."})
        if self.pad_mode == "femto_geometry_experimental" and (
                len(self.pad_profile_sha256) != 64 or validation.get("experimental") is not True
                or validation.get("production_validated") is not False
                or validation.get("empirical_validation") != "not_evaluated"
                or validation.get("scope") != ["print", "replay"]
                or validation.get("calib_version") != calib_version
                or validation.get("profile_sha256") != self.pad_profile_sha256):
            blockers.append({"code": "pad_experiment_profile_invalid", "message": failure_message})

        async def search(url):
            state = _decode(await self.http.get(f"{url}/v1/status", timeout=3))
            if state.get("ready") is not True or state.get("model_version") != active_recog:
                raise ValueError("search not ready or recognition model differs")
            return state

        enrollment_blockers = []
        try:
            await self._any("search", self.search_urls, search)
        except Unavailable:
            enrollment_blockers.append({"code": "search_unavailable", "message": "얼굴 검색 서비스가 준비되지 않았습니다."})
            blockers.extend(enrollment_blockers)
        result = {**out, "ready": not blockers, "blockers": blockers,
                  "enrollment_ready": not enrollment_blockers, "enrollment_blockers": enrollment_blockers,
                  "enrollment_pad_enabled": False}
        if self.pad_mode in ("femto_geometry_pilot", "femto_geometry_experimental"):
            result["pad_profile_sha256"] = self.pad_profile_sha256
        return result

    async def search(self, model_version: str, probes: np.ndarray, k: int = 5) -> list:
        body = msgpack.packb({"model_version": model_version, "probes": probes.astype("<f4").tobytes(),
                              "n": int(len(probes)), "k": k})

        async def call(url):
            out = _decode(await self.http.post(f"{url}/v1/search", content=body, headers=MSGPACK, timeout=5.0))
            res = [{"subject_id": uuid.UUID(str(r["subject_id"])), "score": float(r["score"])} for r in out["results"]]
            if not all(math.isfinite(r["score"]) for r in res):
                raise ValueError("non-finite score")
            return res
        return await self._any("search", self.search_urls, call)

    async def encrypt(self, subject_id: uuid.UUID, items: list) -> list:
        try:
            r = await self.http.post(f"{self.kms_url}/v1/encrypt", headers={**MSGPACK, **self.kms_auth}, timeout=5.0,
                                     content=msgpack.packb({"subject_id": str(subject_id), "items": items}))
            out = _decode(r)["items"]
            if len(out) != len(items) or not all(isinstance(c, bytes) and len(c) > 12 for c in out):
                raise ValueError("malformed kms answer")
            return out
        except Exception as e:  # noqa: BLE001
            raise Unavailable(f"kms: {e!r}") from e

    async def destroy_key(self, subject_id: uuid.UUID):
        """-> KMS answer {"destroyed": bool} or None if the KMS could not be reached / failed."""
        try:
            return _decode(await self.http.delete(f"{self.kms_url}/v1/keys/{subject_id}", headers=self.kms_auth,
                                                  timeout=5.0))
        except Exception:  # noqa: BLE001
            return None

    async def applied_event_ids(self) -> list:
        """applied_event_id of every search replica (None for a replica that does not answer)."""
        async def one(url):
            try:
                return int(_decode(await self.http.get(f"{url}/v1/status", timeout=2.0))["applied_event_id"])
            except Exception:  # noqa: BLE001
                return None
        return list(await asyncio.gather(*(one(u) for u in self.search_urls)))

    async def reload(self, urls: list) -> list:
        """POST /v1/admin/reload on each replica, one after the other (rolling) -> [ok per replica]."""
        out = []
        for url in urls:
            try:
                r = await self.http.post(f"{url}/v1/admin/reload", headers=self.admin_auth, timeout=120.0)
                r.raise_for_status()
                out.append(True)
            except Exception:  # noqa: BLE001
                out.append(False)
        return out
