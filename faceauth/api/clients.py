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


def _decode(r: httpx.Response):
    r.raise_for_status()
    if "msgpack" in r.headers.get("content-type", ""):
        return msgpack.unpackb(r.content, raw=False)
    return r.json()


class Deps:
    def __init__(self, engine_urls: list, search_urls: list, kms_url: str, kms_token: str, admin_token: str):
        self.engine_urls, self.search_urls, self.kms_url = engine_urls, search_urls, kms_url
        self.kms_auth = {"Authorization": f"Bearer {kms_token}"}
        self.admin_auth = {"Authorization": f"Bearer {admin_token}"}
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=2.0))

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

    async def analyze(self, mode: str, frames: list) -> dict:
        body = msgpack.packb({"mode": mode, "frames": [{m: f[m] for m in ("rgb", "depth", "ir")} for f in frames]})

        async def call(url):
            out = _decode(await self.http.post(f"{url}/v1/analyze", content=body, headers=MSGPACK, timeout=60.0))
            if not (isinstance(out["versions"]["recog"], str) and int(out["dim"]) > 0
                    and len(out["frames"]) == len(frames) and all(isinstance(f, dict) for f in out["frames"])):
                raise ValueError("malformed engine answer")
            return out
        return await self._any("engine", self.engine_urls, call)

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
