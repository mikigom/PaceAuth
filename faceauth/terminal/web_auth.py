"""LAN pilot accounts and single-camera enrollment/authentication operations.

The browser never receives signing keys, account assertions, embeddings or
approval bearer tokens. The explicitly configured LAN console uses anonymous
operator sessions; password login remains the default. Final decisions come
from the existing mTLS API. Raw streams remain in memory; one enrollment
thumbnail is encrypted at rest only with separate photo consent.
"""
import datetime
import hashlib
import hmac
import json
import logging
import os
import secrets
import threading
import time
import unicodedata
from collections import OrderedDict
from pathlib import Path

import httpx
from cryptography.hazmat.primitives import serialization
from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field, field_validator

from faceauth.common import token
from faceauth.common.packet import encode_frame
from faceauth.terminal.client import DeviceClient, make_account_assertion
from faceauth.terminal.guide import Guide, HINTS
from faceauth.terminal.accounts import Accounts, SESSION_SECONDS, _password_hash

LOG = logging.getLogger(__name__)
# Uvicorn configures its own loggers, leaving application INFO records hidden by default.
# This dedicated stream contains only aggregate capture counts and timings, never identity or pose data.
GUIDE_LOG = logging.getLogger("faceauth.enrollment.metrics")
if not GUIDE_LOG.handlers:
    guide_handler = logging.StreamHandler()
    guide_handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    GUIDE_LOG.addHandler(guide_handler)
GUIDE_LOG.setLevel(logging.INFO)
GUIDE_LOG.propagate = False
COOKIE = "paceauth_pilot"
CONSOLE_COOKIE = "paceauth_console"
CONSENT_VERSION = "v1"
CONSENT_TEXT = (
    "카메라 앞에 있는 본인의 동의를 받아 얼굴 등록·인증을 진행합니다. 연속 영상은 저장하지 않으며, "
    "등록한 얼굴 특징은 암호화하여 서버에 저장합니다. 등록 사진 저장에 별도로 동의한 경우에만 "
    "얼굴 등록 때의 정면 사진 1장을 축소·암호화해 저장하고, 관리자가 조회·삭제할 수 있습니다. 센서 검증을 선택하면 점수·품질 수치와 익명 대상 코드를 "
    "검증 기록으로 저장합니다. 안내 시간 안에 촬영을 마치지 못한 경우도 비응답으로 기록하며, "
    "직접 취소한 촬영은 검증 기록에 저장하지 않습니다. 이 서비스의 별도 테스트 계정만 사용합니다."
)
FINAL_STATES = {"succeeded", "rejected", "error", "cancelled"}


class Credentials(BaseModel):
    account_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]{2,63}$")
    password: str = Field(min_length=10, max_length=128)


class Profile(BaseModel):
    display_name: str = Field(min_length=1, max_length=80)

    @field_validator("display_name")
    @classmethod
    def clean_name(cls, value):
        value = unicodedata.normalize("NFC", value.strip())
        if not value or any(unicodedata.category(c).startswith("C") for c in value):
            raise ValueError("표시 이름에 빈 이름이나 제어 문자를 사용할 수 없습니다.")
        return value


class Signup(Credentials):
    display_name: str | None = Field(default=None, max_length=80)

    @field_validator("display_name")
    @classmethod
    def clean_name(cls, value):
        return Profile.clean_name(value) if value is not None else None


class ManagedCreation(BaseModel):
    account_id: str | None = Field(default=None, pattern=r"^[a-z0-9][a-z0-9_.-]{2,63}$")
    display_name: str | None = Field(default=None, max_length=80)
    password: str | None = Field(default=None, min_length=10, max_length=128)

    @field_validator("display_name")
    @classmethod
    def clean_name(cls, value):
        return Profile.clean_name(value) if value is not None else None


class Consent(BaseModel):
    version: str
    agreed: bool


class CaptureRequest(BaseModel):
    consent: Consent
    photo_consent: bool = False
    re_enroll: bool = False
    guide_mode: str = Field(default="steps", pattern=r"^(steps|circle)$")
    client_request_id: str | None = Field(default=None, min_length=8, max_length=64,
                                         pattern=r"^[a-zA-Z0-9_-]+$")


class ObservationRequest(CaptureRequest):
    target_account_id: str | None = Field(default=None, pattern=r"^[a-z0-9][a-z0-9_.-]{2,63}$")
    observation_mode: str = Field(default="diagnostic", pattern=r"^(diagnostic|auth|enroll)$")
    kind: str = Field(pattern=r"^(live|print|replay|mask_rigid|mask_silicone)$")
    split: str = Field(pattern=r"^(calibration|test)$")
    subject_id: str = Field(pattern=r"^[a-zA-Z0-9_.-]{1,64}$")
    instrument_id: str = Field(default="", pattern=r"^[a-zA-Z0-9_.-]{0,64}$")
    conditions: str = Field(default="", max_length=300)


class CancelRequest(BaseModel):
    id: str


class AccountStatus(BaseModel):
    enabled: bool


class AccountDeletion(BaseModel):
    confirm_account_id: str


class CaptureCancelled(Exception):
    pass


class LiveAuth:
    def __init__(self, camera, *, enabled=None, state_dir=None, client_factory=None, open_console=None):
        self.camera = camera
        self.enabled = (os.getenv("FA_LIVE_AUTH_ENABLED", "0") == "1") if enabled is None else enabled
        self.open_console = self.enabled and (
            os.getenv("FA_LIVE_OPEN_CONSOLE", "0") == "1" if open_console is None else bool(open_console))
        self._console_key = secrets.token_bytes(32)
        self.state_dir = Path(state_dir or os.getenv("FA_LIVE_STATE_DIR", "/var/lib/paceauth"))
        self.secrets_dir = Path(os.getenv("FA_TERMINAL_SECRETS", "/app/secrets"))
        self.device_id = os.getenv("FA_TERMINAL_DEVICE", "dev-001")
        self.gateway = os.getenv("FA_TERMINAL_GATEWAY", "https://gateway:8443")
        self.client_factory = client_factory or (
            lambda calib: DeviceClient(self.gateway, self.device_id, secrets_dir=self.secrets_dir,
                                       calib_version=calib, timeout=15))
        self.accounts = Accounts(self.state_dir / "accounts.sqlite3") if self.enabled else None
        self.lock = threading.RLock()
        self.management_lock = threading.RLock()
        self.stop_event = threading.Event()
        self.poll_thread = self.operation_thread = None
        self.operations = OrderedDict()
        self.active = None
        self.backend = {"ready": False, "capture_ready": False, "blockers": [
            {"code": "backend_starting", "message": "인증 서버의 준비 상태를 확인하고 있습니다."}]}
        self.backend_at = 0.0

    def console_session(self, secret):
        """Validate a browser operator session without creating a user identity."""
        if not self.open_console or not isinstance(secret, str) or len(secret) > 128:
            return False
        try:
            stamp, nonce, signature = secret.split(".")
            issued = int(stamp)
            if not 0 <= time.time() - issued < SESSION_SECONDS or len(nonce) != 32:
                return False
            expected = hmac.new(self._console_key, f"{stamp}.{nonce}".encode(), hashlib.sha256).hexdigest()
            return hmac.compare_digest(signature, expected)
        except (ValueError, TypeError):
            return False

    def browser_secret(self, request):
        return request.cookies.get(CONSOLE_COOKIE if self.open_console else COOKIE)

    def status_for_browser(self, request, response):
        secret = self.browser_secret(request)
        if self.open_console and not self.console_session(secret):
            # Only same-origin browser status requests bootstrap the operator.
            origin = request.headers.get("origin")
            expected = f"{request.url.scheme}://{request.url.netloc}"
            if request.headers.get("sec-fetch-site") == "cross-site" or (origin and origin != expected):
                raise HTTPException(403, "다른 사이트에서 보낸 요청은 허용하지 않습니다.")
            payload = f"{int(time.time())}.{secrets.token_hex(16)}"
            signature = hmac.new(self._console_key, payload.encode(), hashlib.sha256).hexdigest()
            secret = f"{payload}.{signature}"
            response.set_cookie(CONSOLE_COOKIE, secret, max_age=SESSION_SECONDS, httponly=True,
                                samesite="strict", secure=request.url.scheme == "https", path="/")
        return self.status(secret)

    def start(self):
        if self.enabled:
            self.poll_thread = threading.Thread(target=self._poll, daemon=True, name="auth-readiness")
            self.poll_thread.start()

    def stop(self):
        self.stop_event.set()
        with self.lock:
            if self.active:
                self.operations[self.active]["cancel"].set()
        for thread in (self.operation_thread, self.poll_thread):
            if thread:
                thread.join(timeout=16)

    def _poll(self):
        while not self.stop_event.is_set():
            client = None
            try:
                calib = self.camera.status()["camera"]["info"].get("calib_version", "unavailable")
                client = self.client_factory(calib)
                status = client.readiness()
                if not isinstance(status, dict) or not isinstance(status.get("blockers"), list):
                    raise ValueError("Malformed readiness response")
            except Exception:
                status = {"ready": False, "capture_ready": False, "blockers": [
                    {"code": "backend_unavailable", "message": "인증 서버에 연결할 수 없습니다. 운영자 진단을 확인하세요."}]}
            finally:
                if client:
                    client.close()
            with self.lock:
                self.backend, self.backend_at = status, time.monotonic()
            self.stop_event.wait(3)

    def status(self, secret=None):
        account = self.accounts.account(secret) if self.accounts and not self.open_console else None
        console_ready = self.console_session(secret)
        with self.lock:
            backend = dict(self.backend)
            recent = time.monotonic() - self.backend_at < 20
            own = next((o for o in reversed(self.operations.values())
                        if secret and o["owner"] == Accounts.digest(secret)), None)
        camera = self.camera.status()
        calib = camera["camera"]["info"].get("calib_version")
        camera_ready = camera["camera"]["state"] == "streaming"
        same_calib = bool(calib and calib == backend.get("calib_version"))
        blockers = list(backend.get("blockers", [])) if self.enabled else [
            {"code": "auth_disabled", "message": "웹 등록·인증 기능이 설정되지 않았습니다."}]
        if self.enabled and not recent:
            blockers.append({"code": "readiness_stale", "message": "인증 서버의 최신 상태를 기다리고 있습니다."})
        if not camera_ready:
            blockers.append({"code": "camera_unavailable", "message": "카메라 연결을 확인해 주세요."})
        elif self.enabled and recent and not same_calib:
            blockers.append({"code": "calib_mismatch", "message": "센서 설정과 서버에 등록된 보정 정보가 다릅니다."})
        common = self.enabled and recent and camera_ready and same_calib
        return {"enabled": self.enabled, "pilot": True,
                "console_mode": self.open_console, "console_ready": console_ready,
                "console_session_id": Accounts.digest(secret)[:16] if console_ready else None,
                "management_allowed": console_ready if self.open_console else bool(
                    self.accounts and self.accounts.profile(account).get("is_admin")),
                "ready": bool(common and backend.get("ready") is True and not blockers),
                "capture_ready": bool(common and backend.get("capture_ready") is True),
                "pad_enabled": backend.get("pad_enabled", True),
                "pad_mode": backend.get("pad_mode", "enabled"),
                "blockers": blockers, "pad_validation": backend.get("pad_validation"),
                "account": self.accounts.profile(account) if self.accounts else {
                    "authenticated": False, "account_id": None, "display_name": None},
                "consent": {"version": CONSENT_VERSION, "text": CONSENT_TEXT.replace(
                    "관리자가 조회·삭제할 수 있습니다", "이 페이지를 사용하는 사람이 조회·삭제할 수 있습니다"
                ) if self.open_console else CONSENT_TEXT},
                "operation_id": own["id"] if own else None}

    @staticmethod
    def public(operation):
        result = {k: operation[k] for k in ("id", "kind", "state", "message", "guide", "result", "client_request_id",
                                          "observation_mode", "pad_enabled", "pad_mode", "target_account_id",
                                          "target_display_name", "guide_latency_ms", "guide_diagnostics") if k in operation}
        sample_at = operation.get("_guide_sample_at")
        # Sample start, rather than response completion: a slow inference must not look like a fresh pose.
        result["guide_age_ms"] = round(max(0.0, time.monotonic() - sample_at) * 1000, 1) if sample_at is not None else None
        return result

    def operation(self, secret, operation_id=None, client_request_id=None):
        owner = Accounts.digest(secret)
        with self.lock:
            op = self.operations.get(operation_id) if operation_id else next(
                (o for o in reversed(self.operations.values()) if o["owner"] == owner
                 and (client_request_id is None or o["client_request_id"] == client_request_id)), None)
            if op is not None and op["owner"] != owner:
                raise HTTPException(404, "작업을 찾을 수 없습니다.")
            if operation_id and op is None:
                raise HTTPException(404, "작업을 찾을 수 없습니다.")
            return self.public(op) if op else None

    def _management_call(self, method, *args):
        client = None
        try:
            calib = self.camera.status()["camera"]["info"].get("calib_version", "unavailable")
            client = self.client_factory(calib)
            result = getattr(client, method)(*args)
            if not isinstance(result, dict):
                raise ValueError("malformed_management_response")
            if method == "account_summary":
                rows = result.get("accounts")
                if (not isinstance(rows, list) or len(rows) != len(set(args[0]))
                        or any(not isinstance(row, dict) or not isinstance(row.get("account_id"), str)
                               or not isinstance(row.get("enabled"), bool)
                               or row.get("face_status") not in
                               ("none", "active", "pending_review", "disabled", "revoked") for row in rows)
                        or {row.get("account_id") for row in rows} != set(args[0])):
                    raise ValueError("malformed_account_summary")
            elif result.get("account_id") != args[0]:
                raise ValueError("management_account_mismatch")
            elif method == "account_status" and result.get("enabled") is not args[1]:
                raise ValueError("management_status_mismatch")
            elif method == "delete_account_face" and any(
                    not isinstance(result.get(key), bool) for key in ("revoked", "key_destroyed", "index_removed")):
                raise ValueError("malformed_deletion_response")
            return result
        except (OSError, RuntimeError, httpx.HTTPError, ValueError) as exc:
            LOG.warning("Account management backend unavailable: %s", type(exc).__name__)
            raise HTTPException(503, "계정 관리 서버의 응답을 확인하지 못했습니다. 잠시 후 다시 시도해 주세요.") from None
        finally:
            if client:
                client.close()

    def _management_idle(self):
        with self.lock:
            if self.active:
                raise HTTPException(409, "촬영이 끝난 뒤 계정을 변경해 주세요.")

    def managed_accounts(self):
        accounts = self.accounts.list_accounts()
        try:
            remote = {}
            for start in range(0, len(accounts), 500):
                result = self._management_call("account_summary", [a["account_id"] for a in accounts[start:start + 500]])
                remote.update({row["account_id"]: row for row in result["accounts"]})
            for account in accounts:
                row = remote.get(account["account_id"], {})
                account.update(face_status=row.get("face_status", "unknown"), enrolled_at=row.get("enrolled_at"))
                account["enabled"] = account["enabled"] and row.get("enabled", True)
            return {"accounts": accounts, "management_ready": True, "message": ""}
        except HTTPException as exc:
            return {"accounts": [{**a, "face_status": "unknown", "enrolled_at": None} for a in accounts],
                    "management_ready": False, "message": exc.detail}

    def managed_account(self, account_id):
        return next((a for a in self.managed_accounts()["accounts"] if a["account_id"] == account_id), None)

    def create_account(self, body):
        with self.management_lock:
            if self.accounts.get(body.account_id):
                raise HTTPException(409, "이미 사용 중인 테스트 계정입니다.")
            # A deleted ID can be reused, but never attach a new password to an
            # orphaned, still-enrolled identity from another account database.
            rows = self._management_call("account_summary", [body.account_id])["accounts"]
            if not rows or rows[0].get("face_status") not in ("none", "revoked"):
                raise HTTPException(409, "이미 얼굴이 등록된 계정 ID입니다. 다른 ID를 사용해 주세요.")
            self._management_call("account_status", body.account_id, True)
            return self.accounts.create(body.account_id, body.password, body.display_name)

    def change_account_status(self, account_id, enabled, actor):
        with self.management_lock:
            self._management_idle()
            if not self.accounts.get(account_id):
                raise HTTPException(404, "계정을 찾을 수 없습니다.")
            if account_id == actor and not enabled:
                raise HTTPException(409, "현재 로그인한 관리자는 비활성화할 수 없습니다.")
            # Central eligibility commits first; stale search results cannot approve
            # a disabled identity. Local sessions are then revoked by the store.
            self._management_call("account_status", account_id, enabled)
            self.accounts.set_enabled(account_id, enabled, allow_last_admin=self.open_console)
            return self.managed_account(account_id)

    def delete_account(self, account_id, actor):
        with self.management_lock:
            self._management_idle()
            if account_id == actor:
                raise HTTPException(409, "현재 로그인한 관리자 계정은 삭제할 수 없습니다.")
            if not self.accounts.get(account_id):
                raise HTTPException(404, "계정을 찾을 수 없습니다.")
            self._management_call("account_status", account_id, False)
            self.accounts.set_enabled(account_id, False, allow_last_admin=self.open_console)
            result = self._management_call("delete_account_face", account_id)
            if not all(result.get(k) is True for k in ("revoked", "key_destroyed", "index_removed")):
                raise HTTPException(503, "계정은 비활성화했습니다. 얼굴 정보 삭제 반영을 확인한 뒤 다시 삭제해 주세요.")
            self.accounts.delete(account_id, allow_last_admin=self.open_console)
            with self.lock:
                for key in [key for key, op in self.operations.items() if op["account"] == account_id]:
                    self.operations.pop(key)
            return {"deleted": True, "account_id": account_id}

    def begin(self, kind, body, secret, account):
        if body.consent.agreed is not True or body.consent.version != CONSENT_VERSION:
            raise HTTPException(400, "현재 동의문을 확인하고 동의해 주세요.")
        if kind == "observe" and body.kind != "live" and not body.instrument_id:
            raise HTTPException(400, "사진·화면·가면을 구분하는 공격 매체 코드를 입력해 주세요.")
        readiness = self.status(secret)
        required = "capture_ready" if kind == "observe" else "ready"
        if not readiness[required]:
            raise HTTPException(503, "촬영·인증 준비 항목이 완료되지 않았습니다.")
        with self.management_lock, self.lock:
            console = self.console_session(secret)
            actor = None if self.open_console else self.accounts.account(secret)
            target = self.accounts.get(account)
            if (not console and not actor) or not target or not target["enabled"]:
                raise HTTPException(403, "활성 계정만 얼굴을 등록하거나 인증할 수 있습니다.")
            if not console and actor != account and (kind != "enroll" or not self.accounts.profile(actor)["is_admin"]):
                raise HTTPException(403, "다른 계정의 얼굴 등록은 관리자만 할 수 있습니다.")
            fingerprint = hashlib.sha256(body.model_dump_json().encode()).hexdigest()
            if body.client_request_id:
                previous = next((o for o in self.operations.values()
                                 if o["owner"] == Accounts.digest(secret)
                                 and o["client_request_id"] == body.client_request_id), None)
                if previous:
                    if (previous["kind"] != kind or previous["request_hash"] != fingerprint
                            or previous["account"] != account):
                        raise HTTPException(409, "동일한 요청 번호로 다른 작업을 시작할 수 없습니다.")
                    return self.public(previous)
            if self.active:
                raise HTTPException(409, "다른 촬영이 진행 중입니다. 완료 후 다시 시도해 주세요.")
            op = {"id": secrets.token_urlsafe(18), "kind": kind, "state": "capturing",
                  "observation_mode": body.observation_mode if kind == "observe" else None,
                  "pad_enabled": readiness["pad_enabled"], "pad_mode": readiness["pad_mode"],
                  "message": "카메라에서 50~80cm 떨어져 정면을 바라봐 주세요.",
                  "guide": None, "result": None, "owner": Accounts.digest(secret), "account": account,
                  "actor": actor, "console": console, "target_account_id": account, "target_display_name": target["display_name"],
                  "client_request_id": body.client_request_id, "request_hash": fingerprint,
                  "cancel": threading.Event(), "consent_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
            self.operations[op["id"]] = op
            while len(self.operations) > 32:
                self.operations.popitem(last=False)
            self.active = op["id"]
            self.operation_thread = threading.Thread(target=self._run, args=(op, body, secret), daemon=True,
                                                     name="face-operation")
            self.operation_thread.start()
            return self.public(op)

    def cancel(self, secret, operation_id):
        with self.lock:
            self.operation(secret, operation_id)  # owner check
            op = self.operations[operation_id]
            if op["state"] == "processing":
                raise HTTPException(409, "이미 서버에 제출한 요청의 결과를 기다리고 있습니다.")
            if op["state"] not in FINAL_STATES:
                op["cancel"].set()
            return self.public(op)

    def _update(self, op, **fields):
        with self.lock:
            op.update(fields)

    def _check(self, op, secret):
        actor = self.accounts.account(secret) if not op.get("console") else None
        target = self.accounts.get(op["account"])
        if (op["cancel"].is_set() or self.stop_event.is_set() or actor != op.get("actor", op["account"])
                or not target or not target["enabled"]
                or (op.get("console") and not self.console_session(secret))
                or (not op.get("console") and actor != op["account"] and not self.accounts.profile(actor).get("is_admin"))):
            raise CaptureCancelled

    def _frames(self, op, secret, session, timeout):
        # Establish the barrier AFTER receiving the unpredictable API challenge.
        after = self.camera.capture_sequence()
        not_before = time.monotonic() + 0.2
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._check(op, secret)
            capture = self.camera.capture_after(after, timeout=1)
            if capture is None:
                if self.camera.status()["camera"]["state"] != "streaming":
                    raise RuntimeError("camera_unavailable")
                continue
            after, arrived, frame = capture
            if arrived < not_before:
                continue
            yield frame

    def _run(self, op, body, secret):
        client = None
        observation_context = None
        guide_attempts = 0
        guide_diagnostics = {"samples": 0, "quality_rejected": 0, "pose_unmatched": 0, "reasons": {},
                             "collected_frames": 0, "completed_directions": 0,
                             "latency_mean_ms": 0.0, "latency_max_ms": 0.0}
        latency_total_ms = 0.0
        try:
            calib = self.camera.status()["camera"]["info"]["calib_version"]
            client = self.client_factory(calib)
            observing = op["kind"] == "observe"
            if observing:
                initial = client.readiness()
                binding = (initial.get("pad_validation") or {}).get("binding")
                if (initial.get("capture_ready") is not True or initial.get("calib_version") != calib
                        or not isinstance(binding, dict) or not binding):
                    raise RuntimeError("readiness_changed")
                # Snapshot the engine actually available before capture. A model
                # reload must not relabel an earlier failed attempt as new data.
                observation_context = {"calib_version": calib, "model_binding": json.loads(json.dumps(binding))}
            capture_mode = body.observation_mode if observing else ("enroll" if op["kind"] == "enroll" else "auth")
            purpose = "enroll" if capture_mode == "enroll" else "auth"
            action = "enroll" if purpose == "enroll" else "pilot-check"
            session = client.create_session(purpose, action)
            if purpose == "enroll":
                guide = Guide(timeout_s=90, mode=body.guide_mode)
                refs = []
            else:
                guide = None
                refs = []
            timeout = 90 if guide else 15
            last_sample = 0.0
            for frame in self._frames(op, secret, session, timeout):
                if frame["calib_version"] != calib:
                    raise RuntimeError("calib_changed")
                now = time.monotonic()
                interval = 0.2 if capture_mode == "diagnostic" else 1 / 15 if guide else 0.1
                if now - last_sample < interval:
                    continue
                last_sample = now
                encoded = encode_frame(frame["rgb"], frame["depth"], frame["ir"])
                ref = {"image": encoded, "ts": dict(frame["ts_ms"]), "captured_at_ms": frame["captured_at_ms"]}
                if capture_mode == "diagnostic":
                    refs.append(ref)
                    self._update(op, message=f"검증 표본 촬영 중 · {len(refs)}/5",
                                 guide={"prompt": "정면을 바라봐 주세요", "ring_frac": len(refs) / 5})
                    if len(refs) >= 5:
                        break
                    continue
                analysis = client.capture_guide(session, encoded, ref["ts"], purpose=purpose)["frame"]
                guide_attempts += 1
                analyzed_at = time.monotonic()
                latency_ms = max(0.0, analyzed_at - now) * 1000
                latency_total_ms += latency_ms
                guide_diagnostics["samples"] += 1
                guide_diagnostics["latency_mean_ms"] = round(latency_total_ms / guide_attempts, 1)
                guide_diagnostics["latency_max_ms"] = round(max(guide_diagnostics["latency_max_ms"], latency_ms), 1)
                if guide:
                    state = guide.feed(analysis["pose"], analysis["ok"], ref, analysis["score"],
                                       t=analyzed_at, reasons=analysis["reasons"])
                    guide_diagnostics["collected_frames"] = sum(min(have, need) for have, need in state["coverage"].values())
                    guide_diagnostics["completed_directions"] = sum(have >= need for have, need in state["coverage"].values())
                    if not state["quality_ok"]:
                        guide_diagnostics["quality_rejected"] += 1
                    elif state["reason_code"] in ("between_directions", "pose_margin"):
                        guide_diagnostics["pose_unmatched"] += 1
                    reasons = {reason if reason in HINTS else "quality_unknown" for reason in analysis["reasons"]}
                    if state["reason_code"]:
                        reasons.add(state["reason_code"])
                    for reason in reasons:
                        counts = guide_diagnostics["reasons"]
                        counts[reason] = counts.get(reason, 0) + 1
                    self._update(op, guide=state, message=state["prompt"], _guide_sample_at=now,
                                 guide_latency_ms=round(latency_ms, 1),
                                 guide_diagnostics={**guide_diagnostics, "reasons": dict(guide_diagnostics["reasons"])})
                    if state["done"] or state["timed_out"]:
                        break
                else:
                    # Only retain near-frontal candidates, not a whole video.
                    pose = analysis.get("pose")
                    if analysis["ok"] and pose and abs(pose[0]) <= 20 and abs(pose[1]) <= 15:
                        refs.append(ref)
                    hint = next((HINTS[r] for r in analysis["reasons"] if r in HINTS), "정면을 바라봐 주세요")
                    self._update(op, message=hint, guide={"prompt": hint, "ring_frac": len(refs) / 3},
                                 _guide_sample_at=now, guide_latency_ms=round(latency_ms, 1))
                    if len(refs) >= 3:
                        break
            self._check(op, secret)
            if guide:
                if not guide.done:
                    raise RuntimeError("capture_incomplete")
                kept = guide.frames()
                refs = [ref for _, ref in kept]
                labels = [label for label, _ in kept]
            elif len(refs) < (5 if capture_mode == "diagnostic" else 3):
                raise RuntimeError("capture_incomplete")
            # Recheck readiness and serialize cancellation with final submission.
            current = self.status(secret)
            if not current["capture_ready" if op["kind"] == "observe" else "ready"]:
                raise RuntimeError("readiness_changed")
            with self.lock:
                self._check(op, secret)
                op.update(state="processing", message="서버에서 촬영 결과를 확인하고 있습니다.")
            images, timestamps = [r["image"] for r in refs], [r["ts"] for r in refs]
            captured_at = refs[0]["captured_at_ms"]
            if observing:
                out = client.observe(session, images, timestamps, captured_at_ms=captured_at, purpose=purpose)
                if (out.get("diagnostic") is not True or out.get("capture_purpose") != purpose
                        or out.get("calib_version") != calib or len(out.get("frames", [])) != len(refs)
                        or out.get("model_binding") != observation_context["model_binding"]):
                    raise RuntimeError("observation_mismatch")
                result = self._record_observation(op, body, out)
                ok, message = True, "검증 표본을 기록했습니다. 이 결과는 인증 승인이 아닙니다."
            elif guide:
                method = "admin" if op.get("actor", op["account"]) != op["account"] else "password"
                assertion = make_account_assertion(op["account"], session["challenge"], method, ttl_s=180,
                                                   key_path=self.secrets_dir / "account.key")
                out = client.enroll(session, images, timestamps, labels, assertion,
                                    {"version": CONSENT_VERSION, "agreed_at": op["consent_at"]},
                                    captured_at_ms=captured_at, re_enroll=body.re_enroll)
                ok = out.get("status") == "active"
                message = "얼굴 등록이 완료되었습니다." if ok else (
                    "중복 등록 여부를 운영자가 확인해야 합니다." if out.get("status") == "pending_review" else "얼굴 등록이 거절되었습니다.")
                result = {k: out[k] for k in ("status", "reason", "coverage", "pad_enabled", "pad_mode") if k in out}
                if ok:
                    self._check(op, secret)
                    try:
                        if body.re_enroll:
                            # A photo from an earlier enrollment must not describe
                            # newly replaced face features, even if the new save fails.
                            self.accounts.delete_photo(op["account"])
                        if body.photo_consent:
                            frontal = next((ref for label, ref in kept if label == "C"), None)
                            if frontal is None:
                                raise ValueError("no_frontal_photo")
                            self.accounts.store_photo(op["account"], frontal["image"]["rgb"],
                                                      consented_at=op["consent_at"])
                            result["photo_saved"] = True
                    except Exception:
                        # Enrollment is already committed; an optional image failure
                        # cannot turn that real enrollment into a rejected result.
                        LOG.warning("Unable to update registration photo")
                        result["photo_saved"] = False
                        message = "얼굴은 등록했지만 사진을 갱신하지 못했습니다. 계정 관리에서 사진을 확인해 주세요."
            else:
                out = client.authenticate(session, images, timestamps, captured_at_ms=captured_at)
                ok = out.get("decision") == "approved"
                if ok:
                    pub = serialization.load_pem_public_key((self.secrets_dir / "approval.pub").read_bytes())
                    claims = token.verify(pub, out.get("approval_token", ""))
                    if ((claims.get("dev"), claims.get("act"), claims.get("sid"), claims.get("acct")) !=
                            (self.device_id, action, session["session_id"], op["account"])):
                        raise RuntimeError("approval_scope_mismatch")
                message = "얼굴 인증에 성공했습니다." if ok else (
                    "추가 인증이 필요합니다." if out.get("decision") == "step_up" else "얼굴 인증이 거절되었습니다. 다시 시도해 주세요.")
                result = {k: out[k] for k in ("decision", "reason", "pad_enabled", "pad_mode") if k in out}
            self._update(op, state="succeeded" if ok else "rejected", message=message, result=result)
        except CaptureCancelled:
            self._update(op, state="cancelled", message="촬영을 취소했습니다.")
        except Exception as exc:
            known = {"capture_incomplete", "camera_unavailable", "calib_changed", "already_enrolled",
                     "approval_scope_mismatch", "readiness_changed"}
            reason = str(exc) if isinstance(exc, RuntimeError) and str(exc) in known else "service_error"
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 409:
                try:
                    detail = exc.response.json()
                except ValueError:
                    detail = {}
                if isinstance(detail, dict):
                    if (exc.request.url.path == "/v1/enrollments"
                            and detail.get("reason") == "already_enrolled"):
                        reason = "already_enrolled"
                    elif detail.get("detail") == "session":
                        reason = "capture_session_expired"
            messages = {"capture_incomplete": "촬영 시간이 초과되었습니다. 안내에 따라 다시 시도해 주세요.",
                        "camera_unavailable": "카메라 연결이 끊겼습니다. 다시 연결한 뒤 시도해 주세요.",
                        "calib_changed": "촬영 중 센서 설정이 변경되었습니다. 다시 시작해 주세요.",
                        "already_enrolled": "이 테스트 계정에는 이미 얼굴이 등록되어 있습니다.",
                        "capture_session_expired": "촬영 세션이 만료되었습니다. 다시 시작해 주세요.",
                        "approval_scope_mismatch": "현재 테스트 계정에 대한 승인 결과를 확인할 수 없습니다.",
                        "readiness_changed": "서버 또는 센서 상태가 변경되었습니다. 준비 상태를 확인해 주세요."}
            LOG.warning("Face operation failed: %s", type(exc).__name__)
            result = {"reason": reason}
            message = messages.get(reason, "서버 처리 결과를 확인할 수 없습니다. 준비 상태를 확인해 주세요.")
            if (observation_context is not None and body.observation_mode in ("auth", "enroll")
                    and guide_attempts and reason in ("capture_incomplete", "capture_session_expired")):
                recorded = self._record_capture_nonresponse(op, body, secret, client, observation_context, reason)
                if recorded:
                    result.update(recorded)
                    message += " 이 시도는 검증 비응답으로 기록했습니다."
            self._update(op, state="error", message=message, result=result)
        finally:
            if guide_attempts and (op["kind"] == "enroll" or getattr(body, "observation_mode", None) == "enroll"):
                # Operational counts only. Never log identities, frame references, images or raw poses.
                GUIDE_LOG.info("Enrollment guide summary: %s", json.dumps(guide_diagnostics, sort_keys=True))
            if client:
                client.close()
            # Local frame/guide references disappear when this worker returns.
            with self.lock:
                self.active = None

    def _record_capture_nonresponse(self, op, body, secret, client, initial, reason):
        """Keep selection failures in study denominators without inventing PAD scores."""
        try:
            current = client.readiness()  # Direct mTLS response, not a stale poll cache.
            camera = self.camera.status()["camera"]
            if (current.get("capture_ready") is not True or camera["state"] != "streaming"
                    or camera["info"].get("calib_version") != initial["calib_version"]
                    or current.get("calib_version") != initial["calib_version"]
                    or (current.get("pad_validation") or {}).get("binding") != initial["model_binding"]):
                return None
            with self.lock:
                self._check(op, secret)
                op.update(state="processing", message="완료되지 않은 검증 시도를 비응답으로 기록하고 있습니다.")
                # This is an explicit unmeasured attempt marker, not a sampled
                # frame. The evaluator counts it as nonresponse, with no score.
                return self._record_observation(op, body, {**initial, "frames": [
                    {"pad": None, "ok": False, "quality": {}, "reasons": [reason]}]},
                    capture_outcome="selection_timeout" if reason == "capture_incomplete" else "session_expired")
        except Exception:
            LOG.warning("Unable to record capture nonresponse")
            return None

    def _record_observation(self, op, body, out, *, capture_outcome=None):
        from faceauth.common.pad_validation import make_observation
        conditions = {"lighting": body.conditions, "mode": body.observation_mode}
        if capture_outcome:
            conditions["capture_outcome"] = capture_outcome
        observation = make_observation(
            presentation_id=op["id"], subject_id=body.subject_id, consent_id=op["id"],
            consent=True,
            kind=body.kind, split=body.split, instrument_id=body.instrument_id,
            calib_version=out["calib_version"], binding=out["model_binding"], frames=out["frames"],
            conditions=conditions)
        path = self.state_dir / "pad-observations.jsonl"
        fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "a") as stream:
            stream.write(json.dumps(observation, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return {"recorded": True, "presentation_id": op["id"], "authentication_approved": False}


def auth_router(service):
    router = APIRouter(prefix="/api/auth")

    def mutation(request):
        if not service.enabled:
            raise HTTPException(503, "웹 인증이 설정되지 않았습니다.")
        # Non-simple custom header + JSON blocks cross-origin form/fetch writes.
        if request.headers.get("x-paceauth-action") != "1":
            raise HTTPException(403, "요청 출처를 확인할 수 없습니다.")
        origin = request.headers.get("origin")
        expected = f"{request.url.scheme}://{request.url.netloc}"
        if origin is not None and origin != expected:
            raise HTTPException(403, "다른 사이트에서 보낸 요청은 허용하지 않습니다.")
        if request.headers.get("sec-fetch-site") == "cross-site":
            raise HTTPException(403, "다른 사이트에서 보낸 요청은 허용하지 않습니다.")

    def identity(request):
        if not service.enabled:
            raise HTTPException(503, "웹 인증이 설정되지 않았습니다.")
        secret = service.browser_secret(request)
        if service.open_console:
            if not service.console_session(secret):
                raise HTTPException(401, "페이지를 새로고침해 관리 화면에 연결해 주세요.")
            return secret, None
        account = service.accounts.account(secret)
        if not account:
            raise HTTPException(401, "테스트 계정으로 로그인해 주세요.")
        return secret, account

    def manager(request):
        secret, account = identity(request)
        if not service.open_console and not service.accounts.profile(account).get("is_admin"):
            raise HTTPException(403, "계정 관리는 관리자만 사용할 수 있습니다.")
        return secret, account

    def login_response(request, response, secret):
        previous = request.cookies.get(COOKIE)
        if previous:
            service.accounts.revoke(previous)
        response.set_cookie(COOKIE, secret, max_age=SESSION_SECONDS, httponly=True, samesite="strict",
                            secure=request.url.scheme == "https", path="/")
        account = service.accounts.account(secret)
        return {"account": service.accounts.profile(account)}

    def switching_account(request):
        secret = request.cookies.get(COOKIE)
        if secret and service.accounts.account(secret):
            op = service.operation(secret)
            if op and op["state"] not in FINAL_STATES:
                service.cancel(secret, op["id"])

    def password_mode():
        if service.open_console:
            raise HTTPException(409, "이 페이지에서는 로그인 없이 계정을 관리하고 데모를 사용할 수 있습니다.")

    def own_target(account):
        if account is None:
            raise HTTPException(400, "먼저 등록하거나 인증할 계정을 선택해 주세요.")
        return account

    @router.post("/signup")
    def signup(body: Signup, request: Request, response: Response):
        mutation(request)
        password_mode()
        switching_account(request)
        service.accounts.rate_limit(request.client.host if request.client else "unknown")
        service.create_account(body)
        return login_response(request, response, service.accounts.issue(body.account_id))

    @router.post("/profile")
    def profile(body: Profile, request: Request):
        mutation(request)
        _, account = identity(request)
        return {"account": service.accounts.rename(own_target(account), body.display_name)}

    @router.get("/accounts")
    def list_accounts(request: Request):
        with service.management_lock:
            manager(request)
            return service.managed_accounts()

    @router.post("/accounts", status_code=201)
    def create_account(body: ManagedCreation, request: Request):
        mutation(request)
        with service.management_lock:
            manager(request)
            if service.open_console:
                if body.display_name is None:
                    raise HTTPException(422, "표시 이름을 입력해 주세요.")
                body = Signup(account_id=body.account_id or "person-" + secrets.token_hex(8),
                              display_name=body.display_name, password=secrets.token_urlsafe(48))
            else:
                if body.account_id is None or body.password is None:
                    raise HTTPException(422, "계정 ID와 비밀번호를 입력해 주세요.")
                body = Signup(**body.model_dump())
            return {"account": service.create_account(body)}

    @router.post("/accounts/{account_id}/profile")
    def account_profile(account_id: str, body: Profile, request: Request):
        mutation(request)
        with service.management_lock:
            manager(request)
            service.accounts.rename(account_id, body.display_name)
            return {"account": service.managed_account(account_id)}

    @router.post("/accounts/{account_id}/status")
    def account_status(account_id: str, body: AccountStatus, request: Request):
        mutation(request)
        with service.management_lock:
            _, actor = manager(request)
            return {"account": service.change_account_status(account_id, body.enabled, actor)}

    @router.post("/accounts/{account_id}/delete")
    def account_delete(account_id: str, body: AccountDeletion, request: Request):
        mutation(request)
        with service.management_lock:
            _, actor = manager(request)
            if body.confirm_account_id != account_id:
                raise HTTPException(400, "삭제할 계정을 다시 확인해 주세요.")
            return service.delete_account(account_id, actor)

    @router.get("/accounts/{account_id}/photo")
    def account_photo(account_id: str, request: Request):
        with service.management_lock:
            manager(request)
            photo = service.accounts.photo(account_id)
            if photo is None:
                raise HTTPException(404, "등록 사진이 없습니다.")
            return Response(photo, media_type="image/jpeg", headers={"Cache-Control": "no-store, max-age=0"})

    @router.post("/accounts/{account_id}/photo/delete")
    def account_photo_delete(account_id: str, request: Request):
        mutation(request)
        with service.management_lock:
            manager(request)
            service._management_idle()
            service.accounts.delete_photo(account_id)
            return {"account": service.managed_account(account_id)}

    @router.post("/accounts/{account_id}/enroll", status_code=202)
    def account_enroll(account_id: str, body: CaptureRequest, request: Request):
        mutation(request)
        with service.management_lock:
            secret, _ = manager(request)
            return service.begin("enroll", body, secret, account_id)

    @router.post("/accounts/{account_id}/authenticate", status_code=202)
    def account_authenticate(account_id: str, body: CaptureRequest, request: Request):
        mutation(request)
        with service.management_lock:
            secret, _ = manager(request)
            return service.begin("authenticate", body, secret, account_id)

    @router.post("/login")
    def login(body: Credentials, request: Request, response: Response):
        mutation(request)
        password_mode()
        switching_account(request)
        service.accounts.rate_limit(request.client.host if request.client else "unknown")
        return login_response(request, response, service.accounts.login(body.account_id, body.password))

    @router.post("/logout")
    def logout(request: Request, response: Response):
        mutation(request)
        password_mode()
        secret, _ = identity(request)
        op = service.operation(secret)
        if op and op["state"] not in FINAL_STATES:
            service.cancel(secret, op["id"])
        service.accounts.revoke(secret)
        response.delete_cookie(COOKIE, path="/")
        return {"ok": True}

    @router.post("/enroll", status_code=202)
    def enroll(body: CaptureRequest, request: Request):
        mutation(request)
        secret, account = identity(request)
        return service.begin("enroll", body, secret, own_target(account))

    @router.post("/authenticate", status_code=202)
    def authenticate(body: CaptureRequest, request: Request):
        mutation(request)
        secret, account = identity(request)
        return service.begin("authenticate", body, secret, own_target(account))

    @router.post("/observe", status_code=202)
    def observe(body: ObservationRequest, request: Request):
        mutation(request)
        secret, account = identity(request)
        target = own_target(body.target_account_id) if service.open_console else account
        return service.begin("observe", body, secret, target)

    @router.get("/operation")
    def operation(request: Request, id: str | None = None, client_request_id: str | None = None):
        secret, _ = identity(request)
        return service.operation(secret, id, client_request_id)

    @router.post("/cancel")
    def cancel(body: CancelRequest, request: Request):
        mutation(request)
        secret, _ = identity(request)
        return service.cancel(secret, body.id)

    return router
