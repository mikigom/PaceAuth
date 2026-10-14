"""Pilot identity and operation boundaries, using explicit synthetic fixtures."""
import json
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np
import httpx
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi.testclient import TestClient

from faceauth.common import token
from faceauth.terminal.guide import ORDER, TARGETS
from faceauth.terminal.live import CameraWorker, create_app
from faceauth.terminal.web_auth import Accounts, COOKIE, LiveAuth

HEADERS = {"X-PaceAuth-Action": "1", "Origin": "http://testserver"}
CREDENTIALS = {"account_id": "alice", "password": "test-password-123"}
CONSENT = {"consent": {"version": "v1", "agreed": True}}


class FakeCamera(CameraWorker):
    def __init__(self):
        super().__init__()
        self.state = "streaming"
        self.info = {"calib_version": "test-calib"}
        self.last_frame = time.monotonic()
        self.images = {"rgb": b"synthetic-jpeg"}
        self.sequence = 0

    def start(self):
        pass

    def stop(self):
        pass

    def status(self):
        self.last_frame = time.monotonic()
        return super().status()

    def capture_sequence(self):
        return self.sequence

    def capture_after(self, after, timeout=1):
        time.sleep(0.005)
        self.sequence += 1
        stamp = int(time.time() * 1000)
        frame = {"rgb": np.zeros((48, 64, 3), np.uint8), "depth": np.ones((48, 64), np.uint16) * 600,
                 "ir": np.ones((48, 64), np.uint8) * 100, "calib_version": "test-calib",
                 "ts_ms": {key: stamp for key in ("rgb", "depth", "ir")}, "captured_at_ms": stamp}
        return self.sequence, time.monotonic(), frame


class FakeClient:
    def __init__(self, key):
        self.key = key
        self.calls = 0
        self.account = "alice"
        self.enrollment_status = "active"
        self.block = None
        self.created = self.submitted = 0
        self.account_enabled = {}
        self.account_faces = {}

    def account_summary(self, ids):
        return {"accounts": [{"account_id": account_id, "enabled": self.account_enabled.get(account_id, True),
                              "face_status": self.account_faces.get(account_id, "none"), "enrolled_at": None}
                             for account_id in ids]}

    def account_status(self, account_id, enabled):
        self.account_enabled[account_id] = enabled
        return self.account_summary([account_id])["accounts"][0]

    def delete_account_face(self, account_id):
        self.account_faces[account_id] = "revoked"
        return {"account_id": account_id, "revoked": True, "key_destroyed": True, "index_removed": True}

    def readiness(self):
        return {"capture_ready": True, "calib_version": "test-calib",
                "pad_validation": {"binding": {"model_version": "fixture@1"}}}

    def close(self):
        pass

    def create_session(self, purpose, action):
        self.created += 1
        self.calls = 0
        return {"session_id": "fixture-session", "challenge": "unpredictable-fixture", "purpose": purpose}

    def capture_guide(self, session, frame, ts, purpose):
        if self.block:
            self.block.wait(2)
        direction = ORDER[min(self.calls // 2, len(ORDER) - 1)] if purpose == "enroll" else "C"
        self.calls += 1
        return {"frame": {"ok": True, "pose": [*TARGETS[direction], 0], "score": 1, "reasons": []}}

    def enroll(self, session, frames, timestamps, labels, assertion, consent, **kwargs):
        self.submitted += 1
        self.enrollment = {"frames": frames, "timestamps": timestamps, "labels": labels,
                           "assertion": assertion, "consent": consent, **kwargs}
        account = token.verify(self.key.public_key(), assertion)["acct"]
        self.account_faces[account] = self.enrollment_status
        return {"status": self.enrollment_status, "reason": "ok"}

    def authenticate(self, session, frames, timestamps, **kwargs):
        self.submitted += 1
        now = int(time.time())
        signed = token.sign(self.key, {"dev": "dev-001", "act": "pilot-check", "sid": session["session_id"],
                                       "acct": self.account, "iat": now, "exp": now + 60, "amr": ["face"]})
        return {"decision": "approved", "approval_token": signed}

    def observe(self, session, frames, timestamps, **kwargs):
        self.submitted += 1
        self.observed_purpose = kwargs["purpose"]
        assert session["purpose"] == kwargs["purpose"]
        return {"diagnostic": True, "capture_purpose": kwargs["purpose"],
                "calib_version": "test-calib", "model_binding": {"model_version": "fixture@1"},
                "frames": [{"pad": {"live": False, "nonresponse": False, "score": 0.1}, "ok": True,
                            "emb": b"not-to-be-saved", "rgb": b"not-to-be-saved", "quality": {}, "reasons": []}
                           for _ in frames]}


class LocalAuth(LiveAuth):
    def start(self):
        pass  # Explicit contract fixture: no actual network/engine.


@pytest.fixture
def setup(tmp_path):
    camera = FakeCamera()
    key = Ed25519PrivateKey.generate()
    backend = FakeClient(key)
    service = LocalAuth(camera, enabled=True, state_dir=tmp_path / "state", client_factory=lambda _: backend)
    service.secrets_dir = tmp_path / "secrets"
    service.secrets_dir.mkdir()
    (service.secrets_dir / "approval.pub").write_bytes(key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo))
    (service.secrets_dir / "account.key").write_bytes(key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    service.backend = {"ready": True, "capture_ready": True, "calib_version": "test-calib", "blockers": []}
    service.backend_at = time.monotonic()
    with TestClient(create_app(camera, service)) as client:
        yield client, service, backend


def signup(client):
    response = client.post("/api/auth/signup", json=CREDENTIALS, headers=HEADERS)
    assert response.status_code == 200, response.text
    return response


def finish(client, op):
    deadline = time.monotonic() + 6
    while time.monotonic() < deadline:
        response = client.get("/api/auth/operation", params={"id": op["id"]})
        assert response.status_code == 200
        current = response.json()
        if current["state"] in ("succeeded", "error", "rejected", "cancelled"):
            return current
        time.sleep(0.02)
    raise AssertionError("operation did not finish")


def test_password_login_cookie_and_durable_hashes(setup):
    client, service, _ = setup
    response = signup(client)
    cookie = response.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie
    assert client.get("/api/status").json()["auth"]["account"]["account_id"] == "alice"
    assert CREDENTIALS["password"].encode() not in service.accounts.path.read_bytes()
    assert client.post("/api/auth/logout", headers=HEADERS).status_code == 200
    assert client.post("/api/auth/login", json={**CREDENTIALS, "password": "bad-password-123"}, headers=HEADERS).status_code == 401
    assert client.post("/api/auth/login", json=CREDENTIALS, headers=HEADERS).status_code == 200


def test_browser_cross_origin_writes_and_weak_credentials_are_rejected(setup):
    client, _, _ = setup
    assert client.post("/api/auth/signup", json=CREDENTIALS).status_code == 403
    assert client.post("/api/auth/signup", json=CREDENTIALS,
                       headers={**HEADERS, "Origin": "https://outside.invalid"}).status_code == 403
    assert client.post("/api/auth/signup", json={**CREDENTIALS, "password": "short"}, headers=HEADERS).status_code == 422
    assert client.post("/api/auth/authenticate", json=CONSENT, headers=HEADERS).status_code == 401


def test_display_names_migrate_without_changing_passwords_or_sessions(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    from faceauth.terminal.web_auth import _password_hash
    salt = b"0123456789abcdef"
    hashed = _password_hash(CREDENTIALS["password"], salt)
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE accounts (id TEXT PRIMARY KEY,salt BLOB NOT NULL,password_hash BLOB NOT NULL)")
        db.execute("INSERT INTO accounts VALUES (?,?,?)", ("alice", salt, hashed))
    accounts = Accounts(path)
    secret = accounts.login("alice", CREDENTIALS["password"])
    assert accounts.profile("alice")["display_name"] == "alice"
    assert accounts.rename("alice", "홍길동")["display_name"] == "홍길동"
    reopened = Accounts(path)
    assert reopened.account(secret) == "alice"
    assert reopened.profile("alice")["display_name"] == "홍길동"
    with reopened.connect() as db:
        assert tuple(db.execute("SELECT salt,password_hash FROM accounts").fetchone()) == (salt, hashed)


def test_signup_and_profile_name_changes_require_own_session_and_safe_input(setup):
    client, service, _ = setup
    assert client.post("/api/auth/profile", json={"display_name": "누구"}, headers=HEADERS).status_code == 401
    response = client.post("/api/auth/signup", json={**CREDENTIALS, "display_name": " 홍길동 "}, headers=HEADERS)
    assert response.status_code == 200 and response.json()["account"]["display_name"] == "홍길동"
    service.accounts.signup("bob", "different-password", "밥")
    response = client.post("/api/auth/profile", json={"display_name": "김테스트"}, headers=HEADERS)
    assert response.status_code == 200
    assert client.get("/api/status").json()["auth"]["account"]["display_name"] == "김테스트"
    assert service.accounts.profile("bob")["display_name"] == "밥"
    for name in ("  ", "a" * 81, "name\u202e", "name\nline"):
        assert client.post("/api/auth/profile", json={"display_name": name}, headers=HEADERS).status_code == 422
    assert client.post("/api/auth/profile", json={"display_name": "changed"},
                       headers={**HEADERS, "Origin": "https://elsewhere.invalid"}).status_code == 403


def test_login_rate_limit_and_session_revocation(setup):
    client, service, _ = setup
    signup(client)
    secret = client.cookies.get(COOKIE)
    assert service.accounts.account(secret) == "alice"
    service.accounts.revoke(secret)
    assert service.accounts.account(secret) is None
    for _ in range(9):
        assert client.post("/api/auth/login", json={**CREDENTIALS, "password": "bad-password-123"}, headers=HEADERS).status_code == 401
    assert client.post("/api/auth/login", json=CREDENTIALS, headers=HEADERS).status_code == 429


def test_missing_consent_or_pad_validation_never_starts_a_capture(setup):
    client, service, backend = setup
    signup(client)
    assert client.post("/api/auth/enroll", json={"consent": {"version": "v1", "agreed": False}}, headers=HEADERS).status_code == 400
    service.backend.update(ready=False, blockers=[{"code": "pad_missing", "message": "PAD not validated"}])
    assert client.post("/api/auth/authenticate", json=CONSENT, headers=HEADERS).status_code == 503
    assert backend.created == 0
    assert client.get("/api/status").json()["auth"]["capture_ready"] is True


def test_expired_readiness_and_calibration_mismatch_fail_closed(setup):
    client, service, _ = setup
    signup(client)
    service.backend_at -= 21
    assert client.post("/api/auth/authenticate", json=CONSENT, headers=HEADERS).status_code == 503
    service.backend_at = time.monotonic()
    service.backend["calib_version"] = "different-camera"
    assert client.get("/api/status").json()["auth"]["ready"] is False


def test_signed_approval_is_verified_and_not_returned_to_browser(setup):
    client, _, backend = setup
    signup(client)
    response = client.post("/api/auth/authenticate", json=CONSENT, headers=HEADERS)
    assert response.status_code == 202
    result = finish(client, response.json())
    assert result["state"] == "succeeded"
    assert result["result"] == {"decision": "approved"}
    assert "approval_token" not in json.dumps(result)
    assert backend.submitted == 1


@pytest.mark.parametrize("kind", ["enroll", "authenticate"])
def test_explicit_pad_disabled_pilot_exposes_actual_mode_and_runs_identity_flow(setup, kind):
    client, service, backend = setup
    signup(client)
    service.backend.update(pad_enabled=False, pad_mode="disabled_pilot", pad_validation={"ready": False})
    original = getattr(backend, kind)

    def result_with_mode(*args, **kwargs):
        return {**original(*args, **kwargs), "pad_enabled": False, "pad_mode": "disabled_pilot"}

    setattr(backend, kind, result_with_mode)
    status = client.get("/api/status").json()["auth"]
    assert status["ready"] is True and status["pad_enabled"] is False
    assert status["pad_validation"]["ready"] is False
    response = client.post("/api/auth/" + kind, json=CONSENT, headers=HEADERS)
    assert response.status_code == 202
    operation = response.json()
    assert operation["pad_enabled"] is False and operation["pad_mode"] == "disabled_pilot"
    result = finish(client, operation)
    assert result["state"] == "succeeded"
    assert result["result"]["pad_enabled"] is False
    assert result["result"]["pad_mode"] == "disabled_pilot"
    assert "approval_token" not in json.dumps(result)


def test_other_account_approval_cannot_approve_current_browser(setup):
    client, _, backend = setup
    signup(client)
    backend.account = "someone-else"
    response = client.post("/api/auth/authenticate", json=CONSENT, headers=HEADERS)
    result = finish(client, response.json())
    assert result["state"] == "error"
    assert result["result"]["reason"] == "approval_scope_mismatch"


@pytest.mark.parametrize("path, detail, expected", [
    ("/v1/capture/guide", {"detail": "session"}, "capture_session_expired"),
    ("/v1/enrollments", {"reason": "already_enrolled"}, "already_enrolled"),
    ("/v1/capture/guide", {"reason": "already_enrolled"}, "service_error"),
])
def test_backend_conflicts_keep_their_actual_meaning(setup, path, detail, expected):
    client, _, backend = setup
    signup(client)

    def conflict(*args, **kwargs):
        response = httpx.Response(409, request=httpx.Request("POST", "https://gateway" + path), json=detail)
        response.raise_for_status()

    backend.capture_guide = conflict
    response = client.post("/api/auth/authenticate", json=CONSENT, headers=HEADERS)
    result = finish(client, response.json())
    assert result["state"] == "error"
    assert result["result"]["reason"] == expected


@pytest.mark.parametrize("status, expected", [("active", "succeeded"), ("pending_review", "rejected")])
def test_enrollment_uses_verified_account_consent_and_real_capture_metadata(setup, status, expected):
    client, service, backend = setup
    signup(client)
    backend.enrollment_status = status
    response = client.post("/api/auth/enroll", json=CONSENT, headers=HEADERS)
    result = finish(client, response.json())
    assert result["state"] == expected
    submitted = backend.enrollment
    assert len(submitted["frames"]) == 18
    assert set(submitted["labels"]) == set(ORDER)
    claims = token.verify(backend.key.public_key(), submitted["assertion"])
    assert claims["acct"] == "alice" and claims["nonce"] == "unpredictable-fixture"
    assert claims["method"] == "password"
    stamps = [ts["rgb"] for ts in submitted["timestamps"]]
    assert all(a < b for a, b in zip(stamps, stamps[1:]))


def test_public_guide_age_includes_processing_and_keeps_aging_without_new_samples(monkeypatch):
    from faceauth.terminal import web_auth
    operation = {"id": "fixture", "guide": {"sample_seq": 3}, "_guide_sample_at": 10.0,
                 "guide_latency_ms": 240.0, "account": "private", "owner": "private"}
    monkeypatch.setattr(web_auth.time, "monotonic", lambda: 10.25)
    first = LiveAuth.public(operation)
    assert first["guide_age_ms"] == 250 and first["guide_latency_ms"] == 240
    monkeypatch.setattr(web_auth.time, "monotonic", lambda: 10.8)
    stale = LiveAuth.public(operation)
    assert stale["guide_age_ms"] == 800 and stale["guide"]["sample_seq"] == 3
    assert "_guide_sample_at" not in first and "owner" not in first and "account" not in first
    assert LiveAuth.public({"guide": None})["guide_age_ms"] is None


def test_guide_timing_quality_counts_and_private_summary_use_real_samples(setup, monkeypatch):
    from faceauth.terminal import web_auth
    client, service, backend = setup
    signup(client)
    original_encode, original_capture, original_update = web_auth.encode_frame, backend.capture_guide, service._update
    encoded_at, sampled, reports = [], [], []
    calls = 0

    def encode(*args):
        encoded_at.append(time.monotonic())
        time.sleep(.01)                 # include encoding, not just server latency, in freshness
        return original_encode(*args)

    def capture(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            time.sleep(.06)
            return {"frame": {"pose": [25, 0, 0], "ok": False, "score": 1,
                              "reasons": ["blur", "depth_invalid"]}}
        if calls == 2:
            return {"frame": {"pose": None, "ok": False, "score": 0, "reasons": ["new_sensor_failure"]}}
        if calls == 3:
            return {"frame": {"pose": [12, 0, 0], "ok": True, "score": 1, "reasons": []}}
        return original_capture(*args, **kwargs)

    def update(operation, **fields):
        if "_guide_sample_at" in fields:
            sampled.append((fields["_guide_sample_at"], LiveAuth.public(fields)))
        return original_update(operation, **fields)

    monkeypatch.setattr(web_auth, "encode_frame", encode)
    monkeypatch.setattr(backend, "capture_guide", capture)
    monkeypatch.setattr(service, "_update", update)
    monkeypatch.setattr(web_auth.GUIDE_LOG, "info", lambda message, report: reports.append(json.loads(report)))
    response = client.post("/api/auth/enroll", json={**CONSENT, "guide_mode": "circle"}, headers=HEADERS)
    result = finish(client, response.json())
    service.operation_thread.join(2)
    assert result["state"] == "succeeded" and len(backend.enrollment["frames"]) == 18
    assert len(sampled) == len(encoded_at) == 21
    assert all(start <= encoded for (start, _), encoded in zip(sampled, encoded_at))
    assert sampled[0][1]["guide_age_ms"] >= 70 and sampled[0][1]["guide_latency_ms"] >= 70
    assert [sample[1]["guide"]["sample_seq"] for sample in sampled] == list(range(21))
    # Fifteen per second is a cap, not concurrent work or a promise about the hardware's frame rate.
    assert all(b[0] - a[0] >= 1 / 15 for a, b in zip(sampled, sampled[1:]))
    diagnostics = result["guide_diagnostics"]
    assert diagnostics["samples"] == 21 and diagnostics["quality_rejected"] == 2 and diagnostics["pose_unmatched"] == 1
    assert diagnostics["collected_frames"] == 18 and diagnostics["completed_directions"] == 9
    assert diagnostics["reasons"] == {"blur": 1, "depth_invalid": 1, "quality_unknown": 1, "between_directions": 1}
    assert diagnostics["latency_max_ms"] >= 70 and diagnostics["latency_mean_ms"] >= 10
    assert reports == [diagnostics]
    assert set(reports[0]) == {"samples", "quality_rejected", "pose_unmatched", "reasons", "collected_frames",
                              "completed_directions", "latency_mean_ms", "latency_max_ms"}
    assert "alice" not in json.dumps(reports) and "pose" not in reports[0]


def test_guide_metrics_logger_is_visible_without_enabling_other_application_logs():
    import logging
    from faceauth.terminal import web_auth
    assert web_auth.GUIDE_LOG.isEnabledFor(logging.INFO)
    assert any(isinstance(handler, logging.StreamHandler) for handler in web_auth.GUIDE_LOG.handlers)
    assert not web_auth.GUIDE_LOG.propagate


def test_duplicate_start_owner_isolation_and_request_id_recovery(setup):
    client, service, backend = setup
    signup(client)
    secret = client.cookies.get(COOKIE)
    backend.block = threading.Event()
    body = {**CONSENT, "client_request_id": "request-12345678"}
    first = client.post("/api/auth/authenticate", json=body, headers=HEADERS).json()
    second = client.post("/api/auth/authenticate", json=body, headers=HEADERS).json()
    assert first["id"] == second["id"]
    assert backend.created <= 1
    assert client.get("/api/auth/operation?client_request_id=unknown-id").json() is None
    assert client.get("/api/auth/operation?client_request_id=request-12345678").json()["id"] == first["id"]
    foreign_secret = service.accounts.signup("bob", "different-password")
    with pytest.raises(Exception) as rejected:
        service.operation(foreign_secret, first["id"])
    assert rejected.value.status_code == 404
    assert client.post("/api/auth/enroll", json=CONSENT, headers=HEADERS).status_code == 409
    assert client.post("/api/auth/cancel", json={"id": first["id"]}, headers=HEADERS).status_code == 200
    backend.block.set()
    assert finish(client, first)["state"] == "cancelled"
    assert backend.submitted == 0


@pytest.mark.parametrize("mode, count, purpose", [("diagnostic", 5, "auth"), ("auth", 3, "auth"), ("enroll", 18, "enroll")])
def test_observation_modes_use_actual_selectors_without_enrollment_or_approval(setup, mode, count, purpose):
    client, service, backend = setup
    signup(client)
    service.backend.update(ready=False, blockers=[{"code": "pad_missing", "message": "PAD not validated"}])
    body = {**CONSENT, "observation_mode": mode, "kind": "live", "split": "calibration",
            "subject_id": "subject-001", "conditions": "indoor"}
    response = client.post("/api/auth/observe", json=body, headers=HEADERS)
    assert response.status_code == 202
    result = finish(client, response.json())
    assert result["state"] == "succeeded", result
    assert result["result"]["authentication_approved"] is False
    raw = (service.state_dir / "pad-observations.jsonl").read_text()
    assert "not-to-be-saved" not in raw and "alice" not in raw
    row = json.loads(raw)
    assert row["conditions"]["mode"] == mode
    assert len(row["frames"]) == count
    assert all("emb" not in frame and "rgb" not in frame for frame in row["frames"])
    assert backend.observed_purpose == purpose
    assert not hasattr(backend, "enrollment")
    assert result["observation_mode"] == mode


def test_observation_response_must_match_calibration_and_purpose(setup):
    client, service, backend = setup
    signup(client)
    real_observe = backend.observe

    def wrong_response(*args, **kwargs):
        result = real_observe(*args, **kwargs)
        result["capture_purpose"] = "enroll"
        return result

    backend.observe = wrong_response
    body = {**CONSENT, "observation_mode": "auth", "kind": "live", "split": "calibration", "subject_id": "subject-001"}
    response = client.post("/api/auth/observe", json=body, headers=HEADERS)
    assert finish(client, response.json())["state"] == "error"
    assert not (service.state_dir / "pad-observations.jsonl").exists()


@pytest.mark.parametrize("mode", ["auth", "enroll"])
def test_observation_selection_timeout_is_recorded_as_unmeasured_nonresponse(setup, mode):
    client, service, backend = setup
    signup(client)
    frames = service._frames

    def one_unacceptable_frame(*args, **kwargs):
        yield next(frames(*args, **kwargs))

    service._frames = one_unacceptable_frame
    backend.capture_guide = lambda *a, **kw: {"frame": {"ok": False, "pose": None, "score": 0, "reasons": ["no_face"]}}
    body = {**CONSENT, "observation_mode": mode, "kind": "print", "split": "test",
            "subject_id": "subject-001", "instrument_id": "paper-01"}
    response = client.post("/api/auth/observe", json=body, headers=HEADERS)
    result = finish(client, response.json())
    assert result["state"] == "error" and result["result"]["recorded"] is True
    assert result["result"]["authentication_approved"] is False
    row = json.loads((service.state_dir / "pad-observations.jsonl").read_text())
    assert row["conditions"] == {"lighting": "", "mode": mode, "capture_outcome": "selection_timeout"}
    assert row["frames"] == [{"pad": {"live": False, "nonresponse": True, "score": None},
                              "ok": False, "quality": {}, "reasons": ["capture_incomplete"]}]
    assert row["binding"] == {"model_version": "fixture@1"}
    assert backend.submitted == 0


@pytest.mark.parametrize("changed", ["binding", "calibration", "unavailable"])
def test_timeout_never_relabels_a_changed_or_unavailable_engine(setup, changed):
    client, service, backend = setup
    signup(client)
    frames = service._frames

    def one_frame(*args, **kwargs):
        yield next(frames(*args, **kwargs))

    service._frames = one_frame
    readiness = backend.readiness
    calls = 0

    def changed_readiness():
        nonlocal calls
        calls += 1
        state = readiness()
        if calls > 1:
            if changed == "binding":
                state["pad_validation"]["binding"]["model_version"] = "fixture@2"
            elif changed == "calibration":
                state["calib_version"] = "new-camera"
            else:
                state["capture_ready"] = False
        return state

    backend.readiness = changed_readiness
    body = {**CONSENT, "observation_mode": "auth", "kind": "live", "split": "test", "subject_id": "subject-001"}
    response = client.post("/api/auth/observe", json=body, headers=HEADERS)
    result = finish(client, response.json())
    assert result["result"] == {"reason": "capture_incomplete"}
    assert not (service.state_dir / "pad-observations.jsonl").exists()


def test_cancelled_observation_is_not_counted_as_pad_nonresponse(setup):
    client, service, backend = setup
    signup(client)
    backend.block = threading.Event()
    body = {**CONSENT, "observation_mode": "auth", "kind": "live", "split": "test", "subject_id": "subject-001"}
    response = client.post("/api/auth/observe", json=body, headers=HEADERS)
    operation = response.json()
    assert client.post("/api/auth/cancel", json={"id": operation["id"]}, headers=HEADERS).status_code == 200
    backend.block.set()
    assert finish(client, operation)["state"] == "cancelled"
    assert not (service.state_dir / "pad-observations.jsonl").exists()
    assert backend.submitted == 0


def test_session_expiry_after_guide_measurement_is_retained_as_nonresponse(setup):
    client, service, backend = setup
    signup(client)
    guide = backend.capture_guide

    def expires_after_first_measurement(*args, **kwargs):
        if backend.calls:
            response = httpx.Response(409, request=httpx.Request("POST", "https://gateway/v1/capture/guide"),
                                      json={"detail": "session"})
            response.raise_for_status()
        return guide(*args, **kwargs)

    backend.capture_guide = expires_after_first_measurement
    body = {**CONSENT, "observation_mode": "auth", "kind": "live", "split": "test", "subject_id": "subject-001"}
    response = client.post("/api/auth/observe", json=body, headers=HEADERS)
    result = finish(client, response.json())
    assert result["result"]["reason"] == "capture_session_expired"
    assert result["result"]["recorded"] is True
    row = json.loads((service.state_dir / "pad-observations.jsonl").read_text())
    assert row["conditions"]["capture_outcome"] == "session_expired"
    assert row["frames"][0]["pad"]["score"] is None


def test_raw_capture_wait_does_not_return_pre_challenge_or_failed_frames():
    camera = CameraWorker()
    camera.state = "streaming"
    camera.frames = 4
    camera.last_frame = time.monotonic()
    camera.capture = {"calib_version": "old"}
    assert camera.capture_after(4, timeout=0) is None
    camera.frames = 5
    assert camera.capture_after(4, timeout=0)[0] == 5
    camera.state = "error"
    assert camera.capture_after(4, timeout=0) is None
