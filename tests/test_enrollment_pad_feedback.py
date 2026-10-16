"""Managed enrollment omits PAD without relaxing identity, consent, or auth checks."""
import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from faceauth.api import app as api
from faceauth.terminal.guide import TARGETS
from test_web_auth import CONSENT, HEADERS, finish, setup, signup

ORIGINAL_SESSION = api._session


@pytest.fixture
def enrollment_api(monkeypatch):
    state = {"integrity": [], "consume": True, "audits": [], "analyses": 0,
             "consent": True, "enabled": True, "stored": 0, "search": []}
    frames = [{"ok": True, "n_faces": 1, "emb": np.array([1, 0], dtype="<f4").tobytes(),
               "pose": [*pose, 0], "reasons": [], "pad": {"live": False, "score": 0.123}}
              for pose in TARGETS.values() for _ in range(2)]
    state["analysis"] = {"frames": frames, "dim": 2, "versions": {"recog": "fixture@1"},
                         "pad_enabled": True, "pad_mode": "enabled", "calib_version": "test-calib",
                         "pad_validation": {"ready": False, "calib_version": "test-calib"}}

    @asynccontextmanager
    async def transaction():
        yield None

    @asynccontextmanager
    async def connection():
        yield SimpleNamespace(transaction=transaction)

    async def device(*args):
        return {"id": "synthetic-device", "calib_version": "test-calib"}

    async def session(*args):
        assert args[3] == "enroll"
        return {"id": "synthetic-session", "challenge": "synthetic-challenge"}, None

    async def active(*args):
        return "fixture@1"

    async def absent(*args):
        return None

    async def control(*args, **kwargs):
        return {"enabled": state["enabled"], "enrollment_epoch": 0}

    async def consume(*args):
        return state["consume"]

    async def audit(*args, **kwargs):
        state["audits"].append(kwargs["detail"])

    async def analyze(mode, *args):
        assert mode == "enroll"
        state["analyses"] += 1
        return state["analysis"]

    async def search(*args):
        return state["search"]

    async def encrypt(*args):
        return [b"synthetic-ciphertext"] * 4

    async def store(*args):
        state["stored"] += 1
        return 1

    async def body():
        return b"synthetic-packet"

    monkeypatch.setattr(api.C, "cfg", SimpleNamespace(pad_enabled=True, min_dirs=9, t_same=.6, t_dup=.9))
    monkeypatch.setattr(api.C, "pool", SimpleNamespace(connection=connection))
    monkeypatch.setattr(api.C, "deps", SimpleNamespace(analyze=analyze, search=search, encrypt=encrypt))
    monkeypatch.setattr(api, "_packet", lambda _: ({"device_id": "synthetic-device", "capture_purpose": "auth"},
                                                   b"sig", [{}], "session"))
    monkeypatch.setattr(api, "_device", device)
    monkeypatch.setattr(api, "_session", session)
    monkeypatch.setattr(api, "_active_recog", active)
    monkeypatch.setattr(api, "_assertion", lambda *_: ({"acct": "synthetic-account"}, None))
    monkeypatch.setattr(api, "_one", absent)
    monkeypatch.setattr(api, "_account_control", control)
    monkeypatch.setattr(api, "_integrity", lambda *_: state["integrity"])
    monkeypatch.setattr(api, "_consent", lambda _: "synthetic-consent-at" if state["consent"] else None)
    monkeypatch.setattr(api, "_consume", consume)
    monkeypatch.setattr(api, "_store", store)
    monkeypatch.setattr(api, "audit", audit)
    return state, SimpleNamespace(body=body)


@pytest.mark.parametrize("pad", [None, {"live": False}, {"live": False, "nonresponse": True}])
def test_enrollment_accepts_without_pad_while_profile_is_unvalidated(enrollment_api, monkeypatch, pad):
    state, request = enrollment_api
    for frame in state["analysis"]["frames"]:
        frame["pad"] = pad
    monkeypatch.setattr(api, "_pad_validated", lambda *_: pytest.fail("enrollment must not require PAD validation"))
    monkeypatch.setattr(api.policy, "pad_live", lambda *_: pytest.fail("enrollment must not require PAD verdicts"))
    result = json.loads(asyncio.run(api.enroll(request)).body)
    assert result["status"] == "active" and result["reason"] == "ok"
    assert result["pad_enabled"] is True and result["enrollment_pad_enabled"] is False
    assert state["stored"] == 1 and state["audits"][0]["enrollment_pad_enabled"] is False
    assert "enrollment_feedback" not in result


@pytest.mark.parametrize("failure,expected", [
    ("integrity", "denied"), ("consent", "consent"), ("quality", "retry"),
    ("coverage", "coverage"), ("same_person", "retry"), ("replay", "session"), ("disabled", "denied"),
])
def test_enrollment_non_pad_safeguards_remain(enrollment_api, failure, expected):
    state, request = enrollment_api
    if failure == "integrity":
        state["integrity"] = ["bad_signature"]
    elif failure == "consent":
        state["consent"] = False
    elif failure == "quality":
        state["analysis"]["frames"][0].update(ok=False, reasons=["blur"])
    elif failure == "coverage":
        state["analysis"]["frames"] = state["analysis"]["frames"][:2]
    elif failure == "same_person":
        state["analysis"]["frames"][0]["emb"] = np.array([0, 1], dtype="<f4").tobytes()
    elif failure == "replay":
        state["consume"] = False
    elif failure == "disabled":
        state["enabled"] = False
    result = json.loads(asyncio.run(api.enroll(request)).body)
    assert result["status"] == "rejected" and result["reason"] == expected
    assert result["enrollment_pad_enabled"] is False and state["stored"] == 0
    assert "enrollment_feedback" not in result and "subject_id" not in result
    if failure in ("integrity", "consent", "disabled"):
        assert state["analyses"] == 0


def test_authentication_pad_failure_keeps_coarse_response(enrollment_api):
    result = asyncio.run(api._auth_final({"id": "session", "action_id": "pilot-check"},
                                        {"id": "device"}, "pad_fail"))
    assert result == {"decision": "rejected", "reason": "denied", "pad_enabled": True, "pad_mode": "enabled"}


@pytest.mark.parametrize("pad_mode", ["enabled", "femto_geometry_experimental"])
def test_terminal_exposes_registration_policy_without_relabeling_auth(setup, pad_mode):
    client, service, backend = setup
    signup(client)
    service.backend.update(ready=False, enrollment_ready=True, enrollment_blockers=[],
                           pad_mode=pad_mode, pad_profile_sha256=None,
                           blockers=[{"code": "pad_validation_required", "message": "PAD unavailable"}])
    status = client.get("/api/status").json()["auth"]
    assert not status["ready"] and status["enrollment_ready"]
    assert status["enrollment_pad_enabled"] is False and status["pad_enabled"] is True
    assert client.post("/api/auth/authenticate", json=CONSENT, headers=HEADERS).status_code == 503
    response = client.post("/api/auth/enroll", json=CONSENT, headers=HEADERS)
    assert response.status_code == 202
    assert response.json()["enrollment_pad_enabled"] is False
    outcome = finish(client, response.json())
    assert outcome["state"] == "succeeded"
    assert outcome["result"]["enrollment_pad_enabled"] is False


@pytest.mark.parametrize("failure", ["search", "stale", "camera", "calibration"])
def test_terminal_enrollment_readiness_keeps_non_pad_dependencies(setup, failure):
    client, service, backend = setup
    signup(client)
    service.backend.update(enrollment_ready=True, enrollment_blockers=[])
    if failure == "search":
        service.backend.update(enrollment_ready=False, enrollment_blockers=[{"code": "search_unavailable"}])
    elif failure == "stale":
        service.backend_at = 0
    elif failure == "camera":
        service.camera.state = "error"
    else:
        service.backend["calib_version"] = "different-sensor"
    status = client.get("/api/status").json()["auth"]
    assert not status["enrollment_ready"] and status["enrollment_blockers"]
    assert client.post("/api/auth/enroll", json=CONSENT, headers=HEADERS).status_code == 503


def test_terminal_does_not_forward_obsolete_enrollment_pad_feedback(setup):
    client, _, backend = setup
    signup(client)
    backend.enroll = lambda *args, **kwargs: {"status": "rejected", "reason": "denied",
                                             "enrollment_feedback": "depth_check", "score": .123}
    response = client.post("/api/auth/enroll", json=CONSENT, headers=HEADERS)
    outcome = finish(client, response.json())
    assert outcome["state"] == "rejected"
    assert outcome["result"]["enrollment_pad_enabled"] is False
    assert "깊이 검사" not in outcome["message"]
    assert "enrollment_feedback" not in outcome["result"] and "score" not in outcome["result"]


def test_auth_session_cannot_be_repurposed_for_pad_free_enrollment(enrollment_api, monkeypatch):
    state, request = enrollment_api

    async def auth_session_row(*args):
        return {"device_id": "synthetic-device", "purpose": "auth", "unconsumed": True,
                "live": True, "challenge": "synthetic-challenge"}

    monkeypatch.setattr(api, "_session", ORIGINAL_SESSION)
    monkeypatch.setattr(api, "_one", auth_session_row)
    result = asyncio.run(api.enroll(request))
    assert result["status"] == "rejected" and result["reason"] == "session"
    assert state["analyses"] == 0 and state["stored"] == 0


def test_enrollment_api_does_not_depend_on_experimental_profile(enrollment_api, monkeypatch):
    state, request = enrollment_api
    monkeypatch.setattr(api.C, "cfg", SimpleNamespace(pad_enabled=True,
        pad_mode="femto_geometry_experimental", pad_profile_sha256="a" * 64, min_dirs=9, t_same=.6, t_dup=.9))
    state["analysis"].update(pad_mode="femto_geometry_experimental",
                             pad_experiment={"ready": False, "profile_sha256": None})
    result = json.loads(asyncio.run(api.enroll(request)).body)
    assert result["status"] == "active" and result["enrollment_pad_enabled"] is False
    assert state["stored"] == 1
