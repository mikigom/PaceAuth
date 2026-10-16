"""Experimental enforcement never claims empirical validation or borrows a gate.

All frames, responses and identities here are explicit synthetic fixtures.
"""
import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from faceauth.api import app as api, policy
from faceauth.api.clients import Deps, Unavailable
from faceauth.common import token
from faceauth.engine import main as engine
from test_engine_capture import models
from test_open_console import connect, console, create
from test_web_auth import CONSENT, HEADERS, finish, setup

MODE = "femto_geometry_experimental"
PIN = "a" * 64
CLAIMS = {"pad_enabled": True, "pad_mode": MODE, "pad_profile_sha256": PIN}


def status(ready=True):
    return {"ready": ready, "experimental": True, "production_validated": False,
            "empirical_validation": "not_evaluated", "scope": ["print", "replay"],
            "calib_version": "bolt", "profile_sha256": PIN,
            "reasons": [] if ready else ["profile_binding_mismatch"]}


def payload():
    return {"ready": True, "capture_ready": True, "versions": {"recog": "fixture@1"},
            "calib_version": "bolt", **CLAIMS, "pad_experiment": status(),
            "pad_validation": {"ready": False, "calib_version": "bolt",
                               "reasons": ["pad_real_sensor_validation_missing"]}}


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setattr(api.C, "cfg", SimpleNamespace(pad_enabled=True, pad_mode=MODE, pad_profile_sha256=PIN))


def test_explicit_mode_enforces_pad_without_changing_empirical_validation(configured):
    body = payload()
    assert api._pad_required(body) and api._pad_validated(body, "bolt")
    assert api._pad_mode() == CLAIMS
    assert body["pad_validation"]["ready"] is False
    assert body["pad_experiment"]["empirical_validation"] == "not_evaluated"


@pytest.mark.parametrize("key,value", [
    ("ready", False), ("ready", None), ("experimental", False),
    ("production_validated", True), ("production_validated", None),
    ("empirical_validation", "passed"), ("empirical_validation", None),
    ("calib_version", "other"), ("scope", ["print"]), ("profile_sha256", "b" * 64),
])
def test_experiment_missing_or_mismatched_configuration_never_passes(configured, key, value):
    body = payload()
    body["pad_experiment"][key] = value
    assert not api._pad_validated(body, "bolt")
    if key == "profile_sha256":
        with pytest.raises(Unavailable, match="pad_profile_mismatch"):
            api._pad_required(body)


@pytest.mark.parametrize("other_mode", ["enabled", "femto_geometry_pilot", "disabled_pilot"])
def test_no_cross_mode_gate_borrowing(configured, monkeypatch, other_mode):
    body = payload()
    body.pop("pad_experiment")
    body.update(pad_validation={"ready": True, "calib_version": "bolt"}, pad_pilot_validation=status())
    assert not api._pad_validated(body, "bolt")
    body = payload()
    monkeypatch.setattr(api.C, "cfg", SimpleNamespace(
        pad_enabled=other_mode != "disabled_pilot", pad_mode=other_mode, pad_profile_sha256=PIN))
    with pytest.raises(Unavailable, match="pad_mode_mismatch"):
        api._pad_required(body)
    if other_mode != "disabled_pilot":
        assert not api._pad_validated(body, "bolt")


@pytest.mark.parametrize("variant", ["valid", "no_pin", "wrong_pin", "missing_status", "not_ready", "calib",
                                    "mode", "disabled", "validated_claim", "production_claim"])
def test_dependency_readiness_is_explicit_and_fails_closed(variant):
    async def run():
        deps = Deps(["http://engine"], ["http://search"], "http://kms", "k", "a", pad_mode=MODE,
                    pad_profile_sha256="" if variant == "no_pin" else PIN)
        await deps.http.aclose()
        body = payload()
        if variant == "wrong_pin":
            body["pad_experiment"]["profile_sha256"] = "b" * 64
        elif variant == "missing_status":
            body.pop("pad_experiment")
        elif variant == "not_ready":
            body["pad_experiment"] = status(False)
        elif variant == "calib":
            body["pad_experiment"]["calib_version"] = "other"
        elif variant == "mode":
            body["pad_mode"] = "femto_geometry_pilot"
        elif variant == "disabled":
            body["pad_enabled"] = False
        elif variant == "validated_claim":
            body["pad_experiment"]["empirical_validation"] = "passed"
        elif variant == "production_claim":
            body["pad_experiment"]["production_validated"] = True
        deps.http = httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(
            200, json=deepcopy(body) if req.url.host == "engine" else
            {"ready": True, "model_version": "fixture@1"})))
        try:
            result = await deps.readiness("bolt", "fixture@1")
            assert result["ready"] is (variant == "valid")
            if variant == "valid":
                assert result["pad_profile_sha256"] == PIN and result["pad_enabled"] is True
                assert result["pad_validation"]["ready"] is False
                assert result["pad_experiment"]["empirical_validation"] == "not_evaluated"
        finally:
            await deps.close()
    asyncio.run(run())


def test_experimental_mode_cannot_be_configured_as_disabled():
    with pytest.raises(ValueError, match="inconsistent PAD mode"):
        Deps([], [], "", "", "", pad_mode=MODE, pad_enabled=False, pad_profile_sha256=PIN)


@pytest.mark.parametrize("change", ["mode", "pin", "disabled"])
def test_analysis_refuses_a_mixed_replica_response(change):
    async def run():
        deps = Deps(["http://engine"], [], "", "", "", pad_mode=MODE, pad_profile_sha256=PIN)
        await deps.http.aclose()
        body = payload()
        if change == "mode":
            body["pad_mode"] = "femto_geometry_pilot"
        elif change == "pin":
            body["pad_experiment"]["profile_sha256"] = "b" * 64
        else:
            body["pad_enabled"] = False
        deps.http = httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200, json=body)))
        try:
            with pytest.raises(Unavailable, match="pad_.*_mismatch"):
                await deps.analyze("auth", [{key: b"synthetic" for key in ("rgb", "depth", "ir")}], "bolt")
        finally:
            await deps.close()
    asyncio.run(run())


@pytest.mark.parametrize("ready", [False, True])
def test_engine_readiness_exposes_configuration_without_claiming_validation(monkeypatch, ready):
    monkeypatch.setattr(engine, "PAD_MODE", MODE)
    monkeypatch.setattr(engine, "PAD_ENABLED", True)
    m = models(monkeypatch, ready=False)
    m.pad_experiment = SimpleNamespace(status=Mock(return_value=status(ready)))
    monkeypatch.setattr(engine.E, "models", m)
    try:
        result = asyncio.run(engine.readiness("bolt"))
        assert result["ready"] is ready and result["capture_ready"] is True
        assert result["pad_enabled"] is True and result["pad_mode"] == MODE
        assert result["pad_validation"]["ready"] is False
        assert result["pad_experiment"]["empirical_validation"] == "not_evaluated"
        assert "pad_pilot_validation" not in result
    finally:
        m.pool.shutdown()


@pytest.mark.parametrize("mode", ["auth", "enroll"])
@pytest.mark.parametrize("ready", [False, True])
def test_engine_enforces_geometry_and_leaves_cnn_and_production_gate_unused(monkeypatch, mode, ready):
    monkeypatch.setattr(engine, "PAD_MODE", MODE)
    monkeypatch.setattr(engine, "PAD_ENABLED", True)
    m = models(monkeypatch, ready=False)
    m.pad_experiment = SimpleNamespace(status=Mock(return_value=status(ready)), decide=Mock(return_value={
        "score": 0.0, "live": False, "nonresponse": False, "failed": ["nose_protrusion_mm"]}))
    try:
        out = m.analyze_batch([({"mode": mode, "calib_version": "bolt"}, [None])])[0]
        assert out["pad_enabled"] is True and out["pad_mode"] == MODE
        assert out["pad_validation"]["ready"] is False and "pad_pilot_validation" not in out
        assert out["pad_experiment"]["ready"] is ready
        assert out["versions"]["pad"] == MODE + "@" + PIN
        m.pad.cnn.assert_not_called()
        m.pad.decide.assert_not_called()
        if mode == "enroll":
            assert out["frames"][0]["pad"] is None
            m.pad_experiment.decide.assert_not_called()
            m.rec.embed.assert_called_once()
        elif ready:
            m.pad_experiment.decide.assert_called_once()
            assert out["frames"][0]["pad"]["live"] is False
            assert not policy.pad_live(out["frames"])
        else:
            m.pad_experiment.decide.assert_not_called()
            m.rec.embed.assert_not_called()
            assert out["frames"][0]["reasons"] == ["pad_validation_required"]
    finally:
        m.pool.shutdown()


def test_missing_experiment_profile_blocks_final_capture_but_not_diagnostics(monkeypatch):
    monkeypatch.setattr(engine, "PAD_MODE", MODE)
    monkeypatch.setattr(engine, "PAD_ENABLED", True)
    m = models(monkeypatch)
    try:
        final = m.analyze_batch([({"mode": "auth", "calib_version": "bolt"}, [None])])[0]
        assert final["frames"][0]["reasons"] == ["pad_validation_required"]
        assert final["pad_experiment"]["reasons"] == ["pad_experiment_profile_missing"]
        diagnostic = m.analyze_batch([({"mode": "observe", "capture_purpose": "auth", "calib_version": "bolt"}, [None])])[0]
        assert "pad_candidate" in diagnostic["frames"][0]
        m.pad.cnn.assert_not_called()
        m.rec.embed.assert_not_called()
    finally:
        m.pool.shutdown()


@pytest.mark.parametrize("bad", [None, {"live": False}, {"live": 1}, {"nonresponse": True}])
def test_one_failed_or_missing_frame_never_gets_a_majority_vote(bad):
    frames = [{"pad": {"live": True}} for _ in range(18)]
    frames[7]["pad"] = bad
    assert not policy.pad_live(frames)


@pytest.mark.parametrize("change", [None, "missing_claims", "claim_profile", "claim_mode", "claim_disabled",
                                    "response_profile", "response_mode"])
def test_targetless_identification_binds_actual_signed_experimental_approval(console, change):
    client, service, backend = console
    connect(client)
    create(client, "alice")
    service.backend.update(**CLAIMS, pad_experiment=status(), pad_validation={"ready": False})
    original = backend.authenticate

    def authenticate(*args, **kwargs):
        out = original(*args, **kwargs)
        claims = token.verify(backend.key.public_key(), out["approval_token"])
        if change != "missing_claims":
            claims.update(CLAIMS)
        if change == "claim_profile":
            claims["pad_profile_sha256"] = "b" * 64
        elif change == "claim_mode":
            claims["pad_mode"] = "femto_geometry_pilot"
        elif change == "claim_disabled":
            claims["pad_enabled"] = False
        out.update(CLAIMS, approval_token=token.sign(backend.key, claims))
        if change == "response_profile":
            out["pad_profile_sha256"] = "c" * 64
        elif change == "response_mode":
            out["pad_mode"] = "enabled"
        return out

    backend.authenticate = authenticate
    state = client.get("/api/status").json()["auth"]
    assert state["pad_experiment"]["empirical_validation"] == "not_evaluated"
    operation = client.post("/api/auth/identify", json={"start": True}, headers=HEADERS)
    assert operation.status_code == 202
    result = finish(client, operation.json())
    if change is None:
        assert result["state"] == "succeeded" and result["result"]["matched_account_id"] == "alice"
        assert result["result"]["pad_profile_sha256"] == PIN
    else:
        assert result["state"] == "error" and result["result"]["reason"] == "approval_scope_mismatch"
        assert "matched_account_id" not in result["result"]


@pytest.mark.parametrize("search_ready", [False, True])
def test_registration_readiness_is_independent_of_missing_pad_profile(search_ready):
    async def run():
        deps = Deps(["http://engine"], ["http://search"], "", "", "", pad_mode=MODE, pad_profile_sha256=PIN)
        await deps.http.aclose()
        body = payload()
        body["pad_experiment"] = {**status(False), "profile_sha256": None}
        deps.http = httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(
            200, json=deepcopy(body) if req.url.host == "engine" else
            {"ready": search_ready, "model_version": "fixture@1"})))
        try:
            result = await deps.readiness("bolt", "fixture@1")
            assert result["ready"] is False and result["pad_enabled"] is True
            assert result["enrollment_ready"] is search_ready and result["enrollment_pad_enabled"] is False
            assert bool(result["enrollment_blockers"]) is not search_ready
        finally:
            await deps.close()
    asyncio.run(run())


@pytest.mark.parametrize("mode,purpose,accepted", [
    ("enroll", None, True), ("guide", "enroll", True), ("auth", "enroll", False),
])
def test_missing_pad_profile_never_becomes_an_authentication_escape(mode, purpose, accepted):
    async def run():
        deps = Deps(["http://engine"], [], "", "", "", pad_mode=MODE, pad_profile_sha256=PIN)
        await deps.http.aclose()
        body = {**payload(), "dim": 2, "frames": [{}], "capture_purpose": purpose}
        body["pad_experiment"] = {**status(False), "profile_sha256": None}
        deps.http = httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200, json=body)))
        try:
            if accepted:
                result = await deps.analyze(mode, [{key: b"synthetic" for key in ("rgb", "depth", "ir")}],
                                            "bolt", capture_purpose=purpose)
                assert result["frames"] == [{}]
            else:
                with pytest.raises(Unavailable, match="pad_profile_mismatch"):
                    await deps.analyze(mode, [{key: b"synthetic" for key in ("rgb", "depth", "ir")}],
                                       "bolt", capture_purpose=purpose)
        finally:
            await deps.close()
    asyncio.run(run())
