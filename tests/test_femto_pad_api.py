"""An experimental depth profile cannot borrow the production PAD release gate."""
import asyncio
from copy import deepcopy
from types import SimpleNamespace

import httpx
import pytest

from faceauth.api import app as api
from faceauth.api.clients import Deps, Unavailable

PIN = "a" * 64


def pilot_payload():
    return {"ready": True, "capture_ready": True, "versions": {"recog": "fixture@1"},
            "calib_version": "bolt", "pad_enabled": True, "pad_mode": "femto_geometry_pilot",
            "pad_validation": {"ready": False, "calib_version": "bolt",
                               "reasons": ["pad_real_sensor_validation_missing"]},
            "pad_pilot_validation": {"ready": True, "experimental": True,
                                     "scope": ["print", "replay"], "calib_version": "bolt",
                                     "profile_sha256": PIN, "reasons": []}}


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setattr(api.C, "cfg", SimpleNamespace(pad_enabled=True, pad_mode="femto_geometry_pilot",
                                                      pad_profile_sha256=PIN))


def test_pilot_requires_its_own_profile_and_leaves_production_gate_false(configured):
    body = pilot_payload()
    assert api._pad_required(body)
    assert api._pad_validated(body, "bolt")
    assert body["pad_validation"]["ready"] is False
    assert api._pad_mode()["pad_profile_sha256"] == PIN
    assert not api._pad_validated(body, "other-camera")


@pytest.mark.parametrize("key,value", [("ready", False), ("experimental", False),
    ("scope", ["print", "replay", "mask_rigid"]), ("calib_version", "simulator"),
    ("profile_sha256", "b" * 64)])
def test_pilot_rejects_incomplete_or_mismatched_evidence(configured, key, value):
    body = pilot_payload()
    body["pad_pilot_validation"][key] = value
    assert not api._pad_validated(body, "bolt")
    if key == "profile_sha256":
        with pytest.raises(Unavailable, match="pad_profile_mismatch"):
            api._pad_required(body)


def test_pilot_cannot_borrow_production_release(configured):
    body = pilot_payload()
    body.pop("pad_pilot_validation")
    body["pad_validation"]["ready"] = True
    assert not api._pad_validated(body, "bolt")


def test_production_cannot_borrow_pilot_evidence(monkeypatch):
    monkeypatch.setattr(api.C, "cfg", SimpleNamespace(pad_enabled=True))
    assert not api._pad_validated(pilot_payload(), "bolt")


@pytest.mark.parametrize("variant", ["valid", "wrong_pin", "no_pin", "wrong_scope", "not_ready"])
def test_readiness_checks_profile_pin_scope_and_readiness(variant):
    async def check():
        pin = "" if variant == "no_pin" else PIN
        deps = Deps(["http://engine"], ["http://search"], "http://kms", "k", "a",
                    pad_mode="femto_geometry_pilot", pad_profile_sha256=pin)
        await deps.http.aclose()
        body = pilot_payload()
        if variant == "wrong_pin":
            body["pad_pilot_validation"]["profile_sha256"] = "b" * 64
        elif variant == "wrong_scope":
            body["pad_pilot_validation"]["scope"] = ["print"]
        elif variant == "not_ready":
            body["pad_pilot_validation"]["ready"] = False
        deps.http = httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(
            200, json=deepcopy(body) if req.url.host == "engine" else
            {"ready": True, "model_version": "fixture@1"})))
        try:
            result = await deps.readiness("bolt", "fixture@1")
            assert result["ready"] is (variant == "valid")
            if variant == "valid":
                assert result["pad_profile_sha256"] == PIN
                assert result["pad_validation"]["ready"] is False
        finally:
            await deps.close()
    asyncio.run(check())
