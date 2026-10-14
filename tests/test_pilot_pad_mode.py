"""The explicit pilot opt-out skips PAD alone; missing/mismatched mode information fails closed."""
import asyncio
from types import SimpleNamespace

import httpx
import msgpack
import pytest

from faceauth.api import app as api
from faceauth.api.clients import Deps, Unavailable


def payload(enabled):
    return {"ready": False, "capture_ready": True, "versions": {"recog": "fixture@1"}, "dim": 2,
            "calib_version": "bolt", "pad_validation": {"ready": False, "calib_version": "bolt",
            "reasons": ["pad_real_sensor_validation_missing"]}, "frames": [{"pad": None}],
            "pad_enabled": enabled, "pad_mode": "enabled" if enabled else "disabled_pilot"}


@pytest.mark.parametrize("enabled", [True, False])
def test_readiness_requires_pad_by_default_but_explicit_pilot_skips_evidence(enabled):
    async def check():
        deps = Deps(["http://engine"], ["http://search"], "http://kms", "k", "a", pad_enabled=enabled)
        await deps.http.aclose()
        def handler(request):
            body = payload(enabled) if request.url.host == "engine" else {"ready": True, "model_version": "fixture@1"}
            return httpx.Response(200, json=body)
        deps.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            state = await deps.readiness("bolt", "fixture@1")
            assert state["ready"] is (not enabled) and state["capture_ready"] is True
            assert state["pad_enabled"] is enabled and state["pad_validation"]["ready"] is False
        finally:
            await deps.close()
    asyncio.run(check())


@pytest.mark.parametrize("enabled,reported", [(True, False), (False, True), (False, None), (True, "true")])
def test_mismatch_never_allows_analysis_or_capture_readiness(enabled, reported):
    async def check():
        deps = Deps(["http://engine"], ["http://search"], "http://kms", "k", "a", pad_enabled=enabled)
        await deps.http.aclose()
        body = payload(reported)
        deps.http = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body)))
        try:
            state = await deps.readiness("bolt", "fixture@1")
            assert state["ready"] is False and state["capture_ready"] is False
            assert state["blockers"][0]["code"] == "pad_mode_mismatch"
            with pytest.raises(Unavailable, match="pad_mode_mismatch"):
                await deps.analyze("auth", [{"rgb": b"r", "depth": b"d", "ir": b"i"}], "bolt")
        finally:
            await deps.close()
    asyncio.run(check())


def test_disabled_pilot_still_requires_search_and_transmits_mode():
    async def check():
        deps = Deps(["http://engine"], ["http://search"], "http://kms", "k", "a", pad_enabled=False)
        await deps.http.aclose()
        def handler(request):
            if request.method == "POST":
                sent = msgpack.unpackb(request.content, raw=False)
                assert sent["pad_enabled"] is False and sent["pad_mode"] == "disabled_pilot"
            return httpx.Response(200, json=payload(False) if request.url.host == "engine" else {"ready": False})
        deps.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            state = await deps.readiness("bolt", "fixture@1")
            assert not state["ready"] and state["blockers"][0]["code"] == "search_unavailable"
            result = await deps.analyze("auth", [{"rgb": b"r", "depth": b"d", "ir": b"i"}], "bolt")
            assert result["frames"][0]["pad"] is None
        finally:
            await deps.close()
    asyncio.run(check())


def test_api_independently_checks_the_mode_and_defaults_to_enabled(monkeypatch):
    monkeypatch.setattr(api.C, "cfg", SimpleNamespace())
    assert api._pad_required(payload(True)) is True
    with pytest.raises(Unavailable):
        api._pad_required(payload(False))
    monkeypatch.setattr(api.C, "cfg", SimpleNamespace(pad_enabled=False))
    assert api._pad_required(payload(False)) is False
    with pytest.raises(Unavailable):
        api._pad_required(payload(True))
