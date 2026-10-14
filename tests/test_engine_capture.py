"""CPU contract tests: guide/observe must never run recognition; missing PAD validation never approves."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import msgpack
import numpy as np
import pytest

from faceauth.engine import main as engine


@pytest.mark.parametrize("invalid", [False, True])
def test_warmup_exercises_all_models_and_rejects_nonfinite_outputs(invalid):
    m = engine.Models.__new__(engine.Models)
    m.det = SimpleNamespace(detect=Mock(return_value=[]))
    m.pose = SimpleNamespace(batch_size=4, predict_crops=Mock(return_value=np.zeros((4, 3))))
    m.rec = SimpleNamespace(batch_size=8, dim=512,
                           embed_aligned=Mock(return_value=np.ones((8, 512))))
    m.pad = SimpleNamespace(size=16, cnn=Mock(return_value=np.full(8, np.nan if invalid else 0.5)))
    if invalid:
        with pytest.raises(RuntimeError, match="warm-up returned invalid"):
            m._warmup()
    else:
        out = m._warmup()
        assert out["synthetic"] is True and out["pose_batch"] == 4 and out["recognition_batch"] == 8
    assert m.det.detect.call_count == m.pose.predict_crops.call_count == 1
    assert m.rec.embed_aligned.call_count == m.pad.cnn.call_count == 1


def models(monkeypatch, ready=False):
    m = engine.Models.__new__(engine.Models)
    m.versions = {"recog": "fixture@1", "pad": "fixture@1"}
    m.rec = SimpleNamespace(dim=2, embed=Mock(return_value=np.array([[1, 0]], np.float32)))
    m.pad_validation = SimpleNamespace(binding={"fixture": "binding"}, status=lambda calibration: {
        "ready": ready, "calib_version": calibration, "reasons": [] if ready else ["pad_validation_required"]})
    m.pool = ThreadPoolExecutor(1)
    m.pose = SimpleNamespace(crop="demo", predict_crops=Mock(return_value=[[30, 0, 0]]))
    m.pad = SimpleNamespace(size=16, zoom=1, fx=385, cnn=Mock(return_value=[0.9]), decide=Mock(return_value={
        "score": 0.0, "live": False, "nonresponse": True, "cnn": 0.9, "checks": {"depth": float("nan")},
        "failed": ["depth"]}))
    face = (np.zeros((16, 16, 3), np.uint8), np.ones((16, 16), np.uint16), np.ones((16, 16), np.uint8),
            {"bbox": [1, 1, 12, 12], "kps": np.zeros((5, 2), np.float32)})
    m._frame = lambda _, purpose="enroll": ({"reasons": [], "quality": {"blur": 120}, "det_score": 0.8,
                           "pad": None, "emb": None}, face)
    monkeypatch.setattr(engine, "face_features", lambda *args: (np.zeros((3, 16, 16)), {}))
    return m


def test_guide_enforces_auth_pose_and_skips_pad_and_embeddings(monkeypatch):
    m = models(monkeypatch)
    try:
        out = m.analyze_batch([({"mode": "guide", "capture_purpose": "auth", "calib_version": "bolt"}, [None])])[0]
        frame = out["frames"][0]
        assert not frame["ok"] and frame["reasons"] == ["pose_out_of_range"]
        assert frame["score"] == 0.8 and frame["pad"] is None and frame["emb"] is None
        assert not out["pad_validation"]["ready"]
        m.rec.embed.assert_not_called()
        m.pad.cnn.assert_not_called()
    finally:
        m.pool.shutdown()


@pytest.mark.parametrize("purpose", ["auth", "enroll"])
def test_observe_keeps_nonresponse_and_sanitizes_nonfinite_without_embedding(monkeypatch, purpose):
    m = models(monkeypatch)
    try:
        out = m.analyze_batch([({"mode": "observe", "capture_purpose": purpose, "calib_version": "bolt"}, [None])])[0]
        assert out["frames"][0]["pad"]["checks"]["depth"] is None
        assert out["frames"][0]["pad"]["nonresponse"] is True
        assert out["model_binding"] == m.pad_validation.binding
        assert out["capture_purpose"] == purpose
        assert out["frames"][0]["ok"] is (purpose == "enroll")  # same 30-degree yaw accepted only for enrollment
        m.rec.embed.assert_not_called()
        m.pad.cnn.assert_called_once()
    finally:
        m.pool.shutdown()


def test_auth_pose_rule_cannot_be_overridden_by_capture_purpose(monkeypatch):
    m = models(monkeypatch, ready=True)
    try:
        out = m.analyze_batch([({"mode": "auth", "capture_purpose": "enroll", "calib_version": "bolt"}, [None])])[0]
        assert out["frames"][0]["reasons"] == ["pose_out_of_range"]
        m.rec.embed.assert_called_once()
        assert out["frames"][0]["emb"] == np.array([1, 0], dtype="<f4").tobytes()
    finally:
        m.pool.shutdown()


@pytest.mark.parametrize("mode", ["auth", "enroll"])
def test_final_modes_require_validation_before_running_any_model(monkeypatch, mode):
    m = models(monkeypatch)
    m.process = Mock(side_effect=AssertionError("unvalidated frames must never reach inference"))
    try:
        out = m.analyze_batch([({"mode": mode, "calib_version": "bolt"}, [None])])[0]
        assert out["frames"][0]["reasons"] == ["pad_validation_required"]
        assert not out["frames"][0]["ok"]
    finally:
        m.pool.shutdown()


@pytest.mark.parametrize("mode", ["auth", "enroll"])
def test_disabled_pilot_skips_pad_but_retains_pose_quality_and_embeddings(monkeypatch, mode):
    monkeypatch.setattr(engine, "PAD_ENABLED", False)
    monkeypatch.setattr(engine, "PAD_MODE", "disabled_pilot")
    m = models(monkeypatch, ready=False)
    try:
        out = m.analyze_batch([({"mode": mode, "calib_version": "bolt"}, [None])])[0]
        assert out["pad_enabled"] is False and out["pad_mode"] == "disabled_pilot"
        assert out["pad_validation"]["ready"] is False
        frame = out["frames"][0]
        assert frame["pad"] is None and frame["emb"] is not None
        assert frame["ok"] is (mode == "enroll")  # 30 degree yaw still fails auth quality
        m.pad.cnn.assert_not_called()
        m.pad.decide.assert_not_called()
        m.rec.embed.assert_called_once()
        m.pose.predict_crops.assert_called_once()
    finally:
        m.pool.shutdown()


def test_disabled_pilot_diagnostic_still_measures_pad_without_embedding(monkeypatch):
    monkeypatch.setattr(engine, "PAD_ENABLED", False)
    monkeypatch.setattr(engine, "PAD_MODE", "disabled_pilot")
    m = models(monkeypatch, ready=False)
    try:
        out = m.analyze_batch([({"mode": "observe", "capture_purpose": "auth", "calib_version": "bolt"}, [None])])[0]
        assert out["frames"][0]["pad"] is not None and out["frames"][0]["emb"] is None
        m.pad.cnn.assert_called_once()
        m.rec.embed.assert_not_called()
    finally:
        m.pool.shutdown()


def test_engine_rejects_api_pilot_configuration_mismatch(monkeypatch):
    monkeypatch.setattr(engine, "PAD_ENABLED", False)
    monkeypatch.setattr(engine, "PAD_MODE", "disabled_pilot")
    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=engine.app), base_url="http://engine") as client:
            response = await client.post("/v1/analyze", content=msgpack.packb({
                "mode": "auth", "calib_version": "bolt", "frames": [{}]}))
            assert response.status_code == 503 and response.json()["detail"] == "pad_mode_mismatch"
    asyncio.run(check())


@pytest.mark.parametrize("calibration", [None, ""])
def test_analyze_rejects_missing_calibration(calibration):
    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=engine.app), base_url="http://engine") as client:
            response = await client.post("/v1/analyze", content=msgpack.packb({
                "mode": "auth", "calib_version": calibration, "frames": [{}]}))
            assert response.status_code == 400
    asyncio.run(check())


@pytest.mark.parametrize("purpose,count", [(None, 3), ("auth", 18), ("enroll", 2), ("enroll", 25)])
def test_observe_frame_limits_and_explicit_purpose(purpose, count):
    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=engine.app), base_url="http://engine") as client:
            response = await client.post("/v1/analyze", content=msgpack.packb({
                "mode": "observe", "calib_version": "bolt", "capture_purpose": purpose, "frames": [{}] * count}))
            assert response.status_code == 400
    asyncio.run(check())
