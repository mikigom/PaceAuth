"""Selection and isolation contracts for live recognition and largest-person authentication."""
import asyncio
import queue
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import msgpack
import numpy as np
import pytest

from faceauth.engine import main as engine


def face(box, score=0.9):
    x1, y1, x2, y2 = box
    return {"bbox": np.array(box, np.float32), "score": score,
            "kps": np.array([[x1 + (x2 - x1) * x, y1 + (y2 - y1) * y]
                             for x, y in [(0.3, 0.3), (0.7, 0.3), (0.5, 0.5), (0.3, 0.7), (0.7, 0.7)]],
                            np.float32)}


@pytest.fixture
def measured_models():
    m = engine.Models.__new__(engine.Models)
    m.dets = queue.Queue()
    detector = SimpleNamespace(detect=Mock(return_value=[]))
    m.dets.put(detector)
    m.pool = ThreadPoolExecutor(2)
    m.versions = {"recog": "test@1", "pad": "test@1"}
    m.pad_validation = SimpleNamespace(binding={"test": "binding"}, status=lambda _: {"ready": True})
    m.pose = SimpleNamespace(crop="demo", predict_crops=Mock(side_effect=lambda crops, _: np.zeros((len(crops), 3))))
    m.rec = SimpleNamespace(dim=2, embed=Mock(side_effect=lambda rgbs, kps: np.array(
        [[points[0, 0], 1] for points in kps], np.float32)))
    m.pad = SimpleNamespace(cnn=Mock(side_effect=AssertionError("live must never infer PAD")))
    rgb = np.random.default_rng(42).integers(70, 180, (240, 320, 3), dtype=np.uint8)
    decoded = (rgb, np.full((240, 320), 600, np.uint16), np.full((240, 320), 120, np.uint8))
    yield m, detector, decoded
    m.pool.shutdown()


@pytest.mark.parametrize("mode", ["auth", "guide", "observe"])
def test_auth_always_selects_largest_not_highest_confidence(measured_models, mode):
    m, detector, decoded = measured_models
    largest, smaller = face([100, 20, 250, 200], 0.6), face([5, 30, 85, 120], 0.99)
    detector.detect.return_value = [smaller, largest]
    context = {"mode": mode, "capture_purpose": "auth"}
    frame, pairs = m._prepare_frame((context, decoded))
    assert frame["bbox"] == largest["bbox"].tolist()
    assert frame["n_faces"] == 1 and frame["detected_faces"] == 2 and frame["selection"] == "largest"
    assert len(pairs) == 1


def test_equal_area_selection_is_stable_when_detector_order_changes(measured_models):
    m, detector, decoded = measured_models
    left, right = face([20, 40, 100, 140], 0.6), face([200, 20, 280, 120], 0.99)
    selected = []
    for faces in ([right, left], [left, right]):
        detector.detect.return_value = faces
        selected.append(m._frame(decoded, "auth")[0]["bbox"])
    assert selected == [left["bbox"].tolist()] * 2


def test_selection_uses_visible_area_and_drops_invalid_or_offscreen_faces(measured_models):
    m, detector, decoded = measured_models
    edge, inside = face([-100, -20, 80, 180]), face([100, 20, 260, 200])
    outside, tiny = face([330, 20, 410, 150]), face([-100, 20, 20, 150])
    invalid = face([0, 20, 100, 150])
    invalid["bbox"][2] = np.nan
    detector.detect.return_value = [edge, outside, invalid, inside, tiny]
    selected = m._frame(decoded, "auth")[0]
    assert selected["bbox"] == inside["bbox"].tolist() and selected["detected_faces"] == 2
    result = m.process([({"mode": "live"}, [decoded])])[0][0]
    assert result["n_faces"] == 2
    assert [f["bbox"] for f in result["faces"]] == [inside["bbox"].tolist(), [0, 0, 80, 180]]
    assert edge["bbox"].tolist() == [-100, -20, 80, 180]  # detector-owned arrays remain untouched


@pytest.mark.parametrize("depth", [1800, 3100])
def test_live_group_distance_does_not_relax_authentication(measured_models, monkeypatch, depth):
    monkeypatch.setattr(engine, "PAD_ENABLED", False)
    m, detector, decoded = measured_models
    detector.detect.return_value = [face([20, 20, 170, 210])]
    decoded[1][:] = depth
    live, auth, enroll = m.process([({"mode": "live"}, [decoded]), ({"mode": "auth"}, [decoded]),
                                   ({"mode": "enroll"}, [decoded])])
    result = live[0]["faces"][0]
    assert result["ok"] is (depth == 1800)
    assert (result["emb"] is not None) is (depth == 1800)
    for selected in (auth[0], enroll[0]):
        assert not selected["ok"] and "depth_invalid" in selected["reasons"]


def test_invalid_largest_does_not_fall_back_to_smaller_valid_person(measured_models, monkeypatch):
    monkeypatch.setattr(engine, "PAD_ENABLED", False)
    m, detector, decoded = measured_models
    largest, smaller = face([150, 10, 310, 220]), face([10, 40, 100, 150])
    detector.detect.return_value = [smaller, largest]
    decoded[1][10:220, 150:310] = 0
    result = m.process([({"mode": "auth"}, [decoded])])[0][0]
    assert result["bbox"] == largest["bbox"].tolist()
    assert not result["ok"] and "depth_invalid" in result["reasons"]
    assert m.rec.embed.call_args.args[1][0].tolist() == largest["kps"].tolist()


@pytest.mark.parametrize("mode", ["enroll", "guide", "observe"])
def test_enrollment_stays_strict_when_multiple_people_enter(measured_models, mode):
    m, detector, decoded = measured_models
    detector.detect.return_value = [face([10, 20, 90, 140]), face([150, 20, 300, 200])]
    result = m.process([({"mode": mode, "capture_purpose": "enroll"}, [decoded])])[0][0]
    assert result["n_faces"] == 2 and result["reasons"] == ["multi_face"] and not result["ok"]
    assert result["emb"] is None
    m.pose.predict_crops.assert_not_called()
    m.rec.embed.assert_not_called()
    m.pad.cnn.assert_not_called()


def test_live_faces_have_independent_embeddings_without_pad_or_validation(measured_models):
    m, detector, decoded = measured_models
    m.pad_validation.status = lambda _: {"ready": False}
    detector.detect.return_value = [face([10, 20, 90, 140]), face([150, 20, 300, 200])]
    result = m.analyze_batch([({"mode": "live", "calib_version": "bolt"}, [decoded])])[0]
    frame = result["frames"][0]
    assert frame["n_faces"] == 2 and frame["image_size"] == [320, 240] and not frame["truncated"]
    assert all(f["ok"] and f["pad"] is None for f in frame["faces"])
    assert len(set(f["emb"] for f in frame["faces"])) == 2
    assert not result["pad_validation"]["ready"]
    m.pad.cnn.assert_not_called()


def test_live_bad_quality_keeps_box_but_has_no_identity_embedding(measured_models):
    m, detector, decoded = measured_models
    detector.detect.return_value = [face([150, 10, 310, 220]), face([10, 40, 100, 150])]
    decoded[1][10:220, 150:310] = 0
    result = m.process([({"mode": "live"}, [decoded])])[0][0]
    bad, good = result["faces"]
    assert "depth_invalid" in bad["reasons"] and bad["bbox"] and bad["emb"] is None and not bad["ok"]
    assert good["ok"] and good["emb"] is not None
    assert len(m.rec.embed.call_args.args[0]) == 1


def test_live_pose_failure_has_box_but_no_identity_embedding(measured_models):
    m, detector, decoded = measured_models
    detector.detect.return_value = [face([20, 20, 170, 210])]
    m.pose.predict_crops.side_effect = lambda crops, _: np.full((len(crops), 3), 40)
    result = m.process([({"mode": "live"}, [decoded])])[0][0]["faces"][0]
    assert not result["ok"] and result["reasons"] == ["pose_out_of_range"] and result["emb"] is None
    m.rec.embed.assert_not_called()


def test_live_face_limit_is_explicit_and_keeps_largest_first(measured_models):
    m, detector, decoded = measured_models
    detector.detect.return_value = [face([i, 10, i + 80, 110]) for i in range(20)]
    result = m.process([({"mode": "live"}, [decoded])])[0][0]
    assert result["n_faces"] == 20 and result["truncated"] and len(result["faces"]) == engine.MAX_LIVE_FACES
    assert [f["bbox"][0] for f in result["faces"]] == list(range(16))


def test_live_empty_decode_failure_and_mixed_batch_stay_separate(measured_models):
    m, detector, decoded = measured_models
    result = m.process([({"mode": "live"}, [decoded]), ({"mode": "live"}, ["decode_error"]),
                        ({"mode": "enroll"}, [decoded])])
    assert result[0][0]["ok"] and result[0][0]["faces"] == [] and result[0][0]["image_size"] == [320, 240]
    assert not result[1][0]["ok"] and result[1][0]["reasons"] == ["decode_error"]
    assert not result[2][0]["ok"] and result[2][0]["reasons"] == ["no_face"]
    m.rec.embed.assert_not_called()


@pytest.mark.parametrize("count", [1, 2])
def test_live_endpoint_accepts_exactly_one_frame(measured_models, monkeypatch, count):
    analyze = AsyncMock(return_value={"frames": [{"faces": [], "n_faces": 0}]})
    monkeypatch.setattr(engine.E, "analyze", analyze)
    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=engine.app), base_url="http://engine") as client:
            response = await client.post("/v1/analyze", content=msgpack.packb({"mode": "live", "calib_version": "bolt",
                "pad_enabled": engine.PAD_ENABLED, "pad_mode": engine.PAD_MODE, "frames": [{}] * count}))
            assert response.status_code == (200 if count == 1 else 400)
            assert analyze.call_count == (1 if count == 1 else 0)
    asyncio.run(check())
