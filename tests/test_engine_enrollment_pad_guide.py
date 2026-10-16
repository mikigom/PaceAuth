"""Enrollment skips PAD; authentication and explicit observations retain their own gates."""
import hashlib
from unittest.mock import Mock

import pytest

from faceauth.api import policy
from faceauth.common.femto_pad_experiment import (
    FEATURE_NAMES, FEATURE_SCHEMA, FemtoPADExperiment, build_experiment,
)
from faceauth.common.pad_validation import canonical_json
from faceauth.engine import main as engine
from faceauth.terminal.guide import Guide
from test_engine_capture import models


def measurement(plane=10.0, nose=20.0):
    features = dict.fromkeys(FEATURE_NAMES, 1.0)
    features.update(plane_residual_mm=plane, nose_protrusion_mm=nose, depth_median_mm=600.0)
    return {"schema": FEATURE_SCHEMA, "valid": True, "features": features, "reasons": []}


@pytest.fixture
def experimental(monkeypatch, tmp_path):
    monkeypatch.setattr(engine, "PAD_MODE", "femto_geometry_experimental")
    monkeypatch.setattr(engine, "PAD_ENABLED", True)
    m = models(monkeypatch)
    raw = canonical_json(build_experiment(binding=m.candidate_binding, calib_version="bolt"))
    path = tmp_path / "synthetic-experiment.json"
    path.write_bytes(raw)
    m.pad_experiment = FemtoPADExperiment(path, hashlib.sha256(raw).hexdigest())
    m._candidate = Mock(return_value=measurement())
    m.pose.predict_crops.return_value = [[0, 0, 0]]
    yield m
    m.pool.shutdown()


def analyze(m, mode="guide", purpose="enroll"):
    return m.analyze_batch([({"mode": mode, "capture_purpose": purpose, "calib_version": "bolt"}, [None])])[0]


@pytest.mark.parametrize("mode", ["guide", "enroll"])
@pytest.mark.parametrize("pad_mode", ["enabled", "disabled_pilot", "femto_geometry_pilot", "femto_geometry_experimental"])
def test_enrollment_skips_every_pad_model_and_validation_gate(experimental, monkeypatch, mode, pad_mode):
    m = experimental
    monkeypatch.setattr(engine, "PAD_MODE", pad_mode)
    monkeypatch.setattr(engine, "PAD_ENABLED", pad_mode != "disabled_pilot")
    m.pad_experiment = m.pad_pilot = None
    m._candidate.side_effect = AssertionError("enrollment must not measure PAD")
    frame = analyze(m, mode)["frames"][0]
    assert frame["ok"] is True and frame["reasons"] == [] and frame["pad"] is None
    assert (frame["emb"] is not None) is (mode == "enroll")
    assert "pad_candidate" not in frame
    m._candidate.assert_not_called()
    m.pad.cnn.assert_not_called()
    m.pad.decide.assert_not_called()
    m.pose.predict_crops.assert_called_once()


@pytest.mark.parametrize("mode", ["guide", "enroll"])
def test_enrollment_preserves_frame_quality_rejection(experimental, mode):
    m = experimental
    original = m._frame

    def blurry(*args):
        frame, face = original(*args)
        frame["reasons"] = ["blur"]
        return frame, face

    m._frame = blurry
    frame = analyze(m, mode)["frames"][0]
    assert frame["ok"] is False and frame["reasons"] == ["blur"]
    assert frame["pad"] is None
    m._candidate.assert_not_called()


def test_enrollment_guide_collects_without_pad_depth_preflight(experimental):
    m = experimental
    m._candidate.side_effect = AssertionError("enrollment guide must not run PAD")
    frame = analyze(m)["frames"][0]
    guide = Guide(mode="circle")
    for instant in (0.0, 1.0, 2.0, 3.0):
        state = guide.feed(frame["pose"], frame["ok"], "synthetic-ref", frame["score"],
                           t=instant, reasons=frame["reasons"])
    assert state["coverage"]["C"] == (2, 2)
    assert len(guide.frames()) == 2


@pytest.mark.parametrize("ready", [False, True])
def test_authentication_cannot_use_enrollment_purpose_to_skip_pad(experimental, ready):
    m = experimental
    m._candidate.return_value = measurement(nose=0.0)
    if not ready:
        m.pad_experiment = None
    out = analyze(m, "auth", purpose="enroll")
    frame = out["frames"][0]
    if ready:
        assert frame["pad"]["live"] is False
        assert not policy.pad_live(out["frames"])
        m._candidate.assert_called_once()
    else:
        assert frame["reasons"] == ["pad_validation_required"]
        m._candidate.assert_not_called()
        m.rec.embed.assert_not_called()


@pytest.mark.parametrize("purpose", ["enroll", "auth"])
def test_observation_retains_failed_measurements(experimental, purpose):
    m = experimental
    m._candidate.return_value = {"schema": FEATURE_SCHEMA, "valid": False,
                                 "features": {}, "reasons": ["nose_coverage"]}
    m.pad_experiment = None
    frame = analyze(m, "observe", purpose)["frames"][0]
    assert frame["ok"] is True and frame["reasons"] == []
    assert frame["pad_candidate"]["reasons"] == ["nose_coverage"]
    assert frame["pad"] is None and frame["emb"] is None
    m.rec.embed.assert_not_called()
