"""Synthetic endpoint contracts for numeric-only PAD candidate diagnostics."""
import asyncio
import copy
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from faceauth.api import app as api
from faceauth.api.clients import Unavailable
from faceauth.common.pad_validation import CANDIDATE_FEATURES


@pytest.fixture
def observation(monkeypatch):
    binding = {"schema": "faceauth.femto_pad.binding.v1", "feature_schema": "faceauth.femto_pad.features.v1",
               "feature_code_sha256": "a" * 64, "detector_model_json_sha256": "b" * 64,
               "pipeline_sha256": "c" * 64}
    candidate = {"schema": binding["feature_schema"], "valid": True, "reasons": [],
                 "features": dict.fromkeys(CANDIDATE_FEATURES, 1.0)}
    data = {"versions": {}, "model_binding": {"fixture": "old-cnn"}, "calib_version": "test-calib",
            "candidate_binding": binding, "frames": [{"ok": True, "quality": {}, "reasons": [],
                "pad": {"score": 0.0, "live": False, "nonresponse": False}, "pad_candidate": candidate,
                "emb": b"never-expose", "rgb": b"never-expose"}]}

    @asynccontextmanager
    async def transaction():
        yield None

    @asynccontextmanager
    async def connection():
        yield SimpleNamespace(transaction=transaction)

    async def capture(*args, **kwargs):
        return ({"capture_purpose": "auth"}, [{}], {"id": "synthetic-session"},
                {"id": "synthetic-device", "calib_version": "test-calib"}, "fake-recog")

    async def consume(*args, **kwargs):
        return True

    async def audit(*args, **kwargs):
        pass

    async def analyze(*args, **kwargs):
        return copy.deepcopy(data)

    monkeypatch.setattr(api, "_capture_packet", capture)
    monkeypatch.setattr(api, "_consume", consume)
    monkeypatch.setattr(api, "audit", audit)
    monkeypatch.setattr(api, "_actor", lambda *_: "synthetic-device")
    monkeypatch.setattr(api, "_check_model", lambda *_: None)
    monkeypatch.setattr(api.C, "pool", SimpleNamespace(connection=connection))
    monkeypatch.setattr(api.C, "deps", SimpleNamespace(analyze=analyze))
    return data


def test_observe_forwards_numeric_candidate_without_images_or_live_claim(observation):
    observation["candidate_binding"]["secret"] = "never-expose"
    observation["frames"][0]["pad_candidate"].update(rgb=b"never-expose", live=True)
    observation["frames"][0]["pad_candidate"]["features"]["landmarks"] = [1, 2, 3]
    result = asyncio.run(api.capture_observe(None))
    assert result["diagnostic"] is True
    assert "secret" not in result["candidate_binding"]
    frame = result["frames"][0]
    assert set(frame) == {"ok", "reasons", "quality", "pad", "pad_candidate"}
    assert set(frame["pad_candidate"]) == {"schema", "valid", "reasons", "features"}
    assert set(frame["pad_candidate"]["features"]) == set(CANDIDATE_FEATURES)
    assert frame["pad_candidate"]["valid"] is True and frame["pad"]["live"] is False


@pytest.mark.parametrize("damage", ["missing_binding", "bad_binding", "invalid_feature"])
def test_observe_rejects_unbound_or_malformed_candidate(observation, damage):
    if damage == "missing_binding":
        observation.pop("candidate_binding")
    elif damage == "bad_binding":
        observation["candidate_binding"]["feature_code_sha256"] = "unknown"
    else:
        observation["frames"][0]["pad_candidate"]["features"]["valid_ratio"] = {"image": "never-expose"}
    with pytest.raises(Unavailable):
        asyncio.run(api.capture_observe(None))


def test_legacy_observation_without_candidate_remains_supported(observation):
    observation.pop("candidate_binding")
    observation["frames"][0].pop("pad_candidate")
    result = asyncio.run(api.capture_observe(None))
    assert result["candidate_binding"] is None and "pad_candidate" not in result["frames"][0]
