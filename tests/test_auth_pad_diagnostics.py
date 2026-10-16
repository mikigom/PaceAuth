"""Generated-frame contracts: internal counts never become biometric output or decisions."""
import asyncio
import json
from contextlib import asynccontextmanager
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from faceauth.api import app as api
from faceauth.common.femto_pad_experiment import FemtoPADExperiment, build_experiment
from faceauth.engine import main as engine
from test_engine_capture import models

MODE, PIN = "femto_geometry_experimental", "a" * 64


@pytest.fixture
def auth_api(monkeypatch):
    state = {"audits": [], "searches": 0, "approvals": 0, "consume": True, "integrity": []}
    frame = {"ok": True, "n_faces": 1, "reasons": [],
             "emb": np.array([1, 0], dtype="<f4").tobytes(),
             "pad": {"live": True, "nonresponse": False, "failed": [], "score": .98}}
    state["analysis"] = {"frames": [deepcopy(frame) for _ in range(3)], "dim": 2,
        "versions": {"recog": "fixture@1"}, "pad_enabled": True, "pad_mode": MODE,
        "calib_version": "bolt", "pad_experiment": {
            "ready": True, "experimental": True, "production_validated": False,
            "empirical_validation": "not_evaluated", "scope": ["print", "replay"],
            "profile_sha256": PIN, "calib_version": "bolt"}}

    @asynccontextmanager
    async def transaction():
        yield None

    @asynccontextmanager
    async def connection():
        yield SimpleNamespace(transaction=transaction)

    async def device(*args):
        return {"id": "synthetic-device", "calib_version": "bolt"}

    async def session(*args):
        return {"id": "synthetic-session", "action_id": "identify"}, None

    async def active(*args):
        return "fixture@1"

    async def analyze(*args):
        return deepcopy(state["analysis"])

    async def search(*args):
        state["searches"] += 1
        return [{"subject_id": "synthetic-subject", "score": .99}]

    async def locked(*args):
        return {"status": "active", "consent_version": "v1", "account_id": "synthetic-account"}, True

    async def consume(*args):
        return state["consume"]

    async def approve(*args):
        state["approvals"] += 1
        return "synthetic-approval"

    async def audit(*args, **kwargs):
        state["audits"].append(kwargs["detail"])

    async def body():
        return b"synthetic-packet"

    monkeypatch.setattr(api.C, "cfg", SimpleNamespace(pad_enabled=True, pad_mode=MODE, pad_profile_sha256=PIN,
        t_same=.7, t_id=.8, margin=.1, high_risk=set(), consent_version="v1"))
    monkeypatch.setattr(api.C, "pool", SimpleNamespace(connection=connection))
    monkeypatch.setattr(api.C, "deps", SimpleNamespace(analyze=analyze, search=search))
    monkeypatch.setattr(api, "_packet", lambda _: ({"device_id": "synthetic-device", "challenge": "fixture"},
                                                  b"fixture", [{}, {}, {}], "synthetic-session"))
    for name, value in {"_device": device, "_session": session, "_active_recog": active,
                        "_locked_subject": locked, "_consume": consume, "_approve": approve,
                        "audit": audit}.items():
        monkeypatch.setattr(api, name, value)
    monkeypatch.setattr(api, "_integrity", lambda *_: state["integrity"])
    return state, SimpleNamespace(body=body)


def test_failed_threshold_and_nonresponse_are_separate_in_private_audit(auth_api):
    state, request = auth_api
    frames = state["analysis"]["frames"]
    frames[1]["pad"] = {"live": False, "nonresponse": False, "failed": ["nose_protrusion_mm"]}
    frames[2]["pad"] = {"live": False, "nonresponse": True,
                         "failed": ["nose_coverage", "depth_blocks", "nose_coverage"]}
    result = asyncio.run(api.authenticate(request))
    assert result == {"decision": "rejected", "reason": "denied", "pad_enabled": True,
                      "pad_mode": MODE, "pad_profile_sha256": PIN}
    audit = state["audits"][0]
    assert audit["reason"] == "pad_fail" and state["searches"] == state["approvals"] == 0
    assert audit["pad_diagnostics"] == {
        "schema_version": 1, "total_frames": 3, "passed_frames": 1, "failed_frames": 2,
        "nonresponse_frames": 1, "unmeasured_frames": 0, "truncated": False,
        "reason_counts": {"nose_protrusion_mm": 1, "nose_coverage": 1, "depth_blocks": 1}}


@pytest.mark.parametrize("stage", ["approved", "quality", "validation", "same_person", "expired"])
def test_post_inference_paths_keep_diagnostics_private_and_policy_unchanged(auth_api, stage):
    state, request = auth_api
    if stage == "quality":
        state["analysis"]["frames"][0].update(ok=False, reasons=["depth_invalid"], pad=None)
    elif stage == "validation":
        state["analysis"]["pad_experiment"]["ready"] = False
        for frame in state["analysis"]["frames"]:
            frame["pad"] = None
    elif stage == "same_person":
        state["analysis"]["frames"][0]["emb"] = np.array([0, 1], dtype="<f4").tobytes()
    elif stage == "expired":
        state["consume"] = False
    result = asyncio.run(api.authenticate(request))
    assert "pad_diagnostics" not in result and "reason_counts" not in json.dumps(result)
    assert result["decision"] == ("approved" if stage == "approved" else "rejected")
    assert state["approvals"] == int(stage == "approved")
    counts = state["audits"][0]["pad_diagnostics"]
    assert counts["total_frames"] == 3
    assert counts["passed_frames"] == {"quality": 2, "validation": 0}.get(stage, 3)
    assert counts["unmeasured_frames"] == {"quality": 1, "validation": 3}.get(stage, 0)


def test_packet_rejection_does_not_invent_measurements(auth_api):
    state, request = auth_api
    state["integrity"] = ["bad_signature"]
    result = asyncio.run(api.authenticate(request))
    assert result["decision"] == "rejected"
    assert "pad_diagnostics" not in state["audits"][0]


@pytest.mark.parametrize("failed", [None, "private-face-name", {}, ["private-face-name"],
    [None, [], {}, float("nan")], ["nose_protrusion_mm", "nose_protrusion_mm", "private-face-name"]])
def test_diagnostics_allowlist_deduplicates_and_never_serializes_arbitrary_data(auth_api, failed):
    state, _ = auth_api
    value = {"frames": [{"pad": {"live": False, "nonresponse": True, "failed": failed,
                                  "features": {"private-face-name": float("nan")},
                                  "rgb": b"private-face-name", "score": float("inf")},
                         "emb": b"private-face-name"}]}
    summary = api._auth_pad_diagnostics(value)["pad_diagnostics"]
    serialized = json.dumps(summary, allow_nan=False)
    assert "private-face-name" not in serialized and "features" not in serialized and "score" not in serialized
    assert summary["failed_frames"] == summary["nonresponse_frames"] == 1
    assert summary["reason_counts"] == ({"nose_protrusion_mm": 1}
        if isinstance(failed, list) and "nose_protrusion_mm" in failed else {"unspecified_failure": 1})


def test_diagnostics_counts_and_reason_lists_are_bounded(auth_api):
    summary = api._auth_pad_diagnostics({"frames": [{"pad": {
        "live": False, "nonresponse": True, "failed": ["depth_coverage"] * 10000}}] * 10000})["pad_diagnostics"]
    assert summary["total_frames"] == summary["failed_frames"] == summary["nonresponse_frames"] == 16
    assert summary["reason_counts"] == {"depth_coverage": 16} and summary["truncated"]


@pytest.mark.parametrize("mode", ["disabled_pilot", "enabled", "femto_geometry_pilot"])
def test_other_pad_modes_do_not_invent_experimental_diagnostics(auth_api, mode):
    api.C.cfg.pad_mode = mode
    assert api._auth_pad_diagnostics({"frames": [{"pad": {"live": False}}]}) == {}


@pytest.mark.parametrize("nonresponse,failed", [
    (False, ["plane_residual_mm", "nose_protrusion_mm"]),
    (True, ["depth_coverage", "nose_coverage"]),
    (True, ["measurement_invalid"]), (False, []),
])
def test_engine_preserves_only_allowlisted_condition_names(monkeypatch, nonresponse, failed):
    monkeypatch.setattr(engine, "PAD_MODE", MODE)
    m = models(monkeypatch, ready=True)
    m.pad_experiment = SimpleNamespace(decide=Mock(return_value={
        "score": .123, "live": not failed, "nonresponse": nonresponse, "failed": failed,
        "features": {"secret": 4}, "rgb": b"secret"}))
    try:
        frame = m.process([({"mode": "auth"}, [None])])[0][0]
        assert frame["pad"] == {"score": .123, "live": not failed, "nonresponse": nonresponse,
                                 "failed": sorted(failed)}
    finally:
        m.pool.shutdown()


def test_real_classifier_threshold_and_coverage_diagnostics_reach_api(monkeypatch, auth_api):
    from faceauth.common.femto_pad_experiment import FEATURE_NAMES, FEATURE_SCHEMA

    monkeypatch.setattr(engine, "PAD_MODE", MODE)
    m = models(monkeypatch, ready=True)
    classifier = FemtoPADExperiment.__new__(FemtoPADExperiment)
    classifier.profile = build_experiment(binding=m.candidate_binding, calib_version="bolt")
    classifier.reasons = []
    m.pad_experiment = classifier
    measurements = [
        {"schema": FEATURE_SCHEMA, "valid": True, "reasons": [],
         "features": dict.fromkeys(FEATURE_NAMES, 1.0)},
        {"schema": FEATURE_SCHEMA, "valid": False, "reasons": ["nose_depth_discontinuity"],
         "features": dict.fromkeys(FEATURE_NAMES)},
    ]
    monkeypatch.setattr(m, "_candidate", lambda _: measurements.pop(0))
    try:
        outputs = [m.process([({"mode": "auth"}, [None])])[0][0] for _ in range(2)]
        counts = api._auth_pad_diagnostics({"frames": outputs})["pad_diagnostics"]
        assert counts["failed_frames"] == 2 and counts["nonresponse_frames"] == 1
        assert counts["reason_counts"] == {"nose_protrusion_mm": 1, "nose_depth_discontinuity": 1}
    finally:
        m.pool.shutdown()
