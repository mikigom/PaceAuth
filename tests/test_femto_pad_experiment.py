"""Engineering configuration mechanics, never real-sensor PAD validation."""

import copy
import hashlib
import json

import pytest

from faceauth.common.femto_pad_experiment import (
    BINDING_SCHEMA, ENGINEERING_THRESHOLDS, EXPERIMENT_MODE, FEATURE_NAMES, FEATURE_SCHEMA,
    INITIAL_THRESHOLDS, SOURCE_PROFILE_SHA256, FemtoPADExperiment, build_experiment,
)
from faceauth.common.pad_validation import canonical_json


BINDING = {
    "schema": BINDING_SCHEMA, "feature_schema": FEATURE_SCHEMA,
    "feature_code_sha256": "a" * 64, "detector_model_json_sha256": "b" * 64,
    "pipeline_sha256": "c" * 64,
}


def measurement(*, plane=10.0, nose=20.0):
    features = dict.fromkeys(FEATURE_NAMES, 1.0)
    features.update(plane_residual_mm=plane, nose_protrusion_mm=nose, depth_median_mm=600.0)
    return {"schema": FEATURE_SCHEMA, "valid": True, "features": features, "reasons": []}


def manifest():
    return build_experiment(binding=BINDING, calib_version="test-sensor")


def load(tmp_path, value=None):
    # json.dumps deliberately allows non-finite values for malformed-file tests.
    raw = json.dumps(manifest() if value is None else value, sort_keys=True).encode()
    path = tmp_path / "experiment.json"
    path.write_bytes(raw)
    return FemtoPADExperiment(path, hashlib.sha256(raw).hexdigest())


def test_builder_records_explicit_relaxation_without_claiming_recalibration():
    value = manifest()
    assert value["thresholds"] == {
        "plane_residual_mm": 0.30, "nose_protrusion_mm": 3.0,
    }
    assert value["threshold_provenance"]["source_profile_sha256"] == SOURCE_PROFILE_SHA256
    assert value["threshold_provenance"]["kind"] == "operator_requested_engineering_relaxation"
    assert value["threshold_provenance"]["previous_thresholds"] == INITIAL_THRESHOLDS
    assert value["threshold_provenance"]["transferred_without_recalibration"] is False
    assert value["threshold_provenance"]["recalibrated"] is False
    assert value["mode"] == EXPERIMENT_MODE
    assert value["empirical_validation"] == "not_evaluated"
    assert value["production_validated"] is False
    assert "calibration" not in value and "evaluation" not in value
    assert "eligible_for_local_pilot" not in value
    assert json.loads(canonical_json(value)) == value
    value["thresholds"]["nose_protrusion_mm"] = 0
    value["threshold_provenance"]["previous_thresholds"]["nose_protrusion_mm"] = 0
    value["candidate_binding"]["pipeline_sha256"] = "d" * 64
    assert manifest()["thresholds"] == ENGINEERING_THRESHOLDS
    assert manifest()["threshold_provenance"]["previous_thresholds"]["nose_protrusion_mm"] == 4.563589938099597
    assert BINDING["pipeline_sha256"] == "c" * 64


def test_ready_only_reports_pinned_unvalidated_configuration(tmp_path):
    experiment = load(tmp_path)
    status = experiment.status("test-sensor", BINDING)
    assert status["ready"] is True and status["reasons"] == []
    assert status["scope"] == ["print", "replay"]
    assert status["experimental"] is True and status["production_validated"] is False
    assert status["empirical_validation"] == "not_evaluated"
    assert len(status["profile_sha256"]) == 64
    assert not any(name in status for name in ("eligible_for_local_pilot", "evaluation", "calibration"))
    assert experiment.decide(measurement()) == {
        "score": 1.0, "live": True, "nonresponse": False, "failed": [],
    }


@pytest.mark.parametrize("pin", [None, "", "0" * 63, "A" * 64, "0" * 64])
def test_missing_invalid_or_incorrect_pins_never_enable(tmp_path, pin):
    path = tmp_path / "experiment.json"
    path.write_bytes(canonical_json(manifest()))
    experiment = FemtoPADExperiment(path, pin)
    assert not experiment.status("test-sensor", BINDING)["ready"]
    assert experiment.decide(measurement())["nonresponse"] is True
    assert not experiment.decide(measurement())["live"]


def test_missing_file_is_closed(tmp_path):
    experiment = FemtoPADExperiment(tmp_path / "absent", "a" * 64)
    assert experiment.status("test-sensor", BINDING)["reasons"] == ["profile_missing_or_invalid"]


@pytest.mark.parametrize("change", ["sensor", "extractor", "detector", "pipeline", "feature_version", "binding_version", "missing"])
def test_each_request_requires_same_sensor_and_complete_v2_binding(tmp_path, change):
    experiment = load(tmp_path)
    binding, calib = dict(BINDING), "test-sensor"
    if change == "sensor":
        calib = "other-sensor"
    elif change in ("extractor", "detector", "pipeline"):
        key = {"extractor": "feature_code_sha256", "detector": "detector_model_json_sha256", "pipeline": "pipeline_sha256"}[change]
        binding[key] = "d" * 64
    elif change == "feature_version":
        binding["feature_schema"] = "faceauth.femto_pad.features.v1"
    elif change == "binding_version":
        binding["schema"] = "faceauth.femto_pad.binding.v0"
    else:
        binding.pop("pipeline_sha256")
    assert not experiment.status(calib, binding)["ready"]
    # Request-specific probes must not corrupt other concurrent valid requests.
    assert experiment.status("test-sensor", BINDING)["ready"]
    assert experiment.decide(measurement())["live"]


@pytest.mark.parametrize("change", [
    "old_schema", "old_features", "old_binding", "wrong_mode", "production_claim", "validation_claim",
    "evaluation", "calibration", "extra_field", "source_digest", "source_claim", "wrong_algorithm", "mask_scope",
])
def test_artifacts_cannot_borrow_old_validation_or_expand_their_scope(tmp_path, change):
    value = manifest()
    if change == "old_schema":
        value["schema"] = "faceauth.femto_pad.profile.v1"
    elif change == "old_features":
        value["feature_schema"] = "faceauth.femto_pad.features.v1"
    elif change == "old_binding":
        value["candidate_binding"]["feature_schema"] = "faceauth.femto_pad.features.v1"
    elif change == "wrong_mode":
        value["mode"] = "femto_geometry_pilot"
    elif change == "production_claim":
        value["production_validated"] = True
    elif change == "validation_claim":
        value["empirical_validation"] = "passed"
    elif change == "source_digest":
        value["threshold_provenance"]["source_profile_sha256"] = "0" * 64
    elif change == "source_claim":
        value["threshold_provenance"]["recalibrated"] = True
    elif change == "wrong_algorithm":
        value["algorithm"] = "any_frame_passes"
    elif change == "mask_scope":
        value["scope"].append("mask_rigid")
    else:
        value[change] = {"eligible_for_local_pilot": True}
    experiment = load(tmp_path, value)
    assert not experiment.status("test-sensor", BINDING)["ready"]
    assert experiment.decide(measurement())["nonresponse"]


@pytest.mark.parametrize("threshold", [0, -1, 101, float("nan"), float("inf"), True, "4.56", None, 4.0, 5.0])
def test_thresholds_are_positive_bounded_finite_and_match_the_revision(tmp_path, threshold):
    value = manifest()
    value["thresholds"]["nose_protrusion_mm"] = threshold
    experiment = load(tmp_path, value)
    assert not experiment.status("test-sensor", BINDING)["ready"]
    assert not experiment.decide(measurement())["live"]


def test_frame_policy_requires_both_minima_without_averaging(tmp_path):
    experiment = load(tmp_path)
    thresholds = manifest()["thresholds"]
    boundary = measurement(plane=thresholds["plane_residual_mm"], nose=thresholds["nose_protrusion_mm"])
    assert experiment.decide(boundary)["live"]
    weak = measurement(nose=thresholds["nose_protrusion_mm"] - .001)
    decisions = [experiment.decide(value) for value in (measurement(), weak, measurement())]
    assert [result["live"] for result in decisions] == [True, False, True]
    assert decisions[1]["failed"] == ["nose_protrusion_mm"]
    flat = experiment.decide(measurement(plane=0, nose=0))
    assert not flat["live"] and not flat["nonresponse"]
    assert set(flat["failed"]) == set(ENGINEERING_THRESHOLDS)


def test_relaxed_auth_minima_accept_borderline_geometry_but_still_require_both(tmp_path):
    experiment = load(tmp_path)
    borderline = measurement(plane=.32, nose=3.5)
    assert all(borderline["features"][key] < value for key, value in INITIAL_THRESHOLDS.items())
    assert experiment.decide(borderline)["live"]
    assert experiment.decide(measurement(plane=.299, nose=20))["failed"] == ["plane_residual_mm"]
    assert experiment.decide(measurement(plane=10, nose=2.999))["failed"] == ["nose_protrusion_mm"]


def test_relaxed_minima_preserve_generated_surface_and_missing_depth_rejections(tmp_path):
    from scripts.diagnose_femto_nose import run_diagnostics

    # Generated geometry tests exercise the real extractor, not physical PAD performance.
    experiment = load(tmp_path)
    scenarios = run_diagnostics()["scenarios"]
    reject = {"planar_surface", "curved_surface", "creased_surface", "measurement_artifact"}
    accept = {"localized_relief", "landmark_error", "cross_modal_displacement"}
    assert reject <= {row["family"] for row in scenarios}
    for row in scenarios:
        decision = experiment.decide(row)
        if row["family"] in reject:
            assert not decision["live"], row["scenario"]
        elif row["family"] in accept:
            assert decision["live"], row["scenario"]


@pytest.mark.parametrize("reason", ["depth_missing", "nose_coverage", "nose_depth_discontinuity", "ir_missing"])
def test_invalid_depth_and_geometry_remain_nonresponses(tmp_path, reason):
    value = measurement()
    value.update(valid=False, reasons=[reason])
    value["features"] = dict.fromkeys(FEATURE_NAMES)
    result = load(tmp_path).decide(value)
    assert result == {"score": 0.0, "live": False, "nonresponse": True, "failed": [reason]}


@pytest.mark.parametrize("change", [
    "old_version", "no_depth", "zero_depth", "nan", "infinity", "bool", "string", "coverage",
    "wrong_valid", "missing_reason", "unknown_reason", "conflicting_reason", "missing_features",
])
def test_malformed_or_old_measurements_never_pass(tmp_path, change):
    value = measurement()
    if change == "old_version":
        value["schema"] = "faceauth.femto_pad.features.v1"
    elif change == "no_depth":
        value["features"].pop("depth_median_mm")
    elif change == "zero_depth":
        value["features"]["depth_median_mm"] = 0
    elif change in ("nan", "infinity", "bool", "string"):
        value["features"]["nose_protrusion_mm"] = {
            "nan": float("nan"), "infinity": float("inf"), "bool": True, "string": "20",
        }[change]
    elif change == "coverage":
        value["features"]["nose_valid_ratio"] = 1.01
    elif change == "wrong_valid":
        value["valid"] = 1
    elif change == "missing_reason":
        value["valid"] = False
    elif change == "unknown_reason":
        value.update(valid=False, reasons=["made_up"])
    elif change == "conflicting_reason":
        value["reasons"] = ["depth_missing"]
    else:
        value.pop("features")
    result = load(tmp_path).decide(value)
    assert result == {"score": 0.0, "live": False, "nonresponse": True, "failed": ["measurement_invalid"]}


def test_builder_rejects_legacy_binding_and_does_not_mutate_inputs():
    old = copy.deepcopy(BINDING)
    old["feature_schema"] = "faceauth.femto_pad.features.v1"
    with pytest.raises(ValueError):
        build_experiment(binding=old, calib_version="test-sensor")
    with pytest.raises(ValueError):
        build_experiment(binding=BINDING, calib_version=" ")
    assert old["feature_schema"] == "faceauth.femto_pad.features.v1"
