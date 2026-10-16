"""Physical PAD evidence cannot be replaced by labels, frames or signatures alone."""
import hashlib
import json

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from faceauth.common.pad_validation import (PADValidationGate, SPECIES, binomial_upper95, canonical_json,
    evaluate_observations, make_observation, model_binding, CANDIDATE_SCHEMA, CANDIDATE_FEATURES,
    candidate_binding, clean_candidate_binding, clean_candidate_measurement)


CALIB = "femto-bolt-test-calibration"


@pytest.fixture
def model(tmp_path):
    directory = tmp_path / "model"
    directory.mkdir()
    weights = b"isolated test weights, not an executable model"
    (directory / "weights.bin").write_bytes(weights)
    meta = {"name": "test-pad", "version": "1", "kind": "pad", "files": {"weights": "weights.bin"},
            "sha256": {"weights": hashlib.sha256(weights).hexdigest()},
            "params": {"calib_version": CALIB, "fx": 497.4, "threshold": 0.99}}
    (directory / "MODEL.json").write_text(json.dumps(meta))
    return directory


def observation(binding, attempt="one", kind="live", *, split="test", person=None, instrument=None,
                frames=None, consent=True, candidate=None):
    return make_observation(presentation_id=attempt, subject_id=person or "person-" + attempt,
                            consent_id="study-consent-" + attempt, consent=consent, kind=kind, split=split,
                            instrument_id=instrument or ("instrument-" + attempt if kind != "live" else None),
                            calib_version=CALIB, binding=binding,
                            frames=frames or [{"pad": {"live": kind == "live", "nonresponse": False,
                                                     "score": 1.0 if kind == "live" else 0.0}, "ok": True}],
                            conditions={"mode": "auth", "distance_mm": 650}, candidate_binding=candidate)


def candidate_fixture():
    return {"schema": "faceauth.femto_pad.binding.v1", "feature_schema": CANDIDATE_SCHEMA,
            "feature_code_sha256": "a" * 64, "detector_model_json_sha256": "b" * 64,
            "pipeline_sha256": "c" * 64}


def candidate_measurement():
    return {"schema": CANDIDATE_SCHEMA, "valid": True, "reasons": [],
            "features": dict.fromkeys(CANDIDATE_FEATURES, 1.0)}


def approved_report(binding):
    # These are test-fixture thresholds, never defaults in product code.
    rows = []
    for mode in ("enroll", "auth"):
        for kind in ("live", *SPECIES):
            for i in range(30):
                row = observation(binding, f"{mode}-{kind}-{i}", kind)
                row["conditions"]["mode"] = mode
                rows.append(row)
    policy = {"reference": "fixture-only-policy", "approved_by": "fixture-reviewer",
              "required_species": list(SPECIES), "required_modes": ["enroll", "auth"], "min_live_presentations": 30,
              "min_attack_presentations": 30, "min_live_subjects": 10, "min_attack_instruments": 3,
              "max_apcer_upper95": .1, "max_bpcer_upper95": .1,
              "max_apnrr_upper95": .1, "max_bpnrr_upper95": .1}
    independent = {"assessor": "fixture-independent-assessor", "report_id": "fixture-report",
                   "report_sha256": "0" * 64, "independent": True}
    return evaluate_observations(rows, binding=binding, calib_version=CALIB,
                                 policy=policy, independent_test=independent)


def signed_report(model, report):
    key = Ed25519PrivateKey.generate()
    pub = model / "release.pub"
    pub.write_bytes(key.public_key().public_bytes(serialization.Encoding.PEM,
                                                serialization.PublicFormat.SubjectPublicKeyInfo))
    raw = canonical_json(report)
    (model / "PAD_VALIDATION.json").write_bytes(raw)
    (model / "PAD_VALIDATION.json.sig").write_bytes(key.sign(raw))
    return pub


def test_same_calibration_without_evidence_still_blocks(model):
    status = PADValidationGate(model).status(CALIB)
    assert not status["ready"]
    assert status["reasons"] == ["pad_real_sensor_validation_missing"]


def test_binding_uses_actual_detector_not_logical_name_directory(model):
    detector = model.parent / "scrfd_34g_320"
    detector.mkdir()
    raw = b'{"name":"scrfd_34g","version":"2","params":{"size":320}}'
    (detector / "MODEL.json").write_bytes(raw)
    (model.parent / "registry.json").write_text(json.dumps({"models": [
        {"name": "scrfd_34g", "version": "2", "dir": "scrfd_34g_320"}]}))
    expected = hashlib.sha256(raw).hexdigest()
    assert model_binding(model)["detector_model_json_sha256"] == expected
    assert model_binding(model, detector_dir=detector)["detector_model_json_sha256"] == expected


def test_signed_independent_sufficient_evidence_can_qualify(model):
    report = approved_report(model_binding(model))
    assert report["eligible_for_review"] is True
    public = signed_report(model, report)
    gate = PADValidationGate(model, public)
    assert gate.status(CALIB)["ready"]
    assert "pad_calibration_mismatch" in gate.status("other-sensor")["reasons"]


@pytest.mark.parametrize("tamper", ["report", "settings", "weights"])
def test_signed_evidence_cannot_survive_tampering(model, tamper):
    report = approved_report(model_binding(model))
    public = signed_report(model, report)
    if tamper == "report":
        (model / "PAD_VALIDATION.json").write_bytes(canonical_json({**report, "unit": "frame"}))
    elif tamper == "settings":
        meta = json.loads((model / "MODEL.json").read_text())
        meta["params"]["threshold"] = .1
        (model / "MODEL.json").write_text(json.dumps(meta))
    else:
        (model / "weights.bin").write_bytes(b"new weights")
    assert not PADValidationGate(model, public).status(CALIB)["ready"]


@pytest.mark.parametrize("removed", ["policy", "independent_test"])
def test_signature_does_not_substitute_missing_validation(model, removed):
    report = approved_report(model_binding(model))
    report[removed] = None
    # Even stale/fraudulent success flags are ignored; metrics/policy are checked.
    report["eligible_for_review"], report["blockers"] = True, []
    public = signed_report(model, report)
    assert not PADValidationGate(model, public).status(CALIB)["ready"]


def test_reported_rates_are_recomputed_from_counts(model):
    report = approved_report(model_binding(model))
    report["summary"]["print"]["apcer"].update(errors=30, value=0, upper95=0)
    public = signed_report(model, report)
    assert "pad_print_max_apcer_upper95_not_met" in PADValidationGate(model, public).status(CALIB)["reasons"]


def test_collection_requires_explicit_study_consent(model):
    for consent in (False, None, "true", 1):
        with pytest.raises(ValueError, match="consent"):
            observation(model_binding(model), consent=consent)


def test_collection_strips_biometrics_and_retains_nonresponse(model):
    frames = [{"rgb": b"private", "depth": b"private", "ir": b"private", "emb": b"private",
               "kps": [1, 2], "pad": None, "quality": {"blur": 2.0}, "reasons": ["no_face"]}]
    row = observation(model_binding(model), frames=frames)
    assert "private" not in json.dumps(row)
    assert set(row["frames"][0]) == {"pad", "ok", "quality", "reasons"}
    report = evaluate_observations([row], binding=model_binding(model), calib_version=CALIB)
    assert report["summary"]["live"]["nonresponse"]["errors"] == 1
    assert report["summary"]["live"]["bpcer"]["trials"] == 0
    assert not report["eligible_for_review"]


def test_diagnostic_distributions_retain_real_physics_failures_without_nan(model):
    binding = model_binding(model)
    frame = {"pad": {"live": False, "nonresponse": False, "score": 0, "cnn": .8,
                     "checks": {"ir_median": 17, "relief_mm": float("nan")},
                     "failed": ["ir_median", "relief_mm"]}}
    row = observation(binding, frames=[frame])
    row["conditions"]["mode"] = "diagnostic"
    report = evaluate_observations([row], binding=binding, calib_version=CALIB)
    stats = report["diagnostic_summary"]["live"]["diagnostics"]
    assert stats["quantiles"]["ir_median"]["median"] == 17
    assert "relief_mm" not in stats["quantiles"]
    assert stats["failed_check_frames"] == {"ir_median": 1, "relief_mm": 1}
    canonical_json(report)


def test_unmeasured_selection_timeout_counts_once_without_invented_score_or_frame(model):
    binding = model_binding(model)
    row = observation(binding, frames=[{"pad": None, "ok": False, "reasons": ["capture_incomplete"]}])
    row["conditions"]["capture_outcome"] = "selection_timeout"
    report = evaluate_observations([row], binding=binding, calib_version=CALIB)
    result = report["summary"]["live"]
    assert result["presentations"] == result["nonresponse"]["errors"] == result["capture_nonresponses"] == 1
    assert result["bpcer"]["trials"] == 0
    assert result["diagnostics"] == {"frame_count": 0, "quantiles": {}, "failed_check_frames": {}}


def test_many_adjacent_frames_count_as_one_presentation(model):
    binding = model_binding(model)
    frames = [{"pad": {"live": True, "nonresponse": False, "score": .999}} for _ in range(300)]
    frames[-1]["pad"]["live"] = False
    row = observation(binding, frames=frames)
    report = evaluate_observations([row], binding=binding, calib_version=CALIB)
    result = report["summary"]["live"]
    assert result["presentations"] == 1
    assert result["bpcer"]["errors"] == result["bpcer"]["trials"] == 1
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_observations([row, row], binding=binding, calib_version=CALIB)


def test_diagnostic_frames_cannot_be_counted_as_auth_or_enrollment_evidence(model):
    binding = model_binding(model)
    row = observation(binding)
    row["conditions"]["mode"] = "diagnostic"
    report = evaluate_observations([row], binding=binding, calib_version=CALIB)
    assert report["diagnostic_test_presentations"] == 1
    assert report["diagnostic_summary"]["live"]["presentations"] == 1
    assert report["deployment_test_presentations"] == 0
    assert report["summary"]["live"]["presentations"] == 0
    assert all(report["by_mode"][mode]["live"]["presentations"] == 0 for mode in ("auth", "enroll"))
    row["split"] = "calibration"
    report = evaluate_observations([row], binding=binding, calib_version=CALIB)
    assert report["diagnostic_presentations"] == 1
    assert report["diagnostic_test_presentations"] == 0
    assert report["diagnostic_by_split"]["calibration"]["live"]["presentations"] == 1


@pytest.mark.parametrize("shared", ["person", "instrument"])
def test_subject_and_instrument_split_leakage_rejected(model, shared):
    binding = model_binding(model)
    fields = {shared: "shared-id"}
    rows = [observation(binding, "a", "print", split="calibration", **fields),
            observation(binding, "b", "print", split="test", **fields)]
    with pytest.raises(ValueError, match="leakage"):
        evaluate_observations(rows, binding=binding, calib_version=CALIB)


def test_synthetic_and_mixed_model_records_rejected(model):
    binding = model_binding(model)
    for key, value in (("source", "synthetic"), ("binding", {}), ("calib_version", "other")):
        row = observation(binding)
        row[key] = value
        with pytest.raises(ValueError):
            evaluate_observations([row], binding=binding, calib_version=CALIB)


def test_no_decisions_cannot_appear_as_zero_attack_error(model):
    report = approved_report(model_binding(model))
    metrics = report["summary"]["print"]
    metrics["nonresponse"].update(errors=60, trials=60)
    metrics["apcer"].update(errors=0, trials=0)
    public = signed_report(model, report)
    status = PADValidationGate(model, public).status(CALIB)
    assert not status["ready"]
    assert "pad_print_max_apcer_upper95_not_met" in status["reasons"]


def test_exact_upper_bound_known_values():
    assert binomial_upper95(0, 0) is None
    assert binomial_upper95(0, 1) == pytest.approx(.95)
    assert binomial_upper95(0, 100) == pytest.approx(.029513049607)
    assert binomial_upper95(5, 10) == pytest.approx(.777558898991)
    assert binomial_upper95(10, 10) == 1
    with pytest.raises(ValueError):
        binomial_upper95(True, 10)


def test_candidate_schema_matches_measurement_implementation():
    from faceauth.models.femto_pad import FEATURE_NAMES, FEATURE_SCHEMA
    assert CANDIDATE_SCHEMA == FEATURE_SCHEMA
    assert CANDIDATE_FEATURES == FEATURE_NAMES


def test_candidate_binding_tracks_actual_geometry_and_detector_sources(tmp_path, monkeypatch):
    from faceauth.common import pad_validation as module
    detector = tmp_path / "detector"
    detector.mkdir()
    (detector / "MODEL.json").write_text('{"fixture":"detector"}')
    first = candidate_binding(detector_dir=detector)
    assert clean_candidate_binding(first) == first
    assert first["feature_code_sha256"] == module._hash(
        module.Path(module.__file__).resolve().parents[1] / "models/femto_pad.py")
    actual_hash = module._hash
    monkeypatch.setattr(module, "_hash", lambda path: "d" * 64 if str(path).endswith("models/femto_pad.py") else actual_hash(path))
    second = candidate_binding(detector_dir=detector)
    assert second["feature_code_sha256"] != first["feature_code_sha256"]
    assert second["pipeline_sha256"] != first["pipeline_sha256"]
    (detector / "MODEL.json").write_text('{"fixture":"different detector"}')
    assert candidate_binding(detector_dir=detector)["detector_model_json_sha256"] != second["detector_model_json_sha256"]


def test_candidate_collection_whitelists_only_finite_features_and_does_not_relabel_old_pad(model):
    measurement = {**candidate_measurement(), "emb": b"private", "score": 1, "live": True}
    measurement["features"]["raw_depth"] = ["private"]
    candidate = {**candidate_fixture(), "name": "private"}
    frame = {"pad": {"live": False, "nonresponse": False, "score": 0}, "pad_candidate": measurement}
    row = observation(model_binding(model), frames=[frame], candidate=candidate)
    assert "private" not in json.dumps(row)
    assert row["candidate_binding"] == candidate_fixture()
    assert row["frames"][0]["pad_candidate"] == candidate_measurement()
    assert row["frames"][0]["pad"]["live"] is False
    report = evaluate_observations([row], binding=model_binding(model), calib_version=CALIB)
    assert report["summary"]["live"]["bpcer"]["errors"] == 1
    assert not report["eligible_for_review"]


def test_candidate_collection_requires_binding_and_keeps_missing_attempts(model):
    with pytest.raises(ValueError, match="requires its binding"):
        observation(model_binding(model), frames=[{"pad_candidate": candidate_measurement()}])
    row = observation(model_binding(model), frames=[{"pad": None}], candidate=candidate_fixture())
    candidate = row["frames"][0]["pad_candidate"]
    assert candidate["valid"] is False and candidate["reasons"] == ["not_measured"]
    assert all(value is None for value in candidate["features"].values())


@pytest.mark.parametrize("bad", [True, "1.2", [1.2], {"nested": 1.2}, float("nan"), float("inf"), None])
def test_candidate_valid_measurement_cannot_contain_nonnumeric_or_missing_features(bad):
    measurement = candidate_measurement()
    measurement["features"]["plane_residual_mm"] = bad
    with pytest.raises(ValueError):
        clean_candidate_measurement(measurement)


@pytest.mark.parametrize("field,value", [("schema", "other"), ("feature_schema", "other"),
    ("pipeline_sha256", "not-a-hash"), ("feature_code_sha256", ["a" * 64])])
def test_candidate_binding_rejects_malformed_provenance(field, value):
    with pytest.raises(ValueError):
        clean_candidate_binding({**candidate_fixture(), field: value})


def test_candidate_failure_reasons_are_codes_not_freeform_metadata():
    for reasons in (["private account name"], [[1, 2]], ["depth_missing"]):
        measurement = candidate_measurement()
        measurement["reasons"] = reasons
        with pytest.raises(ValueError):
            clean_candidate_measurement(measurement)
    invalid = {"schema": CANDIDATE_SCHEMA, "valid": False, "reasons": ["depth_missing"],
               "features": {"valid_ratio": 0.0, "plane_residual_mm": float("nan")}}
    assert clean_candidate_measurement(invalid)["features"]["plane_residual_mm"] is None


def test_historical_candidate_version_is_preserved_and_cannot_be_relabelled(model):
    from faceauth.common.pad_validation import LEGACY_CANDIDATE_SCHEMA
    old_binding = {**candidate_fixture(), "feature_schema": LEGACY_CANDIDATE_SCHEMA}
    old_measurement = {**candidate_measurement(), "schema": LEGACY_CANDIDATE_SCHEMA}
    assert clean_candidate_binding(old_binding)["feature_schema"] == LEGACY_CANDIDATE_SCHEMA
    assert clean_candidate_measurement(old_measurement)["schema"] == LEGACY_CANDIDATE_SCHEMA
    with pytest.raises(ValueError, match="schema does not match"):
        observation(model_binding(model), frames=[{"pad_candidate": old_measurement}], candidate=candidate_fixture())
    row = observation(model_binding(model), frames=[{"pad": None}], candidate=old_binding)
    assert row["frames"][0]["pad_candidate"]["schema"] == LEGACY_CANDIDATE_SCHEMA
    assert not row["frames"][0]["pad_candidate"]["valid"]
