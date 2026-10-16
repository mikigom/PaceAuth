"""Synthetic feature fixtures verify policy mechanics, not real-sensor accuracy."""

import hashlib
import json
import stat

import pytest

from faceauth.common.femto_pad_profile import FemtoPADProfile, evaluate_profile, fit_profile
from faceauth.common.pad_validation import CANDIDATE_FEATURES, CANDIDATE_SCHEMA, OBSERVATION_SCHEMA, canonical_json
from scripts.calibrate_femto_pad import main

BINDING = {"schema": "faceauth.femto_pad.binding.v1", "feature_schema": CANDIDATE_SCHEMA,
           "feature_code_sha256": "a" * 64, "detector_model_json_sha256": "b" * 64, "pipeline_sha256": "c" * 64}


def measurement(plane=10, nose=20, valid=True):
    features = dict.fromkeys(CANDIDATE_FEATURES, 1.0)
    features.update(plane_residual_mm=plane, nose_protrusion_mm=nose)
    return {"schema": CANDIDATE_SCHEMA, "valid": valid, "features": features,
            "reasons": [] if valid else ["depth_missing"]}


def rows(split="calibration"):
    data = []
    for kind, features in (("live", (10, 20)), ("print", (1, 15)), ("replay", (8, 1))):
        for i in range(3):
            mode = "diagnostic"
            if split == "test":
                mode = "auth" if i == 0 else "enroll" if i == 1 and kind == "live" else "diagnostic"
            data.append({"schema": OBSERVATION_SCHEMA, "source": "real_sensor", "consent": True,
                         "consent_id": "synthetic-test-fixture", "presentation_id": f"{split}-{kind}-{i}",
                         "subject_id": f"subject-{i}", "instrument_id": f"{kind}-{i}" if kind != "live" else None,
                         "kind": kind, "split": split, "calib_version": "test-sensor",
                         "candidate_binding": BINDING, "conditions": {"mode": mode},
                         "frames": [{"pad_candidate": measurement(*features)} for _ in range(3)]})
    return data


def fit(data=None):
    return fit_profile(rows() if data is None else data, binding=BINDING, calib_version="test-sensor")


def test_v1_history_stays_v1_and_cannot_supply_v2_runtime_measurements(tmp_path):
    from copy import deepcopy
    from faceauth.common.pad_validation import LEGACY_CANDIDATE_SCHEMA
    legacy_binding = {**BINDING, "feature_schema": LEGACY_CANDIDATE_SCHEMA}
    def legacy(split):
        result = deepcopy(rows(split))
        for row in result:
            row["candidate_binding"] = legacy_binding
            for frame in row["frames"]:
                frame["pad_candidate"]["schema"] = LEGACY_CANDIDATE_SCHEMA
        return result
    profile = fit_profile(legacy("calibration"), binding=legacy_binding, calib_version="test-sensor")
    assert profile["feature_schema"] == LEGACY_CANDIDATE_SCHEMA
    evaluated = evaluate_profile(profile, legacy("test"))
    loader = load(tmp_path, evaluated)
    assert not loader.status("test-sensor", BINDING)["ready"]
    assert not loader.decide(measurement())["live"]
    with pytest.raises(ValueError, match="schema does not match"):
        mismatched = legacy("calibration")
        mismatched[0]["frames"][0]["pad_candidate"] = measurement()
        fit_profile(mismatched, binding=legacy_binding, calib_version="test-sensor")


def load(tmp_path, profile):
    path = tmp_path / "profile.json"
    raw = canonical_json(profile)
    path.write_bytes(raw)
    return FemtoPADProfile(path, hashlib.sha256(raw).hexdigest())


def test_fit_uses_calibration_only_and_counts_presentations():
    calibrated = fit()
    hostile_test = rows("test")
    for row in hostile_test:
        row["frames"][0]["pad_candidate"] = measurement(1000, 1000)
    assert fit(rows() + hostile_test) == calibrated
    assert calibrated["calibration"]["summary"]["live"]["presentations"] == 3
    assert calibrated["calibration"]["mode_counts"] == {"diagnostic": 9}
    assert calibrated["production_validated"] is False
    assert all(value > 0 for value in calibrated["thresholds"].values())


def test_frozen_evaluation_and_runtime_success_are_explicitly_experimental(tmp_path):
    calibrated = fit()
    evaluated = evaluate_profile(calibrated, rows("test"))
    assert evaluated["thresholds"] == calibrated["thresholds"]
    assert "evaluation" not in calibrated
    assert evaluated["evaluation"]["eligible_for_local_pilot"]
    assert evaluated["evaluation"]["subject_overlap"]
    assert evaluated["evaluation"]["instrument_overlap"]
    assert evaluated["evaluation"]["independent_assessment"] is False
    assert evaluated["evaluation"]["summary"]["print"]["error_upper95"] > .6
    profile = load(tmp_path, evaluated)
    status = profile.status("test-sensor", BINDING)
    assert status["ready"] and status["experimental"] and status["scope"] == ["print", "replay"]
    assert status["calib_version"] == "test-sensor" and not status["production_validated"]
    assert profile.decide(measurement())["score"] == 1.0
    assert not profile.decide(measurement(1, 1))["live"]
    assert profile.decide(measurement(valid=False))["nonresponse"]
    assert profile.decide({})["nonresponse"]


@pytest.mark.parametrize("change", ["old_scores", "unconsented", "simulated", "binding", "duplicate", "few", "missing_live", "overlap"])
def test_fit_rejects_unusable_or_inseparable_data(change):
    data = rows()
    if change == "old_scores":
        data[0]["frames"] = [{"pad": {"score": 1.0}}]
    elif change == "unconsented":
        data[0]["consent"] = False
    elif change == "simulated":
        data[0]["source"] = "simulator"
    elif change == "binding":
        data[0]["candidate_binding"] = {**BINDING, "pipeline_sha256": "d" * 64}
    elif change == "duplicate":
        data[1]["presentation_id"] = data[0]["presentation_id"]
    elif change == "few":
        data.pop()
    elif change == "missing_live":
        data[0]["frames"][0]["pad_candidate"] = measurement(valid=False)
    else:
        data[3]["frames"] = [{"pad_candidate": measurement(10, 20)}]
    with pytest.raises(ValueError):
        fit(data)


@pytest.mark.parametrize("change", ["live_reject", "live_missing", "attack_accept", "attack_all_missing"])
def test_test_failures_do_not_change_thresholds_or_enable_runtime(tmp_path, change):
    calibrated, data = fit(), rows("test")
    if change == "live_reject":
        data[0]["frames"][1]["pad_candidate"] = measurement(1, 1)
    elif change == "live_missing":
        data[0]["frames"][1]["pad_candidate"] = measurement(valid=False)
    elif change == "attack_accept":
        data[3]["frames"] = [{"pad_candidate": measurement()}]
    else:
        for row in data[3:6]:
            row["frames"] = [{"pad_candidate": measurement(valid=False)}]
    evaluated = evaluate_profile(calibrated, data)
    assert evaluated["thresholds"] == calibrated["thresholds"]
    assert not evaluated["evaluation"]["eligible_for_local_pilot"]
    profile = load(tmp_path, evaluated)
    assert not profile.status("test-sensor", BINDING)["ready"]
    assert not profile.decide(measurement())["live"]


def test_distinct_test_presentations_required_even_with_other_labels():
    data = rows("test")
    data[0]["presentation_id"] = "calibration-print-0"
    with pytest.raises(ValueError, match="overlap"):
        evaluate_profile(fit(), data)


@pytest.mark.parametrize("missing", ["all_actual_flows", "live_enroll", "print_auth", "replay_auth"])
def test_pilot_requires_real_auth_and_enrollment_selection_modes(missing):
    data = rows("test")
    for row in data:
        if (missing == "all_actual_flows"
                or missing == f"{row['kind']}_{row['conditions']['mode']}"):
            row["conditions"]["mode"] = "diagnostic"
    evaluated = evaluate_profile(fit(), data)
    assert not evaluated["evaluation"]["eligible_for_local_pilot"]
    assert evaluated["evaluation"]["reasons"]


@pytest.mark.parametrize("change", ["threshold", "calib", "binding", "counts", "no_evaluation", "boolean_only"])
def test_malformed_or_stale_pinned_artifacts_remain_closed(tmp_path, change):
    evaluated = evaluate_profile(fit(), rows("test"))
    if change == "threshold":
        evaluated["thresholds"]["nose_protrusion_mm"] = .01
    elif change == "calib":
        evaluated["calib_version"] = "other"
    elif change == "binding":
        evaluated["candidate_binding"]["pipeline_sha256"] = "d" * 64
    elif change == "counts":
        evaluated["evaluation"]["summary"]["print"]["responding"] = 99
    elif change == "no_evaluation":
        evaluated.pop("evaluation")
    else:
        evaluated["evaluation"] = {"eligible_for_local_pilot": True}
    profile = load(tmp_path, evaluated)
    assert not profile.status("test-sensor", BINDING)["ready"]


def test_actual_file_hash_and_runtime_provenance_required(tmp_path):
    evaluated = evaluate_profile(fit(), rows("test"))
    profile = load(tmp_path, evaluated)
    assert not FemtoPADProfile(tmp_path / "profile.json", "0" * 64).status("test-sensor", BINDING)["ready"]
    assert not FemtoPADProfile(tmp_path / "absent", None).status("test-sensor", BINDING)["ready"]
    assert not profile.status("other-sensor", BINDING)["ready"]
    assert not profile.status("test-sensor", {**BINDING, "pipeline_sha256": "e" * 64})["ready"]
    # Failed readiness probes cannot corrupt a concurrent valid request's
    # decision; request-bound provenance checks remain the engine's gate.
    assert profile.status("test-sensor", BINDING)["ready"]
    assert profile.decide(measurement())["live"]


def test_cli_exclusive_private_artifacts_and_frozen_evaluation(tmp_path):
    observations = tmp_path / "data.jsonl"
    observations.write_text("\n".join(json.dumps(row) for row in rows() + rows("test")))
    calibrated, evaluated = tmp_path / "fit.json", tmp_path / "evaluated.json"
    command = ["fit", "--observations", str(observations), "--calib-version", "test-sensor", "--output", str(calibrated)]
    assert main(command) == 0
    assert stat.S_IMODE(calibrated.stat().st_mode) == 0o600
    assert main(command) == 2
    assert main(["evaluate", "--observations", str(observations), "--profile", str(calibrated), "--output", str(evaluated)]) == 0
    assert stat.S_IMODE(evaluated.stat().st_mode) == 0o600
    assert json.loads(evaluated.read_text())["evaluation"]["eligible_for_local_pilot"]


def test_cli_explicitly_skips_legacy_rows_without_relabeling_them(tmp_path, capsys):
    legacy = [{"split": "calibration", "kind": "live", "frames": [{"pad": {"score": 1.0}}]}] * 32
    observations, output = tmp_path / "mixed.jsonl", tmp_path / "fit.json"
    observations.write_text("\n".join(json.dumps(row) for row in legacy + rows()))
    assert main(["fit", "--observations", str(observations), "--calib-version", "test-sensor", "--output", str(output)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["legacy_skipped"] == 32 and summary["provenance_excluded"] == 0
    assert json.loads(output.read_text())["calibration"]["summary"]["live"]["presentations"] == 3


def test_cli_selects_explicit_binding_and_rejects_malformed_present_binding(tmp_path, capsys):
    data = rows()
    other = {**data[0], "candidate_binding": {**BINDING, "pipeline_sha256": "f" * 64}}
    observations, binding, output = tmp_path / "data.jsonl", tmp_path / "binding.json", tmp_path / "fit.json"
    binding.write_text(json.dumps(BINDING))
    observations.write_text("\n".join(json.dumps(row) for row in data + [other]))
    base = ["fit", "--observations", str(observations), "--calib-version", "test-sensor", "--output", str(output)]
    assert main(base) == 2
    capsys.readouterr()
    assert main(base + ["--candidate-binding", str(binding)]) == 0
    assert json.loads(capsys.readouterr().out)["provenance_excluded"] == 1
    observations.write_text(json.dumps({"candidate_binding": None}) + "\n")
    assert main(base) == 2
