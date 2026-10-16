"""Synthetic cohort/audit tests; these do not establish sensor accuracy."""

import copy
from datetime import datetime, timezone
import hashlib
import json
import stat

import pytest

from faceauth.common.femto_pad_profile import FemtoPADProfile, fit_profile, profile_payload_sha256
from faceauth.common.pad_validation import CANDIDATE_FEATURES, CANDIDATE_SCHEMA, OBSERVATION_SCHEMA, canonical_json
from scripts.evaluate_femto_runtime import RECONCILIATION_SCHEMA, create_protocol, evaluate_protocol, main

BINDING = {"schema": "faceauth.femto_pad.binding.v1", "feature_schema": CANDIDATE_SCHEMA,
           "feature_code_sha256": "a" * 64, "detector_model_json_sha256": "b" * 64, "pipeline_sha256": "c" * 64}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(rows):
    return b"".join(canonical_json(row) + b"\n" for row in rows)


def sample(kind, mode, identifier, split="test"):
    features = dict.fromkeys(CANDIDATE_FEATURES, 1.0)
    features.update(plane_residual_mm=10 if kind == "live" else 1, nose_protrusion_mm=20 if kind == "live" else 1)
    frame = {"ok": True, "reasons": [], "pad_candidate": {
        "schema": CANDIDATE_SCHEMA, "valid": True, "reasons": [], "features": features}}
    return {"schema": OBSERVATION_SCHEMA, "source": "real_sensor", "consent": True,
            "consent_id": "synthetic-consent", "presentation_id": identifier, "subject_id": "synthetic-subject",
            "instrument_id": None if kind == "live" else "synthetic-" + kind, "kind": kind, "split": split,
            "calib_version": "synthetic-camera", "candidate_binding": copy.deepcopy(BINDING),
            "conditions": {"mode": mode}, "frames": [copy.deepcopy(frame) for _ in range(18 if mode == "enroll" else 3)]}


def runtime_rows():
    return [sample("live", mode, f"new-live-{i}") for i, mode in enumerate(("auth", "enroll", "enroll"))] + [
        sample(kind, "auth", f"new-{kind}-{i}") for kind in ("print", "replay") for i in range(3)]


@pytest.fixture
def inputs():
    baseline = [sample(kind, "diagnostic", f"calibration-{kind}-{i}", "calibration")
                for kind in ("live", "print", "replay") for i in range(3)]
    profile = fit_profile(baseline, binding=BINDING, calib_version="synthetic-camera")
    profile_raw, journal = canonical_json(profile) + b"\n", b""
    summary = {"live": {"presentations": 6, "accepted": 5, "responding": 6, "nonresponses": 0, "errors": 1},
               "print": {"presentations": 8, "accepted": 0, "responding": 7, "nonresponses": 1, "errors": 0},
               "replay": {"presentations": 3, "accepted": 0, "responding": 3, "nonresponses": 0, "errors": 0}}
    prior = canonical_json({"frozen_profile_sha256": sha(profile_raw), "correction_sha256": sha(journal),
                            "eligible_for_local_pilot": False, "test_summary": summary})
    return [encoded(baseline), profile_raw, journal, prior]


def reconciliation(protocol):
    return {"schema": RECONCILIATION_SCHEMA, "protocol_sha256": sha(protocol),
            "started_attempts": 9, "recorded_presentations": 9, "cancelled_attempts": 0,
            "unrecorded_attempts": 0, "operator_confirmed": True,
            "confirmed_at": datetime.now(timezone.utc).isoformat()}


def evaluate(inputs, appended, protocol=None):
    protocol_raw = canonical_json(create_protocol(*inputs)) if protocol is None else protocol
    return evaluate_protocol(inputs[0] + encoded(appended), *inputs[1:], protocol_raw, sha(protocol_raw),
                             canonical_json(reconciliation(protocol_raw)))


def test_passing_cohort_keeps_frozen_base_and_prior_failure(inputs, tmp_path):
    originals = copy.deepcopy(inputs)
    report, profile = evaluate(inputs, runtime_rows())
    assert report["eligible_for_local_pilot"] and not report["reasons"]
    assert report["included_rows"] == 9
    assert report["counts"]["live"] == {"auth": 1, "enroll": 2}
    assert report["quality_passing_responding_auth_rejections"] == {"print": 3, "replay": 3}
    cohort = profile["evaluation"]["prospective_runtime_cohort"]
    assert cohort["prior_diagnostic"]["summary"]["live"]["errors"] == 1
    assert cohort["prior_diagnostic"]["summary"]["live"]["presentations"] == 6
    assert cohort["prior_diagnostic"]["failure_resolved"] is False
    assert cohort["source_prefix"]["sha256"] == sha(inputs[0])
    assert len(cohort["included_row_sha256"]) == 9
    assert cohort["attempt_reconciliation"]["operator_confirmed"] is True
    assert len(cohort["attempt_reconciliation_sha256"]) == 64
    assert profile_payload_sha256(profile) == profile_payload_sha256(json.loads(inputs[1]))
    assert inputs == originals
    raw = canonical_json(profile)
    path = tmp_path / "qualified.json"
    path.write_bytes(raw)
    assert FemtoPADProfile(path, sha(raw)).status("synthetic-camera", BINDING)["ready"]


@pytest.mark.parametrize("field,reason", [
    (0, "original_observation_prefix_changed"), (1, "frozen_profile_hash_mismatch"),
    (2, "corrections_hash_mismatch"), (3, "prior_diagnostic_report_mismatch")])
def test_frozen_sources_cannot_change(inputs, field, reason):
    protocol = canonical_json(create_protocol(*inputs))
    changed = list(inputs)
    changed[field] += b" "
    if field == 0:
        changed[0] = b" " + inputs[0][1:]
    report, profile = evaluate_protocol(changed[0] + encoded(runtime_rows()), *changed[1:], protocol, sha(protocol))
    assert reason in report["reasons"] and profile is None


def test_protocol_pin_and_evaluator_code_are_bound(inputs, monkeypatch):
    protocol = canonical_json(create_protocol(*inputs))
    report, result = evaluate_protocol(inputs[0], *inputs[1:], protocol, "0" * 64)
    assert report["reasons"] == ["protocol_hash_mismatch"] and result is None
    monkeypatch.setattr("scripts.evaluate_femto_runtime._source_hashes", lambda: {"changed": "f" * 64})
    report, result = evaluate_protocol(inputs[0], *inputs[1:], protocol, sha(protocol))
    assert report["reasons"] == ["protocol_evaluator_source_mismatch"] and result is None


@pytest.mark.parametrize("change", ["binding", "calib", "schema", "consent", "source", "duplicate", "old_id"])
def test_new_matching_rows_are_not_silently_filtered(inputs, change):
    rows = runtime_rows()
    if change == "binding":
        rows[-1]["candidate_binding"]["pipeline_sha256"] = "f" * 64
    elif change == "calib":
        rows[-1]["calib_version"] = "different-camera"
    elif change == "schema":
        rows[-1]["schema"] = "invalid"
    elif change == "consent":
        rows[-1]["consent"] = False
    elif change == "source":
        rows[-1]["source"] = "simulator"
    else:
        rows[-1]["presentation_id"] = rows[0]["presentation_id"] if change == "duplicate" else "calibration-live-0"
    report, profile = evaluate(inputs, rows)
    assert profile is None and not report["eligible_for_local_pilot"]
    assert report["included_rows"] == 9 and len(report["included_row_sha256"]) == 9
    assert ("runtime_presentation_not_new_or_unique" if change in ("duplicate", "old_id")
            else "runtime_observation_provenance_mismatch") in report["reasons"]


def test_required_live_rotation_count_cannot_be_replaced_by_more_auth(inputs):
    rows = runtime_rows()
    rows[2] = sample("live", "auth", "new-live-2")
    report, profile = evaluate(inputs, rows)
    assert report["reasons"] == ["runtime_excess_live_auth", "runtime_insufficient_live_enroll"] and profile is None
    # The old evaluator alone would allow auth2/enroll1; no eligible artifact is emitted.
    assert report["evaluation"]["eligible_for_local_pilot"]


@pytest.mark.parametrize("failure", ["quality", "frame_count", "invalid_geometry", "rejection", "timeout"])
def test_bad_live_attempts_remain_and_cannot_be_replaced(inputs, failure):
    rows = runtime_rows()
    bad = sample("live", "enroll", "failed-attempt")
    if failure == "quality":
        bad["frames"][0]["ok"] = False
        bad["frames"][0]["reasons"] = ["pose_out_of_range"]
    elif failure == "frame_count":
        bad["frames"].pop()
    elif failure == "invalid_geometry":
        bad["frames"][0]["pad_candidate"].update(valid=False, reasons=["depth_missing"])
    elif failure == "rejection":
        bad["frames"][0]["pad_candidate"]["features"]["nose_protrusion_mm"] = 0
    else:
        bad["conditions"]["capture_outcome"] = "selection_timeout"
        bad["frames"] = [bad["frames"][0]]
        bad["frames"][0].update(ok=False, reasons=["capture_incomplete"])
        bad["frames"][0]["pad_candidate"].update(valid=False, reasons=["not_measured"])
    rows.insert(0, bad)
    report, profile = evaluate(inputs, rows)
    assert report["included_rows"] == 10 and report["counts"]["live"]["enroll"] == 3
    assert profile is None and report["reasons"]
    assert report["evaluation"]["summary"]["live"]["presentations"] == 4


def test_attacks_must_reach_pad_after_quality_gate_and_are_all_retained(inputs):
    rows = runtime_rows()
    for row in rows:
        if row["kind"] == "print":
            row["frames"][0].update(ok=False, reasons=["pose_out_of_range"])
    report, profile = evaluate(inputs, rows)
    assert report["evaluation"]["eligible_for_local_pilot"]
    assert report["reasons"] == ["runtime_print_quality_passing_auth_rejection_missing"]
    assert report["included_rows"] == 9 and profile is None
    rows.append(sample("print", "auth", "quality-passing-print"))
    report, profile = evaluate(inputs, rows)
    assert profile is None and report["counts"]["print"]["auth"] == 4
    assert "runtime_excess_print_auth" in report["reasons"]
    assert report["quality_passing_responding_auth_rejections"]["print"] == 1


@pytest.mark.parametrize("change", ["missing", "protocol", "unconfirmed", "cancelled", "unrecorded", "started",
                                     "recorded", "timestamp", "predated", "future", "extra_field"])
def test_release_requires_pinned_explicit_operator_attempt_reconciliation(inputs, change):
    protocol = canonical_json(create_protocol(*inputs))
    declaration = reconciliation(protocol)
    if change == "protocol":
        declaration["protocol_sha256"] = "0" * 64
    elif change == "unconfirmed":
        declaration["operator_confirmed"] = False
    elif change in ("cancelled", "unrecorded"):
        declaration[change + "_attempts"] = 1
    elif change == "started":
        declaration["started_attempts"] = 10
    elif change == "recorded":
        declaration["recorded_presentations"] = 8
    elif change == "timestamp":
        declaration["confirmed_at"] = "2026-10-15T00:00:00"
    elif change == "predated":
        declaration["confirmed_at"] = "2020-01-01T00:00:00+00:00"
    elif change == "future":
        declaration["confirmed_at"] = "2999-01-01T00:00:00+00:00"
    elif change == "extra_field":
        declaration["unrecognized"] = True
    raw = None if change == "missing" else canonical_json(declaration)
    report, profile = evaluate_protocol(inputs[0] + encoded(runtime_rows()), *inputs[1:], protocol, sha(protocol), raw)
    assert profile is None and report["included_rows"] == 9
    assert report["reasons"] == ["runtime_attempt_reconciliation_" + ("missing" if change == "missing" else "invalid")]
    assert report["attempt_reconciliation_sha256"] == (None if raw is None else sha(raw))


def test_extra_failed_attack_and_other_rows_are_not_hidden(inputs):
    rows = runtime_rows()
    passed_attack = sample("print", "auth", "accepted-attack")
    passed_attack["frames"] = sample("live", "auth", "unused")["frames"]
    other = [sample("live", "diagnostic", "new-diagnostic"), sample("live", "enroll", "new-calibration", "calibration")]
    report, profile = evaluate(inputs, rows + [passed_attack] + other)
    assert report["included_rows"] == 10 and report["other_appended_rows"] == 2
    assert len(report["other_appended_row_sha256"]) == 2
    assert report["other_appended_reason_counts"] == {
        "not_test_split": 1, "out_of_scope_kind": 0, "out_of_scope_mode": 1}
    assert report["evaluation"]["summary"]["print"]["accepted"] == 1
    assert "pilot_print_not_verified" in report["reasons"] and profile is None


@pytest.mark.parametrize("first", [True, False])
def test_duplicate_selected_id_in_other_appended_row_blocks_regardless_of_order(inputs, first):
    rows = runtime_rows()
    other = sample("live", "diagnostic", rows[0]["presentation_id"])
    rows = [other] + rows if first else rows + [other]
    report, profile = evaluate(inputs, rows)
    assert profile is None and "runtime_presentation_not_new_or_unique" in report["reasons"]
    assert report["included_rows"] == 9 and report["other_appended_rows"] == 1


def test_incomplete_new_log_line_is_not_ignored(inputs):
    protocol = canonical_json(create_protocol(*inputs))
    report, profile = evaluate_protocol(inputs[0] + b'{"incomplete":', *inputs[1:], protocol, sha(protocol))
    assert report["reasons"] == ["observation_log_incomplete_line"] and profile is None
    with pytest.raises(ValueError, match="incomplete_line"):
        create_protocol(inputs[0].rstrip(b"\n"), *inputs[1:])


def test_existing_corrections_apply_without_rewriting_originals(inputs):
    original = sample("replay", "diagnostic", "prior-mislabeled")
    inputs[0] += encoded([original])
    correction = {"schema": "faceauth.pad.label_correction.v1", "presentation_id": original["presentation_id"],
                  "original_row_sha256": sha(canonical_json(original)), "original_kind": "replay",
                  "original_instrument_id": original["instrument_id"], "corrected_kind": "live",
                  "corrected_instrument_id": None, "reason": "Operator corrected physical label",
                  "corrected_at": "2026-10-15T00:00:00+00:00"}
    inputs[2] = encoded([correction])
    prior = json.loads(inputs[3])
    prior["correction_sha256"] = sha(inputs[2])
    inputs[3] = canonical_json(prior)
    originals = copy.deepcopy(inputs)
    report, profile = evaluate(inputs, runtime_rows())
    assert profile is not None and report["included_rows"] == 9
    assert report["corrections_sha256"] == sha(inputs[2])
    assert inputs == originals


def test_cli_private_exclusive_outputs_and_no_profile_until_eligible(inputs, tmp_path, capsys):
    paths = [tmp_path / name for name in ("observations.jsonl", "frozen.json", "corrections.jsonl", "prior.json")]
    common = []
    for option, path, raw in zip(("observations", "profile", "corrections", "prior-report"), paths, inputs):
        path.write_bytes(raw)
        common.extend(["--" + option, str(path)])
    protocol, report, profile = [tmp_path / name for name in ("protocol.json", "report.json", "qualified.json")]
    create = ["create", *common, "--output", str(protocol)]
    assert main(create) == 0
    digest = json.loads(capsys.readouterr().out)["protocol_sha256"]
    assert digest == sha(protocol.read_bytes())
    assert stat.S_IMODE(protocol.stat().st_mode) == 0o600
    assert main(create) == 2
    evaluate_command = ["evaluate", *common, "--protocol", str(protocol), "--protocol-sha256", digest,
                        "--report", str(report), "--output-profile", str(profile)]
    assert main(evaluate_command) == 2
    assert json.loads(report.read_bytes())["included_rows"] == 0
    assert not profile.exists() and stat.S_IMODE(report.stat().st_mode) == 0o600
    paths[0].write_bytes(inputs[0] + encoded(runtime_rows()))
    # Each evaluation report is exclusive, including insufficient ones.
    assert main(evaluate_command) == 2 and not profile.exists()
    new_report = tmp_path / "report-passing.json"
    evaluate_command[evaluate_command.index(str(report))] = str(new_report)
    declaration = tmp_path / "reconciliation.json"
    declaration.write_bytes(canonical_json(reconciliation(protocol.read_bytes())))
    evaluate_command.extend(["--attempt-reconciliation", str(declaration)])
    assert main(evaluate_command) == 0
    assert stat.S_IMODE(profile.stat().st_mode) == 0o600
    assert paths[1].read_bytes() == inputs[1]
    assert json.loads(new_report.read_bytes())["eligible_for_local_pilot"]
