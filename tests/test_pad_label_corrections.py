"""Auditable metadata corrections preserve original observations and capture facts."""

import copy
import hashlib
import json

import pytest

from faceauth.common.pad_validation import canonical_json
from scripts import calibrate_femto_pad as cli


BINDING = {"schema": "faceauth.femto_pad.binding.v1", "feature_schema": "faceauth.femto_pad.features.v1",
           "feature_code_sha256": "a" * 64, "detector_model_json_sha256": "b" * 64,
           "pipeline_sha256": "c" * 64}


def observation():
    return {"presentation_id": "synthetic-presentation", "kind": "replay", "instrument_id": "synthetic-display",
            "split": "test", "conditions": {"mode": "diagnostic"}, "calib_version": "synthetic-sensor",
            "candidate_binding": dict(BINDING), "frames": [{"measurement": index} for index in range(5)]}


def correction(row):
    return {"schema": cli.CORRECTION_SCHEMA, "presentation_id": row["presentation_id"],
            "original_row_sha256": hashlib.sha256(canonical_json(row)).hexdigest(),
            "original_kind": row["kind"], "original_instrument_id": row["instrument_id"],
            "corrected_kind": "live", "corrected_instrument_id": None,
            "reason": "Participant explicitly corrected the selected physical label",
            "corrected_at": "2026-10-15T13:00:00+09:00"}


def test_correction_changes_only_labels_and_deep_copies_every_row():
    row = observation()
    untouched = {**observation(), "presentation_id": "another-synthetic-presentation"}
    rows = [row, untouched]
    original = copy.deepcopy(rows)
    journal = [correction(row)]
    result = cli.apply_label_corrections(rows, journal)
    assert rows == original and journal == [correction(row)]
    assert result[0] == {**row, "kind": "live", "instrument_id": None}
    assert result[0]["conditions"] == {"mode": "diagnostic"}
    assert result[0]["split"] == "test" and len(result[0]["frames"]) == 5
    result[0]["frames"][0]["measurement"] = 999
    result[1]["conditions"]["mode"] = "auth"
    assert rows == original


def test_instrument_only_correction_is_supported():
    row = observation()
    record = {**correction(row), "corrected_kind": "replay", "corrected_instrument_id": "corrected-display"}
    result = cli.apply_label_corrections([row], [record])
    assert result[0] == {**row, "instrument_id": "corrected-display"}


@pytest.mark.parametrize("change", ["unknown_target", "duplicate_target", "duplicate_correction", "changed_row",
                                   "wrong_hash", "wrong_original_kind", "wrong_original_instrument"])
def test_correction_requires_exact_unique_original_record(change):
    row = observation()
    rows, journal = [row], [correction(row)]
    if change == "unknown_target":
        journal[0]["presentation_id"] = "unknown-synthetic-presentation"
    elif change == "duplicate_target":
        rows.append(copy.deepcopy(row))
    elif change == "duplicate_correction":
        journal.append(copy.deepcopy(journal[0]))
    elif change == "changed_row":
        row["frames"][0]["measurement"] = 999
    elif change == "wrong_hash":
        journal[0]["original_row_sha256"] = "0" * 64
    elif change == "wrong_original_kind":
        journal[0]["original_kind"] = "print"
    else:
        journal[0]["original_instrument_id"] = "wrong-display"
    with pytest.raises(ValueError):
        cli.apply_label_corrections(rows, journal)


@pytest.mark.parametrize("extra", ["frames", "conditions", "split", "binding", "subject_id", "consent"])
def test_journal_cannot_modify_capture_or_identity_fields(extra):
    row = observation()
    record = {**correction(row), extra: "forbidden modification"}
    with pytest.raises(ValueError, match="schema or fields"):
        cli.apply_label_corrections([row], [record])


@pytest.mark.parametrize("field,value", [
    ("schema", "unknown"), ("presentation_id", ""), ("original_row_sha256", None),
    ("corrected_kind", "unknown"), ("corrected_kind", None), ("corrected_instrument_id", "display"),
    ("reason", "  "), ("reason", None), ("corrected_at", "2026-10-15T13:00:00"),
    ("corrected_at", "not a date"), ("corrected_at", None),
])
def test_invalid_journal_fields_are_rejected(field, value):
    row = observation()
    record = {**correction(row), field: value}
    with pytest.raises(ValueError):
        cli.apply_label_corrections([row], [record])


@pytest.mark.parametrize("instrument", [None, "", "  ", 123, []])
def test_attack_correction_requires_nonempty_instrument(instrument):
    row = observation()
    record = {**correction(row), "corrected_kind": "print", "corrected_instrument_id": instrument}
    with pytest.raises(ValueError):
        cli.apply_label_corrections([row], [record])


@pytest.mark.parametrize("command", ["fit", "evaluate"])
def test_cli_applies_journal_without_rewriting_sources_and_logs_only_counts_and_hashes(tmp_path, monkeypatch, capsys, command):
    row = observation()
    row["split"] = "calibration" if command == "fit" else "test"
    observations, corrections = tmp_path / "observations.jsonl", tmp_path / "corrections.jsonl"
    original = canonical_json(row) + b"\n"
    journal = canonical_json(correction(row)) + b"\n"
    observations.write_bytes(original)
    corrections.write_bytes(journal)
    output = tmp_path / "output.json"
    seen = []

    def fake_fit(rows, **kwargs):
        seen.extend(rows)
        return {"fixture": "calibration"}

    def fake_evaluate(profile, rows):
        seen.extend(rows)
        return {"fixture": "evaluation", "evaluation": {"eligible_for_local_pilot": True, "reasons": []}}

    monkeypatch.setattr(cli, "fit_profile", fake_fit)
    monkeypatch.setattr(cli, "evaluate_profile", fake_evaluate)
    args = [command, "--observations", str(observations), "--corrections", str(corrections), "--output", str(output)]
    if command == "fit":
        args.extend(["--calib-version", "synthetic-sensor"])
    else:
        profile = tmp_path / "profile.json"
        profile.write_bytes(canonical_json({"candidate_binding": BINDING, "calib_version": "synthetic-sensor"}))
        args.extend(["--profile", str(profile)])
    assert cli.main(args) == 0
    assert seen == [{**row, "kind": "live", "instrument_id": None}]
    assert observations.read_bytes() == original and corrections.read_bytes() == journal
    stdout = capsys.readouterr().out
    summary = json.loads(stdout)
    assert summary["label_corrections"] == 1
    assert summary["corrections_sha256"] == hashlib.sha256(journal).hexdigest()
    assert row["presentation_id"] not in stdout and row["instrument_id"] not in stdout
    assert correction(row)["reason"] not in stdout


def test_invalid_correction_never_writes_output(tmp_path, capsys):
    row = observation()
    observations, corrections, output = (tmp_path / name for name in ("rows.jsonl", "corrections.jsonl", "profile.json"))
    observations.write_bytes(canonical_json(row))
    record = {**correction(row), "original_row_sha256": "0" * 64}
    corrections.write_bytes(canonical_json(record))
    assert cli.main(["fit", "--observations", str(observations), "--corrections", str(corrections),
                     "--calib-version", "synthetic-sensor", "--output", str(output)]) == 2
    assert not output.exists()
    assert row["presentation_id"] not in capsys.readouterr().err
