"""Synthetic robustness regressions; passing tests do not validate physical PAD."""

import hashlib
import json
import stat
from pathlib import Path

import numpy as np
import pytest

from faceauth.models import femto_pad
from scripts.diagnose_femto_nose import KPS, main, run_diagnostics


@pytest.fixture(scope="module")
def report():
    return run_diagnostics()


@pytest.fixture(scope="module")
def scenarios(report):
    return {row["scenario"]: row for row in report["scenarios"]}


def test_small_landmark_displacement_retains_localized_nose_relief(scenarios):
    original = scenarios["frontal_relief"]
    shifted = scenarios["nose_landmark_offset_8px"]
    assert original["valid"] and shifted["valid"]
    baseline, changed = original["features"], shifted["features"]
    assert baseline["nose_protrusion_mm"] > 10
    assert .8 < changed["nose_protrusion_mm"] / baseline["nose_protrusion_mm"] < 1.2
    assert abs(changed["plane_residual_mm"] - baseline["plane_residual_mm"]) < 1
    assert changed["nose_valid_ratio"] > .9 and changed["surround_valid_ratio"] > .9


def test_cross_modal_relief_displacement_retains_localized_nose_relief(scenarios):
    original = scenarios["frontal_relief"]["features"]
    shifted = scenarios["relief_tip_offset_8px"]
    assert shifted["valid"]
    assert .8 < shifted["features"]["nose_protrusion_mm"] / original["nose_protrusion_mm"] < 1.2
    assert shifted["features"]["plane_residual_mm"] > 8


@pytest.mark.parametrize("name", ["flat", "tilted_plane_-0.12", "tilted_plane_0.05", "tilted_plane_0.25",
                                  "rotated_plane_0_0", "rotated_plane_-30_0", "rotated_plane_30_0",
                                  "rotated_plane_0_-20", "rotated_plane_0_20", "rotated_plane_25_15"])
def test_physical_planes_do_not_gain_relief_from_perspective_tilt(scenarios, name):
    row = scenarios[name]
    assert row["valid"]
    assert row["features"]["plane_residual_mm"] < 1
    assert row["features"]["quadratic_residual_mm"] < 1
    assert abs(row["features"]["nose_protrusion_mm"]) < 1


def test_smooth_bend_and_crease_are_distinct_from_flat_but_not_proof_of_life(scenarios):
    bend, crease = scenarios["smooth_bend"], scenarios["creased_surface"]
    assert bend["valid"] and crease["valid"]
    assert bend["features"]["plane_residual_mm"] > 5
    assert bend["features"]["quadratic_residual_mm"] < 1
    assert abs(bend["features"]["nose_protrusion_mm"]) < 1
    assert crease["features"]["quadratic_residual_mm"] > 2
    assert crease["features"]["nose_protrusion_mm"] < 2
    assert "live" not in crease and "score" not in crease


def test_independent_noise_is_not_confused_with_a_registered_frontal_bump(scenarios):
    noise = scenarios["independent_depth_noise"]
    assert noise["valid"]
    assert noise["features"]["plane_residual_mm"] < 3
    assert noise["features"]["quadratic_residual_mm"] < 3
    assert abs(noise["features"]["nose_protrusion_mm"]) < 3


def test_depth_step_artifact_is_unmeasurable_despite_high_global_relief(scenarios, report):
    artifact = scenarios["depth_step_artifact"]
    relief = scenarios["frontal_relief"]
    assert not artifact["valid"] and relief["valid"]
    assert artifact["reasons"] == ["nose_depth_discontinuity"]
    assert artifact["features"]["nose_protrusion_mm"] is None
    for key in ("plane_residual_mm", "quadratic_residual_mm"):
        assert artifact["features"][key] > relief["features"][key]
    assert report["comparisons"]["depth_step_exceeds_frontal_in_both_relief_residuals"]
    assert not report["comparisons"]["depth_step_exceeds_frontal_in_signed_nose_relief"]
    assert report["comparisons"]["depth_step_measurement_rejected"]


@pytest.mark.parametrize("name", ["rotated_toy_relief_0_0", "rotated_toy_relief_-30_0", "rotated_toy_relief_30_0",
                                  "rotated_toy_relief_0_-20", "rotated_toy_relief_0_20", "rotated_toy_relief_25_15"])
def test_rotated_toy_is_measurable_or_conservatively_rejects_depth_boundaries(scenarios, name):
    # This idealized counterexample prevents claiming a real angle fix from
    # toy rendering. Real landmark error, occlusion and sensor effects are absent.
    row = scenarios[name]
    if row["valid"]:
        assert row["features"]["nose_protrusion_mm"] > 10
    else:
        assert row["reasons"] == ["nose_depth_discontinuity"]
        assert row["features"]["nose_protrusion_mm"] is None
    assert row["family"] == "rotated_toy_relief"


def test_report_is_numerical_synthetic_evidence_without_a_profile_or_decision(report):
    assert report["source"] == "synthetic_only" and report["deployable_profile"] is False
    assert report["historical_test_reclassified"] is False
    assert "thresholds" not in report and "eligible_for_local_pilot" not in report
    assert len(report["scenarios"]) == 31
    for row in report["scenarios"]:
        assert "landmarks" not in row and "depth" not in row and "image" not in row
        assert all(value is None or type(value) is float for value in row["features"].values())
    json.dumps(report, allow_nan=False)
    assert report["extractor_source_sha256"] == hashlib.sha256(Path(femto_pad.__file__).read_bytes()).hexdigest()


def test_exclusive_private_cli_artifact_does_not_modify_extractor(tmp_path, capsys, monkeypatch, report):
    original_source = Path(femto_pad.__file__).read_bytes()
    original_landmarks = KPS.copy()
    # The renderer is exercised above; reuse its report to test file handling.
    monkeypatch.setattr("scripts.diagnose_femto_nose.run_diagnostics", lambda: report)
    output = tmp_path / "synthetic-diagnostics.json"
    assert main(["--output", str(output)]) == 0
    stdout = json.loads(capsys.readouterr().out)
    assert stdout["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert stdout["source"] == "synthetic_only" and stdout["deployable_profile"] is False
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        main(["--output", str(output)])
    assert Path(femto_pad.__file__).read_bytes() == original_source
    np.testing.assert_array_equal(KPS, original_landmarks)
