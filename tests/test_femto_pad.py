"""Synthetic geometry contracts only; these tests establish no physical PAD accuracy."""

import json

import numpy as np
import pytest

from faceauth.common.align import ARCFACE_DST
from faceauth.models.femto_pad import FEATURE_NAMES, FEATURE_SCHEMA, measure_features

KPS = 64 + (ARCFACE_DST * (128 / 112) - 64) / 1.6
YY, XX = np.mgrid[:128, :128]
X, Y = (XX - 64) / 24, (YY - 73) / 28
IR = np.full((128, 128), 150, np.uint8)


def measure(z, ir=IR, kps=KPS):
    return measure_features(np.rint(z).clip(0, 65535).astype(np.uint16), ir, kps)


def test_flat_depth_is_measurable_but_never_declared_live():
    output = measure(np.full((128, 128), 600.0))
    assert output["valid"] and not output["reasons"]
    assert output["schema"] == FEATURE_SCHEMA
    assert set(output) == {"schema", "valid", "reasons", "features"}
    assert set(output["features"]) == set(FEATURE_NAMES)
    assert output["features"]["valid_ratio"] == 1.0
    for key in ("plane_residual_mm", "quadratic_residual_mm", "nose_protrusion_mm"):
        assert abs(output["features"][key]) < 1e-6
    json.dumps(output, allow_nan=False)


@pytest.mark.parametrize("tilt", [0.05, -.12, .25])
def test_inverse_depth_plane_removes_physical_screen_tilt(tilt):
    # A plane in 3D projects to an affine inverse depth, not affine depth.
    output = measure(600 / (1 + tilt * X + .06 * Y))
    assert output["valid"]
    assert output["features"]["depth_iqr_mm"] > 20
    assert output["features"]["plane_residual_mm"] < .8
    assert abs(output["features"]["nose_protrusion_mm"]) < .8


def test_quadratic_reference_distinguishes_smooth_bend_from_localized_nose():
    bent = 600 + 35 * X * X + 10 * Y * Y
    nose = 30 * np.exp(-((XX - KPS[2, 0]) / 5.5) ** 2 - ((YY - KPS[2, 1]) / 6) ** 2)
    paper, relief = measure(bent), measure(bent - nose)
    assert paper["valid"] and relief["valid"]
    assert paper["features"]["plane_residual_mm"] > 5
    assert paper["features"]["quadratic_residual_mm"] < 1.5
    assert abs(paper["features"]["nose_protrusion_mm"]) < 2
    assert relief["features"]["nose_protrusion_mm"] > 10
    assert relief["features"]["quadratic_residual_mm"] > paper["features"]["quadratic_residual_mm"]


def test_block_medians_limit_independent_depth_noise():
    rng = np.random.default_rng(7)
    output = measure(600 + rng.normal(0, 5, (128, 128)))
    assert output["valid"]
    assert output["features"]["plane_residual_mm"] < 3
    assert abs(output["features"]["nose_protrusion_mm"]) < 3


def test_sparse_random_holes_remain_measurable():
    rng = np.random.default_rng(4)
    depth = np.full((128, 128), 600.0)
    depth[rng.random(depth.shape) < .08] = 0
    output = measure(depth)
    assert output["valid"]
    assert .85 < output["features"]["valid_ratio"] < 1
    assert output["features"]["plane_residual_mm"] < 1e-6


@pytest.mark.parametrize("missing", ["all", "nose", "surround", "half", "background"])
def test_missing_spatial_coverage_never_fabricates_geometry(missing):
    depth = np.full((128, 128), 600.0)
    if missing == "all":
        depth[:] = 0
    elif missing == "nose":
        depth[65:86, 54:75] = 0
    elif missing == "surround":
        depth[:, :58] = depth[:, 71:] = 0
    elif missing == "half":
        depth[:, :65] = 0
    else:
        depth[65:86, 54:75] = 3000
    output = measure(depth)
    assert not output["valid"] and output["reasons"]
    assert output["features"]["nose_protrusion_mm"] is None
    json.dumps(output, allow_nan=False)


def test_ir_is_only_a_diagnostic_and_missing_ir_fails_measurement():
    depth = np.full((128, 128), 600.0)
    dark, bright = measure(depth, np.full_like(IR, 10)), measure(depth, np.full_like(IR, 240))
    assert dark["valid"] and bright["valid"]
    assert dark["features"]["nose_protrusion_mm"] == bright["features"]["nose_protrusion_mm"]
    assert dark["features"]["ir_median"] == 10
    assert bright["features"]["ir_median"] == 240
    missing = measure(depth, np.zeros_like(IR))
    assert not missing["valid"] and "ir_missing" in missing["reasons"]


@pytest.mark.parametrize("depth,ir", [
    (None, IR), (np.zeros((128, 128), np.float32), IR),
    (np.full((128, 128), np.nan), IR), (np.zeros((128, 128), np.uint16), IR[:64]),
    (np.zeros((128, 128, 1), np.uint16), IR), (np.zeros((8, 8), np.uint16), IR[:8, :8]),
    (np.zeros((128, 128), np.uint16), IR.astype(np.uint16)),
])
def test_bad_arrays_return_finite_failure(depth, ir):
    output = measure_features(depth, ir, KPS)
    assert not output["valid"] and output["reasons"] == ["invalid_input"]
    json.dumps(output, allow_nan=False)


@pytest.mark.parametrize("kps", [
    None, [], np.zeros((5, 2)), np.full((5, 2), np.nan), np.full((5, 2), np.inf),
    KPS * .1, KPS + 500, KPS[:, ::-1], KPS[[0, 1, 2, 4, 3]],
])
def test_degenerate_or_nonfinite_landmarks_fail(kps):
    output = measure(np.full((128, 128), 600.0), kps=kps)
    assert not output["valid"]
    assert output["reasons"][0] in ("invalid_landmarks", "landmarks_too_small")
    json.dumps(output, allow_nan=False)


def test_inputs_are_not_mutated_and_output_contains_no_images_or_landmarks():
    depth = np.full((128, 128), 600, np.uint16)
    ir, kps = IR.copy(), KPS.copy()
    measure_features(depth, ir, kps)
    np.testing.assert_array_equal(depth, np.full_like(depth, 600))
    np.testing.assert_array_equal(ir, IR)
    np.testing.assert_array_equal(kps, KPS)


@pytest.mark.parametrize("offset", [(-8, 0), (8, 0), (0, -8), (0, 8), (-6, -6), (6, -6), (-6, 6), (6, 6)])
@pytest.mark.parametrize("kind", ["landmark", "depth"])
def test_localization_retains_relief_for_small_offsets_in_all_directions(offset, kind):
    dx, dy = offset
    kps = KPS.copy()
    if kind == "landmark":
        kps[2] += [dx, dy]
        dx = dy = 0
    nose = 30 * np.exp(-((XX - KPS[2, 0] - dx) / 5.5) ** 2 - ((YY - KPS[2, 1] - dy) / 6) ** 2)
    output = measure(600 + 35 * X * X + 10 * Y * Y - nose, kps=kps)
    assert output["valid"]
    assert output["features"]["nose_protrusion_mm"] > 10
    assert output["schema"] == "faceauth.femto_pad.features.v2"


@pytest.mark.parametrize("angle", range(0, 180, 15))
def test_oblique_straight_creases_do_not_gain_localized_nose_relief(angle):
    theta = np.deg2rad(angle)
    axis = ((XX - KPS[2, 0]) * np.cos(theta) + (YY - KPS[2, 1]) * np.sin(theta)) / 24
    output = measure(600 + 35 * np.abs(axis))
    assert output["valid"]
    assert output["features"]["nose_protrusion_mm"] < 3


@pytest.mark.parametrize("size", [1, 2, 3])
def test_isolated_depth_spikes_cannot_become_a_localized_nose(size):
    depth = np.full((128, 128), 600.)
    depth[74:74 + size, 63:63 + size] = 500
    output = measure(depth)
    assert not output["valid"] or output["features"]["nose_protrusion_mm"] < 2


@pytest.mark.parametrize("axis", [XX, YY])
def test_coherent_depth_steps_fail_measurement_even_with_complete_coverage(axis):
    output = measure(600 + 100 * (axis > (64 if axis is XX else 75)))
    assert not output["valid"]
    assert "nose_depth_discontinuity" in output["reasons"]
    assert output["features"]["nose_protrusion_mm"] is None


def test_search_cannot_escape_missing_depth_at_original_nose():
    depth = 600 + 35 * X * X + 10 * Y * Y
    # A strong neighbouring bump cannot excuse absent depth at the RGB nose.
    depth -= 45 * np.exp(-((XX - KPS[2, 0] - 8) / 5.5) ** 2 - ((YY - KPS[2, 1]) / 6) ** 2)
    depth[((XX - KPS[2, 0]) / 5.5) ** 2 + ((YY - KPS[2, 1]) / 6) ** 2 <= 1] = 0
    output = measure(depth)
    assert not output["valid"] and "nose_coverage" in output["reasons"]
    assert output["features"]["nose_protrusion_mm"] is None
