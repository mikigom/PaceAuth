"""Offline synthetic diagnostics for the current Femto geometry extractor.

These artificial depth surfaces are not people, attacks, camera measurements,
calibration data, or a fresh test set. No PAD classifier is fitted or produced.
The runner documents landmark sensitivity and the ambiguity of relief features.
Pixel offsets refer to the 128-by-128 synthetic input, not sensor-image pixels.
Its rotated heightfield uses explicit 3D rotation and perspective projection,
but has ideal depth/landmarks and no realistic anatomy or ToF sensor physics.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from faceauth.common.align import ARCFACE_DST
from faceauth.common.pad_validation import canonical_json
from faceauth.models import femto_pad

SCHEMA = "faceauth.femto_pad.synthetic_nose_diagnostics.v2"
KPS = 64 + (ARCFACE_DST * (128 / 112) - 64) / 1.6
YY, XX = np.mgrid[:128, :128]
X, Y = (XX - 64) / 24, (YY - 73) / 28
DESCRIPTORS = ("plane_residual_mm", "quadratic_residual_mm", "nose_plane_protrusion_mm", "nose_protrusion_mm")


def measure(depth, landmarks=None):
    """Exercise the installed extractor with generated arrays only."""
    return femto_pad.measure_features(
        np.rint(depth).clip(0, 65535).astype(np.uint16), np.full((128, 128), 150, np.uint8),
        KPS.copy() if landmarks is None else landmarks)


def curved_surface(*, tip_offset_px=0.0, nose=True):
    depth = 600 + 35 * X * X + 10 * Y * Y
    if nose:
        depth -= 30 * np.exp(-((XX - KPS[2, 0] - tip_offset_px) / 5.5) ** 2
                             - ((YY - KPS[2, 1]) / 6) ** 2)
    return depth


def rotated_heightfield(yaw_deg=0.0, pitch_deg=0.0, *, flat=False):
    """Render a toy 3D relief using nearest-depth z-buffer visibility.

    The dense source surface prevents raster holes in the central face patch.
    Landmarks are projected from the same 3D surface without detection error.
    A flat variant is a physical plane, including its perspective depth tilt.
    Returns ephemeral arrays; the report contains numerical descriptors only.
    """
    distance, focal = 650.0, 240.0
    cx, cy = map(float, KPS[2])
    def relief(x, y):
        return np.zeros_like(x) if flat else (
            35 * (x / 65) ** 2 + 10 * (y / 75) ** 2
            - 30 * np.exp(-(x / 15) ** 2 - (y / 16) ** 2))

    yaw, pitch = np.deg2rad([yaw_deg, pitch_deg])
    ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    rotation = rx @ ry

    def project(points):
        camera = points @ rotation.T + [0, 0, distance]
        pixels = camera[:, :2] / camera[:, 2, None] * focal + [cx, cy]
        return pixels, camera[:, 2]

    y, x = np.meshgrid(np.linspace(-130, 130, 701), np.linspace(-115, 115, 621), indexing="ij")
    pixels, z = project(np.column_stack((x.ravel(), y.ravel(), relief(x, y).ravel())))
    ix, iy = np.rint(pixels).astype(int).T
    visible = (ix >= 0) & (ix < 128) & (iy >= 0) & (iy < 128) & (z > 0)
    depth = np.full(128 * 128, np.inf)
    np.minimum.at(depth, iy[visible] * 128 + ix[visible], z[visible])
    depth[~np.isfinite(depth)] = 0
    xy = (KPS - [cx, cy]) * distance / focal
    landmarks, _ = project(np.column_stack((xy, relief(xy[:, 0], xy[:, 1]))))
    return depth.reshape(128, 128), landmarks


def scenarios():
    """Fixed deterministic experiments, independent of any held-out observations."""
    yield "flat", "planar_surface", {}, np.full((128, 128), 600.0), KPS
    for tilt in (-.12, .05, .25):
        yield f"tilted_plane_{tilt}", "planar_surface", {"inverse_depth_x_slope": tilt}, 600 / (1 + tilt * X + .06 * Y), KPS
    yield "smooth_bend", "curved_surface", {}, curved_surface(nose=False), KPS
    yield "creased_surface", "creased_surface", {}, 600 + 35 * np.abs(X) + 10 * Y * Y, KPS
    noise = np.random.default_rng(7).normal(0, 5, (128, 128))
    yield "independent_depth_noise", "measurement_artifact", {"noise_sigma_mm": 5}, 600 + noise, KPS
    yield "depth_step_artifact", "measurement_artifact", {"step_mm": 100}, 600 + 100 * (XX > 64), KPS
    depth = curved_surface()
    yield "frontal_relief", "localized_relief", {}, depth, KPS
    for offset in (2, 4, 6, 8, 10):
        landmarks = KPS.copy()
        landmarks[2, 0] += offset
        yield f"nose_landmark_offset_{offset}px", "landmark_error", {"nose_landmark_x_offset_px": offset}, depth, landmarks
        yield f"relief_tip_offset_{offset}px", "cross_modal_displacement", {"relief_tip_x_offset_px": offset}, curved_surface(tip_offset_px=offset), KPS
    for flat in (True, False):
        for yaw, pitch in ((0, 0), (-30, 0), (30, 0), (0, -20), (0, 20), (25, 15)):
            depth, landmarks = rotated_heightfield(yaw, pitch, flat=flat)
            kind = "rotated_plane" if flat else "rotated_toy_relief"
            yield f"{kind}_{yaw}_{pitch}", kind, {"yaw_deg": yaw, "pitch_deg": pitch}, depth, landmarks


def run_diagnostics():
    results = []
    for name, family, parameters, depth, landmarks in scenarios():
        result = measure(depth, landmarks)
        results.append({"scenario": name, "family": family, "parameters": parameters, **result})
    indexed = {row["scenario"]: row for row in results}
    baseline = indexed["frontal_relief"]["features"]
    offsets = []
    for offset in (2, 4, 6, 8, 10):
        row = indexed[f"nose_landmark_offset_{offset}px"]
        offsets.append({"offset_px": offset, "measurement_valid": row["valid"],
                        **{key: row["features"][key] for key in DESCRIPTORS}})
    artifact = indexed["depth_step_artifact"]["features"]
    return {"schema": SCHEMA, "source": "synthetic_only", "deployable_profile": False,
            "synthetic_input_width_px": 128, "synthetic_input_height_px": 128,
            "historical_test_reclassified": False,
            "extractor_source_sha256": hashlib.sha256(Path(femto_pad.__file__).read_bytes()).hexdigest(),
            "experiment_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scenarios": results,
            "comparisons": {"frontal_relief": {key: baseline[key] for key in DESCRIPTORS},
                            "nose_landmark_offsets": offsets,
                            "depth_step_exceeds_frontal_in_both_relief_residuals": all(
                                artifact[key] > baseline[key] for key in ("plane_residual_mm", "quadratic_residual_mm")),
                            "depth_step_exceeds_frontal_in_signed_nose_relief":
                                artifact["nose_protrusion_mm"] is not None and
                                artifact["nose_protrusion_mm"] > baseline["nose_protrusion_mm"],
                            "depth_step_measurement_rejected": not indexed["depth_step_artifact"]["valid"],
                            "landmark_offset_8px_retained_nose_fraction":
                                indexed["nose_landmark_offset_8px"]["features"]["nose_protrusion_mm"] /
                                baseline["nose_protrusion_mm"]},
            "interpretation": [
                "Measurement coverage does not establish nose-tip localization.",
                "Bounded patch localization reduces the tested synthetic landmark-displacement sensitivity.",
                "Large plane and quadratic residuals can also describe depth discontinuities, not living faces.",
                "Coherent local depth steps are rejected as unmeasurable rather than granted a nose score.",
                "Directional prominence limits the tested straight crease, but does not prove resistance to bent photographs.",
                "Ideal rotated heightfields do not establish behavior on real rotated human faces.",
                "Patch localization remains an unvalidated experiment: folds, occluders and depth artifacts can create peaks.",
                "This report fits no thresholds, supplies no alternative PAD decision and changes no prior evaluation."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional exclusive 0600 numerical report; default is stdout")
    args = parser.parse_args(argv)
    report = run_diagnostics()
    if args.output:
        raw = canonical_json(report) + b"\n"
        descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
        print(json.dumps({"sha256": hashlib.sha256(raw).hexdigest(), "scenarios": len(report["scenarios"]),
                          "source": "synthetic_only", "deployable_profile": False}))
    else:
        print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
