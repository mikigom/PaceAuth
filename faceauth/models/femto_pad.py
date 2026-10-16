"""Measured Femto Bolt facial geometry, not a live/spoof classifier.

The input is the existing synchronized, color-registered depth (millimetres)
and log-mapped IR stream. IR is diagnostic intensity, not evidence of skin.
``valid`` means that spatial measurements could be made; even a perfectly
flat display can be valid. No threshold in this module grants authentication.

Inverse depth is affine on a physical plane under perspective projection.
Removing that plane avoids mistaking a tilted photograph for facial relief.
A quadratic reference and a localized nose measurement additionally describe
curved surfaces, but do not establish resistance to masks or bent photographs.
Real presentations are required to calibrate and test a downstream decision.

Version 2 localizes a supported nose-sized patch within 10 canonical pixels of
the RGB landmark. ``nose_protrusion_mm`` is the smaller of its surround-relative
relief and its minimum prominence across eight directions. A long crease cannot
obtain a nose descriptor merely by protruding along one axis. Coherent jumps in
the measured patch fail measurement, including genuine self-occlusion when it
prevents measuring a continuous surface. This changes descriptor semantics;
version 1 thresholds and evaluations do not validate version 2 behavior.
"""

import cv2
import numpy as np

from faceauth.common.align import ARCFACE_DST
from faceauth.models.pad import SIZE, ZOOM, crop_affine

FEATURE_SCHEMA = "faceauth.femto_pad.features.v2"
FEATURE_NAMES = (
    "valid_ratio", "surface_ratio", "valid_blocks_ratio", "depth_median_mm", "depth_iqr_mm",
    "plane_residual_mm", "quadratic_residual_mm", "nose_plane_protrusion_mm", "nose_protrusion_mm",
    "nose_valid_ratio", "surround_valid_ratio", "ir_median", "ir_iqr",
)


def _empty(reason):
    return {"schema": FEATURE_SCHEMA, "valid": False, "reasons": [reason],
            "features": dict.fromkeys(FEATURE_NAMES)}


def _finish(features, reasons):
    # Incomplete/degenerate measurements never leak NaN into signed JSON.
    clean = {key: float(value) if value is not None and np.isfinite(value) else None
             for key, value in features.items()}
    if not reasons and any(value is None for value in clean.values()):
        reasons = ["geometry_degenerate"]
    return {"schema": FEATURE_SCHEMA, "valid": not reasons, "reasons": list(dict.fromkeys(reasons)),
            "features": clean}


def _design(x, y, quadratic=False):
    columns = [np.ones_like(x), x, y]
    if quadratic:
        columns.extend([x * x, x * y, y * y])
    return np.stack(columns, axis=-1)


def _fit_inverse(A, z, median):
    """Robust inverse-depth fit, with a depth-equivalent scale in millimetres."""
    target = median * median * (1.0 / z - 1.0 / median)
    keep = np.ones(len(z), dtype=bool)
    for _ in range(3):
        selected = A[keep]
        if len(selected) < 2 * A.shape[1] or np.linalg.matrix_rank(selected) != A.shape[1]:
            raise ValueError("insufficient independent surface samples")
        coeff = np.linalg.lstsq(selected, target[keep], rcond=None)[0]
        residual = target - A @ coeff
        middle = np.median(residual[keep])
        spread = 1.4826 * np.median(np.abs(residual[keep] - middle))
        keep = np.abs(residual - middle) <= max(3.0 * spread, 2.0)
    return coeff


def _predict(A, coeff, median):
    inverse = 1.0 / median + (A @ coeff) / (median * median)
    if not np.isfinite(inverse).all() or np.any(inverse <= 0):
        raise ValueError("invalid depth surface")
    return 1.0 / inverse


def _residual_rms(residual):
    # A few ToF edge outliers should not dominate a face-level descriptor.
    magnitude = np.abs(residual)
    return np.sqrt(np.mean(np.minimum(magnitude, np.quantile(magnitude, .95)) ** 2))


def _nose_patch(xx, yy, core, cx, cy):
    return core & (((xx - cx) / 5.5) ** 2 + ((yy - cy) / 6.0) ** 2 <= 1)


def _localize_nose(xx, yy, core, surface, residual, nx, ny):
    """Find a supported patch within a bounded landmark-error neighbourhood.

    This uses a median over a nose-sized region, not the nearest depth pixel.
    A candidate must retain depth in every quadrant, and its entire patch must
    fit inside the facial core. The same search is applied to flat/curved attack
    surfaces; it is a measurement correction, never evidence of a living face.
    """
    candidates = []
    for dy in range(-10, 11, 2):
        for dx in range(-10, 11, 2):
            if dx * dx + dy * dy > 100:
                continue
            cx, cy = nx + dx, ny + dy
            ellipse = ((xx - cx) / 5.5) ** 2 + ((yy - cy) / 6.0) ** 2 <= 1
            if np.any(ellipse & ~core):
                continue
            selected = ellipse & surface
            if np.mean(surface[ellipse]) < .85:
                continue
            quadrants = [(xx < cx) & (yy < cy), (xx >= cx) & (yy < cy),
                         (xx < cx) & (yy >= cy), (xx >= cx) & (yy >= cy)]
            if any(np.mean(surface[ellipse & quadrant]) < .70 for quadrant in quadrants):
                continue
            score = np.median(residual[selected])
            candidates.append((float(score), -(dx * dx + dy * dy), cx, cy))
    if not candidates:
        raise ValueError("nose_coverage")
    _, _, cx, cy = max(candidates)
    return float(cx), float(cy)


def _has_depth_discontinuity(depth, surface, neighbourhood):
    """Reject coherent jumps; neither a missing pixel nor a spike is a nose.

    A 3x3 median suppresses isolated ToF impulses. More than two neighbouring
    pairs with a > 20 mm jump in the canonical crop indicate a depth boundary,
    rather than a continuously sampled nasal surface. This is a conservative
    measurement-quality bound, not a calibrated PAD decision threshold.
    Missing samples are never interpolated for measurement or this check.
    """
    smoothed = cv2.medianBlur(depth.astype(np.float32), 3)
    # Do not allow zero-filled holes in a median window to create an apparent
    # jump between two otherwise valid pixels.
    supported = cv2.erode(surface.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    neighbourhood = neighbourhood & supported
    count = 0
    for axis in (0, 1):
        first = np.s_[:-1, :] if axis == 0 else np.s_[:, :-1]
        second = np.s_[1:, :] if axis == 0 else np.s_[:, 1:]
        eligible = neighbourhood[first] & neighbourhood[second]
        count += np.count_nonzero(eligible & (np.abs(smoothed[first] - smoothed[second]) > 20.0))
    return count > 2


def _nose_prominence(xx, yy, core, surface, residual, nx, ny, nose):
    """Directional prominence distinguishes a localized peak from a long ridge.

    Each flank must have independently supported depth. Averaging the opposite
    flanks cancels a remaining linear slope. Taking the minimum contrast across
    eight directions also covers oblique creases rather than only x/y ridges.
    These millimetre descriptors do not classify presentations as live.
    """
    centre = np.median(residual[nose & surface])
    axes = []
    for angle in np.arange(8) * np.pi / 8:
        dx, dy = 10 * np.cos(angle), 10 * np.sin(angle)
        flanks = []
        for sign in (-1, 1):
            region = ((xx - nx - sign * dx) / 3) ** 2 + ((yy - ny - sign * dy) / 3) ** 2 <= 1
            if np.mean((core & surface)[region]) < .70:
                raise ValueError("surround_coverage")
            flanks.append(np.median(residual[region & core & surface]))
        axes.append(centre - np.mean(flanks))
    return float(min(axes))


def measure_features(depth, ir, kps5):
    """Return finite scalar descriptors; ``valid`` is NOT a PAD decision.

    Inputs must be registered uint16 depth in millimetres and uint8 IR of the
    same shape, plus SCRFD's five image-coordinate landmarks. Missing coverage
    and malformed arrays fail measurement instead of producing a live score.
    No images, landmarks, or intermediate surface arrays are returned/stored.
    """
    if (not isinstance(depth, np.ndarray) or not isinstance(ir, np.ndarray)
            or depth.ndim != 2 or ir.ndim != 2 or depth.shape != ir.shape
            or depth.dtype != np.uint16 or ir.dtype != np.uint8
            or min(depth.shape, default=0) < 32 or max(depth.shape, default=0) > 8192):
        return _empty("invalid_input")
    try:
        kps = np.asarray(kps5, dtype=np.float64)
    except (ValueError, TypeError, OverflowError):
        return _empty("invalid_landmarks")
    if kps.shape != (5, 2) or not np.isfinite(kps).all():
        return _empty("invalid_landmarks")
    h, w = depth.shape
    if np.any(kps < 0) or np.any(kps[:, 0] >= w) or np.any(kps[:, 1] >= h):
        return _empty("invalid_landmarks")
    eye = kps[1] - kps[0]
    mouth = kps[4] - kps[3]
    vertical = kps[3:5].mean(0) - kps[:2].mean(0)
    eye_width, mouth_width, height = np.linalg.norm(eye), np.linalg.norm(mouth), np.linalg.norm(vertical)
    if eye_width < 12 or mouth_width < 6 or height < 12:
        return _empty("landmarks_too_small")
    cross = eye[0] * vertical[1] - eye[1] * vertical[0]
    if (cross / (eye_width * height) < .35 or np.dot(eye, mouth) / (eye_width * mouth_width) < .25
            or not .25 <= height / eye_width <= 3 or not .15 <= mouth_width / eye_width <= 2):
        return _empty("invalid_landmarks")
    try:
        M = crop_affine(kps, SIZE, ZOOM)
        transformed = kps @ M[:, :2].T + M[:, 2]
        canonical = SIZE / 2 + (ARCFACE_DST * (SIZE / 112) - SIZE / 2) / ZOOM
        if not np.isfinite(M).all() or np.max(np.linalg.norm(transformed - canonical, axis=1)) > 12:
            return _empty("invalid_landmarks")
        dc = cv2.warpAffine(depth, M, (SIZE, SIZE), flags=cv2.INTER_NEAREST).astype(np.float64)
        ic = cv2.warpAffine(ir, M, (SIZE, SIZE), flags=cv2.INTER_NEAREST)
    except (ValueError, np.linalg.LinAlgError, cv2.error):
        return _empty("invalid_landmarks")

    yy, xx = np.mgrid[:SIZE, :SIZE]
    core = ((xx - 64) / 24) ** 2 + ((yy - 73) / 28) ** 2 <= 1
    nx, ny = transformed[2]
    nose = _nose_patch(xx, yy, core, nx, ny)
    # Exclude the nose and its immediate transition from the reference fit.
    surround = core & (((xx - nx) / 9) ** 2 + ((yy - ny) / 10) ** 2 >= 1)
    valid = dc > 0
    features = dict.fromkeys(FEATURE_NAMES)
    features["valid_ratio"] = np.mean(valid[core])
    features["ir_median"] = np.median(ic[core])
    features["ir_iqr"] = np.subtract(*np.quantile(ic[core], [.75, .25]))
    if not np.any(valid & core):
        return _finish(features, ["depth_missing"])
    median = np.median(dc[core & valid])
    features["depth_median_mm"] = median
    features["depth_iqr_mm"] = np.subtract(*np.quantile(dc[core & valid], [.75, .25]))
    # Exclude background holes/registration fringes from surface fitting while
    # retaining and checking the discarded fraction as measurement coverage.
    surface = valid & (np.abs(dc - median) <= 150)
    features["surface_ratio"] = np.mean(surface[core])
    features["nose_valid_ratio"] = np.mean(surface[nose]) if np.any(nose) else 0.0
    features["surround_valid_ratio"] = np.mean(surface[surround]) if np.any(surround) else 0.0
    reasons = []
    if features["surface_ratio"] < .65:
        reasons.append("depth_coverage")
    if features["nose_valid_ratio"] < .70:
        reasons.append("nose_coverage")
    if features["surround_valid_ratio"] < .65:
        reasons.append("surround_coverage")
    if np.mean(ic[core] > 0) < .60:
        reasons.append("ir_missing")

    samples, possible = [], 0
    for y in range(0, SIZE, 4):
        for x in range(0, SIZE, 4):
            sl = np.s_[y:y + 4, x:x + 4]
            interior = core[sl]
            if np.count_nonzero(interior) < 8:
                continue
            possible += 1
            selected = interior & surface[sl]
            if np.count_nonzero(selected) < max(8, .5 * np.count_nonzero(interior)):
                continue
            samples.append((np.median(xx[sl][selected]), np.median(yy[sl][selected]),
                            np.median(dc[sl][selected])))
    features["valid_blocks_ratio"] = len(samples) / max(possible, 1)
    if features["valid_blocks_ratio"] < .65 or len(samples) < 24:
        reasons.append("depth_blocks")
    if reasons:
        return _finish(features, reasons)

    sample = np.array(samples)
    sx, sy, z = sample.T
    x, y = (sx - 64) / 24, (sy - 73) / 28
    reference = ((sx - nx) / 9) ** 2 + ((sy - ny) / 10) ** 2 >= 1
    # A one-sided or narrow strip of depth cannot define the nose reference.
    quadrants = [(x < 0) & (y < 0), (x >= 0) & (y < 0), (x < 0) & (y >= 0), (x >= 0) & (y >= 0)]
    if any(np.count_nonzero(reference & quadrant) < 3 for quadrant in quadrants):
        return _finish(features, ["surround_coverage"])
    try:
        plane, quadratic = _design(x, y), _design(x, y, True)
        pc = _fit_inverse(plane, z, median)
        qc = _fit_inverse(quadratic, z, median)
        features["plane_residual_mm"] = _residual_rms(z - _predict(plane, pc, median))
        features["quadratic_residual_mm"] = _residual_rms(z - _predict(quadratic, qc, median))
        # Locate relief using the global robust surface first; then refit the
        # surrounding surface with the localized nose excluded. The old fixed
        # landmark ellipse could measure cheek depth after a small offset.
        residual = np.zeros_like(dc)
        cx, cy = (xx[core] - 64) / 24, (yy[core] - 73) / 28
        residual[core] = _predict(_design(cx, cy, True), qc, median) - dc[core]
        nx, ny = _localize_nose(xx, yy, core, surface, residual, nx, ny)
        nose = _nose_patch(xx, yy, core, nx, ny)
        surround = core & (((xx - nx) / 9) ** 2 + ((yy - ny) / 10) ** 2 >= 1)
        reference = ((sx - nx) / 9) ** 2 + ((sy - ny) / 10) ** 2 >= 1
        features["nose_valid_ratio"] = np.mean(surface[nose])
        features["surround_valid_ratio"] = np.mean(surface[surround])
        if (features["surround_valid_ratio"] < .65
                or any(np.count_nonzero(reference & quadrant) < 3 for quadrant in quadrants)):
            return _finish(features, ["surround_coverage"])
        # Check the measured nose plus a one-pixel rim. A face viewed at an
        # angle can have genuine self-occlusion in the wider surrounding area;
        # it must not invalidate a continuous, fully observed nose patch.
        neighbourhood = core & (((xx - nx) / 6.5) ** 2 + ((yy - ny) / 7) ** 2 <= 1)
        if _has_depth_discontinuity(dc, surface, neighbourhood):
            return _finish(features, ["nose_depth_discontinuity"])
        rc = _fit_inverse(quadratic[reference], z[reference], median)
        n = nose & surface
        ax, ay = (xx[n] - 64) / 24, (yy[n] - 73) / 28
        features["nose_plane_protrusion_mm"] = np.median(_predict(_design(ax, ay), pc, median) - dc[n])
        referenced = np.median(_predict(_design(ax, ay, True), rc, median) - dc[n])
        features["nose_protrusion_mm"] = min(referenced, _nose_prominence(
            xx, yy, core, surface, residual, nx, ny, nose))
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        if str(exc) in ("nose_coverage", "surround_coverage"):
            return _finish(features, [str(exc)])
        return _finish(features, ["geometry_degenerate"])
    return _finish(features, [])
