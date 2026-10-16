"""Pinned, explicitly unvalidated Femto depth enforcement for the local demo.

This artifact is an engineering configuration, not a calibration or evaluation
report. In particular, ``ready`` only means that the configured artifact matches
the sensor and running pipeline. It never qualifies either PAD validation gate.
The old frozen profile and its failed physical trials are left untouched.
"""

import hashlib
import json
import math
from pathlib import Path

from faceauth.common.pad_validation import canonical_json


EXPERIMENT_SCHEMA = "faceauth.femto_pad.experiment.v1"
EXPERIMENT_MODE = "femto_geometry_experimental"
FEATURE_SCHEMA = "faceauth.femto_pad.features.v2"
BINDING_SCHEMA = "faceauth.femto_pad.binding.v1"
FEATURE_NAMES = (
    "valid_ratio", "surface_ratio", "valid_blocks_ratio", "depth_median_mm", "depth_iqr_mm",
    "plane_residual_mm", "quadratic_residual_mm", "nose_plane_protrusion_mm", "nose_protrusion_mm",
    "nose_valid_ratio", "surround_valid_ratio", "ir_median", "ir_iqr",
)
MEASUREMENT_REASONS = frozenset((
    "invalid_input", "invalid_landmarks", "landmarks_too_small", "depth_missing",
    "depth_coverage", "nose_coverage", "surround_coverage", "ir_missing", "depth_blocks",
    "geometry_degenerate", "not_measured", "measurement_error", "nose_depth_discontinuity",
))
SOURCE_PROFILE_SHA256 = "907454b8797ec94c9688b6252f271d851ef4e336ef7a4c6f83b9154a858f770d"
# Preserve the old v1 initialization as provenance, never as v2 validation.
INITIAL_THRESHOLDS = {
    "plane_residual_mm": 0.3536019972420733,
    "nose_protrusion_mm": 4.563589938099597,
}
# Operator-requested, modest relaxation for authentication. These are engineering
# settings, not a fit to historical test rows or a physically evaluated v2 profile.
# Complete depth/IR measurements and BOTH minima are still required per frame.
ENGINEERING_THRESHOLDS = {"plane_residual_mm": 0.30, "nose_protrusion_mm": 3.0}
THRESHOLD_PROVENANCE = {
    "kind": "operator_requested_engineering_relaxation",
    "source_profile_sha256": SOURCE_PROFILE_SHA256,
    "source_feature_schema": "faceauth.femto_pad.features.v1",
    "previous_thresholds": INITIAL_THRESHOLDS,
    "transferred_without_recalibration": False,
    "recalibrated": False,
    "revision": "auth-relaxed-20261016",
}


def _is_digest(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _clean_binding(value):
    keys = {"schema", "feature_schema", "feature_code_sha256", "detector_model_json_sha256", "pipeline_sha256"}
    if (not isinstance(value, dict) or set(value) != keys
            or value.get("schema") != BINDING_SCHEMA or value.get("feature_schema") != FEATURE_SCHEMA
            or any(not _is_digest(value[key]) for key in keys - {"schema", "feature_schema"})):
        raise ValueError("experimental PAD requires a complete v2 pipeline binding")
    return dict(value)


def build_experiment(*, binding, calib_version):
    """Build a new configuration without collecting data or asserting a test pass.

    The caller writes canonical JSON exclusively and pins the resulting file
    SHA-256 in the API and engines. This function performs no filesystem writes.
    """
    profile = {
        "schema": EXPERIMENT_SCHEMA,
        "mode": EXPERIMENT_MODE,
        "experimental": True,
        "production_validated": False,
        "empirical_validation": "not_evaluated",
        "scope": ["print", "replay"],
        "feature_schema": FEATURE_SCHEMA,
        "algorithm": "all_frames_conjunctive_minimums",
        "candidate_binding": _clean_binding(binding),
        "calib_version": calib_version,
        "thresholds": dict(ENGINEERING_THRESHOLDS),
        "threshold_provenance": {**THRESHOLD_PROVENANCE, "previous_thresholds": dict(INITIAL_THRESHOLDS)},
    }
    _validate_experiment(profile)
    return profile


def _validate_experiment(profile):
    keys = {"schema", "mode", "experimental", "production_validated", "empirical_validation", "scope",
            "feature_schema", "algorithm", "candidate_binding", "calib_version", "thresholds",
            "threshold_provenance"}
    if (not isinstance(profile, dict) or set(profile) != keys
            or profile.get("schema") != EXPERIMENT_SCHEMA or profile.get("mode") != EXPERIMENT_MODE
            or profile.get("experimental") is not True or profile.get("production_validated") is not False
            or profile.get("empirical_validation") != "not_evaluated"
            or profile.get("scope") != ["print", "replay"] or profile.get("feature_schema") != FEATURE_SCHEMA
            or profile.get("algorithm") != "all_frames_conjunctive_minimums"
            or not isinstance(profile.get("calib_version"), str) or not profile["calib_version"].strip()
            or profile.get("threshold_provenance") != THRESHOLD_PROVENANCE):
        raise ValueError("invalid unvalidated experimental PAD configuration")
    _clean_binding(profile.get("candidate_binding"))
    thresholds = profile.get("thresholds")
    if (not isinstance(thresholds, dict) or set(thresholds) != set(ENGINEERING_THRESHOLDS)
            or any(type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 100
                   for value in thresholds.values())
            or thresholds != ENGINEERING_THRESHOLDS):
        raise ValueError("experimental engineering thresholds must match the pinned engineering revision")
    canonical_json(profile)


def _clean_measurement(value):
    if (not isinstance(value, dict) or value.get("schema") != FEATURE_SCHEMA
            or type(value.get("valid")) is not bool or not isinstance(value.get("features"), dict)):
        raise ValueError("experimental PAD requires a v2 measurement")
    reasons = value.get("reasons")
    if (not isinstance(reasons, list)
            or any(not isinstance(reason, str) or reason not in MEASUREMENT_REASONS for reason in reasons)
            or bool(reasons) == value["valid"]):
        raise ValueError("invalid experimental PAD measurement reasons")
    features = {}
    for name in FEATURE_NAMES:
        item = value["features"].get(name)
        if item is not None and (type(item) not in (int, float) or not math.isfinite(item)):
            raise ValueError("experimental PAD features must be finite numeric scalars")
        features[name] = float(item) if item is not None else None
    if value["valid"]:
        if any(item is None for item in features.values()) or features["depth_median_mm"] <= 0:
            raise ValueError("valid experimental PAD measurement requires complete depth features")
        for name in ("valid_ratio", "surface_ratio", "valid_blocks_ratio", "nose_valid_ratio", "surround_valid_ratio"):
            if not 0 <= features[name] <= 1:
                raise ValueError("experimental PAD coverage must be bounded")
    return {"valid": value["valid"], "reasons": list(dict.fromkeys(reasons)), "features": features}


class FemtoPADExperiment:
    """Fail-closed experimental classifier with pure, request-bound readiness.

    Callers must check ``status`` for each request before ``decide`` and require
    all selected frames to pass. A failed probe never changes another request's
    status. No method grants production or empirical validation.
    """

    def __init__(self, path, expected_sha256):
        self.profile, self.profile_sha256, self.reasons = None, None, []
        try:
            if not _is_digest(expected_sha256):
                raise ValueError("experimental profile SHA-256 pin required")
            raw = Path(path).read_bytes()
            self.profile_sha256 = hashlib.sha256(raw).hexdigest()
            if self.profile_sha256 != expected_sha256:
                self.reasons = ["profile_hash_mismatch"]
                return
            profile = json.loads(raw)
            _validate_experiment(profile)
            self.profile = profile
        except (OSError, ValueError, TypeError, KeyError, OverflowError):
            self.reasons = ["profile_missing_or_invalid"]

    def status(self, calib_version, binding):
        reasons = list(self.reasons)
        if self.profile is not None:
            if self.profile["calib_version"] != calib_version:
                reasons.append("profile_calibration_mismatch")
            try:
                matches = _clean_binding(binding) == self.profile["candidate_binding"]
            except (ValueError, TypeError, KeyError):
                matches = False
            if not matches:
                reasons.append("profile_binding_mismatch")
        return {
            "ready": self.profile is not None and not reasons,
            "reasons": reasons,
            "profile_sha256": self.profile_sha256,
            "calib_version": self.profile["calib_version"] if self.profile else None,
            "scope": ["print", "replay"],
            "experimental": True,
            "production_validated": False,
            "empirical_validation": "not_evaluated",
        }

    def decide(self, measurement):
        if self.profile is None or self.reasons:
            return {"score": 0.0, "live": False, "nonresponse": True, "failed": ["profile_not_ready"]}
        try:
            value = _clean_measurement(measurement)
            if not value["valid"]:
                return {"score": 0.0, "live": False, "nonresponse": True, "failed": value["reasons"]}
            failed = [name for name, threshold in self.profile["thresholds"].items()
                      if value["features"][name] < threshold]
            return {"score": 0.0 if failed else 1.0, "live": not failed, "nonresponse": False, "failed": failed}
        except (ValueError, TypeError, KeyError, OverflowError):
            return {"score": 0.0, "live": False, "nonresponse": True, "failed": ["measurement_invalid"]}
