"""Calibration and frozen local-pilot evaluation of numerical Femto PAD features.

Three presentations per class are an engineering smoke-test minimum, not a PAD
standard, statistical performance target, or independent security assessment.
Profiles never qualify the production PADValidationGate. No images are read or
stored, and adjacent frames are never counted as independent presentations.
"""

import hashlib
import json
import math
from pathlib import Path

from faceauth.common.pad_validation import (
    CANDIDATE_SCHEMAS, OBSERVATION_SCHEMA, binomial_upper95, canonical_json,
    clean_candidate_binding, clean_candidate_measurement,
)

PROFILE_SCHEMA = "faceauth.femto_pad.profile.v1"
EVALUATION_SCHEMA = "faceauth.femto_pad.pilot_evaluation.v1"
FEATURES = ("plane_residual_mm", "nose_protrusion_mm")
KINDS = ("live", "print", "replay")
MIN_PRESENTATIONS = 3


def _digest(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _identifier(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("nonempty pseudonymous presentation/subject/instrument IDs required")
    return hashlib.sha256(value.encode()).hexdigest()


def _hash_list(value):
    return (isinstance(value, list)
            and all(isinstance(x, str) and len(x) == 64 and all(c in "0123456789abcdef" for c in x)
                    for x in value) and len(value) == len(set(value)))


def profile_payload_sha256(profile):
    """Bind evaluation to the entire frozen profile, excluding evaluation itself."""
    return _digest({key: value for key, value in profile.items() if key != "evaluation"})


def _rows(rows, split, binding, calib_version):
    selected, identifiers = [], set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("observation must be a mapping")
        if row.get("split") != split:
            continue
        if row.get("kind") not in KINDS:
            continue  # Masks are explicitly outside this experimental profile.
        if (row.get("schema") != OBSERVATION_SCHEMA or row.get("source") != "real_sensor"
                or row.get("consent") is not True or row.get("calib_version") != calib_version
                or clean_candidate_binding(row.get("candidate_binding")) != binding):
            raise ValueError("only consented real-sensor candidate observations with matching provenance are usable")
        _identifier(row.get("consent_id"))
        pid = _identifier(row.get("presentation_id"))
        if pid in identifiers:
            raise ValueError("duplicate presentation ID; frames are not independent trials")
        identifiers.add(pid)
        frames = row.get("frames")
        if not isinstance(frames, list) or not frames:
            raise ValueError("presentation requires measured frames")
        measurements = []
        for frame in frames:
            if not isinstance(frame, dict) or "pad_candidate" not in frame:
                raise ValueError("old CNN scores cannot calibrate the geometry candidate")
            measurement = clean_candidate_measurement(frame["pad_candidate"])
            if measurement["schema"] != binding["feature_schema"]:
                raise ValueError("measurement schema does not match its binding")
            measurements.append(measurement)
        conditions = row.get("conditions", {})
        if not isinstance(conditions, dict):
            raise ValueError("invalid conditions")
        mode = conditions.get("mode", "unspecified")
        if mode not in ("auth", "enroll", "diagnostic"):
            mode = "unspecified"
        selected.append({"id": pid, "subject": _identifier(row.get("subject_id")),
                         "instrument": None if row["kind"] == "live" else _identifier(row.get("instrument_id")),
                         "kind": row["kind"], "mode": mode, "measurements": measurements})
    if any(sum(r["kind"] == kind for r in selected) < MIN_PRESENTATIONS for kind in KINDS):
        raise ValueError("at least three physical presentations each of live, print, and replay are required")
    return selected


def _decision(measurement, thresholds):
    value = clean_candidate_measurement(measurement)
    if not value["valid"]:
        return {"score": 0.0, "live": False, "nonresponse": True, "failed": value["reasons"]}
    failed = [key for key in FEATURES if value["features"][key] < thresholds[key]]
    return {"score": 0.0 if failed else 1.0, "live": not failed, "nonresponse": False, "failed": failed}


def _presentation(row, thresholds):
    decisions = [_decision(m, thresholds) for m in row["measurements"]]
    return all(d["live"] for d in decisions), any(d["nonresponse"] for d in decisions)


def _summary(rows, thresholds):
    result = {}
    for kind in KINDS:
        presentations = [row for row in rows if row["kind"] == kind]
        outcomes = [_presentation(row, thresholds) for row in presentations]
        accepted = sum(live for live, _ in outcomes)
        nonresponses = sum(nr for _, nr in outcomes)
        responding = len(presentations) - nonresponses
        errors = responding - accepted if kind == "live" else accepted
        result[kind] = {"presentations": len(presentations), "accepted": accepted,
                        "nonresponses": nonresponses, "responding": responding,
                        "error_metric": "bpcer" if kind == "live" else "apcer", "errors": errors,
                        "error_rate": errors / responding if responding else None,
                        "error_upper95": binomial_upper95(errors, responding),
                        "nonresponse_upper95": binomial_upper95(nonresponses, len(presentations))}
    return result


def _provenance(rows):
    return {"presentation_hashes": sorted(row["id"] for row in rows),
            "subject_hashes": sorted({row["subject"] for row in rows}),
            "instrument_hashes": sorted({row["instrument"] for row in rows if row["instrument"]}),
            "mode_counts": {mode: sum(row["mode"] == mode for row in rows)
                            for mode in sorted({row["mode"] for row in rows})},
            "observations_sha256": _digest(rows)}


def fit_profile(rows, *, binding, calib_version):
    """Fit only calibration rows; refuse inseparable measurements, never relax."""
    binding = clean_candidate_binding(binding)
    if not isinstance(calib_version, str) or not calib_version.strip():
        raise ValueError("sensor calibration version required")
    data = _rows(rows, "calibration", binding, calib_version)
    live = [row for row in data if row["kind"] == "live"]
    if any(not m["valid"] for row in live for m in row["measurements"]):
        raise ValueError("all calibration live presentations require valid spatial measurements")
    limits = {key: min(m["features"][key] for row in live for m in row["measurements"]) for key in FEATURES}
    if any(not math.isfinite(value) or value <= 0 for value in limits.values()):
        raise ValueError("live geometry does not support positive thresholds; collect new measurements")
    attacks = [row for row in data if row["kind"] != "live" and all(m["valid"] for m in row["measurements"])]
    if any(not any(row["kind"] == kind for row in attacks) for kind in ("print", "replay")):
        raise ValueError("each calibration attack type requires a responding physical presentation")
    minima = [{key: min(m["features"][key] for m in row["measurements"]) for key in FEATURES} for row in attacks]
    candidates = {}
    for key in FEATURES:
        upper = limits[key]
        boundaries = sorted({0.0, upper, *(max(0.0, min(upper, row[key])) for row in minima)})
        points = [(a + b) / 2 for a, b in zip(boundaries, boundaries[1:]) if a < b]
        # Keep work bounded for large study files, retaining the strictest end.
        if len(points) > 128:
            points = [points[round(i * (len(points) - 1) / 127)] for i in range(128)]
        candidates[key] = points
    best, best_margin = None, -1.0
    for first in candidates[FEATURES[0]]:
        for second in candidates[FEATURES[1]]:
            thresholds = dict(zip(FEATURES, (first, second)))
            live_margin = min((limits[key] - thresholds[key]) / limits[key] for key in FEATURES)
            attack_margin = min(max((thresholds[key] - row[key]) / limits[key] for key in FEATURES)
                                for row in minima)
            margin = min(live_margin, attack_margin)
            if margin > best_margin:
                best, best_margin = thresholds, margin
    if best is None or best_margin <= 0:
        raise ValueError("calibration live and attack geometry overlap; this profile cannot separate them")
    return {"schema": PROFILE_SCHEMA, "experimental": True, "scope": ["print", "replay"],
            "feature_schema": binding["feature_schema"], "algorithm": "all_frames_conjunctive_minimums",
            "candidate_binding": binding, "calib_version": calib_version, "thresholds": best,
            "calibration": {**_provenance(data), "summary": _summary(data, best),
                            "minimum_presentations_per_kind": MIN_PRESENTATIONS,
                            "normalized_separation_margin": best_margin},
            "production_validated": False}


def _validate_base(profile):
    if (not isinstance(profile, dict) or profile.get("schema") != PROFILE_SCHEMA
            or profile.get("experimental") is not True or profile.get("production_validated") is not False
            or profile.get("scope") != ["print", "replay"] or profile.get("feature_schema") not in CANDIDATE_SCHEMAS
            or profile.get("algorithm") != "all_frames_conjunctive_minimums"
            or not isinstance(profile.get("calib_version"), str) or not profile["calib_version"].strip()):
        raise ValueError("invalid experimental geometry profile")
    binding = clean_candidate_binding(profile.get("candidate_binding"))
    if profile["feature_schema"] != binding["feature_schema"]:
        raise ValueError("profile feature schema does not match its binding")
    thresholds = profile.get("thresholds")
    if (not isinstance(thresholds, dict) or set(thresholds) != set(FEATURES)
            or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                   for value in thresholds.values())):
        raise ValueError("positive finite geometry thresholds required")
    calibration = profile.get("calibration")
    if not isinstance(calibration, dict) or any(not _hash_list(calibration.get(key))
            for key in ("presentation_hashes", "subject_hashes", "instrument_hashes")):
        raise ValueError("invalid calibration provenance")
    if len(calibration["presentation_hashes"]) < MIN_PRESENTATIONS * len(KINDS):
        raise ValueError("insufficient calibration presentations")
    margin = calibration.get("normalized_separation_margin")
    if type(margin) not in (int, float) or not math.isfinite(margin) or margin <= 0:
        raise ValueError("positive calibration separation margin required")
    total = 0
    for kind in KINDS:
        summary = calibration.get("summary", {}).get(kind, {})
        count, accepted, nr, responding, errors = [summary.get(key) for key in
                                                  ("presentations", "accepted", "nonresponses", "responding", "errors")]
        if (any(type(value) is not int or value < 0 for value in (count, accepted, nr, responding, errors))
                or count < MIN_PRESENTATIONS or nr + responding != count or accepted > responding
                or errors != (responding - accepted if kind == "live" else accepted)
                or (kind == "live" and accepted != count)
                or (kind != "live" and (accepted != 0 or responding < 1))):
            raise ValueError("invalid calibration presentation counts")
        total += count
    if total != len(calibration["presentation_hashes"]):
        raise ValueError("calibration presentation count mismatch")
    canonical_json(profile)


def _evaluation_reasons(profile):
    evaluation = profile.get("evaluation")
    if not isinstance(evaluation, dict):
        return ["pilot_evaluation_missing"]
    if (evaluation.get("schema") != EVALUATION_SCHEMA
            or evaluation.get("frozen_profile_sha256") != profile_payload_sha256(profile)
            or evaluation.get("independent_assessment") is not False
            or evaluation.get("split") != "test"):
        return ["pilot_evaluation_binding_mismatch"]
    ids = evaluation.get("presentation_hashes")
    if (not _hash_list(ids) or set(ids) & set(profile["calibration"]["presentation_hashes"])):
        return ["pilot_test_presentation_overlap"]
    reasons = []
    total = 0
    try:
        for kind in KINDS:
            summary = evaluation["summary"][kind]
            values = [summary[key] for key in ("presentations", "accepted", "nonresponses", "responding", "errors")]
            if any(type(value) is not int or value < 0 for value in values):
                raise ValueError("invalid counts")
            count, accepted, nr, responding, errors = values
            if responding + nr != count or accepted > responding or errors != (responding - accepted if kind == "live" else accepted):
                raise ValueError("inconsistent counts")
            total += count
            if count < MIN_PRESENTATIONS:
                reasons.append("pilot_insufficient_" + kind)
            if kind == "live" and accepted != count:
                reasons.append("pilot_live_not_all_accepted")
            if kind != "live" and (accepted > 0 or responding == 0):
                reasons.append("pilot_" + kind + "_not_verified")
            per_mode = evaluation["by_mode"]
            columns = ("presentations", "accepted", "nonresponses", "responding", "errors")
            for mode in ("auth", "enroll", "diagnostic", "unspecified"):
                counts = per_mode[mode][kind]
                if any(type(counts[key]) is not int or counts[key] < 0 for key in columns):
                    raise ValueError("invalid mode counts")
                if (counts["responding"] + counts["nonresponses"] != counts["presentations"]
                        or counts["accepted"] > counts["responding"]
                        or counts["errors"] != (counts["responding"] - counts["accepted"]
                                                if kind == "live" else counts["accepted"])):
                    raise ValueError("inconsistent mode counts")
            if any(sum(per_mode[mode][kind][key] for mode in per_mode) != summary[key] for key in columns):
                raise ValueError("mode totals differ from presentation totals")
            for mode in (("auth", "enroll") if kind == "live" else ("auth",)):
                counts = per_mode[mode][kind]
                if kind == "live" and counts["accepted"] < 1:
                    reasons.append(f"pilot_live_{mode}_missing")
                if kind != "live" and (counts["responding"] < 1 or counts["accepted"] > 0):
                    reasons.append(f"pilot_{kind}_{mode}_not_verified")
        if total != len(ids):
            raise ValueError("presentation count mismatch")
    except (KeyError, TypeError, ValueError):
        reasons.append("pilot_evaluation_invalid")
    return reasons


def evaluate_profile(profile, rows):
    """Evaluate a frozen profile on test presentations; never adjust thresholds."""
    _validate_base(profile)
    result = json.loads(canonical_json(profile))
    result.pop("evaluation", None)
    data = _rows(rows, "test", result["candidate_binding"], result["calib_version"])
    provenance = _provenance(data)
    calibration = result["calibration"]
    if set(provenance["presentation_hashes"]) & set(calibration["presentation_hashes"]):
        raise ValueError("test presentation IDs overlap calibration")
    result["evaluation"] = {"schema": EVALUATION_SCHEMA, "split": "test",
                            "frozen_profile_sha256": profile_payload_sha256(result), **provenance,
                            "summary": _summary(data, result["thresholds"]),
                            "by_mode": {mode: _summary([row for row in data if row["mode"] == mode], result["thresholds"])
                                        for mode in ("auth", "enroll", "diagnostic", "unspecified")},
                            "subject_overlap": bool(set(provenance["subject_hashes"]) & set(calibration["subject_hashes"])),
                            "instrument_overlap": bool(set(provenance["instrument_hashes"]) & set(calibration["instrument_hashes"])),
                            "independent_assessment": False, "production_validated": False,
                            "tested_modes": sorted(provenance["mode_counts"])}
    reasons = _evaluation_reasons(result)
    result["evaluation"].update(eligible_for_local_pilot=not reasons, reasons=reasons)
    return result


class FemtoPADProfile:
    """Pinned pilot profile; callers must check request provenance with status.

    Status is pure: a readiness probe for another camera must not change the
    result of a concurrent valid capture. The engine checks status against each
    request's calibration and current binding before calling decide.
    """

    def __init__(self, path, expected_sha256):
        self.profile, self.profile_sha256, self.reasons = None, None, []
        try:
            if not _hash_list([expected_sha256]):
                raise ValueError("profile SHA-256 pin required")
            raw = Path(path).read_bytes()
            self.profile_sha256 = hashlib.sha256(raw).hexdigest()
            if self.profile_sha256 != expected_sha256:
                self.reasons = ["profile_hash_mismatch"]
                return
            profile = json.loads(raw)
            _validate_base(profile)
            self.profile = profile
            self.reasons = _evaluation_reasons(profile)
        except (OSError, ValueError, TypeError, KeyError):
            self.reasons = ["profile_missing_or_invalid"]

    def status(self, calib_version, binding):
        reasons = list(self.reasons)
        if self.profile is not None:
            if self.profile["calib_version"] != calib_version:
                reasons.append("profile_calibration_mismatch")
            if self.profile["candidate_binding"] != binding:
                reasons.append("profile_binding_mismatch")
        return {"ready": self.profile is not None and not reasons, "reasons": reasons, "profile_sha256": self.profile_sha256,
                "calib_version": self.profile["calib_version"] if self.profile else None,
                "scope": ["print", "replay"], "experimental": True, "production_validated": False}

    def decide(self, measurement):
        if self.profile is None or self.reasons:
            return {"score": 0.0, "live": False, "nonresponse": True, "failed": ["profile_not_ready"]}
        try:
            if not isinstance(measurement, dict) or measurement.get("schema") != self.profile["feature_schema"]:
                raise ValueError("measurement feature schema does not match profile")
            return _decision(measurement, self.profile["thresholds"])
        except (ValueError, TypeError, KeyError):
            return {"score": 0.0, "live": False, "nonresponse": True, "failed": ["measurement_invalid"]}
