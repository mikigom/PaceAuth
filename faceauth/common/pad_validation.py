"""Real-sensor PAD evidence, separate from inference and from capture consent.

RFP_KO §4 requires independent physical-presentation testing; it does not fix
acceptance limits. Consequently there are deliberately NO default passing limits.
An operator-reviewed policy and independent test reference must accompany the
signed evidence. Neither matching a calibration name nor signing empty evidence
can qualify a model. This module never opens a camera or writes captured images.
"""
import hashlib
import json
import math
from pathlib import Path

SCHEMA = "faceauth.pad.evidence.v1"
OBSERVATION_SCHEMA = "faceauth.pad.observation.v1"
SPECIES = ("print", "replay", "mask_rigid", "mask_silicone")
SPLITS = ("train", "calibration", "test")
SOURCE_FILES = ("models/pad.py", "models/scrfd.py", "common/align.py",
                "common/packet.py", "terminal/orbbec.py", "engine/main.py", "api/policy.py",
                "terminal/live.py", "terminal/web_auth.py", "terminal/pipeline.py", "terminal/guide.py")
LEGACY_CANDIDATE_SCHEMA = "faceauth.femto_pad.features.v1"
CANDIDATE_SCHEMA = "faceauth.femto_pad.features.v2"
CANDIDATE_SCHEMAS = (LEGACY_CANDIDATE_SCHEMA, CANDIDATE_SCHEMA)
CANDIDATE_BINDING_SCHEMA = "faceauth.femto_pad.binding.v1"
CANDIDATE_FEATURES = (
    "valid_ratio", "surface_ratio", "valid_blocks_ratio", "depth_median_mm", "depth_iqr_mm",
    "plane_residual_mm", "quadratic_residual_mm", "nose_plane_protrusion_mm", "nose_protrusion_mm",
    "nose_valid_ratio", "surround_valid_ratio", "ir_median", "ir_iqr",
)
CANDIDATE_REASONS = frozenset(("invalid_input", "invalid_landmarks", "landmarks_too_small", "depth_missing",
    "depth_coverage", "nose_coverage", "surround_coverage", "ir_missing", "depth_blocks",
    "geometry_degenerate", "nose_depth_discontinuity", "not_measured", "measurement_error"))
# These descriptors use PAD's alignment helpers, but never its CNN or weights.
# Capture/selection/storage changes invalidate collection provenance as well.
CANDIDATE_SOURCE_FILES = (*SOURCE_FILES, "models/femto_pad.py", "common/pad_validation.py",
                          "common/femto_pad_profile.py", "common/femto_pad_experiment.py", "api/app.py", "api/clients.py",
                          "terminal/client.py")


def _hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def candidate_binding(*, detector_dir):
    """Fingerprint the feature-only candidate independently of the old PAD CNN.

    The engine supplies the detector from its verified signed registry. This is
    collection provenance, not a release signature or permission to authenticate.
    Calibration remains a separate required observation field.
    """
    root = Path(__file__).resolve().parents[1]
    sources = {name: _hash(root / name) for name in CANDIDATE_SOURCE_FILES}
    return {"schema": CANDIDATE_BINDING_SCHEMA, "feature_schema": CANDIDATE_SCHEMA,
            "feature_code_sha256": sources["models/femto_pad.py"],
            "detector_model_json_sha256": _hash(Path(detector_dir) / "MODEL.json"),
            "pipeline_sha256": hashlib.sha256(canonical_json(sources)).hexdigest()}


def clean_candidate_binding(value):
    """Keep only versioned provenance and SHA-256 digests, never freeform data."""
    if (not isinstance(value, dict) or value.get("schema") != CANDIDATE_BINDING_SCHEMA
            or value.get("feature_schema") not in CANDIDATE_SCHEMAS):
        raise ValueError("invalid PAD candidate binding")
    clean = {"schema": CANDIDATE_BINDING_SCHEMA, "feature_schema": value["feature_schema"]}
    for key in ("feature_code_sha256", "detector_model_json_sha256", "pipeline_sha256"):
        digest = value.get(key)
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid PAD candidate digest")
        clean[key] = digest
    return clean


def clean_candidate_measurement(value):
    """Whitelist feature scalars; valid describes measurement, never liveness."""
    if value is None:
        value = {"schema": CANDIDATE_SCHEMA, "valid": False, "reasons": ["not_measured"], "features": {}}
    if (not isinstance(value, dict) or value.get("schema") not in CANDIDATE_SCHEMAS
            or type(value.get("valid")) is not bool or not isinstance(value.get("features"), dict)):
        raise ValueError("invalid PAD candidate measurement")
    reasons = value.get("reasons")
    if (not isinstance(reasons, list) or any(not isinstance(r, str) or r not in CANDIDATE_REASONS for r in reasons)
            or bool(reasons) == value["valid"]):
        raise ValueError("invalid PAD candidate reasons")
    features = {}
    for name in CANDIDATE_FEATURES:
        item = value["features"].get(name)
        if item is not None and type(item) not in (int, float):
            raise ValueError("PAD candidate features must be numeric scalars")
        features[name] = float(item) if item is not None and math.isfinite(item) else None
    if value["valid"] and any(item is None for item in features.values()):
        raise ValueError("valid PAD candidate measurement requires every finite feature")
    return {"schema": value["schema"], "valid": value["valid"],
            "reasons": list(dict.fromkeys(reasons)), "features": features}


def model_binding(model_dir, *, detector_dir=None):
    """Fingerprint actual weights, all PAD settings, and the capture/inference code.

    Signed-registry verification remains the engine loader's responsibility.
    This binding prevents an old evaluation surviving changed weights/settings.
    """
    directory = Path(model_dir)
    meta = json.loads((directory / "MODEL.json").read_text())
    if meta.get("kind") != "pad":
        raise ValueError("not a PAD model")
    weight = Path(meta["files"]["weights"])
    if weight.is_absolute() or ".." in weight.parts or not weight.parts:
        raise ValueError("invalid PAD weight path")
    actual = _hash(directory / weight)
    if actual != meta["sha256"]["weights"]:
        raise ValueError("PAD weight hash mismatch")
    root = Path(__file__).resolve().parents[1]
    sources = {name: _hash(root / name) for name in SOURCE_FILES}
    # The engine uses SCRFD landmarks, whereas the old synthetic evaluation used
    # ground-truth landmarks. Bind the actual detector artifact when installed.
    if detector_dir is None:
        registry = directory.parent / "registry.json"
        if registry.exists():
            entries = [e for e in json.loads(registry.read_text())["models"] if e["name"] == "scrfd_34g"]
            if len(entries) != 1:
                raise ValueError("select the active detector_dir explicitly")
            relative = Path(entries[0]["dir"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("invalid detector model path")
            detector_dir = directory.parent / relative
    detector = Path(detector_dir) / "MODEL.json" if detector_dir is not None else None
    return {"model_version": f'{meta["name"]}@{meta["version"]}',
            "model_json_sha256": _hash(directory / "MODEL.json"), "weights_sha256": actual,
            "pipeline_sha256": hashlib.sha256(canonical_json(sources)).hexdigest(),
            "detector_model_json_sha256": _hash(detector) if detector is not None else None}


def make_observation(*, presentation_id, subject_id, consent_id, consent, kind, split,
                     instrument_id, calib_version, binding, frames, conditions, candidate_binding=None):
    """Whitelist a consented attempt into a score-only record suitable for JSONL.

    IDs must be opaque study pseudonyms, not account names. The caller obtains
    explicit study consent, authenticates the experiment operator, and selects
    frames using the actual enrollment/auth flow. No embeddings, landmarks,
    images, tokens or account assertions are retained. A row is ONE presentation,
    not an independent trial for every adjacent video frame.
    """
    if consent is not True or not all(isinstance(v, str) and v.strip() for v in
                                     (presentation_id, subject_id, consent_id, calib_version)):
        raise ValueError("explicit study consent and opaque IDs are required")
    if kind not in ("live", *SPECIES) or split not in SPLITS:
        raise ValueError("unknown physical presentation kind or split")
    if kind != "live" and not (isinstance(instrument_id, str) and instrument_id.strip()):
        raise ValueError("physical attack instrument ID required")
    if not isinstance(frames, list) or not frames:
        raise ValueError("a presentation requires measured frames")
    if not isinstance(conditions, dict):
        raise ValueError("conditions must be a mapping")
    candidate = clean_candidate_binding(candidate_binding) if candidate_binding is not None else None
    cleaned = []
    for frame in frames:
        if "pad_candidate" in frame and candidate is None:
            raise ValueError("PAD candidate measurement requires its binding")
        pad = frame.get("pad")
        if pad is None:
            decision = {"live": False, "nonresponse": True, "score": None}
        else:
            if not isinstance(pad, dict) or type(pad.get("live")) is not bool \
                    or type(pad.get("nonresponse")) is not bool:
                raise ValueError("invalid PAD decision")
            score = pad.get("score")
            missing_score = score is None and pad["nonresponse"] and not pad["live"]
            if not missing_score and (not isinstance(score, (int, float)) or isinstance(score, bool) \
                    or not math.isfinite(score) or not 0 <= score <= 1):
                raise ValueError("PAD score must be finite and between zero and one")
            if pad["live"] and pad["nonresponse"]:
                raise ValueError("a non-response cannot be live")
            decision = {"live": pad["live"], "nonresponse": pad["nonresponse"],
                        "score": None if missing_score else float(score)}
            for key in ("cnn", "checks", "failed"):
                if key in pad:
                    decision[key] = _finite_measurements(pad[key])
        item = {"pad": decision, "ok": frame.get("ok") is True,
                "reasons": [str(r) for r in frame.get("reasons", [])],
                "quality": _finite_measurements(frame.get("quality", {}))}
        if candidate is not None:
            measurement = frame.get("pad_candidate")
            if measurement is None:
                measurement = {"schema": candidate["feature_schema"], "valid": False,
                               "reasons": ["not_measured"], "features": {}}
            item["pad_candidate"] = clean_candidate_measurement(measurement)
            if item["pad_candidate"]["schema"] != candidate["feature_schema"]:
                raise ValueError("PAD candidate feature schema does not match its binding")
        cleaned.append(item)
    # Deliberately whitelist conditions so a complete frame/account object cannot
    # accidentally be persisted through this metadata field.
    allowed = ("lighting", "distance_mm", "pose", "eyewear", "location_id", "mode", "pai_subtype", "capture_outcome")
    selected = {key: _finite_measurements(conditions[key]) for key in allowed if key in conditions}
    row = {"schema": OBSERVATION_SCHEMA, "source": "real_sensor", "presentation_id": presentation_id,
           "subject_id": subject_id, "consent_id": consent_id, "consent": True, "kind": kind,
           "split": split, "instrument_id": instrument_id if kind != "live" else None,
           "calib_version": calib_version, "binding": dict(binding), "frames": cleaned,
           "conditions": selected}
    if candidate is not None:
        row["candidate_binding"] = candidate
    canonical_json(row)  # reject non-JSON metadata; no lossy default=str on arrays
    return row


def _finite_measurements(value):
    if isinstance(value, dict):
        return {str(k): _finite_measurements(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_measurements(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError("measurement is not a JSON scalar")


def binomial_upper95(errors, trials):
    """Exact one-sided 95% Clopper-Pearson bound, standard-library only."""
    if type(errors) is not int or type(trials) is not int or not 0 <= errors <= trials:
        raise ValueError("invalid binomial counts")
    if not trials:
        return None
    if errors == trials:
        return 1.0
    if errors == 0:
        return -math.expm1(math.log(0.05) / trials)
    coeff = [math.lgamma(trials + 1) - math.lgamma(i + 1) - math.lgamma(trials - i + 1)
             for i in range(errors + 1)]
    lo, hi = errors / trials, 1.0
    for _ in range(55):
        p = (lo + hi) / 2
        terms = [c + i * math.log(p) + (trials - i) * math.log1p(-p) for i, c in enumerate(coeff)]
        largest = max(terms)
        log_cdf = largest + math.log(sum(math.exp(v - largest) for v in terms))
        if log_cdf > math.log(0.05):
            lo = p
        else:
            hi = p
    return (lo + hi) / 2


def _rate(errors, trials):
    return {"errors": errors, "trials": trials, "value": errors / trials if trials else None,
            "upper95": binomial_upper95(errors, trials)}


def _diagnostics(rows):
    """Frame distributions explain domain shift; frame count is NOT trial count."""
    values, failed = {}, {}
    frame_count = 0
    for row in rows:
        if row["conditions"].get("capture_outcome") in ("selection_timeout", "session_expired"):
            continue  # Explicit unmeasured nonresponse markers are not physical frames.
        for frame in row["frames"]:
            frame_count += 1
            pad = frame["pad"]
            measurements = {key: pad.get(key) for key in ("score", "cnn")}
            measurements.update(pad.get("checks", {}))
            for name, value in measurements.items():
                if type(value) in (int, float) and math.isfinite(value):
                    values.setdefault(name, []).append(float(value))
            for name in pad.get("failed", []):
                failed[name] = failed.get(name, 0) + 1
    quantiles = {}
    for name, numbers in values.items():
        numbers.sort()
        def percentile(fraction):
            position = fraction * (len(numbers) - 1)
            lower = int(position)
            upper = min(lower + 1, len(numbers) - 1)
            return numbers[lower] + (position - lower) * (numbers[upper] - numbers[lower])
        quantiles[name] = {"n_frames": len(numbers), "p05": percentile(.05),
                           "median": percentile(.5), "p95": percentile(.95)}
    return {"frame_count": frame_count, "quantiles": quantiles, "failed_check_frames": failed}


def summarize(rows):
    """Session/attempt metrics; non-responses are explicit, not silently removed."""
    result = {}
    for kind in ("live", *SPECIES):
        selected = [r for r in rows if r["kind"] == kind]
        nr = sum(any(f["pad"]["nonresponse"] for f in r["frames"]) for r in selected)
        accepted = sum(all(f["pad"]["live"] for f in r["frames"]) for r in selected)
        decisions = len(selected) - nr
        key, errors = ("bpcer", decisions - accepted) if kind == "live" else ("apcer", accepted)
        result[kind] = {"presentations": len(selected), "subjects": len({r["subject_id"] for r in selected}),
                        "instruments": len({r["instrument_id"] for r in selected if r["instrument_id"]}),
                        "capture_nonresponses": sum(r["conditions"].get("capture_outcome") in
                                                    ("selection_timeout", "session_expired") for r in selected),
                        key: _rate(errors, decisions), "nonresponse": _rate(nr, len(selected)),
                        "accepted_overall": _rate(accepted, len(selected)),
                        "diagnostics": _diagnostics(selected)}
    return result


def evaluate_observations(rows, *, binding, calib_version, policy=None, independent_test=None):
    """Build an UNSIGNED report; no threshold fitting and no inference changes."""
    normalized, ids = [], set()
    groups = {split: {"subjects": set(), "instruments": set()} for split in SPLITS}
    for row in rows:
        if row.get("schema") != OBSERVATION_SCHEMA or row.get("source") != "real_sensor":
            raise ValueError("synthetic/unknown evidence cannot validate a real sensor")
        if row.get("binding") != binding or row.get("calib_version") != calib_version:
            raise ValueError("mixed model, pipeline or sensor calibration")
        fields = {k: row[k] for k in ("presentation_id", "subject_id", "consent_id", "consent", "kind",
                  "split", "instrument_id", "calib_version", "binding", "frames", "conditions")}
        if "candidate_binding" in row:
            fields["candidate_binding"] = row["candidate_binding"]
        clean = make_observation(**fields)
        if clean["presentation_id"] in ids:
            raise ValueError("duplicate presentation: adjacent frames are not independent trials")
        ids.add(clean["presentation_id"])
        group = groups[clean["split"]]
        group["subjects"].add(clean["subject_id"])
        if clean["instrument_id"]:
            group["instruments"].add(clean["instrument_id"])
        normalized.append(clean)
    for i, first in enumerate(SPLITS):
        for second in SPLITS[i + 1:]:
            if any(groups[first][key] & groups[second][key] for key in ("subjects", "instruments")):
                raise ValueError("subject/attack instrument leakage between splits")
    test = [r for r in normalized if r["split"] == "test"]
    deployment = [r for r in test if r["conditions"].get("mode") in ("enroll", "auth")]
    diagnostic = [r for r in normalized if r["conditions"].get("mode") not in ("enroll", "auth")]
    report = {"schema": SCHEMA, "source": "real_sensor", "calib_version": calib_version,
              "binding": binding, "observation_sha256": hashlib.sha256(canonical_json(normalized)).hexdigest(),
              "counts": {split: sum(r["split"] == split for r in normalized) for split in SPLITS},
              "decision_rule": "all_frames_live", "unit": "physical_presentation",
              "split_disjoint": True, "summary": summarize(deployment),
              "deployment_test_presentations": len(deployment),
              "diagnostic_presentations": len(diagnostic),
              "diagnostic_test_presentations": sum(r["split"] == "test" for r in diagnostic),
              "diagnostic_summary": summarize(diagnostic),
              "diagnostic_by_split": {split: summarize([r for r in diagnostic if r["split"] == split])
                                      for split in SPLITS},
              "by_mode": {mode: summarize([r for r in test if r["conditions"].get("mode") == mode])
                          for mode in ("enroll", "auth")}, "policy": policy,
              "independent_test": independent_test}
    report["blockers"] = evidence_blockers(report)
    report["eligible_for_review"] = not report["blockers"]
    return report


def evidence_blockers(report):
    """Re-evaluate signed report counts against explicit operator-selected limits."""
    reasons = []
    if report.get("schema") != SCHEMA or report.get("source") != "real_sensor" \
            or report.get("decision_rule") != "all_frames_live" \
            or report.get("unit") != "physical_presentation" or report.get("split_disjoint") is not True:
        return ["pad_evidence_protocol_invalid"]
    independent = report.get("independent_test")
    if not isinstance(independent, dict) or not all(independent.get(k) for k in
                                                    ("assessor", "report_id", "report_sha256")) \
            or independent.get("independent") is not True:
        reasons.append("pad_independent_test_missing")
    policy = report.get("policy")
    limits = ("max_apcer_upper95", "max_bpcer_upper95", "max_apnrr_upper95", "max_bpnrr_upper95")
    minimums = ("min_live_presentations", "min_attack_presentations", "min_live_subjects", "min_attack_instruments")
    if not isinstance(policy, dict) or not policy.get("reference") or not policy.get("approved_by") \
            or set(policy.get("required_species", [])) != set(SPECIES) \
            or set(policy.get("required_modes", [])) != {"enroll", "auth"} \
            or any(type(policy.get(k)) not in (int, float) or not math.isfinite(policy[k])
                   or not 0 <= policy[k] < 1 for k in limits) \
            or any(type(policy.get(k)) is not int or policy[k] < 1 for k in minimums):
        return reasons + ["pad_acceptance_policy_missing_or_invalid"]
    try:
        scopes = [("", report["summary"])] + [(mode + "_", report["by_mode"][mode])
                                              for mode in policy["required_modes"]]
        for prefix, summary in scopes:
            for kind in ("live", *SPECIES):
                metrics = summary[kind]
                live = kind == "live"
                minimum = policy["min_live_presentations" if live else "min_attack_presentations"]
                coverage = "subjects" if live else "instruments"
                coverage_min = policy["min_live_subjects" if live else "min_attack_instruments"]
                if type(metrics["presentations"]) is not int or metrics["presentations"] < minimum \
                        or type(metrics[coverage]) is not int or metrics[coverage] < coverage_min:
                    reasons.append(f"pad_insufficient_{prefix}{kind}_presentations")
                nr, decision = metrics["nonresponse"], metrics["bpcer" if live else "apcer"]
                if nr["trials"] != metrics["presentations"] \
                        or decision["trials"] != nr["trials"] - nr["errors"]:
                    raise ValueError("inconsistent counts")
                for rate, limit in ((decision, "max_bpcer_upper95" if live else "max_apcer_upper95"),
                                    (nr, "max_bpnrr_upper95" if live else "max_apnrr_upper95")):
                    upper = binomial_upper95(rate["errors"], rate["trials"])
                    if upper is None or upper > policy[limit]:
                        reasons.append(f"pad_{prefix}{kind}_{limit}_not_met")
    except (KeyError, TypeError, ValueError, OverflowError):
        reasons.append("pad_evidence_metrics_invalid")
    return sorted(set(reasons))


class PADValidationGate:
    """Fail-closed deployment status. No simulator exemption and no silent bypass.

    The report defaults to MODEL_DIR/PAD_VALIDATION.json; its detached Ed25519
    signature is PATH.sig and is checked with the existing trusted release key.
    Load from an already verified registry model directory. Recreate the gate on
    model reload. A status is metadata, never permission to bypass per-frame PAD.
    """

    def __init__(self, model_dir, release_pub=None, evidence_path=None, *, detector_dir=None):
        self.model_dir = Path(model_dir)
        self.release_pub = Path(release_pub) if release_pub else None
        self.evidence_path = Path(evidence_path) if evidence_path else self.model_dir / "PAD_VALIDATION.json"
        self.error = None
        try:
            self.meta = json.loads((self.model_dir / "MODEL.json").read_text())
            self.binding = model_binding(self.model_dir, detector_dir=detector_dir)
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.meta, self.binding, self.error = {}, {}, type(error).__name__

    def status(self, calib_version):
        reasons = []
        if self.error:
            reasons.append("pad_model_binding_invalid")
        if not calib_version or self.meta.get("params", {}).get("calib_version") != calib_version:
            reasons.append("pad_calibration_mismatch")
        path = self.evidence_path
        if not path.is_file():
            reasons.append("pad_real_sensor_validation_missing")
        elif self.release_pub is None:
            reasons.append("pad_validation_trust_key_missing")
        else:
            try:
                from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
                from cryptography.hazmat.primitives.serialization import load_pem_public_key
                raw = path.read_bytes()
                if len(raw) > 4 << 20:
                    raise ValueError("evidence too large")
                public = load_pem_public_key(self.release_pub.read_bytes())
                if not isinstance(public, Ed25519PublicKey):
                    raise ValueError("validation key must be Ed25519")
                public.verify(Path(str(path) + ".sig").read_bytes(), raw)
                report = json.loads(raw)
                if report.get("binding") != self.binding or report.get("calib_version") != calib_version:
                    reasons.append("pad_validation_binding_mismatch")
                reasons.extend(evidence_blockers(report))
            except Exception:  # untrusted/malformed evidence never enables authentication
                reasons.append("pad_validation_signature_or_report_invalid")
        return {"ready": not reasons, "reasons": sorted(set(reasons)),
                "model_version": self.binding.get("model_version"), "calib_version": calib_version,
                "model_calib_version": self.meta.get("params", {}).get("calib_version"),
                "binding": self.binding}
