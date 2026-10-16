"""Declare and evaluate a prospective, limited Femto PAD runtime pilot cohort.

No fitting, camera access, activation, log rewriting, or production validation.
All matching recorded attempts after the declared byte boundary remain in the
fixed nine-presentation cohort, including failures and timeouts. Extra attempts
block release. Explicit operator reconciliation covers unrecorded interruptions.
"""

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import calibrate_femto_pad as calibration_cli
from faceauth.common import femto_pad_profile, pad_validation
from faceauth.common.femto_pad_profile import _decision, _validate_base, evaluate_profile
from faceauth.common.pad_validation import OBSERVATION_SCHEMA, canonical_json, clean_candidate_binding

PROTOCOL_SCHEMA = "faceauth.femto_pad.runtime_protocol.v1"
REPORT_SCHEMA = "faceauth.femto_pad.runtime_report.v1"
RECONCILIATION_SCHEMA = "faceauth.femto_pad.runtime_attempt_reconciliation.v1"
KINDS = ("live", "print", "replay")
MODES = ("auth", "enroll")
REQUIRED = {"live": {"auth": 1, "enroll": 2}, "print": {"auth": 3, "enroll": 0},
            "replay": {"auth": 3, "enroll": 0}}
FRAME_COUNTS = {"auth": 3, "enroll": 18}
RULES = {"split": "test", "kinds": list(KINDS), "modes": list(MODES),
         "exact_presentations": REQUIRED, "complete_frame_counts": FRAME_COUNTS,
         "all_matching_appended_attempts": True, "replace_failures": False,
         "all_live_require_good_final_quality": True,
         "quality_passing_responding_auth_rejections_per_attack_kind": 1,
         "operator_attempt_reconciliation_required": True,
         "started_attempts": 9, "recorded_presentations": 9,
         "cancelled_attempts": 0, "unrecorded_attempts": 0,
         "attack_scope": ["auth"], "live_scope": list(MODES)}
KNOWN_LIMITATION = "prior_diagnostic_live_rejection_remains_unfixed"


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _source_hashes():
    return {"runtime_evaluator": _sha(Path(__file__).read_bytes()),
            "label_corrections": _sha(Path(calibration_cli.__file__).read_bytes()),
            "profile_evaluator": _sha(Path(femto_pad_profile.__file__).read_bytes()),
            "observation_validation": _sha(Path(pad_validation.__file__).read_bytes())}


def _rows(raw, *, complete=False):
    if complete and raw and not raw.endswith(b"\n"):
        raise ValueError("observation_log_incomplete_line")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("observation_log_invalid_rows")
    return rows


def _prior_summary(raw, profile_sha, corrections_sha):
    prior = json.loads(raw)
    if (prior.get("frozen_profile_sha256") != profile_sha
            or prior.get("correction_sha256") != corrections_sha
            or prior.get("eligible_for_local_pilot") is not False):
        raise ValueError("prior_diagnostic_report_provenance_mismatch")
    # Retain only aggregate counts, never paths, identifiers, or free-text notes.
    summary = {kind: {key: prior["test_summary"][kind][key] for key in
                      ("presentations", "accepted", "nonresponses", "responding", "errors")}
               for kind in KINDS}
    for kind, counts in summary.items():
        if (any(type(value) is not int or value < 0 for value in counts.values())
                or counts["responding"] + counts["nonresponses"] != counts["presentations"]
                or counts["accepted"] > counts["responding"]
                or counts["errors"] != (counts["responding"] - counts["accepted"]
                                        if kind == "live" else counts["accepted"])):
            raise ValueError("prior_diagnostic_report_invalid_counts")
    if summary["live"] != {"presentations": 6, "accepted": 5, "nonresponses": 0,
                           "responding": 6, "errors": 1}:
        raise ValueError("prior_diagnostic_known_failure_missing")
    return {"report_sha256": _sha(raw), "summary": summary,
            "eligible_for_local_pilot": False, "known_limitation": KNOWN_LIMITATION,
            "failure_resolved": False}


def create_protocol(observations, profile_raw, corrections, prior_report):
    """Freeze a declaration before collection; the caller writes it exclusively."""
    profile = json.loads(profile_raw)
    _validate_base(profile)
    if "evaluation" in profile:
        raise ValueError("unevaluated_frozen_profile_required")
    rows = _rows(observations, complete=True)
    journal = _rows(corrections)
    calibration_cli.apply_label_corrections(rows, journal)
    return {"schema": PROTOCOL_SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(),
            "experimental": True, "production_validated": False,
            "frozen_profile_sha256": _sha(profile_raw),
            "candidate_binding": copy.deepcopy(profile["candidate_binding"]),
            "calib_version": profile["calib_version"],
            "source_prefix": {"bytes": len(observations), "sha256": _sha(observations), "rows": len(rows)},
            "corrections_sha256": _sha(corrections), "correction_count": len(journal),
            "prior_diagnostic": _prior_summary(prior_report, _sha(profile_raw), _sha(corrections)),
            "rules": copy.deepcopy(RULES), "evaluator_sources_sha256": _source_hashes()}


def _mode(row):
    conditions = row.get("conditions")
    return conditions.get("mode") if isinstance(conditions, dict) else None


def _good_capture(row):
    frames, mode = row.get("frames"), _mode(row)
    return (isinstance(frames, list) and len(frames) == FRAME_COUNTS.get(mode)
            and row["conditions"].get("capture_outcome") not in ("selection_timeout", "session_expired")
            and all(isinstance(frame, dict) and frame.get("ok") is True and frame.get("reasons") == []
                    for frame in frames))


def _quality_rejection(row, thresholds):
    if _mode(row) != "auth" or not _good_capture(row):
        return False
    try:
        decisions = [_decision(frame["pad_candidate"], thresholds) for frame in row["frames"]]
        return not any(value["nonresponse"] for value in decisions) and not all(value["live"] for value in decisions)
    except (ValueError, TypeError, KeyError):
        return False


def _reconcile(raw, protocol_sha256, recorded, declared_at):
    """Validate an operator declaration, not an independent attempt ledger."""
    if raw is None:
        raise ValueError("runtime_attempt_reconciliation_missing")
    try:
        value = json.loads(raw)
        fields = {"schema", "protocol_sha256", "started_attempts", "recorded_presentations",
                  "cancelled_attempts", "unrecorded_attempts", "operator_confirmed", "confirmed_at"}
        if (not isinstance(value, dict) or set(value) != fields
                or value["schema"] != RECONCILIATION_SCHEMA or value["protocol_sha256"] != protocol_sha256
                or value["operator_confirmed"] is not True):
            raise ValueError
        for key in ("started_attempts", "recorded_presentations", "cancelled_attempts", "unrecorded_attempts"):
            if type(value[key]) is not int or value[key] != RULES[key]:
                raise ValueError
        confirmed = datetime.fromisoformat(value["confirmed_at"].replace("Z", "+00:00"))
        declared = datetime.fromisoformat(declared_at.replace("Z", "+00:00"))
        if (confirmed.tzinfo is None or confirmed.utcoffset() is None or confirmed < declared
                or confirmed > datetime.now(timezone.utc) or value["recorded_presentations"] != recorded):
            raise ValueError
    except (ValueError, TypeError, KeyError, AttributeError):
        raise ValueError("runtime_attempt_reconciliation_invalid") from None
    return value


def evaluate_protocol(observations, profile_raw, corrections, prior_report, protocol_raw, protocol_sha256,
                      attempt_reconciliation=None):
    """Return an aggregate report and an eligible artifact, or None on any failure."""
    report = {"schema": REPORT_SCHEMA, "reviewed_at": datetime.now(timezone.utc).isoformat(),
              "experimental": True, "production_validated": False, "eligible_for_local_pilot": False,
              "protocol_sha256": _sha(protocol_raw), "source_sha256": _sha(observations),
              "source_bytes": len(observations), "corrections_sha256": _sha(corrections),
              "attempt_reconciliation_sha256": _sha(attempt_reconciliation) if attempt_reconciliation is not None else None,
              "reasons": []}
    reasons = report["reasons"]
    result = None
    try:
        if _sha(protocol_raw) != protocol_sha256:
            raise ValueError("protocol_hash_mismatch")
        protocol = json.loads(protocol_raw)
        if (protocol.get("schema") != PROTOCOL_SCHEMA or protocol.get("rules") != RULES
                or protocol.get("experimental") is not True or protocol.get("production_validated") is not False):
            raise ValueError("protocol_schema_or_rules_mismatch")
        if protocol.get("evaluator_sources_sha256") != _source_hashes():
            raise ValueError("protocol_evaluator_source_mismatch")
        if protocol.get("frozen_profile_sha256") != _sha(profile_raw):
            raise ValueError("frozen_profile_hash_mismatch")
        if protocol.get("corrections_sha256") != _sha(corrections):
            raise ValueError("corrections_hash_mismatch")
        prior = _prior_summary(prior_report, _sha(profile_raw), _sha(corrections))
        if protocol.get("prior_diagnostic") != prior:
            raise ValueError("prior_diagnostic_report_mismatch")
        report["prior_diagnostic"] = prior
        profile = json.loads(profile_raw)
        _validate_base(profile)
        if ("evaluation" in profile or protocol.get("candidate_binding") != profile["candidate_binding"]
                or protocol.get("calib_version") != profile["calib_version"]):
            raise ValueError("frozen_profile_provenance_mismatch")
        prefix = protocol["source_prefix"]
        length = prefix["bytes"]
        if (type(length) is not int or length < 0 or length > len(observations)
                or _sha(observations[:length]) != prefix["sha256"]):
            raise ValueError("original_observation_prefix_changed")
        before = _rows(observations[:length], complete=True)
        if len(before) != prefix["rows"]:
            raise ValueError("original_observation_row_count_changed")
        appended = _rows(observations[length:], complete=True)
        raw_rows = before + appended
        rows = calibration_cli.apply_label_corrections(raw_rows, _rows(corrections))
        report.update(source_rows=len(rows), source_prefix=prefix,
                      frozen_profile_sha256=_sha(profile_raw), appended_rows=len(appended),
                      evaluator_sources_sha256=_source_hashes())
        selected, other = [], []
        other_reasons = {"not_test_split": 0, "out_of_scope_kind": 0, "out_of_scope_mode": 0}
        identifier_counts = Counter(row["presentation_id"] for row in rows
                                    if isinstance(row.get("presentation_id"), str))
        previous_ids = {row.get("presentation_id") for row in before if isinstance(row.get("presentation_id"), str)}
        for row in rows[len(before):]:
            if row.get("split") == "test" and row.get("kind") in KINDS and _mode(row) in MODES:
                selected.append(row)
                try:
                    matching = (row.get("schema") == OBSERVATION_SCHEMA and row.get("source") == "real_sensor"
                                and row.get("consent") is True and row.get("calib_version") == profile["calib_version"]
                                and clean_candidate_binding(row.get("candidate_binding")) == profile["candidate_binding"])
                except (ValueError, TypeError, KeyError):
                    matching = False
                if not matching:
                    reasons.append("runtime_observation_provenance_mismatch")
                identifier = row.get("presentation_id")
                if (not isinstance(identifier, str) or not identifier.strip()
                        or identifier in previous_ids or identifier_counts[identifier] != 1):
                    reasons.append("runtime_presentation_not_new_or_unique")
            else:
                other.append(row)
                reason = ("not_test_split" if row.get("split") != "test" else "out_of_scope_kind"
                          if row.get("kind") not in KINDS else "out_of_scope_mode")
                other_reasons[reason] += 1
        counts = {kind: {mode: sum(row["kind"] == kind and _mode(row) == mode for row in selected)
                         for mode in MODES} for kind in KINDS}
        report.update(included_rows=len(selected), counts=counts,
                      included_row_sha256=[_sha(canonical_json(row)) for row in selected],
                      other_appended_rows=len(other),
                      other_appended_reason_counts=other_reasons,
                      other_appended_row_sha256=[_sha(canonical_json(row)) for row in other])
        for kind, modes in REQUIRED.items():
            for mode, required in modes.items():
                if counts[kind][mode] < required:
                    reasons.append(f"runtime_insufficient_{kind}_{mode}")
                elif counts[kind][mode] > required:
                    reasons.append(f"runtime_excess_{kind}_{mode}")
        try:
            report["attempt_reconciliation"] = _reconcile(
                attempt_reconciliation, protocol_sha256, len(selected), protocol["created_at"])
        except ValueError as error:
            reasons.append(str(error))
        quality_failures = {mode: sum(row["kind"] == "live" and _mode(row) == mode and not _good_capture(row)
                                      for row in selected) for mode in MODES}
        report["live_incomplete_or_quality_failed"] = quality_failures
        for mode, count in quality_failures.items():
            if count:
                reasons.append(f"runtime_live_{mode}_incomplete_or_quality_failed")
        attack_checks = {kind: sum(row["kind"] == kind and _quality_rejection(row, profile["thresholds"])
                                  for row in selected) for kind in ("print", "replay")}
        report["quality_passing_responding_auth_rejections"] = attack_checks
        for kind, count in attack_checks.items():
            if count < 1:
                reasons.append(f"runtime_{kind}_quality_passing_auth_rejection_missing")
        # Produce full threshold outcomes even when an additional cohort rule
        # fails; the artifact is withheld unless every rule and old gate pass.
        if all(sum(counts[kind].values()) >= 3 for kind in KINDS):
            result = evaluate_profile(profile, selected)
            report["evaluation"] = copy.deepcopy(result["evaluation"])
            reasons.extend(result["evaluation"]["reasons"])
    except (ValueError, TypeError, KeyError, UnicodeError) as error:
        # These errors originate in structural/provenance validators. Do not
        # include arbitrary JSON fields, IDs, or operator text in diagnostics.
        reason = str(error)
        allowed = {"protocol_hash_mismatch", "protocol_schema_or_rules_mismatch", "protocol_evaluator_source_mismatch",
                   "frozen_profile_hash_mismatch", "corrections_hash_mismatch", "prior_diagnostic_report_mismatch",
                   "prior_diagnostic_report_provenance_mismatch", "prior_diagnostic_known_failure_missing",
                   "prior_diagnostic_report_invalid_counts", "frozen_profile_provenance_mismatch",
                   "original_observation_prefix_changed", "original_observation_row_count_changed",
                   "observation_log_incomplete_line", "observation_log_invalid_rows"}
        reasons.append(reason if reason in allowed else "runtime_observations_or_protocol_invalid")
    report["reasons"] = sorted(set(reasons))
    report["eligible_for_local_pilot"] = result is not None and not report["reasons"]
    if not report["eligible_for_local_pilot"]:
        return report, None
    result["evaluation"]["prospective_runtime_cohort"] = {
        key: copy.deepcopy(value) for key, value in report.items() if key != "evaluation"}
    return report, result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("create", "evaluate"):
        command = commands.add_parser(name)
        for option in ("observations", "profile", "corrections", "prior-report"):
            command.add_argument("--" + option, type=Path, required=True)
        if name == "create":
            command.add_argument("--output", type=Path, required=True)
        else:
            command.add_argument("--protocol", type=Path, required=True)
            command.add_argument("--protocol-sha256", required=True)
            command.add_argument("--report", type=Path, required=True)
            command.add_argument("--attempt-reconciliation", type=Path,
                                 help="Explicit operator declaration of all nine starts and no unrecorded interruptions")
            command.add_argument("--output-profile", type=Path,
                                 help="Created only if every frozen threshold and prospective cohort check passes")
    args = parser.parse_args(argv)
    try:
        inputs = [path.read_bytes() for path in (args.observations, args.profile, args.corrections, args.prior_report)]
        if args.command == "create":
            protocol = create_protocol(*inputs)
            digest = calibration_cli._write(args.output, protocol)
            print(json.dumps({"protocol_sha256": digest, "source_prefix": protocol["source_prefix"],
                              "exact_presentations": REQUIRED, "prior_diagnostic": protocol["prior_diagnostic"]}))
            return 0
        reconciliation = args.attempt_reconciliation.read_bytes() if args.attempt_reconciliation else None
        report, profile = evaluate_protocol(*inputs, args.protocol.read_bytes(), args.protocol_sha256, reconciliation)
        # Never overwrite a previous report, protocol or eligible artifact.
        digest = calibration_cli._write(args.report, report)
        profile_sha = calibration_cli._write(args.output_profile, profile) if profile is not None and args.output_profile else None
        print(json.dumps({"report_sha256": digest, "profile_sha256": profile_sha,
                          "eligible_for_local_pilot": report["eligible_for_local_pilot"],
                          "included_rows": report.get("included_rows", 0), "reasons": report["reasons"]}))
        return 0 if report["eligible_for_local_pilot"] else 2
    except (OSError, ValueError, TypeError, KeyError):
        print("Femto runtime protocol failed: input invalid/unavailable or output already exists.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
