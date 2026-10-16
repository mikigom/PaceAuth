"""Fit/evaluate an experimental Femto Bolt profile from existing numerical observations.

No camera access, automatic activation, or production PAD validation occurs.
Output files are created exclusively with mode 0600. Fitting reads calibration
rows only; evaluation uses frozen thresholds and distinct test presentations.
"""

import argparse
import copy
from datetime import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from faceauth.common.femto_pad_profile import evaluate_profile, fit_profile
from faceauth.common.pad_validation import SPECIES, canonical_json, clean_candidate_binding

CORRECTION_SCHEMA = "faceauth.pad.label_correction.v1"
CORRECTION_FIELDS = frozenset(("schema", "presentation_id", "original_row_sha256", "original_kind",
    "original_instrument_id", "corrected_kind", "corrected_instrument_id", "reason", "corrected_at"))


def apply_label_corrections(rows, corrections):
    """Apply an explicit correction journal to a copy, preserving capture facts.

    The original log and frozen profiles remain unchanged. A correction binds
    one original record and can change only its physical label/instrument, never
    its frames, capture mode, split, measurements or provenance. Errors avoid
    printing identifiers or operator-provided free text.
    """
    if not isinstance(rows, list) or not isinstance(corrections, list):
        raise ValueError("observations and corrections must be lists")
    indices = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("observation must be a mapping")
        identifier = row.get("presentation_id")
        if isinstance(identifier, str):
            indices.setdefault(identifier, []).append(index)
    result, seen = copy.deepcopy(rows), set()
    for correction in corrections:
        if (not isinstance(correction, dict) or set(correction) != CORRECTION_FIELDS
                or correction.get("schema") != CORRECTION_SCHEMA):
            raise ValueError("invalid label correction schema or fields")
        identifier = correction["presentation_id"]
        if not isinstance(identifier, str) or not identifier.strip() or identifier in seen:
            raise ValueError("label correction requires a unique presentation identifier")
        seen.add(identifier)
        matched = indices.get(identifier, [])
        if len(matched) != 1:
            raise ValueError("label correction must match exactly one original observation")
        row = rows[matched[0]]
        digest = correction["original_row_sha256"]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
                or hashlib.sha256(canonical_json(row)).hexdigest() != digest):
            raise ValueError("label correction original observation hash mismatch")
        if (correction["original_kind"] != row.get("kind")
                or correction["original_instrument_id"] != row.get("instrument_id")):
            raise ValueError("label correction original metadata mismatch")
        for prefix in ("original", "corrected"):
            kind, instrument = correction[prefix + "_kind"], correction[prefix + "_instrument_id"]
            if (kind not in ("live", *SPECIES)
                    or (kind == "live" and instrument is not None)
                    or (kind != "live" and (not isinstance(instrument, str) or not instrument.strip()))):
                raise ValueError("invalid label correction kind or instrument")
        reason, timestamp = correction["reason"], correction["corrected_at"]
        if not isinstance(reason, str) or not reason.strip() or not isinstance(timestamp, str):
            raise ValueError("label correction requires a reason and timestamp")
        try:
            corrected_at = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            if corrected_at.tzinfo is None or corrected_at.utcoffset() is None:
                raise ValueError
        except ValueError:
            raise ValueError("label correction requires a timezone-aware ISO timestamp") from None
        result[matched[0]]["kind"] = correction["corrected_kind"]
        result[matched[0]]["instrument_id"] = correction["corrected_instrument_id"]
    return result


def _write(path, value):
    raw = canonical_json(value) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
    return hashlib.sha256(raw).hexdigest()


def _candidate_rows(rows):
    selected, legacy = [], 0
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("observation must be a mapping")
        if "candidate_binding" not in row:
            legacy += 1
            continue
        # Present but malformed candidate provenance is never silently skipped.
        clean_candidate_binding(row["candidate_binding"])
        selected.append(row)
    return selected, legacy


def _select(rows, binding, calib_version):
    selected = [row for row in rows if clean_candidate_binding(row["candidate_binding"]) == binding
                and row.get("calib_version") == calib_version]
    return selected, len(rows) - len(selected)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("fit", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--observations", type=Path, required=True)
        command.add_argument("--corrections", type=Path,
                             help="Explicit label correction journal; never rewrites the original observations")
        command.add_argument("--output", type=Path, required=True)
        if name == "fit":
            command.add_argument("--calib-version", required=True)
            command.add_argument("--candidate-binding", type=Path,
                                 help="Expected binding JSON; otherwise require a unique calibration binding")
        else:
            command.add_argument("--profile", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        rows = [json.loads(line) for line in args.observations.read_text().splitlines() if line.strip()]
        corrections, corrections_sha256 = [], None
        if args.corrections:
            raw = args.corrections.read_bytes()
            corrections = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
            rows = apply_label_corrections(rows, corrections)
            corrections_sha256 = hashlib.sha256(raw).hexdigest()
        rows, legacy_skipped = _candidate_rows(rows)
        if args.command == "fit":
            if args.candidate_binding:
                binding = clean_candidate_binding(json.loads(args.candidate_binding.read_text()))
            else:
                bindings = {canonical_json(clean_candidate_binding(row.get("candidate_binding"))) for row in rows
                            if row.get("split") == "calibration" and row.get("calib_version") == args.calib_version
                            and row.get("kind") in ("live", "print", "replay")}
                if len(bindings) != 1:
                    raise ValueError("a unique matching candidate binding is required")
                binding = json.loads(bindings.pop())
            rows, provenance_excluded = _select(rows, binding, args.calib_version)
            result = fit_profile(rows, binding=binding, calib_version=args.calib_version)
            eligible = False
        else:
            profile = json.loads(args.profile.read_text())
            rows, provenance_excluded = _select(rows, clean_candidate_binding(profile.get("candidate_binding")),
                                               profile.get("calib_version"))
            result = evaluate_profile(profile, rows)
            eligible = result["evaluation"]["eligible_for_local_pilot"]
        digest = _write(args.output, result)
        print(json.dumps({"output": str(args.output), "sha256": digest, "experimental": True,
                          "scope": ["print", "replay"], "eligible_for_local_pilot": eligible,
                          "legacy_skipped": legacy_skipped, "provenance_excluded": provenance_excluded,
                          "label_corrections": len(corrections), "corrections_sha256": corrections_sha256,
                          "production_validated": False,
                          "reasons": result.get("evaluation", {}).get("reasons", ["pilot_evaluation_missing"])}, indent=2))
        return 0 if args.command == "fit" or eligible else 2
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"Femto PAD profile failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
