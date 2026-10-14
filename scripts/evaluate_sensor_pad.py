"""Evaluate consented, score-only physical PAD observations; never captures images.

    python scripts/evaluate_sensor_pad.py status --model-dir data/models/pad_rgbdir_v2 \
        --calib-version femto-bolt-v1-50beeef1ae82fb27
    python scripts/evaluate_sensor_pad.py evaluate --observations study.jsonl \
        --model-dir MODEL_DIR --calib-version CALIB --output report.json

Without a reviewed --policy and --independent-test reference, evaluation still
reports useful measurements but cannot pass the deployment gate. See
docs/FEMTO_PAD_VALIDATION.md for the study protocol and explicit release process.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from faceauth.common.pad_validation import (PADValidationGate, canonical_json, evidence_blockers,
                                           evaluate_observations, model_binding)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("status", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--model-dir", type=Path, required=True)
        command.add_argument("--detector-dir", type=Path)
        command.add_argument("--calib-version", required=True)
        if name == "status":
            command.add_argument("--release-pub", type=Path)
            command.add_argument("--evidence", type=Path)
        else:
            command.add_argument("--observations", type=Path, required=True)
            command.add_argument("--policy", type=Path)
            command.add_argument("--independent-test", type=Path)
            command.add_argument("--output", type=Path, required=True)
    sign = commands.add_parser("sign", help="explicit release review: sign a qualifying report")
    sign.add_argument("--report", type=Path, required=True)
    sign.add_argument("--release-key", type=Path, required=True)
    sign.add_argument("--model-dir", type=Path, required=True)
    sign.add_argument("--detector-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "status":
            result = PADValidationGate(args.model_dir, args.release_pub, args.evidence,
                                       detector_dir=args.detector_dir).status(args.calib_version)
            print(json.dumps(result, indent=2))
            return 0 if result["ready"] else 2
        if args.command == "evaluate":
            rows = [json.loads(line) for line in args.observations.read_text().splitlines() if line.strip()]
            read = lambda path: json.loads(path.read_text()) if path else None
            result = evaluate_observations(rows, binding=model_binding(args.model_dir, detector_dir=args.detector_dir),
                                           calib_version=args.calib_version, policy=read(args.policy),
                                           independent_test=read(args.independent_test))
            # Exclusive create avoids silently replacing a reviewed report. No key
            # access occurs in evaluation, and no signature is generated here.
            with args.output.open("xb") as out:
                out.write(canonical_json(result) + b"\n")
            print(json.dumps({"output": str(args.output), "counts": result["counts"],
                              "blockers": result["blockers"], "signed": False}, indent=2))
            return 0 if result["eligible_for_review"] else 2
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives.serialization import load_pem_private_key
        raw = args.report.read_bytes()
        report = json.loads(raw)
        blockers = evidence_blockers(report)
        if report.get("binding") != model_binding(args.model_dir, detector_dir=args.detector_dir):
            blockers.append("pad_validation_binding_mismatch")
        meta = json.loads((args.model_dir / "MODEL.json").read_text())
        if report.get("calib_version") != meta["params"]["calib_version"]:
            blockers.append("pad_calibration_mismatch")
        if blockers:
            raise ValueError("report cannot qualify: " + ", ".join(blockers))
        key = load_pem_private_key(args.release_key.read_bytes(), password=None)
        if not isinstance(key, Ed25519PrivateKey):
            raise ValueError("release key must be Ed25519")
        target = Path(str(args.report) + ".sig")
        with target.open("xb") as out:
            out.write(key.sign(raw))
        print(json.dumps({"signature": str(target), "reviewed_report": str(args.report)}))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"PAD evaluation failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
