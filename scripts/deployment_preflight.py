#!/usr/bin/env python3
"""Read-only host preflight for the Docker / Femto Bolt deployment (stdlib only).

Exit 0 means the inspected prerequisites passed, not that live enrollment has been
validated. Exit 1 lists blockers. No model, driver, secret or service is modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SECRET_FILES = (
    "ca.crt", "gateway.crt", "gateway.key", "release.pub", "approval.key", "approval.pub",
    "account.pub", "kms_master.key", "kms_token_api", "kms_token_search", "admin_token",
    "db_postgres_password", "db_api_password", "db_ro_password", "db_kms_password", "db_repl_password",
)


def dotenv(path: Path) -> dict[str, str]:
    """Read literal compose settings without executing shell expressions."""
    result = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            try:
                words = shlex.split(value, comments=True)
            except ValueError:
                continue
            result[key.strip()] = " ".join(words)
    return result


def command(args: list[str], timeout: int = 15) -> tuple[int, str]:
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
        return proc.returncode, (proc.stdout + proc.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_child(base: Path, value: str) -> Path:
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"invalid model path: {value!r}")
    path = base / relative
    if not path.resolve().is_relative_to(base.resolve()):
        raise ValueError(f"model path leaves its directory: {value!r}")
    return path


def inspect(args: argparse.Namespace) -> dict:
    checks = []

    def add(name, status, summary, details=None, remedy=None):
        item = {"id": name, "status": status, "summary": summary}
        if details is not None:
            item["details"] = details
        if remedy:
            item["remedy"] = remedy
        checks.append(item)

    cameras = []
    for node in sorted(Path("/sys/bus/usb/devices").glob("*")):
        try:
            if (node / "idVendor").read_text().strip().lower() != "2bc5":
                continue
            if (node / "idProduct").read_text().strip().lower() != "066b":
                continue
            cameras.append({"usb_path": node.name, "speed_mbps": float((node / "speed").read_text())})
        except (OSError, ValueError):
            continue
    add("camera_usb", "ok" if cameras else "blocked", "Femto Bolt USB device detected" if cameras else
        "Femto Bolt (2bc5:066b) is not visible on this host", cameras,
        None if cameras else "Connect the powered sensor to this Docker host and check lsusb.")
    if cameras:
        fast = any(item["speed_mbps"] >= 5000 for item in cameras)
        add("camera_usb_speed", "ok" if fast else "warning", "USB 3 link available" if fast else
            "The OS reports a USB 2 link; validate the selected capture profile", cameras,
            None if fast else "USB 3 is recommended. Complete frames have been captured on this link; "
            "validate sustained capture with probe_orbbec.py --fps 5 --frames 30.")

    for name, cmd, description in (
        ("docker_daemon", ["docker", "info", "--format", "{{.ServerVersion}}"], "Docker daemon"),
        ("docker_compose", ["docker", "compose", "version", "--short"], "Docker Compose"),
        ("server_image", ["docker", "image", "inspect", args.image, "--format", "{{.Id}}"],
         f"Server image {args.image}"),
    ):
        code, output = command(cmd)
        add(name, "ok" if code == 0 else "blocked", f"{description} {'available' if code == 0 else 'unavailable'}",
            output[:1000], f"docker build -f docker/Dockerfile -t {args.image} ." if name == "server_image" and code else None)

    code, output = command(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total",
                            "--format=csv,noheader,nounits"])
    gpus = []
    if code == 0:
        for row in csv.reader(output.splitlines()):
            if len(row) == 5:
                gpus.append(dict(zip(("index", "uuid", "name", "driver", "memory_mib"),
                                     (value.strip() for value in row))))
    add("nvidia_driver", "ok" if gpus else "blocked", "NVIDIA driver responds" if gpus else
        "NVIDIA driver cannot enumerate GPUs", gpus or output[:1000], None if gpus else
        "Install/load the NVIDIA module for the running kernel, or boot a kernel with its matching NVIDIA module.")
    if gpus:
        selected = [gpu for gpu in gpus if args.gpu_device_id in (gpu["index"], gpu["uuid"])]
        add("gpu_selection", "ok" if selected else "blocked", f"Configured GPU: {args.gpu_device_id}",
            selected or gpus, None if selected else "Set FA_GPU_DEVICE_ID to an available GPU index or UUID.")

    required = [args.detector_dir, "6drepnet360", args.recognizer, "farl_wflw_448", args.pad_dir]
    manifests = {}
    problems = []
    for name in required:
        try:
            directory = safe_child(args.models_dir, name)
            meta = json.loads((directory / "MODEL.json").read_text())
            if not isinstance(meta.get("files"), dict) or not meta["files"]:
                raise ValueError("MODEL.json contains no files")
            manifests[name] = meta
            for role, filename in meta["files"].items():
                path = safe_child(directory, filename)
                if not path.is_file() or path.stat().st_size == 0:
                    problems.append(f"missing: {path}")
                elif args.verify_hashes and sha256(path) != meta.get("sha256", {}).get(role):
                    problems.append(f"SHA-256 mismatch: {path}")
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            problems.append(f"{name}: {exc}")
    add("model_files", "blocked" if problems else "ok", "Required model files " +
        ("are missing or invalid" if problems else ("and hashes verified" if args.verify_hashes else "present")),
        problems or required, "Restore the listed original weights; use --verify-hashes before signing the registry."
        if problems else None)

    missing = [str(args.secrets_dir / name) for name in SECRET_FILES
               if not (args.secrets_dir / name).is_file() or (args.secrets_dir / name).stat().st_size == 0]
    device_files = [f"devices/{args.device_id}{suffix}" for suffix in (".crt", ".key", ".ed25519", ".ed25519.pub")]
    missing += [str(args.secrets_dir / name) for name in device_files if not (args.secrets_dir / name).is_file()]
    add("secrets", "blocked" if missing else "ok", "Deployment secrets " + ("missing" if missing else "present"),
        missing, "Run scripts/init_secrets.py inside the built Docker image; preserve existing keys." if missing else None)

    registry_problems = []
    registry = args.models_dir / "registry.json"
    signature = args.models_dir / "registry.json.sig"
    public_key = args.secrets_dir / "release.pub"
    try:
        entries = json.loads(registry.read_text())["models"]
        if not isinstance(entries, list) or not entries:
            raise ValueError("registry has no models")
        registered_dirs = set()
        for entry in entries:
            directory = safe_child(args.models_dir, entry["dir"])
            registered_dirs.add(entry["dir"])
            if sha256(directory / "MODEL.json") != entry["model_json_sha256"]:
                registry_problems.append(f"manifest hash mismatch: {entry['dir']}")
            for role, filename in entry["files"].items():
                path = safe_child(directory, filename)
                if not path.is_file():
                    registry_problems.append(f"registry references missing file: {path}")
                elif args.verify_hashes and sha256(path) != entry["sha256"][role]:
                    registry_problems.append(f"registry file hash mismatch: {path}")
        for directory in required:
            if directory not in registered_dirs:
                registry_problems.append(f"required model is not registered: {directory}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        registry_problems.append(str(exc))
    if not signature.is_file() or not public_key.is_file():
        registry_problems.append("registry signature or release public key is missing")
    elif shutil.which("openssl"):
        code, output = command(["openssl", "pkeyutl", "-verify", "-pubin", "-inkey", str(public_key),
                                "-rawin", "-in", str(registry), "-sigfile", str(signature)])
        if code:
            registry_problems.append("registry signature verification failed: " + output[:300])
    else:
        add("registry_signature_verification", "warning", "OpenSSL unavailable; signature presence only checked",
            remedy="The engine still verifies the Ed25519 signature during startup.")
    add("model_registry", "blocked" if registry_problems else "ok", "Signed registry " +
        ("incomplete or invalid" if registry_problems else "consistent with local files"), registry_problems,
        "Build and sign a registry containing the restored models with the deployment release key."
        if registry_problems else None)

    pad = manifests.get(args.pad_dir, {})
    actual_calibration = pad.get("params", {}).get("calib_version")
    calibrated = actual_calibration == args.calib_version
    add("pad_calibration", "ok" if calibrated else "blocked", "PAD calibration " +
        ("matches the configured camera" if calibrated else "does not match Femto Bolt"),
        {"model_calibration": actual_calibration, "required_calibration": args.calib_version},
        None if calibrated else "Use a PAD model trained and validated for this Femto Bolt capture profile; "
        "do not relabel a simulator model or bypass PAD checks.")

    return {"prerequisites_passed": not any(item["status"] == "blocked" for item in checks),
            "gpu_device_id": args.gpu_device_id, "calib_version": args.calib_version,
            "hashes_checked": args.verify_hashes, "checks": checks,
            "scope": "Host prerequisites only; service health, device registration and live enrollment require separate validation."}


def main(argv=None) -> int:
    env = {**dotenv(ROOT / ".env"), **os.environ}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Output a machine-readable report")
    parser.add_argument("--verify-hashes", action="store_true", help="Read every required/registered weight to verify SHA-256")
    parser.add_argument("--gpu-device-id", default=env.get("FA_GPU_DEVICE_ID", "0"))
    parser.add_argument("--models-dir", type=Path, default=ROOT / "data/models")
    parser.add_argument("--secrets-dir", type=Path, default=ROOT / "secrets")
    parser.add_argument("--calib-version", default=env.get("FA_CALIB_VERSION", "femto-bolt-v1"))
    parser.add_argument("--recognizer", default=env.get("FA_RECOG_MODEL", "adaface_ir101_wf12m"))
    parser.add_argument("--detector-dir", default="scrfd_34g_320")
    parser.add_argument("--pad-dir", default="pad_rgbdir_v2")
    parser.add_argument("--device-id", default=env.get("FA_DEVICE_ID", "dev-001"))
    parser.add_argument("--image", default="faceauth:dev")
    args = parser.parse_args(argv)
    report = inspect(args)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        for item in report["checks"]:
            print(f"[{item['status'].upper()}] {item['id']}: {item['summary']}")
            if item["status"] != "ok" and item.get("details"):
                details = item["details"]
                print("  " + (details if isinstance(details, str) else json.dumps(details, ensure_ascii=False)))
            if item.get("remedy"):
                print("  Next: " + item["remedy"])
        print("\n" + ("Prerequisites passed. " if report["prerequisites_passed"] else "Full authentication is blocked. ") +
              report["scope"])
    return 0 if report["prerequisites_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
