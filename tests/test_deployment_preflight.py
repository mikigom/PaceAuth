"""Host preflight must report missing and altered artifacts without Docker imports."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import deployment_preflight as preflight


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.args = argparse.Namespace(
            gpu_device_id="0", models_dir=root / "models", secrets_dir=root / "secrets",
            recognizer="recognizer", detector_dir="detector", pad_dir="pad", device_id="dev-001",
            calib_version="femto-bolt-v1", verify_hashes=True, image="faceauth:test",
        )
        self.args.models_dir.mkdir()
        self.args.secrets_dir.mkdir()
        entries = []
        for name in ("recognizer", "detector", "pad", "6drepnet360", "farl_wflw_448"):
            directory = self.args.models_dir / name
            directory.mkdir()
            (directory / "weights.bin").write_bytes(b"test weight bytes")
            meta = {"files": {"weights": "weights.bin"},
                    "sha256": {"weights": preflight.sha256(directory / "weights.bin")},
                    "params": {"calib_version": self.args.calib_version}}
            (directory / "MODEL.json").write_text(json.dumps(meta))
            entries.append({"dir": name, **meta,
                            "model_json_sha256": preflight.sha256(directory / "MODEL.json")})
        (self.args.models_dir / "registry.json").write_text(json.dumps({"models": entries}))
        (self.args.models_dir / "registry.json.sig").write_bytes(b"test signature")
        names = list(preflight.SECRET_FILES) + ["devices/dev-001" + suffix
                                              for suffix in (".crt", ".key", ".ed25519", ".ed25519.pub")]
        for name in names:
            path = self.args.secrets_dir / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b"private fixture data must never appear in the report")

    @staticmethod
    def successful_command(args, timeout=15):
        if args[0] == "nvidia-smi":
            return 0, "0, GPU-test, Test GPU, 580.0, 8192"
        return 0, "available"

    def report(self, command=None):
        with patch.object(preflight, "command", command or self.successful_command), \
                patch.object(preflight.shutil, "which", return_value="/usr/bin/openssl"):
            return preflight.inspect(self.args)

    @staticmethod
    def check(report, name):
        return next(item for item in report["checks"] if item["id"] == name)

    def test_altered_weight_blocks_models_and_registry(self):
        (self.args.models_dir / "recognizer/weights.bin").write_bytes(b"altered")
        report = self.report()
        self.assertEqual(self.check(report, "model_files")["status"], "blocked")
        self.assertEqual(self.check(report, "model_registry")["status"], "blocked")
        self.assertFalse(report["prerequisites_passed"])

    def test_simulator_pad_blocks_femto(self):
        path = self.args.models_dir / "pad/MODEL.json"
        meta = json.loads(path.read_text())
        meta["params"]["calib_version"] = "gemini336-sim-v1"
        path.write_text(json.dumps(meta))
        report = self.report()
        self.assertEqual(self.check(report, "pad_calibration")["status"], "blocked")

    def test_bad_signature_blocks_registry(self):
        def command(args, timeout=15):
            return (1, "Signature Verification Failure") if args[0] == "openssl" else self.successful_command(args)
        report = self.report(command)
        self.assertEqual(self.check(report, "model_registry")["status"], "blocked")

    def test_gpu_selection_and_no_secret_output(self):
        self.args.gpu_device_id = "5"
        report = self.report()
        self.assertEqual(self.check(report, "gpu_selection")["status"], "blocked")
        self.assertNotIn("private fixture data", json.dumps(report))

    def test_model_paths_cannot_leave_directory(self):
        for value in ("../outside", "/etc/passwd", ""):
            with self.subTest(value=value), self.assertRaises(ValueError):
                preflight.safe_child(self.args.models_dir, value)


if __name__ == "__main__":
    unittest.main()
