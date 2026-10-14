"""GPU startup must stop before container changes when prerequisites fail."""
import argparse
import json
import unittest
from unittest.mock import patch

from scripts import start_gpu_backend as startup


class StartupTests(unittest.TestCase):
    def args(self):
        return argparse.Namespace(gpu_device_id=None, check_only=False, json=True, timeout=10)

    def test_host_driver_failure_does_not_call_docker(self):
        with patch.object(startup, "run", side_effect=startup.StartupError("driver absent")) as run, \
                patch.object(startup, "installed_nvidia_kernels", return_value=["next-kernel"]), \
                patch.object(startup.platform, "release", return_value="running-kernel"):
            result = startup.start(self.args())
        self.assertFalse(result["ready"])
        self.assertIn("next-kernel", result["next_action"])
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0][0], "nvidia-smi")

    def test_wrong_gpu_selection_does_not_launch_a_container(self):
        configuration = {"services": {"engine-1": {
            "image": "faceauth:test", "deploy": {"resources": {"reservations": {
                "devices": [{"device_ids": ["5"]}]}}}}}}
        with patch.object(startup, "run", side_effect=["0, GPU-id, Test GPU, 8192, 0, 8192",
                                                      json.dumps(configuration)]) as run:
            result = startup.start(self.args())
        self.assertFalse(result["ready"])
        self.assertIn("GPU '5' is absent", result["error"])
        self.assertEqual(run.call_count, 2)

    def test_healthy_cpu_engine_is_not_accepted(self):
        with self.assertRaises(startup.StartupError):
            startup.validate_service("engine-1", {"ok": True, "providers": ["CPUExecutionProvider"]})

    def test_search_must_finish_rebuilding(self):
        with self.assertRaises(startup.StartupError):
            startup.validate_service("search-1", {"ready": False})
        startup.validate_service("search-1", {"ready": True})

    def test_models_must_execute_before_engine_is_ready(self):
        health = {"ok": True, "providers": ["TensorrtExecutionProvider"]}
        with self.assertRaises(startup.StartupError):
            startup.validate_service("engine-1", health)
        health["warmup"] = {name: True for name in ("synthetic", "detector", "pose", "recognition", "pad")}
        startup.validate_service("engine-1", health)


if __name__ == "__main__":
    unittest.main()
