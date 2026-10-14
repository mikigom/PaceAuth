#!/usr/bin/env python3
"""Check NVIDIA/Docker, then start the existing GPU backend without replacing data or keys.

Run after booting the kernel with the installed NVIDIA driver:
    python3 scripts/start_gpu_backend.py
    python3 scripts/start_gpu_backend.py --check-only --json

This does not install drivers, reboot, build images, change active models, or
recreate database volumes. It checks GPU execution before changing services.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
GPU_SERVICES = ("engine-1", "search-1", "engine-2", "search-2")
PLATFORM_SERVICES = ("db", "db-standby1", "db-standby2", "kms", "api-1", "api-2", "gateway")

GPU_SMOKE = r'''
import json
import numpy as np
import torch
import onnxruntime as ort
import faiss
import tensorrt
if not torch.cuda.is_available():
    raise RuntimeError("PyTorch has no usable CUDA device")
x = torch.arange(16, device="cuda", dtype=torch.float32).reshape(4, 4)
assert torch.isfinite(x @ x.T).all().item()
ort.preload_dlls()
if "CUDAExecutionProvider" not in ort.get_available_providers():
    raise RuntimeError("ONNX Runtime CUDA provider is unavailable")
resource = faiss.StandardGpuResources()
resource.setTempMemory(32 << 20)
index = faiss.GpuIndexFlatIP(resource, 512)
vectors = np.eye(2, 512, dtype=np.float32)
index.add(vectors)
scores, ids = index.search(vectors, 1)
assert ids[:, 0].tolist() == [0, 1] and np.allclose(scores[:, 0], 1)
print(json.dumps({"device": torch.cuda.get_device_name(0), "torch": torch.__version__,
    "cuda": torch.version.cuda, "onnxruntime": ort.__version__, "providers": ort.get_available_providers(),
    "tensorrt": tensorrt.__version__, "faiss": getattr(faiss, "__version__", "unknown"),
    "torch_cuda_execution": True, "faiss_gpu_search": True}))
'''


class StartupError(RuntimeError):
    pass


def run(command, *, env, timeout=30, input_text=None):
    try:
        result = subprocess.run(command, cwd=ROOT, env=env, input=input_text, capture_output=True,
                                text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise StartupError(str(exc)) from exc
    if result.returncode:
        # No commands contain passwords or read secret contents into stdout.
        detail = (result.stderr or result.stdout).strip()[-4000:]
        raise StartupError(f"{' '.join(command[:5])}: {detail}")
    return result.stdout.strip()


def gpu_snapshot(env):
    output = run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free",
                  "--format=csv,noheader,nounits"], env=env)
    rows = []
    for values in csv.reader(output.splitlines()):
        if len(values) != 6:
            raise StartupError("NVIDIA returned an invalid GPU inventory")
        index, uuid, name, total, used, free = (value.strip() for value in values)
        rows.append({"index": index, "uuid": uuid, "name": name,
                     "total_mib": int(total), "used_mib": int(used), "free_mib": int(free)})
    if not rows:
        raise StartupError("NVIDIA returned no GPUs")
    return rows


def selected_gpu(rows, identifier):
    for row in rows:
        if identifier in (row["index"], row["uuid"]):
            return row
    raise StartupError(f"Configured GPU {identifier!r} is absent; available indices: " +
                       ", ".join(row["index"] for row in rows))


def installed_nvidia_kernels():
    return sorted(directory.name for directory in Path("/lib/modules").glob("*")
                  if any((directory / "kernel").glob("nvidia*/nvidia.ko*")))


def validate_service(name, payload):
    if name.startswith("engine-"):
        providers = payload.get("providers") or []
        if (payload.get("ok") is not True or not providers or
                providers[0] not in ("CUDAExecutionProvider", "TensorrtExecutionProvider")):
            raise StartupError(f"{name} is not serving with a GPU execution provider: {payload}")
        warmup = payload.get("warmup") or {}
        if not all(warmup.get(model) is True for model in ("synthetic", "detector", "pose", "recognition", "pad")):
            raise StartupError(f"{name} has not completed synthetic GPU inference warm-up")
    elif payload.get("ready") is not True:
        raise StartupError(f"{name} has not rebuilt its search index: {payload}")


def start(args):
    env = dict(os.environ)
    if args.gpu_device_id is not None:
        env["FA_GPU_DEVICE_ID"] = args.gpu_device_id
    report = {"ready": False, "kernel": platform.release(), "stages": [],
              "scope": "GPU runtime and service readiness; real enrollment/PAD validation remains a separate check."}

    def stage(name, data):
        report["stages"].append({"name": name, **data})
        if not args.json:
            print(f"[OK] {name}", flush=True)

    try:
        # Fail before any Docker mutation if the host driver cannot run.
        try:
            gpus = gpu_snapshot(env)
        except StartupError as exc:
            kernels = installed_nvidia_kernels()
            report["installed_nvidia_kernels"] = kernels
            if kernels and report["kernel"] not in kernels:
                report["next_action"] = (
                    "Reboot into an installed kernel with its NVIDIA module: " + ", ".join(kernels) +
                    ". The currently running kernel has no NVIDIA module.")
            raise exc
        config = json.loads(run(["docker", "compose", "config", "--format", "json"], env=env))
        engine = config["services"]["engine-1"]
        devices = engine["deploy"]["resources"]["reservations"]["devices"]
        identifier = devices[0]["device_ids"][0]
        image = engine["image"]
        stage("host_gpu", {"gpu": selected_gpu(gpus, identifier)})
        run(["docker", "image", "inspect", image], env=env)
        result = run(["docker", "run", "--rm", "-i", "--network", "none", "--read-only",
                      "--tmpfs", "/tmp", "-e", "HOME=/tmp", "--gpus", f"device={identifier}",
                      image, "python", "-"], env=env, timeout=args.timeout, input_text=GPU_SMOKE)
        # CUDA base images may prepend an entrypoint banner to stdout.
        runtime = json.loads(next(line for line in reversed(result.splitlines()) if line.startswith("{")))
        stage("docker_gpu_runtime", runtime)
        if args.check_only:
            report["ready"] = True
            report["scope"] = "Host/Docker GPU runtime only; services were not started."
            return report

        if not args.json:
            print("Starting existing platform services; database volumes and keys are retained.", flush=True)
        run(["docker", "compose", "up", "-d", "--wait", "--wait-timeout", str(args.timeout),
             *PLATFORM_SERVICES], env=env, timeout=args.timeout + 30)
        stage("platform_services", {"services": list(PLATFORM_SERVICES)})

        # Serial starts avoid overlapping TensorRT compilation peaks on an 8 GB GPU.
        for service in GPU_SERVICES:
            if not args.json:
                print(f"Starting {service} and waiting for readiness…", flush=True)
            run(["docker", "compose", "up", "-d", "--no-deps", "--wait", "--wait-timeout",
                 str(args.timeout), service], env=env, timeout=args.timeout + 30)
            endpoint = "8001/healthz" if service.startswith("engine-") else "8002/v1/status"
            payload = json.loads(run(["docker", "compose", "exec", "-T", service, "curl", "-fsS",
                                      f"http://127.0.0.1:{endpoint}"], env=env))
            validate_service(service, payload)
            stage(service, {"health": payload, "gpu": selected_gpu(gpu_snapshot(env), identifier)})

        report["ready"] = True
        report["note"] = ("No face enrollment was performed. Peak inference VRAM under live enrollment must "
                          "be measured before increasing batch sizes or gallery capacity.")
    except (StartupError, KeyError, ValueError, StopIteration) as exc:
        report["error"] = str(exc)
        report["ready"] = False
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Check GPU execution without starting services")
    parser.add_argument("--json", action="store_true", help="Print one JSON report")
    parser.add_argument("--gpu-device-id", help="Override FA_GPU_DEVICE_ID for this run")
    parser.add_argument("--timeout", type=int, default=600, help="Seconds allowed per GPU probe/service startup")
    args = parser.parse_args(argv)
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    report = start(args)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    sys.exit(main())
