"""Face recognition wrappers (docs/IMPLEMENTATION.md §2).

    rec = Recognizer("data/models/kprpe_vitb_wf12m", device="cuda")
    rec.version, rec.dim, rec.align           # "kprpe_vitb_wf12m@1", 512, "arcface5-112@1"
    rec.embed(imgs_rgb, kps5)                 # raw RGB images + 5 InsightFace-order kps -> (n, dim) float32, L2-normalised
    rec.embed_aligned(crops112_rgb)           # (n,112,112,3) uint8 RGB ArcFace-aligned crops -> (n, dim)

Supported MODEL.json `params.arch`:
  insightface_vit_l_dp005_mask_005  InsightFace arcface_torch ViT-L (WebFace42M, CosFace), state_dict `model.pt`.
  cvlface_vit_base_kprpe            CVLFace KP-RPE ViT-B (AdaFace). KP-RPE also consumes 5 keypoints; like the official
                                    eval pipeline for pre-aligned images (InferAlignerKeypointModelPipeline) they are
                                    predicted on the aligned crop by the bundled DFA aligner (files.aligner).
  cvlface_ir101                     CVLFace AdaFace IR-101 (IResNet-101 as in the AdaFace repo), RGB.
  insightface_onnx                  InsightFace model-zoo ArcFace ONNX (buffalo_l w600k_r50, antelopev2 glintr100), run by
                                    ONNX Runtime CUDA EP in fp32 (TF32 convs, ORT default), files.onnx; input
                                    (x - params.input_mean) / params.input_std (insightface ArcFaceONNX: 127.5 / 127.5
                                    when the graph has no Sub/Mul preprocessing), dynamic batch. Construction fails
                                    instead of silently falling back to CPU (session.disable_cpu_ep_fallback).
All: crop = faceauth.common.align.norm_crop (ArcFace 112 template), RGB; torch archs (x/255 - 0.5) / 0.5.
`params.fp16` (signed with MODEL.json) selects fp16 autocast for torch archs; the DFA aligner always runs fp32.
Weights are loaded without code execution (safetensors, or torch.load(weights_only=True), or ONNX) and strictly
(0 missing / 0 unexpected keys). File hashes are verified by the registry before load (§7), not here.
"""
import json
from pathlib import Path

import numpy as np
import torch

from faceauth.common.align import norm_crop


def _state_dict(path: Path, prefix: str = "") -> dict:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        sd = load_file(str(path))
    else:
        sd = torch.load(path, map_location="cpu", weights_only=True)
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def _ort_session(path: Path, device: torch.device):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3  # zoo exports declare output (1, 512) with a dynamic batch: ORT warns on every run
    if device.type == "cuda":
        ort.preload_dlls()  # faceauth:dev: cuDNN is only on the loader path via the nvidia pip packages
        so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")  # a CPU-placed node fails the load
        # kSameAsRequested: the default power-of-two arena growth held ~5.7 GB for IResNet-50 at batch 256 (shared GPU)
        opts = {"arena_extend_strategy": "kSameAsRequested", **({"device_id": device.index} if device.index else {})}
        want = [("CUDAExecutionProvider", opts)]
    else:
        want = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(path), so, providers=want)
    name = want[0][0] if isinstance(want[0], tuple) else want[0]
    if sess.get_providers()[0] != name:  # ORT silently falls back when an EP fails to load
        raise RuntimeError(f"{path}: wanted {name}, ONNX Runtime gave {sess.get_providers()}")
    return sess


class Recognizer:
    def __init__(self, model_dir, device="cuda", batch_size: int = 128):
        d = Path(model_dir)
        meta = json.loads((d / "MODEL.json").read_text())
        if meta["kind"] != "recognition":
            raise ValueError(f"{d}: kind {meta['kind']!r} is not 'recognition'")
        p, files = meta["params"], meta["files"]
        self.version, self.dim, self.align = f"{meta['name']}@{meta['version']}", int(meta["dim"]), p["align"]
        self.fp16, self.batch_size, self.device = bool(p["fp16"]), batch_size, torch.device(device)
        self.backend, self.sess = "torch", None
        if p["arch"] == "insightface_onnx":
            self.backend, self.fp16 = "onnx", False
            self.sess = _ort_session(d / files["onnx"], self.device)
            self.providers = self.sess.get_providers()
            self.input_name = self.sess.get_inputs()[0].name
            self.mean, self.std = float(p["input_mean"]), float(p["input_std"])
            if self.sess.get_outputs()[0].shape[-1] != self.dim:
                raise ValueError(f"{d}: output {self.sess.get_outputs()[0].shape} does not match dim {self.dim}")
            return
        if p["arch"] == "insightface_vit_l_dp005_mask_005":
            from faceauth.models.vendor.arcface_vit import vit_l_dp005_mask_005
            self.net, self.aligner = vit_l_dp005_mask_005(self.dim), None
            self.net.load_state_dict(_state_dict(d / files["weights"]))
        elif p["arch"] == "cvlface_vit_base_kprpe":
            from faceauth.models.vendor.cvlface_dfa import DFALandmarks
            from faceauth.models.vendor.cvlface_kprpe import vit_base_kprpe
            self.net, self.aligner = vit_base_kprpe(self.dim), DFALandmarks()
            self.net.load_state_dict(_state_dict(d / files["weights"], "model.net."))
            self.aligner.load_state_dict(_state_dict(d / files["aligner"], "model.net."))
            self.aligner.eval().to(self.device)
        elif p["arch"] == "cvlface_ir101":
            from faceauth.models.vendor.cvlface_iresnet import ir_101
            self.net, self.aligner = ir_101(self.dim), None
            self.net.load_state_dict(_state_dict(d / files["weights"], "model.net."))
        else:
            raise ValueError(f"{d}: unknown arch {p['arch']!r}")
        self.net.eval().to(self.device)

    @torch.inference_mode()
    def features(self, crops112_rgb) -> np.ndarray:
        """Raw (un-normalised) float32 embeddings of aligned (n,112,112,3) uint8 RGB crops."""
        crops = np.asarray(crops112_rgb)
        if crops.ndim != 4 or crops.shape[1:] != (112, 112, 3) or crops.dtype != np.uint8:
            raise ValueError(f"expected (n,112,112,3) uint8, got {crops.shape} {crops.dtype}")
        out = [np.zeros((0, self.dim), np.float32)]
        if self.sess is not None:
            for i in range(0, len(crops), self.batch_size):
                x = (crops[i:i + self.batch_size].transpose(0, 3, 1, 2).astype(np.float32) - self.mean) / self.std
                out.append(self.sess.run(None, {self.input_name: x})[0].astype(np.float32))
            return np.concatenate(out)
        for i in range(0, len(crops), self.batch_size):
            x = torch.from_numpy(np.ascontiguousarray(crops[i:i + self.batch_size])).to(self.device)
            x = (x.permute(0, 3, 1, 2).float() / 255 - 0.5) / 0.5
            kps = None if self.aligner is None else self.aligner(x)
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.fp16):
                f = self.net(x) if kps is None else self.net(x, kps)
            out.append(f.float().cpu().numpy())
        return np.concatenate(out)

    def embed_aligned(self, crops112_rgb) -> np.ndarray:
        f = self.features(crops112_rgb)
        return f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)

    def embed(self, imgs_rgb, kps5) -> np.ndarray:
        """imgs_rgb[i] (HxWx3 uint8 RGB) with kps5[i] ((5,2) image coords) -> row i."""
        crops = [norm_crop(img, k, 112) for img, k in zip(imgs_rgb, kps5, strict=True)]
        return self.embed_aligned(np.stack(crops) if crops else np.zeros((0, 112, 112, 3), np.uint8))


if __name__ == "__main__":  # python -m faceauth.models.recog <model_dir> [--write-sha256]
    import argparse
    import hashlib

    ap = argparse.ArgumentParser(description="Strict-load a recognition model dir; check (or fill) MODEL.json sha256.")
    ap.add_argument("model_dir")
    ap.add_argument("--write-sha256", action="store_true", help="write computed hashes into MODEL.json")
    a = ap.parse_args()
    d = Path(a.model_dir)
    meta = json.loads((d / "MODEL.json").read_text())
    rec = Recognizer(d, device="cuda" if torch.cuda.is_available() else "cpu")  # raises on missing/unexpected keys
    print(f"{rec.version}: " + (f"ONNX session OK on {rec.providers}" if rec.sess is not None else
                                "strict load OK (0 missing / 0 unexpected keys)") + f", dim {rec.dim}")
    for k, f in meta["files"].items():
        with open(d / f, "rb") as fh:
            h = hashlib.file_digest(fh, "sha256").hexdigest()
        print(f"{k}: {f} sha256 {h} (MODEL.json: {meta['sha256'].get(k)})")
        if meta["sha256"].get(k) not in (None, h):
            raise SystemExit(f"sha256 mismatch for {f}")
        meta["sha256"][k] = h
    if a.write_sha256:
        (d / "MODEL.json").write_text(json.dumps(meta, indent=2) + "\n")
