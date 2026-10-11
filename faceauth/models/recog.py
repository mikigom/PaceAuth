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
Both: crop = faceauth.common.align.norm_crop (ArcFace 112 template), RGB, (x/255 - 0.5) / 0.5.
`params.fp16` (signed with MODEL.json) selects fp16 autocast; the DFA aligner always runs fp32.
Weights are loaded without code execution (safetensors, or torch.load(weights_only=True)) and strictly
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


class Recognizer:
    def __init__(self, model_dir, device="cuda", batch_size: int = 128):
        d = Path(model_dir)
        meta = json.loads((d / "MODEL.json").read_text())
        if meta["kind"] != "recognition":
            raise ValueError(f"{d}: kind {meta['kind']!r} is not 'recognition'")
        p, files = meta["params"], meta["files"]
        self.version, self.dim, self.align = f"{meta['name']}@{meta['version']}", int(meta["dim"]), p["align"]
        self.fp16, self.batch_size, self.device = bool(p["fp16"]), batch_size, torch.device(device)
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
    print(f"{rec.version}: strict load OK (0 missing / 0 unexpected keys), dim {rec.dim}")
    for k, f in meta["files"].items():
        with open(d / f, "rb") as fh:
            h = hashlib.file_digest(fh, "sha256").hexdigest()
        print(f"{k}: {f} sha256 {h} (MODEL.json: {meta['sha256'].get(k)})")
        if meta["sha256"].get(k) not in (None, h):
            raise SystemExit(f"sha256 mismatch for {f}")
        meta["sha256"][k] = h
    if a.write_sha256:
        (d / "MODEL.json").write_text(json.dumps(meta, indent=2) + "\n")
