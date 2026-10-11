"""FaRL ViT-B/16 448px face alignment, WFLW 98 points (facer `farl/wflw/448`, RFP §2, spec §2).

Re-implements the official pre/post-processing without depending on pyfacer:
  * boxes (n,4) [x1,y1,x2,y2] -> FaRL's WFLW train/eval crop (HRNet center/scale box, UpdateCropAndResizeMatrix,
    scale_mu 0.8): the square of side max(w,h) centred on the box fills 0.8*448 px. The box is the extent of the
    98 landmarks (HRNet csv convention); detector boxes have another extent, pass detector keypoints instead.
  * keypoints (n,5,2) -> least-squares similarity onto DST5 (facer get_face_align_matrix procedure). DST5 is the
    mean WFLW-train position of kps5_from_wflw98(GT) inside FaRL's box crop (eval/eval_wflw.py recomputes it).
    facer's own template (_standard_face_pts at target_face_scale 0.8) frames faces ~0.6x smaller than the model
    was trained on and is ~2.5 NME points worse on WFLW (reports/E5_landmarks).
  * `refine` extra passes re-crop from the predicted landmark extent (the training framing again).
  * image/255 -> CLIP mean/std (in the backbone) -> 98 sigmoid heatmaps 112x112 -> soft-argmax (FaRL
    heatmap2points) -> crop px -> inverse similarity -> image px. Decoding happens in crop space as in FaRL's
    training/eval code; pyfacer inverse-warps the heatmaps to full image size first (difference measured in
    reports/E5_landmarks).

Backbone code vendored from facer (MIT, (c) 2022 FacePerceiver), which carries FaRL code (MIT, (c) Microsoft
Corporation): facer/farl/model.py VisualTransformer, facer/face_alignment/network/transformers.py
FaRLVisualFeatures, .../geometry.py heatmap2points. The UPerHead is the official TorchScript file itself
(reviewed: plain tensor ops); its archive also carries the backbone state_dict.
"""
import io
import json
import os
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn as nn
from skimage.transform import SimilarityTransform

SIZE = 448
FACE_SCALE = 0.8  # FaRL scale_mu: landmark extent -> 0.8 * crop
NUM_POINTS = 98
# kps5 template in the 448 crop (left-right symmetrised mean over the 7,500 WFLW train faces, see module doc)
DST5 = np.array([[151.75, 121.29], [295.25, 121.29], [223.50, 208.87], [158.94, 274.31], [288.06, 274.31]])


class _QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class _Block(nn.Module):  # FaRL/CLIP ResidualAttentionBlock, (L, N, D) layout
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads)
        self.ln_1 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(OrderedDict(
            [("c_fc", nn.Linear(d, 4 * d)), ("gelu", _QuickGELU()), ("c_proj", nn.Linear(4 * d, d))]))
        self.ln_2 = nn.LayerNorm(d)

    def forward(self, x):
        h = self.ln_1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.ln_2(x))


class _Backbone(nn.Module):
    """FaRLVisualFeatures('base', forced_input_resolution=448): blocks 3/5/7/11 -> FPN at 1/4, 1/8, 1/16, 1/32."""
    OUT = (3, 5, 7, 11)

    def __init__(self, d: int = 768, layers: int = 12, heads: int = 12, patch: int = 16):
        super().__init__()
        v = self.visual = nn.Module()
        v.conv1 = nn.Conv2d(3, d, patch, patch, bias=False)
        v.class_embedding = nn.Parameter(torch.zeros(d))
        v.positional_embedding = nn.Parameter(torch.zeros((SIZE // patch) ** 2 + 1, d))
        v.ln_pre = nn.LayerNorm(d)
        v.transformer = nn.Module()
        v.transformer.resblocks = nn.ModuleList(_Block(d, heads) for _ in range(layers))
        self.register_buffer("image_mean", torch.zeros(1, 3, 1, 1))
        self.register_buffer("image_std", torch.ones(1, 3, 1, 1))
        self.fpns = nn.ModuleList([
            nn.Sequential(nn.ConvTranspose2d(d, d, 2, 2), nn.BatchNorm2d(d), nn.GELU(), nn.ConvTranspose2d(d, d, 2, 2)),
            nn.ConvTranspose2d(d, d, 2, 2), nn.Identity(), nn.MaxPool2d(2, 2)])

    def forward(self, x):  # (N,3,448,448) RGB in [0,1]
        v = self.visual
        x = v.conv1((x - self.image_mean) / self.image_std)
        n, d, s, _ = x.shape
        x = torch.cat([v.class_embedding.expand(n, 1, d), x.flatten(2).transpose(1, 2)], 1) + v.positional_embedding
        x = v.ln_pre(x).permute(1, 0, 2)
        feats = []
        for i, blk in enumerate(v.transformer.resblocks[: self.OUT[-1] + 1]):
            x = blk(x)
            if i in self.OUT:
                feats.append(x[1:].permute(1, 2, 0).reshape(n, d, s, s))
        return [fpn(f) for fpn, f in zip(self.fpns, feats)]


def crop_matrices(bboxes_or_kps) -> np.ndarray:
    """(n,4) boxes or (n,5,2) keypoints -> (n,2,3) affine image px -> 448 crop px (pixel centres at integers)."""
    a = np.asarray(bboxes_or_kps, np.float64)
    if a.ndim == 3 and a.shape[1:] == (5, 2):
        ms = []
        for k in a:
            t = SimilarityTransform.from_estimate(k, DST5)
            if not t:
                raise ValueError("degenerate keypoints")
            ms.append(t.params[:2])
        return np.array(ms, np.float32).reshape(-1, 2, 3)
    if a.ndim == 2 and a.shape[1] == 4:
        s = FACE_SCALE * SIZE / np.maximum(a[:, 2] - a[:, 0], a[:, 3] - a[:, 1])
        m = np.zeros((len(a), 2, 3))
        m[:, 0, 0] = m[:, 1, 1] = s
        m[:, :, 2] = (SIZE - 1) / 2.0 - s[:, None] * (a[:, :2] + a[:, 2:]) / 2.0
        return m.astype(np.float32)
    raise ValueError(f"expected (n,4) boxes or (n,5,2) keypoints, got shape {a.shape}")


def soft_argmax(hm: torch.Tensor) -> torch.Tensor:
    """(N,K,h,w) activated heatmaps -> (N,K,2) crop px: FaRL heatmap2points, then denormalised to SIZE."""
    n, k, h, w = hm.shape
    s = hm.sum((2, 3)).clamp(min=1e-6)
    x = (hm.sum(2) * torch.arange(w, device=hm.device, dtype=hm.dtype)).sum(-1) / s
    y = (hm.sum(3) * torch.arange(h, device=hm.device, dtype=hm.dtype)).sum(-1) / s
    return (torch.stack([x, y], -1) + 0.5) * torch.tensor([SIZE / w, SIZE / h], device=hm.device) - 0.5


class FaRLLandmarks:
    def __init__(self, model_dir, device="cuda", batch_size: int = 8, bf16: bool = False):
        """bf16: run backbone + head in bfloat16 (~1.7x faster on the A6000, WFLW NME unchanged to 0.01; fp16
        overflows in the head)."""
        meta = json.load(open(os.path.join(model_dir, "MODEL.json")))
        self.dtype = torch.bfloat16 if bf16 else torch.float32
        extra = {"backbone": ""}
        self.head = torch.jit.load(os.path.join(model_dir, meta["files"]["jit"]), map_location="cpu",
                                   _extra_files=extra).eval().to(device).to(self.dtype)
        sd = torch.load(io.BytesIO(extra["backbone"]), map_location="cpu", weights_only=True)
        self.backbone = _Backbone()  # ln_post/proj (CLIP projection) are unused by the alignment model
        self.backbone.load_state_dict({k: v for k, v in sd.items() if not k.startswith(("visual.ln_post.", "visual.proj"))})
        self.backbone.eval().to(device, self.dtype)
        self.device, self.batch_size = device, batch_size
        self.version = f"{meta['name']}@{meta['version']}"

    @torch.inference_mode()
    def infer_crops(self, crops: np.ndarray) -> np.ndarray:
        """(N,448,448,3) float32 RGB 0..255 crops -> (N,98,2) crop px."""
        out = np.zeros((len(crops), NUM_POINTS, 2), np.float32)
        for i in range(0, len(crops), self.batch_size):
            x = torch.from_numpy(crops[i:i + self.batch_size]).to(self.device).permute(0, 3, 1, 2) / 255.0
            hm = self.head(self.backbone(x.to(self.dtype))).float()
            out[i:i + len(x)] = soft_argmax(torch.sigmoid(hm)).cpu().numpy()
        return out

    def predict_batch(self, imgs_rgb, bboxes_or_kps_list, refine: int = 1) -> list:
        """Per image: RGB uint8 HxWx3 + (n,4) boxes or (n,5,2) kps -> list of (n,98,2) image px.
        `refine` extra passes each re-crop from the previous pass's landmark extent (1 pass ~ +1x compute)."""
        out = self._predict(imgs_rgb, bboxes_or_kps_list)
        for _ in range(refine):
            out = self._predict(imgs_rgb, [np.concatenate([p.min(1), p.max(1)], 1) for p in out])
        return out

    def predict(self, img_rgb, bboxes_or_kps, refine: int = 1) -> np.ndarray:
        return self.predict_batch([img_rgb], [bboxes_or_kps], refine)[0]

    def _predict(self, imgs_rgb, bboxes_or_kps_list) -> list:
        mats, crops = [], []
        for img, b in zip(imgs_rgb, bboxes_or_kps_list):
            m = crop_matrices(b)
            if len(m):
                f = np.asarray(img, np.float32)
                crops += [cv2.warpAffine(f, mi, (SIZE, SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=0) for mi in m]
            mats.append(m)
        pts = self.infer_crops(np.stack(crops)) if crops else np.zeros((0, NUM_POINTS, 2), np.float32)
        m = np.concatenate(mats)
        inv = np.array([cv2.invertAffineTransform(mi) for mi in m], np.float32).reshape(-1, 2, 3)
        pts = np.einsum("nij,nkj->nki", inv[:, :, :2], pts) + inv[:, None, :, 2]
        return np.split(pts, np.cumsum([len(mi) for mi in mats])[:-1])


def kps5_from_wflw98(pts) -> np.ndarray:
    """WFLW-98 (98,2) or (n,98,2) -> (5,2)/(n,5,2) InsightFace order: image-left eye centre (mean of contour 60-67),
    image-right eye centre (68-75), nose tip 54, image-left mouth corner 76, image-right mouth corner 82."""
    p = np.asarray(pts, np.float32)
    return np.stack([p[..., 60:68, :].mean(-2), p[..., 68:76, :].mean(-2), p[..., 54, :], p[..., 76, :], p[..., 82, :]], -2)
