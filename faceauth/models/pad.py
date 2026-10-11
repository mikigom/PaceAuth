"""RGB + depth + IR presentation attack detection (ISO/IEC 30107 PAD subsystem).

Decision = CNN live probability >= threshold AND interpretable physics checks pass. Checks on the aligned
face crop (depth in mm, IR in DN):
  valid_ratio   share of valid depth in the face core; below `min_valid_ratio` -> no decision (non-response)
  relief_mm     RMS residual of the face-core depth after removing the best quadratic surface
                (flat or bent paper and screens have ~0 relief; faces have nose / eye sockets / mouth)
  ir_median     median IR in the face core (displays emit and reflect almost no 850 nm)
  shading_corr  correlation of IR with the Lambertian shading predicted from depth normals
`score()` returns the CNN probability gated by the checks (0.0 if any check fails or no decision is
possible), so `score >= threshold` is the full decision; `predict()` returns the details.

Trained and calibrated on SIMULATED data only (faceauth/sim): validates the pipeline, not real-world PAD
performance. Real-sensor training/validation is deferred to the hardware phase.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from faceauth.common.align import estimate_norm

SIZE, ZOOM = 128, 1.6  # crop size; zoom-out around the ArcFace template (keeps head outline / PAI edges)
DEPTH_RANGE_MM = 100.0  # relative depth is clipped to +-100 mm around the face median
CORE_TEMPLATE = (22.0, 30.0, 90.0, 108.0)  # face core x0,y0,x1,y1 in ArcFace-112 template coords


def core_box(size=SIZE, zoom=ZOOM):
    c = size / 2
    x0, y0, x1, y1 = (c + (np.array(CORE_TEMPLATE) * size / 112 - c) / zoom).round().astype(int)
    return y0, y1, x0, x1


def crop_affine(kps, size=SIZE, zoom=ZOOM) -> np.ndarray:
    """2x3 image -> crop: the ArcFace alignment (faceauth.common.align) zoomed out by `zoom` about the centre."""
    M = np.vstack([estimate_norm(np.asarray(kps, np.float32).reshape(5, 2), size), [0, 0, 1]])
    a, c = 1 / zoom, size / 2
    return (np.array([[a, 0, c * (1 - a)], [0, a, c * (1 - a)], [0, 0, 1]]) @ M)[:2].astype(np.float32)


def crops(rgb, depth, ir, kps, size=SIZE, zoom=ZOOM):
    """-> aligned rgb uint8 (S,S,3), depth uint16 (S,S) (nearest, 0 = invalid), ir uint8 (S,S), affine M."""
    M = crop_affine(kps, size, zoom)
    return (cv2.warpAffine(rgb, M, (size, size), flags=cv2.INTER_LINEAR),
            cv2.warpAffine(depth, M, (size, size), flags=cv2.INTER_NEAREST),
            cv2.warpAffine(ir, M, (size, size), flags=cv2.INTER_LINEAR), M)


def to_input(rgb_c, d_c, ir_c, box=None) -> np.ndarray:
    """-> (6,S,S) float32: RGB (3), depth relative to the face-core median / 100 mm (1), depth valid (1), IR (1)."""
    y0, y1, x0, x1 = box or core_box(rgb_c.shape[0])
    v = d_c > 0
    core = d_c[y0:y1, x0:x1]
    med = np.median(core[core > 0]) if (core > 0).any() else (np.median(d_c[v]) if v.any() else 0.0)
    rel = np.where(v, np.clip((d_c.astype(np.float32) - med) / DEPTH_RANGE_MM, -1, 1), 0)
    x = np.concatenate([(rgb_c.transpose(2, 0, 1) / 255.0 - 0.5) / 0.25, rel[None], v[None],
                        (ir_c[None] / 255.0 - 0.5) / 0.25])
    return x.astype(np.float32)


def physics(d_c, ir_c, M, fx, box=None) -> dict:
    """Interpretable checks on aligned depth (mm) / IR crops; M = crop affine, fx = depth focal length (px)."""
    y0, y1, x0, x1 = box or core_box(d_c.shape[0])
    d = d_c[y0:y1, x0:x1].astype(np.float32)
    ir = ir_c[y0:y1, x0:x1].astype(np.float32)
    out = dict(valid_ratio=float((d > 0).mean()), ir_median=float(np.median(ir)), relief_mm=float("nan"),
               shading_corr=float("nan"))
    if out["valid_ratio"] < 0.05:
        return out
    v = (d > 0) & (np.abs(d - np.median(d[d > 0])) < 120)  # face surface only (drop background behind the head)
    # relief: 4x4 block medians (averages sensor noise), best quadratic surface, RMS residual
    gh, gw = d.shape[0] // 4, d.shape[1] // 4
    blk = np.where(v, d, np.nan)[:gh * 4, :gw * 4].reshape(gh, 4, gw, 4).transpose(0, 2, 1, 3).reshape(gh, gw, 16)
    ok = np.isfinite(blk).sum(-1) >= 8
    if ok.sum() >= 12:
        zb = np.nanmedian(np.where(ok[..., None], blk, 0), -1)[ok]
        yy, xx = np.nonzero(ok)
        xx, yy = (xx - gw / 2) / gw, (yy - gh / 2) / gh
        A = np.stack([np.ones_like(xx), xx, yy, xx * xx, xx * yy, yy * yy], 1)
        res = zb - A @ np.linalg.lstsq(A, zb, rcond=None)[0]
        out["relief_mm"] = float(np.sqrt(np.mean(res ** 2)))
    # IR shading vs Lambertian shading predicted from depth normals (co-located illuminator)
    w = cv2.GaussianBlur(v.astype(np.float32), (0, 0), 2)
    zf = cv2.GaussianBlur(np.where(v, d, 0), (0, 0), 2) / np.maximum(w, 1e-3)
    mm_px = float(np.median(d[v])) / fx / np.sqrt(abs(np.linalg.det(M[:, :2])))
    gy, gx = np.gradient(zf / mm_px)
    shade = 1 / np.sqrt(1 + gx ** 2 + gy ** 2)
    m = w > 0.5
    a, b = shade[m], cv2.GaussianBlur(ir, (0, 0), 2)[m]
    if m.sum() > 50 and a.std() > 1e-6 and b.std() > 1e-6:
        out["shading_corr"] = float(np.corrcoef(a, b)[0, 1])
    return out


def face_features(rgb, depth, ir, kps, size=SIZE, zoom=ZOOM, fx=385.0):
    """One face -> (CNN input (6,S,S), physics measurements)."""
    rc, dc, ic, M = crops(rgb, depth, ir, kps, size, zoom)
    box = core_box(size, zoom)
    return to_input(rc, dc, ic, box), physics(dc, ic, M, fx, box)


def _cbr(i, o, s=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class PADNet(nn.Module):
    """Modality stems (RGB | depth+valid | IR) -> mid-level fusion -> live logit. Input (B,6,128,128)."""

    def __init__(self, w=32):
        super().__init__()
        stem = lambda c: nn.Sequential(_cbr(c, 16, 2), _cbr(16, w), _cbr(w, w, 2))  # noqa: E731
        self.rgb, self.dep, self.ir = stem(3), stem(2), stem(1)
        self.fuse = nn.Sequential(_cbr(3 * w, 128, 2), _cbr(128, 128), _cbr(128, 192, 2), _cbr(192, 192),
                                  nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(192, 1))

    def forward(self, x):
        return self.fuse(torch.cat([self.rgb(x[:, :3]), self.dep(x[:, 3:5]), self.ir(x[:, 5:])], 1)).squeeze(1)


def check_failures(ph: dict, checks: dict) -> list:
    """Names of failed physics checks (NaN measurements fail)."""
    lim = {"relief_mm": "min_relief_mm", "ir_median": "min_ir_median", "shading_corr": "min_shading_corr"}
    return [k for k, c in lim.items() if not ph[k] >= checks[c]]


class PAD:
    """pad = PAD("data/models/pad_rgbdir"); pad.score(rgb, depth, ir, kps5) -> (n,) gated live scores.
    Inputs: one frame (rgb HxWx3 uint8, depth HxW uint16 mm, ir HxW uint8) with kps5 (5,2) or (n,5,2), or
    lists of frames with one (5,2) per frame."""

    def __init__(self, model_dir, device="cuda"):
        d = Path(model_dir)
        meta = json.loads((d / "MODEL.json").read_text())
        if meta.get("kind") != "pad":
            raise ValueError("not a PAD model dir")
        prm = meta["params"]
        self.version = f'{meta["name"]}@{meta["version"]}'
        self.threshold = float(prm["threshold"])
        self.checks = prm["checks"]
        self.fx = float(prm["fx"])
        self.size, self.zoom = int(prm["input"]["size"]), float(prm["input"]["zoom"])
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        self.net = PADNet()
        self.net.load_state_dict(torch.load(d / meta["files"]["weights"], map_location="cpu", weights_only=True))
        self.net.to(self.device).eval()

    @staticmethod
    def _faces(rgb, depth, ir, kps5):
        if isinstance(rgb, np.ndarray) and rgb.ndim == 3:
            return [(rgb, depth, ir, k) for k in np.asarray(kps5, np.float32).reshape(-1, 5, 2)]
        if not (len(rgb) == len(depth) == len(ir) == len(kps5)):
            raise ValueError("frame lists and kps5 must have equal length")
        return [(r, d, i, np.asarray(k, np.float32).reshape(5, 2)) for r, d, i, k in zip(rgb, depth, ir, kps5)]

    def features(self, rgb, depth, ir, kps5):
        """-> CNN inputs (n,6,S,S) and physics measurements per face."""
        f = [face_features(r, d, i, k, self.size, self.zoom, self.fx) for r, d, i, k in self._faces(rgb, depth, ir, kps5)]
        return np.stack([x for x, _ in f]) if f else np.zeros((0, 6, self.size, self.size), np.float32), [p for _, p in f]

    @torch.no_grad()
    def cnn(self, x: np.ndarray) -> np.ndarray:
        out = [torch.sigmoid(self.net(torch.from_numpy(x[i:i + 256]).to(self.device))).float().cpu().numpy()
               for i in range(0, len(x), 256)]
        return np.concatenate(out) if out else np.zeros(0, np.float32)

    def decide(self, p: float, c: dict) -> dict:
        """CNN probability + physics measurements -> {"score", "cnn", "live", "nonresponse", "failed", "checks"}."""
        nr = not c["valid_ratio"] >= self.checks["min_valid_ratio"]
        failed = check_failures(c, self.checks)
        gated = float(p) if not nr and not failed else 0.0
        return dict(score=gated, cnn=float(p), live=bool(gated >= self.threshold and gated > 0), nonresponse=bool(nr),
                    failed=failed, checks=c)

    def predict(self, rgb, depth, ir, kps5) -> list:
        x, ph = self.features(rgb, depth, ir, kps5)
        return [self.decide(p, c) for p, c in zip(self.cnn(x), ph)]

    def score(self, rgb, depth, ir, kps5) -> np.ndarray:
        return np.array([o["score"] for o in self.predict(rgb, depth, ir, kps5)], np.float32)
