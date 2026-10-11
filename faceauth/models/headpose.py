"""Head pose wrapper (docs/IMPLEMENTATION.md §2): 6DRepNet (RepVGG-B1g2) and 6DRepNet360 (ResNet-50).

    hp = HeadPose("data/models/6drepnet360", device="cuda")   # or data/models/6drepnet
    hp.version                         # "6drepnet360@1"
    hp.predict(img_rgb, bboxes)        # (n,3) float32 [yaw, pitch, roll] degrees, §0 convention
    hp.predict_crops(crops_rgb)        # same for already-cropped faces (list of HxWx3 uint8 RGB)

Preprocessing (MODEL.json params.crop), each model's own official code path for detector boxes, RGB, ImageNet
mean/std, torchvision transforms:
  "demo": 6DRepNet sixdrepnet/demo.py - box grows by int(0.2*box height) left/right and int(0.2*box width)
          top/bottom (upstream's width/height swap kept), clipped at the origin; Resize(224) + CenterCrop(224).
  "fsa":  the detector crop behind the published 6DRepNet/6DRepNet360 BIWI numbers (FSA-Net
          TYY_create_db_biwi.py, 6DRepNet README: image size 256) - box grows by 0.4*w / 0.4*h on each side,
          clipped, squashed to 256x256 (cv2 bilinear); Resize(256) + CenterCrop(224) (test.py). 6DRepNet360 has no
          inference/demo code, so this is its only official detector-box pipeline.
The network outputs a rotation matrix; upstream's Euler decomposition gives (pitch, yaw, roll) in the 300W-LP /
AFLW2000 (3DDFA) label convention, where yaw > 0 moves the nose toward image LEFT. §0 wants image RIGHT, so yaw is
negated; pitch (> 0 up) and roll (> 0 clockwise in the image) already match. Checked on AFLW2000 labels/landmarks
and horizontal flips (tests/test_headpose.py, reports/E4_headpose).
Weights load without code execution (torch.load(weights_only=True)) and strictly. File hashes are verified by
the registry before load (§7), not here.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from faceauth.models.vendor.sixdrepnet import ARCHS, euler_from_rotmat

UPSTREAM_TO_SPEC = np.array([-1.0, 1.0, 1.0], np.float32)  # (yaw, pitch, roll) sign flips


def demo_crop(img, bbox) -> np.ndarray:
    """6DRepNet demo.py crop of a detector box [x1,y1,x2,y2] (x2/y2 clip when slicing)."""
    x1, y1, x2, y2 = (int(v) for v in bbox)
    w, h = abs(x2 - x1), abs(y2 - y1)
    return np.ascontiguousarray(img[max(0, y1 - int(0.2 * w)):y2 + int(0.2 * w), max(0, x1 - int(0.2 * h)):x2 + int(0.2 * h)])


def fsa_crop(img, bbox, ad: float = 0.4, size: int = 256) -> np.ndarray:
    """FSA-Net TYY_create_db_biwi.py crop of a detector box, squashed to size x size."""
    h_img, w_img = img.shape[:2]
    x1, y1, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
    xw1, yw1 = max(int(x1 - ad * w), 0), max(int(y1 - ad * h), 0)
    xw2, yw2 = min(int(x1 + w + ad * w), w_img - 1), min(int(y1 + h + ad * h), h_img - 1)
    if xw2 <= xw1 or yw2 <= yw1:
        raise ValueError(f"box {bbox} lies outside the {w_img}x{h_img} image")
    return cv2.resize(np.ascontiguousarray(img[yw1:yw2 + 1, xw1:xw2 + 1]), (size, size))


CROPS = {"demo": (demo_crop, 224), "fsa": (fsa_crop, 256)}  # crop fn, torchvision Resize before CenterCrop(224)


class HeadPose:
    def __init__(self, model_dir, device="cuda", batch_size: int = 64):
        d = Path(model_dir)
        meta = json.loads((d / "MODEL.json").read_text())
        if meta["kind"] != "headpose":
            raise ValueError(f"{d}: kind {meta['kind']!r} is not 'headpose'")
        self.version, self.batch_size, self.device = f"{meta['name']}@{meta['version']}", batch_size, torch.device(device)
        self.net = ARCHS[meta["params"]["arch"]]()
        self.net.load_state_dict(torch.load(d / meta["files"]["weights"], map_location="cpu", weights_only=True))
        self.net.eval().to(self.device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        self.crop = meta["params"].get("crop", "demo")
        if self.crop not in CROPS:
            raise ValueError(f"{d}: unknown params.crop {self.crop!r}")

    def predict(self, img_rgb, bboxes) -> np.ndarray:
        """bboxes (n,4) [x1,y1,x2,y2] px -> (n,3). ValueError if a box lies outside the image (empty crop)."""
        img, boxes = np.asarray(img_rgb), np.asarray(bboxes, np.float64)
        if boxes.size and boxes.shape[-1] != 4:
            raise ValueError(f"bboxes must be (n,4) [x1,y1,x2,y2], got {boxes.shape}")
        crop, resize = CROPS[self.crop]
        return self.predict_crops([crop(img, b) for b in boxes.reshape(-1, 4)], resize)

    @torch.inference_mode()
    def predict_crops(self, crops_rgb, resize: int = 224) -> np.ndarray:
        """Resize(resize) + CenterCrop(224): 224 for 'demo' crops, 256 for 'fsa' crops and the AFLW2000 test crop."""
        if resize == 256 and crops_rgb and all(c.shape == (256, 256, 3) and c.dtype == np.uint8 for c in crops_rgb):
            # fsa crops are already 256x256: Resize(256) is the identity, so CenterCrop(224) + ToTensor + Normalize is
            # an exact slice + scale; batched on the device instead of per-crop PIL (E9: 1.2 -> ~0.3 ms/face)
            x = torch.from_numpy(np.ascontiguousarray(np.stack(crops_rgb)[:, 16:240, 16:240])).to(self.device)
            x = (x.permute(0, 3, 1, 2).float() / 255 - self.mean) / self.std
            out = [np.zeros((0, 3), np.float32)]
            for i in range(0, len(x), self.batch_size):
                pyr = torch.rad2deg(euler_from_rotmat(self.net(x[i:i + self.batch_size]))).cpu().numpy()
                out.append(pyr[:, [1, 0, 2]] * UPSTREAM_TO_SPEC)
            return np.concatenate(out).astype(np.float32)
        tf = transforms.Compose([transforms.Resize(resize), transforms.CenterCrop(224), transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        out = [np.zeros((0, 3), np.float32)]
        for i in range(0, len(crops_rgb), self.batch_size):
            batch = []
            for c in crops_rgb[i:i + self.batch_size]:
                if c.ndim != 3 or c.shape[2] != 3 or c.dtype != np.uint8 or c.size == 0:
                    raise ValueError(f"face crop must be non-empty HxWx3 uint8 RGB, got {c.shape} {c.dtype}")
                batch.append(tf(Image.fromarray(np.ascontiguousarray(c))))
            pyr = torch.rad2deg(euler_from_rotmat(self.net(torch.stack(batch).to(self.device)))).cpu().numpy()
            out.append(pyr[:, [1, 0, 2]] * UPSTREAM_TO_SPEC)  # (pitch, yaw, roll) -> (yaw, pitch, roll), §0 signs
        return np.concatenate(out).astype(np.float32)
