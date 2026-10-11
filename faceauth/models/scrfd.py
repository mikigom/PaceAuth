"""SCRFD face detector with 5 keypoints (ONNX Runtime).

Pre/post-processing is a port of InsightFace `python-package/insightface/model_zoo/scrfd.py`
(Jia Guo, insightface.ai; InsightFace code is MIT licensed): top-left letterbox to a square input,
(x - 127.5) / 128 RGB blob, strides 8/16/32 with 2 anchors per location, distance decoding of boxes and
keypoints, greedy NMS with the "+1" area convention, optional `max_num` selection (area minus 2x squared
distance to the image centre). Differences: input is RGB (InsightFace takes BGR and swaps); boxes and
keypoints are mapped back with per-axis scales (new_w/w, new_h/h) where InsightFace divides both axes by
new_h/h, which is off by up to ~0.3% in x when int() truncates new_h (identical for 640x480 or 16:9 frames;
on WIDER FACE val it cost SCRFD-34G 0.17/0.19/1.25 Easy/Medium/Hard AP); results are dicts sorted by score.
"""
import hashlib
import json
import os

import cv2
import numpy as np
import onnxruntime as ort

STRIDES = (8, 16, 32)
NUM_ANCHORS = 2


def distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """(n,2) anchor centres + (n,4) l,t,r,b distances -> (n,4) x1,y1,x2,y2."""
    return np.stack([points[:, 0] - distance[:, 0], points[:, 1] - distance[:, 1],
                     points[:, 0] + distance[:, 2], points[:, 1] + distance[:, 3]], axis=-1)


def distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """(n,2) anchor centres + (n,10) offsets -> (n,5,2) keypoints."""
    return (np.tile(points, 5) + distance).reshape(-1, 5, 2)


def nms(dets: np.ndarray, thresh: float, offset: float = 1.0) -> np.ndarray:
    """Greedy NMS on (n,5) x1,y1,x2,y2,score; drops boxes with inter > thresh * union (IoU > thresh, written
    like mmcv so zero-area boxes do not give 0/0). offset=1 is InsightFace's pixel-inclusive area convention,
    offset=0 matches mmcv's nms (used by the SCRFD WIDER FACE test)."""
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + offset) * (y2 - y1 + offset)
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        w = np.maximum(0.0, np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]) + offset)
        h = np.maximum(0.0, np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]) + offset)
        inter = w * h
        order = rest[inter <= thresh * (areas[i] + areas[rest] - inter)]
    return np.array(keep, dtype=np.int64)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class SCRFD:
    def __init__(self, model_dir, providers=("CUDAExecutionProvider", "CPUExecutionProvider"), input_size=640):
        with open(os.path.join(model_dir, "MODEL.json")) as f:
            meta = json.load(f)
        path = os.path.join(model_dir, meta["files"]["onnx"])
        if _sha256(path) != meta["sha256"]["onnx"]:
            raise ValueError(f"sha256 mismatch for {path}")
        if "CUDAExecutionProvider" in providers:
            ort.preload_dlls()  # faceauth:dev: cuDNN is only on the loader path via the nvidia pip packages
        self.session = ort.InferenceSession(path, providers=list(providers))
        inp = self.session.get_inputs()[0]
        outs = self.session.get_outputs()
        if len(outs) != 9 or len(outs[0].shape) != 2:
            raise ValueError("expected an unbatched 3-stride, 2-anchor SCRFD export with keypoints (9 outputs)")
        if any(isinstance(d, int) and d != input_size for d in inp.shape[2:]):
            raise ValueError(f"model input is fixed at {inp.shape[2:]}, not {input_size}")
        self.version = f"{meta['name']}@{meta['version']}"
        self.input_name = inp.name
        self.output_names = [o.name for o in outs]
        self.input_size = input_size
        self.nms_thresh = 0.4
        # anchor centres (x, y) per stride, each repeated NUM_ANCHORS times
        self.centers = {}
        for s in STRIDES:
            n = input_size // s
            c = np.stack(np.mgrid[:n, :n][::-1], axis=-1).astype(np.float32).reshape(-1, 2) * s
            self.centers[s] = np.repeat(c, NUM_ANCHORS, axis=0)

    def blob(self, img_rgb: np.ndarray):
        """InsightFace letterbox: resize keeping aspect into the top-left of a black square.
        -> ((1,3,S,S) float32 blob, (x, y) scale from image to blob pixels)."""
        size = self.input_size
        im_ratio = float(img_rgb.shape[0]) / img_rgb.shape[1]
        if im_ratio > 1.0:
            new_h = size
            new_w = int(new_h / im_ratio)
        else:
            new_w = size
            new_h = int(new_w * im_ratio)
        scale = np.array([new_w / img_rgb.shape[1], new_h / img_rgb.shape[0]], dtype=np.float32)
        det_img = np.zeros((size, size, 3), dtype=np.uint8)
        det_img[:new_h, :new_w] = cv2.resize(img_rgb, (new_w, new_h))
        blob = (det_img.astype(np.float32) - 127.5) / 128.0
        return blob.transpose(2, 0, 1)[None], scale

    def forward(self, blob: np.ndarray, thresh: float):
        """(1,3,S,S) blob -> (scores (n,), bboxes (n,4), kps (n,5,2)) in blob pixels for scores >= thresh,
        concatenated over strides 8, 16, 32 (not sorted)."""
        outs = self.session.run(self.output_names, {self.input_name: blob})
        scores, bboxes, kpss = [], [], []
        for i, s in enumerate(STRIDES):
            sc = outs[i][:, 0]
            pos = np.where(sc >= thresh)[0]
            c = self.centers[s][pos]
            scores.append(sc[pos])
            bboxes.append(distance2bbox(c, outs[i + 3][pos] * s))
            kpss.append(distance2kps(c, outs[i + 6][pos] * s))
        return np.concatenate(scores), np.concatenate(bboxes), np.concatenate(kpss)

    def detect(self, img_rgb: np.ndarray, thresh: float = 0.5, max_num: int = 0) -> list[dict]:
        blob, scale = self.blob(img_rgb)
        scores, bboxes, kpss = self.forward(blob, thresh)
        order = np.argsort(-scores, kind="stable")
        det = np.hstack((bboxes / np.tile(scale, 2), scores[:, None])).astype(np.float32, copy=False)[order]
        kpss = (kpss / scale)[order]
        keep = nms(det, self.nms_thresh)
        det, kpss = det[keep], kpss[keep]
        if 0 < max_num < det.shape[0]:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            cy, cx = img_rgb.shape[0] // 2, img_rgb.shape[1] // 2
            off2 = ((det[:, 0] + det[:, 2]) / 2 - cx) ** 2 + ((det[:, 1] + det[:, 3]) / 2 - cy) ** 2
            sel = np.argsort(area - off2 * 2.0)[::-1][:max_num]
            sel = sel[np.argsort(-det[sel, 4], kind="stable")]
            det, kpss = det[sel], kpss[sel]
        return [{"bbox": d[:4].copy(), "score": float(d[4]), "kps": k.astype(np.float32)} for d, k in zip(det, kpss)]

    def detect_batch(self, imgs_rgb, thresh: float = 0.5) -> list[list[dict]]:
        # ponytail: loop, the export has a static batch of 1. A batch-dynamic re-export (GN Reshape/Resize
        # constants) measured only ~1.13x throughput at batch 16 on an A6000, so not worth a derived file yet.
        return [self.detect(img, thresh) for img in imgs_rgb]
