"""SCRFD wrapper: smoke test on WIDER FACE val images + horizontal-flip consistency.
Skips when the model or the dataset is not present.  pytest tests/test_scrfd.py"""
import os

import cv2
import numpy as np
import pytest

from faceauth.models.scrfd import SCRFD

MODEL = "data/models/scrfd_34g"
IMGS = "data/datasets/widerface/WIDER_val/images/"
CASES = [  # image, GT boxes x,y,w,h (wider_face_val_bbx_gt.txt)
    ("0--Parade/0_Parade_Parade_0_901.jpg", [[7, 399, 201, 342], [283, 680, 164, 208], [712, 351, 292, 374]]),
    ("11--Meeting/11_Meeting_Meeting_11_Meeting_Meeting_11_633.jpg",
     [[464, 240, 186, 214], [708, 378, 148, 174], [178, 494, 162, 132]]),
    ("1--Handshaking/1_Handshaking_Handshaking_1_762.jpg", [[235, 211, 315, 417], [615, 438, 185, 214]]),
]
pytestmark = pytest.mark.skipif(not (os.path.exists(f"{MODEL}/MODEL.json") and os.path.isdir(IMGS)),
                                reason="SCRFD model or WIDER FACE val not present")


@pytest.fixture(scope="module")
def det():
    return SCRFD(MODEL)


def _load(name):
    return cv2.cvtColor(cv2.imread(IMGS + name), cv2.COLOR_BGR2RGB)


def _iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


@pytest.mark.parametrize("name,gts", CASES)
def test_smoke(det, name, gts):
    img = _load(name)
    faces = det.detect(img)
    scores = [f["score"] for f in faces]
    assert scores == sorted(scores, reverse=True)
    for x, y, w, h in gts:  # every annotated face is found
        assert max(_iou(f["bbox"], [x, y, x + w, y + h]) for f in faces) > 0.5
    for f in faces:
        b, k = f["bbox"], f["kps"]
        assert b.shape == (4,) and k.shape == (5, 2)
        assert ((k >= b[:2]) & (k <= b[2:])).all(), "keypoints outside the box"
        assert k[0, 0] < k[1, 0] and k[3, 0] < k[4, 0], "not InsightFace order (image-left first)"
    (batch,) = det.detect_batch([img])
    assert np.allclose([f["bbox"] for f in batch], [f["bbox"] for f in faces])


@pytest.mark.parametrize("name,gts", CASES)
def test_flip_consistency(det, name, gts):
    img = _load(name)
    width = img.shape[1]
    orig = [f for f in det.detect(img) if f["score"] > 0.6]
    flip = det.detect(np.ascontiguousarray(img[:, ::-1]), thresh=0.3)
    assert orig
    for f in orig:
        back = [[width - g["bbox"][2], g["bbox"][1], width - g["bbox"][0], g["bbox"][3]] for g in flip]
        ious = [_iou(f["bbox"], b) for b in back]
        g = flip[int(np.argmax(ious))]
        kps = g["kps"][[1, 0, 2, 4, 3]].copy()  # mirroring swaps image-left/right points
        kps[:, 0] = width - kps[:, 0]
        size = np.sqrt((f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))
        assert max(ious) > 0.9
        assert np.abs(kps - f["kps"]).max() < 0.05 * size
        assert abs(g["score"] - f["score"]) < 0.1
