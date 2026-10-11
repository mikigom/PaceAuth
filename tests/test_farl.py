"""FaRL landmarks wrapper: geometry (always), WFLW accuracy (model + data present), pyfacer parity (facer importable).

Plain run:   docker run --rm --gpus '"device=5"' -u $(id -u):$(id -g) -e HOME=/tmp -v $PWD:/app -w /app faceauth:dev \
                 python -m pytest -q tests/test_farl.py
pyfacer parity (throwaway container, facer source at /facer = github.com/FacePerceiver/facer@ddd35c7):
  ... -v <facer clone>:/facer:ro -e PYTHONPATH=/app:/facer faceauth:dev \
      bash -c "pip install --user -q validators && python -m pytest -q -s tests/test_farl.py -k pyfacer"
"""
import json
import os

import cv2
import numpy as np
import pytest
import torch

import faceauth.models.farl as farl

MODEL_DIR = "data/models/farl_wflw_448"
WFLW = "data/datasets/wflw"
FACER_DST5 = ((np.array([[196.0, 226.0], [316.0, 226.0], [256.0, 286.0], [220.0, 360.4], [292.0, 360.4]]) / 256.0 - 1.0)
              * 0.8 + 1.0) * 447 / 2.0
have_data = os.path.exists(f"{MODEL_DIR}/MODEL.json") and os.path.exists(f"{WFLW}/WFLW_images")


def test_geometry():
    pts = np.zeros((98, 2), np.float32)
    pts[60:68], pts[68:76], pts[54], pts[76], pts[82] = [10, 20], [50, 20], [30, 40], [15, 60], [45, 60]
    assert np.allclose(farl.kps5_from_wflw98(pts), [[10, 20], [50, 20], [30, 40], [15, 60], [45, 60]])
    assert farl.kps5_from_wflw98(np.stack([pts, pts])).shape == (2, 5, 2)
    a, s = np.deg2rad(20), 0.5  # kps = similarity(DST5) -> crop maps them back onto DST5
    kps = s * farl.DST5 @ np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]]) + [100, 70]
    m = farl.crop_matrices(kps[None])[0]
    assert np.allclose(kps @ m[:, :2].T + m[:, 2], farl.DST5, atol=1e-3)
    m = farl.crop_matrices([[100, 50, 300, 200]])[0]  # square side 200 -> 0.8*448 px, centre -> crop centre
    assert np.allclose(m, [[358.4 / 200, 0, 223.5 - 1.792 * 200], [0, 358.4 / 200, 223.5 - 1.792 * 125]], atol=1e-4)
    assert farl.crop_matrices(np.zeros((0, 5, 2))).shape == (0, 2, 3)
    hm = torch.zeros(1, 1, 112, 112)
    hm[0, 0, 20, 10] = 1  # heatmap px (x=10, y=20) -> crop px ((p + 0.5) * 4 - 0.5)
    assert torch.allclose(farl.soft_argmax(hm), torch.tensor([[[41.5, 81.5]]]))


def _wflw_faces(n, max_px=2e6):
    """First n WFLW test faces in images of <= max_px pixels (pyfacer's full-image heatmaps: 98 x H x W floats)."""
    imgs, gt = [], []
    for r in map(str.split, open(f"{WFLW}/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt")):
        img = cv2.imread(f"{WFLW}/WFLW_images/{r[206]}")
        if img.shape[0] * img.shape[1] <= max_px:
            imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            gt.append(np.array(r[:196], np.float64).reshape(98, 2))
        if len(imgs) == n:
            return imgs, np.array(gt)


def _nme(p, gt):
    return np.linalg.norm(p - gt, axis=-1).mean(-1) / np.linalg.norm(gt[:, 60] - gt[:, 72], axis=-1)


@pytest.mark.skipif(not have_data, reason="model or WFLW missing")
def test_wflw_accuracy():
    imgs, gt = _wflw_faces(32)
    m = farl.FaRLLandmarks(MODEL_DIR, device="cuda" if torch.cuda.is_available() else "cpu")
    boxes = [np.r_[np.floor(g.min(0)), np.ceil(g.max(0))][None] for g in gt]  # (crop squares it around the centre)
    p_box = np.concatenate(m.predict_batch(imgs, boxes, refine=0))
    p_kps = np.concatenate(m.predict_batch(imgs, [farl.kps5_from_wflw98(g)[None] for g in gt]))
    assert p_box.shape == (32, 98, 2)
    assert _nme(p_box, gt).mean() < 0.045 and _nme(p_kps, gt).mean() < 0.05, (_nme(p_box, gt).mean(), _nme(p_kps, gt).mean())


@pytest.mark.skipif(not have_data, reason="model or WFLW missing")
def test_pyfacer_parity(monkeypatch):
    """Same model file through pyfacer's FaRLFaceAlignment vs our kps path with pyfacer's template, 1 pass."""
    pytest.importorskip("facer")
    from facer.face_alignment import FaRLFaceAlignment
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    imgs, gt = _wflw_faces(48)
    kps = farl.kps5_from_wflw98(gt)
    ref = FaRLFaceAlignment("wflw/448", model_path=f"{MODEL_DIR}/face_alignment.farl.wflw.main_ema_jit.pt", device=dev)
    p_ref = []
    with torch.inference_mode():
        for img, k in zip(imgs, kps):
            data = {"image_ids": torch.zeros(1, dtype=torch.long, device=dev),
                    "points": torch.from_numpy(k[None]).float().to(dev)}
            p_ref.append(ref(torch.from_numpy(img).permute(2, 0, 1)[None].to(dev), data)["alignment"].cpu().numpy()[0])
    p_ref = np.array(p_ref)
    monkeypatch.setattr(farl, "DST5", FACER_DST5)
    p = np.concatenate(farl.FaRLLandmarks(MODEL_DIR, device=dev).predict_batch(imgs, [k[None] for k in kps], refine=0))
    io = np.linalg.norm(gt[:, 60] - gt[:, 72], axis=-1)
    d = np.linalg.norm(p - p_ref, axis=-1)
    # pyfacer soft-argmaxes over the image area only, so it cannot place points outside the image (WFLW has such GT)
    inside = np.array([((g >= 0) & (g < img.shape[1::-1])).all() for g, img in zip(gt, imgs)])
    res = {"n_faces": len(gt), "n_faces_all_gt_inside_image": int(inside.sum()), "facer_commit": "ddd35c7",
           "mean_px": float(d.mean()), "p99_px": float(np.percentile(d, 99)), "max_px": float(d.max()),
           "mean_px_inside": float(d[inside].mean()), "max_px_inside": float(d[inside].max()),
           "mean_dist_over_io_pct": float(100 * (d.mean(1) / io).mean()),
           "nme_io_pct_ours": float(100 * _nme(p, gt).mean()), "nme_io_pct_pyfacer": float(100 * _nme(p_ref, gt).mean())}
    print("\npyfacer parity", res)
    if os.path.isdir("reports/E5_landmarks"):
        json.dump(res, open("reports/E5_landmarks/pyfacer_parity.json", "w"), indent=1)
    assert res["mean_px_inside"] < 0.5 and res["mean_dist_over_io_pct"] < 1.0
