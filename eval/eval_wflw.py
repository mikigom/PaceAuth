"""E5 -- FaRL WFLW-98 landmarks on the WFLW test set (2,500 faces; subsets from the official attribute flags).

Protocols (all decode with faceauth.models.farl):
  box        FaRL's own WFLW eval: square box = extent of the GT landmarks with HRNet's csv rounding
             (HRNet issue #3; the HRNet csv itself sits on OneDrive behind a login). 1 pass. Comparable with the paper.
             Also run with bf16=True (box_bf16).
  kps_gt     production path fed with kps5_from_wflw98(GT): DST5 crop, refine 0 and 1.
  facer_gt   same 5 kps with pyfacer's own template (target_face_scale 0.8), 1 pass: pyfacer's crop.
  kps_scrfd  production path fed with SCRFD-34G kps (if faceauth/models/scrfd.py + data/models/scrfd_34g exist),
             plus the kps5 consistency check FaRL vs SCRFD.
Metrics = FaRL scorer: NME / inter-ocular (60,72) and / inter-pupil (96,97), FR@0.1 and AUC@0.1 (inter-ocular).
Also: 5-point (kps5_from_wflw98) error vs GT, latency on a 640x480 frame, predictions -> data/tmp/lmk/.

Run (GPU 5):
  docker run --rm --gpus '"device=5"' --shm-size 16g -u $(id -u):$(id -g) -e HOME=/tmp \
      -v /fdisk2/Codes/jhseo/faceauth:/app -w /app faceauth:dev python eval/eval_wflw.py
"""
import argparse
import hashlib
import json
import os
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
from scipy.integrate import simpson

import faceauth.models.farl as farl

ROOT = "data/datasets/wflw"
MODEL_DIR = "data/models/farl_wflw_448"
OUT = "reports/E5_landmarks"
SUBSETS = ["pose", "expression", "illumination", "makeup", "occlusion", "blur"]  # attribute column order
FACER_DST5 = ((np.array([[196.0, 226.0], [316.0, 226.0], [256.0, 286.0], [220.0, 360.4], [292.0, 360.4]]) / 256.0 - 1.0)
              * 0.8 + 1.0) * (farl.SIZE - 1) / 2.0
# FaRL paper (FaRL-B ep16, logs/paper/...wflw..., epoch 30) and the released facer checkpoint's own note
PAPER = {"full": 3.96, "pose": 6.91, "expression": 4.21, "illumination": 3.97, "makeup": 3.80, "occlusion": 4.71,
         "blur": 4.57, "fr_0.1": 1.76, "auc_0.1": 61.16}
RELEASED_NOTE = {"nme_io_full": 3.8933, "where": "facer/face_alignment/farl.py comment 'inter_ocular 0.038933 epoch 20'; "
                 "matches FaRL logs/reproduce/...wflw_farl-b-ep64... epoch 22 (0.038931)"}


def load(split="test"):
    path = f"{ROOT}/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_{split}.txt"
    rows = [r.split() for r in open(path) if r.strip()]
    gt = np.array([r[:196] for r in rows], np.float64).reshape(-1, 98, 2)
    return gt, np.array([r[200:206] for r in rows], int), [r[206] for r in rows]


def hrnet_box(gt):
    """(98,2) -> [x1,y1,x2,y2]: center (floor(min)+ceil(max))/2, side max(ceil(max)-floor(min)) (= 200*scale)."""
    lo, hi = np.floor(gt.min(0)), np.ceil(gt.max(0))
    c, half = (lo + hi) / 2, (hi - lo).max() / 2
    return np.r_[c - half, c + half]


def kps_template(gt):
    """Mean kps5_from_wflw98(GT) inside the box crop, left-right symmetrised: the source of farl.DST5."""
    m = farl.crop_matrices(np.array([hrnet_box(g) for g in gt]))
    k = np.einsum("nij,nkj->nki", m[:, :, :2], farl.kps5_from_wflw98(gt)) + m[:, None, :, 2]
    t = k.mean(0)
    mirror = np.c_[farl.SIZE - 1 - t[:, 0], t[:, 1]][[1, 0, 2, 4, 3]]
    return (t + mirror) / 2


def nme(pred, gt, i, j):
    return np.linalg.norm(pred - gt, axis=-1).mean(-1) / np.linalg.norm(gt[:, i] - gt[:, j], axis=-1)


def summarize(pred, gt, attrs, seed=0):
    io, ip = nme(pred, gt, 60, 72), nme(pred, gt, 96, 97)
    out = {}
    for name, sel in [("full", np.ones(len(gt), bool))] + [(s, attrs[:, k] == 1) for k, s in enumerate(SUBSETS)]:
        x = np.arange(0.0, 0.1 + 0.0001, 0.0001)  # FaRL AUC_FR scorer (HRNet issue #6 recipe)
        ced = (io[sel][None] <= x[:, None]).mean(1)
        out[name] = {"n": int(sel.sum()), "nme_io": 100 * io[sel].mean(), "nme_ip": 100 * ip[sel].mean(),
                     "fr_0.1": 100 * (1 - ced[-1]), "auc_0.1": 100 * simpson(ced, x=x) / 0.1}
    boot = np.random.default_rng(seed).integers(0, len(io), (2000, len(io)))
    out["full"]["nme_io_ci95"] = (100 * np.percentile(io[boot].mean(1), [2.5, 97.5])).tolist()
    return out


def run(model, images, names, inputs, refine, chunk=32):
    """inputs[i]: (4,) box or (5,2) kps of face i; faces of one image go into one call."""
    by_img = defaultdict(list)
    for i, n in enumerate(names):
        by_img[n].append(i)
    items, preds = list(by_img.items()), np.zeros((len(names), 98, 2))
    for k in range(0, len(items), chunk):
        part = items[k:k + chunk]
        outs = model.predict_batch([images[n] for n, _ in part], [np.stack([inputs[i] for i in idx]) for _, idx in part],
                                   refine)
        for (_, idx), o in zip(part, outs):
            preds[idx] = o
    return preds


def scrfd_kps(images, names, gt, det_dir="data/models/scrfd_34g"):
    """SCRFD kps matched to each GT face (best IoU with the GT landmark box, IoU >= 0.3) or None."""
    from faceauth.models.scrfd import SCRFD  # noqa: owned by the detector contributor
    det, by_img, kps = SCRFD(det_dir), defaultdict(list), [None] * len(names)
    for i, n in enumerate(names):
        by_img[n].append(i)
    for n, idx in by_img.items():
        faces = det.detect(images[n], thresh=0.3)
        for i in idx:
            lo, hi = gt[i].min(0), gt[i].max(0)
            best, best_iou = None, 0.3
            for f in faces:
                b = np.asarray(f["bbox"], float)
                inter = np.prod(np.clip(np.minimum(hi, b[2:]) - np.maximum(lo, b[:2]), 0, None))
                iou = inter / (np.prod(hi - lo) + np.prod(b[2:] - b[:2]) - inter)
                if iou > best_iou:
                    best, best_iou = f, iou
            kps[i] = None if best is None else np.asarray(best["kps"], float)
    return kps


def kps_offsets(k, ref, gt):
    """Per-point offset of k vs ref in a face frame (x = eye axis, y down), / inter-ocular (60,72)."""
    io = np.linalg.norm(gt[:, 60] - gt[:, 72], axis=-1)
    ax = (ref[:, 1] - ref[:, 0]) / np.linalg.norm(ref[:, 1] - ref[:, 0], axis=-1, keepdims=True)
    rot = np.stack([ax, np.c_[-ax[:, 1], ax[:, 0]]], 1)  # rows: x-axis, y-axis
    d = np.einsum("nij,nkj->nki", rot, k - ref) / io[:, None, None]
    names = ["eye_l", "eye_r", "nose", "mouth_l", "mouth_r"]
    return {"mean_dist_io": float(np.linalg.norm(d, axis=-1).mean()),
            "per_point_mean_dist_io": dict(zip(names, np.linalg.norm(d, axis=-1).mean(0).round(4).tolist())),
            "per_point_mean_offset_io_xy": dict(zip(names, d.mean(0).round(4).tolist()))}


def latency(models, img, kps, iters=30, warmup=5):
    """ms per face for each dtype x batch x variant, interleaved round-robin so the shared GPU's varying load hits
    every variant alike; `min` approximates the uncontended speed."""
    fns = {}
    for dt, mdl in models.items():
        for bs in (1, 8):
            inp = [np.repeat(kps[None], bs, 0)]
            crops = np.random.default_rng(0).uniform(0, 255, (bs, farl.SIZE, farl.SIZE, 3)).astype(np.float32)
            fns[f"{dt}_bs{bs}_predict_refine0"] = (bs, lambda m=mdl, i=inp: m.predict_batch([img], i, 0))
            fns[f"{dt}_bs{bs}_predict_refine1"] = (bs, lambda m=mdl, i=inp: m.predict_batch([img], i, 1))
            fns[f"{dt}_bs{bs}_model_only"] = (bs, lambda m=mdl, c=crops: m.infer_crops(c))
    for _, fn in fns.values():
        for _ in range(warmup):
            fn()
    ts = defaultdict(list)
    for _ in range(iters):
        for k, (bs, fn) in fns.items():
            t = time.perf_counter()
            fn()  # returns numpy -> already synchronised
            ts[k].append((time.perf_counter() - t) * 1000 / bs)
    return {k: {"ms_per_face_min": float(np.min(v)), "ms_per_face_p50": float(np.median(v)),
                "ms_per_face_p95": float(np.percentile(v, 95)), "ms_per_face_mean": float(np.mean(v)), "iters": iters}
            for k, v in ts.items()}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="first N test faces only (smoke runs)")
    args = ap.parse_args()
    torch.manual_seed(0)
    gt, attrs, names = load("test")
    if args.limit:
        gt, attrs, names = gt[:args.limit], attrs[:args.limit], names[:args.limit]
    with ThreadPoolExecutor(16) as ex:
        uniq = sorted(set(names))
        images = dict(zip(uniq, ex.map(lambda n: cv2.cvtColor(cv2.imread(f"{ROOT}/WFLW_images/{n}"), cv2.COLOR_BGR2RGB), uniq)))
    model = farl.FaRLLandmarks(MODEL_DIR)
    meta = json.load(open(f"{MODEL_DIR}/MODEL.json"))
    tmpl = kps_template(load("train")[0])
    assert np.abs(tmpl - farl.DST5).max() < 0.01, (tmpl, farl.DST5)
    kps_gt = farl.kps5_from_wflw98(gt)
    res = {"id": "E5", "name": "landmarks", "date": time.strftime("%Y-%m-%d"), "seed": 0,
           "model": {"version": model.version, "sha256": meta["sha256"], "source": meta["source"]},
           "dataset": {"name": "WFLW test", "n_faces": len(gt), "n_images": len(images),
                       "subset_counts": {s: int(attrs[:, k].sum()) for k, s in enumerate(SUBSETS)},
                       "annotations_sha256": sha256(f"{ROOT}/WFLW_annotations.tar.gz")},
           "units": "nme/fr/auc in %", "kps_template_dst5": tmpl.round(2).tolist(),
           "reference": {"farl_paper_ep16": PAPER, "released_checkpoint_note": RELEASED_NOTE}, "protocols": {}}
    preds = {}
    for key, inputs, refine, desc in [
            ("box", [hrnet_box(g) for g in gt], 0, "GT landmark-extent box (HRNet/FaRL protocol), 1 pass"),
            ("kps_gt_refine0", kps_gt, 0, "kps5_from_wflw98(GT) -> DST5 crop, 1 pass"),
            ("kps_gt_refine1", kps_gt, 1, "kps5_from_wflw98(GT) -> DST5 crop + 1 refine pass")]:
        preds[key] = run(model, images, names, inputs, refine)
        res["protocols"][key] = {"description": desc, "refine": refine, "metrics": summarize(preds[key], gt, attrs)}
    farl.DST5, dst5 = FACER_DST5, farl.DST5  # pyfacer's own 5-point template
    preds["facer_gt"] = run(model, images, names, kps_gt, 0)
    farl.DST5 = dst5
    res["protocols"]["facer_gt"] = {"description": "kps5_from_wflw98(GT) -> pyfacer template (face scale 0.8), 1 pass",
                                    "refine": 0, "metrics": summarize(preds["facer_gt"], gt, attrs)}
    for key in ("box", "kps_gt_refine1"):
        res["protocols"][key]["kps5_vs_gt_kps5"] = kps_offsets(farl.kps5_from_wflw98(preds[key]), kps_gt, gt)
    model16 = farl.FaRLLandmarks(MODEL_DIR, bf16=True)
    p = run(model16, images, names, [hrnet_box(g) for g in gt], 0)
    res["protocols"]["box_bf16"] = {"description": "box protocol with bf16=True", "refine": 0,
                                    "max_abs_diff_px_vs_fp32": float(np.abs(p - preds["box"]).max()),
                                    "metrics": summarize(p, gt, attrs)}

    if os.path.exists("faceauth/models/scrfd.py") and os.path.exists("data/models/scrfd_34g/MODEL.json"):
        kps = scrfd_kps(images, names, gt)
        ok = np.array([k is not None for k in kps])
        k_scrfd = np.stack([k for k in kps if k is not None])
        p = preds["kps_scrfd_refine1"] = run(model, images, [n for n, o in zip(names, ok) if o], list(k_scrfd), 1)
        preds["scrfd_matched"] = ok
        res["protocols"]["kps_scrfd_refine1"] = {
            "description": "SCRFD-34G kps (matched to GT faces) -> DST5 crop + 1 refine pass", "refine": 1,
            "n_matched": int(ok.sum()), "metrics": summarize(p, gt[ok], attrs[ok]),
            "kps5_vs_gt_kps5": kps_offsets(farl.kps5_from_wflw98(p), kps_gt[ok], gt[ok])}
        res["scrfd_consistency"] = {
            "n_faces": int(ok.sum()),
            "farl_kps5_vs_scrfd": kps_offsets(farl.kps5_from_wflw98(p), k_scrfd, gt[ok]),
            "gt_kps5_vs_scrfd": kps_offsets(kps_gt[ok], k_scrfd, gt[ok]),
            "gt_pupils_variant_vs_scrfd": kps_offsets(np.concatenate([gt[ok][:, 96:98], kps_gt[ok][:, 2:]], 1), k_scrfd, gt[ok])}
    else:
        res["scrfd_consistency"] = {"skipped": "faceauth/models/scrfd.py or data/models/scrfd_34g/MODEL.json missing"}

    n0 = names[0]
    s = 640 / max(images[n0].shape[:2])  # sensor-sized frame (640x480 canvas) for latency
    img = np.zeros((480, 640, 3), np.uint8)
    small = cv2.resize(images[n0], None, fx=s, fy=s)
    img[:small.shape[0], :small.shape[1]] = small[:480]
    smi = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"]
    res["peak_gpu_mem_mb_eval_two_models"] = torch.cuda.max_memory_allocated() / 2**20
    res["latency"] = {"device": torch.cuda.get_device_name(0), "frame": "640x480", "faces_per_call": "bs",
                      "gpu_util_mem_before": subprocess.run(smi, capture_output=True, text=True).stdout.strip(),
                      **latency({"float32": model, "bf16": model16}, img, kps_gt[0] * s),
                      "gpu_util_mem_after": subprocess.run(smi, capture_output=True, text=True).stdout.strip()}
    os.makedirs(OUT, exist_ok=True)
    os.makedirs("data/tmp/lmk", exist_ok=True)
    np.savez_compressed("data/tmp/lmk/wflw_test_preds.npz", names=np.array(names), gt=gt, **preds)
    json.dump(res, open(f"{OUT}/results.json" if not args.limit else f"data/tmp/lmk/results_limit{args.limit}.json", "w"),
              indent=1)
    for k, v in res["protocols"].items():
        m = v["metrics"]["full"]
        print(f"{k:18s} n={m['n']} NME_io={m['nme_io']:.3f} FR={m['fr_0.1']:.2f} AUC={m['auc_0.1']:.2f} | " +
              " ".join(f"{s}={v['metrics'][s]['nme_io']:.2f}" for s in SUBSETS))
    print(json.dumps({k: v for k, v in res.items() if k in ("scrfd_consistency", "latency")}, indent=1))


if __name__ == "__main__":
    main()
