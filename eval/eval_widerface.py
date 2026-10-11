"""E3: SCRFD on WIDER FACE val (Easy/Medium/Hard AP) + GPU latency.

AP is a port of the official WIDER FACE MATLAB eval_tools (read_pred.m, norm_score.m, evaluation.m,
boxoverlap.m, plot/VOCap.m; same algorithm as SCRFD's mmdet/core/evaluation/widerface.py): scores min-max
normalised over the whole val set, IoU >= 0.5 with the "+1" box convention, greedy per-prediction matching,
predictions matching faces outside the setting are ignored, 1000 score thresholds, VOC AP.

Protocols: see PROTOCOLS.

  python eval/eval_widerface.py --model-dir data/models/scrfd_34g --protocols paper wrapper --latency
"""
import argparse
import json
import os
import subprocess
import time

import cv2
import numpy as np
from scipy.io import loadmat

from faceauth.models.scrfd import SCRFD, nms

WIDER = "data/datasets/widerface"
SETTINGS = ("easy", "medium", "hard")
PROTOCOLS = {
    "paper": "SCRFD tools/test_widerface.py --mode 0 (VGA, single scale), mmdet test pipeline: keep-ratio resize so the "
             "long side is 640 (round half up), normalise, zero-pad to 640x640 in normalised space, score > 0.02, boxes "
             "clipped to the resized image, NMS IoU 0.45 without +1, no max_per_img",
    "wrapper": "SCRFD.detect(img, thresh=0.02) exactly as deployed: InsightFace letterbox (int() sizes, black padding "
               "before normalisation), no clipping, NMS IoU 0.4 with +1",
}


def load_gt(gt_dir=f"{WIDER}/eval_tools"):
    """-> [(event, image_name, gt xywh (n,4) float64, {setting: 0-based indices of faces in that setting})]"""
    full = loadmat(os.path.join(gt_dir, "wider_face_val.mat"))
    sub = {s: loadmat(os.path.join(gt_dir, f"wider_{s}_val.mat"))["gt_list"] for s in SETTINGS}
    out = []
    for i in range(len(full["event_list"])):
        event = str(full["event_list"][i][0][0])
        for j in range(len(full["file_list"][i][0])):
            name = str(full["file_list"][i][0][j][0][0])
            boxes = full["face_bbx_list"][i][0][j][0].astype(np.float64).reshape(-1, 4)
            keep = {s: sub[s][i][0][j][0].reshape(-1).astype(np.int64) - 1 for s in SETTINGS}
            out.append((event, name, boxes, keep))
    return out


def _image_eval(pred, gt, in_setting, iou_thresh=0.5):
    """evaluation.m image_evaluation: pred (n,5) xywh+score sorted desc, gt (m,4) xywh."""
    p = pred[:, :4].copy()
    p[:, 2:] += p[:, :2]
    g = gt.copy()
    g[:, 2:] += g[:, :2]
    iw = np.minimum(p[:, None, 2], g[None, :, 2]) - np.maximum(p[:, None, 0], g[None, :, 0]) + 1
    ih = np.minimum(p[:, None, 3], g[None, :, 3]) - np.maximum(p[:, None, 1], g[None, :, 1]) + 1
    inter = iw * ih
    area_p = (p[:, 2] - p[:, 0] + 1) * (p[:, 3] - p[:, 1] + 1)
    area_g = (g[:, 2] - g[:, 0] + 1) * (g[:, 3] - g[:, 1] + 1)
    ov = inter / (area_p[:, None] + area_g[None, :] - inter)
    ov[(iw <= 0) | (ih <= 0)] = 0
    recall = np.zeros(len(g))
    proposal = np.ones(len(p))
    pred_recall = np.zeros(len(p))
    n_rec = 0
    for h in range(len(p)):
        idx = int(ov[h].argmax())
        if ov[h, idx] >= iou_thresh:
            if not in_setting[idx]:
                recall[idx] = -1
                proposal[h] = -1
            elif recall[idx] == 0:
                recall[idx] = 1
                n_rec += 1
        pred_recall[h] = n_rec
    return pred_recall, proposal


def wider_ap(preds: dict, gt, thresh_num=1000, iou_thresh=0.5) -> dict:
    """preds: {(event, name): (n,5) x,y,w,h,score}. Images without a prediction count as empty."""
    preds = {k: v[np.argsort(-v[:, 4], kind="stable")].astype(np.float64) for k, v in preds.items() if len(v)}
    allsc = np.concatenate([v[:, 4] for v in preds.values()])
    lo, hi = allsc.min(), allsc.max()
    preds = {k: np.column_stack([v[:, :4], (v[:, 4] - lo) / (hi - lo)]) for k, v in preds.items()}  # norm_score.m
    thresh = 1 - np.arange(1, thresh_num + 1) / thresh_num
    aps, counts = {}, {}
    for s in SETTINGS:
        pr = np.zeros((thresh_num, 2))
        count_face = 0
        for event, name, boxes, keep in gt:
            count_face += len(keep[s])
            pred = preds.get((event, name))
            if len(boxes) == 0 or pred is None:
                continue
            in_setting = np.zeros(len(boxes), dtype=bool)
            in_setting[keep[s]] = True
            pred_recall, proposal = _image_eval(pred, boxes, in_setting, iou_thresh)
            n = np.searchsorted(-pred[:, 4], -thresh, side="right")  # #preds with score >= thresh
            has = n > 0
            pr[has, 0] += np.cumsum(proposal == 1)[n[has] - 1]
            pr[has, 1] += pred_recall[n[has] - 1]
        with np.errstate(invalid="ignore", divide="ignore"):
            prec, rec = pr[:, 1] / pr[:, 0], pr[:, 1] / count_face
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.fmax.accumulate(np.concatenate(([0.0], prec, [0.0]))[::-1])[::-1]  # VOCap.m (max ignores NaN)
        i = np.where(mrec[1:] != mrec[:-1])[0] + 1
        aps[s] = float(np.sum((mrec[i] - mrec[i - 1]) * mpre[i]))
        counts[s] = count_face
    return {"ap": aps, "faces": counts}


_THR_PAPER = float(np.nextafter(np.float32(0.02), np.float32(1)))  # mmdet keeps score > 0.02 (float32)


def detect_paper(det: SCRFD, img_rgb):
    h, w = img_rgb.shape[:2]
    size = det.input_size
    scale = min(size / max(h, w), size / min(h, w))  # mmcv.rescale_size with img_scale (640, 640)
    nw, nh = int(w * scale + 0.5), int(h * scale + 0.5)
    x = np.zeros((size, size, 3), np.float32)
    x[:nh, :nw] = (cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR).astype(np.float32) - 127.5) / 128.0
    scores, b, _ = det.forward(x.transpose(2, 0, 1)[None], _THR_PAPER)
    b[:, 0::2] = b[:, 0::2].clip(0, nw)
    b[:, 1::2] = b[:, 1::2].clip(0, nh)
    b /= np.array([nw / w, nh / h, nw / w, nh / h], np.float32)
    d = np.hstack([b, scores[:, None]])
    d = d[nms(d, 0.45, offset=0.0)]
    return np.column_stack([d[:, :2], d[:, 2:4] - d[:, :2], d[:, 4]])


def detect_wrapper(det: SCRFD, img_rgb):
    f = det.detect(img_rgb, thresh=0.02)
    if not f:
        return np.zeros((0, 5))
    b = np.array([x["bbox"] for x in f])
    return np.column_stack([b[:, :2], b[:, 2:] - b[:, :2], [x["score"] for x in f]])


def read_rgb(event, name):
    return cv2.cvtColor(cv2.imread(f"{WIDER}/WIDER_val/images/{event}/{name}.jpg"), cv2.COLOR_BGR2RGB)


def run_protocol(det, gt, protocol, limit=0):
    fn = {"paper": detect_paper, "wrapper": detect_wrapper}[protocol]
    preds, t0 = {}, time.time()
    for event, name, _, _ in gt[:limit or None]:
        preds[(event, name)] = fn(det, read_rgb(event, name))
    return preds, time.time() - t0


def _bench(fn, n, warmup):
    for _ in range(warmup):
        fn()
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1e3)
    return {"p50_ms": float(np.percentile(t, 50)), "p95_ms": float(np.percentile(t, 95)),
            "mean_ms": float(np.mean(t)), "n": n, "warmup": warmup}


def latency(det: SCRFD, gt):
    imgs = [cv2.resize(read_rgb(e, n), (640, 480)) for e, n, _, _ in gt[:16]]  # VGA, the sensor profile
    blob, _ = det.blob(imgs[0])
    feed = {det.input_name: blob}
    run = lambda: det.session.run(det.output_names, feed)  # noqa: E731
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    return {
        "providers": det.session.get_providers(),
        "gpu_state_before": gpu,
        "model_b1": _bench(run, 300, 30),
        "model_b16_loop": _bench(lambda: [run() for _ in range(16)], 50, 5),
        "detect_b1_vga": _bench(lambda: det.detect(imgs[0]), 300, 30),
        "detect_batch16_vga": _bench(lambda: det.detect_batch(imgs), 50, 5),
        "note": "model_* = session.run on a 1x3x640x640 blob incl. host<->device copies; detect_* = end to end "
                "(letterbox, run, decode, NMS) on 640x480 RGB at thresh 0.5; batch 16 = 16 sequential runs "
                "(static batch-1 export). GPU 5 is shared with other jobs, see gpu_state_before.",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="data/models/scrfd_34g")
    ap.add_argument("--protocols", nargs="*", default=["paper", "wrapper"])
    ap.add_argument("--latency", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="debug: first N images only (AP then not comparable)")
    ap.add_argument("--out", default="reports/E3_detection/results.json")
    ap.add_argument("--trt", action="store_true", help="TensorRT fp16 engine exactly as deployed by faceauth/engine")
    args = ap.parse_args()
    if args.trt:
        from faceauth.engine.main import _session_options
        det = SCRFD(args.model_dir, providers=[("TensorrtExecutionProvider", {
            "trt_fp16_enable": True, "trt_engine_cache_enable": True, "trt_engine_cache_path": "/tmp/trt"}),
            "CUDAExecutionProvider"], sess_options=_session_options())
        assert det.session.get_providers()[0] == "TensorrtExecutionProvider", det.session.get_providers()
    else:
        det = SCRFD(args.model_dir)
    suffix = "_trt16" if args.trt else ""
    with open(os.path.join(args.model_dir, "MODEL.json")) as f:
        meta = json.load(f)
    gt = load_gt()
    res = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            res = json.load(f)
    with open(f"{WIDER}/DATASET.json") as f:
        ds = json.load(f)
    res["dataset"] = {"name": ds["name"], "split": "val", "images": len(gt), "license": ds["license"],
                      "gt_sha256": {k: v for k, v in ds["files"].items() if k.endswith(".mat")}}
    res["protocols"] = PROTOCOLS
    entry = res.setdefault("models", {}).setdefault(meta["name"], {})
    entry["model"] = {k: meta[k] for k in ("name", "version", "sha256", "source", "license")}
    entry["published"] = meta["params"].get("published_widerface_val_vga")
    for p in args.protocols:
        preds, secs = run_protocol(det, gt, p, args.limit)
        r = wider_ap(preds, gt if not args.limit else gt[:args.limit])
        r.update(images=len(preds), seconds=round(secs, 1), n_dets=int(sum(len(v) for v in preds.values())))
        entry[f"widerface_val_{p}{suffix}"] = r
        print(meta["name"], p, {k: round(v * 100, 2) for k, v in r["ap"].items()}, f"{len(preds)} images", flush=True)
        if not args.limit:
            raw = f"reports/raw/E3_detection/{meta['name']}_{p}{suffix}.npz"
            os.makedirs(os.path.dirname(raw), exist_ok=True)
            np.savez_compressed(raw, **{f"{e}/{n}": v for (e, n), v in preds.items()})
    if args.latency:
        key = "latency_trt16" if args.trt else "latency_cuda_ep"
        entry[key] = latency(det, gt)
        print(json.dumps(entry[key], indent=1))
    if not args.limit:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
