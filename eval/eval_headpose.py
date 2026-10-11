"""E4 head pose: AFLW2000-3D + BIWI MAE, sign-convention proof, enrollment-range analysis, latency.

    docker run --rm --gpus '"device=5"' --shm-size 16g -u $(id -u):$(id -g) -e HOME=/tmp -e OMP_NUM_THREADS=4 \
      -e PYTHONPATH=/app:/app/data/tmp/pose/pylib -v /fdisk2/Codes/jhseo/faceauth:/app -w /app faceauth:dev \
      python eval/eval_headpose.py            # -> reports/E4_headpose/{results.json, predictions.npz}
(OMP_NUM_THREADS matters on the busy 128-core host: torch's default 128 threads made CPU preprocessing ~30x slower;
 4 threads was the fastest setting for MTCNN under load.) Needs facenet-pytorch 2.6.0 (`pip install --no-deps`) for
MTCNN, the detector of the published BIWI protocol (FSA-Net TYY_create_db_biwi.py, used by 6DRepNet/6DRepNet360).

Protocols (labels in §0 convention; MAE = per-angle |error| wrapped to [0,180] as in 6DRepNet360 test.py, averaged;
'mean' = mean of the three angles):
  AFLW2000 'official': images with all |angle| <= 99 deg (create_filename_list.py), crop from the min/max of the 21
     AFLW `pt2d` points incl. missing points coded (-1, 452) exactly like datasets.AFLW2000 (k=0.2, sequential
     updates, PIL crop with black padding), Resize(256)+CenterCrop(224) (test.py).
  AFLW2000 'lm68': same margins on the box of the 68 fitted landmarks (pt3d_68): what a clean crop gives.
  BIWI 'paper': MTCNN box (prob >= 0.9) -> headpose.fsa_crop (0.4 margin, squashed to 256) + Resize(256) +
     CenterCrop(224); labels in the depth-camera frame, as published. The detection is the one whose centre is
     nearest to the GT head centre projected into the RGB image (FSA-Net used a position heuristic); frames whose
     projection is not inside that box (grown by 50%) count as missed.
  '<detector>_<crop>' (AFLW2000: scrfd_*, BIWI: scrfd_*, mtcnn_demo): detector box -> headpose crop 'demo' or 'fsa'
     (see faceauth/models/headpose.py) = HeadPose.predict with that crop. SCRFD-34G (faceauth.models.scrfd, score
     >= 0.5) is the terminal/engine detector. Association as above (AFLW2000: nearest to the 68-landmark centroid).
     BIWI labels relative to the RGB camera (faceauth.sim.biwi 'pose').
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from PIL import Image

from faceauth.models.headpose import CROPS, HeadPose, demo_crop, fsa_crop
from faceauth.models.scrfd import SCRFD
from faceauth.sim.biwi import iter_frames, project
from faceauth.terminal.guide import CENTER_TOL, TOL_PITCH, TOL_YAW, TARGETS, direction_of

PUBLISHED = {  # MAE yaw, pitch, roll, mean (deg); trained on 300W-LP (+ CMU Panoptic for Full Rotation)
    "6drepnet": {"source": "6DRepNet README / ICIP 2022 (RepVGG-B1g2, 300W-LP)",
                 "aflw2000": [3.63, 4.91, 3.37, 3.97], "biwi": [3.24, 4.48, 2.68, 3.47]},
    "6drepnet360_300wlp": {"source": "arXiv 2309.07654v1 Table 1, row '6DRepNet' (ResNet-50, 300W-LP)",
                           "aflw2000": [3.27, 4.58, 2.98, 3.61], "biwi": [3.23, 5.32, 2.78, 3.78]},
    "6drepnet360": {"source": "arXiv 2309.07654v1 Table 1, row '6DRepNet360' (ResNet-50, 300W-LP + CMU Panoptic)",
                    "aflw2000": [3.73, 5.52, 3.53, 4.26], "biwi": [3.37, 3.87, 2.93, 3.39]},
}
ENROLL_RANGE = (40.0, 25.0)  # |yaw| <= 40, |pitch| <= 25 (GT)


def abs_err(pred, gt) -> np.ndarray:
    d = np.abs(np.asarray(pred, float) - np.asarray(gt, float)) % 360
    return np.minimum(d, 360 - d)


def mae(pred, gt) -> dict:
    e = abs_err(pred, gt)
    if not len(e):
        return {"n": 0}
    return {"yaw": round(float(e[:, 0].mean()), 3), "pitch": round(float(e[:, 1].mean()), 3),
            "roll": round(float(e[:, 2].mean()), 3), "mean": round(float(e.mean()), 3), "n": int(len(e))}


def _frac(a, b):
    return round(a / b, 4) if b else None


def enrollment(pred, gt) -> dict:
    """MAE in the enrollment range and 9-way direction-bin agreement (faceauth.terminal.guide bins)."""
    in_range = (np.abs(gt[:, 0]) <= ENROLL_RANGE[0]) & (np.abs(gt[:, 1]) <= ENROLL_RANGE[1])
    gb = [direction_of(*g[:2]) for g in gt]
    pb = [direction_of(*p[:2]) for p in pred]
    m = 2.0  # terminal Guide margin: frames are kept only 2 deg inside a bin
    pk = [direction_of(*p[:2], TOL_YAW - m, TOL_PITCH - m, (CENTER_TOL[0] - m, CENTER_TOL[1] - m)) for p in pred]
    per = {}
    for name in TARGETS:
        g = [i for i, b in enumerate(gb) if b == name]
        p = [i for i, b in enumerate(pb) if b == name]
        k = [i for i, b in enumerate(pk) if b == name]
        per[name] = {"n_gt": len(g), "recall": _frac(sum(pb[i] == name for i in g), len(g)),
                     "n_pred": len(p), "precision": _frac(sum(gb[i] == name for i in p), len(p)),
                     "n_kept_margin2": len(k), "kept_precision": _frac(sum(gb[i] == name for i in k), len(k))}
    g_all = [i for i, b in enumerate(gb) if b is not None]
    p_all = [i for i, b in enumerate(pb) if b is not None]
    k_all = [i for i, b in enumerate(pk) if b is not None]
    return {
        "range": {"abs_yaw_max": ENROLL_RANGE[0], "abs_pitch_max": ENROLL_RANGE[1]},
        "mae_in_range": mae(pred[in_range], gt[in_range]),
        "bin_accuracy": _frac(sum(pb[i] == gb[i] for i in g_all), len(g_all)),  # GT in a bin -> same predicted bin
        "n_gt_in_bins": len(g_all),
        "bin_precision": _frac(sum(gb[i] == pb[i] for i in p_all), len(p_all)),  # predicted bin -> GT same bin
        "n_pred_in_bins": len(p_all),
        "wrong_bin_rate": _frac(sum(gb[i] is not None and gb[i] != pb[i] for i in p_all), len(p_all)),
        "kept_margin2_precision": _frac(sum(gb[i] == pk[i] for i in k_all), len(k_all)),
        "n_kept_margin2": len(k_all),
        "kept_margin2_wrong_bin_rate": _frac(sum(gb[i] is not None and gb[i] != pk[i] for i in k_all), len(k_all)),
        "per_bin": per,
    }


def retry(fn, *args, **kw):
    """GPU 5 is shared and sometimes completely full: on CUDA OOM wait and retry instead of losing the run."""
    for _ in range(120):
        try:
            return fn(*args, **kw)
        except Exception as e:  # torch OutOfMemoryError / AcceleratorError, onnxruntime CUDA allocation failures
            if "out of memory" not in str(e).lower():
                raise
            print("CUDA out of memory (shared GPU), retrying in 30 s", flush=True)
            time.sleep(30)
    raise RuntimeError("GPU memory did not become available")


def aflw2000(root, limit=0):
    """-> names, images, official crops, lm68 crops, labels (n,3) §0, 68-pt landmarks (n,68,2)."""
    out = {k: [] for k in ("name", "img", "crop_official", "crop_lm68", "gt", "lm68")}
    for f in sorted(Path(root, "AFLW2000").glob("image*.jpg"))[:limit or None]:
        mat = sio.loadmat(f.with_suffix(".mat"))
        pitch, yaw, roll = np.degrees(mat["Pose_Para"][0][:3])
        if max(abs(pitch), abs(yaw), abs(roll)) > 99:
            continue
        img = Image.open(f).convert("RGB")
        lm = mat["pt3d_68"][:2].T
        for key, pts in (("crop_official", mat["pt2d"]), ("crop_lm68", lm.T)):
            x_min, y_min, x_max, y_max = pts[0].min(), pts[1].min(), pts[0].max(), pts[1].max()
            k = 0.20  # datasets.AFLW2000 (sequential in-place updates kept)
            x_min -= 2 * k * abs(x_max - x_min)
            y_min -= 2 * k * abs(y_max - y_min)
            x_max += 2 * k * abs(x_max - x_min)
            y_max += 0.6 * k * abs(y_max - y_min)
            out[key].append(np.asarray(img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))))
        out["name"].append(f.stem)
        out["img"].append(np.asarray(img))
        out["gt"].append([-yaw, pitch, roll])  # 3DDFA yaw > 0 = nose toward image left -> §0
        out["lm68"].append(lm)
    out["gt"], out["lm68"] = np.array(out["gt"], np.float32), np.array(out["lm68"], np.float32)
    return out


def landmark_proxies(lm68) -> np.ndarray:
    """Geometric pose proxies from 68 landmarks (iBUG order): yaw ~ nose-tip offset from the eye midpoint toward
    image right, pitch ~ minus the nose drop below the eyes, roll ~ eye-line angle (clockwise > 0, y down)."""
    le, re = lm68[:, 36:42].mean(1), lm68[:, 42:48].mean(1)  # image-left eye, image-right eye
    mid, nose, mouth = (le + re) / 2, lm68[:, 30], lm68[:, 48:68].mean(1)
    face_h = np.linalg.norm(mouth - mid, axis=1)
    return np.stack([(nose[:, 0] - mid[:, 0]) / face_h, -(nose[:, 1] - mid[:, 1]) / face_h,
                     np.degrees(np.arctan2(re[:, 1] - le[:, 1], re[:, 0] - le[:, 0]))], 1)


def corr(a, b) -> list:
    return [round(float(np.corrcoef(a[:, i], b[:, i])[0, 1]), 4) for i in range(3)]


def detect_mtcnn(mtcnn, imgs):
    """MTCNN on a list of same-size RGB images -> per image (boxes (k,4), probs (k,)) or None."""
    boxes, probs = mtcnn.detect(np.stack(imgs))
    return [None if b is None else (np.asarray(b, float).reshape(-1, 4), np.asarray(p, float).reshape(-1))
            for b, p in zip(boxes, probs)]


def detect_scrfd(scrfd, img):
    """SCRFD-34G (faceauth.models.scrfd, default thresh 0.5) -> (boxes (k,4), scores (k,)) or None."""
    d = retry(scrfd.detect, img)
    return (np.array([x["bbox"] for x in d], float), np.array([x["score"] for x in d], float)) if d else None


def pick_face(det, point, min_score):
    """Box of the detection nearest to `point` among score >= min_score whose box grown by 50% contains it, or None."""
    best = None
    for b, p in zip(*(det or ((), ()))):
        c = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
        inside = abs(point[0] - c[0]) <= b[2] - b[0] and abs(point[1] - c[1]) <= b[3] - b[1]
        if p >= min_score and inside and (best is None or np.linalg.norm(c - point) < best[0]):
            best = (np.linalg.norm(c - point), b)
    return None if best is None else best[1]


@torch.inference_mode()
def latency(hp: HeadPose, img, n_iter=200) -> dict:
    """Per-face ms (median, p95): model forward on a preprocessed batch and full predict() on a 640x480 image."""
    out = {}
    for bs in (1, 16):
        x = torch.randn(bs, 3, 224, 224, device=hp.device)
        boxes = np.array([[260, 150, 380, 310]] * bs, np.float32)
        for name, fn in (("forward", lambda: hp.net(x)), ("predict", lambda: hp.predict(img, boxes))):
            ts = []
            for i in range(n_iter + 20):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                fn()
                torch.cuda.synchronize()
                if i >= 20:
                    ts.append((time.perf_counter() - t0) * 1000)
            out[f"{name}_bs{bs}"] = {"median_ms_per_face": round(float(np.median(ts)) / bs, 3),
                                     "p95_ms_per_face": round(float(np.percentile(ts, 95)) / bs, 3),
                                     "median_ms_per_batch": round(float(np.median(ts)), 3), "n_iter": n_iter}
    return out


def save(out_dir, res, raw=None):
    (out_dir / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    if raw is not None:
        np.savez_compressed(out_dir / "predictions.npz", **raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["data/models/6drepnet", "data/models/6drepnet360",
                                                    "data/tmp/pose/6drepnet360_300wlp"])
    ap.add_argument("--scrfd", default="data/models/scrfd_34g")
    ap.add_argument("--aflw", default="data/datasets/aflw2000")
    ap.add_argument("--biwi", default="data/datasets/biwi")
    ap.add_argument("--biwi-step", type=int, default=1)
    ap.add_argument("--aflw-limit", type=int, default=0, help="smoke test: first N AFLW2000 images only")
    ap.add_argument("--out", default="reports/E4_headpose")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    from facenet_pytorch import MTCNN
    mtcnn = MTCNN(keep_all=True, device=torch.device("cpu"))  # same boxes as on GPU; GPU 5 is contended
    scrfd = retry(SCRFD, args.scrfd)
    res = {"id": "E4_headpose", "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "convention": "§0: yaw>0 nose toward image right (subject's left), pitch>0 up, roll>0 clockwise; degrees",
           "detectors": {"mtcnn": "facenet-pytorch 2.6.0 MTCNN (0.6/0.7/0.7, min face 20, factor 0.709), prob >= 0.9",
                         "scrfd": f"faceauth.models.scrfd.SCRFD('{args.scrfd}'), score >= 0.5"},
           "enrollment_bins": {"targets": TARGETS, "tol_yaw": TOL_YAW, "tol_pitch": TOL_PITCH, "center_tol": CENTER_TOL,
                               "source": "faceauth.terminal.guide.direction_of"},
           "models": {}, "datasets": {}, "aflw2000": {}, "biwi": {}, "latency": {}, "convention_checks": {}}
    raw = {}

    # ---- detection phase: data, MTCNN (CPU) + SCRFD (GPU) boxes, crops
    t0 = time.time()
    a = aflw2000(args.aflw, args.aflw_limit)
    gt = a["gt"]
    res["datasets"]["aflw2000"] = {"n_images": len(list(Path(args.aflw, "AFLW2000").glob("image*.jpg"))),
                                   "n_eval": len(gt), "filter": "all |angle| <= 99 deg"}
    box_s = [pick_face(detect_scrfd(scrfd, im), c, 0.5) for im, c in zip(a["img"], a["lm68"].mean(1))]
    has_s = np.array([b is not None for b in box_s])
    a_crops = {"scrfd_demo": [demo_crop(a["img"][i], box_s[i]) for i in np.flatnonzero(has_s)],
               "scrfd_fsa": [fsa_crop(a["img"][i], box_s[i]) for i in np.flatnonzero(has_s)]}
    prox = landmark_proxies(a["lm68"])
    c = corr(gt, prox)
    res["convention_checks"]["aflw2000_labels_vs_landmark_proxies_pearson"] = {
        "yaw_vs_nose_offset_right": c[0], "pitch_vs_nose_rise": c[1], "roll_vs_eye_line_angle": c[2], "n": len(gt)}
    print("AFLW2000 data", res["datasets"]["aflw2000"], "scrfd", int(has_s.sum()), f"{time.time() - t0:.0f}s",
          flush=True)

    frames, b_crops = [], {"paper": [], "mtcnn_demo": [], "scrfd_demo": [], "scrfd_fsa": []}

    def flush(batch):
        for f, d in zip(batch, detect_mtcnn(mtcnn, [b["rgb"] for b in batch])):
            centre = project(f["K_rgb"], f["head_center"])
            bm, bs = pick_face(d, centre, 0.9), pick_face(detect_scrfd(scrfd, f["rgb"]), centre, 0.5)
            frames.append((f["seq"], f["frame"], f["subject"], f["pose_depthcam"], f["pose"], bm is not None,
                           bs is not None))
            if bm is not None:
                b_crops["paper"].append(fsa_crop(f["rgb"], bm))
                b_crops["mtcnn_demo"].append(demo_crop(f["rgb"], bm))
            if bs is not None:
                b_crops["scrfd_demo"].append(demo_crop(f["rgb"], bs))
                b_crops["scrfd_fsa"].append(fsa_crop(f["rgb"], bs))
        if len(frames) % 2048 < len(batch):
            print("BIWI frames", len(frames), f"{time.time() - t0:.0f}s", flush=True)

    batch = []
    for f in iter_frames(args.biwi, register=False, step=args.biwi_step):
        batch.append(f)
        if len(batch) == 32:
            flush(batch)
            batch = []
    if batch:
        flush(batch)
    fm, fs = np.array([fr[5] for fr in frames]), np.array([fr[6] for fr in frames])
    gt_d, gt_r = np.array([fr[3] for fr in frames], np.float32), np.array([fr[4] for fr in frames], np.float32)
    res["datasets"]["biwi"] = {"n_frames": len(frames), "n_sequences": len({fr[0] for fr in frames}),
                               "n_subjects": len({fr[2] for fr in frames}), "step": args.biwi_step,
                               "mtcnn_detected": int(fm.sum()), "scrfd_detected": int(fs.sum())}
    print("BIWI data", res["datasets"]["biwi"], f"{time.time() - t0:.0f}s", flush=True)
    del scrfd

    # ---- head pose (GPU)
    # GPU 5 is shared and often nearly full: reserve working memory once; the caching allocator keeps it for us.
    retry(torch.empty, 2 << 30, dtype=torch.uint8, device="cuda")
    models = {Path(d).name: retry(HeadPose, d) for d in args.models}
    for (n, m), d in zip(models.items(), args.models):
        meta = json.loads(Path(d, "MODEL.json").read_text())
        res["models"][n] = {"version": m.version, "dir": d, "predict_crop": m.crop, "sha256": meta["sha256"]["weights"],
                            "source": meta["source"], "license": meta["license"], "published": PUBLISHED.get(n)}
    for name, hp in models.items():
        i0 = int(np.flatnonzero(has_s)[0])  # predict() == predict_crops(<its crop>) by construction; check once
        fn, rs = CROPS[hp.crop]
        assert np.allclose(retry(hp.predict, a["img"][i0], box_s[i0][None]),
                           retry(hp.predict_crops, [fn(a["img"][i0], box_s[i0])], rs), atol=1e-3)
        p = {"official": retry(hp.predict_crops, a["crop_official"], resize=256),
             "lm68": retry(hp.predict_crops, a["crop_lm68"], resize=256),
             "lm68_flip": retry(hp.predict_crops, [np.ascontiguousarray(c[:, ::-1]) for c in a["crop_lm68"]], 256),
             "scrfd_demo": retry(hp.predict_crops, a_crops["scrfd_demo"], resize=224),
             "scrfd_fsa": retry(hp.predict_crops, a_crops["scrfd_fsa"], resize=256)}
        g_s = gt[has_s]
        r = {"official": mae(p["official"], gt), "lm68": mae(p["lm68"], gt),
             "scrfd_demo": mae(p["scrfd_demo"], g_s), "scrfd_fsa": mae(p["scrfd_fsa"], g_s),
             "scrfd_n_no_detection": int((~has_s).sum()),
             "official_enrollment": enrollment(p["official"], gt),
             "scrfd_demo_enrollment": enrollment(p["scrfd_demo"], g_s),
             "scrfd_fsa_enrollment": enrollment(p["scrfd_fsa"], g_s)}
        pl, pf = p["lm68"], p["lm68_flip"]
        r["flip"] = {"crop": "lm68", "n": len(gt), "mean_abs_yaw_sum": round(float(np.abs(pl[:, 0] + pf[:, 0]).mean()), 3),
                     "mean_abs_pitch_diff": round(float(np.abs(pl[:, 1] - pf[:, 1]).mean()), 3),
                     "mean_abs_roll_sum": round(float(np.abs(pl[:, 2] + pf[:, 2]).mean()), 3),
                     "yaw_sign_flipped_frac(|yaw|>5)": _frac(int(((np.sign(pl[:, 0]) == -np.sign(pf[:, 0]))
                                                                  & (np.abs(pl[:, 0]) > 5)).sum()),
                                                             int((np.abs(pl[:, 0]) > 5).sum()))}
        r["pred_vs_landmark_proxies_pearson"] = corr(pl, prox)
        r["pred_vs_label_pearson"] = corr(pl, gt)
        res["aflw2000"][name] = r
        raw.update({f"aflw_{name}_{k}": v for k, v in p.items()})
        print(name, "AFLW2000 official", r["official"], "lm68", r["lm68"], "scrfd_demo", r["scrfd_demo"],
              "scrfd_fsa", r["scrfd_fsa"], "flip", r["flip"], flush=True)
    raw.update({"aflw_gt": gt, "aflw_names": np.array(a["name"]), "aflw_scrfd_found": has_s})
    del a, a_crops
    save(out_dir, res)  # partial

    for name, hp in models.items():
        p = {k: retry(hp.predict_crops, v, resize=CROPS["fsa" if k in ("paper", "scrfd_fsa") else "demo"][1])
             for k, v in b_crops.items()}
        r = {"paper": mae(p["paper"], gt_d[fm]), "paper_rgbcam_labels": mae(p["paper"], gt_r[fm]),
             "mtcnn_demo": mae(p["mtcnn_demo"], gt_r[fm]),
             "scrfd_demo": mae(p["scrfd_demo"], gt_r[fs]), "scrfd_fsa": mae(p["scrfd_fsa"], gt_r[fs]),
             "paper_enrollment": enrollment(p["paper"], gt_d[fm]),
             "scrfd_demo_enrollment": enrollment(p["scrfd_demo"], gt_r[fs]),
             "scrfd_fsa_enrollment": enrollment(p["scrfd_fsa"], gt_r[fs])}
        res["biwi"][name] = r
        raw.update({f"biwi_{name}_{k}": v for k, v in p.items()})
        print(name, "BIWI paper", r["paper"], "scrfd_demo", r["scrfd_demo"], "scrfd_fsa", r["scrfd_fsa"], flush=True)
    raw.update({"biwi_gt_depthcam": gt_d, "biwi_gt_rgbcam": gt_r, "biwi_mtcnn_found": fm, "biwi_scrfd_found": fs,
                "biwi_seq": np.array([fr[0] for fr in frames]), "biwi_frame": np.array([fr[1] for fr in frames])})
    save(out_dir, res, raw)  # partial

    # ---- latency (GPU, fp32)
    img = np.random.default_rng(0).integers(0, 255, (480, 640, 3), np.uint8)
    for name, hp in models.items():
        res["latency"][name] = retry(latency, hp, img)
        print(name, "latency", res["latency"][name], flush=True)
    res["latency_env"] = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "precision": "fp32",
                          "cpu_threads": torch.get_num_threads(),
                          "note": "GPU 5 shared with other jobs (100% utilisation by others during the run); "
                                  "predict() includes the crop + PIL resize on CPU and host<->device copies"}
    save(out_dir, res, raw)
    print("wrote", out_dir / "results.json", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
