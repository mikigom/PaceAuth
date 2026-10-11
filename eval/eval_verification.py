"""E1: 1:1 verification on the InsightFace .bin benchmarks (LFW, CFP-FP, AgeDB-30, CALFW, CPLFW) + speed / fp16 check.

    docker run --rm --gpus '"device=5"' --shm-size 16g -u $(id -u):$(id -g) -e HOME=/tmp \
        -v /fdisk2/Codes/jhseo/faceauth:/app -w /app faceauth:dev python eval/eval_verification.py

Protocol (InsightFace eval/verification.py, also used by arcface_torch and CVLFace): every pre-aligned 112x112 image
and its horizontal flip are embedded; "flip" = L2-normalise(f(x) + f(flip(x))) of raw features (the number the model
cards publish), "noflip" = L2-normalise(f(x)). 10-fold accuracy: KFold without shuffle, threshold on squared L2
distance from 0:4:0.01 picked on the 9 training folds. TAR@FAR=1e-3 is pooled over all pairs of a set
(3000-3500 impostor pairs -> at most 3 accepted impostors; very coarse, Wilson 95% CI on TAR only).
Models whose weight files are missing are reported as pending. Writes <out>/results.json and <out>/summary.md.
"""
import argparse
import hashlib
import json
import pickle
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.stats import binomtest

from faceauth.models.recog import Recognizer

BINS = ("lfw", "cfp_fp", "agedb_30", "calfw", "cplfw")
PUBLISHED = {  # flip-test 10-fold accuracy (%)
    "kprpe_vitb_wf12m": {
        "source": "CVLFace README 'Face Recognition Performance Board', row ViT KPRPE / AdaFace / WebFace12M "
                  "(github.com/mk-minchul/CVLface, cvlface/pretrained_models/README.md)",
        "acc": {"lfw": 99.82, "cfp_fp": 99.30, "agedb_30": 98.10, "calfw": 95.93, "cplfw": 95.65}},
    "insightface_vit_l_wf42m": {
        "source": "arcface_torch README publishes only MFR-ALL 97.85, IJB-C TAR@1e-4 98.00 / 1e-5 97.23 for "
                  "WF42M-PFC-0.3 ViT-L; no numbers for these five sets", "acc": {}},
}


class _BinUnpickler(pickle.Unpickler):
    """.bin files are untrusted pickles: only numpy array reconstruction is allowed (no arbitrary callables)."""
    OK = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
          ("numpy", "ndarray"), ("numpy", "dtype")}

    def find_class(self, module, name):
        if (module, name) not in self.OK:
            raise pickle.UnpicklingError(f"forbidden global {module}.{name}")
        if name == "dtype":  # py2-era pickles pass align/copy as ints
            return lambda obj, align=False, copy=False: np.dtype(obj, bool(align), bool(copy))
        return super().find_class(module.replace("numpy.core.", "numpy._core."), name)


def load_bin(path) -> tuple[np.ndarray, np.ndarray]:
    """InsightFace .bin -> (images (2n,112,112,3) uint8 RGB, issame (n,) bool); pairs are images (2i, 2i+1)."""
    with open(path, "rb") as f:
        bins, issame = _BinUnpickler(f, encoding="bytes").load()
    imgs = np.stack([cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                     for b in bins])
    issame = np.asarray(issame, bool)
    if imgs.shape[1:] != (112, 112, 3) or len(imgs) != 2 * len(issame):
        raise ValueError(f"{path}: unexpected layout {imgs.shape} / {issame.shape}")
    return imgs, issame


def l2n(f: np.ndarray) -> np.ndarray:
    f = f.astype(np.float64)
    return f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)


def kfold_accuracy(emb: np.ndarray, issame: np.ndarray, folds: int = 10) -> tuple[float, float]:
    """InsightFace/facenet calculate_roc on L2-normalised embeddings -> (mean %, std %) over folds."""
    dist = np.sum((emb[0::2] - emb[1::2]) ** 2, 1)
    correct = (dist[:, None] < np.arange(0, 4, 0.01)[None]) == issame[:, None]  # (pairs, thresholds)
    accs = []
    for test in np.array_split(np.arange(len(dist)), folds):
        train = np.ones(len(dist), bool)
        train[test] = False
        accs.append(correct[test, np.argmax(correct[train].mean(0))].mean())
    return 100 * float(np.mean(accs)), 100 * float(np.std(accs))


def tar_at_far(emb: np.ndarray, issame: np.ndarray, far: float = 1e-3) -> dict:
    """Pooled over all pairs: accept if cos > thr, thr = the (k+1)-th highest impostor score, k = floor(far*n_imp)."""
    s = np.sum(emb[0::2] * emb[1::2], 1)
    imp, gen = np.sort(s[~issame])[::-1], s[issame]
    thr = float(imp[int(np.floor(far * len(imp) + 1e-9))])
    n_acc = int((gen > thr).sum())
    ci = binomtest(n_acc, len(gen)).proportion_ci(method="wilson")
    return {"tar": 100 * n_acc / len(gen), "tar_ci95": [100 * ci.low, 100 * ci.high], "thr_cos": thr,
            "far": float((imp > thr).mean()), "n_genuine": int(len(gen)), "n_impostor": int(len(imp))}


def bench(rec: Recognizer, crops: np.ndarray, batch: int, iters: int) -> list[dict]:
    """End-to-end embed_aligned latency (H2D, normalise, [DFA], forward, D2H) for one batch of aligned crops.
    fp32 / fp16 iterations are interleaved so both see the same load on the shared GPU; min = least-disturbed run."""
    x, ms, peak = crops[:batch], {False: [], True: []}, {}
    for fp16 in ms:
        rec.fp16 = fp16
        torch.cuda.reset_peak_memory_stats()
        for _ in range(3):  # warm-up
            rec.features(x)
        peak[fp16] = torch.cuda.max_memory_allocated() / 2 ** 20
    for _ in range(iters):
        for fp16 in ms:
            rec.fp16 = fp16
            t0 = time.perf_counter()
            rec.features(x)
            ms[fp16].append(1000 * (time.perf_counter() - t0))
    return [{"precision": "fp16" if k else "fp32", "batch": batch, "iters": iters, "ms_median": float(np.median(v)),
             "ms_p95": float(np.percentile(v, 95)), "ms_min": float(np.min(v)),
             "img_per_s": batch * 1000 / float(np.median(v)), "img_per_s_best": batch * 1000 / float(np.min(v)),
             "peak_mem_mb": peak[k]} for k, v in ms.items()]


def sha256(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def gpu_load() -> str:
    """Visible-GPU utilisation / memory incl. other users' processes (the GPU is shared; speed is indicative)."""
    q = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"]
    return subprocess.run(q, capture_output=True, text=True).stdout.strip()


def evaluate_model(model_dir: Path, data: dict) -> dict:
    meta = json.loads((model_dir / "MODEL.json").read_text())
    missing = [f for f in meta["files"].values() if not (model_dir / f).exists()]
    if missing:
        return {"status": "pending", "reason": f"weight file(s) not present: {missing}", "model_dir": str(model_dir)}
    rec = Recognizer(model_dir, device="cuda")  # strict load: raises on any missing/unexpected key
    out = {"status": "evaluated", "version": rec.version, "model_dir": str(model_dir), "strict_load": "0 missing / 0 unexpected keys",
           "sha256": {k: sha256(model_dir / f) for k, f in meta["files"].items()},
           "sha256_in_model_json": meta["sha256"], "metrics": {}, "speed": []}
    emb32, cos = {}, []
    for prec in ("fp32", "fp16"):
        rec.fp16 = prec == "fp16"
        out["metrics"][prec] = {}
        for name, (imgs, issame) in data.items():
            f0, f1 = rec.features(imgs), rec.features(imgs[:, :, ::-1])
            r = {}
            for mode, emb in (("flip", l2n(f0.astype(np.float64) + f1)), ("noflip", l2n(f0))):
                r[f"acc_{mode}"], r[f"std_{mode}"] = kfold_accuracy(emb, issame)
                r[f"tar_far1e-3_{mode}"] = tar_at_far(emb, issame)
            out["metrics"][prec][name] = r
            if prec == "fp32":
                emb32[name] = l2n(f0)
            else:
                cos.append(np.sum(emb32[name] * l2n(f0), 1))
            print(model_dir.name, prec, name, f"flip {r['acc_flip']:.2f} noflip {r['acc_noflip']:.2f}", flush=True)
    cos = np.concatenate(cos)
    out["fp16_vs_fp32_cos"] = {"n_images": int(len(cos)), "mean": float(cos.mean()), "min": float(cos.min()),
                               "p0.1": float(np.percentile(cos, 0.1))}
    crops = next(iter(data.values()))[0]
    out["gpu_load_before_speed"] = gpu_load()
    for batch, iters in ((1, 200), (32, 50), (128, 20)):
        out["speed"] += bench(rec, crops, batch, iters)
        print(out["speed"][-2:], flush=True)
    out["gpu_load_after_speed"] = gpu_load()
    del rec
    torch.cuda.empty_cache()
    return out


def summary_md(res: dict) -> str:
    L = ["# E1 — Recognition 1:1 verification (LFW, CFP-FP, AgeDB-30, CALFW, CPLFW)", "",
         f"Generated {res['created']} by `eval/eval_verification.py`. GPU: {res['env']['gpu']} (shared with other jobs; "
         "timings indicative). Protocol: InsightFace 10-fold accuracy on pre-aligned 112x112 pairs; "
         "*flip* = embedding of image + horizontal flip (the number model cards publish).", "",
         "Datasets: " + ", ".join(f"{k} {v['pairs']} pairs ({v['genuine']} genuine)" for k, v in res["datasets"].items()),
         ""]
    cols = list(res["datasets"])
    head = "| model | precision | mode | " + " | ".join(cols) + " | mean |"
    sep = "|" + "---|" * (len(cols) + 4)
    L += ["## 10-fold accuracy (%, mean ± std over folds)", "", head, sep]
    for m, r in res["models"].items():
        if r["status"] != "evaluated":
            L.append(f"| {m} | — | — | " + " | ".join("pending" for _ in cols) + " | — |")
            continue
        for prec in ("fp32", "fp16"):
            for mode in ("flip", "noflip"):
                v = [r["metrics"][prec][c][f"acc_{mode}"] for c in cols]
                s = [r["metrics"][prec][c][f"std_{mode}"] for c in cols]
                L.append(f"| {m} | {prec} | {mode} | " + " | ".join(f"{a:.2f} ± {b:.2f}" for a, b in zip(v, s)) +
                         f" | {np.mean(v):.2f} |")
        pub = PUBLISHED.get(m, {}).get("acc", {})
        if pub:
            L.append(f"| {m} | published | flip | " + " | ".join(f"{pub[c]:.2f}" for c in cols) +
                     f" | {np.mean([pub[c] for c in cols]):.2f} |")
            d = [r["metrics"]["fp32"][c]["acc_flip"] - pub[c] for c in cols]
            L.append(f"| {m} | Δ fp32 − published | flip | " + " | ".join(f"{x:+.2f}" for x in d) + " | |")
    L += ["", "## TAR @ FAR=1e-3 (%, flip, fp32; pooled over all pairs, Wilson 95% CI)", "",
          "Only 3000–3500 impostor pairs per set: the threshold is set by ~3 impostor pairs, so these values are coarse.",
          "", "| model | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for m, r in res["models"].items():
        if r["status"] == "evaluated":
            t = [r["metrics"]["fp32"][c]["tar_far1e-3_flip"] for c in cols]
            L.append(f"| {m} | " + " | ".join(f"{x['tar']:.2f} [{x['tar_ci95'][0]:.2f}, {x['tar_ci95'][1]:.2f}]"
                                              for x in t) + " |")
    L += ["", "## Speed (embed_aligned end-to-end, one GPU process) and fp16 agreement", "",
          "GPU 5 is shared with other contributors' jobs: medians include their load (snapshot below); "
          "min = least-disturbed iteration, closer to an idle GPU.", "",
          "| model | precision | batch | ms median | ms p95 | ms min | img/s (median) | img/s (min) | peak GPU MB |",
          "|---|---|---|---|---|---|---|---|---|"]
    for m, r in res["models"].items():
        for s in r.get("speed", []):
            L.append(f"| {m} | {s['precision']} | {s['batch']} | {s['ms_median']:.1f} | {s['ms_p95']:.1f} | "
                     f"{s['ms_min']:.1f} | {s['img_per_s']:.0f} | {s['img_per_s_best']:.0f} | {s['peak_mem_mb']:.0f} |")
    L.append("")
    for m, r in res["models"].items():
        if r["status"] != "evaluated":
            L.append(f"- {m}: **{r['status']}** — {r['reason']}. When present: `python -m faceauth.models.recog "
                     f"{r['model_dir']} --write-sha256` (strict load + hash into MODEL.json), then re-run this script.")
            continue
        c = r["fp16_vs_fp32_cos"]
        dacc = max(abs(r["metrics"]["fp16"][b][f"acc_{k}"] - r["metrics"]["fp32"][b][f"acc_{k}"])
                   for b in cols for k in ("flip", "noflip"))
        sp = {(s["precision"], s["batch"]): s["ms_median"] for s in r["speed"]}
        ok = dacc <= 0.1 and c["min"] >= 0.999  # ponytail: fixed rule (0.1 pp = 6 of 6000 pairs); E2 must re-confirm
        L.append(f"- {m}: cos(fp16, fp32) over {c['n_images']} images: mean {c['mean']:.6f}, 0.1th pct "
                 f"{c['p0.1']:.6f}, min {c['min']:.6f}; max |Δacc| fp16 vs fp32 = {dacc:.2f} pp; fp16 speed-up "
                 f"x{sp[('fp32', 128)] / sp[('fp16', 128)]:.2f} at batch 128, x{sp[('fp32', 1)] / sp[('fp16', 1)]:.2f} "
                 f"at batch 1 -> recommended precision: **{'fp16 autocast' if ok else 'fp32'}** "
                 "(rule: max |Δacc| <= 0.1 pp and min cos >= 0.999). GPU load before/after timing: "
                 f"{r['gpu_load_before_speed']} / {r['gpu_load_after_speed']} (util %, MiB, all users).")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["data/models/kprpe_vitb_wf12m", "data/models/insightface_vit_l_wf42m"])
    ap.add_argument("--data", default="data/datasets/verification")
    ap.add_argument("--bins", nargs="+", default=list(BINS))
    ap.add_argument("--out", default="reports/E1_recognition")
    a = ap.parse_args()
    data = {b: load_bin(Path(a.data) / f"{b}.bin") for b in a.bins}
    res = {"experiment": "E1_recognition", "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "protocol": __doc__.split("\n\n")[1].strip(), "seeds": "none (deterministic protocol, no sampling)",
           "env": {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                   "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                   "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32},
           "datasets": {b: {"file": str(Path(a.data) / f"{b}.bin"), "sha256": sha256(Path(a.data) / f"{b}.bin"),
                            "images": int(len(d[0])), "pairs": int(len(d[1])), "genuine": int(d[1].sum()),
                            "impostor": int((~d[1]).sum())} for b, d in data.items()},
           "published": PUBLISHED, "models": {}}
    for m in a.models:
        res["models"][Path(m).name] = evaluate_model(Path(m), data)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1))
    (out / "summary.md").write_text(summary_md(res))
    print(summary_md(res))


if __name__ == "__main__":
    main()
