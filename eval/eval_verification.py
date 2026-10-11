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
    "insightface_r50_wf600k": {
        "source": "insightface model_zoo/README.md 'Recognition accuracy of python library model packs', row buffalo_l "
                  "(recognition model w600k_r50.onnx); CALFW/CPLFW not published", "acc": {"lfw": 99.83, "cfp_fp": 99.33, "agedb_30": 98.23}},
    "insightface_r100_glint360k": {
        "source": "nothing published for antelopev2 glintr100.onnx. 'reference' = arcface_torch Glint360k r100 re-training "
                  "(2022, glint360k_r100/training.log, step 332000 flip accuracy): not proven to be the same checkpoint",
        "acc": {}, "reference": {"lfw": 99.78, "cfp_fp": 99.10, "agedb_30": 98.65}},
    "adaface_ir101_wf12m": {
        "source": "CVLFace README 'Face Recognition Performance Board', row IR101 / AdaFace / WebFace12M "
                  "(github.com/mk-minchul/CVLface, cvlface/pretrained_models/README.md)",
        "acc": {"lfw": 99.82, "cfp_fp": 99.24, "agedb_30": 98.00, "calfw": 96.12, "cplfw": 94.57}},
}
TRAIN = {"kprpe_vitb_wf12m": "ViT-B KP-RPE, AdaFace, WebFace12M", "insightface_vit_l_wf42m": "ViT-L, CosFace, WebFace42M",
         "insightface_r50_wf600k": "IResNet-50, WebFace600K (buffalo_l)", "insightface_r100_glint360k":
         "IResNet-100, Glint360K (antelopev2)", "adaface_ir101_wf12m": "IResNet-101, AdaFace, WebFace12M"}
MODELS = ["data/models/kprpe_vitb_wf12m", "data/models/insightface_vit_l_wf42m", "data/models/insightface_r50_wf600k",
          "data/models/insightface_r100_glint360k", "data/models/adaface_ir101_wf12m"]


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
    fp32 / fp16 iterations are interleaved so both see the same load on the shared GPU; min = least-disturbed run.
    ONNX models: fp32 only; their memory lives in the ORT arena, invisible to torch (peak_mem_mb None)."""
    x, ms, peak = crops[:batch], {False: [], True: []} if rec.backend == "torch" else {False: []}, {}
    for fp16 in ms:
        rec.fp16 = fp16
        torch.cuda.reset_peak_memory_stats()
        for _ in range(3):  # warm-up
            rec.features(x)
        peak[fp16] = torch.cuda.max_memory_allocated() / 2 ** 20 if rec.backend == "torch" else None
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
    out = {"status": "evaluated", "version": rec.version, "model_dir": str(model_dir), "backend": rec.backend,
           "strict_load": "0 missing / 0 unexpected keys" if rec.backend == "torch" else f"ONNX Runtime {rec.providers}",
           "sha256": {k: sha256(model_dir / f) for k, f in meta["files"].items()},
           "sha256_in_model_json": meta["sha256"], "model_json_fp16": bool(meta["params"]["fp16"]), "metrics": {}, "speed": []}
    emb32, cos = {}, []
    for prec in ("fp32", "fp16") if rec.backend == "torch" else ("fp32",):
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
    if cos:
        cos = np.concatenate(cos)
        out["fp16_vs_fp32_cos"] = {"n_images": int(len(cos)), "mean": float(cos.mean()), "min": float(cos.min()),
                                   "p0.1": float(np.percentile(cos, 0.1))}
    crops = next(iter(data.values()))[0]
    out["gpu_load_before_speed"] = gpu_load()
    for batch, iters in ((1, 200), (32, 50), (128, 20)):
        out["speed"] += bench(rec, crops, batch, iters)
        print(out["speed"][-1], flush=True)
    out["gpu_load_after_speed"] = gpu_load()
    del rec
    torch.cuda.empty_cache()
    return out


def summary_md(res: dict) -> str:
    """Korean summary: accuracy table model x benchmark vs published, TAR@FAR=1e-3, noflip, fp16, speed."""
    cols = list(res["datasets"])
    names = {"lfw": "LFW", "cfp_fp": "CFP-FP", "agedb_30": "AgeDB-30", "calfw": "CALFW", "cplfw": "CPLFW"}
    ev = {m: r for m, r in res["models"].items() if r["status"] == "evaluated"}
    L = ["# E1 — 1:1 검증 정확도 (LFW, CFP-FP, AgeDB-30, CALFW, CPLFW)", "",
         f"생성 {res['created']}, `eval/eval_verification.py`. GPU {res['env']['gpu']} (다른 작업과 공유 — 속도는 참고치). "
         "프로토콜: InsightFace 10-fold 정확도 (사전 정렬 112x112 쌍, 폴드별 학습 9폴드에서 임계값 선택), "
         "*flip* = 원본 + 좌우반전 임베딩 합 (모델 카드가 공개하는 방식). 표의 값은 fp32.", "",
         "데이터: " + ", ".join(f"{names.get(k, k)} {v['pairs']}쌍 (동일인 {v['genuine']})" for k, v in res["datasets"].items()), "",
         "## 10-fold 정확도 (%, flip, fp32) — 측정값 (공개값, Δ)", "",
         "| 모델 | 구조 / 학습 데이터 | " + " | ".join(names.get(c, c) for c in cols) + " | 평균 |", "|" + "---|" * (len(cols) + 3)]
    for m, r in res["models"].items():
        if r["status"] != "evaluated":
            L.append(f"| {m} | {TRAIN.get(m, '')} | " + " | ".join("—" for _ in cols) + " | 가중치 없음 |")
            continue
        pub = PUBLISHED.get(m, {})
        ref = {**pub.get("reference", {}), **pub.get("acc", {})}
        cells = []
        for c in cols:
            v = r["metrics"]["fp32"][c]["acc_flip"]
            cells.append(f"{v:.2f}" + (f" ({ref[c]:.2f}, {v - ref[c]:+.2f})" if c in ref else ""))
        L.append(f"| {m} | {TRAIN.get(m, '')} | " + " | ".join(cells) +
                 f" | {np.mean([r['metrics']['fp32'][c]['acc_flip'] for c in cols]):.2f} |")
    L += ["", "공개값 출처: " + "; ".join(f"{m}: {PUBLISHED[m]['source']}" for m in ev if m in PUBLISHED), "",
          "## noflip 정확도 (%, fp32) — 실제 엔진 경로 (단일 임베딩, 반전 없음)", "",
          "| 모델 | " + " | ".join(names.get(c, c) for c in cols) + " | 평균 |", "|" + "---|" * (len(cols) + 2)]
    for m, r in ev.items():
        v = [r["metrics"]["fp32"][c]["acc_noflip"] for c in cols]
        L.append(f"| {m} | " + " | ".join(f"{x:.2f}" for x in v) + f" | {np.mean(v):.2f} |")
    L += ["", "## TAR @ FAR=1e-3 (%, flip, fp32; 세트별 전체 쌍, Wilson 95% CI)", "",
          "세트당 타인 쌍이 3000–3500개뿐이라 임계값이 타인 쌍 ~3개로 정해진다 (거친 값).", "",
          "| 모델 | " + " | ".join(names.get(c, c) for c in cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for m, r in ev.items():
        t = [r["metrics"]["fp32"][c]["tar_far1e-3_flip"] for c in cols]
        L.append(f"| {m} | " + " | ".join(f"{x['tar']:.2f} [{x['tar_ci95'][0]:.2f}, {x['tar_ci95'][1]:.2f}]" for x in t) + " |")
    L += ["", "## fp16 / 속도 (embed_aligned 전체 경로, batch 128, 공유 GPU)", "",
          "| 모델 | 런타임 | fp32 img/s (중앙값 / 최선) | fp16 img/s (중앙값 / 최선) | fp16 vs fp32 | MODEL.json 정밀도 |",
          "|---|---|---|---|---|---|"]
    for m, r in ev.items():
        sp = {s["precision"]: s for s in r["speed"] if s["batch"] == 128}
        f32 = f"{sp['fp32']['img_per_s']:.0f} / {sp['fp32']['img_per_s_best']:.0f}"
        f16 = f"{sp['fp16']['img_per_s']:.0f} / {sp['fp16']['img_per_s_best']:.0f}" if "fp16" in sp else "—"
        agree = "—"
        if "fp16_vs_fp32_cos" in r:
            c = r["fp16_vs_fp32_cos"]
            dacc = max(abs(r["metrics"]["fp16"][b][f"acc_{k}"] - r["metrics"]["fp32"][b][f"acc_{k}"])
                       for b in cols for k in ("flip", "noflip"))
            agree = f"min cos {c['min']:.5f}, max |Δacc| {dacc:.2f}pp"
        rt = "PyTorch" if r["backend"] == "torch" else "ONNX Runtime CUDA EP (fp32, TF32 conv)"
        L.append(f"| {m} | {rt} | {f32} | {f16} | {agree} | {'fp16' if r['model_json_fp16'] else 'fp32'} |")
    L += ["", "- 정밀도 규칙: max |Δacc| <= 0.1pp 이고 min cos >= 0.999 이면 fp16 autocast 허용 (E2는 MODEL.json 정밀도로 측정).",
          f"- GPU 부하 (측정 전/후, 모든 사용자): " + "; ".join(f"{m}: {r['gpu_load_before_speed']} / {r['gpu_load_after_speed']}"
                                                     for m, r in ev.items())]
    for m, r in res["models"].items():
        if r["status"] != "evaluated":
            L.append(f"- {m}: **평가 불가** — {r['reason']}. RFP 1순위 후보 InsightFace ViT-L/WebFace42M은 공개 체크포인트가 없다 "
                     "(insightface issue #1961). "
                     f"파일 확보 시 `python -m faceauth.models.recog {r['model_dir']} --write-sha256` 후 재실행.")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
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
