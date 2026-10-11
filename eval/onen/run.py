"""E2 (1:N open-set) + E6 (alignment ablation) orchestrator. Every step is resumable and skips finished work.

    D="docker run --rm --gpus '\"device=5\"' --shm-size 32g -u $(id -u):$(id -g) -e HOME=/tmp \
       -v /fdisk2/Codes/jhseo/faceauth:/app -w /app faceauth:dev"
    $D python -m eval.onen.run align [--part 0/3]        # SCRFD-34G (TRT fp16) on CelebA img_celeba + FFHQ -> crops
    $D python -m eval.onen.run protocol                  # seeded protocol on usable images -> reports/E2_1N/protocol.json
    $D python -m eval.onen.run digiface                  # crops of the DigiFace images the protocol selected
    $D python -m eval.onen.run embed --model kprpe_vitb_wf12m [--part 0/2]
    $D python -m eval.onen.run evaluate --model kprpe_vitb_wf12m   # -> reports/E2_1N/<model>/results.json
    $D python -m eval.onen.run e6 --models kprpe_vitb_wf12m insightface_r50_wf600k   # -> reports/E6_alignment/
    $D python -m eval.onen.run summary                   # -> reports/E2_1N/summary.md
    $D python -m eval.onen.run all --models ...          # all of the above, in order

Pipeline (= the deployed engine path): SCRFD 5 kps -> faceauth.common.align.norm_crop 112 -> Recognizer.embed_aligned
(MODEL.json precision, no flip). The labelled face of an image is the detection matching the dataset's annotated
landmarks (CelebA); FFHQ: largest face. Images without it are failures to acquire, counted and excluded (model metric).
Search = exact inner product over all templates, subject score = max over its templates (search service v1),
decision = top1 >= t AND top1 - top2 >= margin (api/policy.identify). Scratch: data/tmp/recog/onen/.
"""
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from eval.onen import datasets, embed, metrics, protocol

WORK = Path("data/tmp/recog/onen")
REPORT = Path("reports/E2_1N")
REPORT_E6 = Path("reports/E6_alignment")
MODELS = ["kprpe_vitb_wf12m", "insightface_r50_wf600k", "insightface_r100_glint360k", "adaface_ir101_wf12m"]
DEPLOYED = "kprpe_vitb_wf12m"  # FA_RECOG_MODEL default (IMPLEMENTATION.md §11)
PROTO = {"name": "celeba_v1", "identity": "celeba", "celeba_images": "img_celeba", "distractors": ["ffhq", "digiface1m"],
         "probe_frac": {"ffhq": 0.5}, "seed": 0, "enroll_frac": 0.5, "n_templates": 4, "min_images": 5,
         "max_probes": 0, "group_size": 3}
STORES = ("celeba", "ffhq", "digiface1m")  # crop store per dataset (paths start with "<name>/")
MARGINS = (0.0, 0.02, 0.05, 0.08, 0.1)
K_TPL, K_SUBJ = 16, 3     # template top-k (>= K_SUBJ x 4 templates/subject: exact top-3 subjects)
TARGET_FPIR, TARGET_G, TARGET_TPL = 1e-6, 1_000_000, 4  # RFP §4 / §1
TRT = [("TensorrtExecutionProvider", {"trt_fp16_enable": True, "trt_engine_cache_enable": True,
                                      "trt_engine_cache_path": str(WORK / "trt")}), "CUDAExecutionProvider"]


def _part(s):
    k, n = (int(x) for x in s.split("/"))
    return k, n


def _sha(path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def detector(ep="cuda", model_dir="data/models/scrfd_34g"):
    """SCRFD-34G on CUDA EP (fp32 reference; default) or TensorRT fp16 (engine build took > 20 min on the shared GPU)."""
    from faceauth.models.scrfd import SCRFD
    (WORK / "trt").mkdir(parents=True, exist_ok=True)  # ORT falls back to CUDA EP if the cache dir is missing
    want = TRT if ep == "trt" else ["CUDAExecutionProvider"]
    det = SCRFD(model_dir, providers=want)
    got = det.session.get_providers()[0]
    if got != ("TensorrtExecutionProvider" if ep == "trt" else "CUDAExecutionProvider"):  # ORT silently falls back
        raise RuntimeError(f"SCRFD wants {ep}, ORT gave {det.session.get_providers()}")
    return det


# ------------------------------------------------------------------------------------------------ data steps
def cmd_align(a):
    """--part k/N: run N processes with k = 0..N-1 after a single `align --init` (creates the stores)."""
    if a.init:
        for name in a.datasets:
            embed.init_store(datasets.load(name).path.tolist(), WORK / "crops" / name)
        return
    det = detector(a.ep)
    for name in reversed(a.datasets) if a.reverse else a.datasets:
        df = datasets.load(name)
        embed.align(df.path.tolist(), WORK / "crops" / name, det=det, hints=datasets.hints(name), part=a.part,
                    workers=a.workers, reverse=a.reverse)


def usable(name) -> pd.DataFrame:
    df = datasets.load(name)
    if name == "digiface1m":  # pre-aligned: every readable image is usable (checked when its crops are built)
        return df
    paths, _, det = embed.load_store(WORK / "crops" / name)
    if paths != df.path.tolist():
        raise ValueError(f"crop store {name} does not match the dataset listing")
    return df[det["ok"]]


def proto_dir() -> Path:
    return WORK / "protocol" / PROTO["name"]


# ------------------------------------------------------------------------------------------------ identity cleaning
# CelebA identity labels are noisy (one person split over several ids, ids mixing people, the same photo under two
# ids) and FFHQ contains celebrities and repeated people of one Flickr account. Uncleaned, ~2.5% of the non-mated
# searches score like mated ones and set every FPIR <= 1e-3 threshold. Cleaning uses the consensus of ALL evaluated
# models (mean cosine over the 4 recognizers), never a single model, with thresholds read off the ensemble score
# distributions (gap between the impostor bulk and the duplicate mode) and checked visually (reports/E2_1N/cleaning.json).
CLEAN_NOTE = ("Consensus of the 4 evaluated recognizers (mean cosine). identity: CelebA identities whose centroids reach "
              "tau_identity form components, all but the largest id are dropped (split identities). ffhq_celebrity: FFHQ "
              "images reaching tau_ffhq_celebrity against any CelebA centroid are dropped. probe: a probe whose best OTHER "
              "gallery subject (excluding its own) of the same source reaches tau_probe is dropped (mislabelled / repeated "
              "person); thresholds sit in the valley between the impostor bulk and the duplicate mode of the consensus "
              "score histograms (stored here); celeba<->ffhq: a few celebrities left in FFHQ (no bulk tail above 0.475).")
CLEAN = {"models": MODELS, "tau_identity": 0.45, "tau_ffhq_celebrity": 0.45,
         "tau_probe": {"celeba->celeba": 0.45, "ffhq->ffhq": 0.60, "celeba->ffhq": 0.50, "ffhq->celeba": 0.50},
         "ffhq_unit": "Flickr account (ffhq-dataset-v2.json)"}


def store_emb(model, name):
    n = len((WORK / "crops" / name / "paths.txt").read_text().split("\n"))
    return np.memmap(WORK / "emb" / model / f"{name}.f32", np.float32, "r").reshape(n, -1)


def ffhq_accounts() -> dict:
    """FFHQ image path -> Flickr account (photo_url path segment) from the official metadata."""
    meta = json.loads((datasets.DATA / "ffhq/archives/ffhq-dataset-v2.json").read_text())
    return {f"ffhq/images256/{int(k):05d}.png": v["metadata"]["photo_url"].rstrip("/").split("/")[4] for k, v in meta.items()}


def clean_identities() -> dict:
    """-> {"celeba_drop": [identity], "ffhq_drop": [path], stats}: consensus centroid duplicates (CelebA ids whose
    centroids reach tau_identity are one component; all but its largest id are dropped) and FFHQ images that match any
    CelebA identity centroid (celebrities)."""
    import torch
    dev = torch.device("cuda")
    cel = datasets.load("celeba")
    ok = embed.load_store(WORK / "crops" / "celeba")[2]["ok"]
    ids, inv = np.unique(cel.subject.to_numpy()[ok], return_inverse=True)
    nimg = np.bincount(inv)
    ff = datasets.load("ffhq")
    okf = embed.load_store(WORK / "crops" / "ffhq")[2]["ok"]
    S = torch.zeros(len(ids), len(ids), device=dev)
    F = torch.zeros(int(okf.sum()), len(ids), device=dev)
    for m in CLEAN["models"]:
        C = np.zeros((len(ids), 512))
        np.add.at(C, inv, store_emb(m, "celeba")[ok])
        C = torch.from_numpy((C / np.linalg.norm(C, axis=1, keepdims=True)).astype(np.float32)).to(dev)
        S += C @ C.T / len(CLEAN["models"])
        F += torch.from_numpy(np.ascontiguousarray(store_emb(m, "ffhq")[okf])).to(dev) @ C.T / len(CLEAN["models"])
    S.fill_diagonal_(-1)
    a, b = (torch.triu(S, 1) >= CLEAN["tau_identity"]).nonzero().cpu().numpy().T
    parent = np.arange(len(ids))

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for x, y in zip(a, b):
        parent[root(x)] = root(y)
    comp = np.array([root(x) for x in range(len(ids))])
    keep = {}
    for i in np.argsort(-nimg, kind="stable"):  # largest id of each component survives
        keep.setdefault(comp[i], i)
    drop = [ids[i] for i in range(len(ids)) if keep[comp[i]] != i]
    fsim = F.max(1).values.cpu().numpy()
    ffhq_drop = ff.path.to_numpy()[okf][fsim >= CLEAN["tau_ffhq_celebrity"]].tolist()
    return {"celeba_drop": sorted(drop), "ffhq_drop": sorted(ffhq_drop),
            "stats": {"celeba_identities": int(len(ids)), "duplicate_pairs": int(len(a)),
                      "components_with_duplicates": int(len({comp[i] for i in np.unique(np.r_[a, b])})),
                      "celeba_identities_dropped": len(drop), "ffhq_images": int(okf.sum()),
                      "ffhq_celebrity_dropped": len(ffhq_drop)}}


def ensemble_top(t, probe_mask, gal_mask, k=6, exclude=None) -> tuple:
    """Consensus top-1 of each probe over the gallery rows: mean over CLEAN models of the subject score (max over the
    subject's templates) for the union of every model's top-k subjects -> (score, subject). exclude[i] = a subject
    never returned for probe i (its own subject: 'best OTHER subject')."""
    from eval.onen.search import Searcher, subject_scores, subject_topk, template_table
    s = Searcher()
    gsub = t.subject.to_numpy()[gal_mask]
    subjects, gs = np.unique(gsub, return_inverse=True)
    table = template_table(gs, len(subjects))
    Es = {m: gather(m, t) for m in CLEAN["models"]}
    cands = []
    for m, E in Es.items():
        D, I = s.knn(E[gal_mask], E[probe_mask], 4 * k)
        cands.append(subject_topk(D, I, gs, k)[1])
    cands = np.concatenate(cands, 1)
    score = np.zeros(cands.shape)
    for m, E in Es.items():
        P, G = E[probe_mask], E[gal_mask]
        score += np.stack([subject_scores(P, cands[:, j], table, G) for j in range(cands.shape[1])], 1) / len(Es)
    score[cands < 0] = -np.inf
    if exclude is not None:
        score[cands == pd.Index(subjects).get_indexer(np.asarray(exclude, object))[:, None]] = -np.inf
    j = np.argmax(score, 1)
    return score[np.arange(len(j)), j], subjects[np.maximum(cands[np.arange(len(j)), j], 0)]


def cmd_protocol(a):
    """Seeded protocol on usable, consensus-cleaned images (see CLEAN): identity-level drops, FFHQ split by Flickr
    account, then probes whose consensus best OTHER subject is a same-source gallery subject above tau_probe."""
    out = proto_dir()
    if (out / "images.csv.gz").exists():
        print(f"{out} exists")
        return
    cl = clean_identities()
    ident = usable("celeba")
    ident = ident[~ident.subject.isin(cl["celeba_drop"])]
    ff = usable("ffhq")
    ff = ff[~ff.path.isin(cl["ffhq_drop"])].copy()
    ff["unit"] = ff.path.map(ffhq_accounts())
    p0 = protocol.build(ident, [("ffhq", ff), ("digiface1m", usable("digiface1m"))], PROTO["probe_frac"],
                        PROTO["seed"], PROTO["enroll_frac"], PROTO["n_templates"], PROTO["min_images"],
                        PROTO["max_probes"], PROTO["group_size"])
    p0["dataset"] = p0.subject.str.split(":").str[0]
    probe = p0.role.isin(["mated", "nonmated"]).to_numpy()
    own = np.where(p0.role.to_numpy()[probe] == "mated", p0.subject.to_numpy()[probe], None)
    sc, sub = ensemble_top(p0, probe, p0.role.isin(["template", "distractor"]).to_numpy(), exclude=own)
    pair = p0.dataset.to_numpy()[probe] + "->" + pd.Series(sub).str.split(":").str[0].to_numpy()
    tau = np.array([CLEAN["tau_probe"].get(x, np.inf) for x in pair])
    drop = np.zeros(len(p0), bool)
    drop[np.flatnonzero(probe)[sc >= tau]] = True
    role_p = p0.role.to_numpy()[probe]
    hist = {f"{r}:{x}": np.histogram(sc[(pair == x) & (role_p == r)], np.arange(0, 1.0001, 0.025))[0].tolist()
            for r in ("mated", "nonmated") for x in np.unique(pair)}
    cl["stats"].update({"probes_dropped": {f"{r}:{x}": int(((pair == x) & (role_p == r) & (sc >= tau)).sum())
                                           for r in ("mated", "nonmated") for x in CLEAN["tau_probe"]},
                        "consensus_best_other_subject_hist (bins of 0.025 from 0)": hist})
    lost = set(p0.group[drop & (p0.group >= 0)])
    p = p0[~drop].drop(columns="dataset").copy()
    p.loc[p.group.isin(lost), "group"] = -1  # a group that lost a member is dissolved
    key = (p.subject + "|" + p.role + "|" + p.group.astype(str)).where(p.group >= 0)
    p["group"] = pd.factorize(key, sort=True)[0]
    p = p.reset_index(drop=True)
    digi = WORK / "crops" / "digiface1m" / "paths.txt"
    if digi.exists() and sorted(p.path[p.subject.str.startswith("digiface1m:")]) != digi.read_text().split("\n"):
        raise ValueError("DigiFace selection differs from the pre-built crop store (per-dataset seed streams broken?)")
    out.mkdir(parents=True, exist_ok=True)
    p.to_csv(out / "images.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    fta = {}
    for n in ("celeba", "ffhq"):
        z = json.loads((WORK / "crops" / n / "align.json").read_text())
        fta[n] = {**z, "fta_rate": 1 - z["ok"] / z["n"]}
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "cleaning.json").write_text(json.dumps({"rules": CLEAN, "note": CLEAN_NOTE, **cl["stats"],
                                                      "celeba_identities_dropped": cl["celeba_drop"]}, indent=1) + "\n")
    meta = {**PROTO, "created": _now(), "counts": protocol.counts(p), "failure_to_acquire": fta,
            "cleaning": "reports/E2_1N/cleaning.json", "counts_before_probe_cleaning": protocol.counts(p0),
            "images_csv": str(out / "images.csv.gz"), "images_csv_sha256": _sha(out / "images.csv.gz"),
            "dataset_json_sha256": {n: _sha(datasets.DATA / n / "DATASET.json") if (datasets.DATA / n / "DATASET.json").exists()
                                    else None for n in ["celeba", *PROTO["distractors"]]}}
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "protocol.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta["counts"], indent=1))


def load_protocol() -> pd.DataFrame:
    t = pd.read_csv(proto_dir() / "images.csv.gz", keep_default_na=False)
    t["dataset"] = t.subject.str.split(":").str[0]
    return t


def digiface_paths() -> list:
    """DigiFace images of the protocol. Its own seeded stream makes the selection independent of the CelebA / FFHQ
    detection results, so DigiFace crops + embeddings can be built before the protocol (cmd_protocol re-checks)."""
    if (proto_dir() / "images.csv.gz").exists():
        t = load_protocol()
        return sorted(t.path[t.dataset == "digiface1m"])
    p = protocol.build(pd.DataFrame({"subject": pd.Series(dtype=str), "path": pd.Series(dtype=str)}),
                       [(n, datasets.load(n)) for n in PROTO["distractors"]], PROTO["probe_frac"], PROTO["seed"],
                       PROTO["enroll_frac"], PROTO["n_templates"], PROTO["min_images"], PROTO["max_probes"],
                       PROTO["group_size"])
    return sorted(p.path[p.subject.str.startswith("digiface1m:")])


def cmd_digiface(a):
    paths = digiface_paths()
    if a.init:
        return embed.init_store(paths, WORK / "crops" / "digiface1m")
    embed.align(paths, WORK / "crops" / "digiface1m", part=a.part, workers=a.workers)


def cmd_embed(a):
    """Exit status 3 while some store is not completely aligned yet (re-run: `until ... embed ...; do sleep 300; done`)."""
    from faceauth.models.recog import Recognizer
    pending = []
    for m in a.models:
        rec = Recognizer(Path("data/models") / m, device="cuda", batch_size=a.batch)
        for name in a.stores:
            if embed.embed(WORK / "crops" / name, rec, WORK / "emb" / m / f"{name}.f32", part=a.part) is None:
                pending.append(f"{m}/{name}")
    if pending:
        print("pending (alignment not finished):", pending, flush=True)
        raise SystemExit(3)


def gather(model, t, stores=STORES, suffix="", allow_missing=False) -> np.ndarray:
    """Protocol rows -> (n, dim) embeddings from the per-dataset stores (all rows must have one unless allow_missing:
    then rows without one are NaN)."""
    dim = json.loads((Path("data/models") / model / "MODEL.json").read_text())["dim"]
    E = np.full((len(t), dim), np.nan, np.float32)
    for name in stores:
        rows = np.flatnonzero((t.dataset == name).to_numpy())
        if not len(rows):
            continue
        paths = (WORK / "crops" / (name + suffix) / "paths.txt").read_text().split("\n")
        idx = pd.Index(paths).get_indexer(t.path.to_numpy()[rows])
        if (idx < 0).any():
            raise ValueError(f"{(idx < 0).sum()} protocol images missing from store {name}{suffix}")
        emb = np.memmap(WORK / "emb" / model / f"{name}{suffix}.f32", np.float32, "r").reshape(len(paths), dim)
        o = np.argsort(idx)
        E[rows[o]] = emb[idx[o]]
    if not allow_missing and not np.isfinite(E).all():
        raise ValueError(f"{model}: {(~np.isfinite(E).all(1)).sum()} protocol images without an embedding")
    return E


def l2n(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def group_probes(E, t, mask):
    """Multi-image probe variant: normalised mean of each full group's embeddings -> (probes, subject per group)."""
    g = t.group.to_numpy()
    sel = mask & (g >= 0)
    ids, inv = np.unique(g[sel], return_inverse=True)
    P = np.zeros((len(ids), E.shape[1]), np.float64)
    np.add.at(P, inv, E[sel])
    subj = pd.Series(t.subject.to_numpy()[sel]).groupby(inv).first().to_numpy()
    return l2n(P), subj


# ------------------------------------------------------------------------------------------------ E2 evaluation
PRIMARY = "celeba"  # non-mated pool for FPIR: adult, identity-labelled (cleaned) CelebA non-enrolled identities. FFHQ
# probes (Flickr: many infants/children, people repeated across accounts, no identity labels) are a secondary pool.


def at_threshold(t, m_top1, m_top2, m_rank1, n_top1, n_top2, margin, m_cl, n_cl) -> dict:
    nm, hit, mis = metrics.outcome_scores(m_top1, m_top2, m_rank1, n_top1, n_top2, margin)
    M, N = len(hit), len(nm)
    fp, h, mi = int((nm >= t).sum()), int((hit >= t).sum()), int((mis >= t).sum())
    return {"threshold": float(t), "margin": margin, "fp": fp, "n_nonmated": N, "fpir": fp / N,
            "fpir_ci95": metrics.cp_ci(fp, N), "fpir_upper95": metrics.upper_bound(fp, N),
            "fp_identities": int(len(np.unique(np.asarray(n_cl)[nm >= t]))),
            "fnir": (M - h) / M, "fnir_ci95": metrics.cp_ci(M - h, M),
            "fnir_boot_ci95": metrics.cluster_boot_ci(hit < t, m_cl, 1000, 0),
            "misid": mi, "misid_rate": mi / M, "misid_ci95": metrics.cp_ci(mi, M),
            "misid_subjects": int(len(np.unique(np.asarray(m_cl)[mis >= t])))}


def operating_point(r, t, margin) -> dict:
    """Measured numbers at a fixed (threshold, margin) from one gallery's per-search arrays: FPIR on the primary pool,
    plus FP counts of the other pools and the dataset of each mis-identified gallery subject."""
    pri = r["n_src"] == PRIMARY
    o = at_threshold(t, r["m_top1"], r["m_top2"], r["m_rank1"], r["n_top1"][pri], r["n_top2"][pri], margin,
                     r["m_cl"], r["n_cl"][pri])
    nm, _, mis = metrics.outcome_scores(r["m_top1"], r["m_top2"], r["m_rank1"], r["n_top1"], r["n_top2"], margin)
    o["fp_by_probe_dataset"] = {k: int(((nm >= t) & (r["n_src"] == k)).sum()) for k in np.unique(r["n_src"])}
    o["misid_by_gallery_dataset"] = {k: int(((mis >= t) & (r["m_wrong_ds"] == k)).sum())
                                     for k in np.unique(r["m_wrong_ds"]) if k}
    return o


def evaluate_model(model, B_max=1000, B_other=200):
    from eval.onen.search import Searcher, subject_scores, subject_topk, template_table
    t0 = time.time()
    t = load_protocol()
    E = gather(model, t)
    role, ds, order = t.role.to_numpy(), t.dataset.to_numpy(), t.order.to_numpy()
    gal_rows = np.isin(role, ("template", "distractor"))
    subjects = np.unique(t.subject.to_numpy()[gal_rows])
    sidx = pd.Index(subjects).get_indexer(t.subject.to_numpy())          # -1 = not enrolled
    subj_ds = pd.Series(subjects).str.split(":").str[0].to_numpy()
    mated, nonm = role == "mated", role == "nonmated"
    n_dist = int(order.max()) + 1
    n_ffhq = int(t[(role == "distractor") & (ds == "ffhq")].subject.nunique())
    sizes = sorted(x for x in {0, 5000, 15000, n_ffhq, n_ffhq + 35000, n_dist} if x <= n_dist)

    gp_m, gs_m = group_probes(E, t, mated)
    gp_n, gs_n = group_probes(E, t, nonm)
    variants = {
        "single": (E[mated], t.subject.to_numpy()[mated], E[nonm], t.subject.to_numpy()[nonm], ds[nonm]),
        "group3": (gp_m, gs_m, gp_n, gs_n, pd.Series(gs_n).str.split(":").str[0].to_numpy()),
    }
    s = Searcher()
    res = {"experiment": "E2_1N", "model": model, "created": _now(), "protocol": PROTO,
           "protocol_sha256": _sha(proto_dir() / "images.csv.gz"), "margins": MARGINS, "galleries": [],
           "embeddings": {n: json.loads((WORK / "emb" / model / f"{n}.json").read_text()) for n in STORES},
           "notes": "FNIR = mated searches not (rank-1 mate AND accepted); FPIR = accepted non-mated searches of the "
                    "primary pool (%s non-enrolled identities; other pools under 'other_pools', FP counts of every pool at "
                    "the primary threshold under fp_by_probe_dataset); misid = mated searches accepted for another "
                    "gallery subject. CIs: Clopper-Pearson (independent searches) and cluster bootstrap over subjects / "
                    "non-mated identities (B=%d at the largest gallery, %d otherwise)." % (PRIMARY, B_max, B_other)}
    raw = {}
    for D in sizes:
        gm = (role == "template") | ((role == "distractor") & (order < D))
        G, Gs = E[gm], sidx[gm]
        table = template_table(Gs, len(subjects))
        gres = {"distractor_subjects": D, "subjects": int(len(np.unique(Gs))), "templates": int(gm.sum()),
                "templates_by_dataset": {k: int(v) for k, v in zip(*np.unique(ds[gm], return_counts=True))},
                "subjects_by_dataset": {k: int(v) for k, v in zip(*np.unique(subj_ds[np.unique(Gs)], return_counts=True))},
                "variants": {}}
        B = B_max if D == n_dist else B_other
        for v, (Pm, sm, Pn, sn, nsrc) in variants.items():
            mate = pd.Index(subjects).get_indexer(sm)
            Dk, Ik = s.knn(G, np.concatenate([Pm, Pn]), K_TPL)
            sc, sid = subject_topk(Dk, Ik, Gs, K_SUBJ)
            M = len(Pm)
            m1, m2, mr = sc[:M, 0], sc[:M, 1], sid[:M, 0] == mate
            n1, n2 = sc[M:, 0], sc[M:, 1]
            pri = nsrc == PRIMARY
            ev = metrics.evaluate(m1, m2, mr, sm, n1[pri], n2[pri], sn[pri], MARGINS, B=B, seed=0)
            ev["nonmated_pool"] = PRIMARY
            pools = {"ffhq": nsrc == "ffhq", "all": np.ones(len(nsrc), bool)}
            ev["other_pools"] = {k: metrics.evaluate(m1, m2, mr, sm, n1[sel], n2[sel], sn[sel], (0.0,), B=0)
                                 for k, sel in pools.items() if sel.any() and not pri.all()}
            mate_sc = subject_scores(Pm, mate, table, G)
            ev["mate_score"] = {"mean": float(mate_sc.mean()), "p01": float(np.quantile(mate_sc, 0.01))}
            # breakdowns at each grid threshold: non-mated source, dataset of the wrongly identified subject
            wrong_ds = np.where(sid[:M, 0] >= 0, subj_ds[np.maximum(sid[:M, 0], 0)], "")
            for mg, block in ev["margins"].items():
                nm, hit, mis = metrics.outcome_scores(m1, m2, mr, n1, n2, float(mg))
                for row in block["grid"]:
                    thr = row["threshold"]  # FP of every pool at the primary-pool threshold
                    row["fp_by_probe_dataset"] = {k: int(((nm >= thr) & (nsrc == k)).sum()) for k in np.unique(nsrc)}
                    row["misid_by_gallery_dataset"] = {k: int(((mis >= thr) & (wrong_ds == k)).sum())
                                                       for k in np.unique(subj_ds)}
            ev["nonmated_by_dataset"] = {k: {"searches": int((nsrc == k).sum()),
                                             "identities": int(len(np.unique(sn[nsrc == k])))} for k in np.unique(nsrc)}
            gres["variants"][v] = ev
            raw[(D, v)] = {"m_top1": m1, "m_top2": m2, "m_rank1": mr, "m_cl": sm, "n_top1": n1, "n_top2": n2,
                           "n_cl": sn, "n_src": nsrc, "mate": mate_sc,
                           "m_sid": sid[:M, 0], "n_sid": sid[M:, 0], "m_wrong_ds": wrong_ds}
            print(f"[{model}] G={gres['subjects']} {v}: " + ", ".join(
                f"FPIR {r['target_fpir']:.0e}: t={r['threshold']:.3f} FNIR={r['fnir']:.4f}"
                for r in ev["margins"]["0.0"]["grid"][:5]) + f" ({time.time() - t0:.0f}s)", flush=True)
        res["galleries"].append(gres)

    # -------- impostor tails per template source (non-mated single probes x every template of the source)
    Pn_all, nsrc_all = E[nonm], ds[nonm]
    hists = {}
    for src in ("celeba", "ffhq", "digiface1m"):
        gsel = gal_rows & (ds == src)
        for psrc in np.unique(nsrc_all):
            hists[(psrc, src)] = s.hist(Pn_all[nsrc_all == psrc], E[gsel])
    np.savez_compressed(WORK / "emb" / model / "impostor_hist.npz",
                        **{f"{p}__{g}": h for (p, g), h in hists.items()})
    tails = {}
    for src in ("celeba", "ffhq", "digiface1m"):  # primary probe pool only (as the measured FPIR)
        tails[src] = metrics.ImpostorTail(hists[(PRIMARY, src)])
    real_counts = hists[(PRIMARY, "celeba")] + hists[(PRIMARY, "ffhq")]
    real = metrics.ImpostorTail(real_counts)
    res["impostor_tails"] = {
        "per_comparison_note": "p(t) = P(score >= t) for one probe x template comparison: non-mated single probes of the "
                               "primary pool (CelebA non-enrolled identities) x templates of the source",
        "real (celeba+ffhq templates)": real.summary(),
        **{src: tl.summary() for src, tl in tails.items()},
        "p_at": {f"{x:.2f}": {"real": float(real.p_emp(x)), **{src: float(tl.p_emp(x)) for src, tl in tails.items()}}
                 for x in (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5)},
        "by_probe_dataset_p_at_0.30": {f"{p}->{g}": float(h[int(round(1.3 / metrics.HIST_W)):].sum() / h.sum())
                                       for (p, g), h in hists.items()}}

    # -------- model check at the measured galleries: predicted vs measured FPIR / FNIR (margin 0, single probes)
    checks = []
    for gres in res["galleries"]:
        D = gres["distractor_subjects"]
        tb = gres["templates_by_dataset"]
        comps = [(tails[k], tb.get(k, 0)) for k in tails]
        r = raw[(D, "single")]
        for row in gres["variants"]["single"]["margins"]["0.0"]["grid"]:
            if row["fp"] >= 3:  # measured value with some support
                checks.append({"gallery_subjects": gres["subjects"], "threshold": row["threshold"],
                               "fpir_measured": row["fpir"], "fpir_model": metrics.fpir_mix(row["threshold"], comps),
                               "fnir_measured": row["fnir"],
                               "fnir_model": metrics.fnir_model(r["mate"], row["threshold"], comps, 0.0)})
    res["model_check"] = {"note": "independence model with per-source empirical/GPD tails vs measurement on the same "
                                  "gallery (single probes, margin 0)", "rows": checks}

    # -------- extrapolation to 1M subjects x 4 templates (real-face tail only; DigiFace is synthetic)
    n_tpl = TARGET_G * TARGET_TPL
    mate = raw[(n_dist, "single")]["mate"]
    ext = metrics.extrapolate(real, mate, n_tpl, TARGET_FPIR, MARGINS)
    sens = []
    for nx in (10_000, 100_000, 1_000_000):
        if nx <= real.n:
            tl = metrics.ImpostorTail(real_counts, n_exceed=nx)
            tt = tl.t_for_p(metrics.p_for_fpir(TARGET_FPIR, n_tpl))
            sens.append({"n_exceed": nx, "u": tl.u, "gpd_shape": float(tl.c), "threshold": tt,
                         "fnir_margin0": metrics.fnir_model(mate, tt, [(tl, n_tpl)], 0.0)})
    ext["gpd_threshold_sensitivity"] = sens
    res["extrapolation_1M_x4"] = ext

    # -------- operating points: extrapolated threshold (and +-0.02) measured on every gallery, all margins
    t_ext = ext["threshold"]
    ops = []
    for tt in sorted({round(t_ext, 4), round(t_ext - 0.02, 4), round(t_ext + 0.02, 4)}):
        for D, v in [(g["distractor_subjects"], "single") for g in res["galleries"]] + [(n_dist, "group3")]:
            for mg in MARGINS:
                ops.append({"gallery_subjects": next(g["subjects"] for g in res["galleries"]
                                                     if g["distractor_subjects"] == D), "variant": v,
                            **operating_point(raw[(D, v)], tt, mg)})
    res["operating_points"] = ops

    # -------- highest-scoring non-mated searches (for manual inspection: label noise / duplicates / infants?)
    r = raw[(n_dist, "single")]
    nm_paths = t.path.to_numpy()[nonm]
    sub_paths = t[gal_rows].groupby("subject").path.first()
    for name, sel in (("top_nonmated_primary", r["n_src"] == PRIMARY), ("top_nonmated_all", np.ones(len(nm_paths), bool))):
        top = np.flatnonzero(sel)[np.argsort(-r["n_top1"][sel])[:20]]
        res[name] = [{"probe": nm_paths[i], "probe_identity": r["n_cl"][i], "score": float(r["n_top1"][i]),
                      "second": float(r["n_top2"][i]), "gallery_subject": subjects[r["n_sid"][i]],
                      "gallery_example": sub_paths[subjects[r["n_sid"][i]]]} for i in top]
    res["seconds"] = round(time.time() - t0)
    out = REPORT / model
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1, default=_json) + "\n")
    for (D, v), r in raw.items():  # per-search arrays: report.py recomputes operating points from them
        np.savez_compressed(WORK / "emb" / model / f"searches_{D}_{v}.npz",
                            **{k: (np.asarray(x, str) if np.asarray(x).dtype == object else x) for k, x in r.items()})
    return res


def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def cmd_evaluate(a):
    for m in a.models:
        evaluate_model(m, a.boot, a.boot_small)


# ------------------------------------------------------------------------------------------------ E6
E6 = {"seed": 1, "enrolled": 400, "nonenrolled": 400}  # FaRL (2 passes of ViT-B/448 per image) dominates the cost


def e6_rows(t) -> pd.DataFrame:
    """Seeded CelebA-only subset of the E2 protocol: enrolled subjects (templates + mated probes) and non-enrolled
    identities (non-mated probes)."""
    rng = np.random.default_rng(E6["seed"])
    c = t[t.dataset == "celeba"]
    enr = np.sort(c[c.role == "template"].subject.unique())
    non = np.sort(c[c.role == "nonmated"].subject.unique())
    keep = set(rng.choice(enr, E6["enrolled"], replace=False)) | set(rng.choice(non, E6["nonenrolled"], replace=False))
    return c[c.subject.isin(keep)].copy()


def cmd_e6_align(a):
    from faceauth.models.farl import FaRLLandmarks, kps5_from_wflw98
    t = load_protocol()
    sub = e6_rows(t)
    paths = sorted(sub.path)
    if a.variant == "scrfd320":   # deployed server detector (scrfd_34g@2, 320 input): re-detect -> its own 5 kps
        embed.align(paths, WORK / "crops" / "celeba_scrfd320", det=detector(a.ep, "data/models/scrfd_34g_320"),
                    hints=datasets.hints("celeba"), shard=2048, part=a.part, workers=a.workers)
        return
    src_paths, _, det = embed.load_store(WORK / "crops" / "celeba")
    kps = det["kps"][pd.Index(src_paths).get_indexer(paths)]
    # bf16: ~1.7x faster, WFLW NME unchanged to 0.01 (FaRLLandmarks docstring / reports/E5_landmarks)
    farl = FaRLLandmarks("data/models/farl_wflw_448", device="cuda", batch_size=8, bf16=True)

    def refine(imgs, k):  # terminal path: FaRL from the SCRFD kps, refine=1 (FaRLLandmarks.predict default)
        pts = farl.predict_batch(imgs, [kk[None] for kk in k])
        return kps5_from_wflw98(np.concatenate(pts)).astype(np.float32)

    embed.align(paths, WORK / "crops" / "celeba_farl", kps_in=kps, refine=refine, shard=2048, part=a.part,
                workers=a.workers, refine_batch=64)


def cmd_e6_embed(a):
    from faceauth.models.recog import Recognizer
    for m in a.models:
        rec = Recognizer(Path("data/models") / m, device="cuda", batch_size=a.batch)
        embed.embed(WORK / "crops" / f"celeba_{a.variant}", rec, WORK / "emb" / m / f"celeba_{a.variant}.f32")


def tar_at_far(gen, imp_hist, fars=(1e-3, 1e-4, 1e-5, 1e-6)) -> list:
    """1:1 TAR at the histogram edge where the impostor survival first drops to <= far (FAR actual reported)."""
    n = imp_hist.sum()
    surv = np.append(np.cumsum(imp_hist[::-1])[::-1], 0) / n
    edges = -1 + metrics.HIST_W * np.arange(metrics.HIST_BINS + 1)
    out = []
    for f in fars:
        if f * n < 1:
            continue
        i = int(np.argmax(surv <= f))
        k = int((np.asarray(gen) >= edges[i]).sum())
        out.append({"far_target": f, "threshold": float(edges[i]), "far": float(surv[i]), "tar": k / len(gen),
                    "tar_ci95": metrics.cp_ci(k, len(gen)), "n_genuine": int(len(gen)), "n_impostor": int(n)})
    return out


def cmd_e6_evaluate(a):
    from eval.onen.search import Searcher, subject_topk, template_table
    t = load_protocol()
    sub_all = e6_rows(t).reset_index(drop=True)
    s = Searcher()
    res = {"experiment": "E6_alignment", "created": _now(), "subset": E6, "protocol": PROTO["name"],
           "protocol_sha256": _sha(proto_dir() / "images.csv.gz"),
           "alignments": {"scrfd": "SCRFD-34G 5 kps (ORT CUDA EP fp32, 640 input) -> norm_crop 112",
                          "farl": "FaRL ViT-B/16 448 WFLW-98 from the same SCRFD kps (refine=1 as the terminal; bf16) -> "
                                  "kps5_from_wflw98 -> norm_crop 112",
                          "scrfd320": "SCRFD-34G re-targeted to 320 input (scrfd_34g@2, deployed server detector) "
                                      "5 kps -> norm_crop 112"},
           "variant": a.variant,
           "counts": {r: int((sub_all.role == r).sum()) for r in ("template", "mated", "nonmated")}, "models": {}}
    for m in a.models:
        # paired: images where the variant found no face are dropped for BOTH alignments (counted below)
        ok = np.isfinite(gather(m, sub_all, stores=("celeba",), suffix=f"_{a.variant}", allow_missing=True)).all(1)
        sub = sub_all[ok].reset_index(drop=True)
        role = sub.role.to_numpy()
        res["models"][m] = {"images_without_face_in_variant": int((~ok).sum()),
                            "by_role": {r: int(((sub_all.role == r) & ~ok).sum()) for r in ("template", "mated", "nonmated")}}
        hits = {}
        for al, suffix in (("scrfd", ""), (a.variant, f"_{a.variant}")):
            E = gather(m, sub, stores=("celeba",), suffix=suffix)
            gm = role == "template"
            subjects = np.unique(sub.subject[gm])
            sidx = pd.Index(subjects).get_indexer(sub.subject)
            G, Gs = E[gm], sidx[gm]
            mated, nonm = role == "mated", role == "nonmated"
            Dk, Ik = s.knn(G, np.concatenate([E[mated], E[nonm]]), K_TPL)
            sc, sid = subject_topk(Dk, Ik, Gs, K_SUBJ)
            M = int(mated.sum())
            ev = metrics.evaluate(sc[:M, 0], sc[:M, 1], sid[:M, 0] == sidx[mated], sub.subject[mated].to_numpy(),
                                  sc[M:, 0], sc[M:, 1], sub.subject[nonm].to_numpy(), (0.0, 0.05),
                                  grid=(1e-2, 1e-3, 1e-4), B=1000, seed=0, det_points=40)
            # 1:1: genuine = mated probe x each own template; impostor = non-mated probe x every template
            table = template_table(Gs, len(subjects))
            rows = table[sidx[mated]]
            gen = np.einsum("nkd,nd->nk", G[np.maximum(rows, 0)], E[mated])[rows >= 0]
            ev["one_to_one"] = tar_at_far(gen, s.hist(E[nonm], G))
            res["models"][m][al] = ev
            nm, hit, _ = metrics.outcome_scores(sc[:M, 0], sc[:M, 1], sid[:M, 0] == sidx[mated], sc[M:, 0], sc[M:, 1])
            hits[al] = {r["target_fpir"]: hit >= r["threshold"] for r in ev["margins"]["0.0"]["grid"]}
            g = ev["margins"]["0.0"]["grid"]
            print(m, al, [(r["target_fpir"], round(r["fnir"], 4)) for r in g],
                  [(x["far_target"], round(x["tar"], 4)) for x in ev["one_to_one"]], flush=True)
        # paired comparison on the same mated probes (each alignment at its own threshold for the FPIR target)
        res["models"][m]["paired_mcnemar"] = {}
        for f, h1 in hits["scrfd"].items():
            h2 = hits[a.variant][f]
            b, c = int((h1 & ~h2).sum()), int((~h1 & h2).sum())
            res["models"][m]["paired_mcnemar"][str(f)] = {
                "hit_scrfd_only": b, f"hit_{a.variant}_only": c,
                "p_exact": float(binomtest(b, b + c).pvalue) if b + c else 1.0}
    a_json = WORK / "crops" / f"celeba_{a.variant}" / "align.json"
    res[f"{a.variant}_align"] = json.loads(a_json.read_text()) if a_json.exists() else None
    REPORT_E6.mkdir(parents=True, exist_ok=True)
    out = "results.json" if a.variant == "farl" else f"results_{a.variant}.json"
    (REPORT_E6 / out).write_text(json.dumps(res, indent=1, default=_json) + "\n")


# ------------------------------------------------------------------------------------------------ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["align", "protocol", "digiface", "embed", "evaluate", "e6-align", "e6-embed",
                                    "e6-evaluate", "summary"])
    ap.add_argument("--datasets", nargs="+", default=["celeba", "ffhq"])
    ap.add_argument("--model")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--part", type=_part, default=(0, 1), help="k/N: process shards with index %% N == k")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--ep", choices=["cuda", "trt"], default="cuda", help="SCRFD execution provider (align)")
    ap.add_argument("--stores", nargs="+", default=list(STORES), help="embed: crop stores to embed")
    ap.add_argument("--reverse", action="store_true", help="align: walk datasets/shards from the end (extra workers)")
    ap.add_argument("--init", action="store_true", help="align/digiface: only create the stores (before parallel parts)")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--variant", choices=["farl", "scrfd320"], default="farl", help="e6-*: alignment compared to SCRFD")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--boot-small", type=int, default=200)
    a = ap.parse_args(argv)
    if a.model:
        a.models = [a.model]
    {"align": cmd_align, "protocol": cmd_protocol, "digiface": cmd_digiface, "embed": cmd_embed,
     "evaluate": cmd_evaluate, "e6-align": cmd_e6_align, "e6-embed": cmd_e6_embed, "e6-evaluate": cmd_e6_evaluate,
     "summary": lambda a: __import__("eval.onen.report", fromlist=["main"]).main()}[a.cmd](a)


if __name__ == "__main__":
    main()
