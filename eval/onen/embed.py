"""Resumable, batched embedding extraction for protocol images.

Raw images:     SCRFD.detect -> pick the labelled face -> recognizer.embed(img, kps5) (recognizer's own alignment).
Aligned crops:  datasets.ALIGNED prefixes -> recognizer.embed_aligned(112x112), no detector.
Every `shard` images go to <out>/shards/<i>.npz (written atomically; existing shards are skipped on restart),
then <out>/emb.npy (float32 (n, dim), NaN rows = failure to acquire) and <out>/extract.json.
"""
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from eval.onen.datasets import ALIGNED


def read_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def pick(dets, hint=None, max_dist=0.5):
    """Index of the labelled face: with annotated landmarks, the detection whose 5 kps are closest (mean
    distance < max_dist x annotated inter-ocular distance, else None); without, the largest box."""
    if not dets:
        return None
    if hint is not None:
        iod = float(np.linalg.norm(hint[0] - hint[1])) + 1e-6
        dist = [float(np.linalg.norm(d["kps"] - hint, axis=1).mean()) / iod for d in dets]
        i = int(np.argmin(dist))
        return i if dist[i] < max_dist else None
    return int(np.argmax([(d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]) for d in dets]))


def _shard(paths, files, rec, det, hints, batch, det_thresh, pool):
    n = len(paths)
    emb = np.full((n, rec.dim), np.nan, np.float32)
    det_score = np.full(n, np.nan, np.float32)
    n_faces = np.full(n, -1, np.int16)
    raw, crops = [], []

    def flush(force=False):
        if raw and (force or len(raw) >= batch):
            idx, imgs, kps = zip(*raw)
            emb[list(idx)] = rec.embed(list(imgs), np.stack(kps))
            raw.clear()
        if crops and (force or len(crops) >= batch):
            idx, c = zip(*crops)
            emb[list(idx)] = rec.embed_aligned(np.stack(c))
            crops.clear()

    for i, (p, img) in enumerate(zip(paths, pool.imap(read_rgb, files, chunksize=8))):
        if img is None:
            continue
        if p.startswith(ALIGNED):
            if img.shape[:2] != (112, 112):
                img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA)
            crops.append((i, img))
        else:
            dets = det.detect(img, det_thresh)
            n_faces[i] = len(dets)
            j = pick(dets, hints.get(p))
            if j is None:
                continue
            det_score[i] = dets[j]["score"]
            raw.append((i, img, np.asarray(dets[j]["kps"], np.float32)))
        flush()
    flush(force=True)
    return emb, det_score, n_faces


def extract(paths, out, rec, det, data_root="data/datasets", hints=None, shard=4096, batch=64, workers=16,
            det_thresh=0.5) -> np.ndarray:
    """-> (len(paths), rec.dim) float32 embeddings, NaN rows where no face / unreadable."""
    out = Path(out)
    (out / "shards").mkdir(parents=True, exist_ok=True)
    hints = hints or {}
    t0, done = time.time(), 0
    with Pool(workers) as pool:
        for s0 in range(0, len(paths), shard):
            f = out / "shards" / f"{s0 // shard:05d}.npz"
            if f.exists():
                continue
            ps = list(paths[s0:s0 + shard])
            emb, ds, nf = _shard(ps, [str(Path(data_root) / p) for p in ps], rec, det, hints, batch, det_thresh, pool)
            tmp = f.with_name(f.stem + ".tmp.npz")
            np.savez(tmp, emb=emb, det_score=ds, n_faces=nf, paths=np.array(ps))
            os.replace(tmp, f)
            done += len(ps)
            print(f"[embed] {min(s0 + shard, len(paths))}/{len(paths)} ({done / (time.time() - t0):.1f} img/s)",
                  flush=True)
    parts = [np.load(out / "shards" / f"{i:05d}.npz") for i in range((len(paths) + shard - 1) // shard)]
    for s0, z in zip(range(0, len(paths), shard), parts):
        if list(z["paths"]) != list(paths[s0:s0 + shard]):
            raise ValueError(f"shard {s0 // shard} was written for a different image list; delete {out}/shards")
    emb = np.concatenate([z["emb"] for z in parts]) if parts else np.zeros((0, rec.dim), np.float32)
    np.save(out / "emb.npy", emb)
    nf = np.concatenate([z["n_faces"] for z in parts]) if parts else np.zeros(0, np.int16)
    meta = {"recognizer": rec.version, "detector": getattr(det, "version", None), "n": len(paths),
            "ok": int(np.isfinite(emb).all(1).sum()), "no_face": int((nf == 0).sum()),
            "unreadable_or_unmatched": int(((nf != 0) & ~np.isfinite(emb).all(1)).sum()),
            "det_thresh": det_thresh, "seconds_this_run": round(time.time() - t0, 1)}
    (out / "extract.json").write_text(json.dumps(meta, indent=2) + "\n")
    return emb
