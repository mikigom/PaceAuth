"""Resumable, sharded extraction shared by all recognizers (E2 / E6). Detection runs once; every model re-uses the crops.

Stage 1  align(paths, store, det, hints)   raw images: SCRFD -> pick the labelled face -> norm_crop 112 (arcface5-112@1,
                                           the alignment of every recognizer here, = what Recognizer.embed does);
                                           datasets.ALIGNED: the 112x112 image itself. kps_in=(n,5,2) skips detection
                                           (re-uses another store's detections) and `refine(imgs, kps) -> kps` swaps the
                                           alignment keypoints (E6: FaRL-derived 5 points).
         -> <store>/paths.txt, crops.u8 ((n,112,112,3) uint8 memmap), det.npz (kps (n,5,2), bbox, det_score, n_faces
            (-1 = not run), ok).
Stage 2  embed(store, rec, out)            -> <out> ((n, dim) float32 memmap, NaN rows where ok is False) + <out>.json.
Shards are marked done (<dir>/shards*/<i>.done) only after their data is flushed, so a killed run resumes. Disjoint
shard subsets (part=(k, N): shards with i % N == k) can run in parallel processes; finish() consolidates.
"""
import json
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from eval.onen.datasets import ALIGNED, DATA, read_rgb
from faceauth.common.align import norm_crop

CROP = (112, 112, 3)


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


def _paths(store: Path, paths) -> list:
    store.mkdir(parents=True, exist_ok=True)
    f = store / "paths.txt"
    paths = [str(p) for p in paths]
    if f.exists():
        if f.read_text().split("\n") != paths:
            raise ValueError(f"{store} was built for a different image list; delete it to rebuild")
    else:
        tmp = f.with_suffix(".tmp")
        tmp.write_text("\n".join(paths))
        tmp.replace(f)
    return paths


def _memmap(path: Path, shape, dtype):
    if path.exists():
        m = np.memmap(path, dtype, "r+")
        if m.size != int(np.prod(shape)):
            raise ValueError(f"{path}: size mismatch, delete it")
        return m.reshape(shape)
    return np.memmap(path, dtype, "w+", shape=tuple(shape))


def _read(args):
    return read_rgb(*args)


def align(paths, store, det=None, hints=None, kps_in=None, refine=None, shard=4096, workers=12, det_thresh=0.5,
          part=(0, 1), data_root=DATA, refine_batch=64):
    store = Path(store)
    paths = _paths(store, paths)
    n = len(paths)
    crops = _memmap(store / "crops.u8", (n, *CROP), np.uint8)
    (store / "shards").mkdir(exist_ok=True)
    hints = hints or {}
    t0, done = time.time(), 0
    with Pool(workers) as pool:
        for si, s0 in enumerate(range(0, n, shard)):
            mark = store / "shards" / f"{si:05d}.done"
            if si % part[1] != part[0] or mark.exists():
                continue
            ps = paths[s0:s0 + shard]
            m = len(ps)
            kps = np.full((m, 5, 2), np.nan, np.float32)
            bbox = np.full((m, 4), np.nan, np.float32)
            score = np.full(m, np.nan, np.float32)
            nf = np.full(m, -1, np.int16)
            ok = np.zeros(m, bool)
            pend = []  # (i, img, kps) waiting for refine

            def flush():
                if not pend:
                    return
                idx, imgs, k = zip(*pend)
                k2 = refine(list(imgs), np.stack(k)) if refine is not None else np.stack(k)
                for i, img, kk in zip(idx, imgs, k2):
                    if np.isfinite(kk).all():
                        crops[s0 + i] = norm_crop(img, kk, 112)
                        kps[i], ok[i] = kk, True
                pend.clear()

            for i, img in enumerate(pool.imap(_read, [(p, data_root) for p in ps], chunksize=8)):
                if img is None:
                    continue
                if ps[i].startswith(ALIGNED):
                    crops[s0 + i] = img if img.shape == CROP else cv2.resize(img, CROP[:2], interpolation=cv2.INTER_AREA)
                    ok[i] = True
                    continue
                if kps_in is not None:
                    k = np.asarray(kps_in[s0 + i], np.float32)
                    if not np.isfinite(k).all():
                        continue
                else:
                    dets = det.detect(img, det_thresh)
                    nf[i] = len(dets)
                    j = pick(dets, hints.get(ps[i]))
                    if j is None:
                        continue
                    k = np.asarray(dets[j]["kps"], np.float32)
                    bbox[i], score[i] = dets[j]["bbox"], dets[j]["score"]
                pend.append((i, img, k))
                if refine is None or len(pend) >= refine_batch:
                    flush()
            flush()
            crops.flush()
            np.savez(store / "shards" / f"{si:05d}.npz", kps=kps, bbox=bbox, det_score=score, n_faces=nf, ok=ok)
            mark.touch()
            done += m
            print(f"[align {store.name}] shard {si} ({min(s0 + shard, n)}/{n}) ok {ok.mean():.3f} "
                  f"{done / (time.time() - t0):.0f} img/s", flush=True)
    return finish_align(store)


def finish_align(store) -> dict | None:
    """Consolidate shard records into det.npz once every shard is done (None while shards are missing)."""
    store = Path(store)
    if (store / "det.npz").exists():
        z = np.load(store / "det.npz")
        return {k: z[k] for k in z.files}
    n = len((store / "paths.txt").read_text().split("\n"))
    files = sorted((store / "shards").glob("*.npz"))
    if not files or not all(f.with_suffix(".done").exists() for f in files):
        return None
    parts = [np.load(f) for f in files]
    if sum(len(p["ok"]) for p in parts) != n:
        return None
    out = {key: np.concatenate([p[key] for p in parts]) for key in ("kps", "bbox", "det_score", "n_faces", "ok")}
    np.savez(store / "det.npz", **out)
    (store / "align.json").write_text(json.dumps({"n": n, "ok": int(out["ok"].sum()),
                                                   "no_face": int((out["n_faces"] == 0).sum()),
                                                   "unmatched": int(((out["n_faces"] > 0) & ~out["ok"]).sum())}, indent=1))
    return out


def load_store(store):
    """-> (paths, crops memmap (n,112,112,3), det dict) of a finished store."""
    store = Path(store)
    paths = (store / "paths.txt").read_text().split("\n")
    det = finish_align(store)
    if det is None:
        raise RuntimeError(f"{store}: alignment not finished")
    return paths, np.memmap(store / "crops.u8", np.uint8, "r").reshape(len(paths), *CROP), det


def embed(store, rec, out, shard=16384, part=(0, 1)) -> np.ndarray | None:
    """Embeddings of a finished store's crops -> (n, dim) float32 memmap at `out` (NaN rows = no crop)."""
    paths, crops, det = load_store(store)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = len(paths)
    emb = _memmap(out, (n, rec.dim), np.float32)
    sdir = out.parent / f"shards_{out.stem}"
    sdir.mkdir(exist_ok=True)
    t0, done = time.time(), 0
    for si, s0 in enumerate(range(0, n, shard)):
        mark = sdir / f"{si:05d}.done"
        if si % part[1] != part[0] or mark.exists():
            continue
        sl = slice(s0, min(s0 + shard, n))
        ok = det["ok"][sl]
        e = np.full((sl.stop - sl.start, rec.dim), np.nan, np.float32)
        if ok.any():
            e[ok] = rec.embed_aligned(np.ascontiguousarray(crops[sl][ok]))
        emb[sl] = e
        emb.flush()
        mark.touch()
        done += int(ok.sum())
        print(f"[embed {rec.version} {out.stem}] {sl.stop}/{n} {done / (time.time() - t0):.0f} img/s", flush=True)
    if len(list(sdir.glob("*.done"))) < (n + shard - 1) // shard:
        return None
    (out.with_suffix(".json")).write_text(json.dumps({"recognizer": rec.version, "store": str(store), "n": n,
                                                      "ok": int(det["ok"].sum())}, indent=1))
    return emb
