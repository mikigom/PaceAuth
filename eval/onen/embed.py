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
import multiprocessing as mp
import os
import time
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
    """Open (r+) or create a zero-filled memmap. Creation is atomic (link of a temp file), so parallel processes
    starting together never truncate each other's data."""
    if not path.exists():
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        np.memmap(tmp, dtype, "w+", shape=tuple(shape)).flush()
        try:
            os.link(tmp, path)
        except FileExistsError:
            pass
        os.unlink(tmp)
    m = np.memmap(path, dtype, "r+")
    if m.size != int(np.prod(shape)):
        raise ValueError(f"{path}: size mismatch, delete it")
    return m.reshape(shape)


def _read(args):
    return read_rgb(*args)


def _claim(f: Path, stale_s=3 * 3600) -> bool:
    """Work queue for parallel align processes: atomically claim a shard (claims older than stale_s without a
    .done marker are taken over: a killed process)."""
    try:
        os.close(os.open(f, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        if time.time() - f.stat().st_mtime > stale_s:
            f.touch()
            return True
        return False


def align(paths, store, det=None, hints=None, kps_in=None, refine=None, shard=4096, workers=12, det_thresh=0.5,
          part=(0, 1), data_root=DATA, refine_batch=64, reverse=False):
    store = Path(store)
    paths = _paths(store, paths)
    n = len(paths)
    _meta(store, n, shard)
    crops = _memmap(store / "crops.u8", (n, *CROP), np.uint8)
    (store / "shards").mkdir(exist_ok=True)
    hints = hints or {}
    t0, done = time.time(), 0
    with mp.get_context("spawn").Pool(workers) as pool:  # spawn: never fork a process holding CUDA/TRT state
        order = list(enumerate(range(0, n, shard)))
        for si, s0 in reversed(order) if reverse else order:
            mark = store / "shards" / f"{si:05d}.done"
            if si % part[1] != part[0] or mark.exists() or not _claim(store / "shards" / f"{si:05d}.claim"):
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
    tmp = store / f"det.{os.getpid()}.tmp.npz"  # parallel parts may finish together: write atomically
    np.savez(tmp, **out)
    (store / "align.json").write_text(json.dumps({"n": n, "ok": int(out["ok"].sum()),
                                                   "no_face": int((out["n_faces"] == 0).sum()),
                                                   "unmatched": int(((out["n_faces"] > 0) & ~out["ok"]).sum())}, indent=1))
    os.replace(tmp, store / "det.npz")
    return out


def init_store(paths, store, shard=4096) -> None:
    """Create paths.txt + the crops memmap once (run before launching parallel parts: 'w+' would truncate)."""
    store = Path(store)
    n = len(_paths(store, paths))
    _meta(store, n, shard)
    _memmap(store / "crops.u8", (n, *CROP), np.uint8).flush()


def _meta(store: Path, n: int, shard: int) -> None:
    f = store / "meta.json"
    if f.exists():
        m = json.loads(f.read_text())
        if (m["n"], m["shard"]) != (n, shard):
            raise ValueError(f"{store} was built with n={m['n']} shard={m['shard']}")
        return
    tmp = store / f"meta.{os.getpid()}.tmp"
    tmp.write_text(json.dumps({"n": n, "shard": shard}))
    os.replace(tmp, f)


def load_store(store):
    """-> (paths, crops memmap (n,112,112,3), det dict) of a finished store."""
    store = Path(store)
    paths = (store / "paths.txt").read_text().split("\n")
    det = finish_align(store)
    if det is None:
        raise RuntimeError(f"{store}: alignment not finished")
    return paths, np.memmap(store / "crops.u8", np.uint8, "r").reshape(len(paths), *CROP), det


def embed(store, rec, out, part=(0, 1)) -> np.ndarray | None:
    """Embeddings of a store's crops -> (n, dim) float32 memmap at `out` (NaN rows = no crop). Works per align shard
    and only on shards whose alignment is done, so it can run while alignment is still going: call again until it
    returns the array (None = some align shards are not done yet)."""
    store, out = Path(store), Path(out)
    paths = (store / "paths.txt").read_text().split("\n")
    n, shard = len(paths), json.loads((store / "meta.json").read_text())["shard"]
    if out.with_suffix(".json").exists():
        return np.memmap(out, np.float32, "r").reshape(n, rec.dim)
    crops = np.memmap(store / "crops.u8", np.uint8, "r").reshape(n, *CROP)
    out.parent.mkdir(parents=True, exist_ok=True)
    emb = _memmap(out, (n, rec.dim), np.float32)
    sdir = out.parent / f"ashards_{out.stem}"
    sdir.mkdir(exist_ok=True)
    k = (n + shard - 1) // shard
    t0, done = time.time(), 0
    for si in range(k):
        mark = sdir / f"{si:05d}.done"
        if si % part[1] != part[0] or mark.exists() or not (store / "shards" / f"{si:05d}.done").exists():
            continue
        ok = np.load(store / "shards" / f"{si:05d}.npz")["ok"]
        sl = slice(si * shard, si * shard + len(ok))
        e = np.full((len(ok), rec.dim), np.nan, np.float32)
        if ok.any():
            e[ok] = rec.embed_aligned(np.ascontiguousarray(crops[sl][ok]))
        emb[sl] = e
        emb.flush()
        mark.touch()
        done += int(ok.sum())
        print(f"[embed {rec.version} {out.stem}] shard {si + 1}/{k} {done / (time.time() - t0):.0f} img/s", flush=True)
    if len(list(sdir.glob("*.done"))) < k:
        return None
    n_ok = int(sum(np.load(f)["ok"].sum() for f in (store / "shards").glob("*.npz")))
    out.with_suffix(".json").write_text(json.dumps({"recognizer": rec.version, "store": str(store), "n": n, "ok": n_ok},
                                                   indent=1))
    return emb
