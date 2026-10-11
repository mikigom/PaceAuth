"""Exact 1:N search for evaluation: faiss brute-force inner product on GPU (same metric and float32 maths as the
production GpuIndexFlatIP), gallery streamed in chunks so ~4M x 512 templates fit a ~10 GB budget, then
per-subject aggregation (subject score = max over its templates, IMPLEMENTATION.md §4).

    s = Searcher()                                   # one faiss GPU resource, bounded temp memory
    D, I, hist = s.knn(gallery, probes, k, hist_rows) # top-k templates per probe (+ impostor histogram)
    sc, sid = subject_topk(D, I, tpl_subject, ks)    # top-ks subjects per probe, descending
"""
import faiss
import faiss.contrib.torch_utils  # noqa: F401  (faiss.knn_gpu accepts torch CUDA tensors: one upload per chunk)
import numpy as np
import torch

from eval.onen.metrics import HIST_BINS, HIST_W


class Searcher:
    def __init__(self, chunk=1_000_000, qbatch=16384, hist_tile=128, temp_mem=512 << 20, device="cuda"):
        self.chunk, self.qbatch, self.hist_tile, self.dev = chunk, qbatch, hist_tile, torch.device(device)
        self.res = None
        if self.dev.type == "cuda":
            self.res = faiss.StandardGpuResources()
            self.res.setTempMemory(temp_mem)  # default reserves ~18% of the card; GPU 5 is shared

    def _knn(self, xq, xb, k):
        if self.res is not None:
            return faiss.knn_gpu(self.res, xq, xb, k, metric=faiss.METRIC_INNER_PRODUCT)
        D, I = faiss.knn(xq.numpy(), xb.numpy(), k, metric=faiss.METRIC_INNER_PRODUCT)  # CPU fallback (tests)
        return torch.from_numpy(D), torch.from_numpy(I)

    def knn(self, gallery: np.ndarray, probes: np.ndarray, k: int, hist_rows=None):
        """-> D (P,k) float32 descending, I (P,k) int64 gallery rows (-1 = none), hist (HIST_BINS,) int64 of
        ALL scores of probes[hist_rows] against the whole gallery (or None)."""
        P, T = len(probes), len(gallery)
        D = np.full((P, k), -np.inf, np.float32)
        I = np.full((P, k), -1, np.int64)
        hist = torch.zeros(HIST_BINS, dtype=torch.int64, device=self.dev) if hist_rows is not None else None
        hrows = np.flatnonzero(hist_rows) if hist_rows is not None else np.zeros(0, int)
        for c0 in range(0, T, self.chunk):
            xb = torch.from_numpy(np.ascontiguousarray(gallery[c0:c0 + self.chunk], np.float32)).to(self.dev)
            kk = min(k, len(xb))
            for q0 in range(0, P, self.qbatch):
                xq = torch.from_numpy(np.ascontiguousarray(probes[q0:q0 + self.qbatch], np.float32)).to(self.dev)
                d, i = self._knn(xq, xb, kk)
                d, i = d.cpu().numpy(), i.cpu().numpy()
                i = np.where(i >= 0, i + c0, -1)
                D[q0:q0 + len(xq)], I[q0:q0 + len(xq)] = merge_topk([(D[q0:q0 + len(xq)], I[q0:q0 + len(xq)]), (d, i)], k)
            for h0 in range(0, len(hrows), self.hist_tile):
                xq = torch.from_numpy(np.ascontiguousarray(probes[hrows[h0:h0 + self.hist_tile]], np.float32)).to(self.dev)
                b = ((xq @ xb.T + 1) / HIST_W).floor_().clamp_(0, HIST_BINS - 1).to(torch.int32)
                hist += torch.bincount(b.flatten(), minlength=HIST_BINS)
            del xb
        return D, I, (hist.cpu().numpy() if hist is not None else None)


def merge_topk(parts, k):
    """Merge per-chunk/per-block top-k lists (each row descending) into the global top-k."""
    D = np.concatenate([p[0] for p in parts], 1)
    I = np.concatenate([p[1] for p in parts], 1)
    o = np.argsort(-D, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(D, o, 1), np.take_along_axis(I, o, 1)


def subject_topk(D, I, tpl_subject, ks):
    """Template top-k -> top-ks subjects (max over each subject's templates), descending; pad -inf / -1.
    Exact when k >= ks * (max templates per subject): every top-ks subject's best template is retrieved."""
    S = np.where(I >= 0, tpl_subject[np.maximum(I, 0)], -1)
    D = np.where(I >= 0, D, -np.inf)
    o = np.lexsort((-D, S), axis=1)                      # group by subject, best template first
    S, D = np.take_along_axis(S, o, 1), np.take_along_axis(D, o, 1)
    dup = np.zeros_like(S, bool)
    dup[:, 1:] = S[:, 1:] == S[:, :-1]
    D = np.where(dup | (S < 0), -np.inf, D)
    o = np.argsort(-D, axis=1, kind="stable")[:, :ks]
    D, S = np.take_along_axis(D, o, 1), np.take_along_axis(S, o, 1)
    S = np.where(np.isfinite(D), S, -1)
    if D.shape[1] < ks:
        pad = ks - D.shape[1]
        D = np.pad(D, ((0, 0), (0, pad)), constant_values=-np.inf)
        S = np.pad(S, ((0, 0), (0, pad)), constant_values=-1)
    return D.astype(np.float32), S


def template_table(tpl_subject, n_subjects):
    """(n_subjects, kmax) gallery rows of each subject, -1 padded (subjects with no template: all -1)."""
    tpl_subject = np.asarray(tpl_subject)
    o = np.argsort(tpl_subject, kind="stable")
    s = tpl_subject[o]
    cnt = np.bincount(s, minlength=n_subjects)
    kmax = max(int(cnt.max()) if len(cnt) else 0, 1)
    rank = np.arange(len(s)) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    table = np.full((n_subjects, kmax), -1, np.int64)
    table[s, rank] = o
    return table


def subject_scores(x, subj, table, gallery, device="cuda", batch=65536):
    """Exact score of x[i] against subject subj[i] = max over that subject's templates (-inf if none)."""
    dev = torch.device(device)
    out = np.full(len(x), -np.inf, np.float32)
    for i0 in range(0, len(x), batch):
        s = np.asarray(subj[i0:i0 + batch])
        rows = np.where(s[:, None] >= 0, table[np.maximum(s, 0)], -1)
        g = torch.from_numpy(np.ascontiguousarray(gallery[np.maximum(rows, 0)], np.float32)).to(dev)
        q = torch.from_numpy(np.ascontiguousarray(x[i0:i0 + batch], np.float32)).to(dev)
        sc = torch.einsum("nkd,nd->nk", g, q).cpu().numpy()
        out[i0:i0 + batch] = np.where(rows >= 0, sc, -np.inf).max(1)
    return out
