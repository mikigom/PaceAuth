"""Exact 1:N search for evaluation: faiss brute-force inner product on GPU (same metric and float32 maths as the
production GpuIndexFlatIP), gallery streamed in chunks so ~4M x 512 templates fit a ~10 GB budget, then
per-subject aggregation (subject score = max over its templates, IMPLEMENTATION.md §4).

    s = Searcher()                                   # one faiss GPU resource, bounded temp memory
    D, I = s.knn(gallery, probes, k)                 # top-k templates per probe
    sc, sid = subject_topk(D, I, tpl_subject, ks)    # top-ks subjects per probe, descending
    h = s.hist(probes, gallery)                      # histogram (metrics.HIST_*) of ALL probe x template scores
"""
import faiss
import faiss.contrib.torch_utils  # noqa: F401  (faiss.knn_gpu accepts torch CUDA tensors: one upload per chunk)
import numpy as np
import torch

from eval.onen.metrics import HIST_BINS, HIST_W


def _finite(x, what):
    x = np.ascontiguousarray(x, np.float32)
    if not np.isfinite(x).all():  # NaN rows (failure to acquire) would come back as garbage neighbours / bins
        raise ValueError(f"{what} contain non-finite rows; drop failed images before searching")
    return x


class Searcher:
    def __init__(self, chunk=1_000_000, qbatch=16384, temp_mem=512 << 20, device="cuda"):
        self.chunk, self.qbatch, self.dev = chunk, qbatch, torch.device(device)
        self.res = None
        if self.dev.type == "cuda":
            self.res = faiss.StandardGpuResources()
            self.res.setTempMemory(temp_mem)  # default reserves ~18% of the card; GPU 5 is shared

    def _knn(self, xq, xb, k):
        if self.res is not None:
            return faiss.knn_gpu(self.res, xq, xb, k, metric=faiss.METRIC_INNER_PRODUCT)
        D, I = faiss.knn(xq.numpy(), xb.numpy(), k, metric=faiss.METRIC_INNER_PRODUCT)  # CPU fallback (tests)
        return torch.from_numpy(D), torch.from_numpy(I)

    def knn(self, gallery: np.ndarray, probes: np.ndarray, k: int):
        """-> D (P,k) float32 descending, I (P,k) int64 gallery rows (-1 = none)."""
        gallery, probes = _finite(gallery, "gallery"), _finite(probes, "probes")
        P, T = len(probes), len(gallery)
        D = np.full((P, k), -np.inf, np.float32)
        I = np.full((P, k), -1, np.int64)
        for c0 in range(0, T, self.chunk):
            xb = torch.from_numpy(gallery[c0:c0 + self.chunk]).to(self.dev)
            kk = min(k, len(xb))
            for q0 in range(0, P, self.qbatch):
                xq = torch.from_numpy(probes[q0:q0 + self.qbatch]).to(self.dev)
                d, i = self._knn(xq, xb, kk)
                d, i = d.cpu().numpy(), i.cpu().numpy()
                i = np.where(i >= 0, i + c0, -1)
                q = slice(q0, q0 + len(xq))
                D[q], I[q] = merge_topk([(D[q], I[q]), (d, i)], k)
            del xb
        return D, I

    def hist(self, probes: np.ndarray, gallery: np.ndarray, tile: int = 256) -> np.ndarray:
        """(HIST_BINS,) int64 counts of every probe x gallery inner product, bin = floor((s + 1) / HIST_W)."""
        probes, gallery = _finite(probes, "probes"), _finite(gallery, "gallery")
        h = torch.zeros(HIST_BINS, dtype=torch.int64, device=self.dev)
        for c0 in range(0, len(gallery), self.chunk):
            xb = torch.from_numpy(gallery[c0:c0 + self.chunk]).to(self.dev)
            for q0 in range(0, len(probes), tile):
                xq = torch.from_numpy(probes[q0:q0 + tile]).to(self.dev)
                b = ((xq @ xb.T + 1) / HIST_W).floor_().clamp_(0, HIST_BINS - 1).to(torch.int32)
                h += torch.bincount(b.flatten(), minlength=HIST_BINS)
            del xb
        return h.cpu().numpy()


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
