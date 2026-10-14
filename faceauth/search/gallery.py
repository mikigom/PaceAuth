"""Exact 1:N template gallery: FAISS flat inner-product search (float32, no approximation) + subject aggregation.

Rows live in fixed-size flat segments (GpuIndexFlatIP on cuda:0, IndexFlatIP on CPU for tests): faiss GPU flat
storage is reallocated and copied on every add (an 8 GB index would need 16 GB for one enrollment), so only the
last segment ever grows. Rows never move; removal is a tombstone (faiss GPU flat has no remove_ids, and its
IDSelector search parameter is silently ignored), filtered on the host.
"""
import numpy as np
import faiss

SEG_ROWS = 1 << 19   # rows per segment: 1 GiB at d=512, bounds the copy/extra memory of an add
MAX_K = 2048         # faiss GPU k-selection limit
NORM_TOL = 1e-3


def _fit(a: np.ndarray, n: int) -> np.ndarray:
    if len(a) >= n:
        return a
    b = np.zeros(max(n, 2 * len(a)), a.dtype)
    b[:len(a)] = a
    return b


class Gallery:
    """Not thread-safe (neither are faiss GPU indexes): the caller serialises add/remove/search."""

    def __init__(self, gpu: bool = True, seg_rows: int = SEG_ROWS):
        self.dim = None                        # set by the first add (the model's embedding size)
        self.seg_rows = seg_rows
        self.res = None
        if gpu:
            self.res = faiss.StandardGpuResources()
            self.res.setTempMemory(512 << 20)
        self.segs = []
        self.n = 0                             # rows incl. tombstones
        self.sub = np.zeros(0, np.int32)       # row -> subject index
        self.alive = np.zeros(0, bool)
        self.row_of = {}                       # template id -> latest row; tombstones require explicit reactivation
        self.subjects = []                     # subject index -> subject id
        self.sub_idx = {}
        self.live = np.zeros(0, np.int32)      # subject index -> live templates
        self.ntotal = 0                        # live templates
        self.nsubjects = 0                     # subjects with >= 1 live template

    def _new_seg(self):
        if self.res is None:
            return faiss.IndexFlatIP(self.dim)
        cfg = faiss.GpuIndexFlatConfig()
        cfg.device = 0
        cfg.useFloat16 = False
        return faiss.GpuIndexFlatIP(self.res, self.dim, cfg)

    def add(self, tids, subject_ids, vecs, *, allow_reactivation=False) -> int:
        """Append templates (vecs: n x d float32, L2-normalised). Live IDs are skipped. -> rows added.
        Tombstones may be re-added only after the caller verifies an active database row and decrypts it again.
        On an exception the gallery may be inconsistent: discard it."""
        vecs = np.asarray(vecs, np.float32)
        if vecs.ndim != 2 or len(vecs) != len(tids) or len(tids) != len(subject_ids):
            raise ValueError("shape mismatch")
        if self.dim is None and len(vecs):
            self.dim = vecs.shape[1]
        if len(vecs) and vecs.shape[1] != self.dim:
            raise ValueError(f"template dim {vecs.shape[1]} != {self.dim}")
        if not np.all(np.abs(np.linalg.norm(vecs, axis=1) - 1) < NORM_TOL):  # also rejects NaN/inf
            raise ValueError("templates must be finite and L2-normalised")
        tids = np.asarray(tids, np.int64)
        keep, seen = [], set()
        for i, t in enumerate(tids.tolist()):
            row = self.row_of.get(t)
            if t not in seen and (row is None or (allow_reactivation and not self.alive[row])):
                seen.add(t)
                keep.append(i)
        if not keep:
            return 0
        x = np.ascontiguousarray(vecs[keep])
        i = 0
        while i < len(x):
            if not self.segs or self.segs[-1].ntotal >= self.seg_rows:
                self.segs.append(self._new_seg())
            take = min(self.seg_rows - self.segs[-1].ntotal, len(x) - i)
            self.segs[-1].add(x[i:i + take])
            i += take
        sidx = []
        for i in keep:
            sid = str(subject_ids[i])
            s = self.sub_idx.get(sid)
            if s is None:
                s = self.sub_idx[sid] = len(self.subjects)
                self.subjects.append(sid)
            sidx.append(s)
        sidx = np.array(sidx, np.int32)
        r0, m = self.n, self.n + len(keep)
        self.sub, self.alive = _fit(self.sub, m), _fit(self.alive, m)
        self.live = _fit(self.live, len(self.subjects))
        self.sub[r0:m], self.alive[r0:m] = sidx, True
        self.row_of.update(zip(tids[keep].tolist(), range(r0, m)))
        u, c = np.unique(sidx, return_counts=True)
        self.nsubjects += int(np.count_nonzero(self.live[u] == 0))
        self.live[u] += c
        self.n = m
        self.ntotal += len(keep)
        return len(keep)

    def remove(self, tids) -> int:
        """Tombstone templates; unknown or already removed ids are no-ops. -> templates removed."""
        # ponytail: tombstones keep their GPU rows and still take top-K slots until a rebuild (admin reload)
        # compacts them; search fails closed if more than ~MAX_K of them crowd one query.
        n = 0
        for t in tids:
            r = self.row_of.get(int(t))
            if r is not None and self.alive[r]:
                self.alive[r] = False
                s = self.sub[r]
                self.live[s] -= 1
                if self.live[s] == 0:
                    self.nsubjects -= 1
                n += 1
        self.ntotal -= n
        return n

    def _knn(self, q: np.ndarray, k: int):
        """Top-k rows (incl. tombstones) over all segments, scores descending."""
        Ds, Is, off = [], [], 0
        for s in self.segs:
            D, I = s.search(q, min(k, s.ntotal))
            Ds.append(D)
            Is.append(np.where(I >= 0, I + off, -1))
            off += s.ntotal
        D, I = np.hstack(Ds), np.hstack(Is)
        if len(self.segs) > 1:
            o = np.argsort(-D, axis=1, kind="stable")[:, :k]
            D, I = np.take_along_axis(D, o, 1), np.take_along_axis(I, o, 1)
        return D, I

    def _top_subjects(self, d, rows, k):
        out, seen = [], set()
        for score, r in zip(d, rows):
            if r < 0 or not self.alive[r]:
                continue
            s = self.sub[r]
            if s not in seen:
                seen.add(s)
                out.append((self.subjects[s], float(score)))
                if len(out) == k:
                    break
        return out

    def search(self, q: np.ndarray, k: int) -> list:
        """q: m x d unit queries -> per query [(subject_id, score)]: the top-k subjects by max template score.

        Exact: K rows are fetched (growing until k distinct live subjects appear or every row was seen). Any subject
        not among them has all templates <= the K-th row score <= the score of every subject returned.
        None (caller fails closed) for a query that would need more than MAX_K rows (tombstones/one subject's rows).
        """
        q = np.ascontiguousarray(q, np.float32)
        out = [[] for _ in range(len(q))]
        if self.n == 0:
            return out
        if q.ndim != 2 or q.shape[1] != self.dim:
            raise ValueError(f"query dim {q.shape[-1]} != {self.dim}")
        todo, K = list(range(len(q))), min(self.n, MAX_K, max(64, 8 * k))
        while todo:
            D, I = self._knn(q[todo], K)
            rest = []
            for j, qi in enumerate(todo):
                out[qi] = self._top_subjects(D[j], I[j], k)
                if len(out[qi]) < k and K < self.n:
                    rest.append(qi)
            if K >= MAX_K:
                for qi in rest:
                    out[qi] = None
                break
            todo, K = rest, min(self.n, MAX_K, 4 * K)
        return out
