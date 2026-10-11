"""Open-set 1:N metrics with exact binomial and subject-level bootstrap CIs, plus model-based extrapolation.

Decision rule (system policy, IMPLEMENTATION.md §6 rule 4): a search with subject scores top1 >= top2 is
*accepted* iff top1 >= t and top1 - top2 >= margin (margin 0 = plain threshold rule).
  FPIR(t)  = accepted non-mated searches / non-mated searches
  FNIR(t)  = mated searches that are NOT (rank-1 = mate AND accepted) / mated searches
  misID(t) = mated searches whose rank-1 is another gallery subject AND accepted (enrolled customer
             identified as someone else; a subset of the FNIR errors, reported separately per RFP §4)
A probe without an embedding (failure to acquire) enters with top1 = top2 = -inf: mated -> miss,
non-mated -> correct reject. Everything here is numpy/scipy on per-search arrays.
"""
import numpy as np
from scipy import stats

ALPHA = 0.05
FPIR_GRID = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
HIST_W = 1e-4                                  # impostor histogram bin width over cosine [-1, 1]
HIST_BINS = int(round(2 / HIST_W))
F32_LOWEST = float(np.finfo(np.float32).min)


# ---------------------------------------------------------------- binomial statistics
def cp_ci(x: int, n: int, alpha: float = ALPHA) -> tuple[float, float]:
    """Two-sided Clopper-Pearson (exact binomial) 1-alpha interval for x/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if x == 0 else stats.beta.ppf(alpha / 2, x, n - x + 1)
    hi = 1.0 if x == n else stats.beta.ppf(1 - alpha / 2, x + 1, n - x)
    return float(lo), float(hi)


def upper_bound(x: int, n: int, alpha: float = ALPHA) -> float:
    """One-sided exact 1-alpha upper bound for x/n. x=0: 1 - alpha**(1/n) ~= 3/n (rule of three)."""
    return 1.0 if x >= n else float(stats.beta.ppf(1 - alpha, x + 1, n - x))


def searches_needed(rate: float, alpha: float = ALPHA) -> int:
    """Independent error-free searches needed for the one-sided upper bound to reach `rate` (1e-6 -> 2,995,732)."""
    return int(np.ceil(np.log(alpha) / np.log1p(-rate)))


# ---------------------------------------------------------------- thresholds and rates
def decision(top1, top2, margin: float = 0.0) -> np.ndarray:
    """Score that has to clear the threshold: top1 where the margin rule holds, else -inf (never accepted)."""
    top1, top2 = np.asarray(top1, np.float64), np.asarray(top2, np.float64)
    with np.errstate(invalid="ignore"):
        return np.where(top1 - top2 >= margin, top1, -np.inf)


def _thr(desc: np.ndarray, k: int) -> float:
    """Threshold accepting only scores strictly above desc[k] (float32 grid: what a float32 search sees)."""
    if k >= len(desc):
        return -np.inf
    return float(np.nextafter(np.float32(max(desc[k], F32_LOWEST)), np.float32(np.inf)))


def threshold_at(nm_dec, fpir: float) -> float:
    """Lowest threshold with measured FPIR <= fpir on non-mated decision scores (ties -> stricter)."""
    desc = np.sort(np.asarray(nm_dec, np.float64))[::-1]
    return _thr(desc, int(np.floor(fpir * len(desc) * (1 + 1e-12))))


def outcome_scores(m_top1, m_top2, m_rank1, n_top1, n_top2, margin=0.0):
    """-> (nm, hit, mis): a search counts at threshold t iff its score >= t.
    nm: non-mated false positives; hit: mated correct accepts; mis: mated mis-identifications."""
    md = decision(m_top1, m_top2, margin)
    r1 = np.asarray(m_rank1, bool)
    return decision(n_top1, n_top2, margin), np.where(r1, md, -np.inf), np.where(r1, -np.inf, md)


# ---------------------------------------------------------------- bootstrap (resample subjects, not images)
def _clusters(c):
    _, inv = np.unique(np.asarray(c), return_inverse=True)
    return inv, int(inv.max()) + 1 if len(inv) else 0


def cluster_boot_ci(err, clusters, B=1000, seed=0, alpha=ALPHA) -> tuple[float, float]:
    """Percentile CI of mean(err) resampling clusters (subjects / identities) with replacement."""
    inv, C = _clusters(clusters)
    if C == 0:
        return (float("nan"), float("nan"))
    e = np.bincount(inv, np.asarray(err, float), C)
    n = np.bincount(inv, minlength=C).astype(float)
    rng = np.random.default_rng(seed)
    r = np.empty(B)
    for b in range(B):
        w = np.bincount(rng.integers(0, C, C), minlength=C)
        r[b] = (w @ e) / (w @ n)
    lo, hi = np.quantile(r, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def boot_fnir_at_fpir(nm, nm_cl, hit, m_cl, fpir, B=1000, seed=0, alpha=ALPHA) -> tuple[float, float]:
    """CI of FNIR@FPIR=fpir with the threshold re-estimated in every replicate (two-sample cluster
    bootstrap: non-mated identities and enrolled subjects resampled independently)."""
    order = np.argsort(-np.asarray(nm), kind="stable")
    desc = np.asarray(nm, np.float64)[order]
    ninv, Cn = _clusters(np.asarray(nm_cl)[order])
    minv, Cm = _clusters(m_cl)
    if Cn == 0 or Cm == 0:
        return (float("nan"), float("nan"))
    hit = np.asarray(hit, np.float64)
    rng = np.random.default_rng(seed)
    r = np.empty(B)
    for b in range(B):
        wn = np.bincount(rng.integers(0, Cn, Cn), minlength=Cn)[ninv]
        cw = np.cumsum(wn)
        j = int(np.searchsorted(cw, np.floor(fpir * cw[-1] * (1 + 1e-12)), side="right"))
        t = _thr(desc, j)
        wm = np.bincount(rng.integers(0, Cm, Cm), minlength=Cm)[minv]
        r[b] = 1 - (wm @ (hit >= t)) / wm.sum()
    lo, hi = np.quantile(r, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


# ---------------------------------------------------------------- full evaluation of one search set
def evaluate(m_top1, m_top2, m_rank1, m_cluster, n_top1, n_top2, n_cluster, margins=(0.0,), grid=FPIR_GRID,
             B=1000, seed=0, det_points=60) -> dict:
    """All open-set numbers for one gallery / probe variant. *_cluster = subject (mated) or identity
    (non-mated) labels used for counting independent searches and for the bootstrap."""
    M, N = len(m_top1), len(n_top1)
    m_cluster, n_cluster = np.asarray(m_cluster), np.asarray(n_cluster)
    n_ids = len(np.unique(n_cluster))
    out = {
        "n_mated_searches": M, "n_mated_subjects": int(len(np.unique(m_cluster))),
        "n_nonmated_searches": N, "n_nonmated_identities": n_ids,
        "rank1_identification_rate": float(np.mean(m_rank1)) if M else float("nan"),
        "min_demonstrable_fpir_95": {"by_search": upper_bound(0, N), "by_identity": upper_bound(0, n_ids),
                                     "note": "one-sided 95% upper bound if zero false positives were observed"},
        "searches_needed_for_fpir_1e-6": searches_needed(1e-6),
        "margins": {},
    }
    for mg in margins:
        nm, hit, mis = outcome_scores(m_top1, m_top2, m_rank1, n_top1, n_top2, mg)
        rows = []
        for f in grid:
            t = threshold_at(nm, f)
            fp, h, mi = int((nm >= t).sum()), int((hit >= t).sum()), int((mis >= t).sum())
            rows.append({
                "target_fpir": f, "threshold": t, "resolvable": bool(f * N >= 1),
                "fpir": fp / N, "fp": fp, "fpir_ci95": cp_ci(fp, N), "fpir_upper95": upper_bound(fp, N),
                "fpir_boot_ci95": cluster_boot_ci(nm >= t, n_cluster, B, seed),
                "fp_identities": int(len(np.unique(n_cluster[nm >= t]))),
                "fnir": (M - h) / M, "fnir_ci95": cp_ci(M - h, M),
                "fnir_boot_ci95": cluster_boot_ci(hit < t, m_cluster, B, seed),
                "fnir_at_fpir_boot_ci95": boot_fnir_at_fpir(nm, n_cluster, hit, m_cluster, f, B, seed),
                "misid": mi, "misid_rate": mi / M, "misid_ci95": cp_ci(mi, M),
                "misid_subjects": int(len(np.unique(m_cluster[mis >= t]))),
            })
        out["margins"][str(mg)] = {"grid": rows, "det": det_curve(nm, hit, det_points)}
    return out


def det_curve(nm, hit, n_points=60) -> dict:
    """(threshold, FPIR, FNIR) from FPIR=1 down to the last resolvable value 1/N, log-spaced."""
    N, M = len(nm), len(hit)
    desc = np.sort(np.asarray(nm, np.float64))[::-1]
    hs = np.sort(np.asarray(hit, np.float64))
    ks = np.unique(np.round(np.logspace(np.log10(N), 0, n_points)).astype(int)) if N else np.array([], int)
    ts = np.array([_thr(desc, k) for k in ks] + [_thr(desc, 0)])
    fpir = (N - np.searchsorted(desc[::-1], ts, side="left")) / max(N, 1)
    fnir = np.searchsorted(hs, ts, side="left") / max(M, 1)
    return {"threshold": ts.tolist(), "fpir": fpir.tolist(), "fnir": fnir.tolist()}


# ---------------------------------------------------------------- model-based extrapolation to G subjects
def fpir_model(p, n_templates):
    """FPIR ~= 1 - (1 - p)^n for n independent impostor comparisons per search (n = G * k templates).
    With positively correlated templates of one subject or heterogeneous probes this is ~an upper bound."""
    return -np.expm1(n_templates * np.log1p(-np.asarray(p, np.float64)))


def p_for_fpir(fpir, n_templates):
    return float(-np.expm1(np.log1p(-fpir) / n_templates))


class ImpostorTail:
    """Per-comparison impostor survival p(t) = P(score >= t) from a histogram of ALL non-mated comparisons
    (bin width HIST_W over [-1, 1]); a generalized Pareto (peaks-over-threshold) fit above u extends it
    below the empirical resolution 1/n. MODEL-BASED: p(t) beyond the data rests on the GPD assumption."""

    def __init__(self, counts, n_exceed=100_000):
        self.counts = np.asarray(counts, np.int64)
        self.n = int(self.counts.sum())
        self.edges = -1 + HIST_W * np.arange(HIST_BINS + 1)
        cnt_ge = np.cumsum(self.counts[::-1])[::-1]               # comparisons >= edges[i]
        self.surv = np.append(cnt_ge, 0) / self.n
        iu = int(np.nonzero(cnt_ge >= min(n_exceed, self.n))[0].max())
        self.u, self.p_u, self.n_u = float(self.edges[iu]), float(self.surv[iu]), int(cnt_ge[iu])
        y = np.repeat(self.edges[iu:-1] + HIST_W / 2 - self.u, self.counts[iu:])
        self.c, _, self.scale = stats.genpareto.fit(y, floc=0)

    def p_emp(self, t):
        """Empirical P(score >= edge) for the bin edge at or below t: exact on edges, else >= P(score >= t)."""
        i = np.clip(np.floor((np.asarray(t, np.float64) + 1) / HIST_W + 1e-6).astype(int), 0, HIST_BINS)
        return self.surv[i]

    def p(self, t):
        t = np.asarray(t, np.float64)
        gpd = self.p_u * stats.genpareto.sf(np.maximum(t - self.u, 0), self.c, 0, self.scale)
        return np.where(t >= self.u, gpd, self.p_emp(t))

    def t_for_p(self, p):
        """Lowest threshold with p(t) <= p (first histogram edge, or the GPD inverse below p_u)."""
        if p >= self.p_u:
            return float(self.edges[int(np.argmax(self.surv <= p))])
        return float(self.u + stats.genpareto.isf(p / self.p_u, self.c, 0, self.scale))

    def summary(self, checks=(1e-6, 1e-7, 1e-8, 1e-9)) -> dict:
        rows = []
        for q in checks:
            if q * self.n >= 10:   # empirical value still backed by >= 10 comparisons
                t = self.t_for_p(q)
                rows.append({"t": t, "p_empirical": float(self.p_emp(t)), "p_model": float(self.p(t))})
        return {"n_comparisons": self.n, "u": self.u, "p_u": self.p_u, "n_exceedances": self.n_u,
                "gpd_shape": float(self.c), "gpd_scale": float(self.scale), "fit_check": rows}


def fnir_model(s_true, t, n_templates, tail: ImpostorTail, margin=0.0):
    """FNIR ~= 1 - mean_j [s_j >= t] * (1 - p(s_j - margin))^n: the mate's score s_j must clear t and beat
    every one of n impostor templates by `margin`. s_true = exact mate score per mated search (-inf = FTA)."""
    s = np.asarray(s_true, np.float64)
    ok = np.isfinite(s) & (s >= t)
    q = np.zeros(len(s))
    q[ok] = np.exp(n_templates * np.log1p(-tail.p(s[ok] - margin)))
    return float(1 - q.mean())


def extrapolate(tail: ImpostorTail, s_true, n_templates, target_fpir=1e-6, margins=(0.0,)) -> dict:
    p_star = p_for_fpir(target_fpir, n_templates)
    t = tail.t_for_p(p_star)
    return {"label": "MODEL-BASED EXTRAPOLATION (independent comparisons + GPD impostor tail), NOT A MEASUREMENT",
            "n_templates": int(n_templates), "target_fpir": target_fpir, "p_per_comparison": p_star,
            "p_star_below_empirical_resolution": bool(p_star * tail.n < 1), "threshold": t,
            "fnir": {str(m): fnir_model(s_true, t, n_templates, tail, m) for m in margins}}
