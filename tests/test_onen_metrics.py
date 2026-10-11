"""E2 1:N harness: metric definitions, CIs, margin rule, misidentification counting, exact search + aggregation, tail
model, protocol disjointness. CPU only (faiss CPU path), synthetic data with known ground truth.

    docker run --rm -u $(id -u):$(id -g) -e HOME=/tmp -v $PWD:/app -w /app faceauth:dev \
        python -m pytest -q tests/test_onen_metrics.py
"""
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.stats import binomtest

from eval.onen import metrics as M
from eval.onen.protocol import build, counts
from eval.onen.search import Searcher, subject_scores, subject_topk, template_table


# ---------------------------------------------------------------------------------------------- binomial statistics
@pytest.mark.parametrize("x,n", [(0, 10), (3, 50), (50, 50), (7, 100000)])
def test_clopper_pearson_matches_scipy_exact(x, n):
    ci = binomtest(x, n).proportion_ci(confidence_level=0.95, method="exact")
    np.testing.assert_allclose(M.cp_ci(x, n), (ci.low, ci.high), rtol=1e-9, atol=1e-12)


def test_rule_of_three_and_searches_needed():
    n = 3_000_000
    assert M.upper_bound(0, n) == pytest.approx(1 - 0.05 ** (1 / n), rel=1e-9)
    assert M.upper_bound(0, n) == pytest.approx(3 / n, rel=0.01)            # rule of three
    assert M.searches_needed(1e-6) == 2_995_731                               # RFP §4 "~3M independent searches"
    assert M.upper_bound(0, M.searches_needed(1e-6)) <= 1e-6 < M.upper_bound(0, M.searches_needed(1e-6) - 1)
    assert M.upper_bound(5, 10) > M.cp_ci(5, 10)[0] and M.upper_bound(10, 10) == 1.0


# ---------------------------------------------------------------------------------------------- decision rule
def test_decision_margin_rule_and_fta():
    top1 = np.array([0.6, 0.6, 0.6, -np.inf, 0.3])
    top2 = np.array([0.2, 0.58, -np.inf, -np.inf, 0.1])
    d = M.decision(top1, top2, 0.05)
    assert d[0] == 0.6 and d[1] == -np.inf and d[2] == 0.6 and d[3] == -np.inf and d[4] == 0.3
    assert (M.decision(top1, top2, 0.0)[:3] == 0.6).all()


def test_threshold_fpir_definition_with_ties():
    nm = np.array([0.9, 0.8, 0.8, 0.7, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4], np.float32)  # N = 10
    for f, expect_fp in [(0.1, 1), (0.2, 1), (0.3, 3), (0.05, 0), (1.0, 10)]:
        t = M.threshold_at(nm, f)
        fp = int((nm >= t).sum())
        assert fp == expect_fp and fp / len(nm) <= f  # 0.2 -> 1 (the tie at 0.8 cannot be split)
    # the threshold is the lowest float32 value that keeps FPIR <= target
    t = M.threshold_at(nm, 0.1)
    assert t > 0.8 and np.float32(t) == np.nextafter(np.float32(0.8), np.float32(1))


def test_outcomes_fnir_and_misid_counting():
    # mated: (top1, top2, rank1 correct)
    m1 = np.array([0.80, 0.70, 0.60, 0.55, 0.50, 0.30])
    m2 = np.array([0.10, 0.68, 0.20, 0.10, 0.45, 0.10])
    r1 = np.array([True, True, False, False, True, True])
    n1 = np.array([0.40, 0.20, 0.10, 0.0])
    n2 = np.array([0.10, 0.10, 0.05, 0.0])
    nm, hit, mis = M.outcome_scores(m1, m2, r1, n1, n2, 0.0)
    t = 0.45
    assert int((hit >= t).sum()) == 3                  # 0.80, 0.70, 0.50 correct and accepted
    assert int((mis >= t).sum()) == 2                  # rank-1 wrong but accepted -> enrolled customer mis-ID
    assert int((nm >= t).sum()) == 0
    nm, hit, mis = M.outcome_scores(m1, m2, r1, n1, n2, 0.05)
    assert int((hit >= t).sum()) == 1 and int((mis >= t).sum()) == 2  # 0.70 vs 0.68 and 0.50 vs 0.45 ambiguous
    ev = M.evaluate(m1, m2, r1, np.arange(6), n1, n2, np.arange(4), margins=(0.0,), grid=(0.25,), B=50)
    row = ev["margins"]["0.0"]["grid"][0]
    # FPIR 0.25 of 4 -> 1 accepted (0.40); threshold just above 0.20; FNIR counts misses AND mis-IDs
    assert row["fp"] == 1 and row["fpir"] == 0.25 and 0.2 < row["threshold"] < 0.4
    assert row["misid"] == 2 and row["fnir"] == pytest.approx(2 / 6)  # t ~ 0.2: all correct ones hit; mis-IDs miss
    assert row["fnir_ci95"][0] <= row["fnir"] <= row["fnir_ci95"][1]


# ---------------------------------------------------------------------------------------------- search + end-to-end
def _synthetic(rng, S=60, k=4, nonmated=200, mated_per=5, d=32, noise=0.35):
    centers = rng.normal(size=(S + nonmated, d))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    unit = lambda x: (x / np.linalg.norm(x, axis=-1, keepdims=True)).astype(np.float32)
    tpl = unit(centers[:S, None] + noise * rng.normal(size=(S, k, d)) / np.sqrt(d) * 3).reshape(-1, d)
    tpl_subj = np.repeat(np.arange(S), k)
    keep = rng.random(len(tpl)) > 0.2  # uneven template counts (some subjects have 1-3)
    keep[::k] = True
    tpl, tpl_subj = tpl[keep], tpl_subj[keep]
    mp = unit(centers[:S, None] + noise * rng.normal(size=(S, mated_per, d)) / np.sqrt(d) * 3).reshape(-1, d)
    mp_subj = np.repeat(np.arange(S), mated_per)
    npb = unit(centers[S:] + noise * rng.normal(size=(nonmated, d)) / np.sqrt(d) * 3)
    return tpl, tpl_subj, mp, mp_subj, npb


def _brute(probes, tpl, tpl_subj, S):
    s = probes @ tpl.T
    sub = np.full((len(probes), S), -np.inf)
    for j in range(S):
        sub[:, j] = s[:, tpl_subj == j].max(1)
    o = np.argsort(-sub, 1)
    return np.take_along_axis(sub, o, 1), o


def test_search_aggregation_exact_vs_brute_force():
    rng = np.random.default_rng(0)
    tpl, ts, mp, ms, npb = _synthetic(rng)
    S = ts.max() + 1
    srch = Searcher(chunk=37, qbatch=50, device="cpu")  # tiny chunks: exercises the chunk merge
    P = np.concatenate([mp, npb])
    D, I = srch.knn(tpl, P, 12)
    sc, sid = subject_topk(D, I, ts, 3)
    bsc, bsid = _brute(P, tpl, ts, S)
    np.testing.assert_allclose(sc, bsc[:, :3], atol=1e-5)
    assert (sid == bsid[:, :3]).all()
    table = template_table(ts, S)
    m = len(mp)
    np.testing.assert_allclose(subject_scores(mp, ms, table, tpl, device="cpu"),
                               bsc[np.arange(m), np.argmax(bsid[:m] == ms[:, None], 1)], atol=1e-5)
    with pytest.raises(ValueError):
        srch.knn(tpl, np.full((1, tpl.shape[1]), np.nan, np.float32), 4)
    h = srch.hist(npb, tpl)
    assert h.sum() == len(npb) * len(tpl)
    s = (npb @ tpl.T).ravel()
    expect = np.bincount(np.clip(np.floor((s.astype(np.float32) + 1) / M.HIST_W), 0, M.HIST_BINS - 1).astype(int),
                         minlength=M.HIST_BINS)
    assert np.abs(h - expect).sum() <= 2 * 1e-3 * len(s)  # float32 rounding at bin edges only


def test_end_to_end_metrics_match_brute_force():
    rng = np.random.default_rng(1)
    tpl, ts, mp, ms, npb = _synthetic(rng, noise=0.6)
    S = ts.max() + 1
    srch = Searcher(device="cpu")
    D, I = srch.knn(tpl, np.concatenate([mp, npb]), 12)
    sc, sid = subject_topk(D, I, ts, 3)
    m = len(mp)
    ev = M.evaluate(sc[:m, 0], sc[:m, 1], sid[:m, 0] == ms, ms, sc[m:, 0], sc[m:, 1], np.arange(len(npb)),
                    margins=(0.0, 0.05), grid=(0.1, 0.02), B=100)
    _, bsid = _brute(np.concatenate([mp, npb]), tpl, ts, S)
    assert (sid[:, 0] == bsid[:, 0]).all()  # identities from brute force; scores = the search's float32 values
    for mg in (0.0, 0.05):
        for row in ev["margins"][str(mg)]["grid"]:
            t = row["threshold"]
            acc = (sc[:, 0] >= t) & (sc[:, 0] - sc[:, 1] >= mg)
            fp = acc[m:].sum()
            hit = (acc[:m] & (bsid[:m, 0] == ms)).sum()
            mis = (acc[:m] & (bsid[:m, 0] != ms)).sum()
            assert 0 < hit < m and mis > 0  # the synthetic set has misses and mis-IDs to count
            assert row["fp"] == fp and fp <= np.floor(row["target_fpir"] * len(npb))
            assert row["fnir"] == pytest.approx(1 - hit / m) and row["misid"] == mis
            # lowest such threshold: one float32 step lower would exceed the target FPIR
            lower = np.nextafter(np.float32(t), np.float32(-1))
            acc2 = (sc[m:, 0] >= lower) & (sc[m:, 0] - sc[m:, 1] >= mg)
            assert acc2.sum() > np.floor(row["target_fpir"] * len(npb)) or acc2.sum() == fp


# ---------------------------------------------------------------------------------------------- bootstrap
def test_cluster_bootstrap_widens_with_correlated_clusters():
    rng = np.random.default_rng(0)
    cl = np.repeat(np.arange(100), 20)
    err_ind = rng.random(2000) < 0.1
    err_cor = np.repeat(rng.random(100) < 0.1, 20)        # all-or-nothing per cluster
    lo_i, hi_i = M.cluster_boot_ci(err_ind, cl, B=500)
    lo_c, hi_c = M.cluster_boot_ci(err_cor, cl, B=500)
    assert lo_i <= err_ind.mean() <= hi_i and lo_c <= err_cor.mean() <= hi_c
    assert (hi_c - lo_c) > 2 * (hi_i - lo_i)
    nm = rng.normal(size=5000)
    hit = rng.normal(2, 1, size=3000)
    lo, hi = M.boot_fnir_at_fpir(nm, np.arange(5000) // 5, hit, np.arange(3000) // 3, 0.01, B=200)
    t = M.threshold_at(nm, 0.01)
    assert lo <= np.mean(hit < t) <= hi


# ---------------------------------------------------------------------------------------------- tail model
def test_impostor_tail_and_independence_model():
    rng = np.random.default_rng(0)
    c, loc, scale = -0.1, -0.3, 0.05  # scores whose tail IS generalized Pareto: the fitting machinery must recover it
    s = stats.genpareto.rvs(c, loc, scale, size=4_000_000, random_state=rng)
    h = np.bincount(np.floor((s + 1) / M.HIST_W).astype(int), minlength=M.HIST_BINS)
    tl = M.ImpostorTail(h, n_exceed=100_000)
    for t in (-0.2, 0.0, 0.05):
        edge = np.floor((t + 1) / M.HIST_W + 1e-6) * M.HIST_W - 1
        assert tl.p_emp(t) == pytest.approx(np.mean(s >= edge), rel=1e-3)
    # 1.6 decades below the 4M-sample resolution. Even with an exactly-GPD tail the shape estimate's sampling error
    # (c -0.093 vs -0.1 here) costs a factor ~2.6 in p: extrapolated probabilities are order-of-magnitude statements.
    t = float(loc + stats.genpareto.isf(1e-8, c, 0, scale))
    assert 1 / 4 < tl.p(t) / 1e-8 < 4
    assert tl.t_for_p(1e-8) == pytest.approx(t, abs=0.01)
    assert tl.t_for_p(1e-3) <= tl.t_for_p(1e-5) <= tl.t_for_p(1e-8)
    assert all(1 / 2 < r["p_model"] / r["p_empirical"] < 2 for r in tl.summary(checks=(1e-4, 1e-5, 1e-6))["fit_check"])
    # independence model: FPIR over n templates and the FNIR model with a mixture
    p = 1e-7
    assert M.fpir_model(p, 4_000_000) == pytest.approx(1 - np.exp(-0.4), rel=1e-6)
    assert M.p_for_fpir(1e-6, 4_000_000) == pytest.approx(2.5e-13, rel=1e-5)
    assert M.fpir_mix(0.05, [(tl, 1000), (tl, 3000)]) == pytest.approx(M.fpir_model(tl.p(0.05), 4000), rel=1e-9)
    mate = np.array([0.15, 0.08, -0.05, -np.inf])
    f = M.fnir_model(mate, 0.0, [(tl, 1000)])
    expect = 1 - np.mean([(1 - tl.p(0.15)) ** 1000, (1 - tl.p(0.08)) ** 1000, 0, 0])
    assert f == pytest.approx(expect, rel=1e-9)
    ext = M.extrapolate(tl, mate, 4_000_000)
    assert ext["p_star_below_empirical_resolution"] and "NOT A MEASUREMENT" in ext["label"]


# ---------------------------------------------------------------------------------------------- protocol
def test_protocol_subject_disjoint_and_seeded():
    rng = np.random.default_rng(0)
    ident = pd.DataFrame([(f"c:{s}", f"c/{s}_{i}.jpg") for s in range(40) for i in range(rng.integers(1, 12))],
                         columns=["subject", "path"])
    dis1 = pd.DataFrame({"subject": [f"f:{i}" for i in range(30)], "path": [f"f/{i}.png" for i in range(30)]})
    dis2 = pd.DataFrame([(f"d:{s}", f"d/{s}_{i}.png") for s in range(10) for i in range(6)], columns=["subject", "path"])
    p = build(ident, [("f", dis1), ("d", dis2)], {"f": 0.5}, seed=3)
    assert p.equals(build(ident, [("f", dis1), ("d", dis2)], {"f": 0.5}, seed=3))
    roles = p.groupby("subject").role.agg(set)
    assert not any({"template", "nonmated"} <= r or {"mated", "nonmated"} <= r or {"distractor", "nonmated"} <= r
                   for r in roles)
    enrolled = roles[roles.map(lambda r: "template" in r)].index
    assert (p[p.role == "template"].groupby("subject").size() <= 4).all()
    sizes = ident.groupby("subject").size()
    assert (sizes[enrolled] >= 5).all()                                     # min_images
    assert set(ident.path) | set(dis1.path) <= set(p.path) <= set(ident.path) | set(dis1.path) | set(dis2.path)
    assert (p[p.subject.str.startswith("d:")].groupby("subject").size() == 4).all()  # n_templates of 6 images
    d = p[p.role == "distractor"].drop_duplicates("subject")
    assert sorted(d.order) == list(range(len(d)))                          # contiguous gallery-size order
    assert (d.subject.str.startswith("f:").to_numpy()[np.argsort(d.order)][:15]).all()  # first dataset first
    assert (p[p.subject.str.startswith("f:") & (p.role == "nonmated")].subject.nunique()) == 15
    g = p[p.group >= 0].groupby("group")
    assert (g.size() == 3).all() and (g.subject.nunique() == 1).all() and (g.role.nunique() == 1).all()
    c = counts(p)
    assert c["max_gallery_subjects"] == len(enrolled) + 15 + 10
    # unit split (FFHQ Flickr account): no unit on both sides, cut moved to a unit boundary
    du = dis1.assign(unit=[f"acc{i // 4}" for i in range(30)])
    pu = build(ident, [("f", du), ("d", dis2)], {"f": 0.5}, seed=3)
    side = pu[pu.subject.str.startswith("f:")].assign(unit=lambda q: q.subject.map(du.set_index("subject").unit))
    assert side.groupby("unit").role.nunique().max() == 1
    assert (side.role == "nonmated").sum() % 4 == 0 or (side.role == "nonmated").sum() == 30
    # per-dataset streams: dropping identity images (e.g. detection failures) leaves the distractor selection unchanged
    p2 = build(ident.iloc[::2], [("f", dis1), ("d", dis2)], {"f": 0.5}, seed=3)
    key = lambda q: q[q.subject.str.startswith(("f:", "d:"))][["path", "subject", "role", "order"]].reset_index(drop=True)
    assert key(p).equals(key(p2))
