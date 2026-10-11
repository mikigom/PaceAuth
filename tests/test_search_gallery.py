"""Gallery exactness vs numpy brute force (CPU faiss always, GPU faiss when a GPU is visible)."""
import faiss
import numpy as np
import pytest

import faceauth.search.gallery as gallery_mod
from faceauth.search.gallery import Gallery

DEVICES = [False] + ([True] if faiss.get_num_gpus() > 0 else [])
TOL = 1e-5


def unit(x):
    x = np.asarray(x, np.float64)
    return (x / np.linalg.norm(x, axis=-1, keepdims=True)).astype(np.float32)


def brute(X, sids, Q):
    """Full subject ranking per query: [(subject, max score)] descending (float64)."""
    S = Q.astype(np.float64) @ X.astype(np.float64).T
    out = []
    for row in S:
        best = {}
        for s, v in zip(sids, row):
            best[s] = max(best.get(s, -np.inf), v)
        out.append(sorted(best.items(), key=lambda kv: -kv[1]))
    return out


def check(got, ranking, k):
    assert len(got) == min(k, len(ranking))
    ref = dict(ranking)
    scores = [s for _, s in got]
    assert len({s for s, _ in got}) == len(got), "duplicate subject"
    assert all(a >= b - TOL for a, b in zip(scores, scores[1:])), "not sorted"
    np.testing.assert_allclose(scores, [s for _, s in ranking[:k]], atol=TOL)
    for sid, score in got:
        assert abs(ref[sid] - score) < TOL               # each subject's score is its max template score
    kth = ranking[k - 1][1] if len(ranking) >= k else -np.inf
    must = {sid for sid, s in ranking if s > kth + TOL}  # clear winners can't be missed; ties may go either way
    assert must <= {sid for sid, _ in got}


@pytest.mark.parametrize("gpu", DEVICES)
@pytest.mark.parametrize("seg_rows", [7, 1 << 19])
def test_exact_vs_bruteforce(gpu, seg_rows):
    rng = np.random.default_rng(0)
    d, ns = 128, 300
    X, sids = [], []
    for s in range(ns):
        c = rng.standard_normal(d)
        for _ in range(4 if s % 3 else rng.integers(1, 5)):        # most subjects have 4 templates
            X.append(c + 0.7 * rng.standard_normal(d))
            sids.append(f"s{s}")
    X = unit(np.array(X))
    X[sids.index("s1")] = X[0]                                      # exact tie across subjects s0 / s1
    g = Gallery(gpu=gpu, seg_rows=seg_rows)
    for i in range(0, len(X), 97):                                  # several adds (segments fill unevenly)
        g.add(np.arange(i, min(i + 97, len(X))) + 1000, sids[i:i + 97], X[i:i + 97])
    assert (g.ntotal, g.nsubjects, g.n) == (len(X), ns, len(X))
    Q = np.vstack([unit(rng.standard_normal((30, d))), X[:10],                # random, exact templates (ties),
                   unit(X[20:30] + 0.3 * unit(rng.standard_normal((10, d))))])  # near templates
    ranking = brute(X, sids, Q)
    for k in (1, 5, 20):
        for got, rk in zip(g.search(Q, k), ranking):
            check(got, rk, k)


@pytest.mark.parametrize("gpu", DEVICES)
def test_one_subject_dominates(gpu):
    """300 templates of A outrank B's only template: top-2 must still be [A, B] (adaptive K)."""
    rng = np.random.default_rng(1)
    d = 64
    q = unit(rng.standard_normal(d))
    A = unit(q + 0.05 * unit(rng.standard_normal((300, d))))
    B = unit(q + 0.6 * unit(rng.standard_normal(d)))
    others = unit(rng.standard_normal((500, d)))
    g = Gallery(gpu=gpu)
    g.add(range(300), ["A"] * 300, A)
    g.add([300], ["B"], B[None])
    g.add(range(301, 801), [f"o{i}" for i in range(500)], others)
    got = g.search(q[None], 2)[0]
    assert [s for s, _ in got] == ["A", "B"]
    assert got[0][1] == pytest.approx(float((A @ q).max()), abs=TOL)
    assert got[1][1] == pytest.approx(float(B @ q), abs=TOL)


@pytest.mark.parametrize("gpu", DEVICES)
def test_tombstones_idempotency_and_counts(gpu):
    rng = np.random.default_rng(2)
    d = 64
    q = unit(rng.standard_normal(d))
    near = unit(q + 0.05 * unit(rng.standard_normal((200, d))))   # 200 templates of 50 subjects, all near q
    g = Gallery(gpu=gpu)
    near_sids = [f"n{i // 4}" for i in range(200)]
    assert g.add(range(200), near_sids, near) == 200
    assert g.add(range(200), near_sids, near) == 0                 # duplicate add events: no-op
    w, w2 = unit(q + 0.5 * unit(rng.standard_normal(d))), unit(rng.standard_normal(d))
    g.add([500, 501], ["W", "W"], np.vstack([w, w2]))
    g.add(range(600, 1600), [f"r{i}" for i in range(1000)], unit(rng.standard_normal((1000, d))))
    assert (g.ntotal, g.nsubjects) == (1202, 1051)

    assert g.remove(range(200)) == 200                             # all near subjects deleted
    assert g.remove(range(200)) == 0 and g.remove([99999]) == 0    # repeated / unknown removes: no-op
    assert (g.ntotal, g.nsubjects, g.n) == (1002, 1001, 1202)
    assert g.add([5], ["n1"], near[5:6]) == 0                      # a removed template is never resurrected
    got = g.search(q[None], 3)[0]                                  # 200 tombstones ahead of the live winner
    assert got[0][0] == "W" and not any(s.startswith("n") for s, _ in got)
    assert got[0][1] == pytest.approx(float(w @ q), abs=TOL)
    g.remove([500])
    assert g.nsubjects == 1001                                     # W still has template 501 -> scored by it
    assert dict(g.search(q[None], 1001)[0])["W"] == pytest.approx(float(w2 @ q), abs=TOL)
    g.remove([501])
    assert g.nsubjects == 1000 and all(s != "W" for s, _ in g.search(q[None], 1000)[0])


@pytest.mark.parametrize("gpu", DEVICES)
def test_fails_closed_beyond_max_k(gpu, monkeypatch):
    monkeypatch.setattr(gallery_mod, "MAX_K", 64)
    rng = np.random.default_rng(3)
    d = 32
    q = unit(rng.standard_normal(d))
    g = Gallery(gpu=gpu)
    g.add(range(100), [f"t{i}" for i in range(100)], unit(q + 0.05 * unit(rng.standard_normal((100, d)))))
    g.add(range(100, 400), [f"r{i}" for i in range(300)], unit(rng.standard_normal((300, d))))
    g.remove(range(100))                                           # 100 tombstones > MAX_K ahead of any live row
    far = unit(rng.standard_normal(d))
    got = g.search(np.vstack([q, far]), 5)
    assert got[0] is None                                          # cannot prove exactness -> caller rejects
    assert got[1] is not None and len(got[1]) == 5                 # other queries in the batch unaffected


def test_rejects_bad_templates():
    g = Gallery(gpu=False)
    with pytest.raises(ValueError):
        g.add([1], ["a"], np.ones((1, 8), np.float32))             # not normalised
    with pytest.raises(ValueError):
        g.add([1], ["a"], np.full((1, 8), np.nan, np.float32))
    g.add([1], ["a"], unit(np.ones((1, 8))))
    with pytest.raises(ValueError):
        g.add([2], ["a"], unit(np.ones((1, 16))))                  # other dimension
    with pytest.raises(ValueError):
        g.search(unit(np.ones((1, 16))), 1)
    assert Gallery(gpu=False).search(unit(np.ones((2, 8))), 5) == [[], []]   # empty gallery: no candidates


def test_batch_loop_mixed_requests_and_cancellation():
    """One GPU call serves every queued request; per-request errors stay per request; a cancelled request
    (client gone) neither gets an answer nor kills the batch thread."""
    import time
    from faceauth.search.app import Mismatch, NotReady, Replica
    rng = np.random.default_rng(4)
    X = unit(rng.standard_normal((10, 8)))
    g = Gallery(gpu=False)
    g.add(range(10), [f"s{i}" for i in range(10)], X)
    r = Replica("", None, None, "m", gpu=False)
    r.gallery, r.model_version, r.applied, r.last_ok = g, "m@1", 7, time.monotonic()
    ok, ok3, bad, gone, dim = (r.submit("m@1", X[3], 2), r.submit("m@1", X[5], 3), r.submit("m@0", X[3], 2),
                               r.submit("m@1", X[3], 2), r.submit("m@1", unit(np.ones(4)), 2))
    gone.cancel()
    r.jobs.put(None)
    r.batch_loop()                                   # drains the queue as one batch, returns on the sentinel
    assert [s for s, _ in ok.result()[0]][:1] == ["s3"] and len(ok.result()[0]) == 2 and ok.result()[1] == 7
    assert ok3.result()[0][0][0] == "s5" and len(ok3.result()[0]) == 3
    with pytest.raises(Mismatch):
        bad.result()
    with pytest.raises(ValueError):
        dim.result()
    assert gone.cancelled()
    r.last_ok -= 60                                  # stale (no successful poll) -> not ready
    late = r.submit("m@1", X[3], 2)
    r.jobs.put(None)
    r.batch_loop()
    with pytest.raises(NotReady):
        late.result()


def test_from_env_wiring(tmp_path, monkeypatch):
    """Secrets come from files (stripped); every KMS client carries the search-scope bearer token."""
    from faceauth.search.app import Replica
    (tmp_path / "tok").write_text("search-token\n")
    (tmp_path / "pw").write_text("ro-pass\n")
    for k, v in {"FA_DB_DSN": "host=db dbname=faceauth user=faceauth_ro", "FA_DB_PASSWORD_FILE": tmp_path / "pw",
                 "FA_KMS_URL": "http://kms:8003", "FA_KMS_TOKEN_FILE": tmp_path / "tok", "FA_RECOG_MODEL": "m"}.items():
        monkeypatch.setenv(k, str(v))
    r = Replica.from_env()
    with r.new_kms() as c:
        assert c.headers["authorization"] == "Bearer search-token" and str(c.base_url) == "http://kms:8003"
    assert (r.password, r.recog, r.dsn) == ("ro-pass", "m", "host=db dbname=faceauth user=faceauth_ro")
