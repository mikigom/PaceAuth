"""Unit tests of the API decision policy (faceauth/api/policy.py) + in-process fail-closed checks of the app.
Runs anywhere faceauth:dev runs (no stack needed)."""
import math
import uuid
from pathlib import Path

import numpy as np
import pytest

from faceauth.api import policy

S = Path(__file__).resolve().parents[1] / "secrets"


def _unit(v):
    return (np.asarray(v, np.float32) / np.linalg.norm(v)).astype(np.float32)


def _frame(ok=True, faces=1, live=True, emb=b"x" * 8, reasons=()):
    return {"ok": ok, "n_faces": faces, "emb": emb, "reasons": list(reasons), "pad": {"live": live, "score": 0.5}}


def test_external_reasons_never_leak_detail():
    assert policy.external("ok") == "ok"
    assert policy.external("ambiguous") == policy.external("no_match") == "not_recognized"
    assert policy.external("subject_inactive") == "not_recognized"
    assert policy.external("session_replayed") == policy.external("challenge_mismatch") == "session"
    assert policy.external("quality:blur,no_face") == "retry"
    for r in ("pad_fail", "bad_signature", "hash_mismatch", "calib_mismatch", "assertion_invalid", "whatever_new"):
        assert policy.external(r) == "denied"


def test_identify_requires_threshold_and_margin():
    a, b = uuid.uuid4(), uuid.uuid4()
    assert policy.identify([], 0.5, 0.05)[:2] == (None, "no_match")
    assert policy.identify([{"subject_id": a, "score": 0.4}], 0.5, 0.05)[:2] == (None, "no_match")
    assert policy.identify([{"subject_id": a, "score": 0.9}], 0.5, 0.05)[:2] == (a, "ok")
    res = [{"subject_id": b, "score": 0.88}, {"subject_id": a, "score": 0.9}]   # unsorted input
    assert policy.identify(res, 0.5, 0.05)[:2] == (None, "ambiguous")          # top-1 alone never suffices
    res = [{"subject_id": b, "score": 0.80}, {"subject_id": a, "score": 0.9}]
    assert policy.identify(res, 0.5, 0.05)[:2] == (a, "ok")
    assert policy.identify([{"subject_id": a, "score": 0.9}], math.nan, 0.05)[0] is None  # NaN config rejects


def test_frame_and_pad_checks_fail_closed():
    assert policy.frame_reasons([_frame(), _frame()]) == []
    assert policy.frame_reasons([_frame(), _frame(ok=False, reasons=["blur"])]) == ["blur"]
    assert policy.frame_reasons([_frame(faces=2)]) == ["not_ok"]          # ok=True but two faces: still rejected
    assert policy.frame_reasons([_frame(emb=None)]) == ["not_ok"]
    assert policy.frame_reasons([]) == ["no_frames"]
    assert policy.pad_live([_frame(), _frame()])
    assert not policy.pad_live([_frame(), _frame(live=False)])
    assert not policy.pad_live([{**_frame(), "pad": None}])
    assert not policy.pad_live([{**_frame(), "pad": {"live": "true"}}])
    assert not policy.pad_live([])


def test_embeddings_parse_and_reject_malformed():
    v = _unit(np.arange(1, 5))
    e = policy.embeddings([{"emb": (3 * v).astype("<f4").tobytes()}], 4)
    assert e.shape == (1, 4) and np.allclose(e[0], v)
    for bad in (None, b"123", np.zeros(4, "<f4").tobytes(), np.full(4, np.nan, "<f4").tobytes()):
        with pytest.raises(ValueError):
            policy.embeddings([{"emb": bad}], 4)


def test_coverage_uses_terminal_bins():
    from faceauth.terminal.guide import direction_of
    poses = [[0, 0, 0], [25, 0, 0], [-25, 0, 0], [0, 15, 0], [0, -15, 0], [25, 15, 0], [-25, 15, 0], [25, -15, 0],
             [-25, -15, 0], [12, 0, 0], None, ["x", 1], [math.nan, 0]]
    dirs, cov = policy.coverage(poses, direction_of)
    assert dirs[:9] == ["C", "L", "R", "U", "D", "UL", "UR", "DL", "DR"] and dirs[9:] == [None] * 4
    assert cov == {d: True for d in policy.DIRECTIONS}
    _, cov = policy.coverage([[0, 0, 0], [25, 0, 0]], direction_of)
    assert sum(cov.values()) == 2 and cov["C"] and cov["L"] and not cov["R"]


def test_same_person_and_templates():
    rng = np.random.default_rng(0)
    a, b = _unit(rng.standard_normal(64)), _unit(rng.standard_normal(64))
    embs = np.stack([_unit(a + 0.05 * rng.standard_normal(64)) for _ in range(9)])
    dirs = ["C", "L", "R", "U", "D", "UL", "UR", "DL", "DR"]
    ok, low = policy.same_person(embs, 0.8)
    assert ok and low > 0.8
    mixed = np.concatenate([embs[:8], b[None]])
    assert not policy.same_person(mixed, 0.5)[0]
    t = policy.templates(embs, dirs)
    assert len(t) == 4 and all(abs(np.linalg.norm(x) - 1) < 1e-5 for x in t)
    assert np.allclose(t[0], embs[0])                                   # slot0 = C
    assert np.allclose(t[1], _unit(embs[[1, 5, 7]].mean(0)))            # subject-left: L, UL, DL
    assert np.allclose(t[2], _unit(embs[[2, 6, 8]].mean(0)))            # subject-right: R, UR, DR
    assert np.allclose(t[3], _unit(embs[[3, 4]].mean(0)))               # vertical: U, D
    t = policy.templates(embs[:2], ["C", "L"])                           # missing groups -> overall mean
    assert np.allclose(t[2], _unit(embs[:2].mean(0))) and np.allclose(t[3], t[2])


@pytest.mark.skipif(not (S / "approval.key").exists(), reason="needs dev secrets (scripts/init_secrets.py)")
def test_app_fails_closed_when_db_is_down(monkeypatch):
    from fastapi.testclient import TestClient

    from faceauth.api.app import app
    env = {"FA_DB_DSN": "host=127.0.0.1 port=1 dbname=x user=x connect_timeout=1", "FA_RECOG_MODEL": "m",
           "FA_T_ID": "0.5", "FA_MARGIN": "0.05", "FA_T_DUP": "0.4", "FA_T_SAME": "0.5", "FA_CONSENT_VERSION": "c",
           "FA_TRUSTED_PROXIES": "testclient"}
    files = {"FA_DB_PASSWORD_FILE": "db_api_password", "FA_KMS_TOKEN_FILE": "kms_token_api",
             "FA_APPROVAL_KEY_FILE": "approval.key", "FA_ACCOUNT_PUB_FILE": "account.pub",
             "FA_ADMIN_TOKEN_FILE": "admin_token"}
    for k, v in {**env, **{k: str(S / v) for k, v in files.items()}}.items():
        monkeypatch.setenv(k, v)
    with TestClient(app) as c:
        r = c.post("/v1/sessions", json={"purpose": "auth", "device_id": "dev-001", "action_id": "pay"},
                   headers={"X-Client-Verify": "SUCCESS", "X-Client-Cert": "x"})
        assert r.status_code == 503 and r.json() == {"detail": "unavailable"}
        assert c.get("/healthz").status_code == 503
        assert c.post("/v1/authentications", content=b"\x00garbage").status_code == 400
