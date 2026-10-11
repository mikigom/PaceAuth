"""Search replica against a real Postgres (deploy/db/faceauth.sql, role faceauth_ro) and a spec §5 fake KMS.

Needs FA_TEST_PG = admin conninfo of a scratch Postgres (each test creates and drops its own database), e.g.
    docker run -d --name fa-search-pg --network fa-search-net -e POSTGRES_PASSWORD=test postgres:16-alpine
    FA_TEST_PG="host=fa-search-pg user=postgres password=test" pytest tests/test_search_replica.py
"""
import importlib.util
import os
import pathlib
import time
import uuid

import httpx
import msgpack
import numpy as np
import psycopg
import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from fastapi.testclient import TestClient

import faceauth.search.app as appmod
from faceauth.search.app import Mismatch, NotReady, Replica, create_app

ADMIN = os.environ.get("FA_TEST_PG")
pytestmark = pytest.mark.skipif(not ADMIN, reason="FA_TEST_PG (scratch Postgres admin conninfo) not set")
ROOT = pathlib.Path(__file__).resolve().parent.parent
D = 64
MV = "recog@1"


class FakeKMS:
    """Spec §5: per-subject AES-256-GCM key, ct = nonce(12) || AESGCM(key).encrypt(nonce, pt, aad=subject_id);
    decrypt -> null once the subject key is destroyed."""

    def __init__(self):
        self.keys, self.fail = {}, False

    def handle(self, req: httpx.Request) -> httpx.Response:
        if self.fail:
            return httpx.Response(503)
        body = msgpack.unpackb(req.content)
        if req.url.path == "/v1/encrypt":
            sid = body["subject_id"]
            key = self.keys.setdefault(sid, AESGCM.generate_key(256))
            items = []
            for pt in body["items"]:
                nonce = os.urandom(12)
                items.append(nonce + AESGCM(key).encrypt(nonce, pt, sid.encode()))
        elif req.url.path == "/v1/decrypt":
            items = []
            for it in body["items"]:
                key = self.keys.get(it["subject_id"])
                items.append(None if key is None else
                             AESGCM(key).decrypt(it["ct"][:12], it["ct"][12:], it["subject_id"].encode()))
        else:
            return httpx.Response(404)
        return httpx.Response(200, content=msgpack.packb({"items": items}))

    def client(self) -> httpx.Client:
        return httpx.Client(base_url="http://kms:8003", transport=httpx.MockTransport(self.handle))


@pytest.fixture
def db():
    """-> (admin connection to a fresh faceauth DB, faceauth_ro conninfo)."""
    name = "fa_search_" + uuid.uuid4().hex[:10]
    with psycopg.connect(ADMIN, dbname="postgres", autocommit=True) as c:
        c.execute(f"CREATE DATABASE {name}")
        c.execute("DO $$ BEGIN CREATE ROLE faceauth_ro LOGIN PASSWORD 'ro';"
                  " EXCEPTION WHEN duplicate_object THEN NULL; END $$")
    admin = psycopg.connect(ADMIN, dbname=name, autocommit=True)
    admin.execute((ROOT / "deploy/db/faceauth.sql").read_text())
    admin.execute("GRANT SELECT ON templates, template_events, model_versions TO faceauth_ro")  # spec §11 role
    yield admin, psycopg.conninfo.make_conninfo(ADMIN, dbname=name, user="faceauth_ro", password="ro")
    admin.close()
    with psycopg.connect(ADMIN, dbname="postgres", autocommit=True) as c:
        c.execute(f"DROP DATABASE {name} WITH (FORCE)")


def unit(x):
    x = np.asarray(x, np.float64)
    return (x / np.linalg.norm(x, axis=-1, keepdims=True)).astype(np.float32)


def activate(conn, version, name="recog"):
    conn.execute("UPDATE model_versions SET active = false WHERE name = %s", (name,))
    conn.execute("INSERT INTO model_versions (name, version, sha256, active, activated_at) VALUES (%s, %s, 'x', true,"
                 " now()) ON CONFLICT (name, version) DO UPDATE SET active = true", (name, version))


def enroll(conn, kms, vecs, mv=MV, active=True):
    """What the API does: subject + encrypted templates + 'add' events under lock 7001, one tx. -> (sid, tids)."""
    sid = str(uuid.uuid4())
    cts = msgpack.unpackb(kms.post("/v1/encrypt", content=msgpack.packb(
        {"subject_id": sid, "items": [v.astype("<f4").tobytes() for v in vecs]})).content)["items"]
    with conn.transaction():
        conn.execute("INSERT INTO subjects (id, account_id, status, consent_version, consent_at)"
                     " VALUES (%s, %s, 'active', 'v1', now())", (sid, "acct-" + sid))
        tids = [conn.execute("INSERT INTO templates (subject_id, template_version, slot, model_version, align_version,"
                             " calib_version, ciphertext, active) VALUES (%s, 1, %s, %s, 'a', 'c', %s, %s)"
                             " RETURNING id",
                             (sid, i, mv, ct, active)).fetchone()[0] for i, ct in enumerate(cts)]
        conn.execute("SELECT pg_advisory_xact_lock(7001)")
        for t in tids:
            conn.execute("INSERT INTO template_events (op, template_id, subject_id, model_version)"
                         " VALUES ('add', %s, %s, %s)", (t, sid, mv))
    return sid, tids


def delete(conn, fake, sid):
    """As DELETE /v1/subjects/{id}/face: revoke, deactivate + NULL ciphertext, 'remove' events; then shred.
    -> max remove event id"""
    with conn.transaction():
        conn.execute("UPDATE subjects SET status = 'revoked', revoked_at = now() WHERE id = %s", (sid,))
        rows = conn.execute("UPDATE templates SET active = false, ciphertext = NULL WHERE subject_id = %s AND active"
                            " RETURNING id, model_version", (sid,)).fetchall()
        conn.execute("SELECT pg_advisory_xact_lock(7001)")
        ids = [conn.execute("INSERT INTO template_events (op, template_id, subject_id, model_version)"
                            " VALUES ('remove', %s, %s, %s) RETURNING id", (t, sid, m)).fetchone()[0] for t, m in rows]
    fake.keys.pop(sid, None)
    return max(ids)


def person(rng, n=4):
    """-> (templates n x D, probes 3 x D) of one synthetic identity."""
    c = rng.standard_normal(D)
    return unit(c + 0.5 * rng.standard_normal((n, D))), unit(c + 0.5 * rng.standard_normal((3, D)))


def ranked(r, probes, k=100, mv=MV):
    """One search through the replica's batch path, without the batch thread. -> (results, applied_event_id)"""
    f = r.submit(mv, unit(probes.mean(0)), k)
    r.search_batch([r.jobs.get_nowait()])
    return f.result(timeout=5)


def max_event(conn):
    return conn.execute("SELECT max(id) FROM template_events").fetchone()[0]


def test_build_skips_shredded_inactive_and_other_versions(db):
    conn, ro = db
    fake, rng = FakeKMS(), np.random.default_rng(0)
    kms = fake.client()
    activate(conn, "1")
    people = [person(rng) for _ in range(5)]
    sa, ta = enroll(conn, kms, people[0][0])
    sb, _ = enroll(conn, kms, people[1][0])
    sc, _ = enroll(conn, kms, people[2][0][:2])                   # subject with 2 templates
    sd, _ = enroll(conn, kms, people[3][0])
    fake.keys.pop(sd)                                             # crypto-shredded (e.g. restored backup): KMS null
    se, _ = enroll(conn, kms, people[4][0], mv="recog@0")         # other model version: never compared
    enroll(conn, kms, people[4][0], active=False)                 # inactive template rows
    r = Replica(ro, None, fake.client, "recog", gpu=False)
    r.build()
    st = r.status()
    assert st["ready"] and st["model_version"] == MV
    assert (st["ntotal"], st["nsubjects"], st["applied_event_id"]) == (10, 3, max_event(conn))
    res, applied = ranked(r, people[0][1])
    assert res[0][0] == sa and applied == st["applied_event_id"]
    q = unit(people[0][1].mean(0))
    assert res[0][1] == pytest.approx(float((people[0][0] @ q).max()), abs=1e-5)   # max over A's templates
    assert {s for s, _ in res} == {sa, sb, sc}
    assert sd not in {s for s, _ in ranked(r, people[3][1])[0]}


def test_events_in_order_idempotent_and_deleted_never_returned(db):
    conn, ro = db
    fake, rng = FakeKMS(), np.random.default_rng(1)
    kms = fake.client()
    activate(conn, "1")
    A, B, C, F = (person(rng) for _ in range(4))
    sa, _ = enroll(conn, kms, A[0])
    sb, _ = enroll(conn, kms, B[0])
    r = Replica(ro, None, fake.client, "recog", gpu=False)
    r.build()
    sc, tc = enroll(conn, kms, C[0])
    assert r.poll() == 4 and r.status()["ntotal"] == 12 and ranked(r, C[1])[0][0][0] == sc
    conn.execute("INSERT INTO template_events (op, template_id, subject_id, model_version) VALUES ('add', %s, %s, %s)",
                 (tc[0], sc, MV))                                  # duplicate add event
    conn.execute("INSERT INTO template_events (op, template_id, subject_id, model_version) VALUES ('remove', 999999,"
                 " %s, %s)", (sc, MV))                             # remove of a template never indexed
    assert r.poll() == 2 and (r.status()["ntotal"], r.status()["nsubjects"]) == (12, 3)
    rm = delete(conn, fake, sa)
    r.poll()
    st = r.status()
    assert st["applied_event_id"] >= rm and (st["ntotal"], st["nsubjects"]) == (8, 2)
    assert sa not in {s for s, _ in ranked(r, A[1])[0]}          # deleted subject never returned, even for its face
    sf, _ = enroll(conn, kms, F[0])                               # add and remove land in the same poll batch
    delete(conn, fake, sf)
    r.poll()
    assert r.status()["ntotal"] == 8 and sf not in {s for s, _ in ranked(r, F[1])[0]}
    assert r.status()["applied_event_id"] == max_event(conn)


def test_snapshot_overlap_is_harmless(db, monkeypatch):
    """Events after the snapshot whose effects the template read already saw are re-applied idempotently."""
    conn, ro = db
    fake, rng = FakeKMS(), np.random.default_rng(2)
    kms = fake.client()
    activate(conn, "1")
    A, B = person(rng), person(rng)
    sa, _ = enroll(conn, kms, A[0])
    sb, _ = enroll(conn, kms, B[0])
    delete(conn, fake, sa)
    r = Replica(ro, None, fake.client, "recog", gpu=False)
    real = Replica._active
    monkeypatch.setattr(Replica, "_active", lambda self, c: (real(self, c)[0], 0, False))   # snapshot taken "long ago"
    r.build()
    st = r.status()
    assert (st["ntotal"], st["nsubjects"], st["applied_event_id"]) == (4, 1, max_event(conn))
    assert [s for s, _ in ranked(r, A[1])[0]] == [sb]


def test_model_activation_rebuilds_and_mismatch_is_409(db):
    conn, ro = db
    fake, rng = FakeKMS(), np.random.default_rng(3)
    kms = fake.client()
    activate(conn, "1")
    A, G = person(rng), person(rng)
    enroll(conn, kms, A[0])
    r = Replica(ro, None, fake.client, "recog", gpu=False)
    r.build()
    activate(conn, "2")
    sg, _ = enroll(conn, kms, G[0], mv="recog@2")
    r.poll()
    assert r.reload.is_set()                                      # activation noticed -> rebuild
    r.build()
    st = r.status()
    assert (st["model_version"], st["ntotal"]) == ("recog@2", 4)
    assert ranked(r, G[1], mv="recog@2")[0][0][0] == sg
    with pytest.raises(Mismatch):                                 # old-model probes are never compared
        ranked(r, G[1], mv="recog@1")


def test_fail_closed(db, monkeypatch):
    conn, ro = db
    fake, rng = FakeKMS(), np.random.default_rng(4)
    kms = fake.client()
    A = person(rng)
    r = Replica(ro, None, fake.client, "recog", gpu=False)
    with pytest.raises(NotReady):                                 # no active recognition model
        r.build()
    assert not r.status()["ready"]
    c = TestClient(create_app(r))                                 # endpoints only, no lifespan threads
    assert c.post("/v1/admin/reload").status_code == 202 and not r.reload.is_set()   # (re)building: not queued
    activate(conn, "1")
    enroll(conn, kms, A[0])
    fake.fail = True
    with pytest.raises(httpx.HTTPStatusError):                    # KMS down -> no index
        r.build()
    assert not r.status()["ready"]
    fake.fail = False
    r.build()
    assert r.status()["ready"]
    assert c.post("/v1/admin/reload").status_code == 202 and r.reload.is_set()
    r.reload.clear()
    r.last_ok -= appmod.MAX_STALE_S + 1                           # no successful poll for too long
    assert not r.status()["ready"]
    with pytest.raises(NotReady):
        ranked(r, A[1])
    r.poll()
    assert r.status()["ready"]
    r.applied += 1000                                             # server is behind the index:
    real, last = Replica._active, r.last_ok
    monkeypatch.setattr(Replica, "_active", lambda self, c: (*real(self, c)[:2], True))
    assert r.poll() == 0 and not r.reload.is_set() and r.last_ok == last   # a standby replaying: wait, go stale
    monkeypatch.setattr(Replica, "_active", real)
    r.poll()
    assert r.reload.is_set()                                      # the primary lost/restored history: rebuild


def test_failover_rewritten_history_rebuilds(db):
    """Sync replication shows a commit on the primary before a standby has it: after a failover, events this replica
    already applied can be gone and their ids reissued for other events (ids >= applied would be skipped)."""
    conn, ro = db
    fake, rng = FakeKMS(), np.random.default_rng(5)
    kms = fake.client()
    activate(conn, "1")
    enroll(conn, kms, person(rng)[0])
    r = Replica(ro, None, fake.client, "recog", gpu=False)
    r.build()
    enroll(conn, kms, person(rng)[0])
    r.poll()
    last = max_event(conn)
    assert r.applied == last and not r.reload.is_set()
    r.poll()                                                      # no new events, history intact: keep serving
    assert not r.reload.is_set()
    conn.execute("UPDATE template_events SET template_id = template_id + 100000 WHERE id = %s", (last,))
    r.poll()                                                      # same id, different event: rebuild
    assert r.reload.is_set()


@pytest.mark.parametrize("gpu", [False, True])
def test_http_service_end_to_end(db, gpu):
    import faiss
    if gpu and faiss.get_num_gpus() == 0:
        pytest.skip("no GPU")
    conn, ro = db
    fake, rng = FakeKMS(), np.random.default_rng(5)
    kms = fake.client()
    activate(conn, "1")
    people = [person(rng) for _ in range(20)]
    sids = [enroll(conn, kms, p[0])[0] for p in people]
    r = Replica(ro, None, fake.client, "recog", gpu=gpu)

    def search(c, probes, mv=MV, k=5, n=None):
        body = {"model_version": mv, "probes": np.asarray(probes, "<f4").tobytes(), "n": n or len(probes), "k": k}
        return c.post("/v1/search", content=msgpack.packb(body))

    def wait(c, cond, timeout=20):
        t = time.monotonic()
        while time.monotonic() - t < timeout:
            st = c.get("/v1/status").json()
            if cond(st):
                return st
            time.sleep(0.05)
        raise AssertionError(st)

    with TestClient(create_app(r)) as c:                          # lifespan: maintenance + batch threads
        st = wait(c, lambda s: s["ready"])
        assert (st["ntotal"], st["nsubjects"], st["model_version"]) == (80, 20, MV)
        resp = search(c, people[7][1])
        assert resp.status_code == 200 and resp.headers["content-type"] == "application/msgpack"
        out = msgpack.unpackb(resp.content)
        assert out["results"][0]["subject_id"] == sids[7] and len(out["results"]) == 5
        assert out["applied_event_id"] == st["applied_event_id"]
        assert search(c, people[7][1][:1]).status_code == 200     # n=1
        assert search(c, people[7][1], mv="recog@0").status_code == 409
        assert search(c, people[7][1] * 2).status_code == 400     # not normalised
        assert search(c, unit(np.ones((2, D + 1)))).status_code == 400   # wrong dimension
        assert search(c, people[7][1], n=2).status_code == 400    # n inconsistent with bytes
        assert search(c, people[7][1], k=0).status_code == 400
        assert c.post("/v1/search", content=b"\xc1").status_code == 400

        rm = delete(conn, fake, sids[7])                          # the API then waits for applied >= rm
        wait(c, lambda s: s["applied_event_id"] >= rm)
        out = msgpack.unpackb(search(c, people[7][1], k=100).content)
        assert sids[7] not in {x["subject_id"] for x in out["results"]} and len(out["results"]) == 19

        assert c.post("/v1/admin/reload").status_code == 202
        st = wait(c, lambda s: s["ready"] and s["tombstones"] == 0)   # rebuilt (compacted)
        assert (st["ntotal"], st["nsubjects"]) == (76, 19)
        conn.execute("UPDATE model_versions SET active = false")   # deactivated -> stop serving
        wait(c, lambda s: not s["ready"])
        assert search(c, people[1][1]).status_code == 503


def load_bulk_module():
    spec = importlib.util.spec_from_file_location("bulk_load", ROOT / "scripts/bulk_load.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_bulk_load_resumable_and_searchable(db):
    conn, ro = db
    bl, fake = load_bulk_module(), FakeKMS()
    activate(conn, "1")
    s1 = bl.load(conn, fake.client, MV, 0, 20, chunk=8, concurrency=4, dim=D, out=lambda s: None)
    s2 = bl.load(conn, fake.client, MV, 0, 30, chunk=8, concurrency=4, dim=D, out=lambda s: None)   # resume
    assert (s1["loaded"], s2["loaded"]) == (20, 10)
    cnt = conn.execute("SELECT (SELECT count(*) FROM subjects), (SELECT count(*) FROM templates),"
                       " (SELECT count(*) FROM template_events WHERE op = 'add')").fetchone()
    assert cnt == (30, 120, 120)
    r = Replica(ro, None, fake.client, "recog", gpu=False)
    r.build()
    assert (r.status()["ntotal"], r.status()["nsubjects"]) == (120, 30)
    sid, acct, centre, _ = bl.synthetic_subject(7, 0, D)
    res = ranked(r, centre[None])[0]
    assert res[0][0] == sid and res[0][1] > 0.7
