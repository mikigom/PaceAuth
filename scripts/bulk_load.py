"""Bulk-load synthetic subjects (4 templates each) through the real write path, for load tests (E9).

    python scripts/bulk_load.py --subjects 1000000 [--start 0] [--chunk 1000] [--concurrency 32]

Per subject: account_id 'synthetic-<n>', status active, consent fields; 4 unit-vector templates around a random
centre, encrypted by KMS POST /v1/encrypt (kms_token_api). Per chunk, one transaction: COPY subjects, COPY
templates, pg_advisory_xact_lock(7001), template_events 'add' rows. Resumable: chunks are atomic and existing
synthetic accounts are skipped (subject ids are uuid5 of the account, so a retried subject reuses its KMS key).
Env: FA_DB_DSN (role faceauth_api), FA_DB_PASSWORD_FILE, FA_KMS_URL, FA_KMS_TOKEN_FILE, FA_RECOG_MODEL,
FA_CONSENT_VERSION. Prints one JSON line per chunk and a summary line.
"""
import argparse
import concurrent.futures
import datetime
import functools
import itertools
import json
import os
import threading
import time
import uuid

import httpx
import msgpack
import numpy as np
import psycopg

NS = uuid.UUID("6f6e2a52-58d4-4c8e-9a51-3f1c0b7a9e10")
ALIGN, CALIB = "synthetic", "gemini336-sim-v1"


def synthetic_subject(n: int, seed: int = 0, dim: int = 512):
    """-> (subject_id, account_id, centre (dim,), templates (4, dim)); deterministic so load tests can build probes.
    Template-centre cosine ~0.8 (same-person-like)."""
    rng = np.random.default_rng([seed, n])
    c = rng.standard_normal(dim)
    c /= np.linalg.norm(c)
    t = c + 0.75 * rng.standard_normal((4, dim)) / np.sqrt(dim)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    return str(uuid.uuid5(NS, f"synthetic-{n}")), f"synthetic-{n}", c.astype(np.float32), t.astype("<f4")


def kms_encrypt(kms: httpx.Client, subject_id: str, items: list) -> list:
    r = kms.post("/v1/encrypt", content=msgpack.packb({"subject_id": subject_id, "items": items}),
                 headers={"Content-Type": "application/msgpack"})
    r.raise_for_status()
    out = msgpack.unpackb(r.content)["items"]
    if len(out) != len(items) or not all(isinstance(c, bytes) for c in out):
        raise ValueError("malformed KMS response")
    return out


def active_model_version(conn, name: str) -> str:
    row = conn.execute("SELECT version FROM model_versions WHERE name = %s AND active", (name,)).fetchone()
    if row is None:
        raise SystemExit(f"no active model_versions row for {name}; pass --model-version")
    return f"{name}@{row[0]}"


def _chunks(conn, start, count, chunk, seed, dim):
    """-> (upto, subjects not yet in the DB) per chunk."""
    for c0 in range(start, start + count, chunk):
        ns = range(c0, min(c0 + chunk, start + count))
        have = {r[0] for r in conn.execute("SELECT id FROM subjects WHERE id = ANY(%s)",
                                           ([uuid.uuid5(NS, f"synthetic-{n}") for n in ns],))}
        subs = [synthetic_subject(n, seed, dim) for n in ns if uuid.uuid5(NS, f"synthetic-{n}") not in have]
        if subs:
            yield ns.stop, subs


def _write(conn, subs, cts, model_version, consent_version):
    """One transaction: subjects, templates, then the outbox rows under lock 7001 (ids visible in commit order)."""
    now = datetime.datetime.now(datetime.timezone.utc)
    with conn.transaction(), conn.cursor() as cur:
        with cur.copy("COPY subjects (id, account_id, status, consent_version, consent_at)"
                      " FROM STDIN (FORMAT BINARY)") as cp:
            cp.set_types(["uuid", "text", "text", "text", "timestamptz"])
            for sid, acct, *_ in subs:
                cp.write_row((uuid.UUID(sid), acct, "active", consent_version, now))
        with cur.copy("COPY templates (subject_id, template_version, slot, model_version, align_version, calib_version,"
                      " ciphertext) FROM STDIN (FORMAT BINARY)") as cp:
            cp.set_types(["uuid", "int4", "int2", "text", "text", "text", "bytea"])
            for (sid, *_), ct in zip(subs, cts):
                for slot, c in enumerate(ct):
                    cp.write_row((uuid.UUID(sid), 1, slot, model_version, ALIGN, CALIB, c))
        cur.execute("SELECT pg_advisory_xact_lock(7001)")
        cur.execute("INSERT INTO template_events (op, template_id, subject_id, model_version)"
                    " SELECT 'add', id, subject_id, model_version FROM templates"
                    " WHERE subject_id = ANY(%s) ORDER BY id",
                    ([uuid.UUID(s[0]) for s in subs],))


def load(conn, new_kms, model_version: str, start: int, count: int, chunk: int = 1000,
         concurrency: int = 32, seed: int = 0, dim: int = 512, consent_version: str = "synthetic-v1", out=print):
    """new_kms() -> httpx.Client, one per worker thread (one client shared by threads failed with EBADF).
    KMS calls of chunk i+1 run while chunk i is written to the DB."""
    t0, done, pending, local, clients = time.monotonic(), 0, None, threading.local(), []

    def encrypt(s):
        if not hasattr(local, "kms"):
            local.kms = new_kms()
            clients.append(local.kms)
        return kms_encrypt(local.kms, s[0], [t.tobytes() for t in s[3]])

    with concurrent.futures.ThreadPoolExecutor(concurrency) as ex:
        for upto, subs in itertools.chain(_chunks(conn, start, count, chunk, seed, dim), [(None, None)]):
            nxt = subs and (upto, subs, [ex.submit(encrypt, s) for s in subs])
            if pending:
                p_upto, p_subs, futs = pending
                t1 = time.monotonic()
                cts = [f.result() for f in futs]
                t2 = time.monotonic()
                _write(conn, p_subs, cts, model_version, consent_version)
                done += len(p_subs)
                out(json.dumps({"upto": p_upto, "loaded": done, "kms_wait_s": round(t2 - t1, 3),
                                "db_s": round(time.monotonic() - t2, 3),
                                "subjects_per_s": round(done / (time.monotonic() - t0), 1)}))
            pending = nxt
    for c in clients:
        c.close()
    el = time.monotonic() - t0
    summary = {"model_version": model_version, "range": [start, start + count], "loaded": done,
               "templates": 4 * done, "seconds": round(el, 1), "subjects_per_s": round(done / el, 1) if el else None}
    out(json.dumps(summary))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", type=int, required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=1000)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--model-version", help="default: FA_RECOG_MODEL@<active version>")
    a = ap.parse_args()
    e = os.environ

    def secret(var):
        with open(e[var]) as f:
            return f.read().strip()

    dsn = e.get("FA_DB_DSN", "host=db,db-standby1,db-standby2 dbname=faceauth user=faceauth_api"
                             " target_session_attrs=read-write")
    pw = secret("FA_DB_PASSWORD_FILE") if "FA_DB_PASSWORD_FILE" in e else None
    new_kms = functools.partial(httpx.Client, base_url=e.get("FA_KMS_URL", "http://kms:8003"), timeout=60.0,
                                headers={"Authorization": "Bearer " + secret("FA_KMS_TOKEN_FILE")})
    with psycopg.connect(dsn, password=pw, autocommit=True) as conn:
        mv = a.model_version or active_model_version(conn, e.get("FA_RECOG_MODEL", "kprpe_vitb_wf12m"))
        load(conn, new_kms, mv, a.start, a.subjects, a.chunk, a.concurrency, a.seed, a.dim,
             e.get("FA_CONSENT_VERSION", "synthetic-v1"))


if __name__ == "__main__":
    main()
