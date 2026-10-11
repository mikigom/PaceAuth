"""Search replica (spec §4): exact 1:N over the active templates of the active recognition model.

    uvicorn --factory faceauth.search.app:create_app --host 0.0.0.0 --port 8002

Build (startup, POST /v1/admin/reload, active model change): model_version = FA_RECOG_MODEL@<active version>;
snapshot = max(template_events.id) BEFORE reading templates; read active templates (server-side cursor), decrypt
via KMS (batched, concurrent; null = key destroyed (crypto-shredded) / unknown / ciphertext failed authentication
-> skipped, counted in the build log); apply events > snapshot; then ready.
Poll: every 200 ms apply template_events with id > applied. Applying is idempotent (adds re-read the template row
and skip inactive/present ones, removes of absent ids are no-ops), so the snapshot/read overlap is harmless.
Rebuild = 503 until the new index is ready (an atomic swap would need a second 8 GB index on the GPU):
reload replicas one at a time.
Fail closed (503): not built, no active model, build error (DB, KMS, corrupt template), no successful poll for
MAX_STALE_S, top-k not provably exact. Event ids going backwards (DB restore) or a model change -> rebuild.
"""
import asyncio
import collections
import concurrent.futures
import functools
import logging
import os
import queue
import threading
import time
from contextlib import asynccontextmanager

import httpx
import msgpack
import numpy as np
import psycopg
from fastapi import FastAPI, HTTPException, Request, Response

from faceauth.search.gallery import Gallery

log = logging.getLogger("faceauth.search")

POLL_S = 0.2
MAX_STALE_S = 5.0
POLL_LIMIT = 5000        # events per poll query
KMS_BATCH = 1000         # items per KMS decrypt call
KMS_CONCURRENCY = 8
FLUSH_ROWS = 1 << 16     # build: decrypted rows per gallery.add (each add copies the open segment once)
MAX_BATCH = 256          # queries per GPU search call
MAX_PROBES = 64
MAX_RESULTS = 100


class NotReady(Exception):
    pass


class Mismatch(Exception):
    pass


def kms_decrypt(kms: httpx.Client, items: list) -> list:
    """[(subject_id, ct)] -> [plaintext | None (key destroyed/unknown or ct failed authentication)]. Else raises."""
    if not items:
        return []
    r = kms.post("/v1/decrypt", content=msgpack.packb({"items": [{"subject_id": s, "ct": c} for s, c in items]}),
                 headers={"Content-Type": "application/msgpack"})
    r.raise_for_status()
    out = msgpack.unpackb(r.content)["items"]
    if len(out) != len(items) or not all(p is None or isinstance(p, bytes) for p in out):
        raise ValueError("malformed KMS response")
    return out


def decrypt_rows(new_kms, batches):
    """Batches of (template_id, subject_id, ct) rows -> yields (tids, sids, vecs) in order, shredded rows dropped.
    At most 2*KMS_CONCURRENCY KMS calls are in flight, so a 4M-row build never holds all ciphertexts.
    new_kms() -> httpx.Client; one per worker thread (a client shared by threads failed with EBADF under load)."""
    local, clients = threading.local(), []

    def call(items):
        if not hasattr(local, "kms"):
            local.kms = new_kms()
            clients.append(local.kms)
        return kms_decrypt(local.kms, items)

    def done(rows, fut):
        keep = [(r[0], r[1], p) for r, p in zip(rows, fut.result()) if p is not None]
        if not keep:
            return [], [], np.zeros((0, 0), np.float32)
        size = len(keep[0][2])
        if size % 4 or any(len(p) != size for _, _, p in keep):
            raise ValueError("template plaintext length mismatch")
        vecs = np.frombuffer(b"".join(p for _, _, p in keep), "<f4").reshape(len(keep), size // 4)
        return [k[0] for k in keep], [k[1] for k in keep], vecs

    try:
        with concurrent.futures.ThreadPoolExecutor(KMS_CONCURRENCY) as ex:
            pending = collections.deque()
            for rows in batches:
                pending.append((rows, ex.submit(call, [(r[1], r[2]) for r in rows])))
                if len(pending) >= 2 * KMS_CONCURRENCY:
                    yield done(*pending.popleft())
            while pending:
                yield done(*pending.popleft())
    finally:
        for c in clients:
            c.close()


class Replica:
    def __init__(self, dsn: str, password: str | None, new_kms, recog_model: str, gpu: bool = True):
        """new_kms() -> httpx.Client for the KMS (base_url + search-scope bearer token)."""
        self.dsn, self.password, self.new_kms, self.recog, self.gpu = dsn, password, new_kms, recog_model, gpu
        self.lock = threading.Lock()   # guards gallery/model_version/applied; faiss GPU indexes are not thread-safe
        self.gallery = None            # written only by the maintenance thread (run/build/poll), under the lock
        self.model_version = None
        self.applied = 0
        self.applied_sig = None        # (op, template_id) of event `applied`: detects ids lost and reissued by a failover
        self.last_ok = 0.0             # monotonic time of the last successful build/poll
        self.reload = threading.Event()
        self.jobs = queue.Queue()
        self.build_seconds = None
        self._conn = None

    @classmethod
    def from_env(cls):
        e = os.environ

        def secret(var):
            with open(e[var]) as f:
                return f.read().strip()

        new_kms = functools.partial(httpx.Client, base_url=e["FA_KMS_URL"], timeout=30.0,
                                    headers={"Authorization": "Bearer " + secret("FA_KMS_TOKEN_FILE")})
        return cls(e["FA_DB_DSN"], secret("FA_DB_PASSWORD_FILE"), new_kms, e.get("FA_RECOG_MODEL", "kprpe_vitb_wf12m"))

    # ---- state -----------------------------------------------------------------------------------------------
    def ready(self) -> bool:
        return self.gallery is not None and time.monotonic() - self.last_ok < MAX_STALE_S

    def status(self) -> dict:
        with self.lock:
            g = self.gallery
            return {"ready": self.ready(), "model_version": self.model_version if g else None,
                    "ntotal": g.ntotal if g else 0, "nsubjects": g.nsubjects if g else 0,
                    "applied_event_id": self.applied if g else 0, "tombstones": g.n - g.ntotal if g else 0,
                    "build_seconds": self.build_seconds}

    # ---- maintenance thread: build + poll -----------------------------------------------------------------------
    def _db(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(self.dsn, password=self.password, autocommit=True)
        return self._conn

    def _close_db(self):
        if self._conn is not None:
            self._conn.close()
        self._conn = None

    def _active(self, conn):
        """-> (model_version | None, max event id, server is a standby)."""
        v, max_id, standby = conn.execute(
            "SELECT (SELECT version FROM model_versions WHERE name = %s AND active),"
            " (SELECT coalesce(max(id), 0) FROM template_events), pg_is_in_recovery()", (self.recog,)).fetchone()
        return (None if v is None else f"{self.recog}@{v}"), max_id, standby

    @staticmethod
    def _sig(conn, eid):
        return tuple(conn.execute("SELECT op, template_id FROM template_events WHERE id = %s", (eid,)).fetchone() or ()) \
            if eid else None

    def build(self):
        with self.lock:
            self.gallery = None        # 503 from here on; frees the old GPU index before the new one is built
            self.model_version = None
        t0 = time.monotonic()
        with psycopg.connect(self.dsn, password=self.password) as conn:
            mv, snapshot, _ = self._active(conn)   # snapshot BEFORE the template read
            if mv is None:
                raise NotReady(f"no active version of {self.recog}")
            g = Gallery(self.gpu)
            with conn.cursor(name="templates") as cur:
                cur.execute("SELECT id, subject_id::text, ciphertext FROM templates"
                            " WHERE active AND model_version = %s AND ciphertext IS NOT NULL ORDER BY id", (mv,))
                tids, sids, vecs = [], [], []
                for t, s, v in decrypt_rows(self.new_kms, iter(lambda: cur.fetchmany(KMS_BATCH), [])):
                    if t:
                        tids += t
                        sids += s
                        vecs.append(v)
                    if len(tids) >= FLUSH_ROWS:
                        g.add(tids, sids, np.vstack(vecs))
                        tids, sids, vecs = [], [], []
                if tids:
                    g.add(tids, sids, np.vstack(vecs))
                skipped = (cur.rownumber or 0) - g.ntotal
        applied = snapshot
        while True:                                            # catch up before declaring ready
            applied, n = self._apply_events(g, mv, applied)
            if n < POLL_LIMIT:
                break
        sig = self._sig(self._db(), applied)
        with self.lock:
            self.gallery, self.model_version, self.applied, self.last_ok = g, mv, applied, time.monotonic()
            self.applied_sig = sig
            self.build_seconds = round(time.monotonic() - t0, 3)
        log.info("search index %s built: %d templates (%d skipped: KMS null), %d subjects, applied_event_id %d, %.1f s",
                 mv, g.ntotal, skipped, g.nsubjects, applied, self.build_seconds)

    def _apply_events(self, g, mv, applied):
        """Apply up to POLL_LIMIT events with id > applied to g. -> (new applied, events read)."""
        conn = self._db()
        ev = conn.execute("SELECT id, op, template_id, model_version FROM template_events"
                          " WHERE id > %s ORDER BY id LIMIT %s", (applied, POLL_LIMIT)).fetchall()
        if not ev:
            return applied, 0
        add_ids = sorted({t for _, op, t, m in ev if op == "add" and m == mv and t not in g.row_of})
        rows = conn.execute("SELECT id, subject_id::text, ciphertext FROM templates WHERE id = ANY(%s)"
                            " AND active AND model_version = %s AND ciphertext IS NOT NULL ORDER BY id",
                            (add_ids, mv)).fetchall() if add_ids else []
        dec = list(decrypt_rows(self.new_kms, [rows[i:i + KMS_BATCH] for i in range(0, len(rows), KMS_BATCH)]))
        removes = [t for _, op, t, m in ev if op == "remove" and m == mv]
        with self.lock:
            try:
                for tids, sids, vecs in dec:
                    if tids:
                        g.add(tids, sids, vecs)
                g.remove(removes)
            except Exception:
                if self.gallery is g:
                    self.gallery = None                # possibly inconsistent: stop serving, rebuild
                raise
        return ev[-1][0], len(ev)

    def poll(self) -> int:
        g, mv, applied = self.gallery, self.model_version, self.applied
        active, max_id, standby = self._active(self._db())
        if max_id < applied and standby:
            return 0        # reconnected to a standby still replaying: wait (stale -> 503 after MAX_STALE_S)
        # Sync replication makes a commit visible on the primary before a standby has it: after a failover the
        # events we applied may be gone and their ids reissued (an id >= applied would then be skipped forever).
        lost = applied and self._sig(self._db(), applied) != self.applied_sig
        if active != mv or max_id < applied or lost:   # model (de)activated, or the primary lost/restored history
            log.warning("search: active model %s (index %s), max event %d (applied %d, history %s) -> rebuild",
                        active, mv, max_id, applied, "rewritten" if lost else "ok")
            self.reload.set()
            return 0
        applied, n = self._apply_events(g, mv, applied)
        sig = self._sig(self._db(), applied) if n else self.applied_sig
        with self.lock:
            if self.gallery is g:
                self.applied, self.applied_sig, self.last_ok = applied, sig, time.monotonic()
        return n

    def run(self, stop: threading.Event):
        while not stop.is_set():
            try:
                if self.gallery is None or self.reload.is_set():
                    self.reload.clear()
                    self.build()
                    continue
                if self.poll() >= POLL_LIMIT or self.reload.is_set():
                    continue
            except Exception as e:  # dependency error: stay/become not ready, retry
                log.warning("search %s failed: %r", "poll" if self.gallery is not None else "build", e)
                self._close_db()
                if self.gallery is None:
                    stop.wait(2.0)
                    continue
            stop.wait(POLL_S)

    # ---- batch thread: GPU search -------------------------------------------------------------------------------
    def submit(self, model_version: str, q: np.ndarray, k: int) -> concurrent.futures.Future:
        f = concurrent.futures.Future()
        self.jobs.put((model_version, q, k, f))
        return f

    def batch_loop(self):
        """Opportunistic micro-batching: everything queued while the previous GPU search ran goes into one call."""
        while True:
            jobs = [self.jobs.get()]
            while len(jobs) < MAX_BATCH and not self.jobs.empty():
                jobs.append(self.jobs.get_nowait())
            # a request whose client went away is cancelled; claiming the rest makes set_result safe
            live = [j for j in jobs if j is not None and j[3].set_running_or_notify_cancel()]
            try:
                if live:
                    self.search_batch(live)
            except Exception as e:  # never let this thread die: every later search would hang
                log.exception("search batch failed")
                for j in live:
                    if not j[3].done():
                        j[3].set_exception(e)
            if None in jobs:
                return

    def search_batch(self, jobs):
        res = {}
        with self.lock:
            g, mv, applied, ready = self.gallery, self.model_version, self.applied, self.ready()
            err = [NotReady() if not ready else Mismatch() if j[0] != mv else
                   ValueError("probe dimension") if g.dim not in (None, len(j[1])) else None for j in jobs]
            idx = [i for i, e in enumerate(err) if e is None]
            try:
                if idx and g.n:
                    res = dict(zip(idx, g.search(np.stack([jobs[i][1] for i in idx]), max(jobs[i][2] for i in idx))))
            except Exception as e:
                log.error("search failed: %r", e)
                for i in idx:
                    err[i] = e
        for i, r in res.items():
            if r is None:   # exact top-k not reachable within MAX_K rows
                err[i] = RuntimeError("search not exact")
        for i, ((_, _, k, f), e) in enumerate(zip(jobs, err)):
            if e is not None:
                f.set_exception(e)
            else:
                f.set_result((res.get(i, [])[:k], applied))


def create_app(replica: Replica | None = None) -> FastAPI:
    if replica is None:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
        logging.getLogger("httpx").setLevel(logging.WARNING)   # one INFO line per KMS call otherwise
        replica = Replica.from_env()

    @asynccontextmanager
    async def lifespan(app):
        stop = threading.Event()
        for target, args in ((replica.run, (stop,)), (replica.batch_loop, ())):
            threading.Thread(target=target, args=args, daemon=True).start()
        yield
        stop.set()
        replica.jobs.put(None)

    app = FastAPI(lifespan=lifespan)
    app.state.replica = replica

    @app.get("/v1/status")
    def status():
        return replica.status()

    @app.post("/v1/admin/reload", status_code=202)
    def reload():
        # Asynchronous: poll /v1/status until ready before reloading the next replica. A request during a build
        # is not queued again: that build (or the poller's model check right after it) already covers it.
        if replica.gallery is not None:
            replica.reload.set()
        return {"reloading": True}

    @app.post("/v1/search")
    async def search(request: Request):
        try:
            req = msgpack.unpackb(await request.body())
            mv, n, k, raw = req["model_version"], req["n"], req.get("k", 5), req["probes"]
            if not (isinstance(mv, str) and type(n) is int and type(k) is int and isinstance(raw, bytes)
                    and 1 <= n <= MAX_PROBES and 1 <= k <= MAX_RESULTS and raw and len(raw) % (4 * n) == 0):
                raise ValueError
            probes = np.frombuffer(raw, "<f4").reshape(n, -1)
        except Exception:
            raise HTTPException(400, "malformed search request")
        if not np.all(np.abs(np.linalg.norm(probes, axis=1) - 1) < 1e-2):
            raise HTTPException(400, "probes must be finite and L2-normalised")
        q = probes.mean(0, dtype=np.float64)          # aggregation v1: q = normalize(mean(probes))
        if np.linalg.norm(q) < 1e-6:
            raise HTTPException(400, "degenerate probe mean")
        q = (q / np.linalg.norm(q)).astype(np.float32)
        try:
            results, applied = await asyncio.wrap_future(replica.submit(mv, q, k))
        except NotReady:
            raise HTTPException(503, "index not ready")
        except Mismatch:
            raise HTTPException(409, "model_version mismatch")
        except ValueError as e:
            raise HTTPException(400, str(e))
        except Exception:
            raise HTTPException(503, "search failed")
        return Response(msgpack.packb({"results": [{"subject_id": s, "score": x} for s, x in results],
                                       "applied_event_id": applied}), media_type="application/msgpack")

    return app
