"""KMS service (internal, kms:8003), docs/IMPLEMENTATION.md §5. Data keys never leave this process unwrapped.

Per-subject 256-bit data keys live in DB `kms`, table subject_keys, wrapped with the master key
(AES-256-GCM, aad = subject_id). Template ciphertext = nonce(12) || AESGCM(subject_key).encrypt(nonce, pt,
aad=subject_id). DELETE = crypto-shredding: wrapped_key -> NULL and a tombstone, so every copy of that subject's
ciphertext (DB, replicas, backups) becomes undecryptable once no kms backup holds the old wrapped key
(docs/OPERATIONS.md §6). Fail closed: any DB error -> 503 and no key material is used.

Scopes: kms_token_api -> encrypt, destroy; kms_token_search -> decrypt. Bodies are msgpack.

    uvicorn faceauth.kms.main:app_from_env --factory --host 0.0.0.0 --port 8003
"""
import hmac
import logging
import os
import threading
import uuid

import msgpack
import psycopg
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from fastapi import FastAPI, HTTPException, Request, Response
from starlette.concurrency import run_in_threadpool

NONCE = 12
MAX_ITEMS = 65536  # per request; search loads its gallery in batches of at most this many
DEFAULT_DSN = "host=db,db-standby1,db-standby2 dbname=kms user=kms target_session_attrs=read-write connect_timeout=3"
log = logging.getLogger("faceauth.kms")


def seal(aead: AESGCM, pt: bytes, aad: bytes) -> bytes:
    nonce = os.urandom(NONCE)
    return nonce + aead.encrypt(nonce, pt, aad)


def unseal(aead: AESGCM, blob: bytes, aad: bytes) -> bytes:
    """Raises InvalidTag on a wrong key / aad or any tampering."""
    return aead.decrypt(blob[:NONCE], blob[NONCE:], aad)


class DB:
    """One connection per worker process. If the kept connection has died (e.g. after a failover) the call is
    retried once on a fresh one: every KMS operation is idempotent, so even a commit in doubt is safe to repeat.
    Any other failure propagates (-> 503) and the next call reconnects.

    ponytail: calls are serialised per process (each is ~1 ms); use psycopg_pool if KMS QPS ever matters.
    No pool on purpose: its exponential reconnect backoff would pad the failover RTO.
    """

    def __init__(self, conninfo: str, password: str | None):
        self.conninfo, self.password = conninfo, password
        self.lock = threading.Lock()
        self.conn = None

    def __call__(self, fn):
        with self.lock:
            kept = self.conn is not None
            try:
                return self._run(fn)
            except psycopg.OperationalError:
                if not kept:
                    raise
                return self._run(fn)

    def _run(self, fn):
        try:
            if self.conn is None:
                self.conn = psycopg.connect(self.conninfo, password=self.password, autocommit=True)
            return fn(self.conn)
        except psycopg.Error:
            if self.conn is not None:
                self.conn.close()
            self.conn = None
            raise


def _sid(v) -> str:
    try:
        return str(uuid.UUID(v))  # canonical form => identical AAD whoever formats the id
    except (TypeError, ValueError, AttributeError):
        raise HTTPException(400, "bad subject_id") from None


def _unpack(body: bytes) -> dict:
    try:
        obj = msgpack.unpackb(body, raw=False)
    except Exception:
        raise HTTPException(400, "malformed msgpack") from None
    if not isinstance(obj, dict) or not isinstance(obj.get("items"), list) or len(obj["items"]) > MAX_ITEMS:
        raise HTTPException(400, "bad body")
    return obj


def _pack(obj) -> Response:
    return Response(msgpack.packb(obj), media_type="application/msgpack")


def create_app(master_key: bytes, api_token: bytes, search_token: bytes, conninfo: str,
               password: str | None = None) -> FastAPI:
    if len(master_key) != 32:
        raise ValueError("master key must be 32 raw bytes")
    if not api_token or not search_token or api_token == search_token:
        raise ValueError("kms tokens must be non-empty and distinct")
    master = AESGCM(master_key)
    tokens = {"api": api_token, "search": search_token}
    db = DB(conninfo, password)
    app = FastAPI(title="faceauth-kms", docs_url=None, redoc_url=None, openapi_url=None)

    def require(req: Request, scope: str) -> None:
        auth = req.headers.get("authorization", "")
        presented = auth[7:].encode() if auth.startswith("Bearer ") else b""
        granted = [s for s, t in tokens.items() if presented and hmac.compare_digest(presented, t)]
        if not granted:
            raise HTTPException(401, "unauthorized")
        if scope not in granted:
            raise HTTPException(403, "forbidden")

    async def run_db(fn):
        try:
            return await run_in_threadpool(db, fn)
        except psycopg.Error:
            raise HTTPException(503, "kms database unavailable") from None

    @app.get("/healthz")
    async def healthz():
        await run_db(lambda c: c.execute("SELECT 1").fetchone())
        return {"ok": True}

    @app.post("/v1/encrypt")
    async def encrypt(req: Request):
        require(req, "api")
        body = _unpack(await req.body())
        sid = _sid(body.get("subject_id"))
        items = body["items"]
        if not all(isinstance(x, bytes) for x in items):
            raise HTTPException(400, "items must be bytes")
        fresh = seal(master, AESGCM.generate_key(256), sid.encode())

        def key_of(conn):
            # Created if absent. The commit waits for a sync standby. A key that is visible but not yet
            # replicated is still safe: whatever later stores the ciphertext commits later in the same WAL stream.
            with conn.transaction():
                conn.execute("INSERT INTO subject_keys (subject_id, wrapped_key) VALUES (%s, %s) "
                             "ON CONFLICT (subject_id) DO NOTHING", (sid, fresh))
                return conn.execute("SELECT wrapped_key FROM subject_keys WHERE subject_id = %s", (sid,)).fetchone()[0]

        wrapped = await run_db(key_of)
        if wrapped is None:
            raise HTTPException(410, "subject key destroyed")
        aead, aad = AESGCM(unseal(master, wrapped, sid.encode())), sid.encode()
        return _pack({"items": [seal(aead, pt, aad) for pt in items]})

    @app.post("/v1/decrypt")
    async def decrypt(req: Request):
        require(req, "search")
        body = _unpack(await req.body())
        try:
            items = [(_sid(x["subject_id"]), x["ct"]) for x in body["items"]]
        except (TypeError, KeyError):
            raise HTTPException(400, "items must be {subject_id, ct}") from None
        if not all(isinstance(ct, bytes) for _, ct in items):
            raise HTTPException(400, "ct must be bytes")
        sids = sorted({uuid.UUID(s) for s, _ in items})
        # Read on every call (no key cache): a destroyed key is never served, whichever replica destroyed it.
        rows = await run_db(lambda c: c.execute(
            "SELECT subject_id, wrapped_key FROM subject_keys WHERE subject_id = ANY(%s)", (sids,)).fetchall())

        def work():
            keys, bad = {}, 0
            for s, w in rows:
                try:
                    keys[str(s)] = AESGCM(unseal(master, w, str(s).encode())) if w is not None else None
                except InvalidTag:  # tampered row or wrong master key: that subject only, loudly
                    log.error("wrapped key of %s failed authentication", s)
            out = []
            for s, ct in items:  # None: unknown subject, destroyed key, or failed authentication
                k = keys.get(s)
                try:
                    out.append(unseal(k, ct, s.encode()) if k is not None and len(ct) > NONCE else None)
                except InvalidTag:
                    out.append(None)
                    bad += 1
            if bad:
                log.warning("decrypt: %d of %d items failed authentication", bad, len(items))
            return out

        return _pack({"items": await run_in_threadpool(work)})

    @app.delete("/v1/keys/{subject_id}")
    async def destroy(subject_id: str, req: Request):
        require(req, "api")
        sid = _sid(subject_id)

        def shred(conn):  # idempotent; the tombstone also blocks re-creating a key for a withdrawn subject
            conn.execute("INSERT INTO subject_keys (subject_id, wrapped_key, destroyed_at) VALUES (%s, NULL, now()) "
                         "ON CONFLICT (subject_id) DO UPDATE SET wrapped_key = NULL, "
                         "destroyed_at = coalesce(subject_keys.destroyed_at, now())", (sid,))

        await run_db(shred)  # returns after the commit is on a sync standby
        return {"destroyed": True}

    return app


def _read(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def app_from_env() -> FastAPI:
    env = os.environ.get
    return create_app(
        _read(env("FA_KMS_MASTER_KEY_FILE", "/run/secrets/kms_master_key")),
        _read(env("FA_KMS_API_TOKEN_FILE", "/run/secrets/kms_token_api")).strip(),
        _read(env("FA_KMS_SEARCH_TOKEN_FILE", "/run/secrets/kms_token_search")).strip(),
        env("FA_KMS_DB_DSN", DEFAULT_DSN),
        _read(env("FA_KMS_DB_PASSWORD_FILE", "/run/secrets/db_kms_password")).decode().strip(),
    )
