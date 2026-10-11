"""Central API (docs/IMPLEMENTATION.md §6, Korean reference docs/API.md): sessions, enrollment, 1:N authentication
policy, step-up, non-biometric fallback, deletion and admin. Device traffic arrives only through the mTLS gateway.

Invariants
- Fail closed: any DB / engine / search / KMS error -> 503, never a relaxed decision path.
- Devices never see candidates, scores or internal reasons (those go to sessions.reason + the audit log).
- A session is consumed (UPDATE ... WHERE consumed_at IS NULL AND expires_at > now() RETURNING) in the same
  transaction as its decision, approval / step-up row and audit row; the audit row is always the LAST statement.
- Template outbox events are written under pg_advisory_xact_lock(7001).

Run: uvicorn faceauth.api.app:app --host 0.0.0.0 --port 8000 --no-proxy-headers --timeout-keep-alive 75
"""
import asyncio
import hashlib
import hmac
import logging
import os
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Literal
from urllib.parse import unquote

import psycopg
from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, Field

from faceauth.api import policy
from faceauth.api.clients import Deps, Unavailable
from faceauth.common import token
from faceauth.common.packet import b64u, parse_packet, verify_packet

log = logging.getLogger("faceauth.api")

TTL = {"enroll": 180, "auth": 30}          # session lifetime, seconds (§6)
APPROVAL_TTL, STEPUP_TTL = 30, 120         # approval token exp = iat + 30 s; step-up must complete within 120 s
LOCKOUT_REJECTS = 10                       # >= 10 rejected auth decisions of a device in 5 min -> 429
MAX_FRAMES = {"enroll": 64, "auth": 16}
MAX_BODY = 32 << 20                        # the gateway enforces the same limit
C = SimpleNamespace(cfg=None, pool=None, deps=None)


class _SubjectGone(Exception):
    """Re-enroll target stopped being active (revoked / sent to review) between the pre-check and the tx."""


def _file(env: str, default: str) -> bytes:
    return Path(os.environ.get(env, default)).read_bytes()


class Config:
    """Environment of docs/IMPLEMENTATION.md §11 (+ secret file paths). Missing values abort startup."""

    def __init__(self):
        e = os.environ
        self.db_dsn = e["FA_DB_DSN"]
        self.db_password = _file("FA_DB_PASSWORD_FILE", "/run/secrets/db_api_password").decode().strip()
        self.kms_url = e.get("FA_KMS_URL", "http://kms:8003")
        self.kms_token = _file("FA_KMS_TOKEN_FILE", "/run/secrets/kms_token_api").decode().strip()
        self.search_urls = [u.strip() for u in e.get("FA_SEARCH_URLS", "http://search-1:8002,http://search-2:8002")
                            .split(",") if u.strip()]
        self.engine_urls = [u.strip() for u in e.get("FA_ENGINE_URLS", "http://engine:8001").split(",") if u.strip()]
        self.recog_model = e["FA_RECOG_MODEL"]
        self.t_id, self.margin, self.t_dup, self.t_same = (float(e[k]) for k in
                                                           ("FA_T_ID", "FA_MARGIN", "FA_T_DUP", "FA_T_SAME"))
        self.consent_version = e["FA_CONSENT_VERSION"]
        self.high_risk = {a.strip() for a in e.get("FA_HIGH_RISK_ACTIONS", "").split(",") if a.strip()}
        self.min_dirs = int(e.get("FA_ENROLL_MIN_DIRS", "9"))
        # device identity headers are honoured only from these peers (hostnames or IPs): the gateway
        self.trusted_proxies = [h.strip() for h in e.get("FA_TRUSTED_PROXIES", "gateway").split(",") if h.strip()]
        self.approval_key = serialization.load_pem_private_key(
            _file("FA_APPROVAL_KEY_FILE", "/run/secrets/approval_key"), None)
        self.account_pub = serialization.load_pem_public_key(_file("FA_ACCOUNT_PUB_FILE", "/run/secrets/account_pub"))
        self.admin_token = _file("FA_ADMIN_TOKEN_FILE", "/run/secrets/admin_token").decode().strip()
        if not (isinstance(self.approval_key, Ed25519PrivateKey) and isinstance(self.account_pub, Ed25519PublicKey)
                and self.search_urls and self.engine_urls and self.admin_token):
            raise RuntimeError("invalid api configuration")


@asynccontextmanager
async def lifespan(_app):
    cfg = Config()
    pool = AsyncConnectionPool(cfg.db_dsn, open=False, min_size=2, max_size=20, timeout=5.0,
                               kwargs={"password": cfg.db_password, "autocommit": True, "row_factory": dict_row},
                               check=AsyncConnectionPool.check_connection)
    await pool.open(wait=False)  # start even if the DB is down: requests then fail closed (503)
    C.cfg, C.pool = cfg, pool
    C.deps = Deps(cfg.engine_urls, cfg.search_urls, cfg.kms_url, cfg.kms_token, cfg.admin_token)
    try:
        yield
    finally:
        await C.deps.close()
        await pool.close()


app = FastAPI(title="faceauth api", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


@app.exception_handler(Unavailable)
@app.exception_handler(psycopg.Error)
async def _fail_closed(request: Request, exc: Exception):
    log.warning("fail closed (503) on %s %s: %r", request.method, request.url.path, exc)
    return JSONResponse({"detail": "unavailable"}, status_code=503)


# ---------------------------------------------------------------- helpers

async def audit(conn, actor: str, event: str, *, subject_id=None, session_id=None, device_id=None, detail=None):
    """Hash-chained, append-only audit row (the trigger sets id/prev_hash/hash). LAST statement of its tx."""
    await conn.execute("INSERT INTO audit_log (actor, event, subject_id, session_id, device_id, detail) "
                       "VALUES (%s, %s, %s, %s, %s, %s)",
                       (actor, event, subject_id, session_id, device_id, Jsonb(detail or {})))


async def _one(conn, sql: str, params=()):
    return await (await conn.execute(sql, params)).fetchone()


def _admin(request: Request):
    got = request.headers.get("authorization", "").encode()
    if not hmac.compare_digest(got, b"Bearer " + C.cfg.admin_token.encode()):
        raise HTTPException(401, "unauthorized")


async def _via_gateway(request: Request) -> bool:
    """X-Client-* headers are trusted only from the gateway (the TCP peer, uvicorn runs with --no-proxy-headers)."""
    peer = request.client.host if request.client else None
    loop = asyncio.get_running_loop()
    for host in C.cfg.trusted_proxies:
        if peer == host:
            return True
        try:
            if any(info[4][0] == peer for info in await loop.getaddrinfo(host, None)):
                return True
        except OSError:
            continue
    return False


async def _device(request: Request, conn, device_id) -> dict:
    """mTLS identity (§6): the gateway-verified client cert must be the claimed device's registered cert
    (sha256 of DER == devices.cert_fingerprint) and the device must be active."""
    device_id, fp, why = str(device_id)[:64], None, None
    if not await _via_gateway(request):
        why = "untrusted_peer"
    elif request.headers.get("x-client-verify") != "SUCCESS":
        why = "client_cert_unverified"
    else:
        try:
            cert = x509.load_pem_x509_certificate(unquote(request.headers["x-client-cert"]).encode())
            fp = hashlib.sha256(cert.public_bytes(serialization.Encoding.DER)).hexdigest()
        except (KeyError, ValueError):
            why = "client_cert_invalid"
    dev = None if why else await _one(conn, "SELECT * FROM devices WHERE id = %s", (device_id,))
    if why is None:
        if dev is None:
            why = "unknown_device"
        elif not hmac.compare_digest(dev["cert_fingerprint"], fp):
            why = "cert_mismatch"
        elif dev["status"] != "active":
            why = "device_disabled"
    if why:
        await audit(conn, "gateway", "device_rejected", device_id=device_id,
                    detail={"reason": why, "path": request.url.path, "cert_fingerprint": fp})
        if why in ("untrusted_peer", "client_cert_unverified", "client_cert_invalid"):
            raise HTTPException(401, "client_certificate_required")
        raise HTTPException(403, "device_not_authorized")
    return dev


def _actor(dev) -> str:
    return f"device:{dev['id']}"


def _packet(body: bytes):
    if len(body) > MAX_BODY:
        raise HTTPException(413, "packet_too_large")
    try:
        header, sig, frames = parse_packet(body)
        return header, sig, frames, uuid.UUID(str(header["session_id"]))
    except (ValueError, KeyError, TypeError):
        raise HTTPException(400, "malformed_packet") from None


def _integrity(header: dict, sig: bytes, frames: list, dev: dict, purpose: str) -> list:
    """Signature, hashes, order, sync (verify_packet) + calibration version + frame count. [] = ok."""
    try:
        errs = verify_packet(header, sig, frames, Ed25519PublicKey.from_public_bytes(bytes(dev["public_key"])))
    except Exception:  # noqa: BLE001 - validly signed but structurally malformed header / frames
        errs = ["malformed_packet"]
    if errs == ["bad_signature"]:
        return errs  # nothing else in the header can be trusted
    if header.get("calib_version") != dev["calib_version"]:
        errs.append("calib_mismatch")
    if len(frames) > MAX_FRAMES[purpose]:
        errs.append("too_many_frames")
    return errs


async def _session(conn, sid: uuid.UUID, device_id: str, purpose: str, challenge):
    """-> (session, reason). session is None when it must not be consumed by this request (absent, another
    device's, wrong purpose, already consumed or expired); else reason is None or "challenge_mismatch"."""
    s = await _one(conn, "SELECT *, consumed_at IS NULL AS unconsumed, expires_at > now() AS live "
                         "FROM sessions WHERE id = %s", (sid,))
    if s is None:
        return None, "session_not_found"
    if s["device_id"] != device_id:
        return None, "session_device_mismatch"
    if s["purpose"] != purpose:
        return None, "session_purpose_mismatch"
    if not s["unconsumed"]:
        return None, "session_consumed"
    if not s["live"]:
        return None, "session_expired"
    if challenge is not None and not hmac.compare_digest(s["challenge"].encode(), str(challenge).encode()):
        return s, "challenge_mismatch"
    return s, None


async def _consume(conn, sid, decision: str, reason: str) -> bool:
    """Atomic single use: False => already consumed / expired meanwhile (replay) -> the caller rejects."""
    row = await _one(conn, "UPDATE sessions SET consumed_at = now(), decision = %s, reason = %s "
                           "WHERE id = %s AND consumed_at IS NULL AND expires_at > now() RETURNING id",
                     (decision, reason, sid))
    return row is not None


async def _active_recog(conn):
    row = await _one(conn, "SELECT version FROM model_versions WHERE name = %s AND active", (C.cfg.recog_model,))
    return f"{C.cfg.recog_model}@{row['version']}" if row else None


def _check_model(an: dict, active):
    """Templates and probes of different model versions are never compared or stored."""
    if active is None or an["versions"]["recog"] != active:
        raise Unavailable(f"recognition model mismatch: engine {an['versions']['recog']!r}, active {active!r}")


def _embeddings(an: dict):
    try:
        return policy.embeddings(an["frames"], int(an["dim"]))
    except ValueError as e:
        raise Unavailable(f"engine: {e}") from e


def _direction_of():
    try:
        from faceauth.terminal.guide import direction_of  # the terminal's bins, so both sides agree
    except ImportError as e:
        raise Unavailable("faceauth.terminal.guide is missing") from e
    return direction_of


def _assertion(tok, nonce):
    """Account-service token {"acct","method","nonce","iat","exp"} bound to `nonce` -> (payload, reason)."""
    try:
        a = token.verify(C.cfg.account_pub, str(tok))
    except (ValueError, TypeError):
        return None, "assertion_invalid"
    if not (isinstance(a.get("acct"), str) and a["acct"] and a.get("nonce") == nonce):
        return None, "assertion_mismatch"
    return a, None


def _factor(a) -> str | None:
    """Customer-owned non-face factor named by an assertion (pin, otp, ...)."""
    m = a.get("method") if a else None
    return m if isinstance(m, str) and m and m != "face" and len(m) <= 32 else None


def _consent(c):
    """-> agreed_at if the consent is for FA_CONSENT_VERSION with a timezone-aware, not-future time, else None."""
    try:
        t = datetime.fromisoformat(str(c["agreed_at"]))
        ok = (c["version"] == C.cfg.consent_version and t.tzinfo is not None
              and t <= datetime.now(timezone.utc) + timedelta(minutes=5))
    except (TypeError, KeyError, ValueError):
        return None
    return t if ok else None


async def _approve(conn, sess, device_id: str, subject_id, account_id: str, amr: list) -> str:
    """Approval row + token scoped to this device and action; exp = iat + 30 s. Face alone is amr ["face"]."""
    aid, iat = uuid.uuid4(), int(time.time())
    if subject_id is not None:  # ponytail: approvals.subject_id is NOT NULL, so a fallback without a face
        # enrollment is recorded in sessions + audit only (schema change requested in the report)
        await conn.execute("INSERT INTO approvals (id, session_id, subject_id, device_id, action_id, amr, issued_at, "
                           "expires_at) VALUES (%s, %s, %s, %s, %s, %s, to_timestamp(%s), to_timestamp(%s))",
                           (aid, sess["id"], subject_id, device_id, sess["action_id"], amr, iat, iat + APPROVAL_TTL))
    return token.sign(C.cfg.approval_key, {"aid": str(aid), "sid": str(sess["id"]), "dev": device_id,
                                           "act": sess["action_id"], "acct": account_id, "amr": amr,
                                           "iat": iat, "exp": iat + APPROVAL_TTL})


async def _events(conn, op: str, template_ids: list):
    """Outbox rows for search replicas, under the 7001 lock so ids become visible in commit order."""
    await conn.execute("SELECT pg_advisory_xact_lock(7001)")
    await conn.execute("INSERT INTO template_events (op, template_id, subject_id, model_version) "
                       "SELECT %s, id, subject_id, model_version FROM templates WHERE id = ANY(%s) ORDER BY id",
                       (op, template_ids))


# ---------------------------------------------------------------- device endpoints (through the mTLS gateway)

class SessionIn(BaseModel):
    purpose: Literal["enroll", "auth"]
    device_id: str = Field(min_length=1, max_length=64)
    action_id: str = Field(min_length=1, max_length=128)


@app.post("/v1/sessions")
async def create_session(body: SessionIn, request: Request):
    async with C.pool.connection() as conn:
        dev = await _device(request, conn, body.device_id)
        rejects = await _one(conn, "SELECT count(*) AS n FROM sessions WHERE device_id = %s AND purpose = 'auth' "
                                   "AND decision = 'rejected' AND created_at > now() - interval '5 minutes'",
                             (dev["id"],))
        if rejects["n"] >= LOCKOUT_REJECTS:
            raise HTTPException(429, "device_locked")
        sid, challenge = uuid.uuid4(), b64u(secrets.token_bytes(32))
        row = await _one(conn, "INSERT INTO sessions (id, purpose, device_id, action_id, challenge, expires_at) "
                               "VALUES (%s, %s, %s, %s, %s, now() + %s * interval '1 second') RETURNING expires_at",
                         (sid, body.purpose, dev["id"], body.action_id, challenge, TTL[body.purpose]))
    return {"session_id": str(sid), "challenge": challenge, "expires_at": row["expires_at"].isoformat()}


@app.post("/v1/authentications")
async def authenticate(request: Request):
    header, sig, frames, sid = _packet(await request.body())
    async with C.pool.connection() as conn:
        dev = await _device(request, conn, header.get("device_id"))
        sess, why = await _session(conn, sid, dev["id"], "auth", header.get("challenge"))
        if sess is None:
            await audit(conn, _actor(dev), "auth_decision", session_id=sid, device_id=dev["id"],
                        detail={"decision": "rejected", "reason": why})
            return {"decision": "rejected", "reason": "session"}
        active = await _active_recog(conn)
    errs = await asyncio.to_thread(_integrity, header, sig, frames, dev, "auth")
    if why or errs:                                                    # 1. device / packet / session
        return await _auth_final(sess, dev, why or errs[0], {"errors": errs})
    an = await C.deps.analyze("auth", frames)
    _check_model(an, active)
    bad = policy.frame_reasons(an["frames"])
    if bad:                                                            # 2. all modalities, one face, quality
        return await _auth_final(sess, dev, "quality:" + ",".join(bad))
    if not policy.pad_live(an["frames"]):                              # 3. PAD is final
        return await _auth_final(sess, dev, "pad_fail")
    subject, why, top1, top2 = policy.identify(await C.deps.search(active, _embeddings(an)), C.cfg.t_id,
                                               C.cfg.margin)               # 4. threshold AND margin
    scores = {"top1": round(top1, 4), "top2": round(top2, 4)}         # audit only, never returned
    return await _auth_final(sess, dev, why, scores, subject)


async def _auth_final(sess, dev, reason: str, detail=None, subject_id=None):
    """Conditions 5-7 in ONE transaction: subject ledger (FOR SHARE vs a concurrent revoke), action risk,
    session consumption, approval / step-up row, audit last."""
    decision, out = "rejected", {}
    async with C.pool.connection() as conn, conn.transaction():
        account = None
        if subject_id is not None:
            s = await _one(conn, "SELECT account_id, status, consent_version FROM subjects WHERE id = %s FOR SHARE",
                           (subject_id,))
            if s and s["status"] == "active" and s["consent_version"] == C.cfg.consent_version:
                account = s["account_id"]
                decision = "step_up" if sess["action_id"] in C.cfg.high_risk else "approved"
            else:
                reason = "subject_inactive"
        if not await _consume(conn, sess["id"], decision, reason):
            decision, reason = "rejected", "session_replayed"
        elif decision == "approved":
            out["approval_token"] = await _approve(conn, sess, dev["id"], subject_id, account, ["face"])
        elif decision == "step_up":
            stepup_id = uuid.uuid4()
            out["step_up_id"] = str(stepup_id)
            await conn.execute("INSERT INTO stepups (id, session_id, subject_id, device_id, action_id, expires_at) "
                               "VALUES (%s, %s, %s, %s, %s, now() + %s * interval '1 second')",
                               (stepup_id, sess["id"], subject_id, dev["id"], sess["action_id"], STEPUP_TTL))
        await audit(conn, _actor(dev), "auth_decision", subject_id=subject_id, session_id=sess["id"],
                    device_id=dev["id"], detail={"decision": decision, "reason": reason,
                                                 "action_id": sess["action_id"], **(detail or {})})
    return {"decision": decision, **out, "reason": policy.external(reason)}


class AssertionIn(BaseModel):
    assertion: str = Field(min_length=1, max_length=4096)


@app.post("/v1/stepups/{stepup_id}/complete")
async def complete_stepup(stepup_id: uuid.UUID, body: AssertionIn, request: Request):
    async with C.pool.connection() as conn:
        st = await _one(conn, "SELECT * FROM stepups WHERE id = %s", (stepup_id,))
        if st is None:
            return {"decision": "rejected", "reason": "session"}
        dev = await _device(request, conn, st["device_id"])
    a, why = _assertion(body.assertion, str(stepup_id))
    method = _factor(a)
    if why is None and method is None:
        why = "assertion_method"
    out = {"decision": "rejected"}
    async with C.pool.connection() as conn, conn.transaction():
        s = await _one(conn, "SELECT account_id, status, consent_version FROM subjects WHERE id = %s FOR SHARE",
                       (st["subject_id"],))
        if why is None and a["acct"] != s["account_id"]:
            why = "assertion_account_mismatch"
        if why is None and not (s["status"] == "active" and s["consent_version"] == C.cfg.consent_version):
            why = "subject_inactive"
        if why is None and await _one(conn, "UPDATE stepups SET completed_at = now() WHERE id = %s AND "
                                            "completed_at IS NULL AND expires_at > now() RETURNING id",
                                      (stepup_id,)) is None:
            why = "session_stepup_used_or_expired"
        if why is None:
            sess = {"id": st["session_id"], "action_id": st["action_id"]}
            out = {"decision": "approved", "approval_token": await _approve(conn, sess, dev["id"], st["subject_id"],
                                                                            s["account_id"], ["face", method])}
        await audit(conn, _actor(dev), "stepup_decision", subject_id=st["subject_id"], session_id=st["session_id"],
                    device_id=dev["id"], detail={"decision": out["decision"], "reason": why or "ok",
                                                 "method": method, "step_up_id": str(stepup_id)})
    return {**out, "reason": policy.external(why or "ok")}


class FallbackIn(BaseModel):
    session_id: uuid.UUID
    assertion: str = Field(min_length=1, max_length=4096)


@app.post("/v1/fallback")
async def fallback(body: FallbackIn, request: Request):
    """Non-biometric alternative (SEC-01): account-service assertion bound to the session challenge."""
    async with C.pool.connection() as conn:
        owner = await _one(conn, "SELECT device_id FROM sessions WHERE id = %s", (body.session_id,))
        if owner is None:
            return {"decision": "rejected", "reason": "session"}
        dev = await _device(request, conn, owner["device_id"])
        sess, why = await _session(conn, body.session_id, dev["id"], "auth", None)
        if sess is None:
            await audit(conn, _actor(dev), "fallback_decision", session_id=body.session_id, device_id=dev["id"],
                        detail={"decision": "rejected", "reason": why})
            return {"decision": "rejected", "reason": "session"}
    a, why = _assertion(body.assertion, sess["challenge"])
    method = _factor(a)
    if why is None and method is None:
        why = "assertion_method"
    decision, out = ("rejected", {}) if why else ("approved", {})
    async with C.pool.connection() as conn, conn.transaction():
        subject = None if why else await _one(conn, "SELECT id FROM subjects WHERE account_id = %s AND "
                                                    "status <> 'revoked'", (a["acct"],))
        if not await _consume(conn, sess["id"], decision, why or "ok"):
            decision, why = "rejected", "session_replayed"
        elif decision == "approved":
            out["approval_token"] = await _approve(conn, sess, dev["id"], subject and subject["id"], a["acct"],
                                                   [method])
        await audit(conn, _actor(dev), "fallback_decision", subject_id=subject and subject["id"],
                    session_id=sess["id"], device_id=dev["id"],
                    detail={"decision": decision, "reason": why or "ok", "method": method,
                            "action_id": sess["action_id"]})
    return {"decision": decision, **out, "reason": policy.external(why or "ok")}


@app.post("/v1/enrollments")
async def enroll(request: Request):
    header, sig, frames, sid = _packet(await request.body())
    cfg = C.cfg
    async with C.pool.connection() as conn:
        dev = await _device(request, conn, header.get("device_id"))
        sess, why = await _session(conn, sid, dev["id"], "enroll", header.get("challenge"))
        if sess is None:
            await audit(conn, _actor(dev), "enroll_decision", session_id=sid, device_id=dev["id"],
                        detail={"status": "rejected", "reason": why})
            return {"status": "rejected", "coverage": {}, "reason": "session"}
        active = await _active_recog(conn)
        a, why_a = _assertion(header.get("account_assertion"), sess["challenge"])
        existing = a and await _one(conn, "SELECT id, status FROM subjects WHERE account_id = %s AND "
                                          "status <> 'revoked'", (a["acct"],))
    errs = await asyncio.to_thread(_integrity, header, sig, frames, dev, "enroll")
    consent_at = _consent(header.get("consent"))
    why = why or (errs[0] if errs else None) or why_a or (None if consent_at else "consent")
    if why:
        return await _enroll_commit(sess, dev, why, detail={"errors": errs})
    if existing and not (header.get("re_enroll") is True and existing["status"] == "active"):
        return await _enroll_commit(sess, dev, "already_enrolled")
    an = await C.deps.analyze("enroll", frames)
    _check_model(an, active)
    fr = an["frames"]
    dirs, cov = policy.coverage([f.get("pose") if f.get("ok") is True else None for f in fr], _direction_of())
    bad = policy.frame_reasons(fr)
    if bad:
        return await _enroll_commit(sess, dev, "quality:" + ",".join(bad), cov)
    if not policy.pad_live(fr):
        return await _enroll_commit(sess, dev, "pad_fail", cov)
    if sum(cov.values()) < cfg.min_dirs:
        return await _enroll_commit(sess, dev, "coverage", cov)
    embs = _embeddings(an)
    same, low = policy.same_person(embs, cfg.t_same)
    if not same:
        return await _enroll_commit(sess, dev, "same_person", cov, {"min_cos": round(low, 4)})
    own = existing["id"] if existing else None
    dups = [r["subject_id"] for r in await C.deps.search(active, embs) if r["subject_id"] != own
            and r["score"] >= cfg.t_dup]   # never merged: a human reviews every candidate
    subject_id = own or uuid.uuid4()
    # ponytail: if the final tx then loses a race, a new subject's KMS key stays orphaned (no ciphertext uses it)
    cts = await C.deps.encrypt(subject_id, [t.astype("<f4").tobytes() for t in policy.templates(embs, dirs)])
    store = {"subject_id": subject_id, "existing": own is not None, "account": a["acct"], "consent_at": consent_at,
             "status": "pending_review" if dups else "active", "ciphertexts": cts, "model_version": active,
             "align_version": str(an["versions"].get("align") or "unknown"), "duplicates": dups}
    return await _enroll_commit(sess, dev, "review" if dups else "ok", cov,
                                {"min_cos": round(low, 4), "duplicates": len(dups)}, store)


async def _enroll_commit(sess, dev, reason: str, coverage=None, detail=None, store=None):
    """Session consumption + subject/templates/outbox/reviews + audit (last) in ONE transaction."""
    status = store["status"] if store else "rejected"
    subject_id, out = (store or {}).get("subject_id"), {}
    decision = {"active": "enrolled", "pending_review": "pending_review"}.get(status, "rejected")
    async with C.pool.connection() as conn, conn.transaction():
        if not await _consume(conn, sess["id"], decision, reason):
            status, reason, store = "rejected", "session_replayed", None
        elif store:
            try:
                async with conn.transaction():  # savepoint: the live-account unique index / a concurrent revoke
                    out = {"subject_id": str(subject_id), "template_version": await _store(conn, dev, store)}
            except (psycopg.errors.UniqueViolation, _SubjectGone) as e:
                status, out = "rejected", {}
                reason = "already_enrolled" if isinstance(e, psycopg.errors.UniqueViolation) else "subject_changed"
                await conn.execute("UPDATE sessions SET decision = 'rejected', reason = %s WHERE id = %s",
                                   (reason, sess["id"]))
        await audit(conn, _actor(dev), "enroll_decision", subject_id=subject_id if out else None,
                    session_id=sess["id"], device_id=dev["id"],
                    detail={"status": status, "reason": reason, **out, **(detail or {}),
                            "directions": sum((coverage or {}).values())})
    body = {"status": status, **out, "coverage": coverage or {}, "reason": policy.external(reason)}
    return JSONResponse(body, status_code=409 if reason == "already_enrolled" else 200)


async def _store(conn, dev, st) -> int:
    sid = st["subject_id"]
    removed = []
    if st["existing"]:  # re-enroll: replace the templates, template_version += 1
        row = await _one(conn, "UPDATE subjects SET template_version = template_version + 1, status = %s, "
                               "consent_version = %s, consent_at = %s, updated_at = now() "
                               "WHERE id = %s AND status = 'active' RETURNING template_version",
                         (st["status"], C.cfg.consent_version, st["consent_at"], sid))
        if row is None:
            raise _SubjectGone
        version = row["template_version"]
        removed = [r["id"] for r in await (await conn.execute(
            "UPDATE templates SET active = false, ciphertext = NULL WHERE subject_id = %s AND active RETURNING id",
            (sid,))).fetchall()]
    else:
        await conn.execute("INSERT INTO subjects (id, account_id, status, consent_version, consent_at) "
                           "VALUES (%s, %s, %s, %s, %s)",
                           (sid, st["account"], st["status"], C.cfg.consent_version, st["consent_at"]))
        version = 1
    added = []
    for slot, ct in enumerate(st["ciphertexts"]):
        added.append((await _one(conn, "INSERT INTO templates (subject_id, template_version, slot, model_version, "
                                       "align_version, calib_version, ciphertext) VALUES (%s, %s, %s, %s, %s, %s, %s) "
                                       "RETURNING id", (sid, version, slot, st["model_version"],
                                                        st["align_version"], dev["calib_version"], ct)))["id"])
    for cand in st["duplicates"]:
        await conn.execute("INSERT INTO enrollment_reviews (id, subject_id, candidate_subject_id) VALUES (%s, %s, %s)",
                           (uuid.uuid4(), sid, cand))
    if removed:
        await _events(conn, "remove", removed)
    await _events(conn, "add", added)
    return version


# ---------------------------------------------------------------- deletion (service-to-service, admin token)

async def _delete_subject(subject_id: uuid.UUID, actor: str, why: str):
    """Revoke (approvals blocked at commit) + deactivate templates + NULL ciphertexts + remove events in one tx;
    then destroy the KMS key (crypto-shredding); then wait <= 5 s for every search replica to apply the events."""
    async with C.pool.connection() as conn, conn.transaction():
        s = await _one(conn, "SELECT status FROM subjects WHERE id = %s FOR UPDATE", (subject_id,))
        if s is None:
            return None
        await conn.execute("UPDATE subjects SET status = 'revoked', revoked_at = coalesce(revoked_at, now()), "
                           "updated_at = now() WHERE id = %s", (subject_id,))
        ids = [r["id"] for r in await (await conn.execute(
            "UPDATE templates SET active = false, ciphertext = NULL WHERE subject_id = %s AND active RETURNING id",
            (subject_id,))).fetchall()]
        if ids:
            await _events(conn, "remove", ids)
        await audit(conn, actor, "subject_revoked", subject_id=subject_id,
                    detail={"reason": why, "templates_deleted": len(ids), "previous_status": s["status"]})
    kms = await C.deps.destroy_key(subject_id)
    async with C.pool.connection() as conn:
        target = (await _one(conn, "SELECT coalesce(max(id), 0) AS m FROM template_events WHERE subject_id = %s",
                             (subject_id,)))["m"]
    deadline, index_removed = time.monotonic() + 5.0, False
    while not index_removed:
        applied = await C.deps.applied_event_ids()
        index_removed = all(a is not None and a >= target for a in applied)
        if index_removed or time.monotonic() > deadline:
            break
        await asyncio.sleep(0.1)
    result = {"revoked": True, "templates_deleted": len(ids), "index_removed": index_removed,
              "key_destroyed": kms is not None}
    async with C.pool.connection() as conn:
        await audit(conn, actor, "face_deleted", subject_id=subject_id,
                    detail={**result, "kms_destroyed": bool(kms.get("destroyed")) if isinstance(kms, dict) else None,
                            "remove_event_id": target})
    return result


@app.delete("/v1/subjects/{subject_id}/face")
async def delete_face(subject_id: uuid.UUID, request: Request):
    _admin(request)
    result = await _delete_subject(subject_id, "service", "withdrawal")
    if result is None:
        raise HTTPException(404, "not_found")
    return result


# ---------------------------------------------------------------- admin (Bearer admin_token)

class DeviceIn(BaseModel):
    device_id: str = Field(pattern=r"^[A-Za-z0-9._-]{1,64}$")
    public_key_pem: str = Field(max_length=4096)   # Ed25519 packet-signing key (SubjectPublicKeyInfo PEM)
    cert_pem: str = Field(max_length=16384)        # mTLS client certificate issued by the device CA
    calib_version: str = Field(min_length=1, max_length=64)


@app.post("/v1/admin/devices")
async def register_device(body: DeviceIn, request: Request):
    _admin(request)
    try:
        pub = serialization.load_pem_public_key(body.public_key_pem.encode())
        if not isinstance(pub, Ed25519PublicKey):
            raise ValueError("not ed25519")
        raw = pub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        der = x509.load_pem_x509_certificate(body.cert_pem.encode()).public_bytes(serialization.Encoding.DER)
    except ValueError:
        raise HTTPException(400, "invalid_key_or_cert") from None
    fp = hashlib.sha256(der).hexdigest()
    try:
        async with C.pool.connection() as conn, conn.transaction():
            await conn.execute("INSERT INTO devices (id, public_key, cert_fingerprint, calib_version) "
                               "VALUES (%s, %s, %s, %s) ON CONFLICT (id) DO UPDATE SET public_key = EXCLUDED.public_key,"
                               " cert_fingerprint = EXCLUDED.cert_fingerprint, calib_version = EXCLUDED.calib_version,"
                               " status = 'active'", (body.device_id, raw, fp, body.calib_version))
            await audit(conn, "admin", "device_registered", device_id=body.device_id,
                        detail={"cert_fingerprint": fp, "calib_version": body.calib_version})
    except psycopg.errors.UniqueViolation:
        raise HTTPException(409, "certificate_in_use") from None
    return {"device_id": body.device_id, "cert_fingerprint": fp, "status": "active"}


class DeviceStatusIn(BaseModel):
    status: Literal["active", "disabled"]


@app.post("/v1/admin/devices/{device_id}/status")
async def set_device_status(device_id: str, body: DeviceStatusIn, request: Request):
    _admin(request)
    async with C.pool.connection() as conn, conn.transaction():
        if await _one(conn, "UPDATE devices SET status = %s WHERE id = %s RETURNING id",
                      (body.status, device_id)) is None:
            raise HTTPException(404, "not_found")
        await audit(conn, "admin", "device_status", device_id=device_id, detail={"status": body.status})
    return {"device_id": device_id, "status": body.status}


@app.get("/v1/admin/reviews")
async def list_reviews(request: Request, status: Literal["open", "approved", "rejected"] = "open"):
    _admin(request)
    async with C.pool.connection() as conn:
        rows = await (await conn.execute("SELECT * FROM enrollment_reviews WHERE status = %s ORDER BY created_at "
                                         "LIMIT 500", (status,))).fetchall()
    return {"reviews": rows}


class ReviewIn(BaseModel):
    decision: Literal["approve", "reject"]
    by: str = Field(min_length=1, max_length=64)


@app.post("/v1/admin/reviews/{review_id}")
async def decide_review(review_id: uuid.UUID, body: ReviewIn, request: Request):
    """approve: the subject becomes active once no review of it is open. reject: the subject is revoked and its
    face data deleted (same flow as DELETE /v1/subjects/{id}/face). Accounts are never merged."""
    _admin(request)
    status = "approved" if body.decision == "approve" else "rejected"
    async with C.pool.connection() as conn, conn.transaction():
        r = await _one(conn, "UPDATE enrollment_reviews SET status = %s, decided_at = now(), decided_by = %s "
                             "WHERE id = %s AND status = 'open' RETURNING subject_id", (status, body.by, review_id))
        if r is None:
            raise HTTPException(409, "review_not_open")
        if status == "approved":
            await conn.execute("UPDATE subjects SET status = 'active', updated_at = now() WHERE id = %s AND "
                               "status = 'pending_review' AND NOT EXISTS (SELECT 1 FROM enrollment_reviews "
                               "WHERE subject_id = %s AND status = 'open')", (r["subject_id"], r["subject_id"]))
        else:
            await conn.execute("UPDATE enrollment_reviews SET status = 'rejected', decided_at = now(), decided_by = %s "
                               "WHERE subject_id = %s AND status = 'open'", (body.by, r["subject_id"]))
        await audit(conn, "admin", "review_decision", subject_id=r["subject_id"],
                    detail={"review_id": str(review_id), "decision": status, "by": body.by})
    out = {"review_id": str(review_id), "status": status, "subject_id": str(r["subject_id"])}
    if status == "rejected":
        out["deletion"] = await _delete_subject(r["subject_id"], "admin", "review_rejected")
    async with C.pool.connection() as conn:
        out["subject_status"] = (await _one(conn, "SELECT status FROM subjects WHERE id = %s",
                                            (r["subject_id"],)))["status"]
    return out


@app.get("/v1/admin/audit/verify")
async def verify_audit(request: Request):
    _admin(request)
    async with C.pool.connection() as conn:
        r = await _one(conn, "SELECT audit_verify() AS bad, (SELECT count(*) FROM audit_log) AS n")
    return {"ok": r["bad"] is None, "first_bad_id": r["bad"], "rows": r["n"]}


class ActivateIn(BaseModel):
    version: str = Field(min_length=1, max_length=64)


@app.post("/v1/admin/models/{name}/activate")
async def activate_model(name: str, body: ActivateIn, request: Request):
    """Activate a registered version (rollback = activate the previous one), then reload engine replicas
    (they re-verify the signed registry) and, for the recognition model, rebuild the search replicas."""
    _admin(request)
    async with C.pool.connection() as conn, conn.transaction():
        if await _one(conn, "SELECT 1 FROM model_versions WHERE name = %s AND version = %s FOR UPDATE",
                      (name, body.version)) is None:
            raise HTTPException(404, "not_found")
        prev = await _one(conn, "UPDATE model_versions SET active = false WHERE name = %s AND active RETURNING version",
                          (name,))
        await conn.execute("UPDATE model_versions SET active = true, activated_at = now() WHERE name = %s AND "
                           "version = %s", (name, body.version))
        await audit(conn, "admin", "model_activated",
                    detail={"name": name, "version": body.version, "previous": prev and prev["version"]})
    engine = await C.deps.reload(C.cfg.engine_urls)
    search = await C.deps.reload(C.cfg.search_urls) if name == C.cfg.recog_model else []
    return {"name": name, "version": body.version, "previous": prev and prev["version"], "engine_reloaded": engine,
            "search_reloaded": search, "ok": all(engine + search)}


@app.get("/healthz")
async def healthz():
    async with C.pool.connection() as conn:
        await conn.execute("SELECT 1")
    return {"ok": True}
