"""API integration tests through the real mTLS gateway (RFP FR-01/02/03, SEC-01, OPS-01; spec §6).

Stack: deploy/compose.api.yml profile "apitest" (postgres + faceauth.sql, api-1, api-2, nginx gateway, contract
fakes for engine/search/KMS in tests/test_api_fakes.py). Run from the repo root:
    docker compose -p fa-api -f docker-compose.yml -f deploy/compose.api.yml --profile apitest up -d
    docker compose -p fa-api -f docker-compose.yml -f deploy/compose.api.yml --profile apitest run --rm \
        -u "$(id -u):$(id -g)" -e HOME=/tmp tester python -m pytest -q -p no:cacheprovider tests/test_api_flow.py
Skipped when the stack is not reachable. Frames are JSON scenarios for the fake engine (see test_api_fakes.py);
the API only forwards frame bytes, so its policy is exercised exactly as with real images.
FA_TEST_REAL=1: real db/kms/search (compose.platform.yml + compose.search.yml) with only the fake engine; the
fault-injection tests that need the fakes are skipped.
"""
import datetime
import json
import os
import ssl
import tempfile
import time
import uuid
from pathlib import Path
from urllib.parse import quote

import httpx
import msgpack
import psycopg
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from faceauth.common import token
from faceauth.common.packet import build_packet

S = Path(__file__).resolve().parents[1] / "secrets"
GATEWAY = os.environ.get("FA_TEST_GATEWAY", "https://gateway:8443")
API = os.environ.get("FA_TEST_API", "http://api-1:8000")         # admin / service-to-service, internal network
FAKES = os.environ.get("FA_TEST_FAKES", "http://engine:8001")
SEARCH = os.environ.get("FA_TEST_SEARCH", "http://search-1:8002,http://search-2:8002").split(",")
REAL = os.environ.get("FA_TEST_REAL") == "1"
fakes_only = pytest.mark.skipif(REAL, reason="needs the contract fakes' fault injection")
RECOG = os.environ.get("FA_RECOG_MODEL", "kprpe_vitb_wf12m")
CONSENT = os.environ.get("FA_CONSENT_VERSION", "v1")
HIGH_RISK = "transfer_high"                                       # in FA_HIGH_RISK_ACTIONS (compose default)
CALIB = "gemini336-sim-v1"
POSES = {"C": (0, 0), "L": (25, 0), "R": (-25, 0), "U": (0, 15), "D": (0, -15),
         "UL": (25, 15), "UR": (-25, 15), "DL": (25, -15), "DR": (-25, -15)}
EXTERNAL = {"ok", "retry", "not_recognized", "denied", "session", "coverage", "consent", "already_enrolled", "review"}
FORBIDDEN = {"score", "scores", "candidate", "candidates", "candidate_subject_id", "top1", "top2", "similarity",
             "results", "margin", "pad"}
PEM = serialization.Encoding.PEM
SEEN = []  # (path, status, json) of every answer a device got: no score / candidate / internal reason may leak


def _up():
    try:
        return httpx.get(f"{API}/healthz", timeout=3).status_code == 200
    except httpx.HTTPError:
        return False


pytestmark = pytest.mark.skipif(not _up(), reason="needs the apitest stack (deploy/compose.api.yml)")


def _secret(name):
    return (S / name).read_text().strip()


ADMIN = {"Authorization": f"Bearer {_secret('admin_token')}"} if (S / "admin_token").exists() else {}


def admin(method, path, **kw):
    return httpx.request(method, API + path, headers=ADMIN, timeout=60, **kw)


def fakes(**state):
    assert httpx.post(f"{FAKES}/_test/state", json=state, timeout=5).status_code == 200


def db():
    return psycopg.connect("host=db dbname=faceauth user=faceauth_api", password=_secret("db_api_password"),
                           autocommit=True)


def q1(sql, *params):
    with db() as c:
        return c.execute(sql, params).fetchone()


def qa(sql, *params):
    with db() as c:
        return c.execute(sql, params).fetchall()


def search_status():
    return [httpx.get(f"{u}/v1/status", timeout=5).json() for u in SEARCH]


def wait_search(pred, timeout=120):
    deadline = time.time() + timeout
    while not all(pred(st) for st in search_status()):
        assert time.time() < deadline, search_status()
        time.sleep(0.2)


def wait_index():
    """Replicas apply template_events every 200 ms: wait until every replica has applied all of them."""
    top = q1("SELECT coalesce(max(id), 0) FROM template_events")[0]
    wait_search(lambda st: st["ready"] and st["applied_event_id"] >= top)


def decision_body(body):
    """Ignore the added explicit mode fields while checking legacy decision contracts."""
    if "pad_enabled" in body:
        assert body["pad_enabled"] is True and body["pad_mode"] == "enabled"
    return {k: v for k, v in body.items() if k not in ("pad_enabled", "pad_mode", "enrollment_pad_enabled")}


def reason_of(session):
    return q1("SELECT reason FROM sessions WHERE id = %s", session["session_id"])[0]


class Device:
    """Simulated terminal: mTLS client cert (dev CA) + Ed25519 packet-signing key, talking to the gateway."""

    def __init__(self, device_id, cert_pem: bytes, key_pem: bytes, sign_key):
        self.id, self.cert_pem, self.sign_key = device_id, cert_pem, sign_key
        d = Path(tempfile.mkdtemp(prefix="fa-api-test-"))
        (d / "c.pem").write_bytes(cert_pem)
        (d / "k.pem").write_bytes(key_pem)
        ctx = ssl.create_default_context(cafile=str(S / "ca.crt"))
        ctx.load_cert_chain(d / "c.pem", d / "k.pem")
        self.http = httpx.Client(base_url=GATEWAY, verify=ctx, timeout=60)

    def post(self, path, *, json=None, content=None):
        headers = {"Content-Type": "application/msgpack"} if content is not None else None
        r = self.http.post(path, json=json, content=content, headers=headers)
        try:
            SEEN.append((path, r.status_code, r.json()))
        except ValueError:
            SEEN.append((path, r.status_code, None))
        return r

    def session(self, purpose="auth", action="pay", device_id=None):
        r = self.post("/v1/sessions", json={"purpose": purpose, "device_id": device_id or self.id, "action_id": action})
        assert r.status_code == 200, r.text
        return r.json()

    def packet(self, sess, frames, *, ts=None, key=None, device_id=None, calib=CALIB, extra=None, challenge=None):
        t0 = int(time.time() * 1000)
        ts = ts or [{"rgb": t0 + 33 * i, "depth": t0 + 33 * i + 2, "ir": t0 + 33 * i - 2} for i in range(len(frames))]
        return build_packet(key or self.sign_key, session_id=sess["session_id"],
                            challenge=challenge or sess["challenge"], device_id=device_id or self.id,
                            calib_version=calib, frames=frames, ts_ms=ts, captured_at_ms=ts[0]["rgb"], extra=extra)


def file_device(device_id):
    d = S / "devices"
    key = serialization.load_pem_private_key((d / f"{device_id}.ed25519").read_bytes(), None)
    return Device(device_id, (d / f"{device_id}.crt").read_bytes(), (d / f"{device_id}.key").read_bytes(), key)


def register(dev):
    pub = dev.sign_key.public_key().public_bytes(PEM, serialization.PublicFormat.SubjectPublicKeyInfo).decode()
    r = admin("POST", "/v1/admin/devices", json={"device_id": dev.id, "public_key_pem": pub,
                                                 "cert_pem": dev.cert_pem.decode(), "calib_version": CALIB})
    assert r.status_code == 200, r.text
    return dev


def new_device(register_it=True):
    """Fresh terminal with a cert from the dev CA (each test gets its own, so lockout never leaks across tests)."""
    ca_key = serialization.load_pem_private_key((S / "ca.key").read_bytes(), None)
    ca = x509.load_pem_x509_certificate((S / "ca.crt").read_bytes())
    device_id, key = f"t-{uuid.uuid4().hex[:10]}", ec.generate_private_key(ec.SECP256R1())
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, device_id)]))
            .issuer_name(ca.subject).public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(days=1)).not_valid_after(now + datetime.timedelta(days=7))
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CLIENT_AUTH]), critical=False)
            .sign(ca_key, hashes.SHA256()))
    dev = Device(device_id, cert.public_bytes(PEM), key.private_bytes(PEM, serialization.PrivateFormat.PKCS8,
                                                                     serialization.NoEncryption()),
                 ed25519.Ed25519PrivateKey.generate())
    return register(dev) if register_it else dev


def frame(ident, pose=(0, 0), live=True, faces=1, **kw):
    s = {"id": ident, "pose": [pose[0], pose[1], 0.0], "live": live, "faces": faces, "noise": 0.05, **kw}
    return {"rgb": json.dumps(s).encode(), "depth": b"depth" + os.urandom(16), "ir": b"ir" + os.urandom(16)}


def enroll_frames(ident, dirs=tuple(POSES), per_dir=2):
    return [frame(ident, POSES[d]) for d in dirs for _ in range(per_dir)]


def assertion(acct, nonce, method="password", ttl=60):
    key = serialization.load_pem_private_key((S / "account.key").read_bytes(), None)
    now = int(time.time())
    return token.sign(key, {"acct": acct, "method": method, "nonce": nonce, "iat": now, "exp": now + ttl})


def consent(version=CONSENT):
    return {"version": version, "agreed_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}


def enroll(dev, ident, acct=None, *, frames=None, re_enroll=False, consent_=None):
    acct = acct or f"acct-{uuid.uuid4().hex[:12]}"
    s = dev.session("enroll", "enroll")
    extra = {"account_assertion": assertion(acct, s["challenge"]), "consent": consent_ or consent(),
             "re_enroll": re_enroll}
    r = dev.post("/v1/enrollments", content=dev.packet(s, enroll_frames(ident) if frames is None else frames,
                                                       extra=extra))
    if r.status_code == 200 and r.json()["status"] != "rejected":
        wait_index()
    return r, acct, s


def auth(dev, ident, action="pay", n=3, **kw):
    s = dev.session("auth", action)
    return dev.post("/v1/authentications", content=dev.packet(s, [frame(ident, **kw) for _ in range(n)])), s


def check_approval(tok, dev, action, acct, amr, session):
    p = token.verify(serialization.load_pem_public_key((S / "approval.pub").read_bytes()), tok)
    assert (p["dev"], p["act"], p["acct"], p["amr"], p["sid"]) == (dev.id, action, acct, amr, session["session_id"])
    assert p["exp"] - p["iat"] == 30 and "mfa" not in p["amr"]
    assert q1("SELECT amr FROM approvals WHERE id = %s", p["aid"])[0] == amr
    return p


@pytest.fixture(scope="module", autouse=True)
def stack():
    with db() as c:
        for v in ("test", "test2"):
            c.execute("INSERT INTO model_versions (name, version, sha256) VALUES (%s, %s, 'fake') "
                      "ON CONFLICT DO NOTHING", (RECOG, v))
    r = admin("POST", f"/v1/admin/models/{RECOG}/activate", json={"version": "test"})
    assert r.status_code == 200 and r.json()["ok"], r.text   # also reloads the engine / search
    wait_search(lambda st: st["ready"] and st["model_version"] == f"{RECOG}@test")
    fakes(engine_fail=False, search_fail=False, kms_fail=False, search_frozen=False)
    for i in range(1, 5):
        register(file_device(f"dev-{i:03d}"))
    yield


@pytest.fixture
def dev():
    return new_device()


def uid(prefix):
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


# ------------------------------------------------------------------ gateway / mTLS identity

def test_gateway_requires_client_certificate_and_tls13():
    ctx = ssl.create_default_context(cafile=str(S / "ca.crt"))
    try:
        r = httpx.get(f"{GATEWAY}/healthz", verify=ctx, timeout=10)
        assert r.status_code == 400  # nginx: "No required SSL certificate was sent"
    except httpx.TransportError:
        pass
    ctx12 = ssl.create_default_context(cafile=str(S / "ca.crt"))
    ctx12.maximum_version = ssl.TLSVersion.TLSv1_2
    ctx12.load_cert_chain(S / "devices" / "dev-001.crt", S / "devices" / "dev-001.key")
    with pytest.raises(httpx.TransportError):
        httpx.get(f"{GATEWAY}/healthz", verify=ctx12, timeout=10)


def test_forged_identity_headers_from_internal_peer_are_rejected():
    pem = (S / "devices" / "dev-001.crt").read_text()
    r = httpx.post(f"{API}/v1/sessions", timeout=10, json={"purpose": "auth", "device_id": "dev-001", "action_id": "x"},
                   headers={"X-Client-Verify": "SUCCESS", "X-Client-Cert": quote(pem)})
    assert r.status_code == 401


def test_cert_of_another_device_is_rejected():
    dev1, dev2 = file_device("dev-001"), file_device("dev-002")
    r = dev2.post("/v1/sessions", json={"purpose": "auth", "device_id": "dev-001", "action_id": "pay"})
    assert r.status_code == 403
    s = dev1.session()
    # dev-002's TLS cert carrying a packet that claims (and is signed as) dev-001
    r = dev2.post("/v1/authentications", content=dev1.packet(s, [frame("nobody")]))
    assert r.status_code == 403
    assert q1("SELECT consumed_at FROM sessions WHERE id = %s", s["session_id"])[0] is None


def test_session_is_bound_to_its_device(dev):
    other = new_device()
    ident = uid("bound")
    r, acct, _ = enroll(dev, ident)
    assert r.json()["status"] == "active"
    s = dev.session()
    r = other.post("/v1/authentications", content=other.packet(s, [frame(ident)] * 3))  # B uses A's session
    assert decision_body(r.json()) == {"decision": "rejected", "reason": "session"}
    assert q1("SELECT consumed_at FROM sessions WHERE id = %s", s["session_id"])[0] is None  # B cannot burn it
    r = dev.post("/v1/authentications", content=dev.packet(s, [frame(ident)] * 3))
    assert r.json()["decision"] == "approved"


def test_disabled_device(dev):
    s = dev.session()
    assert admin("POST", f"/v1/admin/devices/{dev.id}/status", json={"status": "disabled"}).status_code == 200
    assert dev.post("/v1/sessions", json={"purpose": "auth", "device_id": dev.id, "action_id": "pay"}).status_code == 403
    assert dev.post("/v1/authentications", content=dev.packet(s, [frame("x")])).status_code == 403
    assert admin("POST", f"/v1/admin/devices/{dev.id}/status", json={"status": "active"}).status_code == 200
    dev.session()
    assert q1("SELECT count(*) FROM audit_log WHERE device_id = %s AND event = 'device_rejected' AND "
              "detail->>'reason' = 'device_disabled'", dev.id)[0] == 2


def test_unregistered_device_certificate():
    ghost = new_device(register_it=False)  # valid CA-issued cert, but not in devices
    assert ghost.post("/v1/sessions", json={"purpose": "auth", "device_id": ghost.id, "action_id": "p"}).status_code == 403


# ------------------------------------------------------------------ enrollment (FR-01)

def test_enroll_happy_path_then_auth_approved():
    dev = file_device("dev-001")
    ident = uid("alice")
    r, acct, s = enroll(dev, ident)
    body = r.json()
    assert r.status_code == 200 and body["status"] == "active" and body["reason"] == "ok", body
    assert body["template_version"] == 1 and all(body["coverage"].values()) and len(body["coverage"]) == 9
    sid = body["subject_id"]
    rows = qa("SELECT slot, active, ciphertext, model_version, calib_version FROM templates WHERE subject_id = %s "
              "ORDER BY slot", sid)
    assert [r_[0] for r_ in rows] == [0, 1, 2, 3] and all(r_[1] for r_ in rows)
    assert all(len(r_[2]) == 12 + 4 * 512 + 16 for r_ in rows)          # AES-GCM(nonce|ct|tag), never plaintext
    assert {r_[3] for r_ in rows} == {f"{RECOG}@test"} and {r_[4] for r_ in rows} == {CALIB}
    assert q1("SELECT count(*) FROM template_events WHERE subject_id = %s AND op = 'add'", sid)[0] == 4
    assert q1("SELECT decision, consumed_at IS NOT NULL FROM sessions WHERE id = %s", s["session_id"]) == \
        ("enrolled", True)
    assert q1("SELECT account_id, consent_version FROM subjects WHERE id = %s", sid) == (acct, CONSENT)

    r, s = auth(dev, ident, action="pay")
    body = r.json()
    assert body["decision"] == "approved" and body["reason"] == "ok", body
    check_approval(body["approval_token"], dev, "pay", acct, ["face"], s)


def test_enroll_incomplete_coverage(dev):
    r, acct, s = enroll(dev, uid("cov"), frames=enroll_frames(uid("cov"), dirs=("C", "L", "R", "U")))
    body = r.json()
    assert body["status"] == "rejected" and body["reason"] == "coverage", body
    assert body["coverage"] == {d: d in ("C", "L", "R", "U") for d in POSES}
    assert q1("SELECT count(*) FROM subjects WHERE account_id = %s", acct)[0] == 0


def test_enroll_does_not_use_pad_as_registration_gate(dev):
    ident = uid("pad")
    frames = enroll_frames(ident)
    frames[5] = frame(ident, POSES["UL"], live=False)
    r, acct, s = enroll(dev, ident, frames=frames)
    assert r.json()["status"] == "active" and r.json()["enrollment_pad_enabled"] is False
    assert reason_of(s) == "ok"
    assert q1("SELECT count(*) FROM subjects WHERE account_id = %s", acct)[0] == 1


def test_enroll_same_person_failure(dev):
    a = uid("sp-a")
    frames = enroll_frames(a)
    frames[0] = frame(uid("sp-b"), POSES["C"])   # one frame of somebody else
    r, acct, s = enroll(dev, a, frames=frames)
    assert r.json()["status"] == "rejected" and r.json()["reason"] == "retry"
    assert reason_of(s) == "same_person"


def test_enroll_quality_failure_rejects_whole_enrollment(dev):
    frames = enroll_frames(uid("q"))
    frames[3] = frame("q", POSES["L"], faces=2)
    r, acct, s = enroll(dev, "q", frames=frames)
    assert r.json()["status"] == "rejected" and r.json()["reason"] == "retry"
    assert reason_of(s) == "quality:multi_face"


def test_enroll_requires_bound_assertion_and_current_consent(dev):
    s = dev.session("enroll", "enroll")
    extra = {"account_assertion": assertion("acct-x", "not-the-challenge"), "consent": consent(), "re_enroll": False}
    r = dev.post("/v1/enrollments", content=dev.packet(s, enroll_frames(uid("as")), extra=extra))
    assert r.json()["reason"] == "denied" and reason_of(s) == "assertion_mismatch"
    r, acct, s = enroll(dev, uid("cs"), consent_=consent("old-consent"))
    assert r.json()["reason"] == "consent" and r.json()["status"] == "rejected"


def test_enroll_duplicate_goes_to_review_then_approve_or_reject(dev):
    a = uid("dup-a")
    r, _, _ = enroll(dev, a)
    first = r.json()["subject_id"]
    r, _, _ = enroll(dev, {a: 1.0, uid("dup-b"): 0.1})        # near-identical face, other account
    body = r.json()
    assert body["status"] == "pending_review" and body["reason"] == "review", body
    second = body["subject_id"]
    reviews = admin("GET", "/v1/admin/reviews").json()["reviews"]
    mine = [x for x in reviews if x["subject_id"] == second]
    assert len(mine) == 1 and mine[0]["candidate_subject_id"] == first   # never merged automatically
    r = admin("POST", f"/v1/admin/reviews/{mine[0]['id']}", json={"decision": "approve", "by": "op-1"})
    assert r.status_code == 200 and r.json()["subject_status"] == "active"
    assert q1("SELECT status FROM subjects WHERE id = %s", first)[0] == "active"
    assert admin("POST", f"/v1/admin/reviews/{mine[0]['id']}", json={"decision": "reject", "by": "x"}).status_code == 409

    r, _, _ = enroll(dev, {a: 1.0, uid("dup-c"): 0.1})
    third = r.json()["subject_id"]
    rid = [x for x in admin("GET", "/v1/admin/reviews").json()["reviews"] if x["subject_id"] == third][0]["id"]
    r = admin("POST", f"/v1/admin/reviews/{rid}", json={"decision": "reject", "by": "op-2"})
    out = r.json()
    assert out["subject_status"] == "revoked" and out["deletion"]["templates_deleted"] == 4
    assert out["deletion"]["key_destroyed"] and out["deletion"]["index_removed"]


def test_already_enrolled_409_then_re_enroll(dev):
    ident = uid("re")
    r, acct, _ = enroll(dev, ident)
    sid = r.json()["subject_id"]
    r, _, s = enroll(dev, ident, acct)
    assert r.status_code == 409 and r.json()["reason"] == "already_enrolled" and "subject_id" not in r.json()
    assert q1("SELECT decision FROM sessions WHERE id = %s", s["session_id"])[0] == "rejected"
    r, _, _ = enroll(dev, ident, acct, re_enroll=True)
    body = r.json()
    assert body["status"] == "active" and body["subject_id"] == sid and body["template_version"] == 2, body
    assert q1("SELECT count(*) FROM templates WHERE subject_id = %s AND active", sid)[0] == 4
    assert q1("SELECT count(*) FROM templates WHERE subject_id = %s AND NOT active AND ciphertext IS NULL "
              "AND template_version = 1", sid)[0] == 4
    assert q1("SELECT count(*) FROM template_events WHERE subject_id = %s AND op = 'remove'", sid)[0] == 4
    r, s = auth(dev, ident)
    assert r.json()["decision"] == "approved"


def test_large_enrollment_packet_passes_gateway(dev):
    frames = enroll_frames(uid("big"))
    for f in frames:
        f["depth"], f["ir"] = os.urandom(700_000), os.urandom(600_000)   # ~23 MB packet, like real RGB-D-IR
    r, _, _ = enroll(dev, "big", frames=frames)
    assert r.status_code == 200, r.text


# ------------------------------------------------------------------ authentication (FR-02, FR-03, SEC-01)

def test_auth_unregistered_person_rejected(dev):
    r, s = auth(dev, uid("stranger"))
    assert decision_body(r.json()) == {"decision": "rejected", "reason": "not_recognized"}
    assert reason_of(s) == "no_match"


def test_auth_ambiguous_candidates_rejected(dev):
    a = uid("amb-a")
    assert enroll(dev, a)[0].json()["status"] == "active"
    assert enroll(dev, {a: 1.0, uid("amb-b"): 0.1})[0].json()["status"] == "pending_review"
    r, s = auth(dev, a)
    assert decision_body(r.json()) == {"decision": "rejected", "reason": "not_recognized"}
    assert reason_of(s) == "ambiguous"   # top-1 was above T_ID but the margin to top-2 was not


def test_auth_multi_face_selects_largest(dev):
    ident = uid("multi")
    _, acct, _ = enroll(dev, ident)
    s = dev.session()
    people = [{"id": uid("bystander"), "bbox": [0, 0, 100, 100]},
              {"id": ident, "bbox": [200, 100, 500, 450]}]
    r = dev.post("/v1/authentications", content=dev.packet(s, [frame(ident, people=people)] * 3))
    assert r.json()["decision"] == "approved", r.text
    check_approval(r.json()["approval_token"], dev, "pay", acct, ["face"], s)


def test_auth_missing_modality_never_falls_back_to_rgb(dev):
    ident = uid("mm")
    enroll(dev, ident)
    s = dev.session()
    frames = [frame(ident) for _ in range(3)]
    frames[1]["ir"] = b""
    r = dev.post("/v1/authentications", content=dev.packet(s, frames))
    assert decision_body(r.json()) == {"decision": "rejected", "reason": "denied"} and reason_of(s) == "missing_modality"


def test_auth_unsynchronised_frames(dev):
    s = dev.session()
    t = int(time.time() * 1000)
    ts = [{"rgb": t, "depth": t + 1, "ir": t}, {"rgb": t + 33, "depth": t + 33 + 50, "ir": t + 33}]
    r = dev.post("/v1/authentications", content=dev.packet(s, [frame("u"), frame("u")], ts=ts))
    assert r.json()["reason"] == "denied" and reason_of(s) == "unsynchronized"


def test_auth_injection_without_device_key(dev):
    ident = uid("inj")
    enroll(dev, ident)
    s = dev.session()
    forged = dev.packet(s, [frame(ident)] * 3, key=ed25519.Ed25519PrivateKey.generate())  # attacker's key
    r = dev.post("/v1/authentications", content=forged)
    assert decision_body(r.json()) == {"decision": "rejected", "reason": "denied"} and reason_of(s) == "bad_signature"


def test_auth_tampered_frame_hash_mismatch(dev):
    ident = uid("tamper")
    enroll(dev, ident)
    s = dev.session()
    p = msgpack.unpackb(dev.packet(s, [frame(ident)] * 3), raw=False)
    p["frames"][1]["rgb"] = frame(ident, seed=7)["rgb"]   # swap one frame's bytes after signing
    r = dev.post("/v1/authentications", content=msgpack.packb(p))
    assert r.json()["reason"] == "denied" and reason_of(s) == "hash_mismatch"


def test_auth_wrong_calibration_version(dev):
    s = dev.session()
    r = dev.post("/v1/authentications", content=dev.packet(s, [frame("c")], calib="gemini336-sim-v0"))
    assert r.json()["reason"] == "denied" and reason_of(s) == "calib_mismatch"


def test_auth_replayed_packet(dev):
    ident = uid("replay")
    enroll(dev, ident)
    s = dev.session()
    pkt = dev.packet(s, [frame(ident)] * 3)
    assert dev.post("/v1/authentications", content=pkt).json()["decision"] == "approved"
    assert decision_body(dev.post("/v1/authentications", content=pkt).json()) == {"decision": "rejected", "reason": "session"}
    assert q1("SELECT count(*) FROM approvals WHERE session_id = %s", s["session_id"])[0] == 1


def test_auth_reused_challenge(dev):
    ident = uid("chal")
    enroll(dev, ident)
    old, new = dev.session(), dev.session()
    r = dev.post("/v1/authentications", content=dev.packet(new, [frame(ident)] * 3, challenge=old["challenge"]))
    assert decision_body(r.json()) == {"decision": "rejected", "reason": "session"} and reason_of(new) == "challenge_mismatch"


def test_auth_expired_session(dev):
    ident = uid("exp")
    enroll(dev, ident)
    s = dev.session()
    with db() as c:
        c.execute("UPDATE sessions SET expires_at = now() - interval '1 second' WHERE id = %s", (s["session_id"],))
    r = dev.post("/v1/authentications", content=dev.packet(s, [frame(ident)] * 3))
    assert decision_body(r.json()) == {"decision": "rejected", "reason": "session"}


@fakes_only
def test_revoked_subject_rejected_even_if_index_still_returns_it(dev):
    ident = uid("rev")
    sid = enroll(dev, ident)[0].json()["subject_id"]
    assert httpx.get(f"{FAKES}/v1/status", timeout=5).status_code == 200   # replica has applied the adds
    fakes(search_frozen=True)   # replica lag from now on: the index keeps returning the subject
    try:
        out = admin("DELETE", f"/v1/subjects/{sid}/face").json()
        assert out["revoked"] and out["key_destroyed"] and out["index_removed"] is False
        r, s = auth(dev, ident)
        assert decision_body(r.json()) == {"decision": "rejected", "reason": "not_recognized"}
        assert reason_of(s) == "subject_inactive"   # identified by the index, refused by the DB ledger
    finally:
        fakes(search_frozen=False)


def test_high_risk_action_requires_step_up(dev):
    ident = uid("risk")
    _, acct, _ = enroll(dev, ident)
    r, s = auth(dev, ident, action=HIGH_RISK)
    body = r.json()
    assert body["decision"] == "step_up" and "approval_token" not in body and body["step_up_id"], body
    stid = body["step_up_id"]
    bad = dev.post(f"/v1/stepups/{stid}/complete", json={"assertion": assertion(acct, "wrong", "otp")})
    assert decision_body(bad.json()) == {"decision": "rejected", "reason": "denied"}
    face_only = dev.post(f"/v1/stepups/{stid}/complete", json={"assertion": assertion(acct, stid, "face")})
    assert face_only.json()["decision"] == "rejected"          # the added factor must not be the face again
    other = dev.post(f"/v1/stepups/{stid}/complete", json={"assertion": assertion("someone-else", stid, "otp")})
    assert other.json()["decision"] == "rejected"
    ok = dev.post(f"/v1/stepups/{stid}/complete", json={"assertion": assertion(acct, stid, "otp")}).json()
    assert ok["decision"] == "approved", ok
    check_approval(ok["approval_token"], dev, HIGH_RISK, acct, ["face", "otp"], s)
    again = dev.post(f"/v1/stepups/{stid}/complete", json={"assertion": assertion(acct, stid, "otp")})
    assert decision_body(again.json()) == {"decision": "rejected", "reason": "session"}


def test_non_biometric_fallback(dev):
    s = dev.session("auth", "pay")
    acct = uid("acct-fb")                               # no face enrollment needed
    r = dev.post("/v1/fallback", json={"session_id": s["session_id"], "assertion": assertion(acct, s["challenge"], "pin")})
    body = r.json()
    assert body["decision"] == "approved", body
    p = token.verify(serialization.load_pem_public_key((S / "approval.pub").read_bytes()), body["approval_token"])
    assert p["amr"] == ["pin"] and p["acct"] == acct and p["dev"] == dev.id and p["exp"] - p["iat"] == 30
    # the approval ledger records non-biometric approvals too (no subject)
    assert q1("SELECT subject_id IS NULL, account_id, amr FROM approvals WHERE id = %s", p["aid"]) == (True, acct, ["pin"])
    again = dev.post("/v1/fallback", json={"session_id": s["session_id"],
                                           "assertion": assertion(acct, s["challenge"], "pin")})
    assert decision_body(again.json()) == {"decision": "rejected", "reason": "session"}
    s2 = dev.session()
    r = dev.post("/v1/fallback", json={"session_id": s2["session_id"], "assertion": assertion(acct, "x", "pin")})
    assert decision_body(r.json()) == {"decision": "rejected", "reason": "denied"}
    ident = uid("fb")                                   # enrolled customer: approval row references the subject
    _, acct2, _ = enroll(dev, ident)
    s3 = dev.session()
    r = dev.post("/v1/fallback", json={"session_id": s3["session_id"], "assertion": assertion(acct2, s3["challenge"], "otp")})
    p = check_approval(r.json()["approval_token"], dev, "pay", acct2, ["otp"], s3)
    assert q1("SELECT s.account_id FROM approvals a JOIN subjects s ON s.id = a.subject_id WHERE a.id = %s",
              p["aid"])[0] == acct2


def test_device_lockout_after_repeated_rejections(dev):
    for _ in range(10):
        r, _ = auth(dev, uid("nobody"))
        assert r.json()["decision"] == "rejected"
    r = dev.post("/v1/sessions", json={"purpose": "auth", "device_id": dev.id, "action_id": "pay"})
    assert r.status_code == 429


# ------------------------------------------------------------------ deletion (OPS-01)

def test_delete_flow(dev):
    ident = uid("del")
    sid = enroll(dev, ident)[0].json()["subject_id"]
    assert httpx.delete(f"{API}/v1/subjects/{sid}/face", headers={"Authorization": "Bearer nope"}).status_code == 401
    out = admin("DELETE", f"/v1/subjects/{sid}/face").json()
    assert out == {"revoked": True, "templates_deleted": 4, "index_removed": True, "key_destroyed": True}
    assert q1("SELECT status, revoked_at IS NOT NULL FROM subjects WHERE id = %s", sid) == ("revoked", True)
    assert q1("SELECT count(*) FROM templates WHERE subject_id = %s AND (active OR ciphertext IS NOT NULL)", sid)[0] == 0
    if not REAL:
        assert httpx.get(f"{FAKES}/_test/kms/{sid}", timeout=5).json() == {"exists": False}
    r, s = auth(dev, ident)
    assert r.json()["decision"] == "rejected"
    again = admin("DELETE", f"/v1/subjects/{sid}/face").json()   # idempotent
    assert again["templates_deleted"] == 0 and again["revoked"] and again["index_removed"]
    assert admin("DELETE", f"/v1/subjects/{uuid.uuid4()}/face").status_code == 404
    # the account may enroll again after withdrawal
    acct = q1("SELECT account_id FROM subjects WHERE id = %s", sid)[0]
    assert enroll(dev, ident, acct)[0].json()["status"] == "active"


# ------------------------------------------------------------------ fail closed + model rollback

@fakes_only
def test_dependency_failures_fail_closed(dev):
    ident = uid("fc")
    sid = enroll(dev, ident)[0].json()["subject_id"]
    try:
        for state in ({"engine_fail": True}, {"search_fail": True}):
            fakes(**state)
            r, s = auth(dev, ident)
            assert r.status_code == 503 and "approval_token" not in r.text, state
            assert q1("SELECT count(*) FROM approvals WHERE session_id = %s", s["session_id"])[0] == 0
            fakes(engine_fail=False, search_fail=False)
        fakes(kms_fail=True)
        r, acct, s = enroll(dev, uid("fc2"))
        assert r.status_code == 503
        assert q1("SELECT count(*) FROM subjects WHERE account_id = %s", acct)[0] == 0
        out = admin("DELETE", f"/v1/subjects/{sid}/face").json()   # revocation still commits; key not yet destroyed
        assert out["revoked"] and out["key_destroyed"] is False
        fakes(kms_fail=False)
        assert admin("DELETE", f"/v1/subjects/{sid}/face").json()["key_destroyed"] is True   # retry completes it
    finally:
        fakes(engine_fail=False, search_fail=False, kms_fail=False)
    assert auth(dev, uid("fc3"))[0].status_code == 200


def test_model_activation_and_rollback(dev):
    ident = uid("model")
    enroll(dev, ident)
    try:
        r = admin("POST", f"/v1/admin/models/{RECOG}/activate", json={"version": "test2"})
        assert r.json()["ok"] and r.json()["previous"] == "test"
        wait_search(lambda st: st["ready"] and st["model_version"] == f"{RECOG}@test2")
        r, s = auth(dev, ident)      # vectors of other versions are never compared: nobody is enrolled for test2
        assert r.json()["decision"] == "rejected" and reason_of(s) == "no_match"
    finally:
        assert admin("POST", f"/v1/admin/models/{RECOG}/activate", json={"version": "test"}).json()["ok"]
        wait_search(lambda st: st["ready"] and st["model_version"] == f"{RECOG}@test")
    assert auth(dev, ident)[0].json()["decision"] == "approved"
    assert admin("POST", f"/v1/admin/models/{RECOG}/activate", json={"version": "nope"}).status_code == 404


# ------------------------------------------------------------------ signed capture diagnostics (explicit measurements, never decisions)

def capture(dev, path, session, frames, *, purpose="auth", **kw):
    # Diagnostics intentionally expose quality/PAD measurements. Keep decision-response secrecy assertions
    # below scoped to decision endpoints; independently assert no embeddings/identity/raw modalities here.
    return dev.http.post("/v1/capture/" + path, headers={"Content-Type": "application/msgpack"},
                         content=dev.packet(session, frames, extra={"capture_purpose": purpose}, **kw))


def test_capture_guide_is_signed_bound_and_does_not_consume(dev):
    s = dev.session("enroll")
    for _ in range(2):
        r = capture(dev, "guide", s, [frame("guide")], purpose="enroll")
        assert r.status_code == 200, r.text
        f = r.json()["frame"]
        assert f["ok"] and f["score"] > 0 and f["pose"] == [0, 0, 0]
        assert not set(f) & {"pad", "emb", "rgb", "depth", "ir", "subject_id"}
        assert q1("SELECT consumed_at FROM sessions WHERE id=%s", s["session_id"])[0] is None
    for kw in ({"calib": "wrong"}, {"key": ed25519.Ed25519PrivateKey.generate()}):
        assert capture(dev, "guide", s, [frame("guide")], purpose="enroll", **kw).status_code == 400
    assert capture(dev, "guide", s, [frame("guide")], purpose="enroll", challenge="wrong").status_code == 409
    assert capture(dev, "guide", s, [frame("guide")], purpose="auth").status_code == 409
    assert capture(dev, "guide", s, [frame("guide")] * 2, purpose="enroll").status_code == 400


def test_signed_auth_packet_without_challenge_is_rejected(dev):
    s = dev.session()
    r = dev.post("/v1/authentications", content=dev.packet(s, [frame("missing-challenge")],
                                                         extra={"challenge": None}))
    assert r.status_code == 200 and r.json()["decision"] == "rejected"
    assert reason_of(s) == "challenge_missing"


def test_capture_observe_is_single_use_and_never_authenticates(dev):
    s = dev.session()
    r = capture(dev, "observe", s, [frame("diagnostic") for _ in range(3)])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["diagnostic"] is True and body["calib_version"] == CALIB and body["model_binding"]
    assert len(body["frames"]) == 3 and body["frames"][0]["pad"]["cnn"] > 0
    assert not {k for k, _ in _walk(body)} & {"emb", "subject_id", "approval_token"}
    assert all(not set(f) & {"rgb", "depth", "ir"} for f in body["frames"])
    assert q1("SELECT decision,reason FROM sessions WHERE id=%s", s["session_id"]) == ("observed", "pad_diagnostic")
    assert capture(dev, "observe", s, [frame("diagnostic") for _ in range(3)]).status_code == 409
    auth_reply = dev.post("/v1/authentications", content=dev.packet(s, [frame("diagnostic")]))
    assert decision_body(auth_reply.json()) == {"decision": "rejected", "reason": "session"}


def test_enrollment_observation_consumes_enroll_session_without_templates(dev):
    s = dev.session("enroll")
    before = q1("SELECT count(*) FROM templates")[0]
    frames = [frame("diagnostic-enroll", POSES[d]) for d in POSES for _ in range(2)]
    assert len(frames) == 18
    assert capture(dev, "observe", s, frames, purpose="auth").status_code == 400  # auth frame limit
    assert capture(dev, "observe", s, frames[:3], purpose="auth").status_code == 409  # session purpose
    assert capture(dev, "observe", s, frames[:2], purpose="enroll").status_code == 400
    assert capture(dev, "observe", s, frames * 2, purpose="enroll").status_code == 400
    assert q1("SELECT consumed_at FROM sessions WHERE id=%s", s["session_id"])[0] is None
    r = capture(dev, "observe", s, frames, purpose="enroll")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["capture_purpose"] == "enroll" and len(body["frames"]) == 18
    assert not {k for k, _ in _walk(body)} & {"emb", "subject_id", "approval_token"}
    assert q1("SELECT count(*) FROM templates")[0] == before
    assert q1("SELECT decision FROM sessions WHERE id=%s", s["session_id"])[0] == "observed"
    assert capture(dev, "observe", s, frames, purpose="enroll").status_code == 409


@fakes_only
def test_capture_readiness_and_api_gate_missing_validation(dev):
    try:
        fakes(pad_ready=False)
        state = dev.http.get("/v1/terminal/readiness", params={"device_id": dev.id})
        assert state.status_code == 200, state.text
        assert not state.json()["ready"] and state.json()["capture_ready"]
        assert state.json()["blockers"][0]["code"] == "pad_real_sensor_validation_missing"
        # The fake still returns perfect per-frame PAD + embeddings: the API must independently gate them.
        a, s = auth(dev, "unvalidated")
        assert a.json()["decision"] == "rejected" and reason_of(s) == "pad_validation_required"
        e, _, s = enroll(dev, "unvalidated")
        assert e.json()["status"] == "active" and reason_of(s) == "ok"
        assert e.json()["enrollment_pad_enabled"] is False
        s = dev.session()
        assert capture(dev, "guide", s, [frame("unvalidated")]).status_code == 200
        observation = capture(dev, "observe", s, [frame("unvalidated") for _ in range(3)])
        assert observation.status_code == 200 and observation.json()["pad_validation"]["ready"] is False
    finally:
        fakes(pad_ready=True)


@fakes_only
def test_observe_failure_consumes_diagnostic_session(dev):
    s = dev.session()
    try:
        fakes(engine_fail=True)
        assert capture(dev, "observe", s, [frame("offline") for _ in range(3)]).status_code == 503
        state = dev.http.get("/v1/terminal/readiness", params={"device_id": dev.id}).json()
        assert state["ready"] is False and state["capture_ready"] is False
    finally:
        fakes(engine_fail=False)
    assert capture(dev, "observe", s, [frame("offline") for _ in range(3)]).status_code == 409


# ------------------------------------------------------------------ run last

def test_zz_audit_chain_verifies():
    r = admin("GET", "/v1/admin/audit/verify")
    assert r.status_code == 200 and r.json()["ok"] is True and r.json()["rows"] > 50, r.text
    with pytest.raises(psycopg.Error):
        with db() as c:
            c.execute("UPDATE audit_log SET actor = 'x' WHERE id = 1")


def _walk(o):
    if isinstance(o, dict):
        for k, v in o.items():
            yield k, v
            yield from _walk(v)
    elif isinstance(o, list):
        for v in o:
            yield None, v
            yield from _walk(v)


def test_zz_no_scores_candidates_or_internal_reasons_leak():
    assert len(SEEN) > 50
    for path, status, body in SEEN:
        items = list(_walk(body))
        assert not {k for k, _ in items} & FORBIDDEN, (path, body)
        assert not any(isinstance(v, float) for _, v in items), (path, body)
        if isinstance(body, dict) and "reason" in body:
            assert body["reason"] in EXTERNAL, (path, body)
