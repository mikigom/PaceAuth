"""DeviceClient end to end through the real mTLS gateway + API, with the api contributor's contract fakes for
engine / search / KMS (frames are JSON scenarios for the fake engine, not images). Skipped without the stack:

  docker compose -p fa-terminal -f docker-compose.yml -f deploy/compose.api.yml --profile apitest up -d
  docker compose -p fa-terminal -f docker-compose.yml -f deploy/compose.api.yml --profile apitest run --rm \
      -u "$(id -u):$(id -g)" -e HOME=/tmp tester python -m pytest -q -p no:cacheprovider tests/test_terminal_e2e.py
  docker compose -p fa-terminal -f docker-compose.yml -f deploy/compose.api.yml --profile apitest down -v
"""
import datetime
import json
import os
import shutil
import time
import uuid

import httpx
import msgpack
import psycopg
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from faceauth.common import token
from faceauth.common.packet import parse_packet
from faceauth.terminal.client import CALIB_VERSION, SECRETS, DeviceClient, make_account_assertion, synced_ts
from faceauth.terminal.guide import ORDER, TARGETS

GATEWAY = os.environ.get("FA_TEST_GATEWAY", "https://gateway:8443")
API = os.environ.get("FA_TEST_API", "http://api-1:8000")  # admin bootstrap only (internal network)
SEARCH = ("http://search-1:8002", "http://search-2:8002")
RECOG = os.environ.get("FA_RECOG_MODEL", "kprpe_vitb_wf12m")
PEM = serialization.Encoding.PEM


def _up():
    try:
        return httpx.get(f"{API}/healthz", timeout=3).status_code == 200
    except httpx.HTTPError:
        return False


pytestmark = pytest.mark.skipif(not _up(), reason="needs the apitest stack (deploy/compose.api.yml)")
ADMIN = {"Authorization": "Bearer " + (SECRETS / "admin_token").read_text().strip()}


def _db():
    return psycopg.connect("host=db dbname=faceauth user=faceauth_api", autocommit=True,
                           password=(SECRETS / "db_api_password").read_text().strip())


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """A fresh terminal (own cert + signing key in a temp secrets dir), registered through the gateway."""
    with _db() as c:
        c.execute("INSERT INTO model_versions (name, version, sha256) VALUES (%s, 'test', 'fake') ON CONFLICT DO NOTHING",
                  (RECOG,))
    assert httpx.post(f"{API}/v1/admin/models/{RECOG}/activate", json={"version": "test"}, headers=ADMIN,
                      timeout=60).json()["ok"]
    deadline = time.time() + 60
    while not all(st["ready"] and st["model_version"] == f"{RECOG}@test"
                  for st in (httpx.get(f"{u}/v1/status", timeout=5).json() for u in SEARCH)):
        assert time.time() < deadline
        time.sleep(0.2)
    sd = tmp_path_factory.mktemp("secrets")
    (sd / "devices").mkdir()
    shutil.copy(SECRETS / "ca.crt", sd / "ca.crt")
    dev_id = f"term-{uuid.uuid4().hex[:8]}"
    ca_key = serialization.load_pem_private_key((SECRETS / "ca.key").read_bytes(), None)
    ca = x509.load_pem_x509_certificate((SECRETS / "ca.crt").read_bytes())
    key, sign = ec.generate_private_key(ec.SECP256R1()), ed25519.Ed25519PrivateKey.generate()
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, dev_id)]))
            .issuer_name(ca.subject).public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(days=1)).not_valid_after(now + datetime.timedelta(days=1))
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CLIENT_AUTH]), critical=False)
            .sign(ca_key, hashes.SHA256()))
    pkcs8 = serialization.PrivateFormat.PKCS8
    (sd / "devices" / f"{dev_id}.crt").write_bytes(cert.public_bytes(PEM))
    (sd / "devices" / f"{dev_id}.key").write_bytes(key.private_bytes(PEM, pkcs8, serialization.NoEncryption()))
    (sd / "devices" / f"{dev_id}.ed25519").write_bytes(sign.private_bytes(PEM, pkcs8, serialization.NoEncryption()))
    c = DeviceClient(GATEWAY, dev_id, secrets_dir=sd)
    pub = sign.public_key().public_bytes(PEM, serialization.PublicFormat.SubjectPublicKeyInfo).decode()
    r = c.http.post("/v1/admin/devices", headers=ADMIN, extensions={"sni_hostname": "gateway"},  # as terminal_demo --register
                    json={"device_id": dev_id, "public_key_pem": pub, "cert_pem": cert.public_bytes(PEM).decode(),
                          "calib_version": CALIB_VERSION})
    assert r.status_code == 200, r.text
    yield c
    c.close()


def scenario(ident, yaw=0.0, pitch=0.0):
    s = {"id": ident, "pose": [yaw, pitch, 0.0], "live": True, "faces": 1, "noise": 0.05, "seed": uuid.uuid4().int % 2**32}
    return {"rgb": json.dumps(s).encode(), "depth": os.urandom(16), "ir": os.urandom(16)}


def now_ms():
    return int(time.time() * 1000)


def wait_index(timeout=60):
    with _db() as c:
        top = c.execute("SELECT coalesce(max(id), 0) FROM template_events").fetchone()[0]
    deadline = time.time() + timeout
    while not all(httpx.get(f"{u}/v1/status", timeout=5).json()["applied_event_id"] >= top for u in SEARCH):
        assert time.time() < deadline
        time.sleep(0.2)


def check_approval(out, client, action, session_id, acct, amr):
    assert out["decision"] == "approved", out
    p = token.verify(serialization.load_pem_public_key((SECRETS / "approval.pub").read_bytes()), out["approval_token"])
    assert (p["dev"], p["act"], p["sid"], p["acct"], p["amr"]) == (client.device_id, action, session_id, acct, amr)


@pytest.fixture(scope="module")
def enrolled(client):
    ident, acct = f"p-{uuid.uuid4().hex[:8]}", f"acct-{uuid.uuid4().hex[:8]}"
    s = client.create_session("enroll", "enroll")
    labels = [d for d in ORDER for _ in range(2)]
    consent = {"version": os.environ.get("FA_CONSENT_VERSION", "v1"),
               "agreed_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    out = client.enroll(s, [scenario(ident, *TARGETS[d]) for d in labels], synced_ts(range(len(labels)), now_ms()),
                        labels, make_account_assertion(acct, s["challenge"]), consent)
    assert out["status"] == "active" and all(out["coverage"].values()), out
    wait_index()
    return ident, acct


def test_authenticate_approved_and_token_scoped(client, enrolled):
    ident, acct = enrolled
    s = client.create_session("auth", "door-1")
    out = client.authenticate(s, [scenario(ident) for _ in range(3)], synced_ts(range(3), now_ms()))
    check_approval(out, client, "door-1", s["session_id"], acct, ["face"])


def test_replayed_packet_rejected(client, enrolled):
    s = client.create_session("auth", "door-1")
    body = client.packet(s, [scenario(enrolled[0]) for _ in range(3)], synced_ts(range(3), now_ms()))
    assert client._post("/v1/authentications", content=body)["decision"] == "approved"
    assert client._post("/v1/authentications", content=body) == {"decision": "rejected", "reason": "session"}


def test_tampered_frame_and_unknown_person_rejected(client, enrolled):
    s = client.create_session("auth", "door-1")
    header, sig, frames = parse_packet(client.packet(s, [scenario(enrolled[0]) for _ in range(3)],
                                                     synced_ts(range(3), now_ms())))
    frames[1]["rgb"] = json.dumps({"id": "someone-else", "pose": [0, 0, 0]}).encode()  # injected after signing
    out = client._post("/v1/authentications", content=msgpack.packb({"header": header, "signature": sig, "frames": frames}))
    assert out == {"decision": "rejected", "reason": "denied"}
    s = client.create_session("auth", "door-1")
    out = client.authenticate(s, [scenario("never-enrolled") for _ in range(3)], synced_ts(range(3), now_ms()))
    assert out == {"decision": "rejected", "reason": "not_recognized"}


def test_step_up_and_fallback(client, enrolled):
    ident, acct = enrolled
    s = client.create_session("auth", "transfer_high")  # high-risk action (FA_HIGH_RISK_ACTIONS)
    out = client.authenticate(s, [scenario(ident) for _ in range(3)], synced_ts(range(3), now_ms()))
    assert out["decision"] == "step_up" and "approval_token" not in out
    done = client.complete_stepup(out["step_up_id"], make_account_assertion(acct, out["step_up_id"], "otp"))
    check_approval(done, client, "transfer_high", s["session_id"], acct, ["face", "otp"])
    s = client.create_session("auth", "door-1")
    out = client.fallback(s, make_account_assertion(acct, s["challenge"], "pin"))
    check_approval(out, client, "door-1", s["session_id"], acct, ["pin"])


def test_gateway_name_is_verified():
    wrong = DeviceClient(GATEWAY, "dev-001", server_name="api")  # the gateway cert names 'gateway', not 'api'
    with pytest.raises(httpx.ConnectError, match="(?i)hostname"):
        wrong.create_session("auth", "door-1")
