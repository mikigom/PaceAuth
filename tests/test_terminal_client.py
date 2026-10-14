import datetime
import hashlib
import http.server
import json
import ssl
import threading
import time
from contextlib import contextmanager

import httpx
import msgpack
import numpy as np
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from faceauth.common import token
from faceauth.common.packet import MAX_SYNC_SKEW_MS, decode_frame, parse_packet, verify_packet
from faceauth.terminal.client import (SECRETS, DeviceClient, cert_fingerprint, make_account_assertion,
                                      mtls_context, synced_ts)

DEV = "dev-001"
SESSION = {"session_id": "6f1c0e8e-1111-4c4c-8888-000000000001", "challenge": "Y2hhbGxlbmdlLWJ5dGVz", "expires_at": 0}


def _pub(path):
    return serialization.load_pem_public_key(path.read_bytes())


def _frames(n, seed=0):
    rng = np.random.default_rng(seed)
    return [(np.full((48, 64, 3), (200, 100, 30 + i), np.uint8), rng.integers(300, 900, (48, 64), dtype=np.uint16),
             rng.integers(0, 256, (48, 64), dtype=np.uint8)) for i in range(n)]


def _mock_client(captured, reply=None, status=200, content_type="application/json"):
    def handler(request):
        captured.append(request)
        body = reply if reply is not None else {"ok": True}
        data = msgpack.packb(body) if content_type == "application/msgpack" else json.dumps(body).encode()
        return httpx.Response(status, content=data, headers={"content-type": content_type})
    return DeviceClient("https://gateway:8443", DEV, transport=httpx.MockTransport(handler))


def test_auth_packet_verifies_with_device_key():
    captured = []
    c = _mock_client(captured, reply={"decision": "rejected", "reason": "retry"}, content_type="application/msgpack")
    frames, ts = _frames(3), synced_ts([2, 5, 9], start_ms=1_760_000_000_000, seed=1)
    out = c.authenticate(SESSION, frames, ts)
    assert out == {"decision": "rejected", "reason": "retry"}
    req = captured[0]
    assert req.url.path == "/v1/authentications" and req.headers["content-type"] == "application/msgpack"
    assert req.extensions["sni_hostname"] == "gateway"
    header, sig, wire = parse_packet(req.content)
    assert verify_packet(header, sig, wire, _pub(SECRETS / "devices" / f"{DEV}.ed25519.pub")) == []
    assert verify_packet(header, sig, wire, _pub(SECRETS / "devices" / "dev-002.ed25519.pub")) == ["bad_signature"]
    assert (header["session_id"], header["challenge"], header["device_id"], header["calib_version"]) == \
        (SESSION["session_id"], SESSION["challenge"], DEV, "gemini336-sim-v1")
    assert header["captured_at_ms"] == min(ts[0].values())
    assert [m["ts_ms"] for m in header["frames"]] == ts and [m["seq"] for m in header["frames"]] == [0, 1, 2]
    for (rgb, depth, ir), f in zip(frames, wire):
        rgb2, depth2, ir2 = decode_frame(f)
        assert np.array_equal(depth, depth2) and np.array_equal(ir, ir2)  # lossless PNG
        assert np.abs(rgb.mean((0, 1)) - rgb2.mean((0, 1))).max() < 3  # JPEG, RGB channel order kept
    # any change to a signed field or frame bytes is detected
    tampered = dict(header, challenge="other")
    assert verify_packet(tampered, sig, wire, _pub(SECRETS / "devices" / f"{DEV}.ed25519.pub")) == ["bad_signature"]
    wire[1] = dict(wire[1], depth=wire[0]["depth"])
    assert verify_packet(header, sig, wire, _pub(SECRETS / "devices" / f"{DEV}.ed25519.pub")) == ["hash_mismatch"]


def test_enroll_packet_carries_bound_assertion_consent_and_labels():
    captured = []
    c = _mock_client(captured, reply={"status": "active", "subject_id": "x", "coverage": {}, "reason": "ok"})
    labels = ["C", "L", "DR"]
    assertion = make_account_assertion("acct-42", SESSION["challenge"], "password")
    consent = {"version": "v1", "agreed_at": "2026-10-08T00:00:00+00:00"}
    out = c.enroll(SESSION, _frames(3), synced_ts(range(3), 1_760_000_000_000), labels, assertion, consent, re_enroll=True)
    assert out["status"] == "active"
    req = captured[0]
    assert req.url.path == "/v1/enrollments"
    header, sig, wire = parse_packet(req.content)
    assert verify_packet(header, sig, wire, _pub(SECRETS / "devices" / f"{DEV}.ed25519.pub")) == []
    assert [m["label"] for m in header["frames"]] == labels
    assert header["consent"] == consent and header["re_enroll"] is True
    payload = token.verify(_pub(SECRETS / "account.pub"), header["account_assertion"])
    assert payload["acct"] == "acct-42" and payload["nonce"] == SESSION["challenge"] and payload["method"] == "password"
    assert payload["exp"] - payload["iat"] == 60


def test_pre_encoded_frames_are_signed_as_is():
    captured = []
    c = _mock_client(captured)
    enc = {"rgb": b"scenario", "depth": b"d", "ir": b"i"}
    c.authenticate(SESSION, [enc, _frames(1)[0]], synced_ts([0, 1], 1_760_000_000_000))
    header, sig, wire = parse_packet(captured[0].content)
    assert wire[0] == enc and verify_packet(header, sig, wire, _pub(SECRETS / "devices" / f"{DEV}.ed25519.pub")) == []


def test_account_assertion_signature_and_expiry():
    t = make_account_assertion("acct-1", "nonce-1", "otp", ttl_s=30)
    assert token.verify(_pub(SECRETS / "account.pub"), t)["method"] == "otp"
    with pytest.raises(ValueError):
        token.verify(_pub(SECRETS / "approval.pub"), t)  # other key
    with pytest.raises(ValueError, match="expired"):
        token.verify(_pub(SECRETS / "account.pub"), t, now=time.time() + 31)


def test_json_endpoints_and_fail_closed_on_http_errors():
    captured = []
    c = _mock_client(captured, reply={"session_id": "s", "challenge": "c", "expires_at": "t"})
    assert c.create_session("auth", "door-1")["challenge"] == "c"
    assert json.loads(captured[-1].content) == {"purpose": "auth", "device_id": DEV, "action_id": "door-1"}
    c.fallback(SESSION, "tok")
    assert captured[-1].url.path == "/v1/fallback"
    assert json.loads(captured[-1].content) == {"session_id": SESSION["session_id"], "assertion": "tok"}
    c.complete_stepup("st-1", "tok2")
    assert captured[-1].url.path == "/v1/stepups/st-1/complete" and json.loads(captured[-1].content) == {"assertion": "tok2"}
    for status in (401, 409, 429, 503):
        bad = _mock_client([], reply={"decision": "approved"}, status=status)  # body must never be trusted
        with pytest.raises(httpx.HTTPStatusError):
            bad.authenticate(SESSION, _frames(1), synced_ts([0], 1_760_000_000_000))


def test_synced_ts_is_ordered_and_within_sync_tolerance():
    ts = synced_ts(range(0, 300, 7), start_ms=1_000, fps=30, max_skew_ms=3, seed=0)
    rgb = [t["rgb"] for t in ts]
    assert all(b > a for a, b in zip(rgb, rgb[1:])) and rgb[0] == 1_000 and rgb[1] == 1_000 + round(7000 / 30)
    assert all(abs(t["depth"] - t["rgb"]) <= 3 and abs(t["ir"] - t["rgb"]) <= 3 for t in ts)
    assert 3 < MAX_SYNC_SKEW_MS
    assert synced_ts([0, 1], 0, seed=5) == synced_ts([0, 1], 0, seed=5)


def test_capture_client_binds_purpose_calibration_and_gateway_identity():
    requests = []
    c = _mock_client(requests)
    c.calib_version = "femtobolt-test-v1"
    c.readiness()
    assert requests[-1].method == "GET" and requests[-1].url.params["device_id"] == DEV
    assert requests[-1].extensions["sni_hostname"] == "gateway"
    frames, ts = _frames(3), synced_ts(range(3), 1_760_000_000_000)
    c.capture_guide(SESSION, frames[0], ts[0], purpose="enroll")
    c.observe(SESSION, frames, ts)
    for request, purpose, count in zip(requests[1:], ("enroll", "auth"), (1, 3)):
        header, sig, wire = parse_packet(request.content)
        assert header["capture_purpose"] == purpose and header["calib_version"] == c.calib_version
        assert len(wire) == count and verify_packet(header, sig, wire, c.key.public_key()) == []
    with pytest.raises(ValueError):
        c.observe(SESSION, frames[:2], ts[:2])
    c.observe(SESSION, frames * 6, synced_ts(range(18), 1_760_000_000_000), purpose="enroll")
    header, sig, wire = parse_packet(requests[-1].content)
    assert header["capture_purpose"] == "enroll" and len(wire) == 18
    assert verify_packet(header, sig, wire, c.key.public_key()) == []
    with pytest.raises(ValueError):
        c.observe(SESSION, frames * 6, [], purpose="auth")
    with pytest.raises(ValueError):
        c.observe(SESSION, frames * 9, [], purpose="enroll")
    c.close()


def test_final_submission_waits_for_gateway_without_slowing_guidance_or_readiness():
    requests = []
    c = _mock_client(requests)
    frames, ts = _frames(3), synced_ts(range(3), 1_760_000_000_000)
    c.readiness()
    c.capture_guide(SESSION, frames[0], ts[0])
    c.enroll(SESSION, frames, ts, ["C"] * 3, "assertion", {"version": "v1"})
    c.authenticate(SESSION, frames, ts)
    c.observe(SESSION, frames, ts)
    assert [r.extensions["timeout"]["read"] for r in requests] == [10, 10, 95, 95, 95]
    for request in requests[2:]:
        assert request.extensions["timeout"] == {"connect": 2, "read": 95, "write": 30, "pool": 10}
    c.close()


def test_live_recognition_signs_one_auth_frame_with_device_and_challenge():
    requests = []
    c = _mock_client(requests, reply={"demo": True, "faces": []})
    c.create_session("auth", "live-demo")
    assert json.loads(requests[-1].content)["action_id"] == "live-demo"
    out = c.recognize_live(SESSION, _frames(1)[0], synced_ts([0], 1_760_000_000_000)[0])
    request = requests[-1]
    header, sig, frames = parse_packet(request.content)
    assert request.url.path == "/v1/live/recognize" and out["demo"] is True
    assert header["capture_purpose"] == "auth" and header["challenge"] == SESSION["challenge"]
    assert len(frames) == 1 and verify_packet(header, sig, frames, c.key.public_key()) == []
    assert request.extensions["sni_hostname"] == "gateway" and request.extensions["timeout"]["read"] == 10
    c.close()


# ---- real TLS handshakes against a local server presenting the gateway certificate ----------------------------

class _Echo(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        der = self.connection.getpeercert(binary_form=True)
        body = json.dumps({"session_id": "s", "challenge": "c", "expires_at": 0,
                           "client_fp": hashlib.sha256(der).hexdigest() if der else None}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


@contextmanager
def _serve(certfile=SECRETS / "gateway.crt", keyfile=SECRETS / "gateway.key"):
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile, keyfile)
    ctx.load_verify_locations(SECRETS / "ca.crt")
    ctx.verify_mode = ssl.CERT_REQUIRED  # like the nginx gateway: mTLS required
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Echo)
    srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"https://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        srv.server_close()


def test_mtls_context_loads():
    ctx = mtls_context(DEV)
    assert ctx.verify_mode == ssl.CERT_REQUIRED and ctx.check_hostname
    assert len(ctx.get_ca_certs()) == 1 and ctx.minimum_version >= ssl.TLSVersion.TLSv1_2
    assert len(cert_fingerprint(DEV)) == 64 and cert_fingerprint(DEV) != cert_fingerprint("dev-002")


def test_mtls_handshake_presents_device_cert_and_checks_gateway_name():
    with _serve() as url:
        c = DeviceClient(url, DEV, timeout=5)  # URL host is an IP; the cert is still checked against 'gateway'
        assert c.create_session("auth", "a")["client_fp"] == cert_fingerprint(DEV)
        wrong = DeviceClient(url, DEV, server_name="api", timeout=5)
        with pytest.raises(httpx.ConnectError, match="(?i)hostname|certificate"):
            wrong.create_session("auth", "a")


def _self_signed(tmp_path):
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "gateway")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=1))
            .add_extension(x509.SubjectAlternativeName([x509.DNSName("gateway")]), critical=False)
            .sign(key, hashes.SHA256()))
    (tmp_path / "rogue.crt").write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    (tmp_path / "rogue.key").write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                           serialization.NoEncryption()))
    return tmp_path / "rogue.crt", tmp_path / "rogue.key"


def test_mtls_rejects_server_not_signed_by_dev_ca(tmp_path):
    with _serve(*_self_signed(tmp_path)) as url:
        with pytest.raises(httpx.ConnectError, match="(?i)certificate"):
            DeviceClient(url, DEV, timeout=5).create_session("auth", "a")


def test_account_management_uses_server_credential_over_verified_mtls(tmp_path):
    captured = []
    client = _mock_client(captured, reply={"accounts": []})
    client.secrets_dir = tmp_path
    (tmp_path / "admin_token").write_text("fixture-management-credential")
    client.account_summary(["alice"])
    client.account_status("alice", False)
    client.delete_account_face("alice")
    assert [r.url.path for r in captured] == [
        "/v1/admin/accounts/summary", "/v1/admin/accounts/alice/status", "/v1/admin/accounts/alice/face"]
    assert [r.method for r in captured] == ["POST", "POST", "DELETE"]
    assert json.loads(captured[1].content) == {"enabled": False}
    for request in captured:
        assert request.headers["authorization"] == "Bearer fixture-management-credential"
        assert request.extensions["sni_hostname"] == "gateway"
    client.close()
