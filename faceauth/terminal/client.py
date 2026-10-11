"""Simulated terminal's device client: mTLS to the gateway + signed capture packets (docs/IMPLEMENTATION.md §6).

Dev keys (scripts/init_secrets.py): secrets/devices/<id>.{crt,key} (mTLS client cert), <id>.ed25519 (packet
signing key), secrets/ca.crt (gateway trust anchor). The server certificate is always verified against
`server_name` ('gateway'), whatever host the URL uses. Frames stay in memory (never written to disk).
"""
import hashlib
import ssl
import time
from pathlib import Path

import httpx
import msgpack
import numpy as np
from cryptography import x509
from cryptography.hazmat.primitives import serialization

from faceauth.common import token
from faceauth.common.packet import build_packet, encode_frame

SECRETS = Path(__file__).resolve().parents[2] / "secrets"
CALIB_VERSION = "gemini336-sim-v1"


def _device_file(secrets_dir, device_id: str, suffix: str) -> Path:
    return Path(secrets_dir) / "devices" / f"{device_id}{suffix}"


def mtls_context(device_id: str, secrets_dir=SECRETS) -> ssl.SSLContext:
    """Client TLS context: trusts only the dev CA, presents the device cert, verifies the server hostname."""
    ctx = ssl.create_default_context(cafile=str(Path(secrets_dir) / "ca.crt"))
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(_device_file(secrets_dir, device_id, ".crt"), _device_file(secrets_dir, device_id, ".key"))
    return ctx


def cert_fingerprint(device_id: str, secrets_dir=SECRETS) -> str:
    """sha256(DER) of the device's mTLS cert, lowercase hex (= devices.cert_fingerprint)."""
    cert = x509.load_pem_x509_certificate(_device_file(secrets_dir, device_id, ".crt").read_bytes())
    return hashlib.sha256(cert.public_bytes(serialization.Encoding.DER)).hexdigest()


def synced_ts(indices, start_ms: int, fps: float = 30.0, max_skew_ms: int = 3, seed=None) -> list:
    """Timestamps of a hardware-synced RGB-D-IR sensor for frames `indices` (increasing) of a stream that
    started at wall-clock `start_ms`: rgb on the frame clock, depth/IR within ±max_skew_ms of it."""
    rng = np.random.default_rng(seed)
    out = []
    for i in indices:
        t = int(start_ms + round(i * 1000.0 / fps))
        d, r = (int(x) for x in rng.integers(-max_skew_ms, max_skew_ms + 1, size=2))
        out.append({"rgb": t, "depth": t + d, "ir": t + r})
    return out


def make_account_assertion(account_id: str, nonce: str, method: str = "password", ttl_s: int = 60,
                           key_path=SECRETS / "account.key") -> str:
    """Simulated external account-ownership service (sims/tests only): {"acct","method","nonce","iat","exp"}."""
    key = serialization.load_pem_private_key(Path(key_path).read_bytes(), None)
    now = int(time.time())
    return token.sign(key, {"acct": account_id, "method": method, "nonce": nonce, "iat": now, "exp": now + ttl_s})


class DeviceClient:
    """frames = [(rgb uint8 HxWx3 RGB, depth uint16 HxW mm, ir uint8 HxW) or an already encoded
    {"rgb","depth","ir"} bytes dict (tests / capture-path attack helpers)] in capture order;
    ts = [{"rgb","depth","ir"} ms] (see synced_ts). Non-2xx answers raise httpx.HTTPStatusError (fail closed)."""

    def __init__(self, base_url: str, device_id: str, secrets_dir=SECRETS, server_name: str = "gateway",
                 calib_version: str = CALIB_VERSION, timeout: float = 10.0, transport=None):
        self.device_id, self.server_name, self.calib_version = device_id, server_name, calib_version
        self.key = serialization.load_pem_private_key(_device_file(secrets_dir, device_id, ".ed25519").read_bytes(), None)
        self.http = httpx.Client(base_url=base_url, verify=mtls_context(device_id, secrets_dir), timeout=timeout,
                                 transport=transport)  # transport: tests only

    def close(self):
        self.http.close()

    def _post(self, path: str, *, json=None, content: bytes | None = None):
        headers = {"Content-Type": "application/msgpack"} if content is not None else None
        r = self.http.post(path, json=json, content=content, headers=headers,
                           extensions={"sni_hostname": self.server_name})  # SNI + certificate name check
        r.raise_for_status()
        if "msgpack" in r.headers.get("content-type", ""):
            return msgpack.unpackb(r.content, raw=False)
        return r.json()

    def packet(self, session: dict, frames, ts, labels=None, captured_at_ms=None, extra=None) -> bytes:
        return build_packet(self.key, session_id=session["session_id"], challenge=session["challenge"],
                            device_id=self.device_id, calib_version=self.calib_version,
                            frames=[f if isinstance(f, dict) else encode_frame(*f) for f in frames], ts_ms=ts,
                            captured_at_ms=min(ts[0].values()) if captured_at_ms is None else captured_at_ms,
                            labels=labels, extra=extra)

    def create_session(self, purpose: str, action_id: str) -> dict:
        return self._post("/v1/sessions", json={"purpose": purpose, "device_id": self.device_id, "action_id": action_id})

    def enroll(self, session: dict, frames, ts, labels, account_assertion: str, consent: dict, re_enroll: bool = False,
               captured_at_ms=None) -> dict:
        extra = {"account_assertion": account_assertion, "consent": consent, "re_enroll": bool(re_enroll)}
        return self._post("/v1/enrollments", content=self.packet(session, frames, ts, labels, captured_at_ms, extra))

    def authenticate(self, session: dict, frames, ts, captured_at_ms=None) -> dict:
        return self._post("/v1/authentications", content=self.packet(session, frames, ts, None, captured_at_ms))

    def fallback(self, session: dict, assertion: str) -> dict:
        return self._post("/v1/fallback", json={"session_id": session["session_id"], "assertion": assertion})

    def complete_stepup(self, step_up_id: str, assertion: str) -> dict:
        return self._post(f"/v1/stepups/{step_up_id}/complete", json={"assertion": assertion})
