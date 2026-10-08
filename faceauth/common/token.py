"""Compact Ed25519-signed tokens: "v1.<b64u(canonical json payload)>.<b64u(signature)>".

Used for approval tokens (issued by api), account-ownership / step-up assertions (issued by the
account service, simulated here) and nothing else. `exp`/`iat` are unix seconds.
"""
import json
import time

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from faceauth.common.packet import b64u, canonical, unb64u


def sign(key: Ed25519PrivateKey, payload: dict) -> str:
    body = canonical(payload)
    return f"v1.{b64u(body)}.{b64u(key.sign(body))}"


def verify(pub: Ed25519PublicKey, token: str, now: float | None = None) -> dict:
    """-> payload; raises ValueError if malformed, badly signed or expired."""
    try:
        v, body, sig = token.split(".")
        if v != "v1":
            raise ValueError
        body = unb64u(body)
        pub.verify(unb64u(sig), body)
        payload = json.loads(body)
    except (ValueError, InvalidSignature, json.JSONDecodeError) as e:
        raise ValueError("invalid_token") from e
    if not isinstance(payload, dict) or payload.get("exp", 0) < (now or time.time()):
        raise ValueError("expired_token")
    return payload
