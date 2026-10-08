"""Capture packet: the signed unit a terminal sends for enrollment / authentication.

Wire format (Content-Type: application/msgpack):
    {"header": {...}, "signature": <Ed25519 over canonical(header)>,
     "frames": [{"rgb": <JPEG>, "depth": <PNG uint16 mm, 0=invalid>, "ir": <PNG uint8>}, ...]}

header:
    session_id, challenge (b64url, from POST /v1/sessions), device_id, calib_version,
    captured_at_ms (wall clock of the first frame; latency is measured from here),
    frames: [{"seq", "ts_ms": {"rgb","depth","ir"}, "sha256": {"rgb","depth","ir"}, "label"}],
    + endpoint fields (enrollment: account_assertion, consent, re_enroll).

The signature binds frame bytes (via sha256), order, timestamps, session, challenge, device and
calibration together, so swapping/reordering/re-using any part invalidates the packet.
"""
import base64
import hashlib
import json

import cv2
import msgpack
import numpy as np
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

MODALITIES = ("rgb", "depth", "ir")
MAX_SYNC_SKEW_MS = 15  # max |ts(rgb) - ts(depth|ir)| within one frame (hardware-synced sensor)


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def unb64u(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def encode_frame(rgb: np.ndarray, depth: np.ndarray, ir: np.ndarray, jpeg_quality: int = 95) -> dict:
    """rgb HxWx3 uint8 (RGB order), depth HxW uint16 mm, ir HxW uint8 -> encoded bytes per modality."""
    out = {}
    ok, b = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    out["rgb"] = b.tobytes()
    ok2, b = cv2.imencode(".png", np.ascontiguousarray(depth, dtype=np.uint16))
    out["depth"] = b.tobytes()
    ok3, b = cv2.imencode(".png", np.ascontiguousarray(ir, dtype=np.uint8))
    out["ir"] = b.tobytes()
    if not (ok and ok2 and ok3):
        raise ValueError("frame encode failed")
    return out


def decode_frame(f: dict):
    """-> (rgb uint8 HxWx3 RGB, depth uint16 HxW, ir uint8 HxW). Raises ValueError on missing/corrupt data."""
    missing = [m for m in MODALITIES if not f.get(m)]
    if missing:
        raise ValueError("missing_" + "_".join(missing))
    rgb = cv2.imdecode(np.frombuffer(f["rgb"], np.uint8), cv2.IMREAD_COLOR)
    depth = cv2.imdecode(np.frombuffer(f["depth"], np.uint8), cv2.IMREAD_UNCHANGED)
    ir = cv2.imdecode(np.frombuffer(f["ir"], np.uint8), cv2.IMREAD_UNCHANGED)
    if rgb is None or depth is None or ir is None:
        raise ValueError("corrupt_frame")
    if depth.dtype != np.uint16 or depth.ndim != 2 or ir.dtype != np.uint8 or ir.ndim != 2:
        raise ValueError("bad_frame_format")
    if not (rgb.shape[:2] == depth.shape == ir.shape):
        raise ValueError("unregistered_frame_size")
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), depth, ir


def build_packet(key: Ed25519PrivateKey, *, session_id: str, challenge: str, device_id: str, calib_version: str,
                 frames: list, ts_ms: list, captured_at_ms: int, labels=None, extra=None) -> bytes:
    """frames: list of encode_frame() dicts; ts_ms: list of {"rgb","depth","ir"} ints."""
    header = {
        "session_id": session_id, "challenge": challenge, "device_id": device_id,
        "calib_version": calib_version, "captured_at_ms": int(captured_at_ms),
        "frames": [{"seq": i, "ts_ms": dict(ts_ms[i]),
                    "sha256": {m: hashlib.sha256(f[m]).hexdigest() for m in MODALITIES},
                    "label": labels[i] if labels else None} for i, f in enumerate(frames)],
        **(extra or {}),
    }
    return msgpack.packb({"header": header, "signature": key.sign(canonical(header)), "frames": frames})


def parse_packet(body: bytes):
    """Structural parse only -> (header, signature, frames). Raises ValueError."""
    try:
        p = msgpack.unpackb(body, raw=False)
        header, sig, frames = p["header"], p["signature"], p["frames"]
    except Exception as e:  # malformed client input
        raise ValueError("malformed_packet") from e
    if not isinstance(header, dict) or not isinstance(sig, bytes) or not isinstance(frames, list):
        raise ValueError("malformed_packet")
    return header, sig, frames


def verify_packet(header: dict, signature: bytes, frames: list, public_key: Ed25519PublicKey,
                  max_skew_ms: int = MAX_SYNC_SKEW_MS) -> list:
    """Integrity checks independent of server state. Returns a list of error codes ([] = ok)."""
    try:
        public_key.verify(signature, canonical(header))
    except InvalidSignature:
        return ["bad_signature"]
    meta = header.get("frames") or []
    if not frames or len(meta) != len(frames):
        return ["frame_count_mismatch"]
    errs, prev_seq, prev_ts = [], -1, None
    for m, f in zip(meta, frames):
        if any(not f.get(k) for k in MODALITIES):
            errs.append("missing_modality")  # FR-03: never fall back to RGB-only
            continue
        if any(hashlib.sha256(f[k]).hexdigest() != m["sha256"].get(k) for k in MODALITIES):
            errs.append("hash_mismatch")
        ts = m.get("ts_ms") or {}
        if any(k not in ts for k in MODALITIES):
            errs.append("missing_timestamp")
            continue
        if max(abs(ts["rgb"] - ts["depth"]), abs(ts["rgb"] - ts["ir"])) > max_skew_ms:
            errs.append("unsynchronized")
        if m["seq"] <= prev_seq or (prev_ts is not None and ts["rgb"] <= prev_ts):
            errs.append("bad_frame_order")
        prev_seq, prev_ts = m["seq"], ts["rgb"]
    return sorted(set(errs))
