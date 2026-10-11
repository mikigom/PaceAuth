"""Capture-path attack helpers for security tests (RFP SEC-01, Design §7, E11).

Feed the outputs into faceauth.common.packet (encode_frame / build_packet):
  sequence()        n frames of one presentation (live or a PAI) + synchronised timestamps
  multi_face()      several live subjects in one scene (must be rejected: multi_face)
  drop()            encoded frame with modalities removed (FR-03: never fall back to RGB-only)
  desync()          timestamps with one modality skewed beyond packet.MAX_SYNC_SKEW_MS
  rgb_only_replay() pre-recorded RGB injected behind the sensor, depth/IR absent or fabricated
Digital injection of *perfect* synthetic live frames is sequence(sim, "live", ...) pushed into the capture
path behind the sensor: PAD cannot distinguish it from a live capture (reports/E7_pad); only sensor-origin
attestation (frames signed inside the camera module, Design §7) can stop it.
"""
import cv2
import numpy as np

from faceauth.common.packet import MAX_SYNC_SKEW_MS, encode_frame
from faceauth.sim.sensor import H, W


def sequence(sim, kind, face_rgb, kps5, n=5, seed=None, t0_ms=0, fps=30.0, distance_mm=None):
    """One presentation (static scene, fresh sensor noise per frame). -> (list of sim outputs, list of ts_ms)."""
    rng = np.random.default_rng(seed)
    sc = sim.scene(kind, face_rgb, kps5, rng, distance_mm)
    frames = [sim.capture([sc], rng) for _ in range(n)]
    ts = []
    for i in range(n):
        t = t0_ms + round(i * 1000 / fps)
        ts.append({"rgb": t, "depth": t + int(rng.integers(0, 4)), "ir": t + int(rng.integers(0, 4))})
    return frames, ts


def multi_face(sim, faces, seed=None) -> dict:
    """faces: [(face_rgb, kps5), ...] (first = primary, nearest); others stand beside / behind it."""
    rng = np.random.default_rng(seed)
    scenes = []
    for i, (img, kps) in enumerate(faces):
        cx = W / 2 + (0 if i == 0 else (-1) ** i * rng.uniform(150, 220))
        z = rng.uniform(400, 650) if i == 0 else rng.uniform(600, 1100)
        scenes.append(sim.scene("live", img, kps, rng, z, background=(i == 0), center=(cx, H / 2 - 25 + rng.uniform(-30, 30))))
    return sim.capture(scenes, rng)


def drop(frame_enc: dict, *modalities) -> dict:
    """Encoded frame ({"rgb","depth","ir"} bytes) with the given modalities missing."""
    return {m: (b"" if m in modalities else v) for m, v in frame_enc.items()}


def desync(ts_ms: list, modality="depth", skew_ms=MAX_SYNC_SKEW_MS + 25) -> list:
    return [dict(t, **{modality: t[modality] + skew_ms}) for t in ts_ms]


def rgb_only_replay(rgb, fill=None) -> dict:
    """Injected RGB frame. fill=None: depth/IR absent; "zero": all-invalid depth + black IR;
    "luma": all-invalid depth + IR faked from the RGB luminance (a naive forger)."""
    zd = np.zeros(rgb.shape[:2], np.uint16)
    ir = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if fill == "luma" else np.zeros(rgb.shape[:2], np.uint8)
    enc = encode_frame(rgb, zd, ir)
    return drop(enc, "depth", "ir") if fill is None else enc

