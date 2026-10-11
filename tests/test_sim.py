"""Sensor simulator + capture-path helpers: determinism, formats, geometry, packet-level attack effects."""
import numpy as np
import pytest
import skimage.data
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from faceauth.common.packet import build_packet, decode_frame, encode_frame, parse_packet, verify_packet
from faceauth.sim import capture
from faceauth.sim.sensor import ATTACKS, H, W, SensorSim

FACE = skimage.data.astronaut()  # public-domain NASA portrait shipped with scikit-image
KPS = np.array([[200, 102], [247, 103], [225, 127], [207, 144], [244, 144]], np.float32)  # hand-labelled


def _depth_at(d, p, r=3):
    x, y = np.round(p).astype(int)
    v = d[y - r:y + r + 1, x - r:x + r + 1]
    return float(np.median(v[v > 0]))


def test_deterministic_and_formats():
    a, b = SensorSim(seed=0).live(FACE, KPS, seed=3), SensorSim(seed=1).live(FACE, KPS, seed=3)
    for k in ("rgb", "depth", "ir", "kps5"):
        assert np.array_equal(a[k], b[k])
    for o in [a] + [SensorSim().attack(k, FACE, KPS, seed=3) for k in ATTACKS]:
        assert o["rgb"].shape == (H, W, 3) and o["rgb"].dtype == np.uint8
        assert o["depth"].shape == (H, W) and o["depth"].dtype == np.uint16
        assert o["ir"].shape == (H, W) and o["ir"].dtype == np.uint8
        assert o["kps5"].shape == (5, 2) and o["calib_version"] == "gemini336-sim-v1"
        assert ((o["kps5"] > 0) & (o["kps5"] < (W, H))).all()
        assert decode_frame(encode_frame(o["rgb"], o["depth"], o["ir"]))[1].dtype == np.uint16


def test_live_has_metric_3d_face():
    o = SensorSim().live(FACE, KPS, seed=5, distance_mm=600)
    eyes = np.mean([_depth_at(o["depth"], p) for p in o["kps5"][:2]])
    assert 560 < eyes < 640  # placed at 600 mm (eye plane)
    assert _depth_at(o["depth"], o["kps5"][2]) < eyes - 15  # nose protrudes towards the camera
    x, y = o["kps5"][2].astype(int)
    assert 0 < o["meta"]["ir_gain"] and np.median(o["ir"][y - 5:y + 5, x - 5:x + 5]) > 60  # lit by the illuminator


def test_print_is_flat_masks_are_not():
    sim = SensorSim()
    for seed in range(3):
        pr = sim.attack("print", FACE, KPS, seed=seed)
        d = [_depth_at(pr["depth"], p) for p in pr["kps5"][:3]]
        assert abs(d[2] - (d[0] + d[1]) / 2) < 6  # flat or gently bent paper
        for kind in ("mask_rigid", "mask_silicone"):
            m = sim.attack(kind, FACE, KPS, seed=seed)
            d = [_depth_at(m["depth"], p, 2) for p in m["kps5"][:3]]
            assert d[2] < (d[0] + d[1]) / 2 - 5, kind  # face-shaped


def test_sensor_b_profile():
    a, b = SensorSim().live(FACE, KPS, seed=2), SensorSim("sensor_b").live(FACE, KPS, seed=2)
    assert b["calib_version"] == "sensor_b-sim-v1" and not np.array_equal(a["ir"], b["ir"])
    with pytest.raises(ValueError):
        SensorSim("nope")


def test_capture_path_helpers():
    sim = SensorSim()
    frames, ts = capture.sequence(sim, "live", FACE, KPS, n=3, seed=0, t0_ms=1000)
    enc = [encode_frame(f["rgb"], f["depth"], f["ir"]) for f in frames]
    key = Ed25519PrivateKey.generate()

    def check(fr, t):
        pkt = build_packet(key, session_id="s", challenge="c", device_id="d", calib_version=sim.calib_version,
                           frames=fr, ts_ms=t, captured_at_ms=1000)
        return verify_packet(*parse_packet(pkt), key.public_key())

    assert check(enc, ts) == []
    assert check(enc, capture.desync(ts)) == ["unsynchronized"]
    assert check([capture.drop(enc[0], "depth")] + enc[1:], ts) == ["missing_modality"]
    with pytest.raises(ValueError):
        decode_frame(capture.rgb_only_replay(frames[0]["rgb"]))
    _, depth, _ = decode_frame(capture.rgb_only_replay(frames[0]["rgb"], "luma"))
    assert not depth.any()
    flip = FACE[:, ::-1].copy()
    kf = np.array([[FACE.shape[1] - 1 - x, y] for x, y in KPS[[1, 0, 2, 4, 3]]], np.float32)
    m = capture.multi_face(sim, [(FACE, KPS), (flip, kf)], seed=0)
    assert len(m["faces"]) == 2 and np.linalg.norm(m["faces"][0].mean(0) - m["faces"][1].mean(0)) > 100
