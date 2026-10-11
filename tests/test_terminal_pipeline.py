"""Terminal pipeline logic with fake models (real-model smoke test: scripts/terminal_demo.py --smoke)."""
import hashlib
import json
import sys
import types

import numpy as np
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from faceauth.common.registry import RegistryError
from faceauth.terminal.guide import HINTS, Guide
from faceauth.common.packet import decode_frame
from faceauth.terminal.pipeline import Pipeline, eye_aspect_ratio, select_auth, wire_frame


def _lmk(ear=0.3):
    p = np.zeros((98, 2))
    for (a, b), lids, x0 in (((60, 64), ((61, 67), (62, 66), (63, 65)), 200), ((68, 72), ((69, 75), (70, 74), (71, 73)), 280)):
        p[a], p[b] = (x0, 200), (x0 + 40, 200)
        for k, (u, d) in enumerate(lids):
            p[u], p[d] = (x0 + 10 * (k + 1), 200 - 20 * ear), (x0 + 10 * (k + 1), 200 + 20 * ear)
    return p


class FakeDet:
    def __init__(self, boxes):
        self.boxes = boxes

    def detect(self, img, thresh=0.5, max_num=0):
        return [{"bbox": np.array(b, np.float32), "score": 0.9, "kps": np.zeros((5, 2), np.float32)} for b in self.boxes]


class FakePose:
    def __init__(self, pose=(0.0, 0.0, 0.0)):
        self.pose = pose

    def predict(self, img, bboxes):
        return np.array([self.pose] * len(bboxes))


class FakeLmk:
    def __init__(self, ear=0.3):
        self.ear = ear

    def predict(self, img, pts):
        return np.stack([_lmk(self.ear)] * len(pts))


FACE = (200, 100, 400, 340)  # 200x240 px
TEXTURE = np.random.default_rng(0).integers(60, 200, (480, 640, 3), dtype=np.uint8)


def _pipe(boxes=(FACE,), pose=(0.0, 0.0, 0.0), ear=0.3):
    return Pipeline(FakeDet(list(boxes)), FakePose(pose), FakeLmk(ear))


def test_eye_aspect_ratio():
    assert abs(eye_aspect_ratio(_lmk(0.3)) - 0.3) < 1e-9 and abs(eye_aspect_ratio(_lmk(0.05)) - 0.05) < 1e-9
    p = _lmk(0.3)
    p[[69, 70, 71]] = p[[75, 74, 73]]  # one eye closed: the smaller eye counts
    assert eye_aspect_ratio(p) == 0.0


def test_good_frame():
    r = _pipe(pose=(3.0, -2.0, 1.0)).analyze(TEXTURE)
    assert r["ok"] and r["reasons"] == [] and r["n_faces"] == 1 and r["pose"] == [3.0, -2.0, 1.0]
    assert r["quality"]["face_px"] == 200 and r["quality"]["blur"] > 100 and 60 < r["quality"]["brightness"] < 200
    assert 0 < r["score"] <= 0.9


def test_face_count_and_size_rules():
    assert _pipe(boxes=()).analyze(TEXTURE)["reasons"] == ["no_face"]
    assert _pipe(boxes=(FACE, (10, 10, 110, 110))).analyze(TEXTURE)["reasons"] == ["multi_face"]
    assert _pipe(boxes=(FACE, (10, 10, 40, 40))).analyze(TEXTURE)["reasons"] == ["multi_face"]  # 30 px >= 60/2
    r = _pipe(boxes=(FACE, (10, 10, 38, 38))).analyze(TEXTURE)                  # 28 px background face: ignored
    assert r["ok"] and r["n_faces"] == 1
    r = _pipe(boxes=((10, 10, 60, 70),)).analyze(TEXTURE)                       # 50 px only face
    assert r["reasons"] == ["small_face"] and r["pose"] is None


def test_quality_rejections():
    flat = np.full((480, 640, 3), 128, np.uint8)
    assert _pipe().analyze(flat)["reasons"] == ["blur"]
    dark = (TEXTURE // 8).astype(np.uint8)
    assert "exposure" in _pipe().analyze(dark)["reasons"]
    r = _pipe(ear=0.05).analyze(TEXTURE)
    assert r["reasons"] == ["eyes_closed"] and not r["ok"] and r["pose"] is not None


def test_feed_drives_guide_with_reasons():
    g = Guide()
    r, s = _pipe(pose=(25.0, 0.0, 0.0)).feed(g, TEXTURE, "f0", t=0.0)
    assert r["ok"] and s["accepted"] and s["direction"] == "L" and g.frames() == [("L", "f0")]
    r, s = _pipe(boxes=()).feed(g, TEXTURE, "f1", t=0.1)
    assert not s["accepted"] and s["prompt"] == HINTS["no_face"]
    r, s = _pipe(ear=0.05).feed(g, TEXTURE, "f2", t=0.2)
    assert not s["accepted"] and s["prompt"] == HINTS["eyes_closed"]


def test_wire_frame_analyses_the_pixels_that_are_sent():
    depth, ir = np.full((480, 640), 600, np.uint16), np.full((480, 640), 90, np.uint8)
    enc, seen = wire_frame(TEXTURE, depth, ir)
    assert np.array_equal(seen, decode_frame(enc)[0]) and not np.array_equal(seen, TEXTURE)  # JPEG, not raw
    assert np.array_equal(decode_frame(enc)[1], depth) and np.array_equal(decode_frame(enc)[2], ir)


def _res(ok, yaw, score):
    return {"ok": ok, "pose": [yaw, 0.0, 0.0], "score": score}


def test_select_auth_best_three_in_capture_order():
    rs = [_res(True, 0, 0.5), _res(True, 0, 0.9), _res(False, 0, 1.0), _res(True, 30, 1.0), _res(True, 5, 0.7),
          _res(True, -3, 0.8)]
    assert select_auth(rs) == [1, 4, 5]          # 0.9, 0.7, 0.8; not-ok and yaw 30 excluded; capture order
    assert select_auth(rs, n=2) == [1, 5]
    assert select_auth(rs[:3]) == []             # only two usable frames -> capture more


# ---- §7: models are loaded only from a verified, signed registry --------------------------------------------

def _signed_models(tmp_path, names=("scrfd_34g_320", "6drepnet360", "farl_wflw_448"), key=None):
    h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    entries = []
    for n in names:
        (tmp_path / n).mkdir()
        (tmp_path / n / "w.bin").write_bytes(n.encode())
        (tmp_path / n / "MODEL.json").write_text(json.dumps({"name": n, "version": "1", "files": {"w": "w.bin"}}))
        entries.append({"name": n, "version": "1", "dir": n, "files": {"w": "w.bin"}, "primary": "w",
                        "sha256": {"w": h(tmp_path / n / "w.bin")}, "model_json_sha256": h(tmp_path / n / "MODEL.json")})
    raw = json.dumps({"models": entries}).encode()
    signer = Ed25519PrivateKey.generate()
    (tmp_path / "registry.json").write_bytes(raw)
    (tmp_path / "registry.json.sig").write_bytes(signer.sign(raw))
    pub = tmp_path / "release.pub"
    pub.write_bytes((key or signer).public_key().public_bytes(serialization.Encoding.PEM,
                                                              serialization.PublicFormat.SubjectPublicKeyInfo))
    return pub


def _fake_wrappers(monkeypatch):
    for mod, cls in (("scrfd", "SCRFD"), ("headpose", "HeadPose"), ("farl", "FaRLLandmarks")):
        monkeypatch.setitem(sys.modules, f"faceauth.models.{mod}",
                            types.SimpleNamespace(**{cls: lambda d, device=None, c=cls: (c, d.name)}))


def test_load_verifies_signed_registry_before_loading(tmp_path, monkeypatch):
    _fake_wrappers(monkeypatch)
    pub = _signed_models(tmp_path)
    p = Pipeline.load(tmp_path, "cpu", release_pub=pub)
    assert (p.detector, p.headpose, p.landmarks) == (("SCRFD", "scrfd_34g_320"), ("HeadPose", "6drepnet360"),
                                                     ("FaRLLandmarks", "farl_wflw_448"))
    with pytest.raises(RegistryError, match="not in the signed registry"):
        Pipeline.load(tmp_path, "cpu", release_pub=pub, headpose="6drepnet")
    (tmp_path / "6drepnet360" / "w.bin").write_bytes(b"tampered")
    with pytest.raises(RegistryError, match="hash mismatch"):
        Pipeline.load(tmp_path, "cpu", release_pub=pub)
    assert Pipeline.load(tmp_path, "cpu", verify=False).headpose == ("HeadPose", "6drepnet360")  # explicit dev opt-out


def test_load_refuses_unsigned_or_foreign_registry(tmp_path, monkeypatch):
    _fake_wrappers(monkeypatch)
    pub = _signed_models(tmp_path, key=Ed25519PrivateKey.generate())  # signed by someone else
    with pytest.raises(RegistryError, match="signature invalid"):
        Pipeline.load(tmp_path, "cpu", release_pub=pub)
    (tmp_path / "registry.json.sig").unlink()
    with pytest.raises(RegistryError, match="unreadable"):
        Pipeline.load(tmp_path, "cpu", release_pub=pub)


def test_eye_check_spares_narrow_open_eyes_and_downward_gaze():
    assert _pipe(ear=0.09).analyze(TEXTURE)["ok"]                                   # narrow but open eyes pass
    assert "eyes_closed" in _pipe(ear=0.03).analyze(TEXTURE)["reasons"]               # clearly closed eyes do not
    assert _pipe(ear=0.03, pose=(0.0, -15.0, 0.0)).analyze(TEXTURE)["ok"]             # looking down: no eye check
