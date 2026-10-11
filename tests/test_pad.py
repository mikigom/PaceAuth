"""PAD wrapper API, physics checks and (if trained) the shipped model on simulator frames."""
import json
from pathlib import Path

import numpy as np
import pytest
import skimage.data
import torch

from faceauth.models.pad import PAD, PADNet, face_features
from faceauth.sim.sensor import SensorSim

FACE = skimage.data.astronaut()
KPS = np.array([[200, 102], [247, 103], [225, 127], [207, 144], [244, 144]], np.float32)
MODEL = Path(__file__).resolve().parents[1] / "data/models/pad_rgbdir"
CHECKS = dict(min_valid_ratio=0.5, min_relief_mm=3.0, min_ir_median=30.0, min_shading_corr=0.1)


def _model_dir(d, threshold):
    torch.save(PADNet().state_dict(), d / "w.pt")
    (d / "MODEL.json").write_text(json.dumps(dict(
        name="pad_test", version="0", kind="pad", files={"weights": "w.pt"}, sha256={},
        params=dict(threshold=threshold, checks=CHECKS, fx=385.0, input=dict(size=128, zoom=1.6)))))
    return d


def test_physics_separates_live_and_print():
    sim = SensorSim()
    for seed in range(3):
        lv, pr = sim.live(FACE, KPS, seed=seed), sim.attack("print", FACE, KPS, seed=seed)
        cl = face_features(lv["rgb"], lv["depth"], lv["ir"], lv["kps5"])[1]
        cp = face_features(pr["rgb"], pr["depth"], pr["ir"], pr["kps5"])[1]
        assert cl["valid_ratio"] > 0.9 and cl["shading_corr"] > 0.3
        assert cl["relief_mm"] > CHECKS["min_relief_mm"] > cp["relief_mm"]


def test_pad_api(tmp_path):
    pad = PAD(_model_dir(tmp_path, 0.0), device="cpu")
    assert pad.version == "pad_test@0"
    o = SensorSim().live(FACE, KPS, seed=1)
    s = pad.score(o["rgb"], o["depth"], o["ir"], o["kps5"])
    assert s.shape == (1,) and 0 < s[0] <= 1  # checks pass -> gated score = CNN probability
    assert pad.score(o["rgb"], o["depth"], o["ir"], np.stack([o["kps5"]] * 2)).shape == (2,)
    assert pad.score([o["rgb"]] * 3, [o["depth"]] * 3, [o["ir"]] * 3, [o["kps5"]] * 3).shape == (3,)
    assert pad.score(o["rgb"], o["depth"], o["ir"], np.zeros((0, 5, 2))).shape == (0,)
    pr = SensorSim().attack("print", FACE, KPS, seed=1)
    d = pad.predict(pr["rgb"], pr["depth"], pr["ir"], pr["kps5"])[0]
    assert "relief_mm" in d["failed"] and d["score"] == 0.0 and not d["live"]
    nr = pad.predict(o["rgb"], np.zeros_like(o["depth"]), o["ir"], o["kps5"])[0]
    assert nr["nonresponse"] and not nr["live"] and nr["score"] == 0.0


@pytest.mark.skipif(not (MODEL / "MODEL.json").exists(), reason="PAD model not trained yet")
def test_trained_model_on_sim_frames():
    pad = PAD(MODEL, device="cpu")
    sim = SensorSim(seed=123)
    live = [sim.live(FACE, KPS) for _ in range(6)]
    att = [sim.attack(k, FACE, KPS) for k in ("print", "replay", "mask_rigid", "mask_silicone") for _ in range(2)]
    ok = [pad.predict(o["rgb"], o["depth"], o["ir"], o["kps5"])[0]["live"] for o in live]
    bad = [pad.predict(o["rgb"], o["depth"], o["ir"], o["kps5"])[0]["live"] for o in att]
    assert sum(ok) >= 5 and not any(bad)
