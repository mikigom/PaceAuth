"""Recognition wrappers + E1 helpers.

Random-init tests need no downloads (they exercise the real strict loaders); the real-weight check runs only when
data/models/<name> weights and data/datasets/verification/lfw.bin are present.
    docker run --rm --gpus '"device=5"' -u $(id -u):$(id -g) -e HOME=/tmp -v $PWD:/app -w /app faceauth:dev \
        python -m pytest -q tests/test_recog.py
"""
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from sklearn.model_selection import KFold

from eval.eval_verification import kfold_accuracy, load_bin, tar_at_far
from faceauth.common.align import ARCFACE_DST
from faceauth.models.recog import Recognizer
from faceauth.models.vendor.arcface_vit import vit_l_dp005_mask_005
from faceauth.models.vendor.cvlface_dfa import DFALandmarks
from faceauth.models.vendor.cvlface_kprpe import vit_base_kprpe

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = Path(__file__).resolve().parents[1]
LFW = ROOT / "data/datasets/verification/lfw.bin"


@pytest.mark.parametrize("arch", ["insightface_vit_l_dp005_mask_005", "cvlface_vit_base_kprpe"])
def test_random_init_loader_and_embed(tmp_path, arch):
    torch.manual_seed(0)
    if arch == "insightface_vit_l_dp005_mask_005":
        torch.save(vit_l_dp005_mask_005().state_dict(), tmp_path / "model.pt")
        files = {"weights": "model.pt"}
    else:  # same key layout as the HF safetensors ("model.net." prefix)
        save_file({"model.net." + k: v for k, v in vit_base_kprpe().state_dict().items()}, tmp_path / "model.safetensors")
        save_file({"model.net." + k: v for k, v in DFALandmarks().state_dict().items()}, tmp_path / "aligner.safetensors")
        files = {"weights": "model.safetensors", "aligner": "aligner.safetensors"}
    meta = {"name": "t", "version": "0", "kind": "recognition", "files": files, "sha256": {}, "dim": 512,
            "params": {"arch": arch, "align": "arcface5-112@1", "fp16": False}}
    (tmp_path / "MODEL.json").write_text(json.dumps(meta))
    rec = Recognizer(tmp_path, device=DEV)  # load_state_dict(strict=True): any missing/unexpected key raises
    assert (rec.version, rec.dim, rec.align) == ("t@0", 512, "arcface5-112@1")
    crops = np.random.default_rng(0).integers(0, 256, (3, 112, 112, 3), dtype=np.uint8)
    e = rec.embed_aligned(crops)
    assert e.shape == (3, 512) and e.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(e, axis=1), 1, atol=1e-5)
    # kps == template -> norm_crop is the identity, so embed() must reproduce embed_aligned()
    assert (np.sum(e * rec.embed(list(crops), [ARCFACE_DST] * 3), 1) > 0.9999).all()
    assert rec.embed([], []).shape == (0, 512)


def test_load_bin_layout_and_rejects_code(tmp_path):
    jpg = [cv2.imencode(".jpg", np.full((112, 112, 3), 40 * i, np.uint8))[1] for i in range(4)]
    p = tmp_path / "t.bin"
    for payload in ((jpg, [True, False]), ([b.tobytes() for b in jpg], [1, 0])):
        p.write_bytes(pickle.dumps(payload, protocol=4))  # ndarray entries (HF mirror) and bytes entries
        imgs, issame = load_bin(p)
        assert imgs.shape == (4, 112, 112, 3) and issame.tolist() == [True, False]

    class Evil:
        def __reduce__(self):
            return (print, ("pwned",))

    p.write_bytes(pickle.dumps(([Evil()], [True])))
    with pytest.raises(pickle.UnpicklingError):
        load_bin(p)


def _upstream_accuracy(emb, issame):  # facenet/InsightFace calculate_roc, written out loop by loop
    dist = np.sum(np.square(emb[0::2] - emb[1::2]), 1)
    thresholds, accs = np.arange(0, 4, 0.01), []
    acc = lambda t, idx: np.mean(np.less(dist[idx], t) == issame[idx])
    for train, test in KFold(n_splits=10, shuffle=False).split(np.arange(len(dist))):
        best = np.argmax([acc(t, train) for t in thresholds])
        accs.append(acc(thresholds[best], test))
    return 100 * np.mean(accs), 100 * np.std(accs)


def test_metrics():
    rng = np.random.default_rng(0)
    n, issame = 600, np.arange(600) % 2 == 0
    a = rng.normal(size=(n, 64))
    for noise in (0.1, 1.2):  # separable / partly overlapping
        b = np.where(issame[:, None], a + noise * rng.normal(size=a.shape), rng.normal(size=a.shape))
        emb = np.empty((2 * n, 64))
        emb[0::2], emb[1::2] = a, b
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        np.testing.assert_allclose(kfold_accuracy(emb, issame), _upstream_accuracy(emb, issame))
    t = tar_at_far(emb, issame)
    assert t["n_impostor"] == 300 and t["far"] <= 1e-3 and t["tar_ci95"][0] <= t["tar"] <= t["tar_ci95"][1]


@pytest.mark.parametrize("name", ["kprpe_vitb_wf12m", "insightface_vit_l_wf42m"])
def test_real_weights_lfw_fold1(name):
    d = ROOT / "data/models" / name
    files = json.loads((d / "MODEL.json").read_text())["files"].values()
    if not LFW.exists() or not all((d / f).exists() for f in files):
        pytest.skip("weights or lfw.bin not present")
    imgs, issame = load_bin(LFW)
    rec = Recognizer(d, device=DEV)
    e = rec.embed_aligned(imgs[:1200])  # LFW fold 1: 300 genuine + 300 impostor pairs
    assert kfold_accuracy(e.astype(np.float64), issame[:600])[0] > 99.0
    # a face scaled 2x and shifted inside a larger canvas is re-aligned by embed() via its keypoints
    big = np.zeros((400, 500, 3), np.uint8)
    big[100:324, 150:374] = cv2.resize(imgs[0], (224, 224), interpolation=cv2.INTER_CUBIC)
    assert float(rec.embed([big], [ARCFACE_DST * 2 + (150, 100)])[0] @ e[0]) > 0.9
