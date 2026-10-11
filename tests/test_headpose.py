"""Head pose (faceauth/models/headpose.py) + BIWI loader (faceauth/sim/biwi.py) checks.

The §0 convention tests on synthetic rotations always run; the model / data tests skip without GPU, weights or data.
"""
from pathlib import Path

import numpy as np
import pytest

from faceauth.sim.biwi import pose_from_rotation, read_depth, register_depth


def _rx(d):
    a = np.radians(d)
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def _ry(d):
    a = np.radians(d)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def _rz(d):
    a = np.radians(d)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def test_biwi_pose_convention():
    # camera x right, y down, z forward; a frontal head looks along -z. §0: yaw > 0 = nose toward image right,
    # pitch > 0 = looking up, roll > 0 = clockwise in the image.
    R = _ry(-20)
    assert (R @ [0, 0, -1])[0] > 0  # nose moved toward image right
    np.testing.assert_allclose(pose_from_rotation(R), [20, 0, 0], atol=1e-4)
    R = _rx(-15)
    assert (R @ [0, 0, -1])[1] < 0  # nose moved up (y down)
    np.testing.assert_allclose(pose_from_rotation(R), [0, 15, 0], atol=1e-4)
    R = _rz(10)
    assert (R @ [1, 0, 0])[1] > 0  # image-right axis turned downward = clockwise
    np.testing.assert_allclose(pose_from_rotation(R), [0, 0, 10], atol=1e-4)
    # mirror (x -> -x) negates yaw and roll, keeps pitch
    M = np.diag([-1.0, 1, 1])
    R = _rz(7) @ _ry(-25) @ _rx(-12)
    p, q = pose_from_rotation(R), pose_from_rotation(M @ R @ M)
    np.testing.assert_allclose(q, p * [-1, 1, -1], atol=1e-4)


def test_depth_rle_and_registration(tmp_path):
    d = np.zeros((4, 6), np.int16)
    d[1, 2:5] = [800, 801, 802]
    d[3, 0] = 900
    flat, runs, i = d.ravel(), [], 0
    while i < flat.size:  # encode like the BIWI writer: (n_empty, n_full, values)
        j = i
        while j < flat.size and flat[j] == 0:
            j += 1
        k = j
        while k < flat.size and flat[k] != 0:
            k += 1
        runs.append(np.array([j - i, k - j], "<i4").tobytes() + flat[j:k].astype("<i2").tobytes())
        i = k
    p = tmp_path / "f_depth.bin"
    p.write_bytes(np.array([6, 4], "<i4").tobytes() + b"".join(runs))
    out = read_depth(p)
    assert out.dtype == np.uint16 and np.array_equal(out, d.astype(np.uint16))
    K = np.array([[5.0, 0, 3], [0, 5.0, 2], [0, 0, 1]])
    assert np.array_equal(register_depth(out, K, K, np.eye(3), np.zeros(3), shape=(4, 6)), out)


def _model_dirs():
    return [d for d in ("data/models/6drepnet", "data/models/6drepnet360") if Path(d, "MODEL.json").exists()]


@pytest.fixture(scope="module")
def aflw():
    root = Path("data/datasets/aflw2000/AFLW2000")
    if not root.exists():
        pytest.skip("AFLW2000-3D not present")
    import scipy.io as sio
    from PIL import Image
    crops, gt = [], []
    for f in sorted(root.glob("image*.jpg"))[:300]:
        mat = sio.loadmat(f.with_suffix(".mat"))
        pitch, yaw, roll = np.degrees(mat["Pose_Para"][0][:3])
        if max(abs(pitch), abs(yaw), abs(roll)) > 40:
            continue
        lm = mat["pt3d_68"][:2]
        x1, y1, x2, y2 = lm[0].min(), lm[1].min(), lm[0].max(), lm[1].max()
        w, h = x2 - x1, y2 - y1
        crops.append(np.asarray(Image.open(f).convert("RGB").crop((int(x1 - .4 * w), int(y1 - .4 * h),
                                                                    int(x2 + .4 * w), int(y2 + .1 * h)))))
        gt.append([-yaw, pitch, roll])  # 3DDFA labels: yaw > 0 = nose toward image left -> §0
    return crops, np.array(gt)


@pytest.mark.parametrize("model_dir", _model_dirs())
def test_headpose_signs_flip_and_accuracy(model_dir, aflw):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    from faceauth.models.headpose import HeadPose
    hp = HeadPose(model_dir)
    crops, gt = aflw
    p = hp.predict_crops(crops)
    f = hp.predict_crops([np.ascontiguousarray(c[:, ::-1]) for c in crops])
    assert p.shape == (len(gt), 3) and p.dtype == np.float32
    err = np.abs(p - gt).mean(0)
    assert np.all(err < 6), err  # published AFLW2000 MAE ~3-6 deg per angle
    big = np.abs(gt[:, 0]) > 10
    assert (np.sign(p[big, 0]) == np.sign(gt[big, 0])).mean() > 0.95
    assert np.abs(p[:, 0] + f[:, 0]).mean() < 3 and np.abs(p[:, 2] + f[:, 2]).mean() < 3  # flip negates yaw, roll
    assert np.abs(p[:, 1] - f[:, 1]).mean() < 3  # and keeps pitch
    img = np.zeros((480, 640, 3), np.uint8)
    assert hp.predict(img, np.zeros((0, 4))).shape == (0, 3)
    assert hp.predict(img, [[200, 150, 330, 310]]).shape == (1, 3)


def test_biwi_frame():
    if not Path("data/datasets/biwi/hpdb/01/rgb.cal").exists():
        pytest.skip("BIWI not present")
    from faceauth.sim.biwi import iter_frames, project
    f = next(iter_frames("data/datasets/biwi", seqs=["01"]))
    assert f["rgb"].shape == (480, 640, 3) and f["depth"].dtype == np.uint16 and f["subject"] == "F01"
    u, v = np.rint(project(f["K_rgb"], f["head_center"])).astype(int)
    z = f["depth"][v - 3:v + 4, u - 3:u + 4]
    assert 20 < f["head_center"][2] - np.median(z[z > 0]) < 150  # registered face surface in front of head centre
