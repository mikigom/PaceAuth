"""BIWI Kinect Head Pose Database loader: real RGB-D sequences with ground-truth head pose.

    from faceauth.sim.biwi import iter_frames
    for f in iter_frames("data/datasets/biwi", seqs=["01"]):
        f["rgb"]     # (480,640,3) uint8 RGB
        f["depth"]   # (480,640) uint16 mm, 0 = invalid; registered to the RGB camera (register=False: raw depth camera)
        f["pose"]    # (3,) float32 [yaw, pitch, roll] deg, §0 convention, head rotation relative to the RGB camera
        f["pose_depthcam"]  # same relative to the depth camera (the frame of the published BIWI labels / FSA-Net)
        f["head_center"]    # (3,) float32 GT head centre, RGB camera coords, mm (x right, y down, z forward)
        f["subject"], f["seq"], f["frame"], f["ts_ms"]   # "F01".."M14", "01".."24", int, float

Data: data/datasets/biwi/hpdb/<seq>/{rgb.cal, depth.cal, frame_NNNNN_{rgb.png, depth.bin, pose.txt}} (see DATASET.json).
24 sequences, 20 subjects (F02/F03/F05/M01 recorded twice), 15,678 frames, Kinect v1 at ~1 m. No IR stream.
Calibration: rgb.cal holds K_rgb and the extrinsics (R, T) with X_rgb = R @ X_depth + T (mm); this direction was
picked over the inverse because the registered depth silhouettes then line up with RGB edges (mean RGB gradient on
depth edges 112.7 vs 42.0 for the inverse and 52.2 unregistered, 85 frames; reports/E4_headpose/summary.md).
Registration forward-warps every depth pixel (z-buffer, nearest pixel); RGB pixels outside the depth view stay 0.
The released depth maps contain only the subject (background removed by the dataset authors): ~15-20% of
pixels valid, z ~0.78-1.02 m.
*_pose.txt: rotation R_h (rows; head template -> depth camera) and the head centre (depth camera, mm).
Timestamps: BIWI has none; ts_ms = frame * 1000/30 (nominal Kinect v1 30 fps). Frame numbers have gaps where
the original recording dropped frames (seq 24: 131 missing numbers), so ts_ms reflects those gaps.
Licence: non-commercial use such as university research and education only.
"""
import struct
from pathlib import Path

import cv2
import numpy as np

SUBJECTS = {  # sequence -> subject (BIWI readme; 4 people recorded twice)
    "01": "F01", "02": "F02", "03": "F03", "04": "F04", "05": "F05", "06": "F06", "07": "M01", "08": "M02",
    "09": "M03", "10": "M04", "11": "M05", "12": "M06", "13": "M07", "14": "M08", "15": "F03", "16": "M09",
    "17": "M10", "18": "F05", "19": "M11", "20": "M12", "21": "F02", "22": "M01", "23": "M13", "24": "M14",
}
FPS = 30.0


def read_cal(path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """*.cal -> K (3,3), R (3,3), T (3,) (lines 0-2, 6-8, 10; line 4 = zero distortion, line 12 = size)."""
    rows = [line.split() for line in Path(path).read_text().splitlines()]
    return np.array(rows[0:3], float), np.array(rows[6:9], float), np.array(rows[10], float)


def read_pose(path) -> tuple[np.ndarray, np.ndarray]:
    """*_pose.txt -> R_h (3,3) head->depth camera, head centre (3,) mm in depth-camera coords."""
    rows = [line.split() for line in Path(path).read_text().splitlines() if line.strip()]
    return np.array(rows[0:3], float), np.array(rows[3], float)


def read_depth(path) -> np.ndarray:
    """Compressed *_depth.bin (int32 w, h; then runs of int32 n_empty, int32 n_full, n_full x int16) -> uint16 mm."""
    b = Path(path).read_bytes()
    w, h = struct.unpack_from("<ii", b, 0)
    out, pos, off = np.zeros(w * h, np.int16), 0, 8
    while pos < w * h:
        n_empty, n_full = struct.unpack_from("<ii", b, off)
        pos, off = pos + n_empty, off + 8
        out[pos:pos + n_full] = np.frombuffer(b, "<i2", n_full, off)
        pos, off = pos + n_full, off + 2 * n_full
    return np.clip(out, 0, None).astype(np.uint16).reshape(h, w)


def register_depth(depth, K_d, K_rgb, R, T, shape=(480, 640)) -> np.ndarray:
    """Depth-camera depth -> RGB camera (X_rgb = R X_d + T), nearest-pixel forward warp keeping the closest z."""
    v, u = np.nonzero(depth)
    z = depth[v, u].astype(np.float64)
    P = np.stack([(u - K_d[0, 2]) * z / K_d[0, 0], (v - K_d[1, 2]) * z / K_d[1, 1], z])
    Q = R @ P + T[:, None]
    uc = np.rint(K_rgb[0, 0] * Q[0] / Q[2] + K_rgb[0, 2]).astype(np.int64)
    vc = np.rint(K_rgb[1, 1] * Q[1] / Q[2] + K_rgb[1, 2]).astype(np.int64)
    ok = (Q[2] > 0) & (uc >= 0) & (uc < shape[1]) & (vc >= 0) & (vc < shape[0])
    out = np.full(shape, np.iinfo(np.int64).max, np.int64)
    np.minimum.at(out, (vc[ok], uc[ok]), np.rint(Q[2, ok]).astype(np.int64))
    out[out == np.iinfo(np.int64).max] = 0
    return np.clip(out, 0, 65535).astype(np.uint16)


def pose_from_rotation(R_head) -> np.ndarray:
    """Head->camera rotation (camera x right, y down, z forward) -> [yaw, pitch, roll] deg, §0 convention.
    Same decomposition as the FSA-Net / Hopenet BIWI labels (on R^T) with their yaw negated (theirs follows 300W-LP:
    nose toward image left = positive)."""
    R = np.asarray(R_head, float).T
    return np.degrees([np.arctan2(-R[2, 0], np.hypot(R[2, 1], R[2, 2])),
                       np.arctan2(R[2, 1], R[2, 2]),
                       -np.arctan2(R[1, 0], R[0, 0])]).astype(np.float32)


def project(K, X) -> np.ndarray:
    """Camera coords (3,) mm -> pixel (2,)."""
    return np.array([K[0, 0] * X[0] / X[2] + K[0, 2], K[1, 1] * X[1] / X[2] + K[1, 2]])


def iter_frames(root="data/datasets/biwi", seqs=None, register=True, step=1):
    """Yield one dict per frame (module docstring), sequences in order, frames in recording order."""
    hp = Path(root) / "hpdb"
    for seq in seqs or sorted(SUBJECTS):
        K_d, _, _ = read_cal(hp / seq / "depth.cal")
        K_rgb, R_ext, T_ext = read_cal(hp / seq / "rgb.cal")
        for p in sorted((hp / seq).glob("frame_*_rgb.png"))[::step]:
            stem = str(p)[:-len("_rgb.png")]
            frame = int(p.name.split("_")[1])
            R_h, center = read_pose(stem + "_pose.txt")
            depth = read_depth(stem + "_depth.bin")
            yield {
                "rgb": cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB),
                "depth": register_depth(depth, K_d, K_rgb, R_ext, T_ext) if register else depth,
                "pose": pose_from_rotation(R_ext @ R_h),
                "pose_depthcam": pose_from_rotation(R_h),
                "head_center": (R_ext @ center + T_ext).astype(np.float32),
                "K_rgb": K_rgb,
                "subject": SUBJECTS[seq], "seq": seq, "frame": frame, "ts_ms": frame * 1000.0 / FPS,
            }


if __name__ == "__main__":  # self-check on real data: python -m faceauth.sim.biwi [root]
    import sys

    f = next(iter_frames(sys.argv[1] if len(sys.argv) > 1 else "data/datasets/biwi", seqs=["01"]))
    assert f["rgb"].shape == (480, 640, 3) and f["rgb"].dtype == np.uint8
    assert f["depth"].shape == (480, 640) and f["depth"].dtype == np.uint16 and (f["depth"] > 0).mean() > 0.05
    u, v = np.rint(project(f["K_rgb"], f["head_center"])).astype(int)
    z = f["depth"][v - 3:v + 4, u - 3:u + 4]
    z = float(np.median(z[z > 0]))
    # the head centre lies inside the skull: the face surface seen at its projection is 2-15 cm closer
    assert 20 < f["head_center"][2] - z < 150, (f["head_center"], z)
    assert np.all(np.abs(f["pose"] - f["pose_depthcam"]) < 2.0), (f["pose"], f["pose_depthcam"])
    print("ok", f["subject"], f["seq"], f["frame"], f["ts_ms"], f["pose"], "head z", f["head_center"][2], "surface z", z)
