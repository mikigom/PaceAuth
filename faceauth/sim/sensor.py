"""Simulated Gemini-336-like RGB + depth + IR sensor (D2C aligned, 640x480) with live subjects and
presentation attacks (PAI species: print, replay, mask_rigid, mask_silicone).

NOT a validated sensor model. It is physically motivated so that capture -> engine -> PAD -> policy can be
built and tested without hardware; PAD results on it validate code paths and metric computation only.
Real-sensor training/validation is deferred to the hardware phase.

Live subject: a source photo + its 5 keypoints (InsightFace order). The photo is placed at 400-800 mm by a 2D
similarity (its head pose is kept, never re-posed). Depth comes from a 3D head: the MediaPipe canonical face
mesh (Apache-2.0, vendored) posed by PnP on the keypoints and personalised (affine + RBF) so its landmarks
project exactly onto them, plus skull/hair, neck and torso primitives, in front of a background plane.
RGB = the photo on the head + synthetic clothes/background. IR = NIR albedo x Lambertian/specular shading
from depth normals, co-located 850 nm illuminator with 1/d^2 falloff, + ambient NIR, corneal glints.
Sensor: stereo depth noise sigma_z = z^2 sigma_d / (f b), stereo + depth-to-colour occlusion shadows,
flying pixels at depth edges, material holes (dark hair, black ink, glass), specular saturation, range
limits, 1 mm quantisation; IR auto-exposure on the face with clamped gain, shot + read noise.
"""
import functools
from pathlib import Path

import cv2
import numpy as np
from scipy.special import ndtr

W, H = 640, 480
DEFAULT_PROFILE = "gemini336-sim-v1"
KINDS = ("live", "print", "replay", "mask_rigid", "mask_silicone")
ATTACKS = KINDS[1:]

PROFILES = {
    # nominal calibration, to be replaced by the real device calibration in the hardware phase
    "gemini336-sim-v1": dict(
        calib_version="gemini336-sim-v1", fx=385.0, fy=385.0, cx=319.5, cy=239.5,
        baseline=50.0, d2c_baseline=15.0, disp_sigma=0.07, z_range=(250.0, 6000.0), hole_rate=0.002,
        ir_offset=(25.0, 0.0, 0.0), ir_ambient=0.03, ir_target=0.55, ir_gain_max=4.0, ir_read=1.2, ir_shot=0.12,
        rgb_gain=(1.0, 1.0, 1.0), rgb_noise=1.5, room_ambient=(0.35, 0.6), bg_dist=(1200.0, 3500.0),
        bg_level=(0.2, 0.8)),
    # unseen sensor / location: noisier depth, weaker illuminator + strong ambient NIR (window), warm white
    # balance, brighter and farther background
    "sensor_b": dict(
        calib_version="sensor_b-sim-v1", fx=400.0, fy=400.0, cx=322.0, cy=236.0,
        baseline=40.0, d2c_baseline=20.0, disp_sigma=0.12, z_range=(280.0, 5000.0), hole_rate=0.006,
        ir_offset=(-30.0, 10.0, 0.0), ir_ambient=0.25, ir_target=0.45, ir_gain_max=3.0, ir_read=2.5, ir_shot=0.25,
        rgb_gain=(1.08, 1.0, 0.86), rgb_noise=3.0, room_ambient=(0.55, 0.85), bg_dist=(1800.0, 6500.0),
        bg_level=(0.5, 0.95)),
}

BG, SKIN, EYE, LIPS, HAIR, NECK, CLOTH, PAI, FRAME = range(9)  # per-pixel material labels
LUMA = np.array([0.299, 0.587, 0.114], np.float32)
ARRAYS = ("z", "lab", "rgb", "alb", "spec", "shin", "inval", "shade")

_DIR = Path(__file__).resolve().parent
EYE_R = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]  # subject right = image left
EYE_L = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
LIPS_IN = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
KP_IDX = (EYE_R, EYE_L, [1], [61], [291])  # InsightFace order: img-left eye, img-right eye, nose, mouth l, r
IOD_MM, EM_MM = 63.2, 70.0  # canonical eye distance, eye-mid to mouth-mid
# replay devices: screen w, h, bezel (mm); face IOD shown on screen (mm); held at distance (mm); invalid-depth
# fraction on the screen (glossy glass reflects the projector pattern away; matte monitors less so)
DEVICES = {"phone": (65, 140, 4, (18, 35), (220, 450), (0.2, 0.8)),
           "tablet": (150, 220, 10, (35, 65), (300, 700), (0.2, 0.8)),
           "monitor": (530, 300, 14, (50, 90), (450, 900), (0.05, 0.5))}


def _kps3d(V):
    return np.stack([V[i].mean(0) for i in KP_IDX])


def _outward(V, F, axis_only=False):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cen = (a + b + c) / 3
    if axis_only:
        cen = cen * (1, 0, 1)
    flip = (np.cross(b - a, c - a) * cen).sum(1) < 0
    F = F.copy()
    F[flip] = F[flip][:, ::-1]
    return F


@functools.cache
def _face_mesh():
    """-> V (n,3) mm in the head frame (x image-right, y down, z away from the camera, origin = eye midpoint),
    F (m,3) outward winding, per-triangle labels."""
    V, F = [], []
    with open(_DIR / "canonical_face_model.obj") as f:
        for line in f:
            p = line.split()
            if p and p[0] == "v":
                V.append([float(x) for x in p[1:4]])
            elif p and p[0] == "f":
                F.append([int(x.split("/")[0]) - 1 for x in p[1:4]])
    V = np.asarray(V) * (10.0, -10.0, -10.0)  # cm, y up, z to viewer -> mm, y down, z away (a rotation)
    V -= _kps3d(V)[:2].mean(0)
    F = np.asarray(F)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    if np.cross(b - a, c - a)[:, 2].sum() > 0:  # front faces must point to -z (camera)
        F = F[:, ::-1].copy()
    lab = np.full(len(F), SKIN, np.uint8)
    lab[np.isin(F, EYE_R + EYE_L).all(1)] = EYE
    lab[np.isin(F, LIPS_IN).all(1)] = LIPS
    return V, F, lab


def _grid_mesh(nu, nv, cylinder):
    u = np.linspace(-np.pi, np.pi, nu, endpoint=False)
    v = np.linspace(0, 1, nv + 1) if cylinder else np.linspace(-np.pi / 2, np.pi / 2, nv + 1)
    uu, vv = np.meshgrid(u, v)
    if cylinder:
        V = np.stack([np.sin(uu), vv, -np.cos(uu)], -1).reshape(-1, 3)
    else:
        V = np.stack([np.sin(uu) * np.cos(vv), np.sin(vv), -np.cos(uu) * np.cos(vv)], -1).reshape(-1, 3)
    i, j = np.meshgrid(np.arange(nv), np.arange(nu), indexing="ij")
    a, b = i * nu + j, i * nu + (j + 1) % nu
    F = np.concatenate([np.stack([a, a + nu, b], -1).reshape(-1, 3), np.stack([b, a + nu, b + nu], -1).reshape(-1, 3)])
    return V, _outward(V, F, axis_only=cylinder)


_SPHERE = _grid_mesh(48, 24, False)
_CYLINDER = _grid_mesh(32, 4, True)


def _smooth_mesh(V, F, iters, lam=0.5, keep_boundary=False):
    """Laplacian smoothing; keep_boundary pins the open boundary (otherwise it shrinks inwards)."""
    e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
    u, cnt = np.unique(e, axis=0, return_counts=True)
    move = np.ones((len(V), 1))
    if keep_boundary:
        move[u[cnt == 1].ravel()] = 0
    e = np.concatenate([e, e[:, ::-1]])
    deg = np.bincount(e[:, 0], minlength=len(V))[:, None]
    for _ in range(int(iters)):
        nb = np.zeros_like(V)
        np.add.at(nb, e[:, 0], V[e[:, 1]])
        V = V + lam * move * (nb / deg - V)
    return V


def _skin(rng):
    return rng.uniform(0.35, 0.95) * np.array([1.0, rng.uniform(0.7, 0.82), rng.uniform(0.55, 0.7)])


def _vertex_normals(V, F):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    vn = np.zeros_like(V)
    for k in range(3):
        np.add.at(vn, F[:, k], fn)
    return vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-9)


def _vary(rng, V):
    """Per-subject face shape variation of the canonical mesh."""
    V = V * (rng.uniform(0.94, 1.06), rng.uniform(0.94, 1.06), rng.uniform(0.85, 1.15))
    nose = _kps3d(V)[2]
    V[:, 2] -= rng.uniform(-4, 5) * np.exp(-((V[:, :2] - nose[:2]) ** 2).sum(1) / (2 * 15.0 ** 2))
    for _ in range(3):  # smooth individual bumps (cheeks, chin, brow)
        c = V[rng.integers(len(V))]
        V[:, 2] += rng.uniform(-3, 3) * np.exp(-((V - c) ** 2).sum(1) / (2 * rng.uniform(15, 30) ** 2))
    return V


def _pose(R):
    """Head pose (yaw, pitch, roll) in degrees, IMPLEMENTATION.md §0 convention."""
    f, r = R @ (0, 0, -1.0), R @ (1.0, 0, 0)
    return [float(np.degrees(np.arctan2(f[0], -f[2]))), float(np.degrees(np.arctan2(-f[1], np.hypot(f[0], f[2])))),
            float(np.degrees(np.arctan2(r[1], r[0])))]


def _rot(pitch, yaw, roll):
    """Degrees -> R = Rz(roll) Ry(yaw) Rx(pitch) (camera axes: x right, y down, z forward)."""
    a, b, c = np.radians([pitch, yaw, roll])
    rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return rz @ ry @ rx


def _smooth_noise(rng, shape, sigma):
    n = cv2.GaussianBlur(rng.standard_normal(shape).astype(np.float32), (0, 0), sigma)
    return n / (n.std() + 1e-8)


def bbox_from_kps(kps):
    k = np.asarray(kps, np.float32)
    (x0, y0), (x1, y1) = k.min(0), k.max(0)
    w, h = x1 - x0, y1 - y0
    return np.array([x0 - 0.6 * w, y0 - 1.0 * h, x1 + 0.6 * w, y1 + 0.8 * h], np.float32)


class SensorSim:
    """sim = SensorSim("gemini336-sim-v1", seed=0); out = sim.live(face_rgb, kps5) / sim.attack("print", ...)
    -> {"rgb" uint8 HxWx3, "depth" uint16 HxW mm (0 invalid), "ir" uint8 HxW, "kps5" (5,2), "bbox" (4,),
        "faces" [(5,2)...], "kind", "profile", "calib_version", "meta"}.
    Pass seed=... per call for a reproducible sample; otherwise the instance RNG advances."""

    def __init__(self, profile: str = DEFAULT_PROFILE, seed=None):
        if profile not in PROFILES:
            raise ValueError(f"unknown profile {profile!r}, have {sorted(PROFILES)}")
        self.profile, self.p = profile, PROFILES[profile]
        self.calib_version = self.p["calib_version"]
        p = self.p
        self.f, self.c = np.array([p["fx"], p["fy"]]), np.array([p["cx"], p["cy"]])
        self.K = np.array([[p["fx"], 0, p["cx"]], [0, p["fy"], p["cy"]], [0, 0, 1.0]])
        self.rng = np.random.default_rng(seed)
        u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        self.uv = (u, v)
        self.rays = np.stack([(u - p["cx"]) / p["fx"], (v - p["cy"]) / p["fy"], np.ones_like(u)], -1)
        r2 = ((u - W / 2) ** 2 + (v - H / 2) ** 2) / (W / 2) ** 2
        self.vignette = (1 - 0.2 * r2).astype(np.float32)

    # ------------------------------------------------------------------ public API
    def _rng(self, seed):
        return np.random.default_rng(seed if seed is not None else self.rng.integers(2 ** 63))

    def live(self, face_rgb, kps5, seed=None, distance_mm=None) -> dict:
        rng = self._rng(seed)
        return self.capture([self.scene("live", face_rgb, kps5, rng, distance_mm)], rng)

    def attack(self, kind, face_rgb, kps5, seed=None, distance_mm=None) -> dict:
        if kind not in ATTACKS:
            raise ValueError(f"unknown PAI species {kind!r}, have {ATTACKS}")
        rng = self._rng(seed)
        return self.capture([self.scene(kind, face_rgb, kps5, rng, distance_mm)], rng)

    def render(self, kind, face_rgb, kps5, seed=None, distance_mm=None) -> dict:
        return self.live(face_rgb, kps5, seed, distance_mm) if kind == "live" else \
            self.attack(kind, face_rgb, kps5, seed, distance_mm)

    def scene(self, kind, face_rgb, kps5, rng, distance_mm=None, background=True, center=None) -> dict:
        """Noise-free scene layers (z, materials, ...). Compose several with capture([...])."""
        kps = np.asarray(kps5, np.float64).reshape(5, 2)
        img = np.asarray(face_rgb)
        if img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
            raise ValueError("face_rgb must be uint8 HxWx3 RGB")
        if kind == "live":
            return self._scene_person(rng, img, kps, distance_mm, background, center, mask=None)
        if kind in ("mask_rigid", "mask_silicone"):
            return self._scene_person(rng, img, kps, distance_mm, background, center, mask=kind)
        if kind in ("print", "replay"):
            return self._scene_flat(rng, img, kps, distance_mm, kind)
        raise ValueError(f"unknown kind {kind!r}")

    def capture(self, scenes, rng=None) -> dict:
        """Apply the sensor model to one scene or a z-composite of several (first = primary face)."""
        rng = rng if rng is not None else self._rng(None)
        sc = _composite(scenes)
        z = np.where(np.isfinite(sc["z"]), sc["z"], 1e5).astype(np.float32)
        n = self._normals(z)
        rgb = self._rgb(rng, sc, n)
        ir, sat, gain = self._ir(rng, sc, z, n)
        depth = self._depth(rng, sc, z, sat)
        k = sc["kps"].astype(np.float32)
        return dict(rgb=rgb, depth=depth, ir=ir, kps5=k, bbox=bbox_from_kps(k),
                    faces=[s["kps"].astype(np.float32) for s in scenes], kind=sc["meta"]["kind"],
                    profile=self.profile, calib_version=self.calib_version, meta=dict(sc["meta"], ir_gain=gain))

    # ------------------------------------------------------------------ geometry
    def _ray(self, pts):
        pts = np.asarray(pts, np.float64).reshape(-1, 2)
        return np.c_[(pts - self.c) / self.f, np.ones(len(pts))]

    def _project(self, X):
        X = np.asarray(X, np.float64).reshape(-1, 3)
        return X[:, :2] / X[:, 2:3] * self.f + self.c

    def _raster(self, V, F):
        """Painter's-order rasterisation (camera-frame mesh) -> triangle ids (H,W) int32, z (H,W) float32."""
        a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        nrm, cen = np.cross(b - a, c - a), (a + b + c) / 3
        vis = ((nrm * cen).sum(1) < 0) & (cen[:, 2] > 50)
        pts = np.round(self._project(V)[F] * 16).astype(np.int32)
        ids = np.full((H, W), -1, np.int32)
        for j in np.flatnonzero(vis)[np.argsort(-cen[vis, 2])]:
            cv2.fillConvexPoly(ids, pts[j], int(j), cv2.LINE_8, 4)
        z = np.full((H, W), np.inf, np.float32)
        m = ids >= 0
        if m.any():
            j = ids[m]
            den = (nrm[j] * self.rays[m]).sum(1)
            zz = (nrm[j] * a[j]).sum(1) / np.where(np.abs(den) < 1e-9, 1e-9, den)
            zs = np.stack([a[j, 2], b[j, 2], c[j, 2]])
            z[m] = np.clip(zz, zs.min(0), zs.max(0))
        return ids, z

    def _render(self, objs):
        z = np.full((H, W), np.inf, np.float32)
        lab = np.zeros((H, W), np.uint8)
        for V, F, L in objs:
            ids, zk = self._raster(V, F)
            m = zk < z
            if isinstance(L, np.ndarray):  # the face mesh always wins over the skull/hair proxy behind it
                m |= np.isfinite(zk) & (lab == HAIR)
            z[m] = zk[m]
            lab[m] = L[ids[m]] if isinstance(L, np.ndarray) else L
        return z, lab

    def _place(self, rng, kps, zt, center=None, roll=8.0):
        """2D similarity photo -> frame so the face appears at depth ~zt (in-plane roll jitter only)."""
        eye, mouth = kps[:2].mean(0), kps[3:].mean(0)
        s = min(self.p["fx"] * IOD_MM / (zt * np.linalg.norm(kps[1] - kps[0])),
                self.p["fx"] * EM_MM / (zt * np.linalg.norm(mouth - eye)))
        a = np.radians(rng.uniform(-roll, roll))
        c_out = np.asarray(center, np.float64) if center is not None else \
            np.array([W / 2 + rng.uniform(-60, 60), H / 2 - 25 + rng.uniform(-35, 35)])
        A = s * np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        S = np.c_[A, c_out - A @ kps.mean(0)]
        return S, kps @ A.T + S[:, 2]

    def _warp(self, img, S):
        """Photo -> frame (float RGB in [0,1]) + feathered validity."""
        img = img.astype(np.float32) / 255
        s = np.sqrt(abs(np.linalg.det(S[:, :2])))
        if s < 0.7:  # anti-alias large photos
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            S = np.c_[S[:, :2] / s, S[:, 2]]
        out = cv2.warpAffine(img, S, (W, H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        valid = cv2.warpAffine(np.ones(img.shape[:2], np.float32), S, (W, H), flags=cv2.INTER_LINEAR)
        valid = cv2.GaussianBlur(cv2.erode(valid, np.ones((5, 5), np.uint8)), (0, 0), 1.5)
        return out, valid

    def _pnp(self, KP, kps_out, zt):
        rvec, tvec = np.zeros(3), self._ray(kps_out.mean(0))[0] * zt
        ok, rvec, tvec = cv2.solvePnP(KP.astype(np.float64), np.ascontiguousarray(kps_out, np.float64), self.K, None,
                                      rvec, tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        R, t = cv2.Rodrigues(rvec)[0], tvec.ravel()
        if not ok or not 150 < t[2] < 3000 or (R @ (0, 0, -1.0))[2] > -0.3:
            raise ValueError("cannot fit a head pose to these keypoints")
        return R, t

    def _personalise(self, V, R, t, kps_out, sigma=30.0):
        """Deform a template mesh in the head frame so its 5 landmarks project exactly onto kps_out."""
        KP = _kps3d(V)
        Pp = (self._ray(kps_out) * (KP @ R.T + t)[:, 2:3] - t) @ R  # landmark rays at template depth, head frame
        A = np.linalg.lstsq(np.c_[KP[:, :2], np.ones(5)], Pp[:, :2], rcond=None)[0]
        V, KP = V.copy(), KP.copy()
        V[:, :2], KP[:, :2] = np.c_[V[:, :2], np.ones(len(V))] @ A, np.c_[KP[:, :2], np.ones(5)] @ A
        G = np.exp(-((KP[:, None] - KP[None]) ** 2).sum(-1) / (2 * sigma ** 2))
        wts = np.linalg.solve(G + 1e-6 * np.eye(5), Pp - KP)
        V += np.exp(-((V[:, None] - KP[None]) ** 2).sum(-1) / (2 * sigma ** 2)) @ wts
        return V, np.linalg.norm(A[:2], axis=0)

    def _fit_head(self, rng, kps, zt, S):
        """Pose a (randomly varied) 3D head on the placed keypoints; rescale the placement until the head
        sits at depth zt. -> head dict incl. the final photo->frame similarity S and keypoints."""
        V0, F, flab = _face_mesh()
        V0 = _vary(rng, V0)
        KP = _kps3d(V0)
        for _ in range(4):
            kps_out = kps @ S[:, :2].T + S[:, 2]
            R, t = self._pnp(KP, kps_out, zt)
            r = t[2] / zt
            if abs(r - 1) < 0.02:
                break
            c = kps_out.mean(0)
            S = np.c_[S[:, :2] * r, c - r * (S[:, :2] @ kps.mean(0))]
        V, sxy = self._personalise(V0, R, t, kps_out)
        if not (0.7 < sxy.min() and sxy.max() < 1.4):
            raise ValueError("implausible keypoint geometry")
        return dict(R=R, t=t, V=V, V0=V0, F=F, flab=flab, sxy=sxy, pose=_pose(R), S=S, kps=kps_out)

    def _person(self, rng, head):
        R, t, (sx, sy) = head["R"], head["t"], head["sxy"]
        (Vs, Fs), (Vc, Fc) = _SPHERE, _CYLINDER
        skull = Vs * (80 * sx, 118 * sy, 90) + (0, -18 * sy, 78)  # meets the face-mesh boundary (1-6 mm gap)
        neck = Vc * (52 * sx, 190, 46) + (0, 40, 55)
        base = R @ (0, 150, 55) + t
        torso = (Vs * (rng.uniform(185, 225), 300, rng.uniform(100, 125))) @ _rot(0, rng.uniform(-15, 15), 0).T
        return [(skull @ R.T + t, Fs, HAIR), (neck @ R.T + t, Fc, NECK), (torso + base + (0, 300, 40), Fs, CLOTH),
                (head["V"] @ R.T + t, head["F"], head["flab"])]

    def _cast_sheet(self, Rp, T, k):
        """Rays vs a flat (k=0) or cylindrically bent (curvature k, axis = sheet y; k>0 convex) sheet whose
        local frame is (Rp, T). -> camera z (nan = miss), sheet coords s (arc length), t (mm)."""
        o, d = -(T @ Rp), self.rays @ Rp
        if k == 0:
            with np.errstate(divide="ignore", invalid="ignore"):
                lam = -o[2] / d[..., 2]
            X = o + lam[..., None] * d
            return np.where(lam > 0, lam, np.nan), X[..., 0], X[..., 1]
        r, sg = 1 / abs(k), np.sign(k)
        zc = sg * r
        a = d[..., 0] ** 2 + d[..., 2] ** 2
        b = 2 * (o[0] * d[..., 0] + (o[2] - zc) * d[..., 2])
        cc = o[0] ** 2 + (o[2] - zc) ** 2 - r ** 2
        disc = b * b - 4 * a * cc
        sq = np.sqrt(np.maximum(disc, 0))
        lam = (-b - sg * sq) / (2 * a)
        X = o + lam[..., None] * d
        phi = np.arctan2(X[..., 0], sg * (zc - X[..., 2]))
        ok = (disc >= 0) & (np.abs(phi) < np.pi / 2) & (lam > 0)
        return np.where(ok, lam, np.nan), r * phi, X[..., 1]

    def _sheet_points(self, Rp, T, k, st):
        s, t = st[:, 0], st[:, 1]
        if k == 0:
            P = np.c_[s, t, np.zeros_like(s)]
        else:
            r, sg = 1 / abs(k), np.sign(k)
            P = np.c_[r * np.sin(s / r), t, sg * r * (1 - np.cos(s / r))]
        return P @ Rp.T + T

    def _pai_pose(self, rng, z, yaw, pitch, roll):
        Rp = _rot(rng.uniform(-pitch, pitch), rng.uniform(-yaw, yaw), rng.uniform(-roll, roll))
        target = (W / 2 + rng.uniform(-60, 60), H / 2 - 25 + rng.uniform(-35, 35))
        return Rp, self._ray(target)[0] * z

    def _moire(self, rng, lo, hi):
        """Low-frequency beat pattern from aliasing a fine periodic structure (halftone / pixel grid)."""
        u, v = self.uv
        out = rng.uniform(lo, hi)
        for _ in range(2):
            th, per = rng.uniform(0, np.pi), rng.uniform(4, 30)
            out = out * np.cos(2 * np.pi * (u * np.cos(th) + v * np.sin(th)) / per + rng.uniform(0, 2 * np.pi))
        return out.astype(np.float32)

    # ------------------------------------------------------------------ scenes
    def _empty(self):
        return dict(z=np.full((H, W), np.inf, np.float32), lab=np.zeros((H, W), np.uint8),
                    rgb=np.zeros((H, W, 3), np.float32), alb=np.zeros((H, W), np.float32),
                    spec=np.zeros((H, W), np.float32), shin=np.full((H, W), 10, np.float32),
                    inval=np.zeros((H, W), np.float32), shade=np.zeros((H, W), np.float32), glints=[])

    def _background(self, rng):
        sc = self._empty()
        n = np.array([rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), -1.0])
        n /= np.linalg.norm(n)
        sc["z"] = (n[2] * rng.uniform(*self.p["bg_dist"]) / (self.rays @ n)).astype(np.float32)
        lo, hi = self.p["bg_level"]
        rgb = np.empty((H, W, 3), np.float32)
        rgb[:] = rng.uniform(lo, hi, 3)
        alb = np.full((H, W), rng.uniform(0.25, 0.8), np.float32)
        rgb *= (1 + rng.uniform(-0.25, 0.25) * np.linspace(-1, 1, H, dtype=np.float32))[:, None, None]
        for _ in range(rng.integers(2, 7)):  # furniture, doors, posters
            (x0, x1), (y0, y1) = np.sort(rng.integers(0, W, 2)), np.sort(rng.integers(0, H, 2))
            rgb[y0:y1, x0:x1] = rng.uniform(lo * 0.5, hi, 3)
            alb[y0:y1, x0:x1] = rng.uniform(0.1, 0.9)
        rgb += 0.05 * _smooth_noise(rng, (H, W), 12)[..., None] + rng.normal(0, 0.015, (H, W, 3)).astype(np.float32)
        sc["rgb"] = np.clip(cv2.GaussianBlur(rgb, (0, 0), 1.5), 0, 1)
        sc["alb"] = np.clip(alb + 0.08 * _smooth_noise(rng, (H, W), 12), 0.03, 0.95)
        sc["inval"] = np.clip((0.12 - sc["alb"]) / 0.12, 0, 0.8)
        return sc

    def _body(self, rng, sc, zb):
        """Attacker's torso behind a hand-held PAI."""
        Vs, Fs = _SPHERE
        base = self._ray((W / 2 + rng.uniform(-80, 80), rng.uniform(330, 470)))[0] * zb
        z, _ = self._render([(Vs * (rng.uniform(185, 225), 300, rng.uniform(100, 125)) + base + (0, 300, 40), Fs, CLOTH)])
        m = z < sc["z"]
        sc["z"][m], sc["lab"][m], sc["rgb"][m], sc["shade"][m] = z[m], CLOTH, rng.uniform(0.05, 0.9, 3), 1
        sc["alb"][m], sc["spec"][m], sc["inval"][m] = rng.uniform(0.1, 0.9), 0, 0.01

    def _scene_person(self, rng, img, kps, zt, background, center, mask):
        """Live subject, or an attacker wearing a rigid / silicone mask textured with the victim's photo."""
        zt = zt if zt is not None else rng.uniform(400, 800)
        head = self._fit_head(rng, kps, zt, self._place(rng, kps, zt, center)[0])  # masks: attacker's own head
        kps_out = head["kps"]
        photo, pv = self._warp(img, head["S"])
        z, lab = self._render(self._person(rng, head))
        sc = self._background(rng) if background else self._empty()
        m = z < sc["z"]
        luma = cv2.GaussianBlur(photo @ LUMA, (0, 0), 1.2)
        skin_m = m & (lab == SKIN) & (pv > 0.5)
        skin_rgb = np.median(photo[skin_m], 0) if skin_m.sum() > 50 else _skin(rng)
        med = float(np.median(luma[skin_m])) if skin_m.sum() > 50 else 0.5
        headp = np.isin(lab, (SKIN, EYE, LIPS, HAIR, NECK))
        own = (lab == EYE) if mask else headp  # attacker: photo eyes (stand-in) only through the eye holes
        if mask == "mask_silicone":
            own = own | (lab == HAIR)  # full-head mask incl. hair
        wph = pv * own
        base = np.empty((H, W, 3), np.float32)
        base[:] = skin_rgb if not mask else _skin(rng)
        base[lab == HAIR] = rng.uniform(0.03, 0.35) * rng.uniform((0.8, 0.7, 0.6), (1.2, 1.0, 0.9))
        base[lab == CLOTH] = rng.uniform(0.05, 0.9, 3)
        gain = 1.0 if not mask else np.where(lab == EYE, rng.uniform(0.35, 0.75), 1.0)[..., None]  # mask shadow
        rgb = wph[..., None] * photo * gain + (1 - wph[..., None]) * base
        a_skin = rng.uniform(0.5, 0.68)
        alb = np.full((H, W), a_skin, np.float32)
        alb_ph = np.clip(a_skin + 0.55 * (luma - med), 0.12, 0.9)
        alb = np.where(np.isin(lab, (SKIN, LIPS)), pv * alb_ph + (1 - pv) * a_skin, alb)
        alb = np.where(lab == EYE, pv * (0.25 + 0.45 * luma) + (1 - pv) * 0.4, alb)
        hair_ph = np.clip(0.06 + 0.55 * luma, 0.04, 0.6) if (not mask or mask == "mask_silicone") else 0
        hw = pv if (not mask or mask == "mask_silicone") else 0
        alb = np.where(lab == HAIR, hw * hair_ph + (1 - hw) * rng.uniform(0.08, 0.45), alb)
        alb = np.where(lab == NECK, 0.95 * a_skin, alb)
        alb = np.where(lab == CLOTH, rng.uniform(0.1, 0.9), alb).astype(np.float32)
        spec = np.where(np.isin(lab, (SKIN, LIPS)), rng.uniform(0.03, 0.12), 0).astype(np.float32)
        inval = np.select([lab == HAIR, lab == EYE, lab == CLOTH],
                          [np.clip((0.25 - alb) / 0.25, 0, 0.9), 0.15, np.clip((0.12 - alb) / 0.12, 0, 0.7)], 0.003)
        for k, v in dict(z=z, lab=lab, rgb=rgb, alb=alb, spec=spec, inval=inval).items():
            sc[k][m] = v[m]
        sc["shin"][m], sc["shade"][m] = 15, (1 - wph)[m]
        iod = np.linalg.norm(kps_out[1] - kps_out[0])
        u, v = self.uv
        for e in kps_out[:2]:  # pupils (dark at 850 nm, not co-axial) + corneal glints
            pup = m & (lab == EYE) & ((u - e[0]) ** 2 + (v - e[1]) ** 2 < (0.05 * iod) ** 2)
            sc["alb"][pup] *= 0.3
            gx, gy = e + rng.normal(0, 0.03 * iod, 2)
            sc["glints"].append((gx, gy, rng.uniform(0.6, 1.2) * (1.0 if not mask else rng.uniform(0.5, 1.0))))
        meta = dict(kind=mask or "live", distance_mm=float(head["t"][2]), pose=head["pose"])
        if mask:
            meta.update(self._put_mask(rng, sc, head, kps_out, photo, pv, mask))
        sc.update(kps=kps_out, meta=meta)
        return sc

    def _put_mask(self, rng, sc, head, kps_out, photo, pv, kind):
        R, t = head["R"], head["t"]
        _, F, flab = _face_mesh()
        if kind == "mask_rigid":  # generic smoothed, flattened face shell standing off the face; eye holes
            V0 = _face_mesh()[0].copy()
            V0[:, 2] *= rng.uniform(0.6, 0.95)
            V0 = _smooth_mesh(V0, F, rng.integers(10, 31))
            keep = ~np.isin(F, EYE_R + EYE_L).any(1)
            off = rng.uniform(5, 20)
            prm = dict(standoff_mm=float(off))
        else:  # thin silicone layer conforming to the attacker's face; eye (and maybe mouth) holes
            th = rng.uniform(2, 6)
            V0 = _smooth_mesh(head["V0"] + th * _vertex_normals(head["V0"], F), F, rng.integers(3, 11), keep_boundary=True)
            keep = flab != EYE
            if rng.random() < 0.5:
                keep &= flab != LIPS
            off = 0.0
            prm = dict(thickness_mm=float(th))
        V, _ = self._personalise(V0, R, t, kps_out)
        V[:, 2] -= off
        ids, z = self._raster(V @ R.T + t, F[keep])
        z = np.where(np.isfinite(z), np.minimum(z, sc["z"] - 1.5), np.inf)  # worn mask: always in front of the face
        m = np.isfinite(z) & (z < sc["z"])
        luma = photo @ LUMA
        g = luma[..., None]
        if kind == "mask_rigid":
            tex = g + rng.uniform(0.5, 0.9) * (photo - g)
            tex = 0.06 + 0.84 * np.clip(tex, 0, 1) ** rng.uniform(0.9, 1.3) * rng.uniform(0.92, 1.05, 3)
            base_ir, k_ink = rng.uniform(0.35, 0.9), rng.uniform(0, 0.5)
            alb = base_ir * (1 - k_ink * (1 - np.clip(luma, 0, 1)))
            spec, shin, shade = rng.uniform(0.02, 0.3), rng.uniform(10, 60), 0.5
            tex = cv2.GaussianBlur(tex.astype(np.float32), (0, 0), rng.uniform(0.6, 1.5))
        else:
            tex = g + rng.uniform(0.7, 1.0) * (photo - g)
            tex = np.clip(tex * rng.uniform(0.92, 1.08, 3), 0, 1)
            tex = cv2.GaussianBlur(tex.astype(np.float32), (0, 0), rng.uniform(0.6, 1.5))
            ls = cv2.GaussianBlur(luma, (0, 0), 1.5)
            alb = rng.uniform(0.35, 0.85) + rng.uniform(0, 0.2) * (ls - float(np.median(ls[m])) if m.any() else 0)
            spec, shin, shade = rng.uniform(0.08, 0.35), rng.uniform(15, 40), 0.3
        fill = _skin(rng)  # mask colour where the photo has no data
        tex = pv[..., None] * tex + (1 - pv[..., None]) * fill
        sc["z"][m], sc["lab"][m], sc["rgb"][m] = z[m], PAI, tex[m]
        sc["alb"][m] = np.clip(np.broadcast_to(alb, (H, W))[m], 0.05, 0.95)
        sc["spec"][m], sc["shin"][m], sc["shade"][m], sc["inval"][m] = spec, shin, shade, 0.005
        sc["glints"] = [gl for gl in sc["glints"] if not m[int(np.clip(gl[1], 0, H - 1)), int(np.clip(gl[0], 0, W - 1))]]
        return prm

    def _scene_flat(self, rng, img, kps, zt, kind):
        """Print (flat / bent paper) or replay (phone / tablet / monitor) held in front of the camera."""
        sc = self._background(rng)
        photo = img.astype(np.float32) / 255
        c = kps.mean(0)
        iod_ph = np.linalg.norm(kps[1] - kps[0])
        if kind == "print":
            mm = rng.uniform(45, 72) / iod_ph  # mm per photo pixel (roughly life-size print)
            hph, wph = photo.shape[:2]
            mg = rng.uniform(0, 20, 4) * (rng.random() < 0.6)
            ext = (-c[0] * mm - mg[0], (wph - c[0]) * mm + mg[1], -c[1] * mm - mg[2], (hph - c[1]) * mm + mg[3])
            zt = zt if zt is not None else rng.uniform(350, 750)
            k = 0.0 if rng.random() < 0.5 else rng.choice([-1.0, 1.0]) / rng.uniform(120, 400)
            Rp, T = self._pai_pose(rng, zt, yaw=25, pitch=20, roll=10)
            g = photo.mean(-1, keepdims=True)
            ink = g + rng.uniform(0.55, 1.0) * (photo - g)
            ink = 0.05 + 0.85 * np.clip(ink, 0, 1) ** rng.uniform(0.85, 1.3) * rng.uniform(0.93, 1.04, 3)
            ink = cv2.GaussianBlur(ink.astype(np.float32), (0, 0), 0.6)
            lam, s, t = self._cast_sheet(Rp, T, k)
            inside = (s >= ext[0]) & (s <= ext[1]) & (t >= ext[2]) & (t <= ext[3])
            tex = cv2.remap(ink, (c[0] + s / mm).astype(np.float32), (c[1] + t / mm).astype(np.float32),
                            cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0.9, 0.9, 0.9))
            tex = tex * (1 + self._moire(rng, 0.02, 0.1))[..., None] + rng.normal(0, 0.01, (H, W, 1)).astype(np.float32)
            luma = np.clip(tex @ LUMA / 0.9, 0, 1)
            k_ink = rng.uniform(0.03, 0.3) if rng.random() < 0.5 else rng.uniform(0.4, 0.9)  # dye vs pigment/toner
            alb = rng.uniform(0.7, 0.92) * (1 - k_ink * (1 - luma))
            glossy = rng.random() < 0.35
            mats = [(inside, PAI, tex, 1.0, alb, rng.uniform(0.15, 0.5) if glossy else 0.02,
                     rng.uniform(30, 80) if glossy else 10, np.clip((0.18 - alb) / 0.18, 0, 0.85))]
            meta = dict(subtype="bent" if k else "flat", bend_radius_mm=float(1 / abs(k)) if k else None,
                        iod_on_pai_mm=float(mm * iod_ph), ink_nir_absorption=float(k_ink), glossy=bool(glossy))
        else:
            dev = rng.choice(list(DEVICES), p=[0.5, 0.3, 0.2])
            sw, sh, bz, iod_rng, z_rng, inv_rng = DEVICES[dev]
            iod_mm = rng.uniform(*iod_rng)
            mm = iod_mm / iod_ph
            zt = zt if zt is not None else rng.uniform(*z_rng)
            off = rng.uniform(-0.15, 0.15, 2) * (sw, sh)
            scr = (-sw / 2 + off[0], sw / 2 + off[0], -sh / 2 + off[1], sh / 2 + off[1])
            k = 0.0
            Rp, T = self._pai_pose(rng, zt, yaw=25, pitch=25, roll=10)
            lam, s, t = self._cast_sheet(Rp, T, 0.0)
            in_scr = (s >= scr[0]) & (s <= scr[1]) & (t >= scr[2]) & (t <= scr[3])
            in_dev = (s >= scr[0] - bz) & (s <= scr[1] + bz) & (t >= scr[2] - bz) & (t <= scr[3] + bz)
            black, white = rng.uniform(0.01, 0.06), rng.uniform(0.75, 1.05)
            disp = (black + (white - black) * photo ** rng.uniform(0.8, 1.25)) * \
                (rng.uniform(0.9, 1.0), rng.uniform(0.96, 1.02), rng.uniform(1.0, 1.12))
            tex = cv2.remap(disp.astype(np.float32), (c[0] + s / mm).astype(np.float32),
                            (c[1] + t / mm).astype(np.float32), cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(black,) * 3)
            u, v = self.uv
            gx, gy, gs = rng.uniform(0, W), rng.uniform(0, H), rng.uniform(10, 80)
            glare = rng.uniform(0, 0.35) * np.exp(-((u - gx) ** 2 + (v - gy) ** 2) / (2 * gs ** 2))
            tex = tex * (1 + self._moire(rng, 0.03, 0.15))[..., None] + glare[..., None]
            bez = np.empty((H, W, 3), np.float32)
            bez[:] = rng.uniform(0.02, 0.25)
            mats = [(in_dev & ~in_scr, FRAME, bez, 1.0, rng.uniform(0.03, 0.2), 0.05, 20, 0.05),
                    (in_scr, PAI, tex, 0.0, rng.uniform(0.015, 0.06), rng.uniform(0.6, 3.0), rng.uniform(150, 600),
                     rng.uniform(*inv_rng))]
            meta = dict(subtype=str(dev), iod_on_pai_mm=float(iod_mm))
        self._body(rng, sc, zt + rng.uniform(150, 500))
        front = np.isfinite(lam) & (lam < sc["z"])
        for region, lab, rgb, shade, alb, spec, shin, inval in mats:
            m = front & region
            for key, val in dict(lab=lab, rgb=rgb, shade=shade, alb=alb, spec=spec, shin=shin, inval=inval).items():
                sc[key][m] = val[m] if np.ndim(val) >= 2 else val
            sc["z"][m] = lam[m]
        kps_out = self._project(self._sheet_points(Rp, T, k, (kps - c) * mm))
        sc.update(kps=kps_out, meta=dict(kind=kind, distance_mm=float(zt), **meta))
        return sc

    # ------------------------------------------------------------------ sensor
    def _normals(self, z):
        X = cv2.GaussianBlur(z, (0, 0), 1.0)[..., None] * self.rays
        gy, gx = np.gradient(X, axis=(0, 1))
        n = np.cross(gy, gx)  # towards the camera
        return n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9)

    def _roi(self, kps):
        c, r = np.mean(kps, 0), 0.9 * np.linalg.norm(kps[1] - kps[0]) + 2
        x0, y0 = np.clip((c - r).astype(int), 0, (W - 1, H - 1))
        x1, y1 = np.clip((c + r).astype(int) + 1, 1, (W, H))
        return slice(y0, max(y1, y0 + 1)), slice(x0, max(x1, x0 + 1))

    def _rgb(self, rng, sc, n):
        p = self.p
        L = np.array([rng.uniform(-0.6, 0.6), rng.uniform(-1.0, -0.2), -1.0])
        L /= np.linalg.norm(L)
        amb = rng.uniform(*p["room_ambient"])
        w = sc["shade"]
        img = sc["rgb"] * (1 - w + w * (amb + (1 - amb) * np.clip(n @ L, 0, 1)))[..., None]
        img = cv2.GaussianBlur((img * np.asarray(p["rgb_gain"]) * rng.uniform(0.85, 1.15)).astype(np.float32), (0, 0), 0.6) * 255
        img += rng.standard_normal(img.shape).astype(np.float32) * p["rgb_noise"] * np.sqrt(np.clip(img, 0, 255) / 128 + 0.2)
        return np.clip(np.round(img), 0, 255).astype(np.uint8)

    def _ir(self, rng, sc, z, n):
        p = self.p
        X = z[..., None] * self.rays
        Lv = np.asarray(p["ir_offset"], np.float32) - X
        dist = np.linalg.norm(Lv, axis=-1)
        l = Lv / dist[..., None]
        h = l - X / np.linalg.norm(X, axis=-1, keepdims=True)
        h /= np.linalg.norm(h, axis=-1, keepdims=True)
        fall = (600.0 / dist) ** 2
        spec = sc["spec"] * np.clip((n * h).sum(-1), 0, 1) ** sc["shin"] * fall
        img = sc["alb"] * np.clip((n * l).sum(-1), 0, 1) * fall + spec + p["ir_ambient"] * sc["alb"]
        u, v = self.uv
        for gx, gy, a in sc["glints"]:
            if 0 <= gx < W and 0 <= gy < H:
                y0, y1, x0, x1 = max(int(gy) - 3, 0), min(int(gy) + 4, H), max(int(gx) - 3, 0), min(int(gx) + 4, W)
                img[y0:y1, x0:x1] += a * fall[int(gy), int(gx)] * np.exp(
                    -((u[y0:y1, x0:x1] - gx) ** 2 + (v[y0:y1, x0:x1] - gy) ** 2) / (2 * 0.8 ** 2))
        img = cv2.GaussianBlur((img * self.vignette).astype(np.float32), (0, 0), 0.7)
        g = float(np.clip(p["ir_target"] / max(float(np.median(img[self._roi(sc["kps"])])), 1e-6), 0.25, p["ir_gain_max"]))
        dn = img * g * 255
        dn += rng.standard_normal(dn.shape).astype(np.float32) * np.sqrt((p["ir_read"] * g) ** 2 + p["ir_shot"] * g * np.clip(dn, 0, None))
        return np.clip(np.round(dn), 0, 255).astype(np.uint8), spec * g * 255 > 200, g

    def _depth(self, rng, sc, z, sat):
        p = self.p
        f = p["fx"]
        inv = np.zeros(z.shape, bool)
        for b, sign in ((p["baseline"], 1), (p["d2c_baseline"], -1)):  # stereo shadow left, D2C shadow right
            d = f * b / z
            for k in range(1, int(min(np.ptp(d), 128)) + 1):
                if sign > 0:
                    inv[:, :-k] |= d[:, k:] - d[:, :-k] >= k
                else:
                    inv[:, k:] |= d[:, :-k] - d[:, k:] >= k
        zc = z.copy()
        zmax, zmin = cv2.dilate(z, np.ones((3, 3))), cv2.erode(z, np.ones((3, 3)))
        edge = zmax - zmin > 25
        r = rng.random(z.shape)
        mix = edge & (r < 0.4)  # flying pixels
        zc[mix] = zmin[mix] + rng.random(int(mix.sum())) * (zmax - zmin)[mix]
        inv |= edge & (r > 0.75)
        nz = _smooth_noise(rng, z.shape, 0.8)
        zc = zc + nz * zc ** 2 * p["disp_sigma"] / (f * p["baseline"])
        inv |= ndtr(_smooth_noise(rng, z.shape, 2.0)) < sc["inval"]  # blobby material holes
        inv |= (rng.random(z.shape) < p["hole_rate"]) | sat
        inv |= (z < p["z_range"][0]) | (z > p["z_range"][1])
        return np.where(inv, 0, np.clip(np.round(zc), 1, 65535)).astype(np.uint16)


def _composite(scenes):
    if len(scenes) == 1:
        return scenes[0]
    out = {k: scenes[0][k].copy() for k in ARRAYS}
    out.update(kps=scenes[0]["kps"], meta=dict(scenes[0]["meta"], faces=len(scenes)))
    glints = [(g, 0) for g in scenes[0]["glints"]]
    owner = np.zeros((H, W), np.int32)
    for i, s in enumerate(scenes[1:], 1):
        m = s["z"] < out["z"]
        for k in ARRAYS:
            out[k][m] = s[k][m]
        owner[m] = i
        glints += [(g, i) for g in s["glints"]]
    out["glints"] = [g for g, i in glints if owner[int(np.clip(g[1], 0, H - 1)), int(np.clip(g[0], 0, W - 1))] == i]
    return out
