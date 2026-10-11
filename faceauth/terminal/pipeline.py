"""Per-frame terminal processing (Design §1 terminal layer): one face -> head pose -> landmarks -> quality.

Feeds the enrollment Guide and picks the best frames of a short authentication sequence. Models follow the
docs/IMPLEMENTATION.md §2 interfaces (SCRFD, HeadPose, FaRLLandmarks). These thresholds are terminal-side
prefilters for guidance only; the engine re-verifies every frame (FR-03).
"""
from pathlib import Path

import cv2
import numpy as np

from faceauth.common.packet import decode_frame, encode_frame

MIN_FACE_PX = 60.0            # bbox short side; ~80 cm away with the 640x480 sim profile
BLUR_MIN = 40.0               # Laplacian variance of the face crop resized to 112x112 grey
EXPOSURE = (40.0, 220.0)      # mean grey level of that crop
EAR_MIN = 0.06                # eye aspect ratio (WFLW-98, smaller eye): only clearly closed eyes. 0.12 rejected open
                              # narrow eyes (measured 0.09-0.19 frontal on BIWI subjects): a fairness problem
EAR_MIN_PITCH = -8.0          # looking down foreshortens the lids (EAR 0.07-0.12 while open): no eye check below this
AUTH_MAX_POSE = (20.0, 15.0)  # |yaw|, |pitch| accepted for authentication frames
RELEASE_PUB = Path(__file__).resolve().parents[2] / "secrets" / "release.pub"
# WFLW-98 eyes: (corner, corner), (upper, lower) lid pairs; image-left eye 60-67, image-right eye 68-75
EYES = (((60, 64), ((61, 67), (62, 66), (63, 65))), ((68, 72), ((69, 75), (70, 74), (71, 73))))


def eye_aspect_ratio(lmk98) -> float:
    """Smaller eye's mean lid gap / eye width (eyes-visible proxy: closed or occluded eyes score low)."""
    p = np.asarray(lmk98, dtype=np.float64)
    ears = [np.mean([np.linalg.norm(p[i] - p[j]) for i, j in lids]) / max(np.linalg.norm(p[a] - p[b]), 1e-6)
            for (a, b), lids in EYES]
    return float(min(ears))


def wire_frame(rgb, depth, ir) -> tuple:
    """Encode a captured frame once for the packet and return (encoded, decoded rgb). Analysing the decoded rgb
    makes the terminal see exactly the pixels the engine will see: JPEG alone moves the pose estimate by up to
    ~3 deg yaw / ~7 deg pitch (measured), enough to put a frame in another direction bin server-side."""
    enc = encode_frame(rgb, depth, ir)
    return enc, decode_frame(enc)[0]


def face_crop_stats(rgb, bbox) -> tuple:
    """(Laplacian variance, mean grey) of the face box clipped to the image, resized to 112x112."""
    h, w = rgb.shape[:2]
    x1, y1 = max(int(bbox[0]), 0), max(int(bbox[1]), 0)
    x2, y2 = min(int(np.ceil(bbox[2])), w), min(int(np.ceil(bbox[3])), h)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return 0.0, 0.0
    g = cv2.resize(cv2.cvtColor(np.ascontiguousarray(rgb[y1:y2, x1:x2]), cv2.COLOR_RGB2GRAY), (112, 112))
    return float(cv2.Laplacian(g, cv2.CV_64F).var()), float(g.mean())


class Pipeline:
    def __init__(self, detector, headpose, landmarks, det_thresh=0.5, min_face=MIN_FACE_PX, blur_min=BLUR_MIN,
                 exposure=EXPOSURE, ear_min=EAR_MIN):
        self.detector, self.headpose, self.landmarks = detector, headpose, landmarks
        self.det_thresh, self.min_face, self.blur_min, self.exposure, self.ear_min = \
            det_thresh, min_face, blur_min, exposure, ear_min

    # detector = the server engine's active one (scrfd_34g@2, 320 input): same boxes -> the server's blur / pose-bin
    # re-measurements agree with what the guide accepted (with different detectors, E8 lost frames near thresholds)
    @classmethod
    def load(cls, models_dir="data/models", device="cuda", detector="scrfd_34g_320", headpose="6drepnet360",
             landmarks="farl_wflw_448", release_pub=RELEASE_PUB, verify=True, **kw):
        """verify (§7): the signed registry and every file hash are checked before any load; RegistryError
        (tampered, unsigned or unlisted model) refuses to start. verify=False is for dev runs without a registry."""
        d = Path(models_dir)
        if verify:
            from faceauth.common.registry import RegistryError, load_registry
            signed = {Path(e["path"]).name for e in load_registry(d, release_pub)}
            missing = sorted({detector, headpose, landmarks} - signed)
            if missing:
                raise RegistryError(f"not in the signed registry: {missing}")
        from faceauth.models.farl import FaRLLandmarks
        from faceauth.models.headpose import HeadPose
        from faceauth.models.scrfd import SCRFD
        return cls(SCRFD(d / detector), HeadPose(d / headpose, device=device),
                   FaRLLandmarks(d / landmarks, device=device), **kw)

    def analyze(self, rgb) -> dict:
        """One RGB frame -> {"ok", "reasons", "n_faces", "bbox", "kps", "det_score", "pose", "quality", "score"}.
        Exactly one face of short side >= min_face is required; any other face >= min_face / 2 => multi_face."""
        r = {"ok": False, "reasons": [], "n_faces": 0, "bbox": None, "kps": None, "det_score": 0.0, "pose": None,
             "quality": {}, "score": 0.0}
        dets = self.detector.detect(rgb, thresh=self.det_thresh)
        sizes = [min(d["bbox"][2] - d["bbox"][0], d["bbox"][3] - d["bbox"][1]) for d in dets]
        r["n_faces"] = sum(s >= self.min_face / 2 for s in sizes)
        if not dets:
            r["reasons"].append("no_face")
            return r
        if r["n_faces"] > 1:
            r["reasons"].append("multi_face")
            return r
        i = int(np.argmax(sizes))
        if sizes[i] < self.min_face:
            r["reasons"].append("small_face")
            return r
        bbox, kps = np.asarray(dets[i]["bbox"], np.float32), np.asarray(dets[i]["kps"], np.float32)
        r.update(bbox=bbox, kps=kps, det_score=float(dets[i]["score"]))
        r["pose"] = [float(v) for v in self.headpose.predict(rgb, bbox[None])[0]]
        lmk = self.landmarks.predict(rgb, kps[None])[0]
        lap, mean = face_crop_stats(rgb, bbox)
        q = r["quality"] = {"face_px": float(sizes[i]), "blur": lap, "brightness": mean, "ear": eye_aspect_ratio(lmk)}
        if q["blur"] < self.blur_min:
            r["reasons"].append("blur")
        if not self.exposure[0] <= q["brightness"] <= self.exposure[1]:
            r["reasons"].append("exposure")
        if q["ear"] < self.ear_min and r["pose"][1] >= EAR_MIN_PITCH:
            r["reasons"].append("eyes_closed")
        r["ok"] = not r["reasons"]
        r["score"] = r["det_score"] * min(1.0, q["blur"] / (3 * self.blur_min))  # ranks frames; saturates when sharp
        return r

    def feed(self, guide, rgb, frame_ref, t=None) -> tuple:
        """Enrollment step: analyze one frame and feed it to the Guide -> (analysis, guide status)."""
        r = self.analyze(rgb)
        return r, guide.feed(r["pose"], r["ok"], frame_ref, r["score"], t=t, reasons=r["reasons"])


def select_auth(results, n=3, max_pose=AUTH_MAX_POSE) -> list:
    """Indices of the n best ok, near-frontal frames of a short sequence, in capture order; [] if fewer than n."""
    good = [i for i, r in enumerate(results)
            if r["ok"] and abs(r["pose"][0]) <= max_pose[0] and abs(r["pose"][1]) <= max_pose[1]]
    best = sorted(good, key=lambda i: -results[i]["score"])[:n]
    return sorted(best) if len(best) == n else []
