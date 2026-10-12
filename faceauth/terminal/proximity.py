"""Hands-free authentication trigger (kiosk UX): authenticate as soon as one person is in the capture range.

RFP §1 assumes cooperative use at 40-80 cm. The depth sensor measures the face distance directly, so "close enough" is
a measurement, not a guess from face size. The trigger fires after `hold` consecutive analysed frames with exactly
one face inside RANGE_MM, a near-frontal pose and passing quality; the user presses nothing. The terminal then captures
the authentication frames after the session/challenge exists (Design §3 order), so a pre-recorded sequence cannot be
replayed into a session issued later.
"""
import numpy as np

from faceauth.terminal.pipeline import AUTH_MAX_POSE

RANGE_MM = (400.0, 800.0)
HOLD = 3
HINTS = {
    "idle": "화면 가까이 오시면 자동으로 인식합니다",
    "far": "조금 더 가까이 와 주세요",
    "too_close": "조금 뒤로 물러나 주세요",
    "adjust": "정면을 바라봐 주세요",
    "multi": "한 분만 화면에 들어와 주세요",
    "ready": "인식 중입니다",
}


def face_distance_mm(depth: np.ndarray, bbox) -> float | None:
    """Median valid depth (mm) in the central half of the face box; None if under 30% of it is valid."""
    x1, y1, x2, y2 = (float(v) for v in bbox)
    cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) / 2, (y2 - y1) / 2
    H, W = depth.shape
    r = depth[max(int(cy - h / 2), 0):min(int(cy + h / 2), H), max(int(cx - w / 2), 0):min(int(cx + w / 2), W)]
    v = r[r > 0]
    return float(np.median(v)) if r.size and v.size >= 0.3 * r.size else None


class ProximityTrigger:
    def __init__(self, range_mm=RANGE_MM, hold: int = HOLD, max_pose=AUTH_MAX_POSE):
        self.range_mm, self.hold, self.max_pose = range_mm, hold, max_pose
        self.streak = 0

    def feed(self, analysis: dict, depth: np.ndarray) -> dict:
        """analysis = terminal Pipeline.analyze() of one frame -> {"state", "distance_mm", "hint", "fire"}."""
        dist = face_distance_mm(depth, analysis["bbox"]) if analysis.get("bbox") is not None else None
        if analysis["n_faces"] > 1:
            state = "multi"
        elif analysis["n_faces"] == 0 or dist is None:
            state = "idle"
        elif dist > self.range_mm[1]:
            state = "far"
        elif dist < self.range_mm[0]:
            state = "too_close"
        elif not (analysis["ok"] and abs(analysis["pose"][0]) <= self.max_pose[0]
                  and abs(analysis["pose"][1]) <= self.max_pose[1]):
            state = "adjust"
        else:
            state = "ready"
        self.streak = self.streak + 1 if state == "ready" else 0
        fire = self.streak == self.hold          # fires once per approach; reset() after the attempt
        return {"state": state, "distance_mm": dist, "hint": HINTS[state], "fire": fire}

    def reset(self):
        self.streak = 0


if __name__ == "__main__":  # self-check
    d = np.full((480, 640), 600, np.uint16)
    good = {"n_faces": 1, "bbox": [270, 190, 370, 310], "ok": True, "pose": [3.0, -2.0, 0.0]}
    t = ProximityTrigger()
    assert [t.feed(good, d)["fire"] for _ in range(4)] == [False, False, True, False]
    t.reset()
    assert t.feed(good, np.full((480, 640), 1200, np.uint16))["state"] == "far"
    assert t.feed(good, np.full((480, 640), 300, np.uint16))["state"] == "too_close"
    assert t.feed(dict(good, pose=[30.0, 0, 0]), d)["state"] == "adjust"
    assert t.feed(dict(good, n_faces=2), d)["state"] == "multi"
    assert t.feed({"n_faces": 0, "bbox": None, "ok": False, "pose": None}, d)["state"] == "idle"
    assert face_distance_mm(np.zeros((480, 640), np.uint16), good["bbox"]) is None
    print("proximity ok")
