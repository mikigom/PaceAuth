"""Enrollment rotation guide: direction bins, template slots and the per-frame guidance state machine.

Pure logic (no I/O, stdlib only) so the API can import the same bins for server-side coverage.
Pose convention (docs/IMPLEMENTATION.md §0): yaw > 0 = subject turns to THEIR left, pitch > 0 = looking up.
Direction names are from the subject's perspective: "L" = the subject's left = yaw +25.
"""
import math
import time

TARGETS = {  # name -> (yaw, pitch) degrees; Design §2 initial range yaw ±25°, pitch ±15°
    "C": (0.0, 0.0),
    "L": (25.0, 0.0), "R": (-25.0, 0.0), "U": (0.0, 15.0), "D": (0.0, -15.0),
    "UL": (25.0, 15.0), "UR": (-25.0, 15.0), "DL": (25.0, -15.0), "DR": (-25.0, -15.0),
}
ORDER = ("C", "L", "UL", "U", "UR", "R", "DR", "D", "DL")  # prompt order: centre, then one sweep around
TOL_YAW, TOL_PITCH, CENTER_TOL = 10.0, 7.0, (8.0, 6.0)   # bins are disjoint with these values
SLOTS = {"C": 0, "L": 1, "UL": 1, "DL": 1, "R": 2, "UR": 2, "DR": 2, "U": 3, "D": 3}  # spec §6 template slots

PROMPTS = {
    "C": "정면을 바라봐 주세요",
    "L": "고개를 왼쪽으로 천천히 돌려주세요",
    "R": "고개를 오른쪽으로 천천히 돌려주세요",
    "U": "고개를 위로 천천히 들어주세요",
    "D": "고개를 아래로 천천히 숙여주세요",
    "UL": "고개를 왼쪽 위로 천천히 돌려주세요",
    "UR": "고개를 오른쪽 위로 천천히 돌려주세요",
    "DL": "고개를 왼쪽 아래로 천천히 돌려주세요",
    "DR": "고개를 오른쪽 아래로 천천히 돌려주세요",
}
# Face ID-style continuous enrollment (default): one instruction, a progress ring that fills as the head sweeps around.
CIRCLE_PROMPT = "고개를 천천히 돌려 원을 완성해 주세요"
RING_TICKS = 60        # one tick per 6 degrees of head-direction azimuth
RING_R_MIN = 0.85      # normalised sweep radius (target ellipse = 1.0) that lights the ring: wide enough for a bin
RING_GAP_MAX = 40.0    # fill the ring between consecutive sweep frames at most this far apart (deg)
HINT_AFTER_S = 2.0     # ring >= 90% but a direction still missing for this long -> name that direction
STALL_S = 3.0          # no new tick and no new frame for this long -> name the missing direction

HINTS = {  # keys = terminal/engine reason codes (+ guide states)
    "no_face": "얼굴이 화면 안에 들어오도록 해주세요",
    "multi_face": "한 분만 화면에 들어와 주세요",
    "small_face": "카메라에 조금 더 가까이 와주세요",
    "blur": "잠시 움직임을 멈춰주세요",
    "exposure": "조명이 너무 어둡거나 밝습니다",
    "eyes_closed": "눈을 뜨고 화면을 바라봐 주세요",
    "too_far": "조금만 덜 움직여주세요",
    "done": "얼굴 촬영이 완료되었습니다",
    "timeout": "시간이 초과되었습니다. 처음부터 다시 시도해주세요",
}


def direction_of(yaw: float, pitch: float, tol_yaw: float = TOL_YAW, tol_pitch: float = TOL_PITCH,
                 center_tol: tuple = CENTER_TOL) -> str | None:
    """Name of the bin containing (yaw, pitch) (inclusive bounds), or None between bins / out of range / NaN."""
    for name, (ty, tp) in TARGETS.items():
        ay, ap = center_tol if name == "C" else (tol_yaw, tol_pitch)
        if abs(yaw - ty) <= ay and abs(pitch - tp) <= ap:
            return name
    return None


def group_of(direction: str) -> int:
    """Template slot: 0 C, 1 subject-left (L, UL, DL), 2 subject-right (R, UR, DR), 3 vertical (U, D). KeyError if unknown."""
    return SLOTS[direction]


def ring_position(yaw: float, pitch: float) -> tuple:
    """(azimuth deg in [0, 360), normalised radius). Pitch is scaled by 25/15 so the yaw +-25 / pitch +-15 target
    ellipse becomes the unit circle: L at 0, UL 45, U 90, UR 135, R 180, DR 225, D 270, DL 315 (subject's view)."""
    y, p = yaw / 25.0, pitch / 15.0
    return math.degrees(math.atan2(p, y)) % 360.0, math.hypot(y, p)


def _overshoot(yaw: float, pitch: float, target: str) -> bool:
    """Pose went past the target's bin along the target's own axis (e.g. yaw 40 while aiming for L)."""
    ty, tp = TARGETS[target]
    sy, sp = (ty > 0) - (ty < 0), (tp > 0) - (tp < 0)  # direction of the target along each axis (0 = none)
    return bool((sy and sy * yaw > abs(ty) + TOL_YAW) or (sp and sp * pitch > abs(tp) + TOL_PITCH))


class Guide:
    """Feed one frame at a time; keeps the `per_dir` best-quality frames of every direction bin.

    Frames are collected opportunistically (any bin that is reached counts), but only `margin` degrees inside a
    bin, so the server's re-estimate lands in the same bin. Feed poses estimated on the frame as sent
    (pipeline.wire_frame); the margin then only absorbs numeric / batching differences.
    Done when every bin holds `per_dir` frames; timed out (final) when `timeout_s` passes from the first frame.

    mode "circle" (default, Face ID-style): look straight first, then one instruction - move the head slowly around
    - while a RING_TICKS progress ring fills wherever the sweep passed (quality ok, radius >= RING_R_MIN) and over
    the whole arc of every completed bin. The bins are still what completes enrollment; if the ring is nearly full
    or nothing new lit up for STALL_S while a bin is missing, the prompt names that direction.
    mode "steps" (accessibility option): the prompt walks ORDER one direction at a time.
    """

    def __init__(self, per_dir: int = 2, timeout_s: float = 60.0, margin: float = 1.0, clock=time.monotonic,
                 mode: str = "circle"):
        if mode not in ("circle", "steps"):
            raise ValueError(f"unknown guide mode {mode!r}")
        self.per_dir, self.timeout_s, self.margin, self.clock, self.mode = per_dir, timeout_s, margin, clock, mode
        self.kept = {d: [] for d in ORDER}  # direction -> [(score, seq, frame_ref)]
        self.seq, self.t0, self.timed_out = 0, None, False
        self.ring = [False] * RING_TICKS
        self.last_az, self.nearly_full_since, self.progress_t = None, None, None

    @property
    def done(self) -> bool:
        return all(len(v) >= self.per_dir for v in self.kept.values())

    def current(self) -> str | None:
        return next((d for d in ORDER if len(self.kept[d]) < self.per_dir), None)

    def coverage(self) -> dict:
        """FR-01 display: direction -> (collected, required)."""
        return {d: (len(self.kept[d]), self.per_dir) for d in ORDER}

    def feed(self, pose, quality_ok: bool, frame_ref=None, quality_score: float = 0.0, *, t: float | None = None,
             reasons=()) -> dict:
        """pose = (yaw, pitch[, roll]) or None (no single face). reasons = rejection codes, used for the prompt."""
        t = self.clock() if t is None else t
        self.t0 = t if self.t0 is None else self.t0
        seq, self.seq = self.seq, self.seq + 1
        direction = None if pose is None else direction_of(pose[0], pose[1])
        accepted = False
        if not (self.done or self.timed_out):
            if t - self.t0 > self.timeout_s:
                self.timed_out = True
            elif quality_ok and direction is not None and self._inside(pose, direction):
                accepted = self._keep(direction, float(quality_score), seq, frame_ref)
            lit, have = sum(self.ring), sum(min(len(v), self.per_dir) for v in self.kept.values())
            self._sweep(pose, quality_ok)
            if direction not in (None, "C") and len(self.kept[direction]) >= self.per_dir:
                self._light(ring_position(*TARGETS[direction])[0], 22.5)  # a completed bin owns its arc
            if self.done:
                self.ring = [True] * RING_TICKS                         # the bins decide; the ring closes with them
            if (sum(self.ring) > lit or sum(min(len(v), self.per_dir) for v in self.kept.values()) > have
                    or self.progress_t is None):                       # new coverage only: replacing a kept frame
                self.progress_t = t                                     # with a better one is not progress
        frac = sum(self.ring) / RING_TICKS
        if frac >= 0.9 and not self.done:
            self.nearly_full_since = t if self.nearly_full_since is None else self.nearly_full_since
        return {"coverage": self.coverage(), "direction": direction, "accepted": accepted, "current": self.current(),
                "prompt": self._prompt(pose, quality_ok, reasons, t), "done": self.done, "timed_out": self.timed_out,
                "ring": [i for i, lit in enumerate(self.ring) if lit], "ring_frac": round(frac, 3)}

    def _sweep(self, pose, quality_ok):
        """Light the ring where the head direction passed (between consecutive sweep frames when close enough)."""
        if pose is None or not quality_ok:
            self.last_az = None
            return
        az, r = ring_position(pose[0], pose[1])
        if r < RING_R_MIN:
            self.last_az = None
            return
        lo, hi = az - 180.0 / RING_TICKS, az + 180.0 / RING_TICKS
        if self.last_az is not None:
            d = (az - self.last_az + 180.0) % 360.0 - 180.0            # signed shortest angular move
            if abs(d) <= RING_GAP_MAX:
                lo, hi = min(lo, az - d), max(hi, az)
        self._light((lo + hi) / 2, (hi - lo) / 2)
        self.last_az = az

    def _light(self, centre, half):
        """Light every tick whose centre lies within `half` degrees of azimuth `centre`."""
        step = 360.0 / RING_TICKS
        for i in range(RING_TICKS):
            if abs(((i + 0.5) * step - centre + 180.0) % 360.0 - 180.0) <= half + step / 2 - 1e-9:
                self.ring[i] = True

    def frames(self) -> list:
        """Kept frames in capture order (packets must list frames in capture order) -> [(direction, frame_ref)]."""
        items = sorted(((seq, d, ref) for d, kept in self.kept.items() for _, seq, ref in kept), key=lambda x: x[0])
        return [(d, ref) for _, d, ref in items]

    def _inside(self, pose, direction) -> bool:
        m = self.margin
        shrunk = direction_of(pose[0], pose[1], TOL_YAW - m, TOL_PITCH - m, (CENTER_TOL[0] - m, CENTER_TOL[1] - m))
        return shrunk == direction

    def _keep(self, direction, score, seq, ref) -> bool:
        kept = self.kept[direction]
        if len(kept) < self.per_dir:
            kept.append((score, seq, ref))
            return True
        worst = min(range(len(kept)), key=lambda i: kept[i][0])
        if score > kept[worst][0]:
            kept[worst] = (score, seq, ref)
            return True
        return False

    def _prompt(self, pose, quality_ok, reasons, t=None) -> str:
        if self.timed_out:
            return HINTS["timeout"]
        if self.done:
            return HINTS["done"]
        hint = next((HINTS[r] for r in reasons if r in HINTS), None)
        if pose is None:
            return hint or HINTS["no_face"]
        if not quality_ok and hint:
            return hint
        cur = self.current()
        if self.mode == "circle":
            if len(self.kept["C"]) < self.per_dir:
                return PROMPTS["C"]
            if t is not None and ((self.nearly_full_since is not None and t - self.nearly_full_since >= HINT_AFTER_S)
                                  or t - self.progress_t >= STALL_S):
                return PROMPTS[cur]                     # ring ~full or no progress, a bin still missing: name it
            return CIRCLE_PROMPT
        return HINTS["too_far"] if _overshoot(pose[0], pose[1], cur) else PROMPTS[cur]
