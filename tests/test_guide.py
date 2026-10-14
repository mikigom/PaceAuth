import math

import pytest

from faceauth.terminal.guide import (CENTER_TOL, CIRCLE_PROMPT, HINTS, ORDER, PROMPTS, RING_TICKS, SLOTS, TARGETS,
                                     TOL_PITCH, TOL_YAW, direction_of, group_of, ring_position)
from faceauth.terminal.guide import Guide as _Guide


def Guide(**kw):  # these tests cover the stepwise prompts (accessibility mode); circle mode is tested at the end
    return _Guide(**{"mode": "steps", **kw})


# ---- bins ------------------------------------------------------------------------------------------------

def test_targets_and_sign_convention():
    # §0: yaw > 0 = subject turns to THEIR left; pitch > 0 = up. Names are from the subject's perspective.
    assert TARGETS == {"C": (0, 0), "L": (25, 0), "R": (-25, 0), "U": (0, 15), "D": (0, -15),
                       "UL": (25, 15), "UR": (-25, 15), "DL": (25, -15), "DR": (-25, -15)}
    assert direction_of(25, 0) == "L" and direction_of(-25, 0) == "R"
    assert direction_of(0, 15) == "U" and direction_of(0, -15) == "D"
    assert direction_of(25, 15) == "UL" and direction_of(-25, 15) == "UR"
    assert direction_of(25, -15) == "DL" and direction_of(-25, -15) == "DR"
    assert direction_of(0, 0) == "C"
    assert set(ORDER) == set(TARGETS) and len(ORDER) == 9 and ORDER[0] == "C"


@pytest.mark.parametrize("name", list(TARGETS))
def test_every_target_centre_maps_to_itself(name):
    assert direction_of(*TARGETS[name]) == name


@pytest.mark.parametrize("yaw,pitch,expected", [
    # centre: |yaw| <= 8, |pitch| <= 6 (inclusive)
    (8, 6, "C"), (-8, -6, "C"), (8.01, 0, None), (0, 6.01, None), (-8.01, 0, None), (0, -6.01, None),
    # L: yaw in [15, 35], pitch in [-7, 7]
    (15, 0, "L"), (14.99, 0, None), (35, 0, "L"), (35.01, 0, None), (25, 7, "L"), (25, -7, "L"),
    (25, 7.01, None), (25, -7.01, None),
    # R mirrors L
    (-15, 0, "R"), (-14.99, 0, None), (-35, 0, "R"), (-35.01, 0, None), (-25, 7, "R"), (-25, 7.01, None),
    # U / D: yaw in [-10, 10], pitch in [8, 22] / [-22, -8]
    (10, 15, "U"), (10.01, 15, None), (-10, 15, "U"), (0, 8, "U"), (0, 7.99, None), (0, 22, "U"),
    (0, 22.01, None), (0, -8, "D"), (0, -7.99, None), (0, -22, "D"), (0, -22.01, None), (-10.01, -15, None),
    # corners
    (15, 8, "UL"), (35, 22, "UL"), (14.99, 15, None), (25, 22.01, None), (-15, 8, "UR"), (-35, 22, "UR"),
    (15, -8, "DL"), (35, -22, "DL"), (-15, -8, "DR"), (-35, -22, "DR"), (-35.01, -22, None),
    # gaps between bins and out of range
    (12, 0, None), (0, 7, None), (12, 7.5, None), (60, 0, None), (0, 40, None), (-90, -45, None), (180, 0, None),
])
def test_bin_boundaries(yaw, pitch, expected):
    assert direction_of(yaw, pitch) == expected


def _matches(yaw, pitch):
    out = []
    for name, (ty, tp) in TARGETS.items():
        ay, ap = CENTER_TOL if name == "C" else (TOL_YAW, TOL_PITCH)
        if abs(yaw - ty) <= ay and abs(pitch - tp) <= ap:
            out.append(name)
    return out


def test_bins_are_disjoint_and_first_match_is_the_only_match():
    for i in range(-240, 241):          # yaw -60..60 step 0.25
        for j in range(-160, 161):      # pitch -40..40 step 0.25
            yaw, pitch = i / 4, j / 4
            m = _matches(yaw, pitch)
            assert len(m) <= 1, (yaw, pitch, m)
            assert direction_of(yaw, pitch) == (m[0] if m else None)


def test_horizontal_flip_swaps_left_and_right():
    swap = {"L": "R", "R": "L", "UL": "UR", "UR": "UL", "DL": "DR", "DR": "DL", "C": "C", "U": "U", "D": "D", None: None}
    for i in range(-160, 161, 3):
        for j in range(-100, 101, 3):
            yaw, pitch = i / 4, j / 4
            assert direction_of(-yaw, pitch) == swap[direction_of(yaw, pitch)]


def test_nan_pose_is_no_direction():
    assert direction_of(math.nan, 0) is None and direction_of(0, math.nan) is None


def test_custom_tolerances():
    assert direction_of(17, 0, tol_yaw=5) is None and direction_of(20, 0, tol_yaw=5) == "L"
    assert direction_of(9, 0, center_tol=(10, 6)) == "C"


# ---- template slots -----------------------------------------------------------------------------------------

def test_group_of_slots():
    assert {d: group_of(d) for d in TARGETS} == {"C": 0, "L": 1, "UL": 1, "DL": 1, "R": 2, "UR": 2, "DR": 2,
                                                 "U": 3, "D": 3}
    assert SLOTS.keys() == TARGETS.keys()
    # slot 1 is the subject's left: every member has yaw > 0 under the §0 convention
    assert all(TARGETS[d][0] > 0 for d in TARGETS if group_of(d) == 1)
    assert all(TARGETS[d][0] < 0 for d in TARGETS if group_of(d) == 2)
    for bad in ("X", "", None, "l"):
        with pytest.raises(KeyError):
            group_of(bad)


# ---- guide state machine -------------------------------------------------------------------------------------

def _feed(g, name, score=1.0, t=0.0, ref=None, ok=True, reasons=()):
    yaw, pitch = TARGETS[name]
    return g.feed((yaw, pitch, 0.0), ok, ref if ref is not None else name, score, t=t, reasons=reasons)


def test_prompt_walks_order_and_status_counts():
    g = Guide()
    s = g.feed(None, False, None, 0.0, t=0.0)
    assert s["current"] == "C" and s["prompt"] == HINTS["no_face"] and not s["accepted"] and s["direction"] is None
    assert PROMPTS["C"] == "정면을 바라봐 주세요"
    assert s["coverage"] == {d: (0, 2) for d in ORDER} and list(s["coverage"]) == list(ORDER)
    for k, name in enumerate(ORDER):
        s = _feed(g, name, t=k)
        assert s["accepted"] and s["direction"] == name and s["coverage"][name] == (1, 2)
        assert s["current"] == name and s["prompt"] == PROMPTS[name]
        s = _feed(g, name, t=k + 0.5)
        assert s["coverage"][name] == (2, 2)
        nxt = ORDER[k + 1] if k + 1 < len(ORDER) else None
        assert s["current"] == nxt
        if nxt:
            assert s["prompt"] == PROMPTS[nxt] and not s["done"]
    assert s["done"] and s["prompt"] == HINTS["done"] and s["current"] is None
    assert PROMPTS["L"] == "고개를 왼쪽으로 천천히 돌려주세요"


def test_opportunistic_collection_of_non_current_bins():
    g = Guide()
    s = _feed(g, "DR")  # prompt is still C, but a DR frame counts
    assert s["accepted"] and s["coverage"]["DR"] == (1, 2) and s["current"] == "C" and s["prompt"] == PROMPTS["C"]


def test_keeps_best_quality_frames_per_direction():
    g = Guide()
    acc = [_feed(g, "C", score=sc, ref=f"c{i}", t=i)["accepted"] for i, sc in enumerate([0.1, 0.5, 0.3, 0.9, 0.2])]
    assert acc == [True, True, True, True, False]  # 0.3 replaces 0.1, 0.9 replaces 0.3, 0.2 loses
    assert sorted(sc for sc, _, _ in g.kept["C"]) == [0.5, 0.9]
    assert [ref for _, ref in g.frames()] == ["c1", "c3"]
    assert not _feed(g, "C", score=0.5, ref="tie", t=9)["accepted"]  # ties keep the earlier frame


def test_rejected_frames_are_not_collected_and_prompt_explains():
    g = Guide()
    s = _feed(g, "C", ok=False, reasons=("blur",))
    assert not s["accepted"] and s["coverage"]["C"] == (0, 2) and s["prompt"] == HINTS["blur"]
    s = g.feed(None, False, None, 0.0, t=0.1, reasons=("multi_face",))
    assert s["prompt"] == HINTS["multi_face"] and s["direction"] is None
    s = _feed(g, "C", ok=False, reasons=("unknown_code",))
    assert s["prompt"] == HINTS["quality_unknown"]
    s = g.feed((12.0, 0.0, 0.0), True, "between", 1.0, t=0.2)  # between C and L: no bin
    assert not s["accepted"] and s["direction"] is None and s["prompt"] == PROMPTS["C"]


def test_frames_near_bin_edges_are_not_kept():
    # 1 degree default safety margin: the server re-estimates pose and must land in the same bin
    g = Guide()
    for yaw, pitch, kept in ((15.5, 0, False), (16.0, 0, True), (25, 6.5, False), (25, 6.0, True), (7.5, 0, False),
                             (7, 5, True), (0, 8.9, False), (9, 9, True)):
        s = g.feed((yaw, pitch, 0.0), True, (yaw, pitch), 1.0, t=0.0)
        assert s["accepted"] is kept and s["direction"] is not None, (yaw, pitch)
    assert Guide(margin=0).feed((15.0, 0.0, 0.0), True, "edge", 1.0, t=0.0)["accepted"]
    assert not Guide(margin=3).feed((17.5, 0.0, 0.0), True, "edge", 1.0, t=0.0)["accepted"]


def test_overshoot_prompt():
    g = Guide()
    for _ in range(2):
        _feed(g, "C")
    assert g.current() == "L"
    assert g.feed((40.0, 0.0, 0.0), True, None, 1.0, t=1)["prompt"] == HINTS["too_far"]
    assert g.feed((-40.0, 0.0, 0.0), True, None, 1.0, t=1)["prompt"] == PROMPTS["L"]  # wrong way: keep asking
    assert g.feed((30.0, 0.0, 0.0), True, None, 1.0, t=1)["prompt"] == PROMPTS["L"]
    g2 = Guide(per_dir=1)
    for name in ("C", "L", "UL", "U", "UR", "R", "DR"):
        _feed(g2, name)
    assert g2.current() == "D"
    assert g2.feed((0.0, -23.0, 0.0), True, None, 1.0, t=1)["prompt"] == HINTS["too_far"]
    assert g2.feed((0.0, 23.0, 0.0), True, None, 1.0, t=1)["prompt"] == PROMPTS["D"]


def test_completion_and_frames_in_capture_order():
    g = Guide()
    seq = ["L", "C", "R", "C", "U", "D", "L", "UL", "UR", "DL", "DR", "R", "U", "D", "UL", "UR", "DL", "DR"]
    for i, name in enumerate(seq):
        s = _feed(g, name, ref=(i, name), t=i)
    assert s["done"] and all(v == (2, 2) for v in s["coverage"].values())
    out = g.frames()
    assert [ref for _, ref in out] == [(i, n) for i, n in enumerate(seq)]  # capture order, not ORDER
    assert all(d == ref[1] for d, ref in out)
    s = _feed(g, "C", score=99.0, t=100)  # after done nothing changes (no timeout either)
    assert not s["accepted"] and s["done"] and not s["timed_out"] and len(g.frames()) == 18


def test_per_dir_parameter():
    g = Guide(per_dir=1)
    for name in ORDER:
        s = _feed(g, name)
    assert s["done"] and len(g.frames()) == 9 and s["coverage"]["C"] == (1, 1)


def test_timeout_is_measured_from_first_frame_and_is_final():
    g = Guide(timeout_s=10.0)
    _feed(g, "C", t=100.0)
    s = _feed(g, "L", t=110.0)  # exactly at the limit: still collecting
    assert s["accepted"] and not s["timed_out"]
    s = _feed(g, "R", t=110.01)
    assert s["timed_out"] and not s["accepted"] and not s["done"] and s["prompt"] == HINTS["timeout"]
    s = _feed(g, "U", t=110.02)
    assert s["timed_out"] and not s["accepted"] and s["coverage"]["U"] == (0, 2)


def test_default_clock_is_used_without_t():
    now = [5.0]
    g = Guide(timeout_s=1.0, clock=lambda: now[0])
    assert g.feed((0, 0, 0), True, "a", 1.0)["accepted"]
    now[0] = 6.5
    assert g.feed((0, 0, 0), True, "b", 1.0)["timed_out"]


# ---- circle mode (default, Face ID-style continuous sweep) --------------------------------------------------

def test_ring_position_puts_targets_evenly_on_the_unit_circle():
    for name, az in (("L", 0), ("UL", 45), ("U", 90), ("UR", 135), ("R", 180), ("DR", 225), ("D", 270), ("DL", 315)):
        a, r = ring_position(*TARGETS[name])
        assert abs((a - az + 180) % 360 - 180) < 1e-6 and abs(r - math.hypot(az % 90 != 0, 1)) < 1e-6, name


def _sweep(g, laps=2.0, fps=10.0, period=5.0, ay=27.0, ap=16.0, hold=1.0, t0=0.0):
    """Look straight for `hold` s, then move the head around an ellipse; returns the last status."""
    n, st = int((hold + laps * period) * fps), None
    for i in range(n):
        t = t0 + i / fps
        th = 2 * math.pi * max(0.0, t - t0 - hold) / period
        pose = (0.0, 0.0, 0.0) if t - t0 < hold else (ay * math.cos(th), ap * math.sin(th), 0.0)
        st = g.feed(pose, True, f"f{i}", 1.0, t=t)
        if st["done"]:
            break
    return st


def test_continuous_sweep_completes_enrollment_and_fills_the_ring():
    g = _Guide()
    assert g.mode == "circle"
    first = g.feed((0.0, 0.0, 0.0), True, "f", 1.0, t=0.0)
    assert first["prompt"] == PROMPTS["C"]                      # look straight first
    st = _sweep(g, t0=0.1)
    assert st["done"] and st["prompt"] == HINTS["done"]
    assert st["ring_frac"] == 1.0 and len(st["ring"]) == RING_TICKS
    assert all(n == 2 for n, _ in g.coverage().values())        # the server's 9 bins are what completed it


def test_after_the_centre_one_instruction_is_given():
    g = _Guide()
    for i in range(3):
        g.feed((0.0, 0.0, 0.0), True, f"c{i}", 1.0, t=i * 0.1)
    assert g.feed((20.0, 5.0, 0.0), True, "x", 1.0, t=0.5)["prompt"] == CIRCLE_PROMPT


def test_small_or_poor_sweeps_light_nothing():
    g = _Guide()
    for i in range(50):                                          # radius 0.4 < RING_R_MIN
        th = 2 * math.pi * i / 50
        g.feed((10 * math.cos(th), 6 * math.sin(th), 0.0), True, f"s{i}", 1.0, t=i * 0.1)
    for i in range(50):                                          # big enough but quality not ok
        th = 2 * math.pi * i / 50
        g.feed((27 * math.cos(th), 16 * math.sin(th), 0.0), False, f"q{i}", 1.0, t=5 + i * 0.1)
    assert sum(g.ring) == 0


def test_ring_fills_between_consecutive_sweep_frames():
    g = _Guide()
    a = g.feed((25.0, 0.0, 0.0), True, "a", 1.0, t=0.0)          # azimuth 0
    y, p = 25 * math.cos(math.radians(30)), 15 * math.sin(math.radians(30))
    b = g.feed((y, p, 0.0), True, "b", 1.0, t=0.1)               # azimuth 30: ticks 0..5 lit
    assert set(range(0, 6)) <= set(b["ring"]) and len(a["ring"]) <= 2


def test_full_ring_with_a_missing_direction_names_it():
    g = _Guide(per_dir=1)
    for i, name in enumerate(ORDER[:-1]):                        # every bin but DL: their arcs light up
        g.feed(TARGETS[name] + (0.0,), True, name, 1.0, t=i * 0.1)
    g.feed((12.5, -13.0, 0.0), True, "a", 1.0, t=1.0)            # azimuth 300, between bins
    st = g.feed((21.65, -7.5, 0.0), True, "b", 1.0, t=1.1)       # azimuth 330, between bins: 300-330 lit
    assert st["ring_frac"] == 1.0 and not st["done"] and st["prompt"] == CIRCLE_PROMPT
    assert g.feed((0.0, 0.0, 0.0), True, "c", 1.0, t=2.0)["prompt"] == CIRCLE_PROMPT
    assert g.feed((0.0, 0.0, 0.0), True, "c", 1.0, t=3.2)["prompt"] == PROMPTS["DL"]


def test_no_progress_names_the_missing_direction():
    g = _Guide(per_dir=1)
    g.feed((0.0, 0.0, 0.0), True, "c", 1.0, t=0.0)               # centre done: the sweep starts
    assert g.feed((5.0, 2.0, 0.0), True, "s", 1.0, t=2.9)["prompt"] == CIRCLE_PROMPT
    assert g.feed((5.0, 2.0, 0.0), True, "s", 1.0, t=3.1)["prompt"] == PROMPTS["L"]
    assert g.feed(TARGETS["L"] + (0.0,), True, "l", 1.0, t=3.2)["prompt"] == CIRCLE_PROMPT  # progress again


def test_better_frames_in_a_full_bin_are_not_progress():
    g = _Guide(per_dir=1)
    g.feed((0.0, 0.0, 0.0), True, "c", 1.0, t=0.0)
    g.feed(TARGETS["L"] + (0.0,), True, "l", 1.0, t=0.1)
    for i in range(40):                                          # keeps finding sharper L frames: stalls anyway
        st = g.feed(TARGETS["L"] + (0.0,), True, f"l{i}", 2.0 + i, t=0.2 + i * 0.1)
    assert st["accepted"] and st["hint_direction"] == "UL"
    assert st["prompt"] == "고개를 돌린 채 턱을 조금 들어주세요"


def test_a_completed_bin_lights_its_whole_arc():
    g = _Guide(per_dir=1)
    st = g.feed(TARGETS["U"] + (0.0,), True, "u", 1.0, t=0.0)
    assert set(range(11, 19)) <= set(st["ring"]) and len(st["ring"]) == 8   # azimuth 90 +- 22.5


@pytest.mark.parametrize("pose", [(math.nan, 0), (0, math.inf), (0, 0, math.nan),
                                  (0,), (None, 0), "000", {}, [0, 0, 0, 0]])
def test_invalid_pose_never_collects_animates_or_leaks_nonfinite_numbers(pose):
    import json
    g = _Guide()
    state = g.feed(pose, True, "invalid", 1, t=0)
    assert state["reason_code"] == "pose_invalid"
    assert state["prompt"] == HINTS["pose_invalid"]
    assert not state["accepted"] and not state["quality_ok"] and not state["motion"]["valid"]
    assert not state["ring"] and not g.frames()
    assert all(value is None for key, value in state["motion"].items() if key != "valid")
    json.dumps(state, allow_nan=False)


@pytest.mark.parametrize("reason", ["multi_face", "blur", "depth_invalid", "ir_invalid"])
def test_quality_rejection_stops_all_pose_feedback_even_with_contradictory_ok(reason):
    g = _Guide()
    state = g.feed((25, 0, 0), True, "invalid", 1, t=0, reasons=(reason,))
    assert state["prompt"] == HINTS[reason] and state["reason_code"] == reason
    assert not state["motion"]["valid"] and not state["quality_ok"]
    assert not state["accepted"] and not state["ring"] and not g.frames()


@pytest.mark.parametrize("pose", [None, (25, 0, 0)])
@pytest.mark.parametrize("reason", ["missing_modality", "decode_error", "unregistered_frame_size", "unexpected_sensor_failure"])
def test_unknown_quality_or_sensor_errors_do_not_prompt_more_head_motion(pose, reason):
    g = _Guide(per_dir=1)
    _feed(g, "C", t=0)
    state = g.feed(pose, False, "bad", 1, t=1, reasons=(reason,))
    expected = reason if reason in HINTS else "quality_unknown"
    assert state["reason_code"] == expected and state["prompt"] == HINTS[expected]
    assert not state["motion"]["valid"] and not state["accepted"]


def test_motion_sequence_is_real_pose_and_overshoot_does_not_grant_ring_or_coverage():
    g = _Guide(per_dir=1)
    first = _feed(g, "C", t=0)
    state = g.feed((43, 3, 0), True, "overshoot", 1, t=3.1)
    assert state["sample_seq"] == first["sample_seq"] + 1 and state["ring_ticks"] == 60
    assert state["motion"]["valid"] and state["motion"]["yaw"] == 43 and state["motion"]["pitch"] == 3
    assert state["phase"] == "refine" and state["hint_direction"] == "L"
    assert state["prompt"] == "고개를 조금 덜 돌려주세요"
    assert state["reason_code"] == "between_directions" and not state["accepted"] and not state["ring"]


def test_real_new_frame_resets_stall_even_after_its_direction_arc_already_lit():
    g = _Guide()
    for index, name in enumerate(("C", "C", "L")):
        _feed(g, name, t=index * .1)
    assert g.progress_t == .2
    g.ring = [True] * RING_TICKS                # no visual tick can mask a missed frame-count update
    assert _feed(g, "L", t=3.1)["phase"] == "sweep"
    assert g.progress_t == 3.1
    _feed(g, "L", score=2, t=3.2)
    assert g.progress_t == 3.1                  # sharper replacement is not new coverage


def test_visual_sweep_does_not_reset_stalled_actual_capture():
    g = _Guide(per_dir=1)
    _feed(g, "C", t=0)
    # Between-bin poses can light new ticks but cannot satisfy the missing directions.
    g.feed((21.65, 7.5, 0), True, "gap1", 1, t=2)
    state = g.feed((12.5, 13, 0), True, "gap2", 1, t=3.1)
    assert state["ring"] and state["phase"] == "refine" and state["hint_direction"] == "L"
    assert g.progress_t == 0 and len(g.frames()) == 1


@pytest.mark.parametrize("pose,expected", [
    ((25, 7.5, 0), "고개를 돌린 채 턱을 조금 내려주세요"),
    ((25, -7.5, 0), "고개를 돌린 채 턱을 조금 들어주세요"),
    ((40, 0, 0), "고개를 조금 덜 돌려주세요"),
    ((12, 0, 0), PROMPTS["L"]),
])
def test_horizontal_refinement_explains_cross_axis_and_too_far_misses(pose, expected):
    g = _Guide(per_dir=1)
    _feed(g, "C", t=0)
    state = g.feed(pose, True, "gap", 1, t=3.1)
    assert state["prompt"] == expected and state["phase"] == "refine"
    assert not state["accepted"] and state["coverage"]["L"] == (0, 1)


def test_quality_loss_breaks_arc_interpolation_and_timeout_cannot_paint():
    g = _Guide(timeout_s=2)
    g.feed((25, 0, 0), True, "a", 1, t=0)
    before = set(g.feed(None, False, t=.1, reasons=("no_face",))["ring"])
    state = g.feed((25 * math.cos(math.radians(30)), 15 * math.sin(math.radians(30)), 0), True, "b", 1, t=.2)
    assert 2 not in state["ring"]                 # no imagined bridge over the missing face
    before = set(state["ring"])
    state = g.feed((-25, 0, 0), True, "late", 1, t=3)
    assert set(state["ring"]) == before and state["phase"] == "timeout"
    assert not state["motion"]["valid"] and not state["accepted"]


def test_complete_requires_all_eighteen_frames_even_when_visual_ring_is_full():
    g = _Guide()
    g.ring = [True] * RING_TICKS
    for name in ORDER:
        state = _feed(g, name, t=0)
    assert state["ring_frac"] == 1 and not state["done"] and len(g.frames()) == 9
    for name in ORDER:
        state = _feed(g, name, t=.1)
    assert state["done"] and state["phase"] == "complete" and len(g.frames()) == 18
    assert not state["motion"]["valid"]
