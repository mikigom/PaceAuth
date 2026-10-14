"""No live label may survive stale frames, a changed identity, or a capture operation."""
import copy
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from faceauth.terminal.live_recognition import FaceTracks, LiveRecognition, MAX_AGE_S


def face(account="alice", bbox=None, status="known"):
    return {"bbox": bbox or [10, 5, 30, 35], "status": status, "account_id": account, "reason": "ok"}


def test_names_need_fresh_agreement_and_cannot_transfer_when_people_cross():
    tracker = FaceTracks()
    names = {"alice": "앨리스", "bob": "밥"}
    left, right = face(), face("bob", [35, 5, 60, 35])
    assert all(f["status"] == "uncertain" for f in tracker.update([left, right], names))
    confirmed = tracker.update([right, left], names)  # Detector order changed, actual positions did not.
    assert [f["display_name"] for f in confirmed] == ["밥", "앨리스"]
    ids = [f["track_id"] for f in confirmed]
    swapped = [face("alice", right["bbox"]), face("bob", left["bbox"])]
    waiting = tracker.update(swapped, names)
    assert [f["track_id"] for f in waiting] == ids
    assert all(f["status"] == "uncertain" and f["display_name"] == "확인 중" for f in waiting)
    assert [f["display_name"] for f in tracker.update(swapped, names)] == ["앨리스", "밥"]
    unknown = tracker.update([face(None, right["bbox"], "unknown")], names)
    assert unknown[0]["display_name"] == "미등록" and unknown[0]["status"] == "unknown"
    assert tracker.update([], names) == [] and tracker.tracks == {}
    assert tracker.update([left], names)[0]["status"] == "uncertain"


@pytest.fixture
def worker():
    camera_state = {"camera": {"state": "streaming", "info": {"calib_version": "fixture"}}}
    camera = SimpleNamespace(status=lambda: copy.deepcopy(camera_state))
    accounts = SimpleNamespace(names=lambda ids: {"alice": "테스트 이름"})
    auth = SimpleNamespace(enabled=True, lock=threading.RLock(), active=None, accounts=accounts)
    instance = LiveRecognition(camera, auth)
    instance.camera_state = camera_state
    return instance


def packet():
    return {"demo": True, "calib_version": "fixture", "image_size": [64, 48], "faces": [face()],
            "pad_enabled": False, "pad_mode": "disabled_demo", "truncated": False}


def capture(age=0):
    return 42, time.monotonic() - age, {"depth": np.ones((48, 64), np.uint16),
                                       "calib_version": "fixture", "captured_at_ms": 12345}


def test_live_output_is_bounded_and_stale_results_are_cleared(worker):
    out = packet()
    out["faces"][0].update(emb=b"private", score=0.99)
    worker._publish(out, capture(), time.monotonic())
    worker._publish(out, capture(), time.monotonic())
    state = worker.snapshot()
    assert state["faces"][0]["display_name"] == "테스트 이름"
    assert state["sequence"] == 42 and state["captured_at_ms"] == 12345
    assert state["faces"][0]["primary"] is True
    assert not {"emb", "score", "account_id"}.intersection(state["faces"][0])
    assert state["authentication_approved"] is False
    worker.arrived_at -= MAX_AGE_S + 1
    assert worker.snapshot()["faces"] == []
    worker._publish(out, capture(MAX_AGE_S + 1), time.monotonic())
    assert worker.snapshot()["state"] == "unavailable" and not worker.tracks.tracks


def test_disconnect_and_operation_start_hide_labels_even_during_an_inflight_request(worker):
    worker._publish(packet(), capture(), time.monotonic())
    worker._publish(packet(), capture(), time.monotonic())
    worker.auth.active = "new-operation"
    assert worker.snapshot()["state"] == "paused" and worker.snapshot()["faces"] == []
    worker.auth.active = None
    worker.camera_state["camera"]["state"] = "error"
    assert worker.snapshot()["state"] == "unavailable" and worker.snapshot()["faces"] == []


@pytest.mark.parametrize("change", [
    lambda p: p.update(demo=False),
    lambda p: p.update(calib_version="different"),
    lambda p: p.update(image_size=[320, 240]),
    lambda p: p.update(approval_token="not-a-demo"),
    lambda p: p.update(pad_enabled=True),
    lambda p: p.update(faces=[face()] * 17),
    lambda p: p["faces"][0].update(bbox=[-1, 0, 10, 20]),
    lambda p: p["faces"][0].update(bbox=[0, 0, float("nan"), 20]),
    lambda p: p["faces"][0].update(status="approved"),
])
def test_malformed_or_unbounded_results_never_display(change):
    out = packet()
    change(out)
    with pytest.raises(ValueError):
        LiveRecognition._validate(out, "fixture", (48, 64))


def test_unknown_largest_face_is_primary_instead_of_a_smaller_known_person():
    out = packet()
    out["faces"] = [face(), face(None, [0, 0, 64, 48], "unknown")]
    faces = LiveRecognition._validate(out, "fixture", (48, 64))
    assert not faces[0].get("primary") and faces[1]["primary"]


def test_account_disable_delete_and_rename_apply_to_cached_live_snapshot(worker):
    worker._publish(packet(), capture(), time.monotonic())
    worker._publish(packet(), capture(), time.monotonic())
    assert worker.snapshot()["faces"][0]["status"] == "known"
    worker.auth.accounts.names = lambda ids: {"alice": "새 이름"}
    assert worker.snapshot()["faces"][0]["display_name"] == "새 이름"
    worker.auth.accounts.names = lambda ids: {}
    hidden = worker.snapshot()["faces"][0]
    assert hidden["status"] == "unknown" and hidden["display_name"] == "미등록"
    # An already-running inference must not republish the disabled/deleted name.
    worker._publish(packet(), capture(), time.monotonic())
    assert worker.snapshot()["faces"][0]["status"] == "unknown"
