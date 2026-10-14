"""Real-camera CLI routing, metadata integrity and ownership (no device access)."""
import builtins
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from faceauth.terminal.guide import ORDER, TARGETS
from scripts import terminal_demo as demo


CALIBRATION = "femto-bolt-v1-test-device"


def args(**changes):
    values = dict(source="orbbec", fps=5, serial="camera-1", frame_timeout_ms=10000,
                  auth_timeout=20, timeout=60, auth_max_frames=30, calib_version=CALIBRATION,
                  models="models", torch_device="cpu", unsigned_models=False,
                  gateway="https://gateway:8443", device="real-device", account="demo-account",
                  action="door-open", register=False, skip_enroll=True, fallback=False,
                  consent_version="v1", re_enroll=False)
    values.update(changes)
    return SimpleNamespace(**values)


def capture(index=0):
    timestamp = 1791900000000 + index * 213
    return {"rgb": np.full((4, 4, 3), 100 + index, np.uint8),
            "depth": np.full((4, 4), 600, np.uint16), "ir": np.full((4, 4), 120, np.uint8),
            "ts_ms": {"rgb": timestamp, "depth": timestamp + 2, "ir": timestamp - 1},
            "captured_at_ms": timestamp, "calib_version": CALIBRATION}


class FakeSensor:
    def __init__(self, events, frames):
        self.events, self.frames = events, list(frames)
        self.calib_version = CALIBRATION
        self.started = False

    def start(self):
        self.events.append("sensor.start")
        self.started = True
        return self

    def read(self, timeout_ms):
        assert self.started
        self.events.append(("sensor.read", timeout_ms))
        return self.frames.pop(0)

    def stop(self):
        self.events.append("sensor.stop")
        self.started = False


class FakeClient:
    def __init__(self, events):
        self.events = events
        self.auth_packet = self.enroll_packet = None

    def create_session(self, purpose, action):
        self.events.append(("session", purpose))
        return {"session_id": purpose + "-session", "challenge": purpose + "-challenge"}

    def authenticate(self, session, frames, ts, captured_at_ms):
        self.auth_packet = (session, frames, ts, captured_at_ms)
        self.events.append("authenticate")
        return {"decision": "rejected", "reason": "test"}

    def enroll(self, session, frames, ts, labels, assertion, consent, **kwargs):
        self.enroll_packet = (session, frames, ts, labels, assertion, consent, kwargs)
        self.events.append("enroll")
        return {"status": "active"}

    def close(self):
        self.events.append("client.close")


class FakePipeline:
    def analyze(self, rgb):
        return {"ok": True, "pose": [0, 0, 0], "score": 1.0}


def forbid_synthetic_timestamps(*_args, **_kwargs):
    raise AssertionError("Real camera capture must not fabricate timestamps")


def test_authentication_starts_after_challenge_keeps_actual_timestamps_and_closes(monkeypatch):
    events, inputs = [], [capture(i) for i in range(3)]
    sensor, client = FakeSensor(events, inputs), FakeClient(events)
    options = args(sensor=sensor)
    monkeypatch.setattr(demo, "synced_ts", forbid_synthetic_timestamps)
    assert demo.authenticate(options, client, FakePipeline()) == "rejected"
    assert events[0:2] == [("session", "auth"), "sensor.start"]
    assert events.index("sensor.stop") < events.index("authenticate")
    assert client.auth_packet[2] == [f["ts_ms"] for f in inputs]
    assert client.auth_packet[3] == inputs[0]["captured_at_ms"]
    assert all(0 < e[1] <= options.frame_timeout_ms for e in events if isinstance(e, tuple) and e[0] == "sensor.read")
    assert not sensor.started


def test_enrollment_preserves_metadata_in_capture_order_and_real_guide_clock(monkeypatch):
    events, inputs = [], [capture(i) for i in range(18)]
    sensor, client = FakeSensor(events, inputs), FakeClient(events)
    guide_times, pose_iter = [], iter([TARGETS[d] for d in ORDER for _ in range(2)])

    class EnrollmentPipeline:
        def feed(self, guide, rgb, frame_ref, t):
            guide_times.append(t)
            return None, guide.feed(next(pose_iter), True, frame_ref, 1.0, t=t)

    monkeypatch.setattr(demo, "synced_ts", forbid_synthetic_timestamps)
    monkeypatch.setattr(demo, "make_account_assertion", lambda *a, **kw: "explicit-demo-assertion")
    assert demo.enroll(args(sensor=sensor), client, EnrollmentPipeline())
    assert events[0:2] == [("session", "enroll"), "sensor.start"]
    assert events.index("sensor.stop") < events.index("enroll")
    assert client.enroll_packet[2] == [f["ts_ms"] for f in inputs]
    assert client.enroll_packet[3] == [d for d in ORDER for _ in range(2)]
    assert client.enroll_packet[6]["captured_at_ms"] == inputs[0]["captured_at_ms"]
    assert guide_times == [f["captured_at_ms"] / 1000 for f in inputs]


def test_real_frame_source_never_imports_simulator_and_closes_on_early_exit(monkeypatch):
    events = []
    sensor = FakeSensor(events, [capture()])
    original_import = builtins.__import__

    def checked_import(name, *a, **kw):
        assert not name.startswith("faceauth.sim"), "Real capture imported simulator"
        return original_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", checked_import)
    source = demo.frame_source(args(sensor=sensor), "auth", None)
    assert next(source)[0] == 0
    source.close()
    assert events[-1] == "sensor.stop"


def test_capture_deadline_bounds_reads_and_drops_frame_that_arrives_too_late(monkeypatch):
    events, now = [], [0.0]
    sensor = FakeSensor(events, [capture()])
    read = sensor.read

    def slow_read(timeout_ms):
        result = read(timeout_ms)
        now[0] += 1
        return result

    sensor.read = slow_read
    monkeypatch.setattr(demo.time, "monotonic", lambda: now[0])
    assert list(demo.frame_source(args(sensor=sensor, auth_timeout=0.02), "auth", None)) == []
    assert ("sensor.read", 20) in events
    assert not sensor.started


def test_changed_calibration_is_rejected_and_sensor_released():
    events = []
    sensor = FakeSensor(events, [capture()])
    sensor.calib_version = "different-camera"
    with pytest.raises(RuntimeError, match="calibration changed"):
        list(demo.frame_source(args(sensor=sensor), "auth", None))
    assert not sensor.started


def test_run_binds_real_calibration_before_registration_and_supports_skip_enroll(monkeypatch):
    events = []
    sensor, client = FakeSensor(events, []), FakeClient(events)
    import faceauth.terminal.orbbec as orbbec
    monkeypatch.setattr(orbbec, "FemtoBoltSensor", lambda **kw: sensor)
    monkeypatch.setattr(demo.Pipeline, "load", lambda *a, **kw: FakePipeline())

    def create_client(url, device, **kw):
        assert kw["calib_version"] == CALIBRATION
        assert events == ["sensor.start", "sensor.stop"]
        events.append("client.create")
        client.calib_version = kw["calib_version"]
        return client

    monkeypatch.setattr(demo, "DeviceClient", create_client)
    monkeypatch.setattr(demo, "register", lambda a, c: events.append(("register", c.calib_version)))
    monkeypatch.setattr(demo, "enroll", lambda *a: pytest.fail("--skip-enroll ignored"))
    monkeypatch.setattr(demo, "authenticate", lambda *a: "approved")
    assert demo.run(args(register=True, skip_enroll=True)) == 0
    assert ("register", CALIBRATION) in events
    assert events[-2:] == ["client.close", "sensor.stop"]


def test_run_closes_http_and_sensor_after_capture_exception(monkeypatch):
    events = []
    sensor, client = FakeSensor(events, []), FakeClient(events)
    import faceauth.terminal.orbbec as orbbec
    monkeypatch.setattr(orbbec, "FemtoBoltSensor", lambda **kw: sensor)
    monkeypatch.setattr(demo.Pipeline, "load", lambda *a, **kw: FakePipeline())
    monkeypatch.setattr(demo, "DeviceClient", lambda *a, **kw: client)

    def fail(*args):
        raise RuntimeError("camera disconnected")

    monkeypatch.setattr(demo, "authenticate", fail)
    with pytest.raises(RuntimeError, match="disconnected"):
        demo.run(args())
    assert events[-2:] == ["client.close", "sensor.stop"]


def test_sim_packet_timestamps_keep_existing_helper(monkeypatch):
    expected = [{"rgb": 7, "depth": 9, "ir": 8}]
    monkeypatch.setattr(demo, "synced_ts", lambda indexes, start: expected if indexes == [12] and start == 3 else None)
    frame = capture()
    ref = demo.capture_ref(args(source="sim"), 12, {"rgb": b"jpeg"}, frame)
    encoded, timestamps, captured = demo.packet_capture(args(source="sim"), [ref], 3)
    assert encoded == [{"rgb": b"jpeg"}] and timestamps == expected and captured == 3


def test_cli_accepts_orbbec_and_usb3_fps_default(monkeypatch):
    seen = []
    monkeypatch.setattr(sys, "argv", ["terminal_demo.py", "--source", "orbbec", "--serial", "camera-1", "--skip-enroll"])
    monkeypatch.setattr(demo, "run", lambda options: seen.append(options) or 0)
    with pytest.raises(SystemExit) as exit_info:
        demo.main()
    assert exit_info.value.code == 0
    assert seen[0].fps == 30 and seen[0].serial == "camera-1" and seen[0].skip_enroll
