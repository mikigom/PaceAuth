"""Calibrated sensor registration and fail-closed capture, without USB hardware."""
import types

import numpy as np
import pytest

from faceauth.terminal.orbbec import CameraCalibration, ColorRegistration, FemtoBoltSensor, SensorError


def camera(w=4, h=3, fx=1, cx=0):
    return CameraCalibration(w, h, np.array([[fx, 0, cx], [0, fx, 0], [0, 0, 1]], np.float64), np.zeros(8))


IDENTITY = (np.eye(3), np.zeros(3))


def registration(color=None, depth=None, ir=None, dc=IDENTITY, di=IDENTITY):
    return ColorRegistration(color or camera(), depth or camera(), ir or camera(), dc, di, (4, 3))


def test_identity_keeps_depth_ir_pixel_pairs_and_invalid_holes():
    rgb = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    depth = np.full((3, 4), 700, np.float64)
    depth[1, 2] = 0
    ir = np.arange(12, dtype=np.uint8).reshape(3, 4) + 10
    out_rgb, out_depth, out_ir = registration().register(rgb, depth, ir, 8)
    assert np.array_equal(out_rgb, rgb)
    assert np.array_equal(out_depth, depth.astype(np.uint16))
    assert out_ir[1, 2] == 0
    assert np.array_equal(out_ir[depth > 0], ir[depth > 0])


def test_translation_projects_depth_and_ir_to_color_together():
    # At depth 500 mm a 500 mm X translation and focal length 1 shifts one pixel.
    rgb = np.zeros((3, 4, 3), np.uint8)
    depth = np.full((3, 4), 500, np.float64)
    ir = np.arange(12, dtype=np.uint8).reshape(3, 4) + 10
    _, d, i = registration(dc=(np.eye(3), np.array([500, 0, 0]))).register(rgb, depth, ir, 8)
    assert not d[:, 0].any() and not i[:, 0].any()
    assert np.array_equal(i[:, 1:], ir[:, :-1])
    assert np.all(d[:, 1:] == 500)


def test_nearest_surface_wins_occlusion_and_retains_its_ir():
    # Color focal length 0.1 maps the four depth columns to the same color pixel.
    depth = np.zeros((3, 4), np.float64)
    depth[0, :2] = (900, 400)
    ir = np.zeros((3, 4), np.uint8)
    ir[0, :2] = (30, 180)
    _, d, i = registration(color=camera(fx=0.1)).register(np.zeros((3, 4, 3), np.uint8), depth, ir, 8)
    assert d[0, 0] == 400 and i[0, 0] == 180
    assert np.count_nonzero(d) == 1


def test_equal_nearest_depth_keeps_first_source_ir_among_occluded_samples():
    # All four sources project onto one pixel. Sources 1 and 2 tie for the
    # nearest surface; stable source ordering must choose source 1's IR.
    depth = np.zeros((3, 4), np.float64)
    depth[0] = (900, 400, 400, 700)
    ir = np.zeros((3, 4), np.uint8)
    ir[0] = (20, 70, 190, 250)
    _, d, i = registration(color=camera(fx=0.1)).register(np.zeros((3, 4, 3), np.uint8), depth, ir, 8)
    assert d[0, 0] == 400 and i[0, 0] == 70
    assert np.count_nonzero(d) == np.count_nonzero(i) == 1


def test_ir_extrinsics_are_applied_instead_of_resizing():
    depth = np.full((3, 4), 500, np.float64)
    ir = np.arange(12, dtype=np.uint8).reshape(3, 4) + 10
    _, d, i = registration(di=(np.eye(3), np.array([500, 0, 0]))).register(
        np.zeros((3, 4, 3), np.uint8), depth, ir, 8)
    assert np.array_equal(i[:, :-1], ir[:, 1:])
    assert not d[:, -1].any() and not i[:, -1].any()


def test_depth_is_color_camera_z_not_raw_depth():
    depth = np.zeros((3, 4), np.float64)
    depth[0, 0] = 500
    _, d, _ = registration(dc=(np.eye(3), np.array([0, 0, 100]))).register(
        np.zeros((3, 4, 3), np.uint8), depth, np.ones((3, 4), np.uint16))
    assert d[0, 0] == 600


def test_fixed_ir_conversion_does_not_change_with_other_pixels():
    rgb, depth = np.zeros((3, 4, 3), np.uint8), np.full((3, 4), 500, np.float64)
    ir = np.full((3, 4), 1024, np.uint16)
    first = registration().register(rgb, depth, ir)[2]
    ir[2, 3] = 65535
    second = registration().register(rgb, depth, ir)[2]
    assert first[0, 0] == second[0, 0] and 0 < first[0, 0] < 255
    assert second[2, 3] == 255


def test_nonfinite_depth_is_invalid_and_shape_changes_fail():
    rgb, depth, ir = np.zeros((3, 4, 3), np.uint8), np.full((3, 4), np.nan), np.ones((3, 4), np.uint8)
    _, d, i = registration().register(rgb, depth, ir, 8)
    assert not d.any() and not i.any()
    with pytest.raises(SensorError, match="dimensions changed"):
        registration().register(rgb, depth[:2], ir, 8)


class FakeFrame:
    def __init__(self, data, timestamp_us, fmt, scale=1):
        self.data, self.timestamp, self.fmt, self.scale = data, timestamp_us, fmt, scale

    def get_data(self): return self.data.tobytes()
    def get_width(self): return self.data.shape[1]
    def get_height(self): return self.data.shape[0]
    def get_timestamp_us(self): return self.timestamp
    def get_format(self): return self.fmt
    def get_depth_scale(self): return self.scale
    def get_pixel_available_bit_size(self): return 16


def frames(timestamp=100000, depth_timestamp=None, ir_timestamp=None, scale=1):
    color = FakeFrame(np.zeros((3, 4, 3), np.uint8), timestamp, "RGB")
    depth = FakeFrame(np.full((3, 4), 500, np.uint16), depth_timestamp or timestamp, "Y16", scale)
    ir = FakeFrame(np.full((3, 4), 1024, np.uint16), ir_timestamp or timestamp, "Y16")
    return types.SimpleNamespace(get_color_frame=lambda: color, get_depth_frame=lambda: depth, get_ir_frame=lambda: ir)


def fake_sensor(queue):
    sensor = FemtoBoltSensor()
    sensor._registration = registration()
    sensor._pipeline = types.SimpleNamespace(wait_for_frames=lambda _: queue.pop(0) if queue else None)
    return sensor


def test_capture_preserves_skew_and_depth_scale(monkeypatch):
    monkeypatch.setattr("faceauth.terminal.orbbec.time.time", lambda: 1000)
    sensor = fake_sensor([frames(depth_timestamp=107000, ir_timestamp=98000, scale=2)])
    f = sensor.read(100)
    assert f["ts_ms"] == {"rgb": 1000000, "depth": 1000007, "ir": 999998}
    assert f["captured_at_ms"] == 1000000 and np.all(f["depth"] == 1000)
    assert sensor.info["ready"] and sensor.info["sync_skew_ms"] == 7


def test_unsynchronized_capture_is_rejected_without_faking_timestamps():
    sensor = fake_sensor([frames(depth_timestamp=116000)])
    with pytest.raises(SensorError, match="unsynchronized"):
        sensor.read(5)
    assert not sensor.info["ready"]


def test_duplicate_timestamps_rejected_and_next_actual_frame_used():
    sensor = fake_sensor([frames(), frames(), frames(timestamp=167000)])
    first = sensor.read(50)
    second = sensor.read(50)
    assert second["ts_ms"]["rgb"] - first["ts_ms"]["rgb"] == 67


def test_missing_modality_fails_closed():
    partial = frames()
    partial.get_ir_frame = lambda: None
    sensor = fake_sensor([partial])
    with pytest.raises(SensorError, match="no complete RGB/depth/IR"):
        sensor.read(5)


def test_ir_bit_depth_cannot_silently_change_calibrated_normalization():
    capture = frames()
    capture.get_ir_frame().get_pixel_available_bit_size = lambda: 12
    sensor = fake_sensor([capture])
    with pytest.raises(SensorError, match="IR bit depth 12 differs"):
        sensor.read(50)


def test_sdk_missing_is_explicit_and_lazy(monkeypatch):
    def missing(_): raise ImportError("missing SDK")
    monkeypatch.setattr("faceauth.terminal.orbbec.importlib.import_module", missing)
    sensor = FemtoBoltSensor()
    with pytest.raises(SensorError, match="Orbbec SDK unavailable"):
        sensor.start()
    assert not sensor.info["ready"] and sensor._pipeline is None
