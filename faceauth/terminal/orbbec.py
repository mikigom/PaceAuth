"""Real Femto Bolt RGB/depth/IR capture using pyorbbecsdk2==2.1.2.

All outputs share an undistorted, centre-cropped 640x480 COLOR camera plane.
Depth and IR travel through the same calibrated 3-D projection and z-buffer;
an IR image must never merely be resized to match a D2C depth image. Missing
pixels remain zero. Device timestamps are preserved, including their skew.
No synthetic or RGB-only capture fallback is provided.
"""
import hashlib
import importlib
import json
import time
from dataclasses import dataclass

import cv2
import numpy as np

MAX_SYNC_SKEW_MS = 15  # faceauth.common.packet.MAX_SYNC_SKEW_MS
OUTPUT_SIZE = (640, 480)


class SensorError(RuntimeError):
    """The physical sensor cannot provide a complete, valid capture."""


def _name(value):
    return getattr(value, "name", str(value).rsplit(".", 1)[-1])


def _sdk_context():
    try:
        sdk = importlib.import_module("pyorbbecsdk")
    except (ImportError, OSError) as exc:
        raise SensorError("Orbbec SDK unavailable; install pyorbbecsdk2==2.1.2 and libusb-1.0-0/libudev1") from exc
    # Configure the static logger BEFORE constructing Context: /app may be read-only.
    sdk.Context.set_logger_to_file(sdk.OBLogLevel.NONE, "/tmp")
    sdk.Context.set_logger_level(sdk.OBLogLevel.ERROR)
    return sdk, sdk.Context()


def discover_devices():
    """List cameras without starting any stream or collecting image data."""
    try:
        _, context = _sdk_context()
        devices = context.query_devices()
        return [{"name": devices.get_device_name_by_index(i),
                 "serial": devices.get_device_serial_number_by_index(i),
                 "connection": devices.get_device_connection_type_by_index(i),
                 "pid": devices.get_device_pid_by_index(i)} for i in range(devices.get_count())]
    except SensorError:
        raise
    except Exception as exc:
        raise SensorError(f"Orbbec discovery failed: {exc}") from exc


@dataclass
class CameraCalibration:
    width: int
    height: int
    matrix: np.ndarray
    distortion: np.ndarray

    @classmethod
    def from_profile(cls, profile):
        p, d = profile.get_intrinsic(), profile.get_distortion()
        model = _name(d.model)
        if model not in {"NONE", "BROWN_CONRADY", "BROWN_CONRADY_K6"}:
            raise SensorError(f"Unsupported camera distortion model: {model}")
        coeffs = [float(getattr(d, k)) for k in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")]
        if model == "NONE":
            coeffs = [0.0] * 8
        k = np.array([[p.fx, 0, p.cx], [0, p.fy, p.cy], [0, 0, 1.0]], np.float64)
        if not np.isfinite(k).all() or p.fx <= 0 or p.fy <= 0 or not np.isfinite(coeffs).all():
            raise SensorError("Invalid factory camera calibration")
        return cls(int(p.width), int(p.height), k, np.asarray(coeffs, np.float64))

    def as_dict(self):
        return {"width": self.width, "height": self.height,
                "matrix": self.matrix.tolist(), "distortion": self.distortion.tolist()}


def _extrinsic(source, target):
    e = source.get_extrinsic_to(target)
    # Orbbec rotation is row-major; translation is millimetres.
    r, t = np.asarray(e.rot, np.float64).reshape(3, 3), np.asarray(e.transform, np.float64).reshape(3)
    if not np.isfinite(r).all() or not np.isfinite(t).all():
        raise SensorError("Invalid factory camera extrinsics")
    return r, t


class ColorRegistration:
    """Factory-calibrated registration with a common visibility z-buffer.

    IR conversion uses a fixed log1p curve over its declared bit range, never
    frame-wise histogram normalization. This preprocessing is part of the
    calibration fingerprint; real-camera PAD must be validated separately.
    """

    def __init__(self, color, depth, ir, depth_to_color, depth_to_ir, output_size=OUTPUT_SIZE):
        self.color, self.depth, self.ir = color, depth, ir
        self.rc, self.tc = depth_to_color
        self.ri, self.ti = depth_to_ir
        self.width, self.height = output_size
        scale = max(self.width / color.width, self.height / color.height)
        self.matrix = color.matrix.copy()
        self.matrix[:2] *= scale
        self.matrix[0, 2] -= (color.width * scale - self.width) / 2
        self.matrix[1, 2] -= (color.height * scale - self.height) / 2
        self.color_maps = cv2.initUndistortRectifyMap(
            color.matrix, color.distortion, None, self.matrix, output_size, cv2.CV_32FC1)
        u, v = np.meshgrid(np.arange(depth.width), np.arange(depth.height))
        pixels = np.stack((u, v), -1).astype(np.float64).reshape(-1, 1, 2)
        criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 1e-8)
        if hasattr(cv2, "undistortPointsIter"):
            xy = cv2.undistortPointsIter(pixels, depth.matrix, depth.distortion, None, None, criteria)
        else:
            # OpenCV 5 merged the iterative overload into undistortPoints.
            xy = cv2.undistortPoints(pixels, depth.matrix, depth.distortion, criteria=criteria)
        self.rays = np.column_stack((xy.reshape(-1, 2), np.ones(depth.width * depth.height)))
        self.same_ir = (depth.width == ir.width and depth.height == ir.height
                        and np.allclose(depth.matrix, ir.matrix) and np.allclose(depth.distortion, ir.distortion)
                        and np.allclose(self.ri, np.eye(3), atol=1e-6) and np.allclose(self.ti, 0, atol=1e-6))

    def as_dict(self):
        return {"algorithm": "femto-color-zbuffer-log-ir-v1", "output_size": [self.width, self.height],
                "output_matrix": self.matrix.tolist(), "color": self.color.as_dict(),
                "depth": self.depth.as_dict(), "ir": self.ir.as_dict(),
                "depth_to_color": [self.rc.tolist(), self.tc.tolist()],
                "depth_to_ir": [self.ri.tolist(), self.ti.tolist()]}

    def register(self, rgb, depth_mm, ir, ir_bits=16):
        if rgb.shape != (self.color.height, self.color.width, 3):
            raise SensorError("Color stream dimensions changed")
        if depth_mm.shape != (self.depth.height, self.depth.width) or ir.shape != (self.ir.height, self.ir.width):
            raise SensorError("Depth/IR stream dimensions changed")
        rgb_out = cv2.remap(rgb, *self.color_maps, cv2.INTER_LINEAR)
        depth_out = np.zeros((self.height, self.width), np.uint16)
        ir_out = np.zeros((self.height, self.width), np.uint8)
        source = np.flatnonzero(np.isfinite(depth_mm.ravel()) & (depth_mm.ravel() > 0))
        if not len(source):
            return rgb_out, depth_out, ir_out
        points = self.rays[source] * depth_mm.ravel()[source, None]
        if self.same_ir:
            intensity = ir.ravel()[source].astype(np.float64)
        else:
            ir_points = points @ self.ri.T + self.ti
            uv, _ = cv2.projectPoints(ir_points, np.zeros(3), np.zeros(3), self.ir.matrix, self.ir.distortion)
            uv = uv.reshape(-1, 2)
            valid = np.isfinite(uv).all(1) & (ir_points[:, 2] > 0)
            # Clip before integer conversion; invalid points are removed below.
            uv = np.rint(np.nan_to_num(uv, nan=-1, posinf=-1, neginf=-1)).astype(np.int64)
            valid &= (uv[:, 0] >= 0) & (uv[:, 0] < self.ir.width) & (uv[:, 1] >= 0) & (uv[:, 1] < self.ir.height)
            points, uv = points[valid], uv[valid]
            intensity = ir[uv[:, 1], uv[:, 0]].astype(np.float64)
        color_points = points @ self.rc.T + self.tc
        z = color_points[:, 2]
        valid = np.isfinite(color_points).all(1) & (z > 0) & (z < 65535.5)
        color_points, intensity, z = color_points[valid], intensity[valid], z[valid]
        if not len(z):
            return rgb_out, depth_out, ir_out
        u = np.rint(self.matrix[0, 0] * color_points[:, 0] / z + self.matrix[0, 2]).astype(np.int64)
        v = np.rint(self.matrix[1, 1] * color_points[:, 1] / z + self.matrix[1, 2]).astype(np.int64)
        valid = (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
        dest, z, intensity = (v[valid] * self.width + u[valid]), z[valid], intensity[valid]
        # Select the nearest surface at each color pixel, and its *same* IR sample.
        # Two linear reductions replace sorting the whole depth cloud twice.
        # Equal-depth collisions keep the first source sample, exactly as the
        # former stable depth sort did; its matching IR value stays attached.
        nearest = np.full(self.width * self.height, np.inf, dtype=z.dtype)
        np.minimum.at(nearest, dest, z)
        candidates = np.flatnonzero(z == nearest[dest])
        first = np.full(self.width * self.height, len(z), dtype=np.int64)
        np.minimum.at(first, dest[candidates], candidates)
        chosen = first[first < len(z)]
        depth_out.ravel()[dest[chosen]] = np.clip(np.rint(z[chosen]), 1, 65535).astype(np.uint16)
        if not 1 <= ir_bits <= 16:
            raise SensorError(f"Invalid IR bit depth: {ir_bits}")
        maximum = float((1 << ir_bits) - 1)
        values = np.clip(intensity[chosen], 0, maximum)
        if ir_bits > 8:
            values = np.log1p(values) * (255.0 / np.log1p(maximum))
        else:
            values *= 255.0 / maximum
        ir_out.ravel()[dest[chosen]] = np.rint(values).astype(np.uint8)
        return rgb_out, depth_out, ir_out


def _profile_description(p):
    return {"width": p.get_width(), "height": p.get_height(), "fps": p.get_fps(), "format": _name(p.get_format())}


def _select_profile(pipeline, sensor_type, *, width, height, fps, formats):
    profiles = pipeline.get_stream_profile_list(sensor_type)
    candidates = [profiles[i].as_video_stream_profile() for i in range(len(profiles))]
    for fmt in formats:
        for p in candidates:
            if (p.get_width(), p.get_height(), p.get_fps(), _name(p.get_format())) == (width, height, fps, fmt):
                return p
    available = [_profile_description(p) for p in candidates]
    raise SensorError(f"Required {sensor_type} {width}x{height}@{fps} profile unavailable; profiles={available}")


def _rgb(frame):
    w, h, fmt = frame.get_width(), frame.get_height(), _name(frame.get_format())
    raw = np.frombuffer(frame.get_data(), np.uint8)
    if fmt == "MJPG":
        bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if bgr is None:
            raise SensorError("Corrupt MJPEG camera frame")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if fmt in {"RGB", "BGR"}:
        data = raw.reshape(h, w, 3)
        return data.copy() if fmt == "RGB" else cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    if fmt in {"YUYV", "YUY2", "UYVY"}:
        conversion = cv2.COLOR_YUV2RGB_UYVY if fmt == "UYVY" else cv2.COLOR_YUV2RGB_YUY2
        return cv2.cvtColor(raw.reshape(h, w, 2), conversion)
    raise SensorError(f"Unsupported color format: {fmt}")


class FemtoBoltSensor:
    """Single-owner camera. start()/read()/stop() or ``with FemtoBoltSensor()``.

    read() returns rgb/depth/ir arrays, real modality ts_ms, captured_at_ms,
    and calib_version. The device clock is mapped to host epoch once; modality
    offsets and frame ordering are never fabricated. info is JSON-compatible.
    """

    def __init__(self, serial=None, fps=15):
        if fps not in (5, 15, 30):
            raise ValueError("Femto Bolt fps must be 5, 15 or 30")
        self.serial, self.fps = serial, int(fps)
        self.info = {"ready": False, "model": "Orbbec Femto Bolt", "fps": fps}
        self.calib_version = ""
        self._pipeline = self._device = self._context = None
        self._clock_offset = self._last_timestamp = None
        self._ir_bits = 16

    def start(self):
        if self._pipeline is not None:
            return self
        try:
            sdk, self._context = _sdk_context()
            devices = self._context.query_devices()
            matches = [i for i in range(devices.get_count())
                       if devices.get_device_pid_by_index(i) == 0x066B
                       and (not self.serial or devices.get_device_serial_number_by_index(i) == self.serial)]
            if len(matches) != 1:
                raise SensorError("No Femto Bolt found; check USB/power and Docker USB access" if not matches else
                                  "Multiple Femto Bolt cameras found; set the camera serial number")
            i = matches[0]
            self.info.update(serial=devices.get_device_serial_number_by_index(i),
                             connection=devices.get_device_connection_type_by_index(i))
            self._device = devices.get_device_by_index(i)
            device_info = self._device.get_device_info()
            self.info.update(firmware=device_info.get_firmware_version())
            sensors = self._device.get_sensor_list()
            supported = [_name(sensors.get_type_by_index(j)) for j in range(len(sensors))]
            self.info["sensors"] = supported
            missing = {"COLOR_SENSOR", "DEPTH_SENSOR", "IR_SENSOR"} - set(supported)
            if missing:
                raise SensorError(f"Femto Bolt exposes no {', '.join(sorted(missing))} on {self.info['connection']} "
                                  f"(firmware {self.info['firmware']}); check USB 3 data cable/port, power and firmware")
            pipeline = sdk.Pipeline(self._device)
            color = _select_profile(pipeline, sdk.OBSensorType.COLOR_SENSOR, width=1280, height=720,
                                    fps=self.fps, formats=("MJPG", "RGB", "YUYV", "YUY2", "BGR"))
            depth = _select_profile(pipeline, sdk.OBSensorType.DEPTH_SENSOR, width=640, height=576,
                                    fps=self.fps, formats=("Y16", "Z16"))
            ir = _select_profile(pipeline, sdk.OBSensorType.IR_SENSOR, width=640, height=576,
                                 fps=self.fps, formats=("Y16", "Y8"))
            self._registration = ColorRegistration(*(CameraCalibration.from_profile(p) for p in (color, depth, ir)),
                                                   _extrinsic(depth, color), _extrinsic(depth, ir))
            calibration = self._registration.as_dict()
            streams = {key: _profile_description(p) for key, p in zip(("rgb", "depth", "ir"), (color, depth, ir))}
            self._ir_bits = 8 if _name(ir.get_format()) == "Y8" else 16
            fingerprint = {"serial": self.info["serial"], "firmware": self.info["firmware"], "streams": streams,
                           "ir_bits": self._ir_bits, "calibration": calibration}
            self.calib_version = "femto-bolt-v1-" + hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:16]
            self.info.update(calib_version=self.calib_version, calibration=calibration,
                             streams=streams, ir_bits=self._ir_bits)
            config = sdk.Config()
            for p in (color, depth, ir):
                config.enable_stream(p)
            config.set_frame_aggregate_output_mode(sdk.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
            pipeline.enable_frame_sync()
            self._pipeline = pipeline  # stop partially started streams if SDK start fails
            pipeline.start(config)
            self._clock_offset = self._last_timestamp = None
            return self
        except Exception as exc:
            self.stop()
            message = str(exc)
            self.info.update(error=message)
            if isinstance(exc, SensorError):
                raise
            raise SensorError(f"Femto Bolt startup failed ({self.info.get('connection', 'unknown USB')}): {message}. "
                              "Check EGL/Mesa runtime libraries, USB data connection, power, and camera firmware.") from exc

    def read(self, timeout_ms=2000):
        if self._pipeline is None:
            raise SensorError("Femto Bolt is not started")
        deadline = time.monotonic() + timeout_ms / 1000
        reason = "no complete RGB/depth/IR frames"
        try:
            while time.monotonic() < deadline:
                remaining = max(1, int((deadline - time.monotonic()) * 1000))
                frames = self._pipeline.wait_for_frames(remaining)
                if frames is None:
                    continue
                color, depth, ir = frames.get_color_frame(), frames.get_depth_frame(), frames.get_ir_frame()
                if color is None or depth is None or ir is None:
                    continue
                stamps = {k: int(f.get_timestamp_us()) / 1000 for k, f in zip(("rgb", "depth", "ir"), (color, depth, ir))}
                skew = max(abs(stamps["rgb"] - stamps[m]) for m in ("depth", "ir"))
                if skew > MAX_SYNC_SKEW_MS:
                    reason = f"unsynchronized camera frames ({skew:.1f} ms skew, maximum {MAX_SYNC_SKEW_MS} ms)"
                    continue
                if self._last_timestamp is not None and stamps["rgb"] <= self._last_timestamp:
                    reason = "non-increasing camera timestamps"
                    continue
                received_at_ms = int(time.time() * 1000)
                if self._clock_offset is None:
                    self._clock_offset = received_at_ms - stamps["rgb"]
                ts_ms = {k: int(round(t + self._clock_offset)) for k, t in stamps.items()}
                raw_depth = np.frombuffer(depth.get_data(), np.uint16).reshape(depth.get_height(), depth.get_width())
                scale = float(depth.get_depth_scale())
                if not np.isfinite(scale) or scale <= 0:
                    raise SensorError("Invalid sensor depth scale")
                dtype = np.uint8 if _name(ir.get_format()) == "Y8" else np.uint16
                raw_ir = np.frombuffer(ir.get_data(), dtype).reshape(ir.get_height(), ir.get_width())
                bits = 8 if dtype == np.uint8 else int(ir.get_pixel_available_bit_size())
                if bits != self._ir_bits:
                    raise SensorError(f"IR bit depth {bits} differs from calibrated {self._ir_bits}-bit stream")
                rgb_out, depth_out, ir_out = self._registration.register(_rgb(color), raw_depth.astype(np.float64) * scale,
                                                                        raw_ir, bits)
                self._last_timestamp = stamps["rgb"]
                self.info.update(ready=True, sync_skew_ms=skew, depth_valid_fraction=float(np.mean(depth_out > 0)))
                self.info.pop("error", None)
                return {"rgb": rgb_out, "depth": depth_out, "ir": ir_out, "ts_ms": ts_ms,
                        "captured_at_ms": ts_ms["rgb"], "calib_version": self.calib_version}
            raise SensorError(f"Femto Bolt capture timed out: {reason}")
        except Exception as exc:
            self.info.update(ready=False, error=str(exc))
            if isinstance(exc, SensorError):
                raise
            raise SensorError(f"Femto Bolt capture failed: {exc}") from exc

    def stop(self):
        pipeline, self._pipeline = self._pipeline, None
        self.info["ready"] = False
        try:
            if pipeline is not None:
                pipeline.stop()
        except Exception:
            pass  # disconnected hardware may already have stopped
        self._device = self._context = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.stop()
