"""Live Femto Bolt camera terminal. Raw captures stay in bounded process memory.

This service exposes sensor readiness separately from HTTP liveness. It never
turns a preview or a simulated PAD result into an authentication approval.
"""
import asyncio
import logging
import os
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

LOG = logging.getLogger(__name__)
UI_DIR = Path(__file__).resolve().parents[2] / "ui"
MAX_FRAME_AGE_S = 3.0


def preview_images(frame):
    """Three display JPEGs; depth/IR source measurements are never changed."""
    rgb = cv2.cvtColor(frame["rgb"], cv2.COLOR_RGB2BGR)
    depth = frame["depth"]
    valid = depth > 0
    level = np.clip((depth.astype(np.float32) - 250) / 1750 * 255, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(255 - level, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    images = {}
    for key, data in (("rgb", rgb), ("depth", colored), ("ir", frame["ir"])):
        ok, encoded = cv2.imencode(".jpg", data, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            raise ValueError("preview encoding failed")
        images[key] = encoded.tobytes()
    h, w = depth.shape
    center = depth[h * 3 // 8:h * 5 // 8, w * 3 // 8:w * 5 // 8]
    center_valid = center[center > 0]
    return images, {
        "depth_valid_ratio": round(float(valid.mean()), 3),
        "depth_median_mm": float(np.median(center_valid)) if center_valid.size else None,
    }


class CameraWorker:
    def __init__(self, factory=None):
        self.factory = factory or self._sensor
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.state = "starting"
        self.message = "카메라 연결을 확인하고 있습니다"
        self.info = {}
        self.images = {}
        self.frames = 0
        self.last_frame = None
        self.frame_times = deque(maxlen=30)
        self.metrics = {"depth_valid_ratio": None, "depth_median_mm": None}
        self.waiters = set()
        self.capture = None
        self.capture_condition = threading.Condition(self.lock)

    @staticmethod
    def _sensor():
        from faceauth.terminal.orbbec import FemtoBoltSensor
        return FemtoBoltSensor(serial=os.getenv("FA_ORBBEC_SERIAL") or None,
                               fps=int(os.getenv("FA_ORBBEC_FPS", "15")))

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True, name="femto-bolt")
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=12)

    def _run(self):
        while not self.stop_event.is_set():
            sensor = None
            try:
                sensor = self.factory()
                sensor.start()
                first_capture = True
                while not self.stop_event.is_set():
                    frame = sensor.read(timeout_ms=10000 if first_capture else 2000)
                    first_capture = False
                    images, metrics = preview_images(frame)
                    now = time.monotonic()
                    with self.lock:
                        self.images, self.metrics = images, metrics
                        self.capture = frame
                        self.info = dict(sensor.info)
                        self.info["calib_version"] = frame["calib_version"]
                        self.last_frame = now
                        self.frame_times.append(now)
                        self.frames += 1
                        self.state = "streaming"
                        self.message = "RGB · 깊이 · IR 실시간 수신 중"
                        self._notify_waiters_locked()
                        self.capture_condition.notify_all()
            except Exception as exc:
                # No image contents or credentials are logged.
                LOG.warning("Camera unavailable: %s", exc)
                with self.lock:
                    if sensor is not None:
                        self.info = dict(getattr(sensor, "info", {}))
                    self.state = "error"
                    self.message = str(exc)[:500]
                    self.images = {}
                    self.capture = None
                    self.frame_times.clear()
                    self.metrics = {"depth_valid_ratio": None, "depth_median_mm": None}
                    self._notify_waiters_locked()
                    self.capture_condition.notify_all()
            finally:
                if sensor is not None:
                    try:
                        sensor.stop()
                    except Exception:
                        LOG.warning("Camera stop failed")
            self.stop_event.wait(3)

    def status(self):
        with self.lock:
            age = None if self.last_frame is None else time.monotonic() - self.last_frame
            fresh = self.state == "streaming" and age is not None and age <= MAX_FRAME_AGE_S
            state = self.state if self.state != "streaming" or fresh else "error"
            message = self.message if state == self.state else "카메라 프레임 수신이 지연되고 있습니다"
            fps = ((len(self.frame_times) - 1) / (self.frame_times[-1] - self.frame_times[0])
                   if len(self.frame_times) > 1 and self.frame_times[-1] > self.frame_times[0] else 0.0)
            return {
                "mode": "sensor_preview",
                "camera": {"state": state, "message": message, "info": dict(self.info)},
                "capture": {"frames": self.frames, "fps": round(fps, 1) if fresh else 0,
                            "age_ms": round(age * 1000) if age is not None else None,
                            **(self.metrics if fresh else {"depth_valid_ratio": None, "depth_median_mm": None})},
                "auth": {"ready": False, "blockers": [
                    {"code": "preview_service", "message": "현재 접속한 서비스는 실센서 확인용입니다. 얼굴 등록·인증은 아직 연결되지 않았습니다."},
                    {"code": "sensor_validation", "message": "기존 인증 모델은 시뮬레이터 기준이며 Femto Bolt 검증이 필요합니다."},
                ]},
            }

    def image(self, modality):
        with self.lock:
            if (self.state != "streaming" or self.last_frame is None
                    or time.monotonic() - self.last_frame > MAX_FRAME_AGE_S):
                return None
            return self.images.get(modality)

    def capture_sequence(self):
        with self.lock:
            return self.frames

    def capture_after(self, after, timeout=1):
        """Bounded raw capture for the local operation worker; never an HTTP route."""
        with self.capture_condition:
            self.capture_condition.wait_for(
                lambda: self.frames != after or self.state == "error" or self.stop_event.is_set(), timeout)
            if (self.frames == after or self.state != "streaming" or self.capture is None
                    or self.last_frame is None or time.monotonic() - self.last_frame > MAX_FRAME_AGE_S):
                return None
            return self.frames, self.last_frame, dict(self.capture)

    def _notify_waiters_locked(self):
        """Wake HTTP consumers without holding a frame queue for each viewer."""
        for loop, future in self.waiters:
            try:
                loop.call_soon_threadsafe(self._resolve_waiter, future)
            except RuntimeError:
                # The request's event loop may already have shut down.
                pass
        self.waiters.clear()

    @staticmethod
    def _resolve_waiter(future):
        if not future.done():
            future.set_result(None)

    async def wait_image(self, modality, after=None, wait_ms=1000):
        """Return the latest JPEG, or wait for a different sequence.

        Each consumer holds only a notification future. Slow clients skip to the
        newest frame, so neither capture nor other viewers wait for their network.
        A None sequence means unavailable; a None payload alone means unchanged.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + wait_ms / 1000
        while True:
            with self.lock:
                age = None if self.last_frame is None else time.monotonic() - self.last_frame
                if (self.state != "streaming" or age is None or age > MAX_FRAME_AGE_S
                        or modality not in self.images):
                    return None, None, None
                sequence, age_ms = self.frames, round(age * 1000)
                if after is None or sequence != after:
                    return self.images[modality], sequence, age_ms
                remaining = deadline - loop.time()
                if remaining <= 0:
                    return None, sequence, age_ms
                future = loop.create_future()
                waiter = (loop, future)
                self.waiters.add(waiter)
            try:
                await asyncio.wait_for(future, remaining)
            except asyncio.TimeoutError:
                pass
            finally:
                with self.lock:
                    self.waiters.discard(waiter)


def create_app(camera=None, auth=None, recognition=None):
    from faceauth.terminal.web_auth import LiveAuth, auth_router
    from faceauth.terminal.live_recognition import LiveRecognition
    camera = camera or CameraWorker()
    auth = auth or LiveAuth(camera)
    recognition = recognition or LiveRecognition(camera, auth,
                                                target_fps=float(os.getenv("FA_LIVE_RECOGNITION_FPS", "15")))

    @asynccontextmanager
    async def lifespan(app):
        camera.start()
        auth.start()
        recognition.start()
        try:
            yield
        finally:
            await asyncio.to_thread(recognition.stop)
            await asyncio.to_thread(auth.stop)
            await asyncio.to_thread(camera.stop)

    app = FastAPI(title="PaceAuth live camera", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.camera = camera
    app.state.auth = auth
    app.state.recognition = recognition
    app.include_router(auth_router(auth))

    @app.middleware("http")
    async def no_store(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store, max-age=0"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
        return response

    @app.get("/")
    def index():
        return FileResponse(UI_DIR / "live.html")

    @app.get("/healthz")
    def health():
        return {"status": "ok", "service": "camera-preview"}

    @app.get("/readyz")
    def ready():
        status = camera.status()
        is_ready = status["camera"]["state"] == "streaming"
        return JSONResponse({"ready": is_ready, "scope": "sensor", "authentication_ready": auth.status()["ready"]},
                            status_code=200 if is_ready else 503)

    @app.get("/api/status")
    def status(request: Request, response: Response):
        out = camera.status()
        out["auth"] = auth.status_for_browser(request, response)
        if out["auth"]["enabled"]:
            out["mode"] = "authentication"
        return out

    @app.get("/api/live/faces")
    def live_faces():
        return recognition.snapshot(touch=True)

    @app.get("/api/preview/{modality}.jpg")
    async def preview(modality: str, after: int | None = Query(default=None, ge=0),
                      wait_ms: int = Query(default=1000, ge=0, le=1000)):
        if modality not in ("rgb", "depth", "ir"):
            raise HTTPException(404, "Unknown modality")
        payload, sequence, age_ms = await camera.wait_image(modality, after, wait_ms)
        if sequence is None:
            raise HTTPException(503, "No fresh camera frame")
        headers = {"X-Frame-Sequence": str(sequence), "X-Frame-Age-Ms": str(age_ms)}
        if payload is None:
            return Response(status_code=204, headers=headers)
        return Response(payload, media_type="image/jpeg", headers=headers)

    app.mount("/static", StaticFiles(directory=UI_DIR), name="static")
    return app


app = create_app()
