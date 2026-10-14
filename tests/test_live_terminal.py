"""The live terminal must never report stale captures as a ready camera."""
import asyncio
import time

import cv2
import numpy as np
from fastapi.testclient import TestClient

from faceauth.terminal.live import CameraWorker, MAX_FRAME_AGE_S, create_app, preview_images


class FixedCamera(CameraWorker):
    def start(self):
        pass

    def stop(self):
        pass


def streaming_camera():
    camera = FixedCamera()
    camera.state = "streaming"
    camera.last_frame = time.monotonic()
    camera.images = {"rgb": b"current-frame"}
    camera.metrics = {"depth_valid_ratio": 0.9, "depth_median_mm": 600}
    return camera


def test_http_liveness_is_separate_from_sensor_and_auth_readiness():
    with TestClient(create_app(FixedCamera())) as client:
        assert client.get("/healthz").status_code == 200
        response = client.get("/readyz")
        assert response.status_code == 503
        assert response.json()["authentication_ready"] is False
        assert client.get("/api/preview/rgb.jpg").status_code == 503
        assert client.get("/api/preview/unknown.jpg").status_code == 404


def test_stale_frames_stop_being_served_and_stop_reporting_measurements():
    camera = streaming_camera()
    with TestClient(create_app(camera)) as client:
        response = client.get("/api/preview/rgb.jpg")
        assert response.status_code == 200
        assert response.content == b"current-frame"
        assert "no-store" in response.headers["cache-control"]
        assert client.get("/readyz").status_code == 200
        camera.last_frame -= MAX_FRAME_AGE_S + 1
        assert client.get("/api/preview/rgb.jpg").status_code == 503
        assert client.get("/readyz").status_code == 503
        status = client.get("/api/status").json()
        assert status["camera"]["state"] == "error"
        assert status["capture"]["depth_median_mm"] is None
        assert status["capture"]["fps"] == 0
        assert status["auth"]["ready"] is False


def test_depth_preview_preserves_sensor_arrays_and_zero_is_invalid():
    frame = {"rgb": np.zeros((48, 64, 3), np.uint8),
             "depth": np.zeros((48, 64), np.uint16), "ir": np.full((48, 64), 100, np.uint8)}
    frame["depth"][18:30, 24:40] = 600
    original = {key: value.copy() for key, value in frame.items()}
    images, metrics = preview_images(frame)
    assert set(images) == {"rgb", "depth", "ir"}
    assert metrics["depth_median_mm"] == 600
    for name, payload in images.items():
        assert cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_UNCHANGED) is not None
        np.testing.assert_array_equal(original[name], frame[name])


def test_capture_failure_clears_last_preview_and_releases_sensor():
    camera = streaming_camera()

    class BrokenSensor:
        stopped = False

        def start(self):
            raise RuntimeError("disconnected")

        def stop(self):
            self.stopped = True
            camera.stop_event.set()

    sensor = BrokenSensor()
    camera.factory = lambda: sensor
    camera._run()
    assert sensor.stopped
    assert camera.state == "error"
    assert camera.image("rgb") is None
    assert camera.status()["capture"]["depth_median_mm"] is None


def test_static_page_is_live_page_with_same_origin_scripts():
    with TestClient(create_app(FixedCamera())) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "/static/live.js" in response.text
        assert client.get("/static/live.js").status_code == 200
        assert "frame-ancestors 'none'" in response.headers["content-security-policy"]


def test_preview_sequence_skips_duplicates_and_recovers_after_server_restart():
    camera = streaming_camera()
    camera.frames = 12
    with TestClient(create_app(camera)) as client:
        first = client.get("/api/preview/rgb.jpg")
        assert first.headers["x-frame-sequence"] == "12"
        assert int(first.headers["x-frame-age-ms"]) >= 0
        unchanged = client.get("/api/preview/rgb.jpg?after=12&wait_ms=0")
        assert unchanged.status_code == 204
        assert unchanged.content == b""
        # An older process may have sent a larger sequence to this browser.
        assert client.get("/api/preview/rgb.jpg?after=999&wait_ms=0").content == b"current-frame"
        assert client.get("/api/preview/rgb.jpg?wait_ms=1001").status_code == 422


def test_new_frame_wakes_all_viewers_without_queueing_old_images():
    camera = streaming_camera()

    async def check():
        requests = [asyncio.create_task(camera.wait_image("rgb", after=0)) for _ in range(3)]
        await asyncio.sleep(0)
        assert len(camera.waiters) == 3

        def publish():
            with camera.lock:
                camera.images = {"rgb": b"newest-frame"}
                camera.frames = 4  # A slow viewer can skip intermediate frames.
                camera.last_frame = time.monotonic()
                camera._notify_waiters_locked()

        await asyncio.to_thread(publish)
        results = await asyncio.wait_for(asyncio.gather(*requests), 1)
        assert all(payload == b"newest-frame" and seq == 4 for payload, seq, _ in results)
        assert not camera.waiters

    asyncio.run(check())


def test_preview_wait_clears_notifications_on_timeout_and_cancellation():
    camera = streaming_camera()

    async def check():
        payload, sequence, _ = await camera.wait_image("rgb", after=0, wait_ms=1)
        assert payload is None and sequence == 0
        assert not camera.waiters
        request = asyncio.create_task(camera.wait_image("rgb", after=0))
        await asyncio.sleep(0)
        request.cancel()
        result = await asyncio.gather(request, return_exceptions=True)
        assert isinstance(result[0], asyncio.CancelledError)
        assert not camera.waiters

    asyncio.run(check())


def test_camera_failure_wakes_waiting_viewer_and_cannot_return_old_image():
    camera = streaming_camera()

    async def check():
        request = asyncio.create_task(camera.wait_image("rgb", after=0))
        await asyncio.sleep(0)
        with camera.lock:
            camera.state = "error"
            camera.images = {}
            camera._notify_waiters_locked()
        assert await asyncio.wait_for(request, 1) == (None, None, None)
        assert not camera.waiters

    asyncio.run(check())


def test_frame_becoming_stale_during_wait_is_unavailable():
    camera = streaming_camera()
    camera.last_frame -= MAX_FRAME_AGE_S - 0.005
    assert asyncio.run(camera.wait_image("rgb", after=0, wait_ms=10)) == (None, None, None)
