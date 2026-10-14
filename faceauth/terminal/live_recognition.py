"""Bounded live demo recognition; labels never grant authentication.

One worker shares the camera's latest frame with the preview and operation
workers. It retains only current boxes and short-lived spatial track metadata,
not source frames or embeddings. Viewers share inference, rather than spawning
one GPU workload per browser.
"""
import logging
import math
import threading
import time
from collections import deque

from faceauth.common.packet import encode_frame

LOG = logging.getLogger(__name__)
MAX_AGE_S = 0.6
MAX_FACES = 16


def overlap(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection)
    return intersection / area if area > 0 else 0


class FaceTracks:
    """Spatial IDs only; a name needs two fresh agreeing recognition results.

    A missed/uncertain/unknown result drops a previous name immediately. A
    crossing or changed identity cannot inherit a neighbouring person's label.
    No boxes are extrapolated after a face disappears.
    """

    def __init__(self):
        self.tracks = {}
        self.next_id = 1

    def clear(self):
        self.tracks.clear()

    def update(self, faces, names):
        pairs = sorted(((overlap(face["bbox"], previous["bbox"]), i, track_id)
                        for i, face in enumerate(faces) for track_id, previous in self.tracks.items()),
                       reverse=True)
        matched, used = {}, set()
        for score, index, track_id in pairs:
            if score >= 0.3 and index not in matched and track_id not in used:
                matched[index] = track_id
                used.add(track_id)
        current, visible = {}, []
        for i, face in enumerate(faces):
            track_id = matched.get(i)
            if track_id is None:
                track_id = self.next_id
                self.next_id += 1
            previous = self.tracks.get(track_id, {})
            account = face.get("account_id") if face["status"] == "known" else None
            count = previous.get("count", 0) + 1 if account and account == previous.get("account") else 1
            current[track_id] = {"bbox": face["bbox"], "account": account, "count": count}
            status = face["status"]
            if status == "known" and count < 2:
                status = "uncertain"
            label = names.get(account, account) if status == "known" else (
                "미등록" if status == "unknown" else "확인 중")
            visible.append({"track_id": track_id, "bbox": list(face["bbox"]), "status": status,
                            "display_name": label, "primary": bool(face.get("primary")),
                            "reason": face.get("reason", "")})
        self.tracks = current
        return visible


class LiveRecognition:
    def __init__(self, camera, auth, *, target_fps=15):
        self.camera, self.auth = camera, auth
        self.enabled = auth.enabled
        self.target_fps = max(1, min(float(target_fps), 30))
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.viewed_at = None
        self.arrived_at = None
        self.times = deque(maxlen=30)
        self.tracks = FaceTracks()
        self.face_accounts = {}
        self.result = {"enabled": self.enabled, "state": "starting", "message": "라이브 인식을 준비하고 있습니다.",
                       "sequence": None, "captured_at_ms": None, "image_size": [640, 480], "faces": [],
                       "fps": 0.0, "latency_ms": None, "truncated": False, "pad_enabled": False,
                       "pad_mode": "disabled_demo", "authentication_approved": False}

    def start(self):
        if self.enabled:
            self.thread = threading.Thread(target=self._run, daemon=True, name="live-recognition")
            self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=16)

    def snapshot(self, *, touch=False):
        now = time.monotonic()
        with self.lock:
            if touch:
                self.viewed_at = now
            out = {**self.result, "faces": [dict(face) for face in self.result["faces"]]}
            accounts = dict(self.face_accounts)
            age = None if self.arrived_at is None else max(0, now - self.arrived_at)
        out["age_ms"] = round(age * 1000) if age is not None else None
        if out["state"] == "running" and (age is None or age > MAX_AGE_S):
            out.update(state="unavailable", message="새 인식 결과를 기다리고 있습니다.", faces=[], fps=0.0)
        if not self.enabled:
            out.update(state="unavailable", message="라이브 인식이 설정되지 않았습니다.", faces=[])
        elif self.camera.status()["camera"]["state"] != "streaming":
            out.update(state="unavailable", message="카메라 연결을 기다리고 있습니다.", faces=[], fps=0.0)
        else:
            with self.auth.lock:
                active = self.auth.active is not None
            if active:
                out.update(state="paused", message="등록·인증 촬영 중에는 라이브 인식을 잠시 멈춥니다.",
                           faces=[], fps=0.0)
        if out["faces"]:
            names = self.auth.accounts.names(accounts.values())
            for face in out["faces"]:
                account = accounts.get(face["track_id"])
                if face["status"] == "known":
                    if account not in names:
                        face.update(status="unknown", display_name="미등록", reason="account_unavailable")
                    else:
                        face["display_name"] = names[account]
        return out

    def _clear(self, state, message):
        self.tracks.clear()
        self.times.clear()
        with self.lock:
            self.arrived_at = None
            self.face_accounts = {}
            self.result.update(state=state, message=message, sequence=None, captured_at_ms=None,
                               faces=[], fps=0.0, latency_ms=None, truncated=False)

    @staticmethod
    def _validate(out, calib, shape):
        height, width = shape
        if (out.get("demo") is not True or out.get("calib_version") != calib
                or out.get("image_size") != [width, height] or out.get("pad_enabled") is not False
                or out.get("pad_mode") != "disabled_demo" or out.get("approval_token")
                or not isinstance(out.get("faces"), list) or len(out["faces"]) > MAX_FACES):
            raise ValueError("Malformed live recognition response")
        faces = []
        for face in out["faces"]:
            bbox = face.get("bbox")
            if (not isinstance(bbox, list) or len(bbox) != 4
                    or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in bbox)
                    or not (0 <= bbox[0] < bbox[2] <= width and 0 <= bbox[1] < bbox[3] <= height)
                    or face.get("status") not in ("known", "unknown", "uncertain")
                    or (face["status"] == "known" and not isinstance(face.get("account_id"), str))):
                raise ValueError("Malformed live face")
            faces.append({key: face[key] for key in ("bbox", "status", "account_id", "reason") if key in face})
        if faces:
            # Selection is geometric; an unknown or low-quality largest face is
            # still the authentication candidate, never a smaller known face.
            primary = max(range(len(faces)), key=lambda i: (
                (faces[i]["bbox"][2] - faces[i]["bbox"][0]) * (faces[i]["bbox"][3] - faces[i]["bbox"][1]),
                -faces[i]["bbox"][0], -faces[i]["bbox"][1]))
            faces[primary]["primary"] = True
        return faces

    def _publish(self, out, capture, started):
        sequence, arrived, frame = capture
        faces = self._validate(out, frame["calib_version"], frame["depth"].shape)
        now = time.monotonic()
        if now - arrived > MAX_AGE_S:
            self._clear("unavailable", "인식 응답이 지연되고 있습니다.")
            return
        names = self.auth.accounts.names([f["account_id"] for f in faces if f["status"] == "known"])
        for face in faces:
            if face["status"] == "known" and face.get("account_id") not in names:
                face.update(status="unknown", reason="account_unavailable")
                face.pop("account_id", None)
        visible = self.tracks.update(faces, names)
        self.times.append(now)
        fps = ((len(self.times) - 1) / (self.times[-1] - self.times[0])
               if len(self.times) > 1 and self.times[-1] > self.times[0] else 0.0)
        with self.lock:
            self.face_accounts = {key: track["account"] for key, track in self.tracks.tracks.items()
                                  if track.get("account")}
            self.arrived_at = arrived
            self.result.update(state="running", message="얼굴을 자동으로 확인하고 있습니다.", sequence=sequence,
                               captured_at_ms=frame["captured_at_ms"], image_size=list(out["image_size"]),
                               faces=visible, fps=round(fps, 1), latency_ms=round((now - started) * 1000, 1),
                               truncated=bool(out.get("truncated")))

    def _run(self):
        client, session, calib, session_at, after, not_before = None, None, None, 0.0, -1, 0.0
        try:
            while not self.stop_event.is_set():
                started = time.monotonic()
                with self.lock:
                    viewed = self.viewed_at is not None and started - self.viewed_at < 2
                with self.auth.lock:
                    active = self.auth.active is not None
                    backend = dict(self.auth.backend)
                    recent = started - self.auth.backend_at < 20
                camera = self.camera.status()["camera"]
                if not viewed or active or camera["state"] != "streaming" or not recent or not backend.get("capture_ready"):
                    reason = ("등록·인증 촬영 중에는 라이브 인식을 잠시 멈춥니다." if active else
                              "화면을 열면 라이브 인식을 시작합니다." if not viewed else "카메라와 인식 서버를 기다리고 있습니다.")
                    self._clear("paused" if active or not viewed else "unavailable", reason)
                    session = None
                    if client:
                        client.close()
                        client = None
                    self.stop_event.wait(0.1)
                    continue
                current_calib = camera["info"].get("calib_version")
                try:
                    if client is None or calib != current_calib:
                        if client:
                            client.close()
                        calib = current_calib
                        client = self.auth.client_factory(calib)
                        session = None
                    if session is None or started - session_at >= 20:
                        session = client.create_session("auth", "live-demo")
                        session_at = time.monotonic()
                        after = self.camera.capture_sequence()
                        not_before = session_at + 0.2
                    capture = self.camera.capture_after(after, timeout=0.25)
                    if capture is None:
                        continue
                    after, arrived, frame = capture
                    if arrived < not_before:
                        continue
                    if frame["calib_version"] != calib:
                        raise ValueError("Calibration changed during live recognition")
                    encoded = encode_frame(frame["rgb"], frame["depth"], frame["ir"])
                    out = client.recognize_live(session, encoded, frame["ts_ms"])
                    # A capture can start while this request is in flight. Never
                    # publish its old labels over the new capture operation.
                    with self.auth.lock:
                        active = self.auth.active is not None
                    current_camera = self.camera.status()["camera"]
                    if active:
                        self._clear("paused", "등록·인증 촬영 중에는 라이브 인식을 잠시 멈춥니다.")
                    elif (current_camera["state"] != "streaming"
                          or current_camera["info"].get("calib_version") != calib):
                        self._clear("unavailable", "카메라 설정과 연결을 확인하고 있습니다.")
                        session = None
                    else:
                        self._publish(out, capture, started)
                    del capture, frame, encoded, out
                except Exception as exc:
                    LOG.warning("Live recognition unavailable: %s", type(exc).__name__)
                    self._clear("error", "라이브 인식 서버에 연결할 수 없습니다. 다시 연결하고 있습니다.")
                    session = None
                    if client:
                        client.close()
                        client = None
                    self.stop_event.wait(0.5)
                self.stop_event.wait(max(0, 1 / self.target_fps - (time.monotonic() - started)))
        finally:
            if client:
                client.close()
            self._clear("unavailable", "라이브 인식이 종료되었습니다.")
