"""Engine: central GPU inference service (docs/IMPLEMENTATION.md §3).

POST /v1/analyze (msgpack) {"mode": "enroll"|"auth"|"guide"|"observe"|"live", "calib_version", "frames"} -> detection,
server-side sensor/quality re-verification (FR-03), head pose, PAD and embedding. The API decides; the engine
only measures.

- Models come only from the signed registry (§7) at the versions active in model_versions; unsigned, re-hashed or
  inactive models refuse to load, and so does any ONNX graph that would run a node on CPU (fail closed).
- Concurrent requests are micro-batched: everything queued while the GPU worker was busy becomes one batch.
- Frames live only in memory (read-only container, tmpfs /tmp).
"""
import asyncio
import os
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import cv2
import msgpack
import numpy as np
import onnxruntime as ort
import psycopg
from fastapi import FastAPI, HTTPException, Request, Response

from faceauth.common.femto_pad_experiment import MEASUREMENT_REASONS
from faceauth.common.packet import decode_frame
from faceauth.common.pad_validation import (CANDIDATE_SCHEMA, PADValidationGate, _finite_measurements, candidate_binding,
                                           clean_candidate_measurement)
from faceauth.common.registry import active_entries, load_registry
from faceauth.models.headpose import CROPS, HeadPose
from faceauth.models.femto_pad import measure_features
from faceauth.models.pad import PAD, face_features
from faceauth.models.recog import Recognizer
from faceauth.models.scrfd import SCRFD
from faceauth.terminal.pipeline import face_crop_stats

DETECTOR, POSE, PAD_MODEL = "scrfd_34g", "6drepnet360", "pad_rgbdir"
# Server-side re-verification thresholds; changing any of them is a new QUALITY version (re-test, Design §7).
# Face size / blur / exposure use the terminal's metric (face_crop_stats) so both sides measure the same thing;
# the auth pose window is wider than the terminal's prefilter (20/15) so terminal-accepted frames are not bounced.
QUALITY = "q@1"
DET_THRESH, MIN_FACE_PX, OTHER_FACE_PX = 0.5, 60.0, 30.0  # faces under 30 px (far background) are not counted
BLUR_MIN, EXPOSURE = 40.0, (40.0, 220.0)
DEPTH_MM, DEPTH_VALID, IR_RANGE = (350, 1000), 0.5, (20, 250)  # 40-80 cm use + margin; face-box medians
LIVE_DEPTH_MM = (350, 3000)  # Read-only group demo; authentication/enrollment retain the close-range policy.
AUTH_MAX_POSE = (25.0, 20.0)
MAX_BATCH_FRAMES = 64
MAX_LIVE_FACES = 16
# Per-replica parallelism (measured: one serial worker = ~90 frames/s, SCRFD 4.9 + PAD 2.3 ms/frame on the CPU-bound
# path). Several SCRFD sessions = several CUDA streams + concurrent pre/post-processing; PAD physics on threads.
DET_SESSIONS = int(os.environ.get("FA_ENGINE_DET_SESSIONS", "2"))   # SCRFD-34G tops out ~350 frames/s per A6000
CPU_THREADS = int(os.environ.get("FA_ENGINE_CPU_THREADS", "6"))
PAD_MODE = os.environ.get("FA_PAD_MODE") or (
    "disabled_pilot" if os.environ.get("FA_PILOT_DISABLE_PAD", "0") == "1" else "enabled")
GEOMETRY_MODES = ("femto_geometry_pilot", "femto_geometry_experimental")
if PAD_MODE not in ("enabled", "disabled_pilot", *GEOMETRY_MODES):
    raise ValueError("unknown FA_PAD_MODE")
PAD_ENABLED = PAD_MODE != "disabled_pilot"
PAD_DIAGNOSTIC_REASONS = MEASUREMENT_REASONS | frozenset((
    "plane_residual_mm", "nose_protrusion_mm", "measurement_invalid", "profile_not_ready",
))


cv2.setNumThreads(1)  # frames are parallelised above; OpenCV's own pool would fight them on a 128-core host


def _env_file(name):
    p = os.environ.get(name)
    return open(p).read().strip() if p else None


def _session_options():
    so = ort.SessionOptions()
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")  # a CPU-placed node fails the load
    return so


class Models:
    """One consistent set of model versions. Built in a worker thread; swapped atomically on reload."""

    def __init__(self, entries: dict, recog_name: str, trt: bool):
        missing = [n for n in (DETECTOR, POSE, PAD_MODEL, recog_name) if n not in entries]
        if missing:
            raise RuntimeError(f"no active signed version for {missing}")
        gpu = "TensorrtExecutionProvider" if trt else "CUDAExecutionProvider"
        prov = ([("TensorrtExecutionProvider", {"trt_fp16_enable": True, "trt_engine_cache_enable": True,
                                                "trt_engine_cache_path": "/tmp/trt"})] if trt else []) \
            + ["CUDAExecutionProvider"]
        self.dets = queue.Queue()
        for _ in range(DET_SESSIONS):
            det = SCRFD(entries[DETECTOR]["path"], providers=prov, sess_options=_session_options())
            got = det.session.get_providers()
            if got[0] != gpu:  # ORT silently falls back when an EP fails to load
                raise RuntimeError(f"detector wants {gpu}, ORT gave {got}")
            self.dets.put(det)
        self.det = det
        self.pool = ThreadPoolExecutor(max(DET_SESSIONS, CPU_THREADS), thread_name_prefix="frame")
        pose_batch = int(os.environ.get("FA_ENGINE_POSE_BATCH_SIZE", "64"))
        recog_batch = int(os.environ.get("FA_ENGINE_RECOG_BATCH_SIZE", "128"))
        if min(pose_batch, recog_batch) < 1:
            raise ValueError("Engine pose/recognition batch sizes must be positive")
        self.pose = HeadPose(entries[POSE]["path"], batch_size=pose_batch)
        self.rec = Recognizer(entries[recog_name]["path"], batch_size=recog_batch)
        self.pad = PAD(entries[PAD_MODEL]["path"])
        self.pad_validation = PADValidationGate(entries[PAD_MODEL]["path"],
            release_pub=os.environ.get("FA_RELEASE_PUB", "/run/secrets/release_pub"),
            detector_dir=entries[DETECTOR]["path"])
        self.candidate_binding = candidate_binding(detector_dir=entries[DETECTOR]["path"])
        self.pad_pilot = None
        self.pad_experiment = None
        if PAD_MODE == "femto_geometry_pilot":
            path, digest = os.environ.get("FA_FEMTO_PAD_PROFILE"), os.environ.get("FA_FEMTO_PAD_PROFILE_SHA256")
            if path and digest:
                from faceauth.common.femto_pad_profile import FemtoPADProfile
                self.pad_pilot = FemtoPADProfile(path, digest)
        elif PAD_MODE == "femto_geometry_experimental":
            path, digest = os.environ.get("FA_FEMTO_PAD_PROFILE"), os.environ.get("FA_FEMTO_PAD_PROFILE_SHA256")
            if path and digest:
                from faceauth.common.femto_pad_experiment import FemtoPADExperiment
                self.pad_experiment = FemtoPADExperiment(path, digest)
        self.versions = {"detector": self.det.version + ("+trt16" if trt else ""), "pose": self.pose.version,
                         "recog": self.rec.version, "align": self.rec.align, "pad": self.pad.version,
                         "quality": QUALITY}
        if PAD_MODE == "femto_geometry_experimental":
            digest = getattr(self.pad_experiment, "profile_sha256", None) or "unconfigured"
            self.versions["pad"] = PAD_MODE + "@" + digest
        self.providers = got
        self.warmup = self._warmup()

    def pilot_status(self, calibration):
        profile = getattr(self, "pad_pilot", None)
        if profile is None:
            return {"ready": False, "reasons": ["pad_pilot_profile_missing"], "profile_sha256": None,
                    "scope": ["print", "replay"], "experimental": True}
        return profile.status(calibration, self.candidate_binding)

    def experiment_status(self, calibration):
        """Configuration integrity is not empirical PAD validation."""
        profile = getattr(self, "pad_experiment", None)
        if profile is None:
            return {"ready": False, "reasons": ["pad_experiment_profile_missing"], "profile_sha256": None,
                    "calib_version": None, "scope": ["print", "replay"], "experimental": True,
                    "production_validated": False, "empirical_validation": "not_evaluated"}
        return profile.status(calibration, self.candidate_binding)

    @staticmethod
    def _candidate(face):
        if face is None:
            return clean_candidate_measurement(None)
        try:
            return clean_candidate_measurement(measure_features(face[1], face[2], face[3]["kps"]))
        except Exception:  # Measurements must fail closed without dropping the failed presentation.
            return clean_candidate_measurement({"schema": CANDIDATE_SCHEMA, "valid": False,
                                                "reasons": ["measurement_error"], "features": {}})

    def _warmup(self):
        """Exercise every GPU model on generated tensors before advertising readiness.

        No sensor frames, PAD decisions, templates or sessions are created. Model
        construction alone does not expose CUDA failures during the first forward pass.
        """
        started = time.monotonic()
        self.det.detect(np.zeros((480, 640, 3), np.uint8), DET_THRESH)
        pose_count = min(8, self.pose.batch_size)
        recog_count = min(8, self.rec.batch_size)
        poses = self.pose.predict_crops([np.zeros((256, 256, 3), np.uint8)] * pose_count, 256)
        embeddings = self.rec.embed_aligned(np.zeros((recog_count, 112, 112, 3), np.uint8))
        pad = self.pad.cnn(np.zeros((8, 6, self.pad.size, self.pad.size), np.float32))
        if (poses.shape != (pose_count, 3) or embeddings.shape != (recog_count, self.rec.dim)
                or pad.shape != (8,) or not all(np.isfinite(x).all() for x in (poses, embeddings, pad))):
            raise RuntimeError("GPU model warm-up returned invalid outputs")
        return {"synthetic": True, "detector": True, "pose": True, "recognition": True, "pad": True,
                "pose_batch": pose_count, "recognition_batch": recog_count, "pad_batch": 8,
                "elapsed_ms": round((time.monotonic() - started) * 1000, 1)}

    def _detect(self, dec):
        """Visible foreground faces, ordered by area with a deterministic coordinate tie break."""
        rgb, _, _ = dec
        det = self.dets.get()
        try:
            faces = det.detect(rgb, DET_THRESH)
        finally:
            self.dets.put(det)
        h, w = rgb.shape[:2]
        big = []
        for face in faces:
            box = np.asarray(face["bbox"], dtype=np.float32).copy()
            if box.shape != (4,) or not np.isfinite(box).all():
                continue
            box[::2] = np.clip(box[::2], 0, w)
            box[1::2] = np.clip(box[1::2], 0, h)
            if min(box[2] - box[0], box[3] - box[1]) >= OTHER_FACE_PX:
                big.append({**face, "bbox": box})
        return sorted(big, key=lambda f: (
            -float((f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1])),
            *(float(v) for v in f["bbox"]), -float(f["score"])))

    @staticmethod
    def _empty_face(n_faces):
        return {"n_faces": n_faces, "bbox": None, "kps": None, "det_score": 0.0, "pose": None, "pad": None,
               "emb": None, "quality": {}}

    def _measure_face(self, dec, f, *, depth_range=DEPTH_MM):
        """Measure one face without changing its identity when a quality check fails."""
        rgb, depth, ir = dec
        out = self._empty_face(1)
        x1, y1, x2, y2 = (int(round(v)) for v in f["bbox"])
        h, w = depth.shape
        x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
        blur, bright = face_crop_stats(rgb, f["bbox"])
        d = depth[y1:y2, x1:x2]
        valid = d[d > 0]
        q = {"face_px": float(min(f["bbox"][2] - f["bbox"][0], f["bbox"][3] - f["bbox"][1])), "blur": blur,
             "brightness": bright, "depth_valid": float(valid.size / max(d.size, 1)),
             "depth_mm": float(np.median(valid)) if valid.size else 0.0,
             "ir_median": float(np.median(ir[y1:y2, x1:x2])) if d.size else 0.0}
        reasons = []
        if not q["face_px"] >= MIN_FACE_PX:
            reasons.append("small_face")
        if not blur >= BLUR_MIN:
            reasons.append("blur")
        if not EXPOSURE[0] <= bright <= EXPOSURE[1]:
            reasons.append("exposure")
        if not (q["depth_valid"] >= DEPTH_VALID and depth_range[0] <= q["depth_mm"] <= depth_range[1]):
            reasons.append("depth_invalid")
        if not IR_RANGE[0] <= q["ir_median"] <= IR_RANGE[1]:
            reasons.append("ir_invalid")
        out.update(bbox=[float(v) for v in f["bbox"]], kps=f["kps"].tolist(), det_score=f["score"], quality=q,
                   reasons=reasons)
        return out, (rgb, depth, ir, f)

    def _frame(self, dec, purpose="enroll"):
        """Enrollment requires one face; authentication always measures the largest face."""
        if isinstance(dec, str):
            return {"ok": False, "reasons": [dec], "n_faces": 0}, None
        big = self._detect(dec)
        if not big or (purpose != "auth" and len(big) != 1):
            out = self._empty_face(len(big))
            out.update(ok=False, reasons=["no_face" if not big else "multi_face"])
            if purpose == "auth":
                out.update(detected_faces=len(big), selection="largest")
            return out, None
        out, face = self._measure_face(dec, big[0])
        if purpose == "auth":
            out.update(detected_faces=len(big), selection="largest")
        return out, face

    def _live_frame(self, dec):
        """Measure each visible face separately; raw coordinates stay in the sensor frame."""
        if isinstance(dec, str):
            return {"ok": False, "reasons": [dec], "faces": [], "n_faces": 0,
                    "image_size": None, "truncated": False}, []
        faces = self._detect(dec)
        measured = [self._measure_face(dec, face, depth_range=LIVE_DEPTH_MM) for face in faces[:MAX_LIVE_FACES]]
        h, w = dec[0].shape[:2]
        return {"ok": True, "reasons": [], "faces": [r for r, _ in measured], "n_faces": len(faces),
                "image_size": [w, h], "truncated": len(faces) > MAX_LIVE_FACES}, measured

    def _prepare_frame(self, item):
        context, dec = item
        if context["mode"] == "live":
            return self._live_frame(dec)
        purpose = (context.get("capture_purpose") if context["mode"] in ("guide", "observe")
                   else context["mode"])
        result, face = self._frame(dec, purpose)
        return result, [(result, face)]

    def process(self, batch):
        """batch: [(capture context, decoded frames)] -> per-request measurements."""
        flat = [(context, d) for context, frames in batch for d in frames]
        prepared = list(self.pool.map(self._prepare_frame, flat))
        results = [result for result, _ in prepared]
        measured = [(context, r, face) for (context, _), (_, items) in zip(flat, prepared) for r, face in items]
        contexts, res, faces = zip(*measured) if measured else ((), (), ())
        candidate_idx = [i for i, context in enumerate(contexts) if context["mode"] == "observe"
                         or (PAD_MODE in GEOMETRY_MODES and context["mode"] == "auth")]
        if candidate_idx:
            measurements = list(self.pool.map(self._candidate, [faces[i] for i in candidate_idx]))
            for i, measurement in zip(candidate_idx, measurements):
                if contexts[i]["mode"] == "observe":
                    res[i]["pad_candidate"] = measurement
                elif faces[i] is not None:
                    profile = self.pad_experiment if PAD_MODE == "femto_geometry_experimental" else self.pad_pilot
                    pad = profile.decide(measurement)
                    res[i]["pad"] = {key: pad[key] for key in ("score", "live", "nonresponse")}
                    if PAD_MODE == "femto_geometry_experimental":
                        # Internal API diagnostics only: carry condition names, never
                        # feature values/crops. This does not change the PAD decision.
                        failed = pad.get("failed")
                        res[i]["pad"]["failed"] = sorted({
                            reason for reason in (failed[:32] if isinstance(failed, list) else [])
                            if isinstance(reason, str) and reason in PAD_DIAGNOSTIC_REASONS
                        })
        idx = [i for i, f in enumerate(faces) if f is not None]
        if idx:
            crop, resize = CROPS[self.pose.crop]
            fs = [faces[i] for i in idx]
            poses = self.pose.predict_crops([crop(r, f["bbox"]) for r, _, _, f in fs], resize)
            for j, i in enumerate(idx):
                r = res[i]
                r["pose"] = [float(v) for v in poses[j]]
                context = contexts[i]
                purpose = (context.get("capture_purpose") if context["mode"] in ("guide", "observe")
                           else context["mode"])
                if purpose in ("auth", "live") and not (abs(poses[j][0]) <= AUTH_MAX_POSE[0]
                                               and abs(poses[j][1]) <= AUTH_MAX_POSE[1]):
                    r["reasons"].append("pose_out_of_range")
                r["score"] = r["det_score"] * min(1.0, r["quality"]["blur"] / (3 * BLUR_MIN))
            recog_idx = [i for i in idx if contexts[i]["mode"] in ("enroll", "auth")
                         or (contexts[i]["mode"] == "live" and not res[i]["reasons"])]
            if recog_idx:
                recog_faces = [faces[i] for i in recog_idx]
                embs = self.rec.embed([r for r, _, _, _ in recog_faces], [f["kps"] for _, _, _, f in recog_faces])
                for i, emb in zip(recog_idx, embs):
                    res[i]["emb"] = emb.astype("<f4").tobytes()
            pad_idx = [i for i in idx if (contexts[i]["mode"] == "observe" and PAD_MODE != "femto_geometry_experimental")
                       or (PAD_ENABLED and PAD_MODE not in GEOMETRY_MODES
                           and contexts[i]["mode"] == "auth")]
            if pad_idx:
                p = self.pad
                feats = list(self.pool.map(lambda x: face_features(x[0], x[1], x[2], x[3]["kps"],
                                                                  p.size, p.zoom, p.fx), [faces[i] for i in pad_idx]))
                pads = [p.decide(c, ph) for c, (_, ph) in zip(p.cnn(np.stack([x for x, _ in feats])), feats)]
                for i, pad in zip(pad_idx, pads):
                    res[i]["pad"] = _finite_measurements(pad) if contexts[i]["mode"] == "observe" else {
                        key: pad[key] for key in ("score", "live", "nonresponse")}
        for r in res:
            r["ok"] = not r["reasons"]
        out, k = [], 0
        for _, frames in batch:
            out.append(results[k:k + len(frames)])
            k += len(frames)
        return out

    def analyze_batch(self, batch):
        """Gate real decisions and bind every response to the exact models used."""
        output, allowed, positions = [], [], []
        for context, frames in batch:
            validation = self.pad_validation.status(context["calib_version"])
            output.append({"versions": dict(self.versions), "dim": self.rec.dim,
                           "pad_enabled": PAD_ENABLED, "pad_mode": PAD_MODE,
                           "capture_purpose": context.get("capture_purpose"),
                           "calib_version": context["calib_version"], "pad_validation": validation,
                           "model_binding": dict(self.pad_validation.binding)})
            if context["mode"] == "observe":
                output[-1]["candidate_binding"] = dict(self.candidate_binding)
            pilot = self.pilot_status(context["calib_version"]) if PAD_MODE == "femto_geometry_pilot" else None
            experiment = (self.experiment_status(context["calib_version"])
                          if PAD_MODE == "femto_geometry_experimental" else None)
            if pilot is not None:
                output[-1]["pad_pilot_validation"] = pilot
                output[-1]["candidate_binding"] = dict(self.candidate_binding)
                if context["mode"] in ("auth", "enroll"):
                    output[-1]["versions"]["pad"] = "femto_geometry_pilot@" + (pilot.get("profile_sha256") or "unconfigured")
            if experiment is not None:
                output[-1]["pad_experiment"] = experiment
                output[-1]["candidate_binding"] = dict(self.candidate_binding)
                if context["mode"] in ("auth", "enroll"):
                    output[-1]["versions"]["pad"] = PAD_MODE + "@" + (experiment.get("profile_sha256") or "unconfigured")
            ready = (experiment["ready"] if experiment is not None else
                     pilot["ready"] if pilot is not None else validation["ready"])
            # Enrollment is an administrator-authorized template operation, not
            # an authentication assertion. Only the server-selected auth mode
            # requires PAD; capture_purpose can never override that boundary.
            if PAD_ENABLED and context["mode"] == "auth" and ready is not True:
                output[-1]["frames"] = [{"ok": False, "reasons": ["pad_validation_required"],
                                          "n_faces": 0, "pad": None, "emb": None} for _ in frames]
            else:
                positions.append(len(output) - 1)
                allowed.append((context, frames))
        if allowed:
            for position, measured in zip(positions, self.process(allowed)):
                output[position]["frames"] = measured
        return output


def _decode(f) -> tuple | str:
    try:
        return decode_frame(f)
    except ValueError as e:
        msg = str(e)
        return "missing_modality" if msg.startswith("missing_") else (
            "unregistered_frame_size" if msg == "unregistered_frame_size" else "decode_error")
    except Exception:  # noqa: BLE001 - any garbage from the wire is a decode error, never a crash
        return "decode_error"


def load_models() -> Models:
    """Signed registry + active versions from the DB -> Models (raises on anything unverifiable)."""
    entries = load_registry(os.environ.get("FA_MODELS_DIR", "/app/data/models"),
                            os.environ.get("FA_RELEASE_PUB", "/run/secrets/release_pub"))
    with psycopg.connect(os.environ["FA_DB_DSN"], password=_env_file("FA_DB_PASSWORD_FILE")) as conn:
        active = active_entries(entries, conn)
    return Models(active, os.environ.get("FA_RECOG_MODEL", "kprpe_vitb_wf12m"),
                  os.environ.get("FA_ENGINE_TRT", "1") == "1")


class Engine:
    def __init__(self):
        self.models: Models | None = None
        self.queue: asyncio.Queue = asyncio.Queue()
        self.gpu = ThreadPoolExecutor(1, thread_name_prefix="gpu")   # model calls are serialized on one worker
        self.cpu = ThreadPoolExecutor(8, thread_name_prefix="decode")
        self.admin_token = _env_file("FA_ADMIN_TOKEN_FILE")
        self.batches = self.frames = 0

    async def analyze(self, context, frames):
        loop = asyncio.get_running_loop()
        decoded = await asyncio.gather(*(loop.run_in_executor(self.cpu, _decode, f) for f in frames))
        fut = loop.create_future()
        await self.queue.put((context, decoded, fut))
        return await fut

    async def worker(self):
        loop = asyncio.get_running_loop()
        while True:
            items = [await self.queue.get()]
            n = len(items[0][1])
            while not self.queue.empty() and n < MAX_BATCH_FRAMES:
                items.append(self.queue.get_nowait())
                n += len(items[-1][1])
            try:
                models = self.models
                outs = await loop.run_in_executor(self.gpu, models.analyze_batch, [(m, d) for m, d, _ in items])
                for (_, _, fut), o in zip(items, outs):
                    if not fut.done():
                        fut.set_result(o)
                self.batches, self.frames = self.batches + 1, self.frames + n
            except Exception as e:  # noqa: BLE001 - fail the requests (API answers 503), keep serving
                for _, _, fut in items:
                    if not fut.done():
                        fut.set_exception(e)


E = Engine()


@asynccontextmanager
async def lifespan(app):
    t = time.time()
    E.models = await asyncio.get_running_loop().run_in_executor(E.gpu, load_models)
    print(f"engine ready in {time.time() - t:.1f}s versions={E.models.versions} providers={E.models.providers}",
          flush=True)
    task = asyncio.create_task(E.worker())
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan)
MSGPACK = "application/msgpack"


@app.post("/v1/analyze")
async def analyze(request: Request):
    try:
        body = msgpack.unpackb(await request.body(), raw=False)
        mode, frames = body["mode"], body["frames"]
        calibration = body.get("calib_version")
        purpose = body.get("capture_purpose")
        if (mode not in ("enroll", "auth", "guide", "observe", "live") or not isinstance(frames, list)
                or not 0 < len(frames) <= 64 or not isinstance(calibration, str) or not 0 < len(calibration) <= 64
                or (mode == "live" and len(frames) != 1)
                or (mode == "guide" and (len(frames) != 1 or purpose not in ("auth", "enroll")))
                or (mode == "observe" and (purpose not in ("auth", "enroll")
                    or not 3 <= len(frames) <= (24 if purpose == "enroll" else 16)))):
            raise ValueError
    except Exception:  # noqa: BLE001
        raise HTTPException(400, "malformed request") from None
    if (body.get("pad_enabled", True) is not PAD_ENABLED
            or body.get("pad_mode", "enabled") != PAD_MODE):
        raise HTTPException(503, "pad_mode_mismatch")
    out = await E.analyze({"mode": mode, "calib_version": calibration, "capture_purpose": purpose}, frames)
    return Response(msgpack.packb(out), media_type=MSGPACK)


@app.get("/v1/readiness")
async def readiness(calib_version: str):
    m = E.models
    if m is None:
        raise HTTPException(503, "models_not_ready")
    validation = m.pad_validation.status(calib_version)
    pilot = m.pilot_status(calib_version) if PAD_MODE == "femto_geometry_pilot" else None
    experiment = m.experiment_status(calib_version) if PAD_MODE == "femto_geometry_experimental" else None
    configured = (experiment["ready"] if experiment is not None else
                  pilot["ready"] if pilot is not None else validation["ready"])
    state = {"ready": not PAD_ENABLED or configured,
            "capture_ready": True, "pad_validation": validation,
            "pad_enabled": PAD_ENABLED, "pad_mode": PAD_MODE,
            "versions": m.versions, "model_binding": m.pad_validation.binding, "calib_version": calib_version,
            "candidate_binding": dict(m.candidate_binding)}
    if pilot is not None:
        state["pad_pilot_validation"] = pilot
    if experiment is not None:
        state["pad_experiment"] = experiment
    return state


@app.get("/healthz")
async def healthz():
    m = E.models
    return {"ok": m is not None, "versions": m and m.versions, "providers": m and m.providers,
            "pad_enabled": PAD_ENABLED, "pad_mode": PAD_MODE,
            "warmup": m and m.warmup,
            "batches": E.batches, "frames": E.frames}


@app.post("/v1/admin/reload")
async def reload(request: Request):
    if E.admin_token and request.headers.get("authorization") != f"Bearer {E.admin_token}":
        raise HTTPException(401)
    try:  # build the new set on the GPU worker (queued behind running batches), then swap
        E.models = await asyncio.get_running_loop().run_in_executor(E.gpu, load_models)
    except Exception as e:  # noqa: BLE001 - keep serving the old verified set; operator sees the failure
        raise HTTPException(503, f"reload refused: {e}") from None
    return {"ok": True, "versions": E.models.versions}
