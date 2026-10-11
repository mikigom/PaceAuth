"""Engine: central GPU inference service (docs/IMPLEMENTATION.md §3).

POST /v1/analyze (msgpack) {"mode": "enroll"|"auth", "frames": [{"rgb","depth","ir"}]} -> per frame: detection,
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

from faceauth.common.packet import decode_frame
from faceauth.common.registry import active_entries, load_registry
from faceauth.models.headpose import CROPS, HeadPose
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
AUTH_MAX_POSE = (25.0, 20.0)
MAX_BATCH_FRAMES = 64
# Per-replica parallelism (measured: one serial worker = ~90 frames/s, SCRFD 4.9 + PAD 2.3 ms/frame on the CPU-bound
# path). Several SCRFD sessions = several CUDA streams + concurrent pre/post-processing; PAD physics on threads.
DET_SESSIONS = int(os.environ.get("FA_ENGINE_DET_SESSIONS", "2"))   # SCRFD-34G tops out ~350 frames/s per A6000
CPU_THREADS = int(os.environ.get("FA_ENGINE_CPU_THREADS", "6"))


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
        self.pose = HeadPose(entries[POSE]["path"])
        self.rec = Recognizer(entries[recog_name]["path"])
        self.pad = PAD(entries[PAD_MODEL]["path"])
        self.versions = {"detector": self.det.version + ("+trt16" if trt else ""), "pose": self.pose.version,
                         "recog": self.rec.version, "align": self.rec.align, "pad": self.pad.version,
                         "quality": QUALITY}
        self.providers = got

    def _frame(self, dec):
        """Detection + sensor/quality checks for one decoded frame -> (result dict, face for later stages | None)."""
        if isinstance(dec, str):
            return {"ok": False, "reasons": [dec], "n_faces": 0}, None
        rgb, depth, ir = dec
        det = self.dets.get()
        try:
            faces = det.detect(rgb, DET_THRESH)
        finally:
            self.dets.put(det)
        big = [f for f in faces if min(f["bbox"][2] - f["bbox"][0], f["bbox"][3] - f["bbox"][1]) >= OTHER_FACE_PX]
        out = {"n_faces": len(big), "bbox": None, "kps": None, "det_score": 0.0, "pose": None, "pad": None,
               "emb": None, "quality": {}}
        if len(big) != 1:
            out.update(ok=False, reasons=["no_face" if not big else "multi_face"])
            return out, None
        f = big[0]
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
        if not (q["depth_valid"] >= DEPTH_VALID and DEPTH_MM[0] <= q["depth_mm"] <= DEPTH_MM[1]):
            reasons.append("depth_invalid")
        if not IR_RANGE[0] <= q["ir_median"] <= IR_RANGE[1]:
            reasons.append("ir_invalid")
        out.update(bbox=[float(v) for v in f["bbox"]], kps=f["kps"].tolist(), det_score=f["score"], quality=q,
                   reasons=reasons)
        return out, (rgb, depth, ir, f)

    def process(self, batch):
        """batch: [(mode, [decoded frame | error str])] -> per request, the list of per-frame answers."""
        flat = [(mode, d) for mode, frames in batch for d in frames]
        res, faces = zip(*self.pool.map(self._frame, [d for _, d in flat])) if flat else ((), ())
        idx = [i for i, f in enumerate(faces) if f is not None]
        if idx:
            crop, resize = CROPS[self.pose.crop]
            fs = [faces[i] for i in idx]
            poses = self.pose.predict_crops([crop(r, f["bbox"]) for r, _, _, f in fs], resize)
            embs = self.rec.embed([r for r, _, _, _ in fs], [f["kps"] for _, _, _, f in fs])
            p = self.pad
            feats = list(self.pool.map(lambda x: face_features(x[0], x[1], x[2], x[3]["kps"], p.size, p.zoom, p.fx), fs))
            pads = [p.decide(c, ph) for c, (_, ph) in zip(p.cnn(np.stack([x for x, _ in feats])), feats)]
            for j, i in enumerate(idx):
                r = res[i]
                r["pose"] = [float(v) for v in poses[j]]
                r["emb"] = embs[j].astype("<f4").tobytes()
                r["pad"] = {"score": pads[j]["score"], "live": pads[j]["live"], "nonresponse": pads[j]["nonresponse"]}
                if flat[i][0] == "auth" and not (abs(poses[j][0]) <= AUTH_MAX_POSE[0]
                                                  and abs(poses[j][1]) <= AUTH_MAX_POSE[1]):
                    r["reasons"].append("pose_out_of_range")
        for r in res:
            r["ok"] = not r["reasons"]
        out, k = [], 0
        for _, frames in batch:
            out.append(list(res[k:k + len(frames)]))
            k += len(frames)
        return out


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

    async def analyze(self, mode, frames):
        loop = asyncio.get_running_loop()
        decoded = await asyncio.gather(*(loop.run_in_executor(self.cpu, _decode, f) for f in frames))
        fut = loop.create_future()
        await self.queue.put((mode, decoded, fut))
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
                outs = await loop.run_in_executor(self.gpu, self.models.process, [(m, d) for m, d, _ in items])
                for (_, _, fut), o in zip(items, outs):
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
        if mode not in ("enroll", "auth") or not isinstance(frames, list) or not 0 < len(frames) <= 64:
            raise ValueError
    except Exception:  # noqa: BLE001
        raise HTTPException(400, "malformed request") from None
    out = await E.analyze(mode, frames)
    m = E.models
    return Response(msgpack.packb({"versions": m.versions, "dim": m.rec.dim, "frames": out}), media_type=MSGPACK)


@app.get("/healthz")
async def healthz():
    m = E.models
    return {"ok": m is not None, "versions": m and m.versions, "providers": m and m.providers,
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
