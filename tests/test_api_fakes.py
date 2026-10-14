"""Contract fakes for the API integration tests: engine (§3), search (§4), KMS (§5) + test controls. No tests here.

One process serves the same app on FAKE_PORTS (default 8001,8002,8003) with network aliases engine, search-1,
search-2, kms (deploy/compose.api.yml, profile apitest):   python -m tests.test_api_fakes
Frames are not images: the fake engine reads a JSON scenario from each frame's "rgb" bytes:
    {"id": "alice" | {"alice": 1, "bob": 0.1}, "pose": [yaw, pitch, roll], "live": true, "faces": 1,
     "reasons": [], "noise": 0.05, "seed": 1}
Search follows the real contract: index of active templates of the active recognition version, built from DB
ciphertexts (KMS-decrypted), updated from template_events (id > applied); POST /_test/state {"search_frozen": true}
stops applying events (replica lag), {"engine_fail"|"search_fail"|"kms_fail": true} makes that service fail.
"""
import asyncio
import hashlib
import json
import os
from pathlib import Path

import msgpack
import numpy as np
import psycopg
import uvicorn
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from fastapi import FastAPI, Request, Response

DIM = 512
SECRETS = Path(os.environ.get("FA_SECRETS_DIR", "/app/secrets"))
RECOG = os.environ.get("FA_RECOG_MODEL", "kprpe_vitb_wf12m")
STATE = {"engine_fail": False, "search_fail": False, "kms_fail": False, "search_frozen": False,
         "pad_ready": True, "pad_enabled": os.environ.get("FA_PILOT_DISABLE_PAD", "0") != "1"}
VERSION = {"recog": None}                                  # engine's loaded recognition version
INDEX = {"model_version": None, "applied": 0, "tpl": {}}    # search: template_id -> (subject_id, vector)
KEYS = {}                                                   # kms: subject_id -> 32-byte key
app = FastAPI()


class _Conn:  # one persistent connection (single event loop), reopened if it broke
    c = None

    def __enter__(self):
        if _Conn.c is None or _Conn.c.closed or _Conn.c.broken:
            _Conn.c = psycopg.connect(os.environ.get("FAKE_DB_DSN", "host=db dbname=faceauth user=faceauth_api"),
                                      password=(SECRETS / "db_api_password").read_text().strip(), autocommit=True)
        return _Conn.c

    def __exit__(self, *exc):
        return False


def _db():
    return _Conn()


def _active_version():
    with _db() as c:
        row = c.execute("SELECT version FROM model_versions WHERE name = %s AND active", (RECOG,)).fetchone()
    return f"{RECOG}@{row[0]}" if row else None


def _pack(obj, status=200):
    return Response(msgpack.packb(obj), status_code=status, media_type="application/msgpack")


def _bearer(request, name):
    return request.headers.get("authorization") == f"Bearer {(SECRETS / name).read_text().strip()}"


@app.post("/_test/state")
async def set_state(request: Request):
    STATE.update(await request.json())
    return STATE


@app.get("/healthz")
async def healthz():
    return {"ok": True}


# ------------------------------------------------------------------ engine
def identity(spec) -> np.ndarray:
    weights = {spec: 1.0} if isinstance(spec, str) else spec
    v = sum(w * np.random.default_rng(int.from_bytes(hashlib.sha256(k.encode()).digest()[:8], "little"))
            .standard_normal(DIM) for k, w in weights.items())
    return v / np.linalg.norm(v)


def _frame(f) -> dict:
    out = {"ok": False, "reasons": [], "n_faces": 0, "bbox": None, "kps": None, "det_score": 0.0, "pose": None,
           "quality": {}, "pad": None, "emb": None}
    if not (f.get("rgb") and f.get("depth") and f.get("ir")):
        return {**out, "reasons": ["missing_modality"]}
    try:
        s = json.loads(f["rgb"])
    except ValueError:
        return {**out, "reasons": ["decode_error"]}
    faces, reasons = int(s.get("faces", 1)), list(s.get("reasons", []))
    reasons += ["no_face"] if faces == 0 else ["multi_face"] if faces > 1 else []
    if faces == 1:
        noise = np.random.default_rng(s.get("seed")).standard_normal(DIM)
        v = identity(s["id"]) + float(s.get("noise", 0.05)) * noise / np.linalg.norm(noise)
        live = bool(s.get("live", True))
        out.update(bbox=[200.0, 120.0, 440.0, 400.0], kps=[[270, 230], [370, 230], [320, 290], [280, 340], [360, 340]],
                   det_score=0.9, score=0.9, pose=list(s.get("pose", [0.0, 0.0, 0.0])),
                   pad={"score": 0.9 if live else 0.1, "live": live, "nonresponse": False,
                        "cnn": 0.9 if live else 0.1, "checks": {"depth": True}, "failed": []},
                   emb=(v / np.linalg.norm(v)).astype("<f4").tobytes())
    return {**out, "n_faces": faces, "reasons": reasons, "ok": not reasons}


def _analyze_frame(frame, mode, capture_purpose):
    try:
        spec = json.loads(frame["rgb"])
    except (ValueError, KeyError):
        return _frame(frame)
    people = spec.get("people")
    if people is None:
        people = [{**spec, "faces": 1}] * int(spec.get("faces", 1))
    people = sorted(people, key=lambda s: -(lambda b: (b[2] - b[0]) * (b[3] - b[1]))(
        s.get("bbox", [200, 120, 440, 400])))

    def face(person):
        out = _frame({**frame, "rgb": json.dumps({**person, "faces": 1}).encode()})
        out["bbox"] = person.get("bbox", out["bbox"])
        return out

    if mode == "live":
        faces = [face(person) for person in people[:16]]
        for item in faces:
            item["pad"] = None
            if not item["ok"]:
                item["emb"] = None
        return {"faces": faces, "n_faces": len(people), "image_size": [640, 480], "truncated": len(people) > 16}
    purpose = capture_purpose if mode in ("guide", "observe") else mode
    if purpose == "auth" and people:
        return {**face(people[0]), "detected_faces": len(people), "selection": "largest"}
    if "people" in spec:
        return _frame({**frame, "rgb": json.dumps({**spec, "faces": len(people)}).encode()})
    return _frame(frame)


@app.post("/v1/analyze")
async def analyze(request: Request):
    if STATE["engine_fail"]:
        return Response(status_code=500)
    req = msgpack.unpackb(await request.body(), raw=False)
    versions = {k: "fake@1" for k in ("detector", "pose", "landmarks", "align", "pad", "quality")}
    frames = [_analyze_frame(f, req["mode"], req.get("capture_purpose")) for f in req["frames"]]
    if not STATE["pad_enabled"] and req["mode"] in ("auth", "enroll"):
        for frame in frames:
            frame["pad"] = None
    return _pack({"versions": {**versions, "recog": VERSION["recog"] or "none"}, "dim": DIM,
                  "pad_enabled": STATE["pad_enabled"],
                  "pad_mode": "enabled" if STATE["pad_enabled"] else "disabled_pilot",
                  "calib_version": req["calib_version"], "pad_validation": _validation(req["calib_version"]),
                  "capture_purpose": req.get("capture_purpose"),
                  "model_binding": {"fixture": "api-contract-only"},
                  "frames": frames})


def _validation(calibration):
    return {"ready": STATE["pad_ready"], "calib_version": calibration,
            "reasons": [] if STATE["pad_ready"] else ["pad_real_sensor_validation_missing"]}


@app.get("/v1/readiness")
async def readiness(calib_version: str):
    if STATE["engine_fail"]:
        return Response(status_code=503)
    return {"ready": not STATE["pad_enabled"] or STATE["pad_ready"], "capture_ready": True,
            "pad_validation": _validation(calib_version), "pad_enabled": STATE["pad_enabled"],
            "pad_mode": "enabled" if STATE["pad_enabled"] else "disabled_pilot",
            "versions": {"recog": VERSION["recog"] or "none"}, "calib_version": calib_version}


# ------------------------------------------------------------------ search
def _add(tid, sid, ct):
    pt = _decrypt(str(sid), bytes(ct)) if ct is not None else None
    if pt is not None:
        INDEX["tpl"][tid] = (str(sid), np.frombuffer(pt, "<f4"))


def _rebuild():
    mv = _active_version()
    INDEX.update(model_version=mv, tpl={})
    with _db() as c:
        INDEX["applied"] = c.execute("SELECT coalesce(max(id), 0) FROM template_events").fetchone()[0]
        for tid, sid, ct in c.execute("SELECT id, subject_id, ciphertext FROM templates WHERE active AND "
                                      "model_version = %s", (mv,)):
            _add(tid, sid, ct)


def _sync():
    if STATE["search_frozen"] or INDEX["model_version"] is None:
        return
    with _db() as c:
        for eid, op, tid, sid, mv in c.execute("SELECT id, op, template_id, subject_id, model_version FROM "
                                               "template_events WHERE id > %s ORDER BY id", (INDEX["applied"],)):
            if op == "add" and mv == INDEX["model_version"]:
                row = c.execute("SELECT ciphertext FROM templates WHERE id = %s AND active", (tid,)).fetchone()
                if row:
                    _add(tid, sid, row[0])
            elif op == "remove":
                INDEX["tpl"].pop(tid, None)
            INDEX["applied"] = eid


@app.get("/v1/status")
async def status():
    if STATE["search_fail"]:
        return Response(status_code=503)
    _sync()
    subjects = {s for s, _ in INDEX["tpl"].values()}
    return {"ready": INDEX["model_version"] is not None, "model_version": INDEX["model_version"],
            "ntotal": len(INDEX["tpl"]), "nsubjects": len(subjects), "applied_event_id": INDEX["applied"]}


@app.post("/v1/search")
async def search(request: Request):
    if STATE["search_fail"] or INDEX["model_version"] is None:
        return Response(status_code=503)
    q = msgpack.unpackb(await request.body(), raw=False)
    if q["model_version"] != INDEX["model_version"]:
        return Response(status_code=409)
    _sync()
    probes = np.frombuffer(q["probes"], "<f4").reshape(int(q["n"]), -1)
    v = probes.mean(0)
    v /= np.linalg.norm(v)
    best = {}
    for sid, t in INDEX["tpl"].values():
        best[sid] = max(best.get(sid, -2.0), float(v @ t))
    top = sorted(best.items(), key=lambda x: -x[1])[: int(q.get("k", 5))]
    return _pack({"results": [{"subject_id": s, "score": sc} for s, sc in top], "applied_event_id": INDEX["applied"]})


@app.post("/v1/admin/reload")
async def reload():  # engine + search share this process: both reload
    VERSION["recog"] = _active_version()
    _rebuild()
    return {"ok": True, "recog": VERSION["recog"], "ntotal": len(INDEX["tpl"])}


# ------------------------------------------------------------------ kms
def _decrypt(subject_id, ct):
    key = KEYS.get(subject_id)
    if key is None:
        return None
    return AESGCM(key).decrypt(ct[:12], ct[12:], subject_id.encode())


@app.post("/v1/encrypt")
async def encrypt(request: Request):
    if STATE["kms_fail"]:
        return Response(status_code=500)
    if not _bearer(request, "kms_token_api"):
        return Response(status_code=403)
    req = msgpack.unpackb(await request.body(), raw=False)
    sid = req["subject_id"]
    key = KEYS.setdefault(sid, AESGCM.generate_key(256))
    items = []
    for pt in req["items"]:
        nonce = os.urandom(12)
        items.append(nonce + AESGCM(key).encrypt(nonce, pt, sid.encode()))
    return _pack({"items": items})


@app.post("/v1/decrypt")
async def decrypt(request: Request):
    if STATE["kms_fail"]:
        return Response(status_code=500)
    if not _bearer(request, "kms_token_search"):
        return Response(status_code=403)
    req = msgpack.unpackb(await request.body(), raw=False)
    return _pack({"items": [_decrypt(i["subject_id"], i["ct"]) for i in req["items"]]})


@app.delete("/v1/keys/{subject_id}")
async def destroy(subject_id: str, request: Request):
    if STATE["kms_fail"]:
        return Response(status_code=500)
    if not _bearer(request, "kms_token_api"):
        return Response(status_code=403)
    return {"destroyed": KEYS.pop(subject_id, None) is not None}


@app.get("/_test/kms/{subject_id}")
async def has_key(subject_id: str):
    return {"exists": subject_id in KEYS}


async def main():
    ports = [int(p) for p in os.environ.get("FAKE_PORTS", "8001,8002,8003").split(",")]
    await asyncio.gather(*(uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=p, log_level="warning")).serve()
                           for p in ports))


if __name__ == "__main__":
    asyncio.run(main())
