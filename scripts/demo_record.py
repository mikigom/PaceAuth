"""Record the demo scenario against the REAL stack, for the kiosk UI replay and the motion-graphic video.

    docker compose run --rm -u $(id -u):$(id -g) -e HOME=/tmp tester python scripts/demo_record.py

Everything shown later is what the system actually did: the RGB-D-IR frames the terminal analysed (SensorSim renders
of real BIWI head-rotation recordings), the guide's prompts and progress ring, the hands-free trigger, the server
decisions, the internal reasons and scores of the audit log (server side only; the terminal never receives them), the
approval token as the relying login service verifies it, and the measured latencies. Scenario:
  1. users A-D enroll with the Face ID-style continuous head sweep (real enrollment API)
  2. A and D come back on ANOTHER BIWI recording session, walk up, are identified hands-free (1:N) and logged in
  3. two people who never enrolled walk up and are rejected
  4. a silicone mask of D's face walks up and is rejected by PAD
Output: data/demo/timeline.json (+ timeline.js for file:// pages) and data/demo/frames/<id>_{rgb,dep,ir}.jpg
BIWI subjects are dataset volunteers; they are shown as "사용자 A-D" / "미등록자 1-2".
"""
import datetime
import json
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import psycopg  # noqa: E402
from cryptography.hazmat.primitives.serialization import load_pem_public_key  # noqa: E402

import enroll_sim  # noqa: E402
from faceauth.common import token  # noqa: E402
from faceauth.common.packet import build_packet  # noqa: E402
from faceauth.sim.biwi import iter_frames  # noqa: E402
from faceauth.sim.sensor import SensorSim  # noqa: E402
from faceauth.terminal.client import SECRETS, make_account_assertion, synced_ts  # noqa: E402
from faceauth.terminal.guide import Guide  # noqa: E402
from faceauth.terminal.pipeline import Pipeline, select_auth, wire_frame  # noqa: E402
from faceauth.terminal.proximity import ProximityTrigger  # noqa: E402

OUT = ROOT / "data/demo"
FPS = 30.0                      # sensor / display rate; the terminal analyses every 3rd frame (~10 fps, measured)
DSN = "host=db,db-standby1,db-standby2 dbname=faceauth user=faceauth_api target_session_attrs=read-write"
# label, BIWI subject, enrollment session, rendering seed; returning users come back on another recording session.
# Each user keeps ONE rendering (backdrop, clothes, skin) for the whole enrollment, chosen among seeds 100-123 so its
# frames clear the terminal blur check in the main directions (per-frame random renderings, as in E8, pass by chance).
USERS = [("A", "M01", ["07"], 111), ("B", "M03", ["09"], 106), ("C", "M07", ["13"], 110), ("D", "F03", ["03"], 104)]
RETURNS = [("A", "22"), ("D", "15")]
STRANGERS = [("미등록자 1", "12"), ("미등록자 2", "16")]   # BIWI M06, M09: never enrolled
STAND_MM = 600.0                # where people stop in front of the kiosk (trigger range 400-800 mm)
WINDOW = 10                     # frames captured after the trigger; the terminal sends the best 3 (as scripts/loadgen.py)


class Rec:
    def __init__(self):
        (OUT / "frames").mkdir(parents=True, exist_ok=True)
        self.n = 0

    def frame(self, o) -> str:
        """Save what the sensor delivered: RGB (as analysed), depth (colour map), IR."""
        fid = f"{self.n:05d}"
        self.n += 1
        cv2.imwrite(str(OUT / "frames" / f"{fid}_rgb.jpg"), cv2.cvtColor(o["rgb"], cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        d = o["depth"].astype(np.float32)
        v = np.clip((1500 - d) / 1100, 0, 1)                      # near = warm; 0 (invalid) -> black
        col = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        col[d == 0] = 0
        cv2.imwrite(str(OUT / "frames" / f"{fid}_dep.jpg"), cv2.resize(col, (320, 240)), [cv2.IMWRITE_JPEG_QUALITY, 85])
        cv2.imwrite(str(OUT / "frames" / f"{fid}_ir.jpg"), cv2.resize(o["ir"], (320, 240)), [cv2.IMWRITE_JPEG_QUALITY, 85])
        return fid


def server_view(conn, session_id) -> dict:
    """Internal reason + audit detail (top1/top2 for identifications): server side, never sent to the device."""
    reason = conn.execute("SELECT reason FROM sessions WHERE id = %s", (session_id,)).fetchone()[0]
    row = conn.execute("SELECT detail FROM audit_log WHERE session_id = %s AND event = 'auth_decision' ORDER BY id DESC "
                       "LIMIT 1", (session_id,)).fetchone()
    detail = row[0] if row else {}
    return {"internal": reason, "top1": detail.get("top1"), "top2": detail.get("top2")}


def checklist(reason: str) -> list:
    """Server pipeline steps for the operator view."""
    steps = ["서명·세션", "센서 동기·품질", "PAD (RGB·Depth·IR)", "1:N 식별", "계정·권한"]
    fail_at = {"ok": None, "pad_fail": 2, "no_match": 3, "ambiguous": 3, "subject_inactive": 4}.get(reason)
    if fail_at is None and reason != "ok":
        fail_at = 1 if reason.startswith("quality") or reason in ("missing_modality", "unsynchronized") else 0
    return [{"step": s, "status": "pass" if fail_at is None or i < fail_at else ("fail" if i == fail_at else "skip")}
            for i, s in enumerate(steps)]


def enroll_user(rec, label, subject, seqs, scene_seed, dev, conn, pipe, sim, k, tries=3):
    """Guided capture + enrollment; a failed attempt (timeout / rejection) is repeated like a person would."""
    pool, attempts = enroll_sim.subject_pool(seqs), []
    for attempt in range(tries):
        account = f"demo-{label.lower()}-{uuid.uuid4().hex[:6]}"
        rng = np.random.default_rng([7, k, attempt])
        speed, react, ticks = float(rng.uniform(30, 45)), 0.5, []

        def on_tick(tick, t, o, r, st):
            ticks.append({"t": round(t, 3), "frame": rec.frame(o),
                          "bbox": None if r["bbox"] is None else [round(float(v), 1) for v in r["bbox"]],
                          "pose": r["pose"] and [round(v, 1) for v in r["pose"]], "ok": r["ok"], "reasons": r["reasons"],
                          "prompt": st["prompt"], "ring": st["ring"], "accepted": st["accepted"],
                          "direction": st["direction"], "coverage": {d: v[0] for d, v in st["coverage"].items()},
                          "done": st["done"]})
        guide = Guide(timeout_s=60)
        done, sensor_s, kept, _ = enroll_sim.capture(sim, pipe, pool, {}, rng, speed, react, 10.0, guide, on_tick=on_tick,
                                                     scene_seed=scene_seed)    # same clothes and backdrop throughout
        seg = {"type": "enroll", "user": label, "subject": subject, "account": account, "ticks": ticks,
               "sensor_s": round(sensor_s, 2), "captured": done}
        if done:
            session = dev.create_session("enroll", "enroll")
            assertion = make_account_assertion(account, session["challenge"], "password", ttl_s=180)
            consent = {"version": "v1", "agreed_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
            t0 = time.perf_counter()
            r = dev.enroll(session, [enc for _, (_, enc) in kept],
                           synced_ts([i for _, (i, _) in kept], int(time.time() * 1000)), [d for d, _ in kept],
                           assertion, consent)
            seg["server"] = {"latency_s": round(time.perf_counter() - t0, 3), "response": r,
                             "frames_sent": len(kept), **server_view(conn, session["session_id"])}
        status = seg.get("server", {}).get("response", {}).get("status")
        attempts.append({"captured": done, "sensor_s": seg["sensor_s"], "status": status})
        print(label, subject, "enroll attempt", attempt, status, f"{sensor_s:.1f}s", flush=True)
        if status == "active":
            seg["attempts"] = attempts
            return seg
    raise SystemExit(f"{label} ({subject}) did not enroll in {tries} attempts: {attempts}")


def frontal_run(seq, max_angle=10.0, n=24):
    """Longest run of near-frontal, upright frames of a BIWI sequence (a person standing in front of the kiosk)."""
    run, best = [], []
    for f in iter_frames(str(ROOT / "data/datasets/biwi"), seqs=[seq], register=False, step=1):
        if all(abs(v) <= max_angle for v in f["pose"][:3]):
            run.append(f["rgb"])
            if len(run) > len(best):
                best = list(run)
            if len(best) >= n:
                break
        else:
            run = []
    return best


def approach(rec, label, kind, photos, dev, conn, pipe, sim, seed, approval_pub, action="login"):
    """A person walks up (1.5 m -> STAND_MM in 2.5 s) and stands; hands-free trigger -> capture window -> best 3
    frames (quality, frontal) -> signed packet -> server."""
    big = lambda dets: max(dets, key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]))  # noqa: E731
    kps = [big(pipe.detector.detect(p))["kps"] for p in photos]
    trig, rng = ProximityTrigger(), np.random.default_rng(seed)
    ticks, session, seg, fired_at = [], None, {"type": "auth", "user": label, "kind": kind}, None
    last = {"state": "idle", "distance_mm": None, "hint": "", "bbox": None, "reasons": []}
    for i in range(int(FPS * 8.0)):
        t = i / FPS
        dist = 1500 - (1500 - STAND_MM) * min(1.0, (t / 2.5)) ** 0.6 if t < 2.5 else STAND_MM
        j = i % len(photos)
        scene = sim.scene(kind, photos[j], kps[j], np.random.default_rng(seed), dist)   # same backdrop, new distance
        o = sim.capture([scene], rng)
        enc, rgb = wire_frame(o["rgb"], o["depth"], o["ir"])
        tk = {"t": round(t, 3), "frame": rec.frame(o)}
        if fired_at is None and i % 3 == 0:                      # terminal analyses ~10 fps
            r = pipe.analyze(rgb)
            p = trig.feed(r, o["depth"])
            last = {**p, "reasons": r["reasons"], "bbox": None if r["bbox"] is None else [round(float(v), 1) for v in r["bbox"]]}
            if p["state"] != "idle" and session is None:          # someone is coming: issue the challenge early
                session = dev.create_session("auth", action)
            if p["fire"]:
                fired_at, auth = i, []
        tk.update({k: last[k] for k in ("state", "hint", "bbox", "reasons")},
                  distance_mm=last["distance_mm"] and round(last["distance_mm"]))
        if fired_at is not None and i > fired_at and len(auth) < WINDOW and "server" not in seg:
            auth.append((enc, i, pipe.analyze(rgb)))                # capture AFTER the trigger (and the session)
            pick = select_auth([a[2] for a in auth]) if len(auth) == WINDOW else None
            if pick == []:                                          # no 3 good frames: capture another window
                auth = []
            elif pick:
                auth = [auth[j][:2] for j in pick]
                ts = synced_ts([k for _, k in auth], int(time.time() * 1000))
                pkt = build_packet(dev.key, session_id=session["session_id"], challenge=session["challenge"],
                                   device_id=dev.device_id, calib_version=dev.calib_version,
                                   frames=[e for e, _ in auth], ts_ms=ts, captured_at_ms=min(ts[0].values()))
                t0 = time.perf_counter()
                r = dev.http.post("/v1/authentications", content=pkt, headers={"Content-Type": "application/msgpack"},
                                  extensions={"sni_hostname": "gateway"})
                lat = time.perf_counter() - t0
                out = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
                sv = server_view(conn, session["session_id"])
                login = None
                if out.get("approval_token"):                     # the relying login service's check (API.md §5.1)
                    pl = token.verify(approval_pub, out["approval_token"])
                    if pl["dev"] != dev.device_id or pl["act"] != action:
                        raise SystemExit(f"approval token for another device/action: {pl}")
                    login = {"account": pl["acct"], "amr": pl["amr"], "ttl_s": pl["exp"] - pl["iat"]}
                seg["server"] = {"trigger_t": round(fired_at / FPS, 3), "sent_t": round(i / FPS, 3),
                                 "latency_s": round(lat, 3), "status": r.status_code,
                                 "response": {k: v for k, v in out.items() if k != "approval_token"},
                                 "approval_token": out["approval_token"][:16] + "…" if out.get("approval_token") else None,
                                 "login": login, **sv, "checklist": checklist(sv["internal"])}
        ticks.append(tk)
        if "server" in seg and i >= int(seg["server"]["sent_t"] * FPS) + int(FPS * 2.0):
            break
    seg["ticks"] = ticks
    if "server" not in seg:
        raise SystemExit(f"{label} {kind}: the trigger never fired: {[(x['t'], x['state'], x['reasons']) for x in ticks[::6]]}")
    s = seg["server"]
    print(label, kind, "->", s["response"], s["internal"], s["top1"], s["latency_s"], flush=True)
    return seg


def gallery(conn) -> tuple:
    """Subjects / templates the 1:N search covers: active subjects' active templates of the active recognizer."""
    return conn.execute(
        "SELECT count(DISTINCT t.subject_id), count(*) FROM templates t JOIN subjects s ON s.id = t.subject_id "
        "WHERE t.active AND s.status = 'active' AND t.model_version = "
        "(SELECT name || '@' || version FROM model_versions WHERE name = 'adaface_ir101_wf12m' AND active)").fetchone()


def main():
    import argparse
    from init_secrets import make_device
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse-enrollments", action="store_true",
                    help="keep the enrolled users and enrollment segments of the last recording; redo the rest")
    a = ap.parse_args()
    rec, sim = Rec(), SensorSim(seed=11)
    pipe = Pipeline.load(str(ROOT / "data/models"))
    old = json.loads((OUT / "timeline.json").read_text()) if a.reuse_enrollments else None
    if old is None:
        enroll_sim.withdraw(["demo-%"])
    dev_id = "kiosk-demo-01"
    make_device(dev_id)
    dev = enroll_sim.register(dev_id)
    approval_pub = load_pem_public_key((SECRETS / "approval.pub").read_bytes())
    conn = psycopg.connect(DSN, password=(ROOT / "secrets/db_api_password").read_text().strip(), autocommit=True)
    if old:
        segs, before = [s for s in old["segments"] if s["type"] == "enroll"], tuple(old["meta"]["gallery_before"].values())
        rec.n = 1 + max(int(tk["frame"]) for s in segs for tk in s["ticks"])
    else:
        before = gallery(conn)
        segs = [enroll_user(rec, label, subject, seqs, seed, dev, conn, pipe, sim, k)
                for k, (label, subject, seqs, seed) in enumerate(USERS)]
    accounts = {s["user"]: s["account"] for s in segs}
    after = gallery(conn)
    faces = {}
    for label, seq in RETURNS:
        faces[label] = frontal_run(seq)
        segs.append(approach(rec, label, "live", faces[label], dev, conn, pipe, sim, 101 + len(segs), approval_pub))
    for label, seq in STRANGERS:
        segs.append(approach(rec, label, "live", frontal_run(seq), dev, conn, pipe, sim, 201 + len(segs), approval_pub))
    tries = []
    for s in range(5):                                            # mask of D's face; keep a PAD refusal to show
        seg = approach(rec, "D", "mask_silicone", faces["D"], dev, conn, pipe, sim, 300 + s, approval_pub)
        tries.append(seg["server"]["internal"])
        if tries[-1] == "pad_fail":
            break
    seg["mask_attempts"] = tries
    segs.append(seg)
    meta = {"created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "gallery_before": {"subjects": before[0], "templates": before[1]},
            "gallery": {"subjects": after[0], "templates": after[1]}, "accounts": accounts, "fps": FPS,
            "device": dev_id, "model": "adaface_ir101_wf12m@1", "detector": "scrfd_34g@2 (320)", "pad": "pad_rgbdir@2",
            "thresholds": {k: float(v) for k, v in (l.split("=") for l in (ROOT / ".env").read_text().splitlines()
                                                    if l.startswith(("FA_T_ID=", "FA_MARGIN=")))},
            "note": "frames = SensorSim renders of BIWI recordings (the pixels the system processed)"}
    doc = {"meta": meta, "segments": segs}
    (OUT / "timeline.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    (OUT / "timeline.js").write_text("window.TIMELINE = " + json.dumps(doc, ensure_ascii=False) + ";\n")
    print("frames", rec.n, "gallery", before, "->", after, "mask tries", tries)


if __name__ == "__main__":
    main()
