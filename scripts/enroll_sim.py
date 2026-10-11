"""E8 enrollment simulation (RFP §4: enrollment success >= 99%, enrollment time p95 <= 20 s; failures, abandonment and
retries reported too).

No hardware and no live people, so a prompt-following simulated user stands in front of the kiosk. It reacts to the
guide's prompt after a reaction delay and turns its head toward the target at a per-trial angular speed (with
jitter), correcting per axis from the terminal's estimated pose (what a kiosk pose indicator shows). At every
terminal analysis tick it shows the REAL BIWI frame of the same person whose ground-truth pose is
closest to its current head pose; SensorSim renders the RGB-D-IR triplet (fresh sensor noise per tick). The real
terminal pipeline + Guide run on every tick, and the bundle goes through the real API (PAD, server-side pose
coverage, same-person, duplicate checks). After each trial the enrollment is withdrawn (DELETE /v1/subjects/{id}/face)
so the next trial of the same person is not a duplicate.

    python scripts/enroll_sim.py --trials 10 --workers 9

Subjects: BIWI subjects held out from PAD training (scripts/train_pad.py BIWI_SPLIT["test"]).
Enrollment time = sensor time from the first prompt to the guide's completion + the server round trip.
"""
import argparse
import datetime
import json
import sys
import time
import uuid
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import httpx  # noqa: E402
import msgpack  # noqa: E402
import numpy as np  # noqa: E402

from faceauth.terminal.client import SECRETS, DeviceClient, make_account_assertion, synced_ts  # noqa: E402
from faceauth.terminal.guide import TARGETS, Guide  # noqa: E402

GATEWAY, ENGINE = "https://gateway:8443", "http://engine:8001"
HELD_OUT = ["02", "03", "05", "06", "07", "09", "10", "13", "15", "16", "18", "21", "22"]  # train_pad BIWI test split
# Fixed before the run: subjects whose recordings contain all 9 guidance directions (GT pose scan). The others (F02,
# M01, M04) cannot show some directions at all, so they are reported separately as recording-coverage limited.
COMPLETE = {"F03", "F05", "F06", "M03", "M07", "M09"}
TIMEOUT_S, MAX_ATTEMPTS = 60.0, 2


def subject_pool(seqs):
    """All frames of one BIWI subject (several sessions): GT pose + RGB, within the guidance range."""
    from faceauth.sim import biwi
    out = []
    for f in biwi.iter_frames(str(ROOT / "data/datasets/biwi"), seqs=seqs, register=False, step=2):
        if abs(f["pose"][0]) <= 45 and abs(f["pose"][1]) <= 35:
            out.append((np.asarray(f["pose"][:2], np.float32), f["rgb"], f"{f['seq']}/{f['frame']}"))
    return out


def register(device_id):
    from init_secrets import make_device
    make_device(device_id)
    d = SECRETS / "devices"
    body = {"device_id": device_id, "public_key_pem": (d / f"{device_id}.ed25519.pub").read_text(),
            "cert_pem": (d / f"{device_id}.crt").read_text(), "calib_version": "gemini336-sim-v1"}
    c = DeviceClient(GATEWAY, device_id)
    r = c.http.post("/v1/admin/devices", json=body, extensions={"sni_hostname": "gateway"},
                    headers={"Authorization": "Bearer " + (SECRETS / "admin_token").read_text().strip()})
    if r.status_code not in (200, 201, 409):
        raise SystemExit(f"register {device_id}: {r.status_code} {r.text}")
    return c


def same_person_min_cos(frames):
    """Diagnostics only (the API decides): min cosine of each kept frame's embedding to their normalised mean."""
    r = httpx.post(f"{ENGINE}/v1/analyze", content=msgpack.packb({"mode": "enroll", "frames": frames}),
                   headers={"Content-Type": "application/msgpack"}, timeout=120)
    an = msgpack.unpackb(r.content, raw=False)
    e = np.array([np.frombuffer(f["emb"], "<f4") for f in an["frames"] if f.get("emb")])
    if len(e) < 2:
        return None, an
    m = e.mean(0)
    return float((e @ (m / np.linalg.norm(m))).min()), an


def capture(sim, pipe, pool, scenes, rng, speed, react, fps, guide, trace=None):
    """One guided capture. -> (done, sensor seconds, frames kept [(dir, (idx, enc))], ticks).
    trace: optional dict filled with tick -> BIWI frame id shown (diagnostics)."""
    from faceauth.terminal.pipeline import wire_frame
    head = rng.normal(0, 3, 2).astype(np.float32)                       # starts roughly frontal
    poses = np.stack([p for p, _, _ in pool])
    lo, hi = poses.min(0), poses.max(0)
    target, since, tick, corr = None, 0.0, 0, np.zeros(2, np.float32)
    while not (guide.done or guide.timed_out):
        t = tick / fps
        cur = guide.current()
        if cur != target:
            target, since, corr = cur, t, np.zeros(2, np.float32)
        if target is not None and t - since >= react:                  # reacts, then turns toward the target
            goal = np.asarray(TARGETS[target], np.float32) + corr      # corr: follows the on-screen pose feedback
            step = goal - head
            dist = float(np.linalg.norm(step))
            head = np.clip(head + step * min(1.0, speed / fps / max(dist, 1e-6)) + rng.normal(0, 0.7, 2), lo, hi)
            head = head.astype(np.float32)
        near = np.argsort(np.linalg.norm(poses - head, axis=1))[:5]      # closest real frames of this person
        i = int(rng.choice(near))
        if trace is not None:
            trace[tick] = pool[i][2]
        if i not in scenes:   # the subject = largest face (BIWI frames can show people in the background)
            dets = pipe.detector.detect(pool[i][1])
            big = max(dets, key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1])) if dets else None
            scenes[i] = sim.scene("live", pool[i][1], big["kps"], np.random.default_rng(i), 600.0) if big else None
        if scenes[i] is None:
            guide.feed(None, False, t=t, reasons=("no_face",))
        else:
            o = sim.capture([scenes[i]], rng)
            enc, rgb = wire_frame(o["rgb"], o["depth"], o["ir"])
            r, _ = pipe.feed(guide, rgb, (tick, enc), t=t)
            if target is not None and r["pose"] is not None and t - since >= react:
                # the kiosk shows where the head is vs the target; the person corrects per axis (integral feedback)
                corr = np.clip(corr + 0.15 * (np.asarray(TARGETS[target], np.float32) - np.asarray(r["pose"][:2], np.float32)),
                               -20.0, 20.0).astype(np.float32)
        tick += 1
    return guide.done, tick / fps, guide.frames(), tick


def trials(job):
    """All trials of one subject, sequential (each ends with a withdrawal). -> list of trial records."""
    import torch
    torch.set_num_threads(2)
    from faceauth.sim.sensor import SensorSim
    from faceauth.terminal.pipeline import Pipeline
    subject, seqs, n_trials, fps, seed, worker = job
    pipe = Pipeline.load(str(ROOT / "data/models"))
    pool, sim, scenes = subject_pool(seqs), SensorSim(seed=seed), {}
    client = register(f"e8-{worker:02d}")
    admin = {"Authorization": "Bearer " + (SECRETS / "admin_token").read_text().strip()}
    out = []
    for k in range(n_trials):
        rng = np.random.default_rng([seed, worker, k])
        speed, react = float(rng.uniform(20, 60)), float(rng.uniform(0.3, 1.0))   # deg/s, s
        account = f"e8-{subject}-{k}-{uuid.uuid4().hex[:6]}"
        rec = {"subject": subject, "trial": k, "speed_deg_s": speed, "react_s": react, "attempts": []}
        for attempt in range(MAX_ATTEMPTS):
            guide = Guide(timeout_s=TIMEOUT_S)
            done, sensor_s, kept, ticks = capture(sim, pipe, pool, scenes, rng, speed, react, fps, guide)
            # Simulation shortcut: a real terminal opens the session before capturing (Design §2) and the capture takes
            # sensor_s; here the simulated capture runs far slower than real time on a shared GPU, so the session is
            # opened right after it to keep the 180 s session TTL from failing trials that are 10-30 s in sensor time.
            session = client.create_session("enroll", "enroll")
            assertion = make_account_assertion(account, session["challenge"], "password", ttl_s=180)
            a = {"captured": done, "sensor_s": sensor_s, "ticks": ticks, "coverage": guide.coverage()}
            if done:
                frames = [enc for _, (_, enc) in kept]
                a["min_cos"], _ = same_person_min_cos(frames)
                t0 = time.perf_counter()
                consent = {"version": "v1", "agreed_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
                try:
                    r = client.enroll(session, frames, synced_ts([i for _, (i, _) in kept], int(time.time() * 1000)),
                                      [d for d, _ in kept], assertion, consent)
                except httpx.HTTPStatusError as e:
                    r = {"status": f"http_{e.response.status_code}"}
                a.update(server_s=time.perf_counter() - t0, status=r.get("status"), reason=r.get("reason"),
                         subject_id=r.get("subject_id"))
            rec["attempts"].append(a)
            if a.get("status") == "active":                              # withdraw: next trial is not a duplicate
                client.http.delete(f"/v1/subjects/{a['subject_id']}/face", headers=admin,
                                   extensions={"sni_hostname": "gateway"}).raise_for_status()
                break
        rec["success"] = any(a.get("status") == "active" for a in rec["attempts"])
        rec["time_s"] = sum(a["sensor_s"] + a.get("server_s", 0.0) for a in rec["attempts"])
        out.append(rec)
        print(json.dumps({k2: v for k2, v in rec.items() if k2 != "attempts"} |
                         {"statuses": [(a.get("status"), a.get("reason"), round(a["sensor_s"], 1)) for a in rec["attempts"]]}),
              flush=True)
    return out


def withdraw(patterns):
    """Withdraw every live enrollment whose account matches a LIKE pattern (leftovers of interrupted runs and the demo
    would otherwise be flagged as duplicates of these subjects -> pending_review)."""
    import psycopg
    client = register("e8-admin")
    admin = {"Authorization": "Bearer " + (SECRETS / "admin_token").read_text().strip()}
    with psycopg.connect("host=db,db-standby1,db-standby2 dbname=faceauth user=faceauth_api target_session_attrs=read-write",
                         password=(ROOT / "secrets/db_api_password").read_text().strip()) as conn:
        ids = [r[0] for p in patterns for r in conn.execute(
            "SELECT id FROM subjects WHERE account_id LIKE %s AND status <> 'revoked'", (p,)).fetchall()]
    for sid in ids:
        client.http.delete(f"/v1/subjects/{sid}/face", headers=admin, extensions={"sni_hostname": "gateway"}).raise_for_status()
    return len(ids)


def cp_ci(x, n, alpha=0.05):
    """Clopper-Pearson interval of a binomial rate."""
    from scipy.stats import beta
    return (float(beta.ppf(alpha / 2, x, n - x + 1)) if x else 0.0,
            float(beta.ppf(1 - alpha / 2, x + 1, n - x)) if x < n else 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--fps", type=float, default=10.0, help="terminal analysis rate (frames analysed per second)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subjects", nargs="*", help="only these BIWI subjects (e.g. a regression check)")
    ap.add_argument("--out", default="reports/E8_enrollment/results.json")
    ap.add_argument("--withdraw", nargs="*", default=["e8-%", "acct-f05%"],
                    help="account LIKE patterns withdrawn before the run (leftovers would be duplicates)")
    a = ap.parse_args()
    print("withdrawn leftovers:", withdraw(a.withdraw), flush=True)
    from faceauth.sim.biwi import SUBJECTS
    subj = {}
    for s in HELD_OUT:
        subj.setdefault(SUBJECTS[s], []).append(s)
    jobs = [(name, seqs, a.trials, a.fps, a.seed, i) for i, (name, seqs) in enumerate(sorted(subj.items()))
            if not a.subjects or name in a.subjects]
    t0 = time.time()
    with Pool(min(a.workers, len(jobs))) as pool:
        recs = [r for rs in pool.map(trials, jobs) for r in rs]
    by_group = {}
    for name, sel in (("complete_coverage", [r for r in recs if r["subject"] in COMPLETE]),
                      ("coverage_limited", [r for r in recs if r["subject"] not in COMPLETE])):
        k = sum(r["success"] for r in sel)
        by_group[name] = {"subjects": sorted({r["subject"] for r in sel}), "trials": len(sel), "success": k,
                          "success_rate": k / len(sel) if sel else None, "success_ci95": cp_ci(k, len(sel)) if sel else None}
    ok = [r for r in recs if r["success"]]
    first = [r for r in recs if r["attempts"][0].get("status") == "active"]
    times = np.array([r["time_s"] for r in ok])
    res = {
        "id": "E8_enrollment", "subjects": sorted(subj), "trials": len(recs), "fps": a.fps, "timeout_s": TIMEOUT_S,
        "max_attempts": MAX_ATTEMPTS,
        "success_rate": len(ok) / len(recs), "success_ci95": cp_ci(len(ok), len(recs)), "by_group": by_group,
        "per_subject": {sub: [r["success"] for r in recs if r["subject"] == sub] for sub in sorted(subj)},
        "first_attempt_success_rate": len(first) / len(recs),
        "abandoned_timeout": sum(not any(x["captured"] for x in r["attempts"]) for r in recs),
        "retries": sum(len(r["attempts"]) - 1 for r in recs),
        "time_s": {"p50": float(np.median(times)), "p95": float(np.percentile(times, 95)), "max": float(times.max())}
        if len(times) else None,
        "server_s_p95": float(np.percentile([x["server_s"] for r in recs for x in r["attempts"] if "server_s" in x], 95)),
        "failure_reasons": {},
        "min_cos": {"p1": float(np.percentile(m, 1)), "p5": float(np.percentile(m, 5)), "p50": float(np.median(m))}
        if len(m := [x["min_cos"] for r in recs for x in r["attempts"] if x.get("min_cos") is not None]) else None,
        "targets": {"success_rate": 0.99, "time_p95_s": 20.0}, "wall_s": time.time() - t0,
    }
    for r in recs:
        for x in r["attempts"]:
            if x.get("status") != "active":
                key = f"{x.get('status') or 'capture_timeout'}:{x.get('reason') or ''}"
                res["failure_reasons"][key] = res["failure_reasons"].get(key, 0) + 1
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    raw = ROOT / "reports/raw/E8"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "trials.json").write_text(json.dumps(recs, indent=1, default=str))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
