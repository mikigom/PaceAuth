"""E9 load test (RFP §4): open-loop Poisson authentication sessions through the real mTLS gateway.

    python scripts/loadgen.py setup [--users 200] [--impostors 100] [--devices 600]
    python scripts/loadgen.py run --rate 100 --duration 300

Target: 100 sessions/s, p95 <= 1.5 s and p99 <= 2.5 s from the first captured frame to the final response, network,
PAD and 1:N search included. Gallery: 1M synthetic subjects x 4 (scripts/bulk_load.py) + the users below.
Runs inside the compose networks (tester service, GPU 5).

setup (once):
  users      seeded CelebA identities (in-the-wild photos, >= 6 each). Every photo -> SensorSim live render (3 frames,
             gemini336-sim-v1). 4 photos -> 4 templates computed by the engine itself (/v1/analyze, mean of the 3 frame
             embeddings) and stored through the enrollment write path (KMS encrypt, templates, 'add' events under lock
             7001); 2 other photos -> probe sessions.
  impostors  CelebA identities never enrolled (probe sessions only).
  devices    N simulated terminals (CA-issued mTLS cert + Ed25519 key) registered through the admin API.
  terminal   per-frame latency of the terminal pipeline (SCRFD + 6DRepNet360 + FaRL + quality) on the rendered frames.
run: each session takes a free terminal: POST /v1/sessions, then POST /v1/authentications with a freshly signed packet
of pre-rendered frames. Latency (Design §3: the session is issued, then the sequence is captured):
  e2e = capture window (10 frames = 333 ms; the terminal sends its best 3) + terminal processing of the last frame
        + packet signing + T_auth (client side, incl. mTLS, gateway, API, engine, search, DB commit).
"""
import argparse
import asyncio
import json
import random
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

from faceauth.common.packet import build_packet, decode_frame, encode_frame  # noqa: E402
from faceauth.terminal.client import CALIB_VERSION, SECRETS, mtls_context, synced_ts  # noqa: E402

OUT = ROOT / "data/tmp/loadgen"
REPORT = ROOT / "reports/E9_load"
CELEBA = ROOT / "data/datasets/celeba"
GATEWAY, ENGINE = "https://gateway:8443", "http://engine:8001"
NS = uuid.UUID("2b0a7e4c-1d0f-4f53-9c51-6f0a3e2d7b11")
WINDOW_MS, AUTH_IDX = 1000 * 10 / 30, (1, 4, 8)   # 10-frame capture; the 3 frames sent (indices in the window)


# ------------------------------------------------------------------------------------------------ setup
def identities(seed, n_users, n_imp, per=6):
    ident = {}
    for line in open(CELEBA / "Anno/identity_CelebA.txt"):
        name, i = line.split()
        ident.setdefault(i, []).append(name)
    ids = sorted((i for i, v in ident.items() if len(v) >= per), key=int)
    random.Random(seed).shuffle(ids)
    return [(i, sorted(ident[i])[:per], True) for i in ids[:n_users]] + \
           [(i, sorted(ident[i])[:2], False) for i in ids[n_users:n_users + n_imp]]


def _render(task):
    from faceauth.sim.capture import sequence
    from faceauth.sim.sensor import SensorSim
    import cv2
    path, kps, seed, dist = task
    img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    try:
        frames, _ = sequence(SensorSim(seed=seed), "live", img, np.asarray(kps, np.float32), n=3, seed=seed,
                             distance_mm=dist)
    except ValueError:
        return None
    return [encode_frame(f["rgb"], f["depth"], f["ir"]) for f in frames]


def analyze(http, frames, mode="enroll"):
    r = http.post(f"{ENGINE}/v1/analyze", content=msgpack.packb({"mode": mode, "frames": frames}),
                  headers={"Content-Type": "application/msgpack"}, timeout=120)
    r.raise_for_status()
    return msgpack.unpackb(r.content, raw=False)


def templates_from(http, photos, mv):
    """Rendered photos (3 frames each) -> (templates: one per photo whose 3 frames are ok + live, align version).
    mv = active recognition model version: vectors of another version are never stored (Design §7)."""
    tpl, align = [], None
    for p in photos:
        if not p:
            continue
        an = analyze(http, p)
        if an["versions"]["recog"] != mv:
            raise SystemExit(f"engine recognizer {an['versions']['recog']} != active {mv}")
        align = an["versions"]["align"]
        good = [f for f in an["frames"] if f["ok"] and f["pad"] and f["pad"]["live"] and f["emb"]]
        if len(good) == len(p):
            e = np.mean([np.frombuffer(f["emb"], "<f4") for f in good], 0)
            tpl.append((e / np.linalg.norm(e)).astype("<f4"))
    return tpl, align


def insert_user(conn, kms, mv, align, account, tpl) -> uuid.UUID:
    """Enrollment write path without the capture (load/security tests): subject + KMS-encrypted templates + 'add'
    events under lock 7001, one transaction. Idempotent per account."""
    sid = uuid.uuid5(NS, account)
    cts = msgpack.unpackb(kms.post("/v1/encrypt", headers={"Content-Type": "application/msgpack"},
                                   content=msgpack.packb({"subject_id": str(sid), "items": [t.tobytes() for t in tpl]})).content)["items"]
    with conn.transaction():
        conn.execute("INSERT INTO subjects (id, account_id, status, consent_version, consent_at) VALUES "
                     "(%s, %s, 'active', 'v1', now()) ON CONFLICT DO NOTHING", (sid, account))
        if conn.execute("SELECT count(*) FROM templates WHERE subject_id = %s AND active", (sid,)).fetchone()[0] == 0:
            tids = [conn.execute("INSERT INTO templates (subject_id, template_version, slot, model_version, "
                                 "align_version, calib_version, ciphertext) VALUES (%s, 1, %s, %s, %s, %s, %s) "
                                 "RETURNING id", (sid, i, mv, align, CALIB_VERSION, c)).fetchone()[0]
                    for i, c in enumerate(cts)]
            conn.execute("SELECT pg_advisory_xact_lock(7001)")
            for t in tids:
                conn.execute("INSERT INTO template_events (op, template_id, subject_id, model_version) "
                             "VALUES ('add', %s, %s, %s)", (t, sid, mv))
    return sid


def cmd_setup(a):
    import psycopg
    import cv2
    from faceauth.models.scrfd import SCRFD
    from init_secrets import make_device
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    people = identities(a.seed, a.users, a.impostors)
    det, tasks = SCRFD(ROOT / "data/models/scrfd_34g"), []
    rng = np.random.default_rng(a.seed)
    for cid, names, enrolled in people:
        for n in names:
            img = cv2.cvtColor(cv2.imread(str(CELEBA / "img_celeba" / n)), cv2.COLOR_BGR2RGB)
            d = det.detect(img)
            big = max(d, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1])) if d else None
            tasks.append((str(CELEBA / "img_celeba" / n), big and big["kps"].tolist(), int(rng.integers(1 << 31)),
                          float(rng.uniform(450, 750))))
    todo = [i for i, t in enumerate(tasks) if t[1]]                      # photos without a detected face: skipped
    rendered = [None] * len(tasks)
    with Pool(a.workers) as pool:
        for i, r in zip(todo, pool.map(_render, [tasks[i] for i in todo], chunksize=2)):
            rendered[i] = r
    print(f"rendered {sum(r is not None for r in rendered)}/{len(rendered)} photos {time.time() - t0:.0f}s", flush=True)

    http = httpx.Client()
    with psycopg.connect(a.dsn, password=(ROOT / a.password_file).read_text().strip()) as conn:
        mv = conn.execute("SELECT name || '@' || version FROM model_versions WHERE name = %s AND active",
                          (a.recog,)).fetchone()[0]
        kms = httpx.Client(base_url=a.kms_url, headers={"Authorization": "Bearer " + (ROOT / a.kms_token_file).read_text().strip()})
        users, k, stats = [], 0, {"enrolled": 0, "impostors": 0, "dropped": 0}
        for cid, names, enrolled in people:
            photos, k = rendered[k:k + len(names)], k + len(names)
            if not enrolled:
                probes = [p for p in photos if p]
                if probes:
                    users.append({"id": f"imp-{cid}", "enrolled": False, "probes": probes})
                    stats["impostors"] += 1
                continue
            tpl, align = templates_from(http, photos[:4], mv)
            probes = [p for p in photos[4:] if p]
            if len(tpl) < 2 or not probes:
                stats["dropped"] += 1
                continue
            account = f"loadtest-{cid}"
            insert_user(conn, kms, mv, align, account, tpl)
            users.append({"id": account, "enrolled": True, "probes": probes})
            stats["enrolled"] += 1
    print(stats, f"{time.time() - t0:.0f}s", flush=True)

    admin = (SECRETS / "admin_token").read_text().strip()
    devices = [f"lg-{i:04d}" for i in range(a.devices)]
    for d in devices:
        make_device(d)
        body = {"device_id": d, "public_key_pem": (SECRETS / "devices" / f"{d}.ed25519.pub").read_text(),
                "cert_pem": (SECRETS / "devices" / f"{d}.crt").read_text(), "calib_version": CALIB_VERSION}
        with httpx.Client(base_url=GATEWAY, verify=mtls_context(d)) as c:
            r = c.post("/v1/admin/devices", json=body, headers={"Authorization": "Bearer " + admin},
                       extensions={"sni_hostname": "gateway"})
            if r.status_code not in (200, 201, 409):
                raise SystemExit(f"device registration failed: {r.status_code} {r.text}")

    term = terminal_bench([p for u in users[:40] for p in u["probes"]])
    (OUT / "users.msgpack").write_bytes(msgpack.packb(users))
    (OUT / "setup.json").write_text(json.dumps({"seed": a.seed, "model_version": mv, "devices": devices,
                                                "stats": stats, "terminal_ms_per_frame": term}, indent=1))
    print("terminal ms/frame", term, f"setup {time.time() - t0:.0f}s")


def terminal_bench(probes):
    """Per-frame latency of the terminal pipeline on the decoded frames it would analyse (GPU 5, idle-ish)."""
    from faceauth.terminal.pipeline import Pipeline
    pipe = Pipeline.load(str(ROOT / "data/models"))
    frames = [decode_frame(f)[0] for p in probes for f in p]
    for f in frames[:5]:
        pipe.analyze(f)
    ts = []
    for f in frames:
        t = time.perf_counter()
        pipe.analyze(f)
        ts.append((time.perf_counter() - t) * 1000)
    return {"n": len(ts), "p50": float(np.median(ts)), "p95": float(np.percentile(ts, 95)), "max": float(max(ts))}


# ------------------------------------------------------------------------------------------------ run
class Terminal:
    def __init__(self, device_id):
        from cryptography.hazmat.primitives import serialization
        self.id = device_id
        self.key = serialization.load_pem_private_key((SECRETS / "devices" / f"{device_id}.ed25519").read_bytes(), None)
        self.http = httpx.AsyncClient(base_url=GATEWAY, verify=mtls_context(device_id), timeout=30.0)

    async def post(self, path, **kw):
        r = await self.http.post(path, extensions={"sni_hostname": "gateway"}, **kw)
        return r.status_code, (r.json() if r.headers.get("content-type", "").startswith("application/json")
                               else msgpack.unpackb(r.content, raw=False) if r.content else None)


async def one(term, user, probe, rec, sched_t, t_start):
    t_wait = time.perf_counter() - sched_t
    t0 = time.perf_counter()
    st, s = await term.post("/v1/sessions", json={"purpose": "auth", "device_id": term.id, "action_id": "door-open"})
    t1 = time.perf_counter()
    out = {"user": user["id"], "enrolled": user["enrolled"], "t": t1 - t_start, "wait_s": t_wait, "session_s": t1 - t0,
           "session_status": st}
    if st == 200:
        start_ms = int(time.time() * 1000)
        ts = synced_ts(AUTH_IDX, start_ms)
        pkt = await asyncio.to_thread(build_packet, term.key, session_id=s["session_id"], challenge=s["challenge"],
                                      device_id=term.id, calib_version=CALIB_VERSION, frames=probe, ts_ms=ts,
                                      captured_at_ms=start_ms)
        t2 = time.perf_counter()
        st2, r = await term.post("/v1/authentications", content=pkt, headers={"Content-Type": "application/msgpack"})
        t3 = time.perf_counter()
        out.update(sign_s=t2 - t1, auth_s=t3 - t2, auth_status=st2, decision=(r or {}).get("decision"),
                   reason=(r or {}).get("reason"))
    rec.append(out)


async def cmd_run_async(a):
    setup = json.loads((OUT / "setup.json").read_text())
    users = msgpack.unpackb((OUT / "users.msgpack").read_bytes(), raw=False)
    enrolled, imps = [u for u in users if u["enrolled"]], [u for u in users if not u["enrolled"]]
    free = asyncio.Queue()
    for d in setup["devices"][:a.devices]:
        free.put_nowait(Terminal(d))
    rng, rec, tasks = random.Random(a.seed), [], []

    async def run_one(user, probe, sched_t, t_start):
        term = await free.get()
        try:
            await one(term, user, probe, rec, sched_t, t_start)
        except Exception as e:  # noqa: BLE001 - count it, keep the load going
            rec.append({"user": user["id"], "enrolled": user["enrolled"], "error": repr(e)[:200]})
        finally:
            free.put_nowait(term)

    total = a.duration + a.warmup
    t_start = time.perf_counter()
    nxt = t_start
    while nxt - t_start < total:
        nxt += rng.expovariate(a.rate)
        await asyncio.sleep(max(0.0, nxt - time.perf_counter()))
        u = rng.choice(imps) if rng.random() < a.impostor_frac else rng.choice(enrolled)
        tasks.append(asyncio.create_task(run_one(u, rng.choice(u["probes"]), nxt, t_start)))
    await asyncio.gather(*tasks)
    return setup, rec, time.perf_counter() - t_start


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def cmd_run(a):
    setup, rec, wall = asyncio.run(cmd_run_async(a))
    m = [r for r in rec if r.get("t", -1) >= a.warmup]                    # measured window only
    ok = [r for r in m if r.get("auth_status") == 200]
    auth = np.array([r["auth_s"] for r in ok]) * 1000
    term_ms = setup["terminal_ms_per_frame"]["p95"]
    sign = np.array([r["sign_s"] for r in ok]) * 1000
    e2e = WINDOW_MS + term_ms + sign + auth
    gen = [r for r in ok if r["enrolled"]]
    imp = [r for r in ok if not r["enrolled"]]
    res = {
        "id": "E9_load", "rate_target": a.rate, "duration_s": a.duration, "warmup_s": a.warmup, "wall_s": wall,
        "devices": a.devices, "impostor_frac": a.impostor_frac, "model_version": setup["model_version"],
        "sessions_measured": len(m), "throughput_per_s": len(m) / a.duration,
        "errors": sum("error" in r for r in m), "http_non_200": sum(r.get("auth_status", 200) != 200 or r.get("session_status") != 200 for r in m if "error" not in r),
        "wait_ms_p99": pct([r["wait_s"] * 1000 for r in m if "wait_s" in r], 99),
        "session_ms": {"p50": pct([r["session_s"] * 1000 for r in m if "session_s" in r], 50),
                       "p95": pct([r["session_s"] * 1000 for r in m if "session_s" in r], 95)},
        "auth_ms": {"p50": pct(auth, 50), "p95": pct(auth, 95), "p99": pct(auth, 99), "max": float(auth.max()) if len(auth) else None},
        "sign_ms_p95": pct(sign, 95),
        "e2e_ms": {"p50": pct(e2e, 50), "p95": pct(e2e, 95), "p99": pct(e2e, 99),
                   "composition": f"{WINDOW_MS:.0f} ms capture window + terminal p95 {term_ms:.1f} ms/frame + signing + T_auth"},
        "targets": {"p95_ms": 1500, "p99_ms": 2500},
        "decisions": {"genuine": {"n": len(gen), "approved": sum(r["decision"] == "approved" for r in gen),
                                  "reasons": {k: sum(r["reason"] == k for r in gen) for k in {r["reason"] for r in gen}}},
                      "impostor": {"n": len(imp), "approved": sum(r["decision"] == "approved" for r in imp),
                                   "reasons": {k: sum(r["reason"] == k for r in imp) for k in {r["reason"] for r in imp}}}},
        "terminal_ms_per_frame": setup["terminal_ms_per_frame"],
    }
    res["pass"] = bool(res["e2e_ms"]["p95"] <= 1500 and res["e2e_ms"]["p99"] <= 2500
                       and res["throughput_per_s"] >= 0.98 * a.rate and res["errors"] == 0)
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / f"results_rate{a.rate:g}.json").write_text(json.dumps(res, indent=1))
    raw = ROOT / "reports/raw/E9"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / f"sessions_rate{a.rate:g}.jsonl").write_text("\n".join(json.dumps(r) for r in rec))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup")
    s.add_argument("--users", type=int, default=200)
    s.add_argument("--impostors", type=int, default=100)
    s.add_argument("--devices", type=int, default=600)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--workers", type=int, default=32)
    s.add_argument("--recog", default="kprpe_vitb_wf12m")
    s.add_argument("--dsn", default="host=db,db-standby1,db-standby2 dbname=faceauth user=faceauth_api target_session_attrs=read-write")
    s.add_argument("--password-file", default="secrets/db_api_password")
    s.add_argument("--kms-url", default="http://kms:8003")
    s.add_argument("--kms-token-file", default="secrets/kms_token_api")
    r = sub.add_parser("run")
    r.add_argument("--rate", type=float, default=100)
    r.add_argument("--duration", type=float, default=300)
    r.add_argument("--warmup", type=float, default=30)
    r.add_argument("--devices", type=int, default=600)
    r.add_argument("--impostor-frac", type=float, default=0.05)
    r.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cmd_setup(a) if a.cmd == "setup" else cmd_run(a)
