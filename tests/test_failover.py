"""E10 availability (RFP §4): single-node failure, RPO=0 for approved enrollments/withdrawals, RTO <= 15 min.

scripts/test_failover.sh drives the scenario from the host (kills containers, runs scripts/failover.sh); this module
is the part that runs inside the compose network:

    python tests/test_failover.py load DIR [--workers 4]   enroll/withdraw loop until DIR/stop exists -> DIR/ops-*.jsonl
    python tests/test_failover.py verify DIR               every acknowledged op is present on the current primary
    python tests/test_failover.py report RAW OUT           RAW/*/ -> OUT/results.json + OUT/summary.md

An enrollment = KMS encrypt (creates the subject key) + one DB tx (subject, 4 templates, outbox events, audit).
A withdrawal = one DB tx (revoke, templates off + ciphertext NULL, remove events, audit) + KMS key destruction.
Each is "acknowledged" only when every step returned. pytest runs only the analysis test at the bottom.
"""
import argparse
import collections
import hashlib
import json
import pathlib
import statistics
import sys
import threading
import time
import uuid

DSN = ("host=db,db-standby1,db-standby2 dbname=faceauth user=faceauth_api target_session_attrs=read-write "
       "connect_timeout=3 keepalives_idle=5 keepalives_interval=2 keepalives_count=3")
KMS = "http://kms:8003"
MODEL = "e10-synthetic@1"
PROBE = b"probe"


def pt(sid: str, slot: int) -> bytes:  # deterministic 2048-byte "template" (512 x float32 size)
    return hashlib.sha256(f"{sid}/{slot}".encode()).digest() * 64


def _secret(name: str) -> str:
    return pathlib.Path("secrets", name).read_text().strip()


def _kms(client, method, path, scope, obj=None):
    import msgpack
    r = client.request(method, KMS + path, content=None if obj is None else msgpack.packb(obj),
                       headers={"Authorization": f"Bearer {_secret('kms_token_' + scope)}",
                                "Content-Type": "application/msgpack"})
    r.raise_for_status()
    return msgpack.unpackb(r.content, raw=False)["items"] if obj is not None else r.json()


def enroll(conn, kms, sid):
    cts = _kms(kms, "POST", "/v1/encrypt", "api", {"subject_id": sid, "items": [pt(sid, s) for s in range(4)] + [PROBE]})
    with conn.transaction():
        conn.execute("INSERT INTO subjects (id, account_id, status, consent_version, consent_at) "
                     "VALUES (%s, %s, 'active', 'e10', now())", (sid, "e10-" + sid))
        ids = [conn.execute("INSERT INTO templates (subject_id, template_version, slot, model_version, align_version, "
                            "calib_version, ciphertext) VALUES (%s, 1, %s, %s, 'e10', 'e10', %s) RETURNING id",
                            (sid, s, MODEL, ct)).fetchone()[0] for s, ct in enumerate(cts[:4])]
        conn.execute("SELECT pg_advisory_xact_lock(7001)")
        for tid in ids:
            conn.execute("INSERT INTO template_events (op, template_id, subject_id, model_version) "
                         "VALUES ('add', %s, %s, %s)", (tid, sid, MODEL))
        conn.execute("INSERT INTO audit_log (actor, event, subject_id) VALUES ('e10', 'enrolled', %s)", (sid,))
    return cts[4]


def withdraw(conn, kms, sid, stop):
    with conn.transaction():
        conn.execute("UPDATE subjects SET status = 'revoked', revoked_at = now(), updated_at = now() WHERE id = %s", (sid,))
        rows = conn.execute("UPDATE templates SET active = false, ciphertext = NULL WHERE subject_id = %s AND active "
                            "RETURNING id", (sid,)).fetchall()
        conn.execute("SELECT pg_advisory_xact_lock(7001)")
        for (tid,) in rows:
            conn.execute("INSERT INTO template_events (op, template_id, subject_id, model_version) "
                         "VALUES ('remove', %s, %s, %s)", (tid, sid, MODEL))
        conn.execute("INSERT INTO audit_log (actor, event, subject_id) VALUES ('e10', 'withdrawn', %s)", (sid,))
    t_db = time.time()
    while True:  # key destruction is part of an acknowledged withdrawal; the DB part is already durable
        try:
            _kms(kms, "DELETE", f"/v1/keys/{sid}", "api")
            return t_db
        except Exception:
            if stop.exists():
                raise
            time.sleep(0.2)


def load(d: pathlib.Path, workers: int) -> None:
    import httpx
    import psycopg
    stop, pw = d / "stop", _secret("db_api_password")

    def worker(w):
        live, conn, i = collections.deque(), None, 0
        with open(d / f"ops-{w}.jsonl", "a", buffering=1) as log, httpx.Client(timeout=10) as kms:
            while not stop.exists():
                i += 1
                op = "withdraw" if live and i % 3 == 0 else "enroll"
                sid, probe = live.popleft() if op == "withdraw" else (str(uuid.uuid4()), None)
                rec = {"op": op, "sid": sid, "w": w, "t0": time.time()}
                try:
                    if conn is None:
                        conn = psycopg.connect(DSN, password=pw, autocommit=True)
                    if op == "enroll":
                        probe = enroll(conn, kms, sid)
                        live.append((sid, probe))
                    else:
                        rec["t_db"] = withdraw(conn, kms, sid, stop)
                    rec.update(ok=True, probe=probe.hex())
                except Exception as e:  # not acknowledged; a failed withdrawal leaves the subject in doubt
                    rec.update(ok=False, err=f"{type(e).__name__}: {e}"[:200])
                    if conn is not None:
                        conn.close()
                    conn = None
                    time.sleep(0.1)
                rec["t"] = time.time()
                log.write(json.dumps(rec) + "\n")

    ts = [threading.Thread(target=worker, args=(w,)) for w in range(workers)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()


def read_ops(d: pathlib.Path) -> list:
    return [json.loads(line) for f in sorted(d.glob("ops-*.jsonl")) for line in f.read_text().splitlines() if line]


def verify(d: pathlib.Path) -> dict:
    import httpx
    import psycopg
    ops = read_ops(d)
    enrolled = {o["sid"]: o for o in ops if o["op"] == "enroll" and o["ok"]}
    withdrawn = {o["sid"]: o for o in ops if o["op"] == "withdraw" and o["ok"]}
    doubtful = {o["sid"] for o in ops if o["op"] == "withdraw" and not o["ok"]}
    keep = sorted(set(enrolled) - set(withdrawn) - doubtful)
    with psycopg.connect(DSN, password=_secret("db_api_password")) as conn, httpx.Client(timeout=120) as kms:
        q = lambda sql: conn.execute(sql, (list(enrolled),)).fetchall()  # noqa: E731
        primary = conn.execute("SELECT current_setting('cluster_name')").fetchone()[0]
        status = dict(q("SELECT id::text, status FROM subjects WHERE id = ANY(%s::uuid[])"))
        live_t = dict(q("SELECT subject_id::text, count(*) FROM templates WHERE subject_id = ANY(%s::uuid[]) "
                        "AND active AND ciphertext IS NOT NULL GROUP BY 1"))
        events = {(s, op): n for s, op, n in q("SELECT subject_id::text, op, count(*) FROM template_events "
                                                "WHERE subject_id = ANY(%s::uuid[]) GROUP BY 1, 2")}
        cts = conn.execute("SELECT subject_id::text, slot, ciphertext FROM templates WHERE subject_id = ANY(%s::uuid[]) "
                           "AND active", (keep,)).fetchall()
        audit_bad_id = conn.execute("SELECT audit_verify()").fetchone()[0]
        lost_enroll = [s for s in keep if status.get(s) != "active" or live_t.get(s) != 4 or events.get((s, "add")) != 4]
        lost_withdraw = [s for s in withdrawn if status.get(s) != "revoked" or live_t.get(s, 0) != 0
                         or events.get((s, "remove")) != 4]
        plain = []
        for i in range(0, len(cts), 8192):  # templates still decrypt (search scope) to what was enrolled
            plain += _kms(kms, "POST", "/v1/decrypt", "search", {"items": [{"subject_id": s, "ct": c} for s, _, c in cts[i:i + 8192]]})
        bad_templates = sum(p != pt(s, slot) for (s, slot, _), p in zip(cts, plain))
        probes = [(s, bytes.fromhex(withdrawn[s]["probe"])) for s in withdrawn]
        shredded = []
        for i in range(0, len(probes), 8192):  # a withdrawn subject's key is gone for good
            shredded += _kms(kms, "POST", "/v1/decrypt", "search", {"items": [{"subject_id": s, "ct": c} for s, c in probes[i:i + 8192]]})
    res = {"primary": primary, "ops": len(ops), "acked_enroll": len(enrolled), "acked_withdraw": len(withdrawn),
           "failed_ops": sum(not o["ok"] for o in ops), "in_doubt_withdraw": len(doubtful),
           "checked_enrolled": len(keep), "lost_enroll": lost_enroll, "lost_withdraw": lost_withdraw,
           "templates_checked": len(cts), "templates_bad": bad_templates + 4 * len(keep) - len(cts),
           "keys_not_shredded": sum(p is not None for p in shredded), "audit_chain_ok": audit_bad_id is None}
    res["rpo_zero"] = not (lost_enroll or lost_withdraw or res["templates_bad"] or res["keys_not_shredded"]
                           or not res["audit_chain_ok"])
    (d / "verify.json").write_text(json.dumps(res, indent=1))
    return res


def analyze(ops: list, meta: dict) -> dict:
    """Timing of one round from the op log and the orchestrator's timestamps (unix seconds)."""
    acks = sorted(o["t"] for o in ops if o["ok"])
    out = {"acks": len(acks), "failed": sum(not o["ok"] for o in ops)}
    t_kill, t_dead = meta["t_kill"], meta["t_dead"]
    if meta["kind"] == "primary":
        after = [t for t in acks if t > t_dead]
        out["rto_s"] = round(after[0] - t_kill, 3) if after else None
        fo = meta.get("failover", {})
        for k in ("detected", "t_fenced", "t_promoted", "t_sync"):
            if k in fo:
                out[f"{k.removeprefix('t_')}_after_kill_s"] = round(fo[k] - t_kill, 3)
    else:  # standby loss: writes must go on; with no standby left they must block (never be acknowledged)
        t_back = meta["t_restart"]
        win = [o for o in ops if o["t"] > t_dead and o["t0"] < t_back]
        out["failed_during_outage"] = sum(not o["ok"] for o in win)
        # grace: a commit flushed by a standby just before it died may be acknowledged a few ms after the kill
        out["acks_during_outage"] = sum(o["ok"] and t_dead + 0.5 < o["t"] < t_back for o in ops)
        out["max_latency_during_outage_s"] = round(max((o["t"] - o["t0"] for o in win), default=0), 3)
        out["outage_s"] = round(t_back - t_kill, 3)
    return out


def report(raw: pathlib.Path, out_dir: pathlib.Path) -> None:
    rounds = []
    for d in sorted(p for p in raw.iterdir() if (p / "meta.json").exists()):
        meta = json.loads((d / "meta.json").read_text())
        vf = d / "verify.json"  # missing = the RPO check itself could not run: never reported as a pass
        ver = json.loads(vf.read_text()) if vf.exists() else {"rpo_zero": False, "verify_error": True, "acked_enroll": 0,
                                                              "acked_withdraw": 0, "failed_ops": None, "audit_chain_ok": False,
                                                              "lost_enroll": [], "lost_withdraw": []}
        rounds.append({"round": d.name, **meta, **analyze(read_ops(d), meta),
                       **{k: v for k, v in ver.items() if not isinstance(v, list)},
                       "lost": len(ver["lost_enroll"]) + len(ver["lost_withdraw"])})
    prim = [r for r in rounds if r["kind"] == "primary"]
    rtos = [r["rto_s"] for r in prim if r["rto_s"] is not None]
    summary = {"primary_failovers": len(prim), "rto_s": {"min": min(rtos), "median": statistics.median(rtos),
                                                         "max": max(rtos)} if rtos else None,
               "rto_target_s": 900, "all_recovered": len(rtos) == len(prim),
               "rpo_zero_all_rounds": all(r["rpo_zero"] for r in rounds),
               "acked_enrollments_checked": sum(r["acked_enroll"] for r in rounds),
               "acked_withdrawals_checked": sum(r["acked_withdraw"] for r in rounds),
               "lost_acknowledged_ops": sum(r["lost"] for r in rounds)}
    env = json.loads((raw / "env.json").read_text()) if (raw / "env.json").exists() else {}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps({"summary": summary, "env": env, "rounds": rounds}, indent=1))
    lines = ["# E10 가용성 / Availability (RFP §4: RTO <= 15 min, RPO = 0)", "",
             "Generated by `scripts/test_failover.sh` (raw logs: `reports/raw/E10/`). 3-node Postgres 16 "
             "(1 primary + 2 standbys, `synchronous_commit=on`, `ANY 1`), KMS in the write path, all on one host "
             "(docker kill = node failure). RTO = kill -> first acknowledged enrollment/withdrawal on the new primary, "
             "including automatic detection (`failover.sh watch 1 3`).", "",
             f"Environment: {json.dumps(env)}", "",
             "| round | scenario | acked enroll | acked withdraw | failed ops | lost acked | RTO s | detect / promote / sync s | audit chain |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rounds:
        timing = (f"{r.get('detected_after_kill_s')} / {r.get('promoted_after_kill_s')} / {r.get('sync_after_kill_s')}"
                  if r["kind"] == "primary" else
                  f"outage {r['outage_s']}, acks during {r['acks_during_outage']}, max latency {r['max_latency_during_outage_s']}")
        lines.append(f"| {r['round']} | {r['scenario']} | {r['acked_enroll']} | {r['acked_withdraw']} | {r['failed_ops']} "
                     f"| {r['lost']} | {r.get('rto_s', '-')} | {timing} | {'ok' if r['audit_chain_ok'] else 'BROKEN'} |")
    s, det = summary, [r["detected_after_kill_s"] for r in prim if "detected_after_kill_s" in r]
    rto = (f"min {s['rto_s']['min']:.2f} s, median {s['rto_s']['median']:.2f} s, max {s['rto_s']['max']:.2f} s"
           if s["rto_s"] else "not recovered")
    lines += ["", f"- RTO over {s['primary_failovers']} primary failovers: {rto} (target <= 900 s); automatic detection "
              f"took {min(det, default=0):.2f}-{max(det, default=0):.2f} s of it.",
              f"- RPO=0: {'held in every round' if s['rpo_zero_all_rounds'] else 'VIOLATED'} — "
              f"{s['acked_enrollments_checked']} acknowledged enrollments and {s['acked_withdrawals_checked']} "
              f"withdrawals re-checked on the surviving primary (row state, outbox events, KMS decrypt of every template, "
              f"withdrawn keys shredded, audit hash chain); lost: {s['lost_acknowledged_ops']}.",
              "- Single standby loss: writes continue (ANY 1). Both standbys lost: commits block, none is acknowledged "
              "until a standby is back (fail closed, never relaxed).",
              "- Limits: one host, so no network partition / disk loss; detection is a 1 s x 3 probe; in-doubt ops "
              "(error after commit was sent) are excluded from the check and counted as failed."]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def test_analyze():
    ops = [{"op": "enroll", "t0": 9.0, "t": 9.5, "ok": True}, {"op": "enroll", "t0": 9.9, "t": 10.2, "ok": False},
           {"op": "enroll", "t0": 10.3, "t": 13.0, "ok": True}, {"op": "enroll", "t0": 13.0, "t": 13.1, "ok": True}]
    r = analyze(ops, {"kind": "primary", "t_kill": 10.0, "t_dead": 10.1, "failover": {"t_sync": 12.0}})
    assert r == {"acks": 3, "failed": 1, "rto_s": 3.0, "sync_after_kill_s": 2.0}
    r = analyze(ops, {"kind": "standby", "t_kill": 10.0, "t_dead": 10.1, "t_restart": 12.5})
    assert r["failed_during_outage"] == 1 and r["acks_during_outage"] == 0 and r["outage_s"] == 2.5
    assert r["max_latency_during_outage_s"] == 2.7
    ops.append({"op": "withdraw", "t0": 10.4, "t": 11.0, "ok": True})  # acknowledged while the standby was down
    assert analyze(ops, {"kind": "standby", "t_kill": 10.0, "t_dead": 10.1, "t_restart": 12.5})["acks_during_outage"] == 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["load", "verify", "report"])
    ap.add_argument("dir", type=pathlib.Path)
    ap.add_argument("out", type=pathlib.Path, nargs="?")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "load":
        load(a.dir, a.workers)
    elif a.cmd == "verify":
        r = verify(a.dir)
        print(json.dumps({k: v for k, v in r.items() if not isinstance(v, list)}))
        sys.exit(0 if r["rpo_zero"] else 1)
    else:
        report(a.dir, a.out)
