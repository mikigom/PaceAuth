"""Search microbenchmark (E9): exact 1:N on GPU at RFP scale -> reports/E9_load/search_microbench.json.

    python -m faceauth.search.bench [--subjects 1000000]   # synthetic gallery, GPU only (no DB/KMS)
    python -m faceauth.search.bench --startup              # replica build from DB + KMS (service env vars)

Sections are merged into the existing JSON, so both modes can be run separately.
"""
import argparse
import asyncio
import datetime
import json
import multiprocessing as mp
import pathlib
import subprocess
import threading
import time

import numpy as np

import faceauth.search.app as appmod
from faceauth.search.gallery import Gallery

OUT = pathlib.Path("reports/E9_load/search_microbench.json")
MV = "bench@1"
D = 512


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def pct(xs):
    xs = np.asarray(xs) * 1e3
    return {"p50_ms": round(float(np.percentile(xs, 50)), 2), "p95_ms": round(float(np.percentile(xs, 95)), 2),
            "p99_ms": round(float(np.percentile(xs, 99)), 2), "samples": len(xs)}


def gpu_used_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.split()
    return int(out[0]) if out else None


def gpu_state():
    out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.split(",")
    return {"utilization_pct": int(out[0]), "memory_used_mib": int(out[1])} if len(out) == 2 else None


def probes_for(centres, s, n, rng):
    return unit(centres[s] + 0.75 / np.sqrt(D) * rng.standard_normal((n, D))).astype("<f4")


def build_gallery(subjects, seed=0):
    """subjects x 4 templates around random centres (template-centre cos ~0.8) -> (gallery, centres of the first
    65536 subjects, seconds spent in gallery.add)."""
    rng, g, t_add, centres = np.random.default_rng(seed), Gallery(gpu=True), 0.0, None
    for s0 in range(0, subjects, 1 << 16):
        m = min(1 << 16, subjects - s0)
        c = unit(rng.standard_normal((m, D), dtype=np.float32))
        t = unit(c[:, None, :] + 0.75 / np.sqrt(D) * rng.standard_normal((m, 4, D), dtype=np.float32)).reshape(-1, D)
        centres = c if centres is None else centres
        sids = [f"s{s}" for s in range(s0, s0 + m) for _ in range(4)]
        t0 = time.perf_counter()
        g.add(np.arange(4 * s0, 4 * (s0 + m)), sids, t)
        t_add += time.perf_counter() - t0
    return g, centres, t_add


class BenchReplica(appmod.Replica):
    """The service's batching/search path on an injected gallery (no DB/KMS maintenance)."""

    def __init__(self, g):
        super().__init__("", None, None, "bench")
        self.gallery, self.model_version, self.last_ok = g, MV, float("inf")
        self.batch_sizes = []

    def run(self, stop):
        stop.wait()

    def search_batch(self, jobs):
        self.batch_sizes.append(len(jobs))
        super().search_batch(jobs)


def open_loop(r, centres, rate, seconds, rng, n_probes=3):
    """Poisson arrivals at `rate`/s through Replica.submit (the batch thread is running)."""
    lat, hits, futs, t_start = [], [], [], time.perf_counter()
    t_next = t_start
    r.batch_sizes.clear()
    while t_next - t_start < seconds:
        time.sleep(max(0.0, t_next - time.perf_counter()))
        s = int(rng.integers(len(centres)))
        q = unit(probes_for(centres, s, n_probes, rng).mean(0)).astype(np.float32)
        t0 = time.perf_counter()
        f = r.submit(MV, q, 5)
        f.add_done_callback(lambda f, t0=t0, s=s: (lat.append(time.perf_counter() - t0),
                                                   hits.append(f.result()[0][0][0] == f"s{s}")))
        futs.append(f)
        t_next += rng.exponential(1 / rate)
    deadline = time.perf_counter() + 10
    done = sum(f.done() for f in futs)
    while done < len(futs) and time.perf_counter() < deadline:
        time.sleep(0.05)
        done = sum(f.done() for f in futs)
    el = time.perf_counter() - t_start
    out = {"offered_per_s": rate, "sent": len(futs), "completed_in_time": done,
           "achieved_per_s": round(done / el, 1), **pct(lat), "top1_correct": round(float(np.mean(hits)), 4),
           "mean_batch": round(float(np.mean(r.batch_sizes)), 2) if r.batch_sizes else None}
    while not all(f.done() for f in futs):     # drain before the next run
        time.sleep(0.1)
    return out


def http_client(url, rate, seconds, seed, n_centres, q):
    """Separate process (no GIL sharing with the server): Poisson load of n=3 probe requests over HTTP."""
    import httpx
    import msgpack
    rng = np.random.default_rng(seed)
    centres = unit(np.random.default_rng(0).standard_normal((n_centres, D), dtype=np.float32))   # = build_gallery's

    async def main():
        lat, codes, hits = [], [], []
        async with httpx.AsyncClient(base_url=url, timeout=30, limits=httpx.Limits(max_connections=1000)) as c:
            async def one(s):
                body = msgpack.packb({"model_version": MV, "probes": probes_for(centres, s, 3, rng).tobytes(),
                                      "n": 3, "k": 5})
                t0 = time.perf_counter()
                r = await c.post("/v1/search", content=body)
                lat.append(time.perf_counter() - t0)
                codes.append(r.status_code)
                hits.append(r.status_code == 200 and msgpack.unpackb(r.content)["results"][0]["subject_id"] == f"s{s}")
            tasks, t_start = [], time.perf_counter()
            t_next = t_start
            while t_next - t_start < seconds:
                await asyncio.sleep(max(0.0, t_next - time.perf_counter()))
                tasks.append(asyncio.create_task(one(int(rng.integers(n_centres)))))
                t_next += rng.exponential(1 / rate)
            await asyncio.gather(*tasks)
            el = time.perf_counter() - t_start
        return {"sent": len(tasks), "achieved_per_s": len(tasks) / el, "non_200": sum(c != 200 for c in codes),
                "lat": lat, "top1_correct": float(np.mean(hits))}

    q.put(asyncio.run(main()))


def bench_gpu(subjects, http_port=18002):
    import faiss
    import uvicorn
    res, rng = {}, np.random.default_rng(1)
    res["gpu_state_at_start"] = gpu_state()          # GPU 5 is shared: other processes skew latencies
    mem0 = gpu_used_mib()
    t0 = time.perf_counter()
    g, centres, t_add = build_gallery(subjects)
    res["gallery"] = {"subjects": subjects, "templates": g.ntotal, "dim": D, "faiss": faiss.__version__,
                      "storage": f"float32 GpuIndexFlatIP x {len(g.segs)} segments of <= {g.seg_rows} rows",
                      "index_bytes": g.ntotal * D * 4, "gpu_mem_delta_mib": gpu_used_mib() - mem0,
                      "add_seconds": round(t_add, 2),
                      "build_seconds_incl_synthesis": round(time.perf_counter() - t0, 1)}
    print(json.dumps(res["gallery"]), flush=True)

    res["single_request"] = {}                       # sequential: aggregate + exact top-5 subjects, no HTTP
    for n in (1, 2, 3):
        lat, hits = [], []
        for i in range(300):
            p = probes_for(centres, i, n, rng)
            t0 = time.perf_counter()
            out = g.search(unit(p.mean(0, dtype=np.float64)).astype(np.float32)[None], 5)
            lat.append(time.perf_counter() - t0)
            hits.append(out[0][0][0] == f"s{i}")
        res["single_request"][f"n_probes={n}"] = {**pct(lat), "top1_correct": round(float(np.mean(hits)), 4)}
    print(json.dumps(res["single_request"]), flush=True)

    res["batched_calls"] = []                        # one gallery.search call with B queries (k=5)
    Q = unit(rng.standard_normal((256, D))).astype(np.float32)
    for b in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        g.search(Q[:b], 5)
        ts = []
        for _ in range(10):
            t0 = time.perf_counter()
            g.search(Q[:b], 5)
            ts.append(time.perf_counter() - t0)
        ms = float(np.median(ts)) * 1e3
        res["batched_calls"].append({"batch": b, "ms_per_call": round(ms, 2), "ms_per_query": round(ms / b, 3),
                                     "queries_per_s": round(1e3 * b / ms, 1)})
    print(json.dumps(res["batched_calls"]), flush=True)

    r = BenchReplica(g)
    threading.Thread(target=r.batch_loop, daemon=True).start()
    res["replica_open_loop"] = {}                    # Replica.submit -> batch thread -> GPU, Poisson arrivals
    for name, max_batch in (("micro_batching", appmod.MAX_BATCH), ("no_batching", 1)):
        saved, appmod.MAX_BATCH = appmod.MAX_BATCH, max_batch
        res["replica_open_loop"][name] = [open_loop(r, centres, rate, 10, rng) for rate in (25, 50, 100, 200, 400)]
        appmod.MAX_BATCH = saved
        print(name, json.dumps(res["replica_open_loop"][name]), flush=True)
    r.jobs.put(None)

    r = BenchReplica(g)                              # full service: uvicorn + FastAPI + msgpack, client in a process
    server = uvicorn.Server(uvicorn.Config(appmod.create_app(r), host="127.0.0.1", port=http_port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.1)
    res["http_open_loop"] = []
    ctx = mp.get_context("spawn")
    for rate in (50, 100, 200, 400):
        procs = max(1, rate // 100)                  # one httpx client process tops out at a few hundred req/s
        q = ctx.Queue()
        ps = [ctx.Process(target=http_client, args=(f"http://127.0.0.1:{http_port}", rate / procs, 10, rate + i,
                                                    len(centres), q)) for i in range(procs)]
        for p in ps:
            p.start()
        parts = [q.get() for _ in ps]
        for p in ps:
            p.join()
        lat = [x for part in parts for x in part.pop("lat")]
        res["http_open_loop"].append({"offered_per_s": rate, "client_processes": procs,
                                      "sent": sum(p["sent"] for p in parts),
                                      "achieved_per_s": round(sum(p["achieved_per_s"] for p in parts), 1),
                                      "non_200": sum(p["non_200"] for p in parts), **pct(lat),
                                      "top1_correct": round(float(np.mean([p["top1_correct"] for p in parts])), 4)})
        print(json.dumps(res["http_open_loop"][-1]), flush=True)
    server.should_exit = True
    return res


def bench_startup():
    """Replica build from the configured DB + KMS (what a restart costs: feeds RTO)."""
    import psycopg
    r = appmod.Replica.from_env()
    with psycopg.connect(r.dsn, password=r.password) as conn, conn.cursor(name="scan") as cur:
        t0 = time.perf_counter()
        cur.execute("SELECT id, subject_id::text, ciphertext FROM templates WHERE active")
        rows = nbytes = 0
        while batch := cur.fetchmany(appmod.KMS_BATCH):
            rows += len(batch)
            nbytes += sum(len(b[2] or b"") for b in batch)
        t_scan = time.perf_counter() - t0
    mem0, state = gpu_used_mib(), gpu_state()
    r.build()
    st = r.status()
    return {"db_scan_only_seconds": round(t_scan, 1), "rows": rows, "ciphertext_bytes": nbytes,
            "build_seconds": st["build_seconds"], "templates": st["ntotal"], "subjects": st["nsubjects"],
            "model_version": st["model_version"], "gpu_mem_delta_mib": gpu_used_mib() - mem0,
            "kms_batch": appmod.KMS_BATCH, "kms_concurrency": appmod.KMS_CONCURRENCY, "gpu_state_at_start": state}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, default=1_000_000)
    ap.add_argument("--startup", action="store_true")
    ap.add_argument("--kms-batch", type=int, default=appmod.KMS_BATCH)
    ap.add_argument("--kms-concurrency", type=int, default=appmod.KMS_CONCURRENCY)
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    a = ap.parse_args()
    appmod.KMS_BATCH, appmod.KMS_CONCURRENCY = a.kms_batch, a.kms_concurrency
    res = json.loads(a.out.read_text()) if a.out.exists() else {}
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True)
    res.update({"id": "E9_search_microbench", "gpu": gpu.stdout.strip() + " (host GPU 5)"})
    if a.startup:
        res.setdefault("startup", []).append({**bench_startup(),
                                              "date": datetime.datetime.now().isoformat(timespec="seconds")})
    else:
        res.update(bench_gpu(a.subjects), date=datetime.datetime.now().isoformat(timespec="seconds"))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
