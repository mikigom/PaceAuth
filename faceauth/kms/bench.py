"""KMS decrypt throughput as the search service sees it at startup (N subjects x 4 templates of 512 float32).

    docker run --rm --network <project>_internal -u $(id -u):$(id -g) -e HOME=/tmp -v $PWD:/app -w /app \
        faceauth:dev python -m faceauth.kms.bench --subjects 100000
Dev stacks only: creates N subject keys in DB kms and crypto-shreds them at the end. Prints one JSON line.
"""
import argparse
import concurrent.futures as cf
import json
import os
import time
import uuid

import httpx
import msgpack


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://kms:8003")
    ap.add_argument("--subjects", type=int, default=100_000)
    ap.add_argument("--batch", type=int, default=8192, help="items per decrypt request")
    ap.add_argument("--concurrency", type=int, default=4, help="parallel decrypt requests")
    ap.add_argument("--secrets", default="secrets")
    a = ap.parse_args()
    tok = {n: open(os.path.join(a.secrets, f"kms_token_{n}")).read().strip() for n in ("api", "search")}
    pts = [os.urandom(2048) for _ in range(4)]

    def call(c, method, path, scope, obj=None):
        r = c.request(method, a.url + path, content=None if obj is None else msgpack.packb(obj),
                      headers={"Authorization": f"Bearer {tok[scope]}", "Content-Type": "application/msgpack"})
        r.raise_for_status()
        return msgpack.unpackb(r.content, raw=False) if obj is not None else r.json()

    sids = [str(uuid.uuid4()) for _ in range(a.subjects)]
    with httpx.Client(timeout=120, limits=httpx.Limits(max_connections=64)) as c, cf.ThreadPoolExecutor(32) as ex:
        t = time.perf_counter()
        cts = list(ex.map(lambda s: call(c, "POST", "/v1/encrypt", "api", {"subject_id": s, "items": pts})["items"], sids))
        t_enc = time.perf_counter() - t
        items = [{"subject_id": s, "ct": ct} for s, row in zip(sids, cts) for ct in row]
        batches = [items[i:i + a.batch] for i in range(0, len(items), a.batch)]
        with cf.ThreadPoolExecutor(a.concurrency) as dx:
            t = time.perf_counter()
            out = list(dx.map(lambda b: call(c, "POST", "/v1/decrypt", "search", {"items": b})["items"], batches))
            t_dec = time.perf_counter() - t
        ok = sum(pt == pts[i % 4] for row in out for i, pt in enumerate(row))  # batch % 4 == 0 keeps slots aligned
        list(ex.map(lambda s: call(c, "DELETE", f"/v1/keys/{s}", "api"), sids))
    n = len(items)
    print(json.dumps({"subjects": a.subjects, "items": n, "item_bytes": 2048, "batch": a.batch,
                      "concurrency": a.concurrency, "decrypt_s": round(t_dec, 2),
                      "decrypt_items_per_s": round(n / t_dec), "projected_4M_items_s": round(4e6 / (n / t_dec), 1),
                      "decrypted_ok": ok, "encrypt_calls_per_s": round(a.subjects / t_enc)}))
    assert ok == n, "decrypt mismatch"


if __name__ == "__main__":
    main()
