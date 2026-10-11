"""E7 PAD: render the synthetic RGB-D-IR dataset and train / calibrate the 3-modality PAD model.

  python scripts/train_pad.py gen                              # -> data/datasets/pad_synth (CelebA faces)
  python scripts/train_pad.py train --protocol seen            # -> data/models/pad_rgbdir (deployed)
  python scripts/train_pad.py train --protocol unseen_species --out data/tmp/pad/unseen_species
Runs in faceauth:dev (GPU 5 = cuda:0, < 10 GB). SYNTHETIC data only (faceauth/sim): results validate the
pipeline and metric computation, not real-world PAD performance (real sensor data: hardware phase).
"""
import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import msgpack
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from faceauth.common.packet import decode_frame, encode_frame  # noqa: E402
from faceauth.models import pad as padm  # noqa: E402
from faceauth.sim.sensor import ATTACKS, DEFAULT_PROFILE, SensorSim  # noqa: E402

DATA = Path(os.environ.get("PAD_SYNTH", ROOT / "data/datasets/pad_synth"))  # env override for smoke tests
CELEBA_IMG = ROOT / "data/tmp/pad/celeba_align/img_align_celeba"  # official img_align_celeba (see DATASET.json)
CELEBA_ANNO = ROOT / "data/datasets/celeba/raw/Anno"
SPECIES = list(ATTACKS)
CHECKS = dict(min_valid_ratio=0.5, min_relief_mm=3.0, min_ir_median=30.0, min_shading_corr=0.1)  # physics priors
BPCER_TARGET = 0.01
PROTOCOLS = {"seen": SPECIES, "unseen_species": ["print", "replay"]}  # species used for training + calibration


# ---------------------------------------------------------------------------------------------- sources
def celeba_sources(img_dir: Path, anno: Path, aligned=True):
    """CelebA images with the official 5 landmarks (image-left eye first) and identity labels."""
    ident = dict(line.split() for line in open(anno / "identity_CelebA.txt"))
    lm = anno / ("list_landmarks_align_celeba.txt" if aligned else "list_landmarks_celeba.txt")
    min_iod = 30.0 if aligned else 40.0
    out = []
    for line in open(lm).read().splitlines()[2:]:
        name, *v = line.split()
        k = np.array(v, np.float32).reshape(5, 2)
        iod = np.linalg.norm(k[1] - k[0])
        frac = (k[2, 0] - k[0, 0]) / max(k[1, 0] - k[0, 0], 1e-6)  # nose between the eyes -> |yaw| moderate
        roll = np.degrees(np.arctan2(k[1, 1] - k[0, 1], k[1, 0] - k[0, 0]))
        if iod >= min_iod and 0.2 < frac < 0.8 and abs(roll) < 25 and k[3, 1] > k[2, 1] > k[0, 1]:
            out.append(dict(img=str(img_dir / name), kps=k.tolist(), group=ident[name], src=f"celeba/{name}"))
    return out


def split_groups(sources, n_groups, per_group, seed):
    """Group (identity) disjoint train/val/test split -> {split: [source, ...]}."""
    by = {}
    for s in sources:
        by.setdefault(s["group"], []).append(s)
    groups = sorted(g for g, v in by.items() if len(v) >= per_group)
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    out, i = {}, 0
    for split, n in zip(("train", "val", "test"), n_groups):
        if i + n > len(groups):
            raise SystemExit(f"not enough identities with >= {per_group} images ({len(groups)})")
        out[split] = [s for g in groups[i:i + n] for s in sorted(by[g], key=lambda s: s["src"])[:per_group]]
        i += n
    return out


# ---------------------------------------------------------------------------------------------- gen
_SIMS = {}


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o.item() if isinstance(o, np.generic) else o


def _render(task):
    """One source image x profile -> all kinds (all or nothing). Returns manifest rows or an error row."""
    sim = _SIMS.setdefault(task["profile"], SensorSim(task["profile"]))
    img = cv2.imread(task["img"], cv2.IMREAD_COLOR)
    if img is None:
        return [dict(task, error="unreadable image")]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    recs = []
    for j, kind in enumerate(task["kinds"]):
        seed = task["seed"] * 16 + j
        try:
            o = sim.render("live" if kind == "inject_live" else kind, img, task["kps"], seed=seed)
        except ValueError as e:
            return [dict(task, error=f"{kind}: {e}")]
        b = msgpack.packb(dict(encode_frame(o["rgb"], o["depth"], o["ir"]), kps5=o["kps5"].tolist(),
                               kind=kind, meta=_jsonable(o["meta"])))
        rel = f"frames/{task['split']}_{task['profile']}/{task['uid']}_{kind}.msgpack"
        recs.append((rel, b, dict(file=rel, sha256=hashlib.sha256(b).hexdigest(), split=task["split"],
                                  profile=task["profile"], kind=kind, group=task["group"], src=task["src"],
                                  seed=seed, meta=_jsonable(o["meta"]))))
    rows = []
    for rel, b, row in recs:
        (DATA / rel).parent.mkdir(parents=True, exist_ok=True)
        (DATA / rel).write_bytes(b)
        rows.append(row)
    return rows


def cmd_gen(a):
    t0 = time.time()
    srcs = celeba_sources(Path(a.celeba_img), Path(a.celeba_anno), aligned=not a.wild)
    splits = split_groups(srcs, a.ids, a.per_id, a.seed)
    kinds = ["live"] + SPECIES
    plan = {"train": [(DEFAULT_PROFILE, kinds)], "val": [(DEFAULT_PROFILE, kinds)],
            "test": [(DEFAULT_PROFILE, kinds + ["inject_live"]), ("sensor_b", kinds)]}
    tasks = []
    for split, items in splits.items():
        for prof, ks in plan[split]:
            for i, s in enumerate(items):
                tasks.append(dict(img=s["img"], kps=s["kps"], group=s["group"], src=s["src"], split=split,
                                  profile=prof, kinds=ks, uid=f"{i:05d}", seed=a.seed * 10 ** 6 + len(tasks)))
    print(f"{len(srcs)} usable CelebA images; tasks {len(tasks)}", flush=True)
    rows, errors = [], []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_render, tasks, chunksize=4)):
            (errors if "error" in r[0] else rows).extend(r)
            if i % 500 == 0:
                print(f"{i}/{len(tasks)} {time.time() - t0:.0f}s", flush=True)
    rows.sort(key=lambda r: r["file"])
    with open(DATA / "manifest.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    counts = {}
    for r in rows:
        key = f"{r['split']}/{r['profile']}/{r['kind']}"
        counts[key] = counts.get(key, 0) + 1
    ids = {sp: len({r["group"] for r in rows if r["split"] == sp}) for sp in ("train", "val", "test")}
    meta = dict(
        name="pad_synth", version="1",
        source="Rendered by faceauth/sim/sensor.py (SensorSim profiles gemini336-sim-v1, sensor_b) from CelebA "
               "align&cropped images (img_align_celeba, 202,599 JPEGs, from the HF mirror "
               "https://huggingface.co/datasets/student/celebA rev 2d31e6555722815c74ea7c845b07c1063dd705e9, "
               "Dataset.zip sha256 9464985fa6cfc84d0fe6b87a6e4f07a275c8bd3dfc8abd713218765de7d62306, extracted to "
               "data/tmp/pad/celeba_align) with the OFFICIAL annotations list_landmarks_align_celeba.txt + "
               "identity_CelebA.txt (data/datasets/celeba/raw/Anno); MediaPipe canonical face mesh (Apache-2.0) "
               "for live/mask geometry. Command: python scripts/train_pad.py gen " + " ".join(sys.argv[2:]),
        license="Derived from CelebA: 'use is RESTRICTED to non-commercial research and educational purposes' "
                "(CelebA README; CUHK does not own the image copyrights). NON-COMMERCIAL / RESEARCH-ONLY. "
                "Simulator assets: MediaPipe canonical_face_model.obj, Apache-2.0.",
        files={"manifest.jsonl": hashlib.sha256((DATA / "manifest.jsonl").read_bytes()).hexdigest()},
        counts=dict(frames=len(rows), per_split_profile_kind=counts, identities=ids, failed_sources=len(errors)),
        notes="SYNTHETIC presentations: live subjects, print (flat/bent), replay (phone/tablet/monitor), "
              "mask_rigid, mask_silicone; 'inject_live' = perfect synthetic live frames used as the digital-"
              "injection attack in eval. Identity-disjoint train/val/test (CelebA identity labels, seed "
              f"{a.seed}). Each frame file is msgpack {{rgb: JPEG q95, depth: PNG uint16 mm, ir: PNG uint8, "
              "kps5, kind, meta} (packet.encode_frame wire format); per-file sha256 in manifest.jsonl. "
              "Not real sensor data: validates the pipeline only.",
        split=dict(seed=a.seed, identities=dict(zip(("train", "val", "test"), a.ids)), images_per_identity=a.per_id),
        errors=errors[:50])
    (DATA / "DATASET.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta["counts"], indent=1), f"{time.time() - t0:.0f}s")


# ---------------------------------------------------------------------------------------------- train
def load_rows(split, profile=DEFAULT_PROFILE, species=SPECIES):
    rows = [json.loads(line) for line in open(DATA / "manifest.jsonl")]
    return [r for r in rows if r["split"] == split and r["profile"] == profile and r["kind"] in ["live"] + species]


def read_frame(row):
    rec = msgpack.unpackb((DATA / row["file"]).read_bytes())
    return (*decode_frame(rec), np.asarray(rec["kps5"], np.float32))


def jitter_kps(k, rng):
    """Detector-like keypoint noise: per point, global shift and scale."""
    iod = np.linalg.norm(k[1] - k[0])
    c = k.mean(0)
    k = c + (k - c) * np.exp(rng.normal(0, 0.05)) + rng.normal(0, 0.04 * iod, 2)
    return (k + rng.normal(0, 0.03 * iod, k.shape)).astype(np.float32)


def augment(rc, dc, ic, rng):
    if rng.random() < 0.5:
        rc, dc, ic = rc[:, ::-1], dc[:, ::-1], ic[:, ::-1]
    r = rc.astype(np.float32) / 255 * rng.uniform(0.75, 1.25) + rng.uniform(-0.08, 0.08)
    g = r.mean(-1, keepdims=True)
    r = np.clip(g + rng.uniform(0.7, 1.3) * (r - g), 0, 1) ** rng.uniform(0.8, 1.25)
    i = np.clip(ic.astype(np.float32) / 255 * rng.uniform(0.7, 1.4), 0, 1) ** rng.uniform(0.8, 1.25)
    i = i + rng.normal(0, rng.uniform(0, 0.02), i.shape)
    d = dc.astype(np.float32)
    d = np.where(d > 0, d + rng.normal(0, rng.uniform(0, 2.0), d.shape), 0)
    if rng.random() < 0.2:  # extra depth holes
        y, x = rng.integers(0, d.shape[0], 2)
        d[y:y + rng.integers(8, 40), x:x + rng.integers(8, 40)] = 0
    return ((np.clip(r, 0, 1) * 255).astype(np.uint8), np.clip(d, 0, 65535).astype(np.uint16),
            (np.clip(i, 0, 1) * 255).astype(np.uint8))


class FrameSet:
    def __init__(self, rows, train, seed=0):
        self.rows, self.train, self.seed, self.epoch = rows, train, seed, 0
        self.box = padm.core_box()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        rgb, depth, ir, k = read_frame(self.rows[i])
        rng = np.random.default_rng((self.seed, self.epoch, i))
        if self.train:
            k = jitter_kps(k, rng)
        rc, dc, ic, _ = padm.crops(rgb, depth, ir, k)
        if self.train:
            rc, dc, ic = augment(rc, dc, ic, rng)
        x = padm.to_input(rc, dc, ic, self.box)
        if self.train:  # modality dropout (never all three)
            for sl in [m for m in (slice(0, 3), slice(3, 5), slice(5, 6)) if rng.random() < 0.1][:2]:
                x[sl] = 0
        return x, np.float32(self.rows[i]["kind"] == "live")


def _feat(args):
    row, size, zoom, fx = args
    return padm.face_features(*read_frame(row), size, zoom, fx)


def score_rows(pad, rows, workers=32, chunk=4096):
    """-> pad.decide() dict per row (decode/crop/physics on CPU workers, CNN on the PAD device)."""
    out = []
    with Pool(workers) as pool:
        for i in range(0, len(rows), chunk):
            f = pool.map(_feat, [(r, pad.size, pad.zoom, pad.fx) for r in rows[i:i + chunk]], chunksize=8)
            out += [pad.decide(p, c) for p, (_, c) in zip(pad.cnn(np.stack([x for x, _ in f])), f)]
    return out


def calibrate(model_dir, rows):
    """Physics checks are priors; the CNN threshold is the largest value keeping val BPCER <= target."""
    pad = padm.PAD(model_dir)
    res = [dict(r, kind=row["kind"]) for r, row in zip(score_rows(pad, rows), rows)]
    bf = [r for r in res if r["kind"] == "live" and not r["nonresponse"]]
    ok = np.sort([r["cnn"] for r in bf if not r["failed"]])
    allowed = int(np.floor(BPCER_TARGET * len(bf))) - (len(bf) - len(ok))
    thr = float(ok[max(allowed, 0)]) if len(ok) else 0.5
    live = lambda r: not r["nonresponse"] and not r["failed"] and r["cnn"] >= thr  # noqa: E731
    q = lambda k: np.nanpercentile([r["checks"][k] for r in bf], [0.5, 5, 50]).round(3).tolist()  # noqa: E731
    val = dict(n_bona_fide=len(bf), bpcer=float(np.mean([not live(r) for r in bf])),
               bpcer_checks_only=float(np.mean([bool(r["failed"]) for r in bf])),
               check_fail_bona_fide={c: int(sum(c in r["failed"] for r in bf)) for c in ("relief_mm", "ir_median", "shading_corr")},
               bona_fide_quantiles_0p5_5_50={k: q(k) for k in ("valid_ratio", "relief_mm", "ir_median", "shading_corr")})
    for sp in sorted({r["kind"] for r in res} - {"live"}):
        a = [r for r in res if r["kind"] == sp]
        dec = [r for r in a if not r["nonresponse"]]
        val[sp] = dict(n=len(a), nonresponse=len(a) - len(dec), apcer=float(np.mean([live(r) for r in dec])) if dec else None)
    return thr, val


def cmd_train(a):
    import torch
    torch.manual_seed(a.seed)
    species = PROTOCOLS[a.protocol]
    tr, va = load_rows("train", species=species), load_rows("val", species=species)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n_live = sum(r["kind"] == "live" for r in tr)
    print(f"protocol {a.protocol}: train {len(tr)} ({n_live} live), val {len(va)}", flush=True)
    ds_tr, ds_va = FrameSet(tr, True, a.seed), FrameSet(va, False)
    gen = torch.Generator().manual_seed(a.seed)
    dl = lambda ds, sh: torch.utils.data.DataLoader(ds, a.batch, shuffle=sh, num_workers=a.workers,  # noqa: E731
                                                    drop_last=sh, generator=gen)
    dev = torch.device("cuda")
    net = padm.PADNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = max(a.epochs * (len(tr) // a.batch), 100)  # (floor only matters for tiny smoke runs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(tr) - n_live) / n_live, device=dev))
    best, log = 1e9, []
    for ep in range(a.epochs):
        ds_tr.epoch = ep
        net.train()
        t0, tl = time.time(), []
        for x, y in dl(ds_tr, True):
            with torch.autocast("cuda", torch.bfloat16):
                loss = lossf(net(x.to(dev, non_blocking=True)).float(), y.to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tl.append(loss.item())
        net.eval()
        ps, ys = [], []
        with torch.no_grad():
            for x, y in dl(ds_va, False):
                ps.append(torch.sigmoid(net(x.to(dev)).float()).cpu().numpy())
                ys.append(y.numpy())
        p, y = np.concatenate(ps), np.concatenate(ys)
        vl = float(-np.mean(y * np.log(p + 1e-7) * lossf.pos_weight.item() + (1 - y) * np.log(1 - p + 1e-7)))
        rec = dict(epoch=ep, train_loss=float(np.mean(tl)), val_loss=vl, val_bpcer05=float(np.mean(p[y == 1] < 0.5)),
                   val_apcer05=float(np.mean(p[y == 0] >= 0.5)), sec=round(time.time() - t0, 1))
        log.append(rec)
        print(json.dumps(rec), flush=True)
        if vl < best:
            best = vl
            torch.save(net.state_dict(), out / "pad_rgbdir.pt")
    meta = dict(
        name="pad_rgbdir" if a.protocol == "seen" else f"pad_rgbdir_{a.protocol}", version="1", kind="pad",
        files={"weights": "pad_rgbdir.pt"}, sha256={"weights": hashlib.sha256((out / "pad_rgbdir.pt").read_bytes()).hexdigest()},
        source="Trained in-house on SYNTHETIC data (faceauth/sim + scripts/train_pad.py, protocol "
               f"{a.protocol}); not validated on any real sensor.",
        license="In-house weights; trained on renders of CelebA images -> treat as NON-COMMERCIAL / RESEARCH-ONLY "
                "(CelebA: non-commercial research and educational use only). Replace by weights trained on "
                "consented real-sensor data before any deployment.",
        params=dict(threshold=0.5, checks=CHECKS, fx=385.0, calib_version=DEFAULT_PROFILE,
                    input=dict(size=padm.SIZE, zoom=padm.ZOOM, channels=["r", "g", "b", "depth_rel", "depth_valid", "ir"],
                               align="faceauth.common.align.estimate_norm zoomed out by `zoom`",
                               depth="(d - median(face core)) / 100 mm clipped to [-1,1], invalid 0",
                               rgb_ir="(x/255 - 0.5)/0.25"),
                    train=dict(protocol=a.protocol, species=species, epochs=a.epochs, batch=a.batch, lr=a.lr, seed=a.seed,
                               dataset_manifest_sha256=hashlib.sha256((DATA / "manifest.jsonl").read_bytes()).hexdigest(),
                               n_train=len(tr), n_val=len(va), log=log)))
    (out / "MODEL.json").write_text(json.dumps(meta, indent=1))
    thr, val = calibrate(out, va)
    meta["params"]["threshold"], meta["params"]["val"] = thr, val
    (out / "MODEL.json").write_text(json.dumps(meta, indent=1))
    print("threshold", thr, json.dumps(val, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--celeba-img", default=str(CELEBA_IMG))
    g.add_argument("--celeba-anno", default=str(CELEBA_ANNO))
    g.add_argument("--wild", action="store_true", help="in-the-wild images + list_landmarks_celeba.txt")
    g.add_argument("--ids", type=int, nargs=3, default=[2000, 500, 1000], help="identities train val test")
    g.add_argument("--per-id", type=int, default=2)
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--workers", type=int, default=48)
    t = sub.add_parser("train")
    t.add_argument("--protocol", choices=list(PROTOCOLS), default="seen")
    t.add_argument("--out", default=str(ROOT / "data/models/pad_rgbdir"))
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--batch", type=int, default=128)
    t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    cmd_gen(a) if a.cmd == "gen" else cmd_train(a)
