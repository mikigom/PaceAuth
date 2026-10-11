"""Deterministic (seeded) subject-disjoint open-set 1:N protocol.

    python -m eval.onen.protocol --name celeba_v1 --identity celeba --distractors ffhq digiface1m \
        --probe-frac ffhq=0.5 --seed 0

Identity dataset: subjects with >= min_images images are eligible for enrolment; a seeded `enroll_frac` of them
is enrolled with `n_templates` random images as gallery templates and the rest as mated probes. Every other
identity is non-enrolled and all its images are non-mated probes. Distractor datasets add gallery subjects
(up to `n_templates` images each) in a seeded `order`, so a gallery of D distractors = order < D; with
--probe-frac NAME=F a fraction F of that dataset's subjects become non-mated probe identities instead.
Mated / non-mated images of one subject are also cut into multi-image probe groups of `group_size`.

Output reports/E2_1N/protocol/<name>/: images.csv.gz (one row per image, sorted by path:
path, subject, role in {template, mated, nonmated, distractor}, group (-1 = none), order (-1 = not a
distractor)) and protocol.json (parameters, counts, dataset sources).
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eval.onen import datasets

OUT = Path("reports/E2_1N/protocol")


def build(identity: pd.DataFrame, distractors=(), probe_frac=None, seed=0, enroll_frac=0.5, n_templates=4,
          min_images=5, max_probes=0, group_size=3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    probe_frac = probe_frac or {}
    rows = []

    def groups(paths, subject, role):
        for i, p in enumerate(paths):
            full = (i // group_size + 1) * group_size <= len(paths)
            rows.append((p, subject, role, i // group_size if full else -1, -1))

    by_subj = identity.groupby("subject", sort=True)["path"].apply(sorted)
    subjects = by_subj.index.to_numpy()
    eligible = subjects[by_subj.map(len).to_numpy() >= min_images]
    enrolled = set(rng.choice(eligible, int(round(enroll_frac * len(eligible))), replace=False))
    for s in subjects:
        paths = list(rng.permutation(by_subj[s]))
        if s in enrolled:
            rows += [(p, s, "template", -1, -1) for p in paths[:n_templates]]
            paths, role = paths[n_templates:], "mated"
        else:
            role = "nonmated"
        groups(paths[:max_probes] if max_probes else paths, s, role)

    order = 0
    for name, df in distractors:
        d = df.groupby("subject", sort=True)["path"].apply(sorted)
        subj = rng.permutation(d.index.to_numpy())
        n_probe = int(round(probe_frac.get(name, 0.0) * len(subj)))
        for s in subj[:n_probe]:
            groups(list(rng.permutation(d[s])), s, "nonmated")
        for s in subj[n_probe:]:
            rows += [(p, s, "distractor", -1, order) for p in rng.permutation(d[s])[:n_templates]]
            order += 1

    out = pd.DataFrame(rows, columns=["path", "subject", "role", "group", "order"])
    key = (out.subject + "|" + out.role + "|" + out.group.astype(str)).where(out.group >= 0)
    out["group"] = pd.factorize(key, sort=True)[0]  # global group id, -1 = not in a full group
    if out.path.duplicated().any():
        raise ValueError("an image appears twice (duplicate paths across datasets?)")
    return out.sort_values("path", kind="stable").reset_index(drop=True)


def counts(p: pd.DataFrame) -> dict:
    c = {}
    for role in ("template", "mated", "nonmated", "distractor"):
        r = p[p.role == role]
        c[role] = {"images": int(len(r)), "subjects": int(r.subject.nunique())}
    for role in ("mated", "nonmated"):
        r = p[(p.role == role) & (p.group >= 0)]
        c[role]["groups"] = int(r.group.nunique())
    c["nonmated"]["by_dataset"] = p[p.role == "nonmated"].groupby(p.subject.str.split(":").str[0]).agg(
        images=("path", "size"), identities=("subject", "nunique")).to_dict("index")
    c["distractor"]["by_dataset"] = p[p.role == "distractor"].groupby(p.subject.str.split(":").str[0]).agg(
        images=("path", "size"), subjects=("subject", "nunique")).to_dict("index")
    c["enrolled_subjects"] = c["template"]["subjects"]
    c["max_gallery_subjects"] = c["template"]["subjects"] + c["distractor"]["subjects"]
    c["max_gallery_templates"] = c["template"]["images"] + c["distractor"]["images"]
    return c


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--identity", default="celeba", choices=sorted(datasets.LOADERS))
    ap.add_argument("--celeba-images", default="img_celeba", help="img_celeba (in-the-wild) | img_align_celeba")
    ap.add_argument("--distractors", nargs="*", default=[], choices=sorted(datasets.LOADERS))
    ap.add_argument("--probe-frac", nargs="*", default=[], help="NAME=F: fraction of a distractor dataset's "
                    "subjects used as non-mated probe identities instead of gallery distractors")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--enroll-frac", type=float, default=0.5)
    ap.add_argument("--n-templates", type=int, default=4)
    ap.add_argument("--min-images", type=int, default=5)
    ap.add_argument("--max-probes", type=int, default=0, help="per subject, 0 = all")
    ap.add_argument("--group-size", type=int, default=3)
    ap.add_argument("--data", default=str(datasets.DATA))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)

    def load(name):
        df = datasets.celeba(a.data, a.celeba_images) if name == "celeba" else datasets.load(name, a.data)
        return df.sort_values("path", kind="stable").reset_index(drop=True)

    pf = {k: float(v) for k, v in (s.split("=") for s in a.probe_frac)}
    p = build(load(a.identity), [(n, load(n)) for n in a.distractors], pf, a.seed, a.enroll_frac, a.n_templates,
              a.min_images, a.max_probes, a.group_size)
    out = Path(a.out) / a.name
    out.mkdir(parents=True, exist_ok=True)
    p.to_csv(out / "images.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    src = {}
    for n in [a.identity, *a.distractors]:
        f = Path(a.data) / n / "DATASET.json"
        src[n] = {"dataset_json_sha256": hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else None}
    meta = {"name": a.name, "seed": a.seed, "identity": a.identity, "celeba_images": a.celeba_images,
            "distractors": a.distractors, "probe_frac": pf, "enroll_frac": a.enroll_frac,
            "n_templates": a.n_templates, "min_images": a.min_images, "max_probes": a.max_probes,
            "group_size": a.group_size, "sources": src, "counts": counts(p),
            "images_csv_sha256": hashlib.sha256((out / "images.csv.gz").read_bytes()).hexdigest()}
    (out / "protocol.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta["counts"], indent=2))


if __name__ == "__main__":
    main()
