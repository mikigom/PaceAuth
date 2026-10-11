"""Deterministic (seeded) subject-disjoint open-set 1:N protocol (built by eval/onen/run.py on usable images only).

Identity dataset: subjects with >= min_images usable images are eligible for enrolment; a seeded `enroll_frac` of
them is enrolled with `n_templates` random images as gallery templates and the rest as mated probes. Every other
identity is non-enrolled and all its images are non-mated probes. Distractor datasets add gallery subjects (up to
`n_templates` images each) in a seeded `order`, continuing across datasets in the given order (so a gallery with
D distractors = order < D: the first datasets fill small galleries); probe_frac {name: F} turns a fraction F of a
dataset's subjects into non-mated probe identities instead (if the dataset has a `unit` column the split is by unit,
moved to a unit boundary, so probes and distractors never share a unit). Mated / non-mated images of one subject are also cut
into multi-image probe groups of `group_size` (variant: probe = normalised mean of the group's embeddings).
Every dataset draws from its own stream np.random.default_rng([seed, i]) (i = 0 identity, 1.. distractors in order),
so a dataset's selection does not depend on another dataset's content (e.g. on which CelebA images were detected).

Table: one row per image, sorted by path: path, subject, role in {template, mated, nonmated, distractor},
group (-1 = none), order (-1 = not a distractor).
"""
import numpy as np
import pandas as pd


def build(identity: pd.DataFrame, distractors=(), probe_frac=None, seed=0, enroll_frac=0.5, n_templates=4,
          min_images=5, max_probes=0, group_size=3) -> pd.DataFrame:
    rng = np.random.default_rng([seed, 0])
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
    for i, (name, df) in enumerate(distractors, 1):
        rng = np.random.default_rng([seed, i])
        d = df.groupby("subject", sort=True)["path"].apply(sorted)
        if "unit" in df:  # split by unit (e.g. FFHQ Flickr account): probes and distractors never share a unit
            units = df.groupby("unit", sort=True)["subject"].apply(lambda x: sorted(set(x)))
            subj = np.array([s for u in rng.permutation(units.index.to_numpy()) for s in units[u]], dtype=object)
        else:
            subj = rng.permutation(d.index.to_numpy())
        n_probe = int(round(probe_frac.get(name, 0.0) * len(subj)))
        if "unit" in df and n_probe:  # move the cut to the next unit boundary
            unit_of = df.drop_duplicates("subject").set_index("subject")["unit"]
            while n_probe < len(subj) and unit_of[subj[n_probe]] == unit_of[subj[n_probe - 1]]:
                n_probe += 1
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
    ds = p.subject.str.split(":").str[0]
    for role in ("template", "mated", "nonmated", "distractor"):
        r = p[p.role == role]
        c[role] = {"images": int(len(r)), "subjects": int(r.subject.nunique())}
    for role in ("mated", "nonmated"):
        r = p[(p.role == role) & (p.group >= 0)]
        c[role]["groups"] = int(r.group.nunique())
    c["nonmated"]["by_dataset"] = p[p.role == "nonmated"].groupby(ds).agg(
        images=("path", "size"), identities=("subject", "nunique")).to_dict("index")
    c["distractor"]["by_dataset"] = p[p.role == "distractor"].groupby(ds).agg(
        images=("path", "size"), subjects=("subject", "nunique")).to_dict("index")
    c["enrolled_subjects"] = c["template"]["subjects"]
    c["max_gallery_subjects"] = c["template"]["subjects"] + c["distractor"]["subjects"]
    c["max_gallery_templates"] = c["template"]["images"] + c["distractor"]["images"]
    return c
