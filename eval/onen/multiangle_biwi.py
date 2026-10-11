"""Face ID-style multi-angle enrollment vs frontal-only on REAL camera frames (BIWI Kinect RGB, two sessions).

    python eval/onen/multiangle_biwi.py   # -> reports/E2_1N/faceid_style_biwi.json
"""
import json, collections, numpy as np
from faceauth.sim import biwi
from faceauth.models.scrfd import SCRFD
from faceauth.models.recog import Recognizer
from faceauth.terminal.guide import direction_of, SLOTS, ORDER
det, rec = SCRFD("data/models/scrfd_34g_320"), Recognizer("data/models/adaface_ir101_wf12m")
rows = []   # (subject, seq, frame, yaw, pitch, emb)
for f in biwi.iter_frames("data/datasets/biwi", register=False, step=2):
    d = det.detect(f["rgb"])
    if not d: continue
    b = max(d, key=lambda x: (x["bbox"][2]-x["bbox"][0]) * (x["bbox"][3]-x["bbox"][1]))
    rows.append((f["subject"], f["seq"], f["frame"], float(f["pose"][0]), float(f["pose"][1]), f["rgb"], b["kps"]))
E = np.concatenate([rec.embed([r[5] for r in rows[i:i+256]], [r[6] for r in rows[i:i+256]]) for i in range(0, len(rows), 256)])
meta = [(r[0], r[1], r[2], r[3], r[4]) for r in rows]
print("frames embedded", len(meta), flush=True)
def unit(v): return v / np.linalg.norm(v)
by_seq = collections.defaultdict(list)
for i, m in enumerate(meta): by_seq[(m[0], m[1])].append(i)
def templates(idx, mode):
    """Enrollment from one session: per direction bin (GT pose) the 2 frames closest to the bin centre."""
    from faceauth.terminal.guide import TARGETS
    per = collections.defaultdict(list)
    for i in idx:
        dname = direction_of(meta[i][3], meta[i][4])
        if dname: per[dname].append(i)
    pick = {k: sorted(v, key=lambda i: np.hypot(meta[i][3] - TARGETS[k][0], meta[i][4] - TARGETS[k][1]))[:2] for k, v in per.items()}
    if "C" not in pick: return None
    if mode == "single_photo": return [E[pick["C"][0]]]
    if mode == "frontal": return [unit(E[pick["C"]].mean(0))]
    if mode == "9dir": return [unit(E[v].mean(0)) for v in pick.values()]
    slots = collections.defaultdict(list)
    for k, v in pick.items(): slots[SLOTS[k]].extend(v)
    return [unit(E[v].mean(0)) for v in slots.values()]                      # "4slot" = deployed policy
def probes(idx, lo, hi):
    """3 consecutive frames (as one authentication) whose |yaw| in [lo, hi) and |pitch| <= 15."""
    out, idx = [], sorted(idx, key=lambda i: meta[i][2])
    for j in range(0, len(idx) - 2, 3):
        w = idx[j:j+3]
        if all(lo <= abs(meta[i][3]) < hi and abs(meta[i][4]) <= 15 for i in w): out.append(unit(E[w].mean(0)))
    return out
PAIRS = {"F02": ("02", "21"), "F03": ("03", "15"), "F05": ("05", "18"), "M01": ("07", "22")}
res = {}
for mode in ("single_photo", "frontal", "4slot", "9dir"):
    gen = {"frontal_probe": [], "turned_probe": []}; imp = []
    gal = {k: templates(v, mode) for k, v in by_seq.items()}
    for subj, (a, b) in PAIRS.items():
        for enr, auth in ((a, b), (b, a)):
            T = gal[(subj, enr)]
            if T is None: continue
            for kind, (lo, hi) in (("frontal_probe", (0, 20)), ("turned_probe", (20, 35))):
                for p in probes(by_seq[(subj, auth)], lo, hi):
                    gen[kind].append(max(float(t @ p) for t in T))
                    imp.append(max(float(t @ p) for (s, q), TT in gal.items() if s != subj and TT for t in TT))
    r = {}
    for kind, g in gen.items():
        g = np.array(g)
        r[kind] = {"n": len(g), "min": round(float(g.min()), 3), "p5": round(float(np.percentile(g, 5)), 3),
                   "median": round(float(np.median(g)), 3), "below_0.60": round(float((g < 0.60).mean()), 4),
                   "below_0.617": round(float((g < 0.617).mean()), 4)} if len(g) else None
    r["best_impostor_over_20_subjects"] = round(float(max(imp)), 3)
    res[mode] = r
    print(mode, json.dumps(r), flush=True)
json.dump(res, open("reports/E2_1N/faceid_style_biwi.json", "w"), indent=1)
