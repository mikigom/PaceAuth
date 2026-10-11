"""E7: PAD evaluation per ISO/IEC 30107-3 on the synthetic RGB-D-IR test split (data/datasets/pad_synth).

  python eval/eval_pad.py   # -> reports/E7_pad/results.json + summary.md

Protocols: (a) seen species, (b) unseen species (model trained without masks, tested on masks),
(c) unseen sensor/location (default-profile model on sensor_b), (d) digital injection (perfect synthetic live
frames; RGB-only replay). SYNTHETIC data: validates the pipeline and metric computation only.
Definitions (per PAI species / bona fide, ISO/IEC 30107-3):
  non-response = no decision possible (frame undecodable / missing modality, or face-core valid depth below
                 min_valid_ratio); APNRR / BPNRR = non-responses / presentations
  APCER = attacks classified bona fide / attacks with a decision;  BPCER = bona fide classified attack /
  bona fide with a decision;  APCER_max = max over species;  ACER = (APCER_max + BPCER) / 2 (a summary figure
  from the PAD-challenge literature, not an ISO/IEC 30107-3 metric; reported for comparability only);
  attack success rate = accepted / all attack presentations (non-responses count as not accepted).
  95% Clopper-Pearson intervals.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import train_pad as tp  # noqa: E402
from faceauth.common.packet import decode_frame  # noqa: E402
from faceauth.models.pad import PAD  # noqa: E402
from faceauth.sim.capture import rgb_only_replay  # noqa: E402



def rate(k, n):
    if n == 0:
        return dict(k=0, n=0, value=None, ci95=None)
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return dict(k=int(k), n=int(n), value=k / n, ci95=[lo, hi])


def summarize(res, rule):
    """res: list of (kind, subtype, decision dict); rule(decision) -> accepted as bona fide."""
    out, species = {}, sorted({k for k, _, _ in res} - {"live"})
    bf = [d for k, _, d in res if k == "live"]
    if bf:
        dec = [d for d in bf if not d["nonresponse"]]
        out["bona_fide"] = dict(n=len(bf), bpnrr=rate(len(bf) - len(dec), len(bf)),
                                bpcer=rate(sum(not rule(d) for d in dec), len(dec)))
    tot_acc = tot_n = 0
    for sp in species:
        a = [(st, d) for k, st, d in res if k == sp]
        dec = [d for _, d in a if not d["nonresponse"]]
        acc = sum(rule(d) for d in dec)
        tot_acc, tot_n = tot_acc + acc, tot_n + len(a)
        out[sp] = dict(n=len(a), apnrr=rate(len(a) - len(dec), len(a)), apcer=rate(acc, len(dec)),
                       attack_success=rate(acc, len(a)))
        subs = sorted({st for st, _ in a if st})
        if len(subs) > 1:
            out[sp]["subtypes"] = {st: dict(n=sum(s == st for s, _ in a),
                                            nonresponse=sum(s == st and d["nonresponse"] for s, d in a),
                                            accepted=sum(s == st and not d["nonresponse"] and rule(d) for s, d in a))
                                   for st in subs}
    ap = [out[sp]["apcer"]["value"] for sp in species if out[sp]["apcer"]["value"] is not None]
    if ap:
        out["apcer_max"] = max(ap)
        out["worst_species"] = species[int(np.argmax([out[sp]["apcer"]["value"] or 0 for sp in species]))]
    if ap and bf:
        out["acer"] = (out["apcer_max"] + out["bona_fide"]["bpcer"]["value"]) / 2
    if species:
        out["attack_success_overall"] = rate(tot_acc, tot_n)
    return out


def run(pad, rows):
    t0 = time.time()
    dec = tp.score_rows(pad, rows)
    print(f"  scored {len(rows)} in {time.time() - t0:.0f}s", flush=True)
    return [(r["kind"], r["meta"].get("subtype"), d) for r, d in zip(rows, dec)]


def evaluate(pad, res):
    thr = pad.threshold
    rules = {"pad": lambda d: d["live"],  # CNN >= threshold AND checks (the deployed decision)
             "cnn_only": lambda d: d["cnn"] >= thr,
             "checks_only": lambda d: not d["failed"]}
    return {name: summarize(res, rule) for name, rule in rules.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seen", type=Path, default=ROOT / "data/models/pad_rgbdir")
    ap.add_argument("--unseen-species", type=Path, default=ROOT / "data/tmp/pad/unseen_species")
    ap.add_argument("--out", type=Path, default=ROOT / "reports/E7_pad")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    MODELS = {"seen": a.seen, "unseen_species": a.unseen_species}
    rows = [json.loads(line) for line in open(tp.DATA / "manifest.jsonl")]
    test = lambda prof, kinds: [r for r in rows if r["split"] == "test" and r["profile"] == prof and r["kind"] in kinds]  # noqa: E731
    pads = {k: PAD(v, a.device) for k, v in MODELS.items() if (v / "MODEL.json").exists()}
    kinds = ["live"] + tp.SPECIES
    out = dict(id="E7", name="pad", synthetic_only=True,
               caveat="All presentations are SIMULATED (faceauth/sim, no real sensor, no real PAI). These numbers "
                      "validate the pipeline and the metric computation only; they are NOT real-world PAD "
                      "performance. Real-sensor training and independent ISO/IEC 30107-3 testing are required "
                      "(hardware phase).",
               dataset=dict(name="pad_synth", manifest_sha256=hashlib.sha256((tp.DATA / "manifest.jsonl").read_bytes()).hexdigest(),
                            counts=json.loads((tp.DATA / "DATASET.json").read_text())["counts"]),
               models={k: dict(version=p.version, threshold=p.threshold, checks=p.checks,
                               weights_sha256=json.loads((MODELS[k] / "MODEL.json").read_text())["sha256"]["weights"])
                       for k, p in pads.items()},
               decision="live = CNN live probability >= threshold AND relief/IR/shading checks pass; "
                        "non-response if the face-core valid depth ratio < min_valid_ratio",
               ci="95% Clopper-Pearson", protocols={})
    P = out["protocols"]
    print("(a) seen", flush=True)
    P["a_seen"] = dict(model="seen", profile=tp.DEFAULT_PROFILE,
                       **evaluate(pads["seen"], run(pads["seen"], test(tp.DEFAULT_PROFILE, kinds))))
    if "unseen_species" in pads:
        print("(b) unseen species", flush=True)
        P["b_unseen_species"] = dict(model="unseen_species (trained on live+print+replay only)", profile=tp.DEFAULT_PROFILE,
                                     **evaluate(pads["unseen_species"], run(pads["unseen_species"], test(tp.DEFAULT_PROFILE, kinds))))
    print("(c) unseen sensor", flush=True)
    P["c_unseen_sensor"] = dict(model="seen", profile="sensor_b", **evaluate(pads["seen"], run(pads["seen"], test("sensor_b", kinds))))
    lat = []  # per-frame PAD latency (crop + physics + CNN), one frame per call, first (warm-up) call dropped
    for r in test(tp.DEFAULT_PROFILE, ["live"])[:51]:
        f = tp.read_frame(r)
        t0 = time.perf_counter()
        pads["seen"].predict(*f)
        lat.append(1000 * (time.perf_counter() - t0))
    out["latency_ms_per_frame"] = dict(device=a.device, n=len(lat[1:]), median=float(np.median(lat[1:])),
                                       p95=float(np.percentile(lat[1:], 95)), note="shared, heavily loaded host")
    print("(d) injection", flush=True)
    inj = run(pads["seen"], test(tp.DEFAULT_PROFILE, ["inject_live"]))
    live_rows = test(tp.DEFAULT_PROFILE, ["live"])
    nr = dict(score=0.0, cnn=0.0, live=False, nonresponse=True, failed=[], checks={})
    rgb_only = {}
    for fill in (None, "zero", "luma"):
        res = []
        for r in live_rows:
            rgb, _, _, k = tp.read_frame(r)
            try:  # the engine decodes before PAD: missing modalities -> no decision (and packet: missing_modality)
                f = decode_frame(rgb_only_replay(rgb, fill))
                d = pads["seen"].predict(*f, k)[0]
            except ValueError:
                d = nr
            res.append((f"rgb_only_{fill or 'absent'}", None, d))
        rgb_only[fill or "absent"] = summarize(res, lambda d: d["live"])[f"rgb_only_{fill or 'absent'}"]
    P["d_injection"] = dict(
        note="Injected behind the sensor (compromised capture path). Perfect synthetic live frames are accepted "
             "at ~(1 - BPCER): PAD cannot stop them; sensor-origin attestation (frames signed in the camera "
             "module) and the signed capture packet are the controls (RFP SEC-01, Design §7). RGB-only replays "
             "are rejected by FR-03 (missing modality / no valid depth), never by a relaxed RGB-only path.",
        synthetic_live=summarize([("inject_live", None, d) for _, _, d in inj], lambda d: d["live"])["inject_live"],
        rgb_only_replay=rgb_only)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "results.json").write_text(json.dumps(out, indent=1))
    (a.out / "summary.md").write_text(summary_md(out))
    print((a.out / "summary.md").read_text())


def pct(r):
    if r is None or r.get("value") is None:
        return "n/a"
    lo, hi = r["ci95"]
    return f"{100 * r['value']:.2f}% [{100 * lo:.2f}, {100 * hi:.2f}] ({r['k']}/{r['n']})"


def summary_md(o):
    L = ["# E7 PAD (ISO/IEC 30107-3 metrics) — SYNTHETIC DATA ONLY", "",
         f"> {o['caveat']}", "",
         f"Decision: {o['decision']}. CI: {o['ci']}. Threshold fixed on the validation split (BPCER target 1%).", "",
         "Models: " + "; ".join(f"`{k}` {m['version']} thr={m['threshold']:.4f}" for k, m in o["models"].items()) + ".",
         f"Physics checks (priors, MODEL.json): {o['models']['seen']['checks']}.", ""]
    for name, p in o["protocols"].items():
        if name == "d_injection":
            continue
        a = p["pad"]
        L += [f"## {name} (model `{p['model']}`, profile `{p['profile']}`)", "",
              "| presentation | n | non-response | APCER / BPCER | attack success |", "|---|---|---|---|---|",
              f"| bona fide | {a['bona_fide']['n']} | {pct(a['bona_fide']['bpnrr'])} | BPCER {pct(a['bona_fide']['bpcer'])} | |"]
        for sp in tp.SPECIES:
            if sp in a:
                L.append(f"| {sp} | {a[sp]['n']} | {pct(a[sp]['apnrr'])} | APCER {pct(a[sp]['apcer'])} | {pct(a[sp]['attack_success'])} |")
        L += ["", f"APCER_max {100 * a.get('apcer_max', float('nan')):.2f}% ({a.get('worst_species')}), "
                  f"ACER {100 * a.get('acer', float('nan')):.2f}%, overall attack success {pct(a.get('attack_success_overall'))}. "
                  f"Ablation APCER_max / BPCER: CNN only {100 * p['cnn_only'].get('apcer_max', float('nan')):.2f}% / "
                  f"{pct(p['cnn_only']['bona_fide']['bpcer'])}; checks only {100 * p['checks_only'].get('apcer_max', float('nan')):.2f}% / "
                  f"{pct(p['checks_only']['bona_fide']['bpcer'])}.", ""]
    d = o["protocols"]["d_injection"]
    L += ["## d_injection (capture-path attacks, model `seen`)", "", d["note"], "",
          f"- perfect synthetic live frames: attack success {pct(d['synthetic_live']['attack_success'])}",
          *[f"- RGB-only replay (depth/IR {k}): non-response {pct(v['apnrr'])}, attack success {pct(v['attack_success'])}"
            for k, v in d["rgb_only_replay"].items()], "",
          "## Limitations of the simulation", "",
          "- Live depth = MediaPipe canonical face mesh personalised to the 5 keypoints (+ random shape variation), "
          "skull/neck/torso primitives; no 3D hair, glasses or expression dynamics. IR albedo is derived from the photo.",
          "- PAI renderers encode assumptions (dye inks partly NIR-transparent, displays dark at 850 nm with specular "
          "glass, masks with uniform NIR albedo) that a CNN can learn exactly; real PAIs, materials and sensors vary "
          "far more. Unseen-species / unseen-sensor rows show how brittle such learned cues are, not real rates.",
          "- Keypoints are the simulator's ground truth (no detector in the loop); the engine uses SCRFD keypoints.",
          "- Per-frame rates; the auth policy requires every frame of a packet to be live, so session-level BPCER "
          "is higher unless PAD scores are aggregated over the sequence.", "",
          "## Comparison with published numbers", "",
          "None applicable: there is no public benchmark for this simulator, and numbers on real RGB-D-IR PAD sets "
          "(CeFA, WMCA, CASIA-SURF; EULA-gated, not obtained) are not comparable. The real-sensor validation "
          "and an independent ISO/IEC 30107-3 test (RFP §4) remain open.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
