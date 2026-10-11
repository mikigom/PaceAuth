# E1 — Recognition 1:1 verification (LFW, CFP-FP, AgeDB-30, CALFW, CPLFW)

Generated 2026-10-08T09:29:12+00:00 by `eval/eval_verification.py`. GPU: NVIDIA RTX A6000 (shared with other jobs; timings indicative). Protocol: InsightFace 10-fold accuracy on pre-aligned 112x112 pairs; *flip* = embedding of image + horizontal flip (the number model cards publish).

Datasets: lfw 6000 pairs (3000 genuine), cfp_fp 7000 pairs (3500 genuine), agedb_30 6000 pairs (3000 genuine), calfw 6000 pairs (3000 genuine), cplfw 6000 pairs (3000 genuine)

## 10-fold accuracy (%, mean ± std over folds)

| model | precision | mode | lfw | cfp_fp | agedb_30 | calfw | cplfw | mean |
|---|---|---|---|---|---|---|---|---|
| kprpe_vitb_wf12m | fp32 | flip | 99.82 ± 0.24 | 99.30 ± 0.33 | 98.10 ± 0.73 | 95.93 ± 1.13 | 95.65 ± 0.83 | 97.76 |
| kprpe_vitb_wf12m | fp32 | noflip | 99.82 ± 0.24 | 99.27 ± 0.38 | 98.03 ± 0.77 | 96.00 ± 1.12 | 95.67 ± 0.78 | 97.76 |
| kprpe_vitb_wf12m | fp16 | flip | 99.82 ± 0.24 | 99.30 ± 0.33 | 98.07 ± 0.73 | 95.93 ± 1.13 | 95.65 ± 0.83 | 97.75 |
| kprpe_vitb_wf12m | fp16 | noflip | 99.82 ± 0.24 | 99.24 ± 0.35 | 98.03 ± 0.77 | 96.02 ± 1.14 | 95.62 ± 0.81 | 97.75 |
| kprpe_vitb_wf12m | published | flip | 99.82 | 99.30 | 98.10 | 95.93 | 95.65 | 97.76 |
| kprpe_vitb_wf12m | Δ fp32 − published | flip | -0.00 | +0.00 | +0.00 | +0.00 | -0.00 | |
| insightface_vit_l_wf42m | — | — | pending | pending | pending | pending | pending | — |

## TAR @ FAR=1e-3 (%, flip, fp32; pooled over all pairs, Wilson 95% CI)

Only 3000–3500 impostor pairs per set: the threshold is set by ~3 impostor pairs, so these values are coarse.

| model | lfw | cfp_fp | agedb_30 | calfw | cplfw |
|---|---|---|---|---|---|
| kprpe_vitb_wf12m | 99.70 [99.43, 99.84] | 98.71 [98.28, 99.04] | 95.47 [94.66, 96.15] | 92.20 [91.19, 93.11] | 90.67 [89.57, 91.66] |

## Speed (embed_aligned end-to-end, one GPU process) and fp16 agreement

GPU 5 is shared with other contributors' jobs: medians include their load (snapshot below); min = least-disturbed iteration, closer to an idle GPU.

| model | precision | batch | ms median | ms p95 | ms min | img/s (median) | img/s (min) | peak GPU MB |
|---|---|---|---|---|---|---|---|---|
| kprpe_vitb_wf12m | fp32 | 1 | 23.2 | 31.3 | 17.5 | 43 | 57 | 473 |
| kprpe_vitb_wf12m | fp16 | 1 | 26.4 | 30.7 | 20.2 | 38 | 50 | 555 |
| kprpe_vitb_wf12m | fp32 | 32 | 212.7 | 304.2 | 98.1 | 150 | 326 | 1191 |
| kprpe_vitb_wf12m | fp16 | 32 | 109.1 | 155.2 | 51.5 | 293 | 621 | 886 |
| kprpe_vitb_wf12m | fp32 | 128 | 821.4 | 1157.9 | 414.3 | 156 | 309 | 3417 |
| kprpe_vitb_wf12m | fp16 | 128 | 390.1 | 569.9 | 175.3 | 328 | 730 | 2191 |

- kprpe_vitb_wf12m: cos(fp16, fp32) over 62000 images: mean 0.999998, 0.1th pct 0.999987, min 0.999829; max |Δacc| fp16 vs fp32 = 0.05 pp; fp16 speed-up x2.11 at batch 128, x0.88 at batch 1 -> recommended precision: **fp16 autocast** (rule: max |Δacc| <= 0.1 pp and min cos >= 0.999). GPU load before/after timing: 100 %, 19078 MiB / 100 %, 19078 MiB (util %, MiB, all users).
- insightface_vit_l_wf42m: **pending** — weight file(s) not present: ['model.pt']. When present: `python -m faceauth.models.recog data/models/insightface_vit_l_wf42m --write-sha256` (strict load + hash into MODEL.json), then re-run this script.
