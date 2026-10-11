# E7 PAD (ISO/IEC 30107-3 metrics) — SYNTHETIC DATA ONLY

> All presentations are SIMULATED (faceauth/sim, no real sensor, no real PAI). These numbers validate the pipeline and the metric computation only; they are NOT real-world PAD performance. Real-sensor training and independent ISO/IEC 30107-3 testing are required (hardware phase).

Decision: live = CNN live probability >= threshold AND relief/IR/shading checks pass; non-response if the face-core valid depth ratio < min_valid_ratio. CI: 95% Clopper-Pearson. Threshold fixed on the validation split (BPCER target 1%).

Models: `seen` pad_rgbdir@2 thr=0.9995; `unseen_species` pad_rgbdir_unseen_species@2 thr=0.9999.
Physics checks (priors, MODEL.json): {'min_valid_ratio': 0.5, 'min_relief_mm': 3.0, 'min_ir_median': 30.0, 'min_shading_corr': 0.1}.

## a_seen (model `seen`, profile `gemini336-sim-v1`)

| presentation | n | non-response | APCER / BPCER | attack success |
|---|---|---|---|---|
| bona fide | 3743 | 0.00% [0.00, 0.10] (0/3743) | BPCER 2.16% [1.72, 2.68] (81/3743) | |
| print | 3743 | 0.00% [0.00, 0.10] (0/3743) | APCER 0.00% [0.00, 0.10] (0/3743) | 0.00% [0.00, 0.10] (0/3743) |
| replay | 3743 | 49.16% [47.54, 50.77] (1840/3743) | APCER 0.00% [0.00, 0.19] (0/1903) | 0.00% [0.00, 0.10] (0/3743) |
| mask_rigid | 3743 | 0.00% [0.00, 0.10] (0/3743) | APCER 0.00% [0.00, 0.10] (0/3743) | 0.00% [0.00, 0.10] (0/3743) |
| mask_silicone | 3743 | 0.00% [0.00, 0.10] (0/3743) | APCER 0.00% [0.00, 0.10] (0/3743) | 0.00% [0.00, 0.10] (0/3743) |

APCER_max 0.00% (mask_rigid), ACER 1.08%, overall attack success 0.00% [0.00, 0.02] (0/14972). Ablation APCER_max / BPCER: CNN only 0.00% / 2.16% [1.72, 2.68] (81/3743); checks only 100.00% / 0.00% [0.00, 0.10] (0/3743).

## a2_pose_diverse (model `seen`, profile `gemini336-sim-v1`)

| presentation | n | non-response | APCER / BPCER | attack success |
|---|---|---|---|---|
| bona fide | 1743 | 0.00% [0.00, 0.21] (0/1743) | BPCER 4.25% [3.35, 5.30] (74/1743) | |
| print | 1743 | 0.00% [0.00, 0.21] (0/1743) | APCER 0.00% [0.00, 0.21] (0/1743) | 0.00% [0.00, 0.21] (0/1743) |
| replay | 1743 | 49.00% [46.62, 51.37] (854/1743) | APCER 0.00% [0.00, 0.41] (0/889) | 0.00% [0.00, 0.21] (0/1743) |
| mask_rigid | 1743 | 0.00% [0.00, 0.21] (0/1743) | APCER 0.00% [0.00, 0.21] (0/1743) | 0.00% [0.00, 0.21] (0/1743) |
| mask_silicone | 1743 | 0.00% [0.00, 0.21] (0/1743) | APCER 0.00% [0.00, 0.21] (0/1743) | 0.00% [0.00, 0.21] (0/1743) |

APCER_max 0.00% (mask_rigid), ACER 2.12%, overall attack success 0.00% [0.00, 0.05] (0/6972). Ablation APCER_max / BPCER: CNN only 0.00% / 4.25% [3.35, 5.30] (74/1743); checks only 100.00% / 0.00% [0.00, 0.21] (0/1743).

## b_unseen_species (model `unseen_species (trained on live+print+replay only)`, profile `gemini336-sim-v1`)

| presentation | n | non-response | APCER / BPCER | attack success |
|---|---|---|---|---|
| bona fide | 3743 | 0.00% [0.00, 0.10] (0/3743) | BPCER 2.48% [2.01, 3.04] (93/3743) | |
| print | 3743 | 0.00% [0.00, 0.10] (0/3743) | APCER 0.00% [0.00, 0.10] (0/3743) | 0.00% [0.00, 0.10] (0/3743) |
| replay | 3743 | 49.16% [47.54, 50.77] (1840/3743) | APCER 0.00% [0.00, 0.19] (0/1903) | 0.00% [0.00, 0.10] (0/3743) |
| mask_rigid | 3743 | 0.00% [0.00, 0.10] (0/3743) | APCER 73.79% [72.35, 75.19] (2762/3743) | 73.79% [72.35, 75.19] (2762/3743) |
| mask_silicone | 3743 | 0.00% [0.00, 0.10] (0/3743) | APCER 89.29% [88.25, 90.26] (3342/3743) | 89.29% [88.25, 90.26] (3342/3743) |

APCER_max 89.29% (mask_silicone), ACER 45.89%, overall attack success 40.77% [39.98, 41.56] (6104/14972). Ablation APCER_max / BPCER: CNN only 89.29% / 2.48% [2.01, 3.04] (93/3743); checks only 100.00% / 0.00% [0.00, 0.10] (0/3743).

## c_unseen_sensor (model `seen`, profile `sensor_b`)

| presentation | n | non-response | APCER / BPCER | attack success |
|---|---|---|---|---|
| bona fide | 2000 | 0.00% [0.00, 0.18] (0/2000) | BPCER 2.00% [1.43, 2.71] (40/2000) | |
| print | 2000 | 0.00% [0.00, 0.18] (0/2000) | APCER 0.00% [0.00, 0.18] (0/2000) | 0.00% [0.00, 0.18] (0/2000) |
| replay | 2000 | 50.90% [48.68, 53.11] (1018/2000) | APCER 0.00% [0.00, 0.37] (0/982) | 0.00% [0.00, 0.18] (0/2000) |
| mask_rigid | 2000 | 0.00% [0.00, 0.18] (0/2000) | APCER 0.00% [0.00, 0.18] (0/2000) | 0.00% [0.00, 0.18] (0/2000) |
| mask_silicone | 2000 | 0.00% [0.00, 0.18] (0/2000) | APCER 0.00% [0.00, 0.18] (0/2000) | 0.00% [0.00, 0.18] (0/2000) |

APCER_max 0.00% (mask_rigid), ACER 1.00%, overall attack success 0.00% [0.00, 0.05] (0/8000). Ablation APCER_max / BPCER: CNN only 0.00% / 1.95% [1.39, 2.66] (39/2000); checks only 100.00% / 0.05% [0.00, 0.28] (1/2000).

## d_injection (capture-path attacks, model `seen`)

Injected behind the sensor (compromised capture path). Perfect synthetic live frames are accepted at ~(1 - BPCER): PAD cannot stop them; sensor-origin attestation (frames signed in the camera module) and the signed capture packet are the controls (RFP SEC-01, Design §7). RGB-only replays are rejected by FR-03 (missing modality / no valid depth), never by a relaxed RGB-only path.

- perfect synthetic live frames: attack success 97.60% [97.05, 98.06] (3653/3743)
- RGB-only replay (depth/IR absent): non-response 100.00% [99.90, 100.00] (3743/3743), attack success 0.00% [0.00, 0.10] (0/3743)
- RGB-only replay (depth/IR zero): non-response 100.00% [99.90, 100.00] (3743/3743), attack success 0.00% [0.00, 0.10] (0/3743)
- RGB-only replay (depth/IR luma): non-response 100.00% [99.90, 100.00] (3743/3743), attack success 0.00% [0.00, 0.10] (0/3743)

## Limitations of the simulation

- Live depth = MediaPipe canonical face mesh personalised to the 5 keypoints (+ random shape variation), skull/neck/torso primitives; no 3D hair, glasses or expression dynamics. IR albedo is derived from the photo.
- PAI renderers encode assumptions (dye inks partly NIR-transparent, displays dark at 850 nm with specular glass, masks with uniform NIR albedo) that a CNN can learn exactly; real PAIs, materials and sensors vary far more. Unseen-species / unseen-sensor rows show how brittle such learned cues are, not real rates.
- Keypoints are the simulator's ground truth (no detector in the loop); the engine uses SCRFD keypoints.
- Per-frame rates; the auth policy requires every frame of a packet to be live, so session-level BPCER is higher unless PAD scores are aggregated over the sequence.

## Comparison with published numbers

None applicable: there is no public benchmark for this simulator, and numbers on real RGB-D-IR PAD sets (CeFA, WMCA, CASIA-SURF; EULA-gated, not obtained) are not comparable. The real-sensor validation and an independent ISO/IEC 30107-3 test (RFP §4) remain open.
