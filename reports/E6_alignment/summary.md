# E6 — 정렬 ablation: SCRFD 5점 vs FaRL 유도 5점 (1:N / 1:1)

생성 2026-10-11T20:18:12+00:00 · `python -m eval.onen.run e6-*`. E2 프로토콜 `celeba_v1`의 CelebA 부분집합 (seed 1): 등록 400명 (템플릿 1,600), mated 7,265회, 미등록 400명 non-mated 7,226회. 갤러리 = 부분집합 등록자만. 정렬만 다르고 검출·이미지·프로토콜은 동일.

- scrfd: SCRFD-34G 5 kps (ORT CUDA EP fp32) -> norm_crop 112 (deployed engine path)
- farl: FaRL ViT-B/16 448 WFLW-98 from the same SCRFD kps (refine=1 as the terminal; bf16) -> kps5_from_wflw98 -> norm_crop 112

FaRL 정렬 실패: 0 / 16091

| 모델 | 정렬 | FNIR@FPIR 1e-2 | 1e-3 | 1e-4 | rank-1 | TAR@FAR 1e-4 | 1e-5 | 1e-6 |
|---|---|---|---|---|---|---|---|---|
| kprpe_vitb_wf12m | scrfd | 2.26 [1.93, 2.63] | 2.42 [2.08, 2.80] | 2.70 [2.34, 3.10] | 98.03 | 96.01 | 95.48 | 94.37 |
| kprpe_vitb_wf12m | farl | 2.38 [2.04, 2.76] | 2.53 [2.18, 2.92] | 2.95 [2.57, 3.36] | 97.92 | 95.91 | 95.37 | 94.46 |
| insightface_r50_wf600k | scrfd | 2.24 [1.92, 2.61] | 2.48 [2.13, 2.86] | 2.74 [2.38, 3.14] | 97.94 | 95.76 | 94.95 | 93.39 |
| insightface_r50_wf600k | farl | 2.35 [2.02, 2.73] | 2.66 [2.30, 3.05] | 3.10 [2.71, 3.52] | 97.89 | 95.70 | 94.78 | 93.17 |

같은 mated probe 쌍비교 (각 정렬의 자체 임계값, margin 0): SCRFD만 성공 / FaRL만 성공 (exact McNemar p)

- kprpe_vitb_wf12m: FPIR 1e-02: 10 / 1 (p=0.012); FPIR 1e-03: 10 / 2 (p=0.039); FPIR 1e-04: 20 / 2 (p=0.00012)
- insightface_r50_wf600k: FPIR 1e-02: 10 / 2 (p=0.039); FPIR 1e-03: 13 / 0 (p=0.00024); FPIR 1e-04: 29 / 3 (p=2.6e-06)

non-mated 7,226회 → FPIR 1e-4 = 허용 FP 0건 (해상도 밖, 비교는 1e-2·1e-3과 쌍비교 위주).

**결론**: 두 모델 모두 FaRL 유도 5점 정렬의 FNIR이 SCRFD 5점보다 나쁘고 (같은 probe 쌍비교에서 SCRFD만 성공한 probe가 유의하게 많음), 1:1 TAR 차이는 0.2%p 내외다. 인식 모델이 RetinaFace/SCRFD 계열 5점으로 정렬된 데이터로 학습됐고 FaRL 유도 입꼬리 위치가 SCRFD와 체계적으로 다르기 때문으로 보인다 (E5 `scrfd_consistency`: 입꼬리 평균 편차 0.012–0.024 IOD). 인식 정렬은 SCRFD kps를 유지하고 FaRL은 랜드마크·품질 용도로만 쓴다 (설계 §4 재검증).

