# E1 — 1:1 검증 정확도 (LFW, CFP-FP, AgeDB-30, CALFW, CPLFW)

생성 2026-10-11T15:07:22+00:00, `eval/eval_verification.py`. GPU NVIDIA RTX A6000 (다른 작업과 공유 — 속도는 참고치). 프로토콜: InsightFace 10-fold 정확도 (사전 정렬 112x112 쌍, 폴드별 학습 9폴드에서 임계값 선택), *flip* = 원본 + 좌우반전 임베딩 합 (모델 카드가 공개하는 방식). 표의 값은 fp32.

데이터: LFW 6000쌍 (동일인 3000), CFP-FP 7000쌍 (동일인 3500), AgeDB-30 6000쌍 (동일인 3000), CALFW 6000쌍 (동일인 3000), CPLFW 6000쌍 (동일인 3000)

## 10-fold 정확도 (%, flip, fp32) — 측정값 (공개값, Δ)

| 모델 | 구조 / 학습 데이터 | LFW | CFP-FP | AgeDB-30 | CALFW | CPLFW | 평균 |
|---|---|---|---|---|---|---|---|
| kprpe_vitb_wf12m | ViT-B KP-RPE, AdaFace, WebFace12M | 99.82 (99.82, -0.00) | 99.30 (99.30, +0.00) | 98.10 (98.10, +0.00) | 95.93 (95.93, +0.00) | 95.65 (95.65, -0.00) | 97.76 |
| insightface_vit_l_wf42m | ViT-L, CosFace, WebFace42M | — | — | — | — | — | 가중치 없음 |
| insightface_r50_wf600k | IResNet-50, WebFace600K (buffalo_l) | 99.82 (99.83, -0.01) | 99.26 (99.33, -0.07) | 98.12 (98.23, -0.11) | 95.95 | 94.48 | 97.52 |
| insightface_r100_glint360k | IResNet-100, Glint360K (antelopev2) | 99.83 (99.78, +0.05) | 99.17 (99.10, +0.07) | 98.28 (98.65, -0.37) | 96.10 | 94.67 | 97.61 |
| adaface_ir101_wf12m | IResNet-101, AdaFace, WebFace12M | 99.82 (99.82, -0.00) | 99.24 (99.24, +0.00) | 98.00 (98.00, +0.00) | 96.12 (96.12, -0.00) | 94.57 (94.57, -0.00) | 97.55 |

공개값 출처: kprpe_vitb_wf12m: CVLFace README 'Face Recognition Performance Board', row ViT KPRPE / AdaFace / WebFace12M (github.com/mk-minchul/CVLface, cvlface/pretrained_models/README.md); insightface_r50_wf600k: insightface model_zoo/README.md 'Recognition accuracy of python library model packs', row buffalo_l (recognition model w600k_r50.onnx); CALFW/CPLFW not published; insightface_r100_glint360k: nothing published for antelopev2 glintr100.onnx. 'reference' = arcface_torch Glint360k r100 re-training (2022, glint360k_r100/training.log, step 332000 flip accuracy): not proven to be the same checkpoint; adaface_ir101_wf12m: CVLFace README 'Face Recognition Performance Board', row IR101 / AdaFace / WebFace12M (github.com/mk-minchul/CVLface, cvlface/pretrained_models/README.md)

## noflip 정확도 (%, fp32) — 실제 엔진 경로 (단일 임베딩, 반전 없음)

| 모델 | LFW | CFP-FP | AgeDB-30 | CALFW | CPLFW | 평균 |
|---|---|---|---|---|---|---|
| kprpe_vitb_wf12m | 99.82 | 99.27 | 98.03 | 96.00 | 95.67 | 97.76 |
| insightface_r50_wf600k | 99.80 | 99.24 | 98.03 | 96.05 | 94.43 | 97.51 |
| insightface_r100_glint360k | 99.83 | 99.14 | 98.42 | 96.18 | 94.57 | 97.63 |
| adaface_ir101_wf12m | 99.82 | 99.31 | 97.98 | 96.08 | 94.63 | 97.57 |

## TAR @ FAR=1e-3 (%, flip, fp32; 세트별 전체 쌍, Wilson 95% CI)

세트당 타인 쌍이 3000–3500개뿐이라 임계값이 타인 쌍 ~3개로 정해진다 (거친 값).

| 모델 | LFW | CFP-FP | AgeDB-30 | CALFW | CPLFW |
|---|---|---|---|---|---|
| kprpe_vitb_wf12m | 99.70 [99.43, 99.84] | 98.71 [98.28, 99.04] | 95.47 [94.66, 96.15] | 92.20 [91.19, 93.11] | 90.67 [89.57, 91.66] |
| insightface_r50_wf600k | 99.73 [99.47, 99.86] | 98.69 [98.25, 99.01] | 96.03 [95.27, 96.67] | 92.27 [91.26, 93.17] | 87.43 [86.20, 88.57] |
| insightface_r100_glint360k | 99.73 [99.47, 99.86] | 98.43 [97.96, 98.79] | 96.73 [96.04, 97.31] | 92.37 [91.36, 93.26] | 88.30 [87.10, 89.40] |
| adaface_ir101_wf12m | 99.73 [99.47, 99.86] | 98.69 [98.25, 99.01] | 96.07 [95.31, 96.71] | 92.03 [91.01, 92.95] | 88.73 [87.55, 89.82] |

## fp16 / 속도 (embed_aligned 전체 경로, batch 128, 공유 GPU)

| 모델 | 런타임 | fp32 img/s (중앙값 / 최선) | fp16 img/s (중앙값 / 최선) | fp16 vs fp32 | MODEL.json 정밀도 |
|---|---|---|---|---|---|
| kprpe_vitb_wf12m | PyTorch | 37 / 38 | 74 / 80 | min cos 0.99983, max |Δacc| 0.05pp | fp16 |
| insightface_r50_wf600k | ONNX Runtime CUDA EP (fp32, TF32 conv) | 266 / 318 | — | — | fp32 |
| insightface_r100_glint360k | ONNX Runtime CUDA EP (fp32, TF32 conv) | 111 / 112 | — | — | fp32 |
| adaface_ir101_wf12m | PyTorch | 71 / 73 | 150 / 157 | min cos 0.99993, max |Δacc| 0.03pp | fp32 |
- adaface_ir101_wf12m: 이 측정 후 위 규칙에 따라 MODEL.json 정밀도를 fp16로 변경 (E2는 변경 후 정밀도).
- ONNX 모델의 공개값 대비 작은 차이는 TF32 때문이 아니다: CUDA EP use_tf32=1/0의 flip 정확도가 소수 셋째 자리까지 같음 (insightface_r50_wf600k tf32=1 cfp_fp 99.257, insightface_r50_wf600k tf32=1 agedb_30 98.117, insightface_r50_wf600k tf32=0 cfp_fp 99.257, insightface_r50_wf600k tf32=0 agedb_30 98.117; tf32_check.json).

- 정밀도 규칙: max |Δacc| <= 0.1pp 이고 min cos >= 0.999 이면 fp16 autocast 허용 (E2는 MODEL.json 정밀도로 측정).
- GPU 부하 (측정 전/후, 모든 사용자): kprpe_vitb_wf12m: 100 %, 42651 MiB / 100 %, 44760 MiB; insightface_r50_wf600k: 100 %, 34142 MiB / 100 %, 34144 MiB; insightface_r100_glint360k: 100 %, 39327 MiB / 100 %, 39327 MiB; adaface_ir101_wf12m: 100 %, 45234 MiB / 100 %, 45238 MiB
- insightface_vit_l_wf42m: **평가 불가** — weight file(s) not present: ['model.pt']. RFP 1순위 후보 InsightFace ViT-L/WebFace42M은 공개 체크포인트가 없다 (insightface issue #1961). 파일 확보 시 `python -m faceauth.models.recog data/models/insightface_vit_l_wf42m --write-sha256` 후 재실행.
