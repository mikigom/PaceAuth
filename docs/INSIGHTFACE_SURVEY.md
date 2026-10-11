# deepinsight/insightface 조사 결과 (2026-10-08, HEAD 81929474)

조사 방법: 4개 관점(인식·검색 / 라이브니스·보안 / 서버·SDK·독립시험 / 검출·3D·데이터셋)으로 레포를 로컬에서 읽고,
관점마다 별도 검증자가 링크·해시·코드를 재확인했다 (검출·3D·데이터셋 관점은 검증 단계가 중단되어 **미검증**).
방침: 내부 연구용이므로 라이선스는 채택 기준이 아니며, 권리 상태는 RFP §2 증빙용으로만 기록한다.

## 결론

- **WF42M ViT-L보다 강한 공개 InsightFace 인식 모델은 없다.** 2.0/2.1의 새 팩(raccoon, cheetah)은 기존
  w600k_r50 / w600k_mbf를 바이트 단위로 그대로 재사용한다. 공개 수치상 KP-RPE ViT-B(IJB-C@1e-4 97.82)가
  buffalo_l w600k_r50(97.25)보다 높다.
- 그래도 **비교 모델로는 가치가 크다**: buffalo_l w600k_r50(IResNet-50, WebFace600K = WebFace12M과 같은 61.8만 ID)
  은 학습 데이터가 KP-RPE와 같아 구조 차이만 비교되고, antelopev2 glintr100(IResNet-100, Glint360K)은 다른 데이터 계열이다.
- **엔진에 바로 반영할 안전장치 2개**: ONNX Runtime이 CUDA 대신 CPU로 조용히 넘어가는 경우를 막는 검사
  (검증자가 우리 SCRFD 세션 3번 중 1번에서 실제로 CPU 폴백을 관찰), ORT 텔레메트리 비활성화.
- 3채널 PAD를 대체할 것은 없다. InsightFace의 유일한 라이브니스 모델은 RGB 단일 프레임 MiniFASNetV2다.
  대신 WFAS 논문 수치(미지의 3D 공격에서 RGB 전용 PAD의 APCER 45~59%)가 FR-03(RGB-only 금지) 설계의 외부 근거가 된다.

## 바로 쓸 것

| 항목 | 위치 | 용도 (RFP) | 다음 작업 |
|---|---|---|---|
| buffalo_l `w600k_r50.onnx` (IResNet-50 @ WebFace600K, 512-d, ArcFace 정렬) sha256 `4c06341c…` | `releases/download/model-zoo/buffalo_l.zip` (288.6 MB) | E1/E2/E6 비교 모델 (RFP §2 비교 요구) | ORT 래퍼 추가, E1·E2 재측정 |
| antelopev2 `glintr100.onnx` (IResNet-100 @ Glint360K) sha256 `4ab1d643…` | `releases/download/model-zoo/antelopev2.zip` (360.7 MB, `antelopev2/` 하위 폴더) | E1/E2 두 번째 비교 모델 (다른 학습 데이터) | 동일 래퍼로 추가 |
| ORT CUDA 강제·감사 (Server `onnx_engine.py` `audit_cuda_profile`) | `server/backend/insightface_server/inference/onnx_engine.py` | 엔진 fail-closed, E9 지연 | 세션 생성 후 `get_providers()` 확인, CPU 폴백이면 기동 거부 (TensorRT EP 허용, CUDA 13 so 이름 주의) |
| `ORT_DISABLE_TELEMETRY=1` | Server 설정 | OPS-01 (생체 노드의 외부 송신 차단) | 이미지 ENV 1줄 |
| 컨테이너 강화: no-new-privileges, cap_drop ALL, pids_limit, 베이스 이미지 digest 고정 | `server/deploy` | OPS-01, RFP §6 SBOM | compose/Dockerfile 반영 |

## 평가해 볼 것

| 항목 | 메모 |
|---|---|
| MiniFASNetV2 RGB 라이브니스 (= InsightFace `liveness.onnx`; 원 가중치 MiniVision Apache-2.0) | E7의 "RGB 전용 기준선" 한 줄. 3채널 PAD 대체 불가 |
| WFAS(CVPR23) 프로토콜 2 (2D 공격으로 학습 → 미지의 3D 공격 시험) | E7 "미학습 공격 유형" 프로토콜 설계 참고. 데이터는 신청제·RGB 전용 |
| `inswapper_128.onnx` (554 MB) | E11 딥페이크 주입 공격 데이터 생성기 (RGB 전용; 주입 경로 방어 시험용) |
| `1k3d68.onnx` (buffalo 팩) | E4 헤드포즈 세 번째 의견. 부호 규약 매핑 필요 |
| InsightFace Server 검색: 정확 전수 탐색 + INT8(×736) 저장, ANN 없음 | 우리 FAISS FP32와 같은 계열. 메모리가 부족해지면 FP16/INT8 검토 (E2 재시험 필요) |
| InsightFace 비공개 모델 평가 (dingo / eagle, MOU 후 임베딩 API, 2주) | 공개되지 않은 ViT-L의 현실적 대안. 얼굴 크롭 외부 전송은 OPS-01과 충돌하므로 온프레미스 평가 문의 필요 |
| NIST FRTE 1:N | 알고리즘 독립 시험 경로. 2026-10-30~11-30 제출 중단. RFP 인수 시험(약 300만 비동일인 검색)을 대체하지는 못함 |
| Glint360K (토렌트/Baidu) | E2 갤러리 확장용 실제 인물 방해 데이터 (36만 ID) |

## 쓰지 않을 것

- raccoon / cheetah 팩: 인식 모델이 기존과 동일하다. 사람 검출·ReID는 키오스크에 무관하다.
- InspireFace C++ SDK: 우리 모델(SCRFD-34G/6DRepNet360/FaRL)을 대체하는 고정 팩이고 IR PAD 모델이 없다. 단말은 ORT C++ + OpenCV 쪽이 단순하다.
- InsightFace Server 저장 구조(SQLite 단일 노드): RPO=0, 키 분리, 감사 요구와 충돌한다. 단, 생체 데이터를 로그에 남기지 않는 규칙은 참고한다.
- IJB-B/C, MegaFace 미러: NIST가 배포를 중단했다(2023-03-14). 합법적 경로가 아니다.
- `retinaface_anticov`: 이름과 달리 마스크 착용 검출기이고 위조 방어가 아니다.
- genderage: 데이터 최소화(OPS-01)에 반한다.
- MFR-Ongoing / IFRT 평가 서버: 현재 오프라인이다.
- 검출: 공식 SCRFD-34G에는 키포인트가 없다. 현재 쓰는 ModelScope SCRFD_34G_GNKPS_v2가 최선이므로 유지한다 (미검증 관점).

## 권리 상태 (RFP §2 증빙용 기록)

| 자산 | 조건 |
|---|---|
| InsightFace 코드 | MIT (README 기재) |
| InsightFace 배포 모델 전부 (buffalo, antelopev2, liveness, inswapper …) | "non-commercial research purposes only". 상용: 인식 팩은 recognition-oss-pack@insightface.ai, inswapper·InspireFace·비공개 모델은 contact@insightface.ai |
| KP-RPE ViT-B (CVLFace) | 학습 데이터(WebFace12M, 연구용)의 라이선스를 따른다 |
| SCRFD-34G GNKPS v2 (ModelScope) | 모델 카드는 MIT라고 표기하지만 학습 데이터 WIDER FACE가 CC BY-NC-ND이므로 법무 검토 필요 |
| MiniFASNet 원 가중치 | Apache-2.0 (학습 데이터 비공개) |
