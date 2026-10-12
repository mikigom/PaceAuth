# faceauth — 1:N 얼굴 인증 시스템 시제품 (RGB · Depth · IR)

`docs/RFP_KO.md`(제안요청서)와 `docs/Design_KO.md`(시스템 디자인)를 따라 만든 시제품입니다. 하드웨어(Gemini 336 센서, 단말)는
이번 단계에서 제외했고, 센서는 시뮬레이터(`faceauth/sim`)로 대체했습니다. 평가 결과는 `reports/EVALUATION_REPORT.md`에 있습니다.

## 구성

```
terminal (sim)  --mTLS-->  gateway (nginx, TLS1.3, client cert)  -->  api-1 / api-2 (FastAPI, policy, audit)
                                                                         |-> engine-1 / engine-2 (GPU: SCRFD-34G TRT, 6DRepNet360, PAD v2, AdaFace IR101)
                                                                         |-> search-1 / search-2 (GPU: FAISS exact IP, 100k x 4)
                                                                         |-> kms (subject keys, crypto-shredding)
                                                                         |-> db / db-standby1 / db-standby2 (Postgres, ANY 1 sync)
```

| 경로 | 내용 |
|---|---|
| `docs/IMPLEMENTATION.md` | 구성요소 간 계약(인터페이스, 데이터, 환경변수) |
| `docs/API.md` | API 설계서 (한국어) |
| `docs/OPERATIONS.md` | 배포·복구 매뉴얼 (한국어) |
| `docs/INSIGHTFACE_SURVEY.md` | InsightFace 자산 조사 |
| `deploy/db/faceauth.sql` | DB 스키마 (해시 체인 감사 로그, outbox) |
| `faceauth/` | 서비스 코드 (`api`, `engine`, `search`, `kms`, `terminal`, `sim`, `models`, `common`) |
| `eval/`, `scripts/` | 평가·운영 스크립트 |
| `reports/` | 평가 결과 (E1–E11), `SBOM.json` |
| `ui/` | 키오스크 화면(`kiosk.html`, 녹화 재생)과 시연 영상 구성(`video.html`) |

## 실행

모든 실행은 Docker 안에서 하고, GPU는 5번만 씁니다(compose `device_ids: ["5"]`).

```bash
docker build -f docker/Dockerfile -t faceauth:dev .                    # 이미지 (버전 고정: requirements.txt)
docker run --rm -u $(id -u):$(id -g) -v $PWD:/app -w /app faceauth:dev python scripts/init_secrets.py
docker run --rm -u $(id -u):$(id -g) -v $PWD:/app -w /app faceauth:dev python scripts/build_registry.py --skip-invalid
docker compose up -d db db-standby1 db-standby2 kms
docker compose run --rm -u $(id -u):$(id -g) -e HOME=/tmp tester python scripts/build_registry.py --skip-invalid \
    --dsn "host=db,db-standby1,db-standby2 dbname=faceauth user=faceauth_api target_session_attrs=read-write" \
    --password-file secrets/db_api_password                             # model_versions 등록·활성화
docker compose up -d engine search-1 search-2 api-1 api-2 gateway
```

모델 가중치와 데이터셋은 `data/`에 있습니다(git 제외, 출처·해시·라이선스는 각 `MODEL.json` / `DATASET.json`).

## 시험

```bash
# 단위·통합 (서비스별)
docker compose -p fa-api -f docker-compose.yml -f deploy/compose.api.yml --profile apitest up -d
docker compose -p fa-api -f docker-compose.yml -f deploy/compose.api.yml --profile apitest run --rm -u "$(id -u):$(id -g)" \
    -e HOME=/tmp tester python -m pytest -q tests/test_api_policy.py tests/test_api_flow.py
# 실제 스택 E2E 보안 시험 (E11)
docker compose run --rm -u $(id -u):$(id -g) -e HOME=/tmp -e FA_E2E=1 tester python -m pytest -q tests/test_e2e_real.py
# 시연: BIWI 실제 고개 회전 영상으로 등록(세션 05) 후 다른 날 촬영본(세션 18)으로 인증
docker compose run --rm -u $(id -u):$(id -g) -e HOME=/tmp tester python scripts/terminal_demo.py --device dev-001 \
    --register --account demo-f05 --source biwi --enroll-seq 05 --auth-seq 18 --auth-max-frames 150
```

평가 스크립트: `eval/eval_verification.py`(E1), `eval/onen/run.py`(E2), `eval/eval_widerface.py`(E3),
`eval/eval_headpose.py`(E4), `eval/eval_wflw.py`(E5), `eval/eval_pad.py`(E7), `scripts/enroll_sim.py`(E8),
`scripts/loadgen.py`(E9), `scripts/test_failover.sh`(E10), `tests/test_e2e_real.py`(E11), `scripts/sbom.sh`(SBOM).

## 시연 UI·영상

최대 등록 고객은 10만 명(×템플릿 4개)이고, 운영 임계값 `FA_T_ID=0.57`은 이 규모 기준입니다(`reports/EVALUATION_REPORT.md` §4.2).
등록 안내는 Face ID 방식(고개를 한 바퀴 돌리면 원형 눈금이 채워짐)이 기본이고, 단계별 안내는 접근성 옵션입니다.
로그인은 깊이 센서로 잰 거리가 40–80 cm에 들어오면 버튼 없이 자동으로 시작합니다.

```bash
# 실제 스택에서 시나리오 녹화 (등록 4명 → 다른 날 재방문 2명 로그인 → 미등록자 2명 거절 → 마스크 거절)
docker compose run --rm -u $(id -u):$(id -g) -e HOME=/tmp tester python scripts/demo_record.py   # -> data/demo/
# 영상 렌더링 (1920x1080, 30 fps; 헤드리스 Chromium + ffmpeg)
docker build -f docker/video.Dockerfile -t faceauth-video:dev .
docker run --rm -u $(id -u):$(id -g) -e HOME=/tmp -v $PWD:/app -w /app faceauth-video:dev python3 scripts/render_video.py
```

`ui/kiosk.html`을 브라우저로 열면 녹화된 장면을 실시간으로 반복 재생합니다. 키오스크 화면에는 단말이 실제로 받는 정보(안내 문구,
판정 결과)만 나오고, 유사도 점수와 내부 사유는 영상의 서버 측 화면(감사 로그)에만 나옵니다.

## 주의

- 내부 연구용 시제품입니다. 사용한 모델·데이터 대부분은 비상업 연구 라이선스입니다(`reports/SBOM.json`).
- PAD는 실제 RGB-D-IR 위조 데이터가 없어 시뮬레이터 데이터로만 학습·평가했습니다. 실센서 학습과 ISO/IEC 30107-3 독립 시험은 하드웨어 단계 과제입니다.
- `secrets/`의 CA·키는 개발용입니다.
