# faceauth — 1:N 얼굴 인증 시스템 시제품 (RGB · Depth · IR)

`docs/RFP_KO.md`(제안요청서)와 `docs/Design_KO.md`(시스템 디자인)를 따라 만든 시제품입니다. 기존 인증 평가는
시뮬레이터(`faceauth/sim`)로 진행했습니다. 평가 결과는 `reports/EVALUATION_REPORT.md`에 있습니다.

## Orbbec Femto Bolt 실센서 연결

라이브 데모의 등록자·미등록자 두 사람 박스 구분과 가까운 등록자 인증을 실제 사용자가 확인했습니다.
이름 입력·수정, 얼굴별 자동 인식, 가장 큰 얼굴의 인증 대상 선택을 배포했습니다
([라이브 데모 검증 기록](reports/deployment/femto_bolt_live_demo_success_20261014.json)).

Docker 웹 단말에 RGB·깊이·IR 미리보기, 로그인 없는 계정 관리·데모 화면, 명시적 동의,
중앙 서버 안내에 따른 얼굴 등록·인증과 PAD 점수 진단을 연결했습니다. 기존 `ui/kiosk.html`은 녹화 재생용입니다.
라이브 화면에는 여러 얼굴의 박스와 이름을 자동으로 표시합니다. 등록자는 초록색, 미등록자는 주황색이며,
품질이 부족하거나 일치가 불확실하면 회색 `확인 중` 박스를 표시합니다. 한 번에 최대 16명을 처리합니다.
현재 배포는 USB 3.1과 AMD 내장 GPU에서 센서 약 30.1 FPS, 브라우저 표시 30 FPS를 확인했습니다.
재부팅 후 커널 `7.0.0-38-generic`과 NVIDIA 드라이버 `580.178.04`에서 Docker CUDA·FAISS 연산을
확인했고, GPU 추론·검색 복제본 각 2개가 준비되었습니다. 현재 내부망 데모는 추가 수집 없이 개선한
**실측 미검증 실험용 PAD**를 인증에 적용합니다. 등록은 PAD 없이 대상 계정을 선택하고 동의한 뒤 진행하며,
라이브 인증은 계정을 미리 고르지 않고 등록된 전체 얼굴에서 찾는 **1:N 방식**입니다.
센서 품질·서명·계정·신원과 프레임별 깊이 형상을 확인합니다. 수정된 PAD의 실제 공격 방어율은 검증 전입니다.

2026-10-14 실제 참여자가 웹에서 9방향 얼굴 등록과 얼굴 일치 확인을 완료했습니다. 서버에서도 등록 활성화,
암호화된 특징 4개 저장, 인증 승인을 확인했습니다. 이는 1명의 동작 확인이며 인식 정확도·PAD 검증 결과는
아닙니다([실제 사용 검증 기록](reports/deployment/femto_bolt_pilot_success_20261014.json)).

```bash
# 기존 backend, secrets, faceauth_edge 네트워크가 준비된 배포에서 실행
docker compose -f compose.live.yml up -d --build
# 같은 내부망에서 http://<이 PC의 IP>:8080 접속 (.env의 FA_LIVE_BIND_IP/FA_LIVE_PORT)
# 추론·검색 서버 시작 및 실제 GPU 연산·서비스 준비 점검
python3 scripts/start_gpu_backend.py
```

`auth.capture_ready`는 센서·중앙 추론 준비 여부, `auth.enrollment_ready`는 등록 준비 여부, `auth.ready`는 인증 준비 여부입니다.
현재 `.env`의 `FA_PAD_MODE=femto_geometry_experimental`과 고정 설정 SHA를 API·엔진에 공통 적용합니다.
엔진은 읽기 전용 `deploy/pad/`의 설정을 사용합니다. 코드·센서·파일 SHA가 다르면 승인하지 않습니다.
모드를 생략하면 기존 production PAD 검증 경로를 사용하며, 새 실험 모드의 설정으로 그 검증을 통과시킬 수 없습니다.
등록은 9방향의 18프레임, 인증은 정면 품질을 통과한 3프레임을 사용합니다. 등록 원은 눈금 두 칸으로 1회 촬영과 2회 완료를 구분합니다.
등록의 기본 품질·동의·동일인 검사는 유지하며, 인증 PAD 최소값은 평면 잔차 0.30mm·코 주변 깊이 차 3.0mm입니다. 테스트 계정과 기존 관측 점수는
`live-state` 볼륨에, 등록한 얼굴 특징은 암호화하여 서버에 보관합니다. 연속 영상은 저장하지 않습니다.
별도 사진 저장 동의가 있으면 등록 때의 정면 사진 1장만 최대 320px로 축소·암호화해 보관합니다.

현재 내부망 페이지는 **로그인 없는 관리자·데모 콘솔**입니다. 상단 **계정 관리**에서 여러 계정을 추가하고
이름·등록 사진·활성 상태를 관리하고 선택한 사람의 얼굴을 등록할 수 있습니다. 라이브 인증에는 계정 선택이 없습니다.
새 계정을 만들 때 비밀번호를 입력하지 않습니다. 비활성 계정은 얼굴 인식·인증에서 제외하며,
계정 삭제는 사진과 암호화된 얼굴 특징까지 폐기합니다. 기존에 사진 없이 등록한 계정은 다시 촬영해야 사진이 생깁니다.
`compose.live.yml`의 `FA_LIVE_OPEN_CONSOLE=1`이 이 동작을 명시적으로 켭니다. 페이지에 접속한 사용자가 관리 기능을
바로 사용하며, 설정을 끄면 기존 계정 로그인·관리자 권한 검사를 사용하는 모드로 돌아갑니다.
응답의 `pad_enabled=true`, `pad_mode=femto_geometry_experimental`로 구분합니다. 등록은 `enrollment_pad_enabled=false`로 구분합니다. 실측 미검증 안내는 PAD 수집 화면에 유지하며 라이브 영상에는 표시하지 않습니다.

데모 준비는 **사람마다 별도 계정·표시 이름 생성 → 본인 동의 → 한 명씩 얼굴 등록** 순서입니다. 등록을 마치고
화면을 열어 두면 여러 사람을 자동으로 인식합니다. 기존 계정도 **계정 관리**에서 표시 이름만
변경할 수 있으며 얼굴 재등록은 필요하지 않습니다. 컬러·깊이·IR 모두 거울 보기이며 박스 좌표도 함께 반전합니다.
라이브 화면은 컬러 영상으로 화면을 채우며, 작은 **인증 시작** 버튼을 누르면 최대 **10초** 동안 등록된 얼굴을 기다립니다.
가장 큰 얼굴의 등록자 인식이 600ms 동안 유지되면 새 촬영으로 `/api/auth/identify`의 전체 갤러리 검색을 실행합니다.
미등록자나 얼굴 없음은 대기를 유지하고, 서버가 승인한 경우에만 확인된 이름과 승인 모션을 보여준 뒤 대기를 종료합니다.
대기는 파란 모서리의 은은한 움직임과 **인증 대기 중** 표시로 구분합니다. 상시 결과 카드는 표시하지 않습니다.
기한 안에 승인된 경우에만 카메라 영상 위에 이름과 승인 모션을 표시합니다. 실패·거절은 표시 없이 남은 시간 안에 다시 시도합니다.
재시도로 10초 제한을 연장하지 않으며 만료되면 기본 대기로 돌아가 다음 시작을 기다립니다. 중지 버튼에 남은 초를 표시합니다.
라이브 인증에는 동의 안내·체크박스를 표시하지 않으며, 명시적인 시작 요청만 보냅니다. 등록·PAD 수집은 기존 동의를 유지합니다.
**인증 중지**는 대기와 화면 결과 표시를 끝냅니다. 진행 중 요청의 정리를 확인해 중복 촬영을 막고, 늦게 도착한 결과는 표시하지 않습니다.
중앙 인증 세션도 남은 시간에 맞춰 만료됩니다. 이미 확정된 서버 승인 기록을 취소하거나 바꾸지는 않습니다.
촬영 대상은 가장 큰 얼굴이며 품질 불합격이어도 작은 얼굴로 바꾸지 않습니다. 프레임 사이에 사람이 바뀌면
해당 시도를 승인하지 않습니다. 등록은 계속 화면에 한 명만 허용합니다.

자동 라이브 인식은 PAD 없는 이름 표시용 데모이며 승인 토큰을 발급하지 않습니다(`pad_mode=disabled_demo`).
라이브 깊이 범위는 350–3,000mm, 등록·인증은 350–1,000mm이고 다른 품질 기준도 적용합니다.
카메라의 최신 프레임을 공유하는 인식 작업자의 기본 목표는 **15 FPS**이며, 브라우저가 같은 결과를 공유합니다.
이는 기존 약 30 FPS 미리보기 측정과 별개이며, 여러 실제 참여자의 동시 인식 속도·정확도는 추가 측정 대상입니다.
등록·인증 중에는 라이브 인식을 잠시 멈추고, 오래되거나 불확실한 결과에는 이전 이름을 표시하지 않습니다.

첫 실센서 진단은 1명의 32시도·160프레임에서 PAD 통과 0프레임이었습니다
([측정 기록](reports/deployment/femto_bolt_first_live_diagnostics_20261014.json)).
센서 프로브, 제한된 키 마운트와 재부팅 후 복구는
[Femto Bolt 배포 문서](docs/FEMTO_BOLT.md), 물리 표본과 평가 gate는
[PAD 검증 문서](docs/FEMTO_PAD_VALIDATION.md)를 참고하세요.

## 구성

```
terminal (Femto web / CLI / sim) --mTLS--> gateway (nginx, TLS1.3, client cert) --> api-1 / api-2 (FastAPI, policy, audit)
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
| `ui/` | 실센서 웹 단말(`live.html`), 녹화 재생(`kiosk.html`), 시연 영상 구성(`video.html`) |

## 실행

모든 서비스는 Docker 안에서 실행합니다. `.env`의 `FA_GPU_DEVICE_ID`로 호스트 GPU를 선택합니다(기본 `0`).

```bash
docker build -f docker/Dockerfile -t faceauth:dev .                    # 이미지 (버전 고정: requirements.txt)
docker run --rm -u $(id -u):$(id -g) -v $PWD:/app -w /app faceauth:dev python scripts/init_secrets.py
docker run --rm -u $(id -u):$(id -g) -v $PWD:/app -w /app faceauth:dev python scripts/build_registry.py --skip-invalid
docker compose up -d db db-standby1 db-standby2 kms
docker compose run --rm -u $(id -u):$(id -g) -e HOME=/tmp tester python scripts/build_registry.py --skip-invalid \
    --dsn "host=db,db-standby1,db-standby2 dbname=faceauth user=faceauth_api target_session_attrs=read-write" \
    --password-file secrets/db_api_password                             # model_versions 등록·활성화
docker compose up -d engine-1 engine-2 search-1 search-2 api-1 api-2 gateway
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
# 과거 시연 경로: PAD 사용 모드에서는 현재 모델의 검증 gate가 등록·인증을 차단함
# BIWI 실제 고개 회전 영상으로 등록(세션 05) 후 다른 날 촬영본(세션 18)으로 인증
docker compose run --rm -u $(id -u):$(id -g) -e HOME=/tmp tester python scripts/terminal_demo.py --device dev-001 \
    --register --account demo-f05 --source biwi --enroll-seq 05 --auth-seq 18 --auth-max-frames 150
```

평가 스크립트: `eval/eval_verification.py`(E1), `eval/onen/run.py`(E2), `eval/eval_widerface.py`(E3),
`eval/eval_headpose.py`(E4), `eval/eval_wflw.py`(E5), `eval/eval_pad.py`(E7), `scripts/enroll_sim.py`(E8),
`scripts/loadgen.py`(E9), `scripts/test_failover.sh`(E10), `tests/test_e2e_real.py`(E11), `scripts/sbom.sh`(SBOM).

## 시연 UI·영상

최대 등록 고객은 10만 명(×템플릿 4개)이고, 운영 임계값 `FA_T_ID=0.57`은 이 규모 기준입니다(`reports/EVALUATION_REPORT.md` §4.2).
등록 안내는 Face ID 방식(고개를 한 바퀴 돌리면 원형 눈금이 채워짐)이 기본이고, 단계별 안내는 접근성 옵션입니다.
아래 녹화 시연의 로그인은 깊이 센서로 잰 거리가 40–80 cm에 들어오면 자동으로 시작합니다.
실센서 웹 단말의 라이브 박스·이름 표시는 자동으로 시작합니다. 현재 내부망 콘솔에서는 로그인 없이 사람별 얼굴을
등록하고, 라이브 화면에서 **인증 시작**으로 1:N 인증 대기를 켭니다. 승인되거나 시작 후 10초가 지나면 대기가 끝납니다.
안내 거리는 50–80 cm입니다. 등록은 한 명씩, 인증은 가장 큰 얼굴을 대상으로 합니다.

```bash
# 과거 시나리오 녹화 경로 (PAD 사용 모드에서는 현재 모델이 검증 gate를 통과하지 못함)
docker compose run --rm -u $(id -u):$(id -g) -e HOME=/tmp tester python scripts/demo_record.py   # -> data/demo/
# 영상 렌더링 (1920x1080, 30 fps; 헤드리스 Chromium + ffmpeg)
docker build -f docker/video.Dockerfile -t faceauth-video:dev .
docker run --rm -u $(id -u):$(id -g) -e HOME=/tmp -v $PWD:/app -w /app faceauth-video:dev python3 scripts/render_video.py
```

`ui/kiosk.html`을 브라우저로 열면 녹화된 장면을 실시간으로 반복 재생합니다. 키오스크 화면에는 단말이 실제로 받는 정보(안내 문구,
판정 결과)만 나오고, 유사도 점수와 내부 사유는 영상의 서버 측 화면(감사 로그)에만 나옵니다.

## 주의

- 내부 연구용 시제품입니다. 사용한 모델·데이터 대부분은 비상업 연구 라이선스입니다(`reports/SBOM.json`).
- 현재 실험용 깊이 PAD는 실제 인증에 적용되지만 실측 성능은 검증 전입니다. 기존 실패 기록은 보존합니다.
  production/pilot 검증 경로는 별도로 유지합니다([측정 결과·배포 상태](docs/FEMTO_PAD_VALIDATION.md)).
- `secrets/`의 CA·키는 개발용입니다.
