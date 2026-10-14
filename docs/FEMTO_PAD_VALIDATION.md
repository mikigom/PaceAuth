# Femto Bolt PAD 상태와 검증 참고 자료

사용자의 명시적 요청으로 현재 내부망 파일럿은 **PAD를 사용하지 않고 얼굴 등록·인증을 진행**합니다.
API와 엔진에 `FA_PILOT_DISABLE_PAD=1`을 함께 설정합니다. 기본값 `0`에서는 기존 PAD 검증 gate와
프레임별 PAD 판정을 그대로 적용합니다. 현재 모드는 `pad_enabled=false`, `pad_mode=disabled_pilot`로
표시하며 생체·위조 여부를 판정하지 않습니다. 센서 품질, 촬영 서명, 계정, 얼굴 신원 판정은 유지합니다.
PAD 비활성화는 모델의 검증 완료를 뜻하지 않으며 `pad_validation`에는 실제 미검증 상태를 남깁니다.
원본 수집이나 PAD 재학습을 현재 파일럿에 추가하지 않습니다.

## 첫 실센서 진단 결과

2026-10-14 동의 후 참여자가 생체로 표시한 진단 32시도·160프레임을 측정했습니다.
원본 영상은 저장하지 않았고 기존 모델·임계값도 바꾸지 않았습니다.

| 범위 | 시도 | 프레임 | 품질 통과 | PAD 물리 검사 통과 | 최종 PAD 통과 |
|---|---:|---:|---:|---:|---:|
| 전체 | 32 | 160 | 136 | 128 | 0 |
| 얼굴 깊이 600–700mm | 2 | 10 | 10 | 9 | 0 |

600–700mm 구간의 실제 깊이는 643–667mm였고 최대 CNN 확률은 0.264854로 기존 임계값
0.999476에 미치지 못했습니다. 거리만 바꾸어 해결되지 않았습니다. 전체에서 물리 검사를 통과한
128프레임도 최종 PAD를 통과하지 못했으므로 물리 검사 실패만이 원인도 아닙니다.
이는 한 참여자의 `calibration`/`diagnostic` 자료이며 공격 표본·독립 시험 자료는 없습니다.
160개의 인접 프레임을 독립 표본 160회나 일반적인 정확도 수치로 해석하지 않습니다.
[최초 실센서 진단 기록](../reports/deployment/femto_bolt_first_live_diagnostics_20261014.json).

점검에서 학습·추론 입력 구성의 불일치나 뚜렷한 GPU 수치 오류는 확인되지 않았습니다.
실제 IR 변환·ToF 깊이·검출 랜드마크와 합성 학습 환경의 차이는 확인되지만, 어느 차이가 주원인인지는
분리하지 못했습니다. 점수 기록으로 원본 프레임을 재현하거나 CNN을 재학습할 수는 없습니다.

기존 `pad_rgbdir@2`의 `MODEL.json`과 E7 보고서는 **시뮬레이션 전용**이라고 명시합니다.
PAD 사용 모드로 평가하면 여전히 `pad_calibration_mismatch`와 `pad_real_sensor_validation_missing`이
남습니다. 아래 자료는 PAD를 끄기 전 상태를 기록한 것이며 현재 비활성 모드의 인증 가능 여부와 구분합니다.
[당시 PAD 준비 상태](../reports/deployment/femto_bolt_pad_readiness_20261014.json),
[GPU 실행·합성 워밍업](../reports/deployment/femto_bolt_gpu_20261014.json).

## 현재 차이

| 항목 | 기존 학습·평가 | Femto Bolt 현재 캡처 |
|---|---|---|
| 보정 | `gemini336-sim-v1`, fx 385 | `femto-bolt-v1-50beeef1ae82fb27`, 출력 컬러좌표 fx 약497.406 |
| 깊이 | 합성 얼굴 메쉬/센서 오차 | 실제 ToF, 가림·반사·멀티패스·무효점과 컬러좌표 재투영 |
| IR | 합성 uint8, 얼굴 반사율·음영 가정 | Y16, 고정 `log1p` 곡선으로 uint8 변환 |
| 랜드마크 | 합성 정답 5점 | 엔진 SCRFD 검출 5점 |
| PAD 사용 시 평가 단위 | 단일 프레임 | 등록·인증 시도의 모든 선택 프레임이 live여야 함 |

PAD는 RGB 3채널, 상대깊이, 유효깊이 마스크, IR의 6채널 CNN과 물리 검사를 함께 씁니다.
현재 임계값은 0.9994763732이며 깊이 유효비율≥0.5, 잔차 relief≥3mm,
IR 중앙값≥30, 음영 상관≥0.1이 필요합니다. 이 수치는 합성 분포에서 정한 것입니다.
fx는 깊이 표면 기울기·IR 음영 상관 계산에 직접 들어갑니다. fx만 고치거나 IR 기준을
낮추거나 보정 이름을 바꾸어도 실제 성능 증적이 생기지 않습니다.

현재 NFOV 640×576 모드의 공식 동작 범위는 0.5–3.86m입니다. 초기 현장 실험은
50–80cm에서 수행하고, 그보다 가까운 구간은 별도 검증합니다.
[Orbbec 공식 하드웨어 사양](https://doc.orbbec.com/documentation/Orbbec%20Femto%20Bolt%20Documentation/Femto%20Bolt%20Hardware%20Specifications).

## PAD를 다시 사용할 때의 검증 참고 절차

다음 절차는 PAD 사용 모드의 검증을 다시 진행할 때 참고할 자료입니다. 현재 파일럿의 진행 조건이 아니며
추가 수집·학습을 시작하라는 지시도 아닙니다. PAD 사용 모드에서는 점수 진단과 인증 승인을 구분하고,
검증 자료가 없는 모델의 등록 완료·인증 승인 토큰 발급을 차단합니다.

1. 참여자가 **점수 기반 PAD 실험**의 목적과 저장 항목에 명시적으로 동의합니다.
   참여자·동의·시도 식별자는 계정명 대신 실험용 무작위 ID를 사용합니다.
2. 실제 얼굴을 50/65/80cm, 정면과 등록의 주변 8방향, 일반/어두운/측광 조건에서
   측정합니다. 안경·표정·다른 날짜·다른 참여자를 포함하고 각 시도를 새로 시작합니다.
   실패·얼굴 미검출·센서 무효·중도 포기도 기록해야 성공 표본만 남는 편향을 줄일 수 있습니다.
3. 동의받은 얼굴 자료로 물리 공격을 준비합니다. `print`: 평면/휘어진 무광·유광 인쇄물,
   `replay`: 서로 다른 휴대폰·태블릿·모니터의 정지영상/동영상,
   `mask_rigid`: 실제 입체 경질 가면, `mask_silicone`: 실제 유연한 실리콘 가면.
   각 물체에 `instrument_id`를 부여합니다. 공격 종류·재질·장소별 결과를 분리합니다.
   가면이 없으면 해당 공격 범위는 **미시험**으로 남깁니다.
4. 학습/보정/최종시험의 사람과 물리 공격 도구를 분리합니다. 모델·임계값을 먼저 고정하고
   최종시험을 실행합니다. 인접한 30 FPS 프레임 300장은 독립 시도 300회가 아닙니다.
5. 현재 세 관측 모드의 수집은 원본·임베딩을 저장하지 않는 점수 측정입니다. 모델 재학습은 별도 원본 수집 동의,
   보관·철회·삭제 방식, 합법적 사용 권한을 정한 후 진행합니다. 점수만으로 CNN을 재학습할 수 없습니다.

위 목록은 현장 진단 계획이며 표준 인증 시험을 대체하지 않습니다. 사진·영상·가면별 오류율은
서로 대체할 수 없습니다. NIST의 소프트웨어 RGB 평가도 공격 도구 종류별 차이를 보고하며,
그 성적을 이 RGB-D-IR 모델에 적용할 수는 없습니다.
[NISTIR 8491](https://nvlpubs.nist.gov/nistpubs/ir/2023/NIST.IR.8491.pdf).

## 웹/엔진 기록 계약

`faceauth.common.pad_validation.make_observation`은 센서를 열거나 파일을 쓰지 않습니다.
웹은 동의를 받은 뒤 라이브 워커의 새 시도 프레임을 엔진 observe 경로에 전달하고,
반환된 `model_binding`과 프레임 측정을 다음처럼 전달합니다.

```python
row = make_observation(
    presentation_id="random-attempt-id", subject_id="study-person-01",
    consent_id="random-consent-id", consent=True,
    kind="live", split="calibration", instrument_id=None,
    calib_version=verified_packet_calibration,
    binding=engine_response["model_binding"], frames=engine_response["frames"],
    conditions={"mode": "diagnostic", "distance_mm": 650, "lighting": "indoor",
                "pose": "C", "location_id": "room-A", "eyewear": "none"},
)
```

`kind`는 `live|print|replay|mask_rigid|mask_silicone`, `split`은
`train|calibration|test`입니다. `frames`에서 보관되는 값은 PAD의
`score/live/nonresponse/cnn/checks/failed`와 품질/실패 사유뿐입니다.
RGB·깊이·IR·랜드마크·임베딩·계정 주장은 제외합니다. PAD 자체가 수행되지 않은 프레임은
비응답으로 남습니다. `binding`은 모델 메타데이터·가중치·검출기 및 처리 코드 hash를 묶습니다.
이 기록은 여전히 실험 참여 기록이므로 제한된 보관 기간과 삭제 절차를 적용해야 합니다.

웹 `/api/auth/observe` 요청의 `observation_mode`는 다음 세 모드를 지원합니다.

| `observation_mode` | 프레임 수집 | 기록의 `conditions.mode` |
|---|---|---|
| `diagnostic` (기본) | 품질 불합격도 포함한 고정 간격 정면 5프레임 | `diagnostic` |
| `auth` | 실제 인증과 동일한 정면·품질 선택기의 3프레임 | `auth` |
| `enroll` | 실제 등록과 동일한 Guide의 9방향별 2프레임, 총18프레임 | `enroll` |

웹이 선택한 모드의 실제 프레임 선택 과정을 수행하고, 해당 모드로 측정 기록을 만듭니다.
이미 수집한 진단 5프레임의 라벨을 `auth`나 `enroll`로 바꾸어 대신하지 않습니다.
모든 관측 모드는 동의 후 점수·물리 검사만 저장하며 얼굴 등록, 템플릿 생성, 인증 승인 또는
승인 토큰 발급을 하지 않습니다. PAD가 아직 미검증이어도 `capture_ready`이면 관측할 수 있지만,
PAD 사용 모드의 등록·인증 `ready` 조건은 그대로 유지됩니다. `observe`는 별도의 진단 경로로
PAD 비활성 파일럿에서도 PAD를 측정할 수 있지만, 현재 사용자 등록·인증 흐름에서는 사용하지 않습니다.

이는 실제 선택 경로의 자료를 수집할 수 있도록 구현한 기능이며 현장 검증 완료를 뜻하지 않습니다.
이미 수집한 32시도를 포함한 `diagnostic` 자료는 승인 근거 통계에 포함하지 않습니다.
`auth`/`enroll` 자료도 아래의 독립 시험·분할·표본·오류 상한 조건을 충족해야 합니다.
`diagnostic_summary`는 별도 진단용입니다.
진단 요약은 보정/시험 자료 모두 보여주고 `diagnostic_by_split`으로 구분합니다. 승인 통계는
오직 최종시험 split의 실제 인증/등록 시도만 사용합니다.
각 요약의 `diagnostics`에는 CNN/최종 점수와 IR·relief·음영 상관 등의 5/50/95백분위수,
검사별 실패 프레임 수를 담습니다. 이 프레임 분포는 센서 도메인 차이를 진단하기 위한
값이며 독립 표본 수나 정확도 신뢰구간을 늘리지 않습니다.

품질·얼굴 안내를 한 번 이상 수행한 `auth`/`enroll` 관측이 안내 시간 초과 또는 세션 만료로
끝나면 그 시도도 비응답으로 저장합니다. 기록의 `conditions.capture_outcome`은
`selection_timeout` 또는 `session_expired`이고, 점수는 `null`이며 실패 사유가 남습니다.
이 명시적인 미측정 표시는 시도 분모와 비응답률에만 포함하고 실제 측정 프레임 수·물리 수치
분포에는 포함하지 않습니다. 시작과 종료에 직접 조회한 엔진 모델 binding·센서 보정이 같아야
기록하며, 모델 재적재 중인 결과를 새 모델의 관측으로 바꾸어 저장하지 않습니다.
사용자가 직접 취소한 촬영은 PAD 오판정이나 비응답으로 계산하지 않고 저장하지 않습니다.
취소, 카메라 분리, 서버 장애, 모델 변경 등 측정을 끝내지 못한 사건은 현장 운영자가 별도
시도 대장에 남겨야 합니다. 자동 JSONL만으로 전체 참가·중도 포기 현황을 알 수는 없습니다.

## PAD 재검증 시 현장 측정 방법

웹에서 별도 테스트 계정을 만들고 로그인한 뒤 촬영 동의와 **센서 검증**을 선택합니다.
GPU 엔진이 준비되어 `capture_ready`가 참이면 최종 인증 승인이 막혀 있어도 측정할 수 있습니다.
먼저 대상 코드를 무작위로 정하고 `calibration`/`live`/`diagnostic`으로 50–80cm 정면 측정을
시작합니다. 같은 위치에서 동의받은 얼굴의 인쇄물과 다른 화면 재생을 각각 독립 시도로 측정하고
공격 매체 코드를 달리 기록합니다. 이어 `auth`, `enroll` 모드의 실제 안내를 수행합니다.
이 첫 진단은 센서 입력과 모델의 차이를 확인하기 위한 것으로, 한 사람의 성공이나 사진 한 장의
거절만으로 PAD 검증 완료를 선언할 수 없습니다. 최종시험 사람·매체는 보정 자료와 분리합니다.

운영자가 점수 기록을 호스트로 복사할 때는 다음 명령을 사용합니다. 실제 기록이 생긴 뒤에만
실행하며, 내보낸 파일에도 서버 기록과 같은 접근·보관·삭제 정책을 적용합니다.

```bash
docker compose -f compose.live.yml exec -T terminal \
  cat /var/lib/paceauth/pad-observations.jsonl > consented-study.jsonl
```

## 평가와 배포 차단

```bash
python scripts/evaluate_sensor_pad.py status \
  --model-dir data/models/pad_rgbdir_v2 \
  --calib-version femto-bolt-v1-50beeef1ae82fb27

python scripts/evaluate_sensor_pad.py evaluate \
  --model-dir data/models/pad_rgbdir_v2 \
  --calib-version femto-bolt-v1-50beeef1ae82fb27 \
  --observations consented-study.jsonl --output unsigned-report.json
```

검증 미완료는 종료 코드 2입니다. 두 번째 명령은 실제 표본이 있을 때 실행합니다.
아직 없는 공격·최종시험 자료를 합성 관측 파일로 대신하지 않습니다.
보고서는 최종시험 split만 평가하고, 모든 프레임 live 규칙을 시도 단위로 적용합니다.
공격별 APCER, 생체 BPCER, 비응답률, 전체 시도 중 통과율과 각각의 **단측 95%
Clopper–Pearson 상한**을 계산합니다. 0회 오류라도 표본이 적으면 상한은 큽니다.
독립성은 ID 중복 검사만으로 입증되지 않으므로 실제 시험 운영자가 확인해야 합니다.

`RFP_KO.md` §4는 PAD 한계값을 별도 합의하도록 요구합니다. 기본 합격 수치나 기본 합격
표본수는 없습니다. 평가에서 다음 정책을 명시적으로 제공해야 합니다 (`--policy FILE`):

- `reference`, `approved_by`: 승인된 시험 계획/운영 담당자
- `required_species`: 네 물리 공격 유형 전체
- `required_modes`: `enroll`, `auth` 모두. 두 흐름의 프레임 선택/포즈가 달라 각각 평가합니다.
- `min_live_presentations`, `min_attack_presentations`, `min_live_subjects`, `min_attack_instruments`
- `max_apcer_upper95`, `max_bpcer_upper95`, `max_apnrr_upper95`, `max_bpnrr_upper95`

이와 함께 `--independent-test FILE`에 독립 시험의 `assessor`, `report_id`,
`report_sha256`, `independent: true`가 필요합니다. 서명 전에 실제 보고서와 현장 범위,
표본 독립성, 합의한 판정 기준을 담당자가 검토해야 합니다. 이 JSON 선언 자체가
독립 시험 수행이나 ISO 인증을 증명하지는 않습니다.

최소 표본수와 오류 상한은 전체 요약 및 각 모드 요약에 모두 적용합니다.
검출기는 엔진이 실제 로드한 디렉터리를 gate의 `detector_dir=`와 CLI의 `--detector-dir`에
전달합니다. 생략할 경우 registry에 검출기 버전이 하나일 때만 자동 선택합니다.

모델 후보가 정식 학습/보정·독립 평가를 마친 후에만 별도 release 검토를 실행합니다.
`sign --report ... --model-dir ... --release-key ...`는 조건을 충족한 보고서에만
Ed25519 서명을 추가하고, 평가 명령은 키를 읽거나 서명하지 않습니다.
기본 배포 위치는 해당 모델 디렉터리의 `PAD_VALIDATION.json`과 `.json.sig`입니다.
신뢰하는 release 공개키로 서명이 검증되고, 모델/가중치/처리코드/센서 보정이 일치하고,
정책·독립 시험·표본·오류율 조건이 모두 충족되어야 gate가 ready를 반환합니다.
모델 자체는 기존 서명 registry를 통해서도 별도로 검증됩니다.

PAD 사용 모드의 런타임은 `PADValidationGate(model_dir, release_pub).status(calib_version)`의
`ready/reasons`를 사용합니다. 승인 gate가 통과해도 각 등록/인증 시도의 기존 PAD·품질·신원
검사를 계속 수행합니다. 영상 주입·재전송·단말 변조 방어는 이 물리 PAD 평가와 별도로 시험합니다.
