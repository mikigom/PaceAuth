# 1:N 얼굴 인증 시스템 — API 명세서 (시제품)

대상: 전용 단말·계정 서비스·운영 도구 개발자. 근거: `docs/RFP_KO.md`(FR-01~03, SEC-01, OPS-01),
`docs/Design_KO.md` §2·§3·§5·§7, 계약 문서 `docs/IMPLEMENTATION.md` §6. 구현: `faceauth/api/`,
게이트웨이: `deploy/nginx/nginx.conf`, 배포: `deploy/compose.api.yml`.

## 1. 구성과 신뢰 경계

```
단말 ──(TLS 1.3 + 단말 인증서, mTLS)──> gateway:8443 ──> api-1 / api-2:8000 ──> engine · search-1/2 · kms · db
계정 서비스·운영 도구 ──(내부망, Bearer admin_token)──────────> api-1 / api-2:8000
```

- 게이트웨이(nginx)는 TLS 1.3만 허용하고, 단말 CA(`secrets/ca.crt`)가 발급한 클라이언트 인증서가 없으면
  연결을 거절한다. 검증된 인증서를 `X-Client-Cert`(URL 인코딩 PEM), `X-Client-Verify`로 전달한다.
  클라이언트가 같은 이름의 헤더를 보내도 게이트웨이가 덮어쓴다.
- API는 **게이트웨이에서 온 연결만**(`FA_TRUSTED_PROXIES`, TCP 피어 주소) 위 헤더를 신뢰한다. 내부망의 다른
  컨테이너가 헤더를 위조하면 401로 거절하고 감사 로그에 남긴다.
- 단말 식별: `sha256(인증서 DER)` == `devices.cert_fingerprint`(요청이 주장하는 `device_id`의 값),
  `X-Client-Verify == SUCCESS`, 단말 상태 `active` — 셋 다 만족해야 한다. 다른 단말의 인증서로는 아무 것도
  할 수 없다(403).
- 원본 프레임은 메모리에서만 처리한다. 게이트웨이 요청 본문 버퍼(32 MB)와 임시 경로는 tmpfs, API 컨테이너는
  읽기 전용 + tmpfs `/tmp`이며 DB에는 원본을 저장하지 않는다.
- API 2중화: 게이트웨이가 api-1/api-2에 분산하고, 연결 실패 시에만 다른 복제본으로 넘긴다(수동 헬스체크,
  `max_fails=3 fail_timeout=10s`). 이미 API에 도달한 POST 요청은 절대 재전송하지 않는다.

## 2. 공통 규약

| 항목 | 규약 |
|---|---|
| 전송 | 단말: `https://gateway:8443` (TLS 1.3, mTLS 필수). 관리: `http://api-1:8000`, `http://api-2:8000`(내부망) |
| 본문 | JSON(`application/json`). 촬영 패킷만 msgpack(`application/msgpack`), 최대 32 MB |
| 응답 | 항상 JSON. 시간은 ISO 8601(UTC 오프셋 포함) |
| 오류 형식 | `{"detail": "<코드>"}` (§8). 판정 결과(거절 포함)는 오류가 아니라 200 응답 본문으로 준다 |
| 비공개 원칙 | 외부 응답에는 **후보·점수·상세 사유가 절대 포함되지 않는다**. 상세 사유는 `sessions.reason`과 감사 로그에만 기록 |
| 장애 시 | DB·engine·search·KMS 중 하나라도 실패하면 503 `{"detail":"unavailable"}`. 완화된 판정 경로는 없다(Fail closed) |

## 3. 단말용 엔드포인트 (게이트웨이 경유, mTLS)

### 3.1 `POST /v1/sessions` — 세션·일회성 challenge 발급

요청 `{"purpose": "enroll"|"auth", "device_id": "dev-001", "action_id": "pay"}`
(`device_id` 1~64자, `action_id` 1~128자. 등록은 관례상 `"enroll"`)

응답 200 `{"session_id": "<uuid>", "challenge": "<b64url 32바이트>", "expires_at": "2026-10-08T09:00:30+00:00"}`

- 유효시간: 등록 180초, 인증 30초. 세션은 **한 번만** 사용된다(판정과 같은 트랜잭션에서 소비).
- 429 `device_locked`: 해당 단말의 최근 5분간 거절된 인증 판정이 10건 이상(단말 잠금, 5분 경과 시 자동 해제).
  단계 인증(step_up)은 실패로 세지 않는다.
- 401/403: §1의 단말 식별 실패.

### 3.2 `POST /v1/enrollments` — 얼굴 등록 (FR-01)

본문: 촬영 패킷(§6) msgpack. `header`에 다음 필드를 추가하고 단말 키로 함께 서명한다.

| 필드 | 내용 |
|---|---|
| `account_assertion` | 계정 서비스가 계정 소유 확인 후 발급한 토큰(§5.2), `nonce` == 세션 challenge |
| `consent` | `{"version": "<FA_CONSENT_VERSION>", "agreed_at": "<ISO 8601, 시간대 포함>"}` |
| `re_enroll` | `true`이면 기존 활성 등록의 템플릿을 교체(재등록) |

응답 `{"status": "active"|"pending_review"|"rejected", "subject_id"?, "template_version"?, "coverage": {...}, "reason"}`

- `coverage`: 서버가 엔진의 머리 자세로 다시 판정한 방향별 수집 여부
  `{"C","L","R","U","D","UL","UR","DL","DR": bool}`(방향은 고객 기준, `faceauth/terminal/guide.py`와 같은 구간).
  세션·무결성 단계에서 거절되면 `{}`.
- `status=active`: 등록 완료. `pending_review`: 다른 고객과 중복 의심 → 운영자 검토 대기(자동 병합 없음, 검토 전
  인증 불가). `rejected`: 아래 `reason` 참조.
- 409 `{"status":"rejected","reason":"already_enrolled"}`: 계정에 살아 있는 등록이 있는데 `re_enroll`이 아님
  (또는 검토 대기 중). 세션은 소비된다.

등록 완료 조건(모두 충족): 계정 확인 토큰 유효·세션 결합, 동의 버전 일치, 모든 프레임 품질 통과(얼굴 정확히 1개,
RGB·Depth·IR 모두 유효), 모든 프레임 PAD 통과, 방향 커버리지 ≥ `FA_ENROLL_MIN_DIRS`(기본 9 = 정면+8방향),
동일인(모든 임베딩이 평균 벡터와 cos ≥ `FA_T_SAME`). 대표 템플릿 4개(슬롯0 정면, 슬롯1 고객 왼쪽 L·UL·DL,
슬롯2 오른쪽 R·UR·DR, 슬롯3 상하 U·D; 각 그룹의 정규화 평균, 없는 그룹은 전체 평균)는 KMS로 암호화해
저장하고, 검색 인덱스에는 outbox(`template_events`)로 반영한다. 재등록 시 이전 템플릿은 비활성화 + 암호문 삭제,
`template_version` += 1.

### 3.3 `POST /v1/authentications` — 1:N 인증 (FR-02, FR-03)

본문: 촬영 패킷(§6) msgpack(인증 세션, 최대 16프레임).

응답 200 `{"decision": "approved"|"rejected"|"step_up", "approval_token"?, "step_up_id"?, "reason"}`

- `approved`: `approval_token`(§5.1) 발급. 이 단말·이 `action_id`·이 세션에만 유효, 30초.
- `step_up`: 고위험 행위(`FA_HIGH_RISK_ACTIONS`). 얼굴 판정은 통과했으나 고객 소유 수단으로 추가 인증 필요 →
  `step_up_id`로 §3.4 호출(120초 이내).
- `rejected`: `reason` 참조. 검색 결과(후보·점수)는 어떤 경우에도 응답에 없다.

판정 정책(§4)의 7개 조건을 모두 만족해야 승인된다. 어느 조건도 다른 조건을 보상하지 않는다.

### 3.4 `POST /v1/stepups/{step_up_id}/complete` — 단계 인증 완료 (SEC-01)

요청 `{"assertion": "<계정 서비스 토큰>"}` — `nonce` == `step_up_id`, `acct` == 인증된 고객 계정,
`method` == 고객 소유 수단(`otp`, `pin` 등; `face` 불가).

응답 `{"decision": "approved", "approval_token", "reason": "ok"}`, `amr = ["face", "<method>"]`.
같은 단말에서만, 한 번만, 만료 전에만 완료할 수 있다. 실패 `{"decision":"rejected","reason":"denied"|"session"|...}`.

### 3.5 `POST /v1/fallback` — 비생체 대체 인증 (SEC-01)

요청 `{"session_id": "<인증 세션>", "assertion": "<계정 서비스 토큰>"}` — `nonce` == 그 세션의 challenge,
`method` == `pin`/`otp` 등(`face` 불가). 얼굴 등록 여부와 무관하게 사용 가능(동의 철회 고객 포함).

응답 `{"decision": "approved", "approval_token", "reason": "ok"}`, `amr = ["<method>"]`(얼굴 미포함).
세션은 소비되며(재사용 불가), 얼굴 거절 후에는 **새 세션**으로 호출한다.

## 4. 판정 정책 (인증)

| # | 조건 | 실패 시 외부 사유 |
|---|---|---|
| 1 | mTLS 인증서 == 단말, 단말 활성, 패킷 서명·해시·순서·동기화 유효, `calib_version` == 단말 등록값, 세션 존재·인증 목적·같은 단말·미만료·미사용·challenge 일치 | `session` / `denied` / 401·403 |
| 2 | 모든 프레임에서 RGB·Depth·IR 모두 디코딩, 얼굴 정확히 1개, 품질 통과(센서 누락 시 RGB-only 전환 없음) | `retry`(품질) / `denied`(모달리티 누락) |
| 3 | 모든 프레임 PAD live. PAD 실패는 유사도와 무관하게 최종 거절 | `denied` |
| 4 | top1 ≥ `FA_T_ID` **그리고** top1 − top2 ≥ `FA_MARGIN`(모호하면 거절, Top-1만으로 승인하지 않음) | `not_recognized` |
| 5 | 고객 상태 active, 동의 버전 == `FA_CONSENT_VERSION`, 철회 아님 — 인덱스가 아니라 **DB 원장**으로 확인(행 잠금 `FOR SHARE`) | `not_recognized` |
| 6 | 행위 위험도: 고위험이면 승인 대신 `step_up` | — |
| 7 | 세션 소비(`UPDATE ... WHERE consumed_at IS NULL AND expires_at > now() RETURNING`) + 판정 + 승인/단계 인증 행 + 감사 로그가 **한 트랜잭션**. 0행이면 재전송으로 보고 거절 | `session` |

검색은 활성 인식 모델 버전으로만 수행하며, 엔진이 보고한 인식 모델 버전이 DB의 활성 버전과 다르면
판정하지 않는다(503). 서로 다른 버전의 벡터는 비교·저장하지 않는다.

## 5. 토큰

형식(`faceauth/common/token.py`): `v1.<b64url(정렬된 JSON)>.<b64url(Ed25519 서명)>`, `iat`/`exp`는 유닉스 초.

### 5.1 승인 토큰 (API 발급, `secrets/approval.key`로 서명, 검증 키 `approval.pub`)

```json
{"aid": "<승인 uuid>", "sid": "<세션 uuid>", "dev": "<단말>", "act": "<action_id>", "acct": "<계정>",
 "amr": ["face"], "iat": 1760000000, "exp": 1760000030}
```

- `exp = iat + 30`. 의존 서비스는 서명·만료와 함께 `dev`·`act`가 자기 요청과 같은지 반드시 확인한다.
- `amr`: 얼굴 단독 `["face"]`(다중요소로 표시하지 않음), 단계 인증 `["face","otp"]`, 대체 인증 `["pin"]` 등.
- 승인 원장: `approvals` 테이블(대체 인증이면서 얼굴 등록이 없는 계정은 세션·감사 로그에만 기록, §10).

### 5.2 계정 확인 토큰 (계정 서비스 발급, API는 `account.pub`로 검증)

```json
{"acct": "<계정>", "method": "password|otp|pin|...", "nonce": "<세션 challenge 또는 step_up_id>",
 "iat": 1760000000, "exp": 1760000060}
```

`nonce`로 특정 세션/단계 인증에 묶이므로 다른 세션에 재사용할 수 없다. 시제품에서는 계정 서비스를 모사한다
(`secrets/account.key`는 시뮬레이터·시험 전용).

## 6. 촬영 패킷 (`faceauth/common/packet.py`)

```
{"header": {session_id, challenge, device_id, calib_version, captured_at_ms,
            frames: [{"seq", "ts_ms": {"rgb","depth","ir"}, "sha256": {"rgb","depth","ir"}, "label"}],
            (등록) account_assertion, consent, re_enroll},
 "signature": Ed25519(단말 키, canonical(header)),
 "frames": [{"rgb": JPEG, "depth": PNG uint16 mm, "ir": PNG uint8}, ...]}
```

서명이 세션·challenge·단말·보정 버전·프레임 순서·시각·원본 해시를 묶는다. 서버 검사:
서명(단말 등록 공개키) → 프레임 수 → 모달리티 누락 → 해시 → 시각 누락 → 동기화(|ts(rgb) − ts(depth/ir)| ≤ 15 ms)
→ 순서(seq·시각 증가) → `calib_version` → 프레임 수 상한(등록 64, 인증 16).

## 7. 사유 코드

외부 사유(응답 `reason`)는 아래 값만 사용한다.

| 외부 `reason` | 의미 | 대표 내부 사유(`sessions.reason`, 감사 로그) |
|---|---|---|
| `ok` | 승인/등록 완료(단계 인증 대기 포함) | `ok` |
| `retry` | 촬영 품질 문제, 다시 촬영 | `quality:no_face`, `quality:multi_face`, `quality:blur`, `quality:pose_out_of_range`, `quality:depth_invalid`, `same_person`(등록) |
| `not_recognized` | 식별 불가 | `no_match`(top1 < 임계값), `ambiguous`(마진 부족), `subject_inactive`(철회·검토 대기·동의 버전 불일치) |
| `denied` | 거부(재시도 무의미) | `pad_fail`, `bad_signature`, `hash_mismatch`, `missing_modality`, `unsynchronized`, `bad_frame_order`, `calib_mismatch`, `assertion_invalid`, `assertion_mismatch`, `assertion_method`, `subject_changed`, `too_many_frames`, `malformed_packet` |
| `session` | 세션 문제, 새 세션 필요 | `session_not_found`, `session_device_mismatch`, `session_purpose_mismatch`, `session_consumed`, `session_expired`, `session_replayed`, `challenge_mismatch`, `session_stepup_used_or_expired` |
| `coverage` | (등록) 방향 수집 부족, `coverage` 참고 | `coverage` |
| `consent` | (등록) 동의 버전 불일치·형식 오류 | `consent` |
| `review` | (등록) 중복 의심, 운영자 검토 대기 | `review` |
| `already_enrolled` | (등록, 409) 이미 등록됨 | `already_enrolled` |

세션 단계에서 거절되면(없음·다른 단말·사용됨·만료) 세션을 소비하지 않는다(다른 단말이 남의 세션을
소모시킬 수 없음). 세션이 이 단말의 유효한 세션이면 무결성·품질·PAD·식별 실패도 세션을 소비한다.

## 8. HTTP 상태 / 오류 코드

| 상태 | `detail` | 의미 |
|---|---|---|
| 200 | — | 판정 결과(승인·거절·단계 인증) |
| 400 | `malformed_packet`, `invalid_key_or_cert` | 패킷 구조 오류, 단말 등록 키/인증서 오류 |
| 401 | `client_certificate_required` | 인증서 없음/미검증/게이트웨이 외 경로 |
| 401 | `unauthorized` | 관리 토큰 불일치 |
| 403 | `device_not_authorized` | 미등록 단말, 다른 단말 인증서, 비활성 단말 |
| 404 | `not_found` | 고객·단말·모델 버전 없음 |
| 409 | `already_enrolled`(본문 §3.2), `review_not_open`, `certificate_in_use` | 상태 충돌 |
| 413 | `packet_too_large` | 32 MB 초과(게이트웨이도 동일 한도) |
| 422 | (FastAPI 검증 오류) | JSON 필드 형식 오류 |
| 429 | `device_locked` | 단말 잠금 |
| 503 | `unavailable` | DB·engine·search·KMS 장애 또는 모델 버전 불일치(Fail closed) |

## 9. 서비스·관리 엔드포인트 (`Authorization: Bearer <admin_token>`, 내부망)

| 엔드포인트 | 요청 → 응답 |
|---|---|
| `DELETE /v1/subjects/{subject_id}/face` | 동의 철회·삭제 → `{"revoked": true, "templates_deleted": n, "index_removed": bool, "key_destroyed": bool}` |
| `POST /v1/admin/devices` | `{"device_id", "public_key_pem"(Ed25519 SPKI PEM), "cert_pem"(단말 CA 발급), "calib_version"}` → `{"device_id","cert_fingerprint","status":"active"}` (재등록 = 키·인증서 교체 + 활성화) |
| `POST /v1/admin/devices/{id}/status` | `{"status": "active"|"disabled"}` (분실·변조 단말 즉시 차단) |
| `GET /v1/admin/reviews?status=open` | 중복 검토 목록 `{"reviews": [{"id","subject_id","candidate_subject_id","status","created_at","decided_at","decided_by"}]}` |
| `POST /v1/admin/reviews/{id}` | `{"decision": "approve"|"reject", "by": "<운영자>"}` → approve: 열린 검토가 없으면 고객 active. reject: 고객 철회 + 얼굴 데이터 삭제(아래와 동일 절차) |
| `GET /v1/admin/audit/verify` | 해시 체인 검증 `{"ok": bool, "first_bad_id": id|null, "rows": n}` |
| `POST /v1/admin/models/{name}/activate` | `{"version"}` → DB 활성 버전 교체 후 engine(및 인식 모델이면 search) 재적재 → `{"name","version","previous","engine_reloaded":[..],"search_reloaded":[..],"ok"}`. 롤백 = 이전 버전 활성화 |
| `GET /healthz` | DB 연결 확인 `{"ok": true}` / 503 |

삭제 절차(OPS-01): ① 한 트랜잭션에서 고객 `revoked`(커밋 즉시 신규 승인 차단), 템플릿 비활성 + 암호문 NULL,
`remove` 이벤트, 감사 로그 → ② KMS 고객 키 파기(암호 삭제, 백업 암호문도 복호 불가) → ③ 모든 검색 복제본의
`applied_event_id`가 제거 이벤트 이상이 될 때까지 최대 5초 대기. KMS·검색 확인이 실패해도 ①은 이미 확정되어
승인은 차단되며, 같은 요청을 다시 보내면(멱등) 나머지 단계를 완료한다. 이미 발급된 승인 토큰은 최대 30초 유효.
철회 후 같은 계정은 새 고객 ID로 다시 등록할 수 있다.

모델 교체: 검색 복제본은 활성 버전 변경을 감지해 재구축하며 그동안 503이다(원본 미보관이므로 새 인식 모델은
재등록이 필요). engine 재적재 실패 시 `ok=false` — 이전 버전을 다시 활성화해 롤백한다.

## 10. 보안 노트

- **재전송**: 세션 1회 사용(원자적 소비), 패킷 서명이 세션·challenge에 결합. 같은 패킷 재전송, 다른 세션의
  challenge 재사용, 만료 세션 → `session`.
- **영상 주입·변조**: 단말 키 없는 패킷(서명 오류), 서명 후 프레임 교체(해시 불일치), 순서·시각 조작,
  비동기 프레임, 보정 버전 불일치 → `denied`. 서버는 원본 3센서를 다시 검증한다(단말 판정 불신).
- **단말 변조/도난**: 인증서와 단말 ID 결합, 관리자가 즉시 비활성화, 반복 거절 시 단말 잠금.
- **정보 노출 최소화**: 외부 응답에 후보·점수·상세 사유 없음. 점수(top1/top2)는 감사 로그에만 기록하고,
  미식별·모호 판정에서는 후보 ID를 기록하지 않음(식별된 고객 ID만 기록).
- **감사**: 모든 판정·관리 행위를 해시 체인 감사 로그(추가 전용, 수정·삭제 트리거 차단)에 판정과 같은
  트랜잭션의 마지막 문장으로 기록한다. `GET /v1/admin/audit/verify`로 검증.
- **키 분리**: 템플릿은 KMS의 고객별 키로 암호화(API는 암호화·파기 권한만, 복호 권한 없음).
- **Fail closed**: 의존 서비스 장애 시 503, 판정 완화 없음. 엔진 응답 형식 오류·비유한 점수도 503.

## 11. 설정 (환경 변수, `deploy/compose.api.yml`)

| 변수 | 기본값/예 | 설명 |
|---|---|---|
| `FA_DB_DSN`, `FA_DB_PASSWORD_FILE` | `host=db,db-standby1,db-standby2 ... target_session_attrs=read-write`, `/run/secrets/db_api_password` | 쓰기 가능한 주 노드에 연결 |
| `FA_KMS_URL`, `FA_KMS_TOKEN_FILE` | `http://kms:8003`, `/run/secrets/kms_token_api` | |
| `FA_SEARCH_URLS`, `FA_ENGINE_URLS` | `http://search-1:8002,http://search-2:8002`, `http://engine:8001` | 복제본 무작위 순서 시도 |
| `FA_RECOG_MODEL` | `kprpe_vitb_wf12m` | 활성 버전은 DB `model_versions` |
| `FA_T_ID`, `FA_MARGIN`, `FA_T_DUP`, `FA_T_SAME` | 0.45, 0.05, 0.40, 0.50 (**임시값**, E2/E8 결과로 고정) | 코사인 유사도 |
| `FA_CONSENT_VERSION` | `v1` | 등록 시 필수, 인증 시 고객 동의 버전과 일치해야 함 |
| `FA_HIGH_RISK_ACTIONS` | `transfer_high,account_change` | 쉼표 구분 |
| `FA_ENROLL_MIN_DIRS` | 9 | |
| `FA_TRUSTED_PROXIES` | `gateway` | 단말 인증서 헤더를 신뢰할 피어 |
| `FA_APPROVAL_KEY_FILE`, `FA_ACCOUNT_PUB_FILE`, `FA_ADMIN_TOKEN_FILE` | `/run/secrets/approval_key`, `account_pub`, `admin_token` | |

## 12. 실행과 시험

```
docker compose -p fa-api -f docker-compose.yml -f deploy/compose.api.yml --profile apitest up -d
docker compose -p fa-api -f docker-compose.yml -f deploy/compose.api.yml --profile apitest run --rm \
    -u "$(id -u):$(id -g)" -e HOME=/tmp tester python -m pytest -q -p no:cacheprovider tests/test_api_flow.py
docker compose -p fa-api -f docker-compose.yml -f deploy/compose.api.yml --profile apitest down -v
```

`apitest` 프로필은 Postgres(`deploy/db/faceauth.sql`)와 engine·search·KMS 계약 모사(`tests/test_api_fakes.py`)를
띄운다. 실제 db·kms·search와 함께 돌릴 때는 `FA_TEST_REAL=1`(장애 주입 시험 2건은 건너뜀).
정책 단위 시험: `python -m pytest tests/test_api_policy.py`.

## 13. 알려진 제한 (시제품)

- 임계값은 임시값이다. 동일인 검사(평균 벡터와의 cos)는 두 사람이 반씩 섞인 등록 묶음에 약하므로(각 cos ≈ 0.71)
  `FA_T_SAME`을 그 이상으로 고정하거나 쌍별 검사로 강화할 것.
- 검색 복제본 반영 지연(폴링 200 ms) 동안 막 등록된 고객은 미식별·중복 검사 누락이 생길 수 있다(거절 방향).
- 대체 인증에서 얼굴 미등록 계정의 승인은 `approvals`에 기록되지 않는다(`approvals.subject_id NOT NULL`).
- 동의 버전을 올리면 기존 고객은 재등록(`re_enroll`) 전까지 인증되지 않는다(동의 갱신 전용 API 없음).
