# 운영 매뉴얼 — 배포 · 복구 (1:N 얼굴 인증 프로토타입)

RFP §6 납품물 "배포·복구 매뉴얼". 대상 요구사항: OPS-01(템플릿 암호화·키 분리, 동의 철회·삭제, 감사 로그,
이중화, 서명 업데이트, 모델별 롤백), RFP §4 가용성(단일 노드 장애 시 승인 완료된 등록·철회 RPO=0, RTO ≤ 15분,
복구 중 승인 조건 완화 금지).

이 문서는 **프로토타입** 기준이다. 모든 노드가 한 호스트의 Docker Compose 위에서 돌고, 센서·단말 하드웨어는
시뮬레이션이다. 운영 환경으로 옮길 때 바꿔야 하는 항목은 §10에 모았다. 서비스 간 계약은 `docs/IMPLEMENTATION.md`
를 따른다.

---

## 1. 구성 요약

| 서비스 | 이미지 / 실행 | 역할 | 비고 |
|---|---|---|---|
| `db` | postgres:16-alpine, `deploy/db/node.sh` | 최초 primary | 이후 역할은 장애에 따라 바뀜 |
| `db-standby1`, `db-standby2` | 동일 | 동기 standby | 볼륨이 비어 있으면 primary를 `pg_basebackup`으로 복제 |
| `kms` | faceauth:dev, uvicorn :8003 (worker 4) | 피험자별 데이터 키, 암복호화, 키 파기 | `faceauth/kms/main.py` |
| `api-*`, `engine`, `search-*`, `gateway` | 각 담당자 compose 파일 | | `deploy/compose.{api,engine,search}.yml` |

- 네트워크: `edge`(단말 → gateway mTLS), `internal`(`internal: true`, 외부 통신 불가). DB·KMS는 `internal`에만 있고
  호스트 포트를 열지 않는다.
- 컨테이너: `read_only: true`, `/tmp`(및 Postgres 소켓 디렉터리)는 tmpfs. DB 데이터는 named volume
  (`db-data`, `db-standby1-data`, `db-standby2-data`)에만 기록된다. Postgres는 uid 70(postgres), KMS는 uid 65534로 실행.
- 비밀정보: `secrets/`(호스트 0700) → compose secrets → 컨테이너의 `/run/secrets/<이름>`.

| 비밀정보 | 사용처 |
|---|---|
| `db_postgres_password` | Postgres 슈퍼유저(최초 initdb에만 사용; 네트워크 접속 불가, 로컬 peer만 허용) |
| `db_api_password`, `db_ro_password`, `db_kms_password`, `db_repl_password` | 역할 `faceauth_api`, `faceauth_ro`, `kms`, `replicator` |
| `kms_master.key` (→ `kms_master_key`) | KMS 마스터 키 32바이트. KMS 컨테이너에만 마운트 |
| `kms_token_api`, `kms_token_search` | KMS Bearer 토큰: api = encrypt + destroy, search = decrypt |
| `release.key` / `release.pub` | 모델 레지스트리 서명 / 검증 (engine은 pub만) |
| `approval.key`, `account.key`, `ca.*`, `gateway.*`, `devices/*`, `admin_token` | API·게이트웨이·단말 (각 담당) |

DB 역할과 접근 범위(`deploy/db/init-primary.sh`, `deploy/db/pg_hba.conf`):

| 역할 | 접근 |
|---|---|
| `faceauth_api` | DB `faceauth` 소유자(모든 테이블 소유). DB `kms` 접속 불가 |
| `faceauth_ro` | DB `faceauth`의 `templates`, `template_events`, `model_versions` SELECT만 (`subjects` 등은 거부) |
| `kms` | DB `kms` 소유자. DB `faceauth` 접속 불가 |
| `replicator` | 복제 연결만 |
| `postgres` | 컨테이너 내부 unix 소켓(peer)만. 네트워크 접속 불가 |

`pg_hba.conf`는 위 (DB, 역할) 쌍만 `samenet`(internal 네트워크)에서 scram-sha-256으로 허용하고 나머지는 모두 거부한다.
두 DB 모두 `REVOKE ALL ... FROM PUBLIC`.

---

## 2. 최초 구축 (빈 호스트에서)

리포지토리 루트에서 실행한다. `.env`의 `COMPOSE_FILE`이 base + 담당자별 compose 파일을 합친다(프로젝트 이름 `faceauth`).

```bash
cd /fdisk2/Codes/jhseo/faceauth

# 2.1 비밀정보 생성 (멱등: 있는 파일은 절대 덮어쓰지 않는다)
docker run --rm -u $(id -u):$(id -g) -v $PWD:/app -w /app faceauth:dev python scripts/init_secrets.py --devices 4

# 2.2 이미지 (버전 고정: requirements.txt, docker/Dockerfile)
docker build -t faceauth:dev -f docker/Dockerfile .

# 2.3 모델 파일 배치 후 서명된 레지스트리 생성 (data/models/<dir>/MODEL.json, §7)
docker run --rm -u $(id -u):$(id -g) -v $PWD:/app -w /app faceauth:dev python scripts/build_registry.py

# 2.4 DB 3노드 + KMS
docker compose up -d db db-standby1 db-standby2 kms
scripts/failover.sh status
#   db primary lsn=... ANY 1 (db_standby1, db_standby2) replicas=[db_standby1:streaming:quorum db_standby2:streaming:quorum]
#   db-standby1 standby lsn=... upstream=db
#   db-standby2 standby lsn=... upstream=db

# 2.5 model_versions 등록 + 이름별 최초 버전 활성화 (감사 로그 기록)
docker compose run --rm -u $(id -u):$(id -g) tester python scripts/build_registry.py \
  --dsn "host=db,db-standby1,db-standby2 dbname=faceauth user=faceauth_api target_session_attrs=read-write"

# 2.6 나머지 서비스
docker compose up -d
```

최초 기동 시 동작:

1. `db`(`FA_PG_INIT=1`)가 initdb(데이터 체크섬 활성) 후 `init-primary.sh`를 실행한다: 역할 4개(비밀번호는
   `/run/secrets`), DB `faceauth`(소유자 faceauth_api) / `kms`(소유자 kms), `faceauth.sql`을 **faceauth_api로**,
   `kms.sql`을 **kms로** 적용(테이블 소유권이 각 서비스 역할에 있음), `faceauth_ro` 권한 부여,
   `synchronous_standby_names = 'ANY 1 (db_standby1, db_standby2)'`.
2. 다른 피어가 이미 응답하면 `db`는 initdb를 **거부**하고 종료한다(빈 primary가 두 개 생기는 사고 방지).
3. standby는 볼륨이 비어 있으면 `FA_PG_PRIMARY`(기본 `db`)를 `pg_basebackup -X stream -R`로 복제하고(실패 시 2초마다
   재시도) standby로 기동한다. 복제 비밀번호는 tmpfs의 `/tmp/pgpass`에만 둔다.
4. standby가 하나도 붙기 전에는 쓰기 커밋이 대기한다. 의도된 동작이다(§4).

---

## 3. 단말(디바이스) 등록

`devices` 테이블 필드: `id`, `public_key`(Ed25519 원시 32바이트, 촬영 패킷 서명 검증용),
`cert_fingerprint`(mTLS 클라이언트 인증서 DER의 sha256, 소문자 hex), `calib_version`(시뮬레이터: `gemini336-sim-v1`),
`status`(`active`/`disabled`).

```bash
docker run --rm -u $(id -u):$(id -g) -v $PWD:/app -w /app faceauth:dev python scripts/init_secrets.py --devices 5  # dev-005 추가, 기존 키 유지
openssl x509 -in secrets/devices/dev-005.crt -outform DER | sha256sum | cut -d' ' -f1                 # cert_fingerprint
openssl pkey -pubin -in secrets/devices/dev-005.ed25519.pub -outform DER | tail -c 32 | xxd -p -c 64   # public_key (hex)
```

- 정식 경로: API `POST /v1/admin/devices` (Bearer `admin_token`, 감사 로그 기록). 요청 본문 형식은 API 문서를 따른다.
- 개발용 대체(감사 로그 없음):
  ```bash
  docker compose run --rm -T tester sh -c 'PGPASSWORD=$(cat secrets/db_api_password) psql -v ON_ERROR_STOP=1 \
    "host=db,db-standby1,db-standby2 dbname=faceauth user=faceauth_api target_session_attrs=read-write"' <<'SQL'
  INSERT INTO devices (id, public_key, cert_fingerprint, calib_version)
  VALUES ('dev-005', '\x<public_key hex>', '<cert_fingerprint>', 'gemini336-sim-v1');
  SQL
  ```
- 분실·탈취: `UPDATE devices SET status = 'disabled'`. API는 인증서 지문과 단말 상태를 매 요청 DB에서 확인하므로
  즉시 차단된다(개발 CA에는 CRL이 없다, §10).

---

## 4. 데이터베이스 이중화 원칙 (RPO = 0)

- `synchronous_commit = on`, primary의 `synchronous_standby_names = 'ANY 1 (<다른 두 노드>)'`.
  커밋 응답은 standby 하나 이상이 WAL을 디스크에 flush한 뒤에만 나간다. 따라서 **응답을 받은 등록·철회는
  어떤 노드 하나가 사라져도 남는다.**
- standby 1대 장애: 남은 1대가 응답하므로 쓰기는 계속된다.
- standby 2대 장애(또는 primary + standby 1대): 커밋이 **대기**한다(fail closed). `synchronous_standby_names`를
  비워서 쓰기를 살리는 것은 복제 없는 커밋 응답(RPO > 0)이 되므로 **금지**한다(RFP §4 "복구 중 승인 조건 완화 금지",
  Design §7). 쓰기 가용성은 standby 재구축(§5.4)으로만 복구한다.
- standby의 application_name = `cluster_name`(`db`, `db_standby1`, `db_standby2`). `pg_basebackup` 자체의 WAL 스트림은
  이름이 달라 동기 standby로 계산되지 않는다.
- 동적 설정(`synchronous_standby_names`, `primary_conninfo`)은 `postgresql.auto.conf`(ALTER SYSTEM)에,
  정적 설정은 `deploy/db/postgresql.conf`에 둔다. 동적 설정을 커맨드라인에 두면 페일오버 스크립트가 바꿀 수 없다.
- 복제 슬롯은 쓰지 않는다(`wal_keep_size = 2GB`). 죽은 standby가 primary 디스크를 채우지 않게 하기 위함이며,
  너무 오래 떨어져 있던 standby는 재구축한다.

애플리케이션 개발자 필수 규칙 (§11에 상세):
쓰기 트랜잭션의 COMMIT을 취소하지 않는다(`statement_timeout`, `pg_cancel_backend`). Postgres는 동기 대기를 취소하면
경고와 함께 **성공**을 돌려주는데, 이 커밋은 복제되지 않았을 수 있다.

---

## 5. 장애 대응: 페일오버와 복구

도구: `scripts/failover.sh` (호스트에서 실행, `FA_PROJECT`=compose 프로젝트, 기본 `faceauth`).

| 명령 | 동작 |
|---|---|
| `status` | 노드별 역할, LSN, 상위 노드, primary의 동기 상태 |
| `primary` | 현재 primary 이름 (없거나 둘 이상이면 실패) |
| `watch [간격초] [횟수]` | primary를 간격마다 점검하고, 연속 실패가 횟수에 이르면 `promote` 1회 실행 후 종료 |
| `promote [이전 primary]` | 펜싱 → 가장 앞선 standby 승격 → 나머지 standby 재지정 → 동기 standby 확인 |
| `rebuild <노드>` | 노드 데이터를 버리고 현재 primary에서 다시 복제해 standby로 기동 |

### 5.1 감지

- 자동: `scripts/failover.sh watch 1 3` (1초마다 점검, 3회 연속 실패 시 승격). 운영 호스트의 supervisor(systemd 등)로
  띄우며, 한 번 승격하면 종료한다. 재구축 후 다시 실행한다(연쇄 자동 승격 방지).
- 수동: 장애를 확인한 운영자가 `scripts/failover.sh promote`. 이전 primary가 아직 primary로 응답하면 거부하고,
  확실할 때만 `FORCE=1`(먼저 kill로 펜싱)을 쓴다.

### 5.2 `promote`가 하는 일 (순서가 안전성의 핵심)

1. **펜싱**: 이전 primary 컨테이너의 재시작 정책을 `no`로 바꾸고 kill. 이후 다시는 쓰기를 받지 못하게 한다.
2. **승격 대상 선택**: 살아 있는 standby 중 디스크에 받은 WAL 위치(LSN)가 가장 큰 노드. 응답된 커밋은 모두
   standby 하나 이상에 있고 WAL은 하나의 연속 스트림이므로, 가장 앞선 standby가 응답된 커밋 전부를 가진다.
   승격 시 Postgres는 받은 WAL을 모두 재생한 뒤 새 타임라인을 연다.
3. 승격 **전에** 대상 노드의 `synchronous_standby_names = 'ANY 1 (<다른 두 노드>)'`. 첫 쓰기부터 동기 조건이 걸린다
   (이전 primary 이름도 포함되어 재구축 후 다시 동기 후보가 된다).
4. `pg_promote()`.
5. 나머지 standby의 `primary_conninfo`를 새 primary로 바꾸고 reload. `recovery_target_timeline = latest`이므로 새
   타임라인을 따라간다.
6. 새 primary에 동기(quorum) standby가 붙을 때까지 대기. 그 전까지 커밋은 대기한다.
7. **저장소 펜싱**: 이전 primary 볼륨에 `standby.signal`을 쓴다. 누가 `docker compose up`으로 다시 띄워도
   읽기 전용 standby로만 올라온다.

마지막 줄에 JSON(타이밍, 노드별 LSN)을 출력한다.

### 5.3 페일오버 직후 점검 (필수)

```bash
scripts/failover.sh status                       # 새 primary + quorum standby 1대 이상
curl -X POST http://search-1:8002/v1/admin/reload  # search 복제본마다 (internal 네트워크의 컨테이너에서)
```

- **search 인덱스 재구성은 필수다.** primary에서 커밋된 행은 standby 응답 전에도 다른 세션에 보인다. 이전 primary가
  죽을 때 응답되지 않은(따라서 사라진) 트랜잭션의 `template_events`를 search가 이미 적용했을 수 있고, 새 primary는
  그 이벤트 id를 다시 발급할 수 있다(시퀀스는 32개 단위로 WAL에 기록). 재구성하지 않으면 유령 템플릿이 남고,
  재발급된 id의 이벤트를 건너뛴다.
- API·KMS·engine은 다중 호스트 DSN(`target_session_attrs=read-write`)으로 자동 재연결한다. 장애 중 요청은
  503/거절(fail closed)이다.
- 감사 로그 해시 체인: `GET /v1/admin/audit/verify` (E10에서 매 라운드 `audit_verify()` 통과 확인).

### 5.4 장애 노드 재구축

```bash
scripts/failover.sh rebuild db      # 예: 이전 primary가 db였던 경우
```

컨테이너를 제거하고, 볼륨의 데이터를 지운 뒤 현재 primary에서 `pg_basebackup`으로 복제해 standby로 띄우고,
primary의 `pg_stat_replication`에 streaming으로 나타날 때까지 기다린다. 새 primary의 동기 후보 목록에 이미 이 노드가
들어 있으므로 별도 설정은 없다. 실제 하드웨어 장애라면 교체 후 같은 명령을 쓴다. 데이터가 큰 경우의 대안은
`pg_rewind`(`wal_log_hints = on`)이지만 스크립트에는 넣지 않았다.

**금지 사항**: 펜싱된 이전 primary를 재구축 없이 primary로 되살리지 않는다. 클러스터가 살아 있는 동안 `db`의 볼륨을
지우고 `up` 하지 않는다(`node.sh`가 피어 응답을 감지해 initdb를 거부하지만, 피어가 모두 내려가 있으면 막지 못한다).

### 5.5 standby 장애

- 1대: 조치 불필요(쓰기 계속). 원인 제거 후 `docker start <컨테이너>` 또는 `scripts/failover.sh rebuild <노드>`.
- 2대: 쓰기가 멈춘다(fail closed). 1대라도 `docker start` 하거나 재구축하면 대기 중이던 커밋이 응답된다.

### 5.6 전체 재시작

`docker compose stop` → `docker compose up -d`. 각 노드는 자기 볼륨의 상태(`standby.signal` 유무)대로 올라오므로
역할이 유지된다.

### 5.7 측정 결과 (E10, `scripts/test_failover.sh`)

결과: `reports/E10_availability/results.json`, `summary.md`. 쓰기 부하(4 스레드)가 KMS 키 생성 + 등록 트랜잭션,
철회 트랜잭션 + KMS 키 파기를 반복하는 동안 primary를 `docker kill`하고, `watch 1 3`이 감지·승격한다.
RTO = kill → 새 primary에서 첫 등록·철회 응답(감지 포함).

측정일 2026-10-08, postgres (PostgreSQL) 16.11, Docker 29.3.0, 한 호스트(다른 작업 부하 공존).

| 시나리오 | 응답된 등록 | 응답된 철회 | 실패(미응답) | 응답 후 유실 | RTO 초 | kill 기준 시각(초) |
|---|---|---|---|---|---|---|
| kill primary db | 2466 | 1228 | 100 | 0 | 7.29 | 감지 3.3 / 승격 6.2 / 동기 standby 7.3 |
| kill primary db-standby1 | 2350 | 1172 | 126 | 0 | 9.28 | 감지 4.3 / 승격 7.4 / 동기 standby 9.4 |
| kill primary db | 2267 | 1132 | 68 | 0 | 6.30 | 감지 3.0 / 승격 5.3 / 동기 standby 6.4 |
| kill standby db | 3414 | 1705 | 0 | 0 | - | 장애 10.9초 동안 응답된 커밋 2528건, 최대 지연 0.17초 |
| kill both standbys | 1500 | 749 | 0 | 0 | - | 장애 8.5초 동안 응답된 커밋 0건, 최대 지연 9.81초 |

- RTO: 최소 6.30초, 중앙값 7.29초, 최대 9.28초 (목표 900초). 감지(1초 × 3회)가 그중 3~4초를 차지한다. 사람이 판단하는 수동 페일오버여도 15분 안에 충분히 들어온다.
- RPO = 0: 모든 라운드에서 응답된 등록 11997건과 철회 5986건을 새 primary에서 다시 확인했다(피험자 상태, 템플릿 4개, outbox 이벤트, KMS로 모든 템플릿 복호화, 철회 피험자의 키 파기, 감사 해시 체인). 유실 0건.
- standby 1대 장애: 실패 0건, 쓰기 계속. standby 2대 장애: 장애 동안 응답된 커밋 0건(대기 후 standby 복귀 시 응답) = fail closed.
- 실패(미응답) 건은 장애 중 거절된 요청과 결과 불명 요청이다. 결과 불명 건은 검사에서 제외된다.
- 한계: 한 호스트라 네트워크 분할·디스크 손실은 시험하지 않았다. 이전 primary 이외 노드의 동시 장애는 RFP 범위(단일 노드) 밖이다
  (수동 확인: primary + standby 1대 동시 장애 시 남은 standby가 승격되고, 쓰기는 노드를 재구축할 때까지 대기했다가 재구축 후 커밋됨).

---

## 6. 백업, 보존 기간, 삭제(크립토 셰딩)

### 6.1 무엇을 어디에

| 대상 | 방법 | 보관 위치 | 보존 |
|---|---|---|---|
| DB `faceauth` | `pg_dump -Fc` (faceauth_api, standby에서) | 백업 저장소 A | 예: 30일 |
| DB `kms` | `pg_dump -Fc` (kms) | **별도** 저장소 B (접근 권한 분리) | **짧게**, 예: 7일 |
| KMS 마스터 키 | DB 백업에 포함하지 않음 | 비밀 저장소/HSM, 2인 통제 에스크로 | 키 교체 시 폐기 |

```bash
mkdir -p data/backup
docker compose run --rm -u $(id -u):$(id -g) tester sh -c 'PGPASSWORD=$(cat secrets/db_api_password) pg_dump -Fc \
  -d "host=db-standby1,db-standby2,db dbname=faceauth user=faceauth_api target_session_attrs=prefer-standby" \
  -f data/backup/faceauth-$(date +%F).dump'
docker compose run --rm -u $(id -u):$(id -g) tester sh -c 'PGPASSWORD=$(cat secrets/db_kms_password) pg_dump -Fc \
  -d "host=db-standby1,db-standby2,db dbname=kms user=kms target_session_attrs=prefer-standby" \
  -f data/backup/kms-$(date +%F).dump'
```

(개발 환경 예시다. 운영에서는 백업 파일을 즉시 각 저장소로 옮기고 로컬 사본은 지운다. faceauth와 kms 백업을 같은
곳에 두면 키 분리가 무너진다.)

### 6.2 삭제 요청(동의 철회)의 효과

`DELETE /v1/subjects/{id}/face`(API)는 한 트랜잭션에서 피험자 revoke(승인 즉시 차단), 템플릿 비활성 + 암호문 NULL,
`remove` 이벤트를 기록한 뒤 KMS `DELETE /v1/keys/{id}`로 **데이터 키를 파기**한다(wrapped_key = NULL, 묘비 행).
이후 그 피험자의 암호문은 어디에 남아 있든(이전 faceauth 백업, WAL, 디스크 잔여 블록) 복호화할 데이터 키가 없다.

남는 위험과 기간:

- **kms 백업**에는 파기 전 wrapped key가 들어 있다. 마스터 키와 함께라면 옛 faceauth 백업의 템플릿을 풀 수 있다.
  → kms 백업 보존 기간이 곧 "백업에서 완전히 삭제될 때까지의 최대 기간"이다. 그래서 kms 백업은 짧게 보존한다.
- DB 파일·WAL·standby 디스크의 잔여 바이트도 wrapped key(마스터 키로 암호화된 상태)를 담을 수 있다.
- 위 두 경우를 즉시 끝내는 방법은 **마스터 키 교체 후 이전 마스터 키 폐기**(§8)다. 그 시점 이전의 모든 kms 사본은
  무의미해진다. 프로토타입에는 교체 도구가 없다(§8, 후속 과제).

### 6.3 복원 원칙

1. faceauth를 복원한다.
2. kms는 **faceauth 백업보다 오래되지 않은** 상태여야 한다. 더 오래되면 그 사이 등록된 피험자의 키가 없어 재등록이 필요하다.
3. **kms를 파기 이전 시점으로 되돌리지 않는다.** 되돌리면 철회된 피험자의 키가 되살아난다. 가능하면 kms는
   복원하지 않고 살아 있는 클러스터의 상태(3중 복제)를 쓴다. 그러면 faceauth를 과거 백업으로 복원해도 철회된
   피험자의 템플릿은 복호화되지 않는다(search가 null을 받아 건너뜀). 크립토 셰딩이 "백업 복원으로 철회가
   되살아나는" 문제를 막는 방식이다.
4. 복원 후 faceauth에서 `status = 'revoked'`인 모든 피험자에 대해 KMS 키 파기를 다시 호출한다(멱등).
5. search 인덱스 재구성(§5.3), 감사 체인 검증.

---

## 7. 모델 배포, 활성화, 롤백

- 모델마다 `data/models/<dir>/MODEL.json`(이름, 버전, 종류, 파일, 파일별 sha256, 출처, 라이선스, 파라미터).
- `scripts/build_registry.py`: 모든 MODEL.json을 모아 실제 파일 해시와 대조하고(불일치·중복·경로 이탈 시 실패,
  `--skip-invalid`이면 그 모델만 제외), `registry.json`(각 MODEL.json의 해시 포함)을 `release.key`(Ed25519)로 서명해
  `registry.json.sig`를 만든다. `--dsn`이면 `model_versions`에 새 (이름, 버전, 주 파일 sha256)을 넣고, 활성 버전이 없는
  이름만 가장 낮은 버전을 활성화한다(감사 로그). 같은 이름·버전의 해시가 바뀌면 거부한다(버전을 올려야 함).
- `faceauth/common/registry.py`: `load_registry()`는 서명과 **모든** 파일·MODEL.json 해시를 확인한 뒤에만 항목을
  돌려준다. `active_entries()`는 DB의 활성 버전이 서명된 레지스트리에 있고 sha256이 일치할 때만 돌려준다.
  어느 하나라도 어긋나면 `RegistryError`(engine은 해당 모델을 로드하지 않음 = fail closed).

절차:

```bash
# 새 버전 추가 (예: recog 2)
cp -r new_model data/models/<dir_v2>   # MODEL.json 포함
docker run --rm -u $(id -u):$(id -g) -v $PWD:/app -w /app faceauth:dev python scripts/build_registry.py
docker compose run --rm -u $(id -u):$(id -g) tester python scripts/build_registry.py --dsn "<FA_DB_DSN>"
# 활성화 (API, 감사 로그 기록)
curl -X POST -H "Authorization: Bearer $(cat secrets/admin_token)" .../v1/admin/models/<name>/activate  # 본문: API 문서
# 반영
curl -X POST http://engine:8001/v1/admin/reload
curl -X POST http://search-1:8002/v1/admin/reload   # 인식 모델을 바꾼 경우, search 복제본마다
```

- **롤백** = 이전 버전 활성화 + reload. 검출·자세·랜드마크·PAD 모델은 즉시 롤백된다.
- **인식 모델**은 벡터가 버전별로 호환되지 않으므로 비교하지 않는다(`templates.model_version`). 원본 영상을 보관하지
  않으므로 새 인식 모델로 바꾸려면 **재등록 계획**이 필요하고, 롤백은 이전 버전 템플릿이 남아 있는 피험자에게만
  유효하다. 새 버전으로만 등록한 피험자는 롤백 후 재등록해야 한다.
- 개발용 SQL 대체(감사 로그를 직접 남김):
  ```sql
  BEGIN;
  UPDATE model_versions SET active = false WHERE name = '<name>' AND active;
  UPDATE model_versions SET active = true, activated_at = now() WHERE name = '<name>' AND version = '<version>';
  INSERT INTO audit_log (actor, event, detail) VALUES ('ops', 'model_activated', '{"name": "<name>", "version": "<version>"}');
  COMMIT;
  ```

---

## 8. 비밀정보 교체

`init_secrets.py`는 기존 파일을 덮어쓰지 않는다. 개발 환경에서 교체하려면 해당 파일을 다른 곳으로 옮기고 다시 실행한 뒤
아래 절차를 따른다.

| 비밀정보 | 절차 | 영향 |
|---|---|---|
| DB 역할 비밀번호 | primary에서 `docker exec <primary> psql -c "ALTER ROLE faceauth_api PASSWORD '...'"` → 파일 교체 → 해당 서비스 재시작 | 재시작 동안 해당 서비스 요청 실패(fail closed) |
| `db_repl_password` | `ALTER ROLE replicator ...` → 파일 교체 → standby 재시작(`/tmp/pgpass`는 기동 시 생성) | 재시작 중인 standby만 일시 이탈 |
| `kms_token_api` / `kms_token_search` | 파일 교체 → kms와 api / search 재시작 | 짧은 503 |
| KMS 마스터 키 | **온라인 교체 미지원(후속 과제)**. 절차: KMS 중지 → 모든 wrapped key를 옛 키로 풀고 새 키로 감싸는 단일 트랜잭션 → 비밀 교체 → KMS 기동 → 옛 키 폐기 | 교체 동안 등록·로딩 불가. 폐기 후 옛 kms 백업은 무의미 (§6.2) |
| `release.key` | 새 키로 `build_registry.py` 재서명 → engine에 새 `release.pub` 배포 → reload | 서명 불일치 시 engine이 모델 로드 거부 |
| `approval.key` | 교체 후 api 재시작, 검증 측에 새 pub 배포 | 토큰 TTL 30초: 그 동안 발급된 토큰 거부 가능 |
| 단말 인증서 / CA | 재발급 → `devices.cert_fingerprint` 갱신 | 해당 단말 재등록 |
| `admin_token` | 파일 교체 → api 재시작 | 관리 호출 재인증 |

---

## 9. KMS 운영 정보

- API: `POST /v1/encrypt`(api 토큰), `POST /v1/decrypt`(search 토큰), `DELETE /v1/keys/{subject_id}`(api 토큰),
  `GET /healthz`. 본문 msgpack. 토큰 없음/불일치 401, 범위 밖 403, 잘못된 입력 400, DB 장애 503(fail closed),
  파기된 피험자에 encrypt 410. decrypt는 피험자 키가 없거나 파기됐거나 인증 태그가 맞지 않으면 그 항목만 `null`.
- 데이터 키는 피험자별 AES-256-GCM, 마스터 키로 감싸 저장(AAD = subject_id). 감싼 키 행을 다른 피험자로 옮기거나
  암호문을 다른 피험자 이름으로 보내면 복호화되지 않는다(테스트로 확인).
- 키 파기는 동기 커밋 뒤에 응답한다(응답된 파기는 RPO = 0). 묘비 행이 같은 피험자 키의 재생성을 막는다.
- 복호화는 요청마다 DB에서 키 상태를 읽는다(풀린 키 캐시 없음). 파기된 키는 어느 KMS 프로세스에서도 즉시 거부된다.
- 연결: 워커 프로세스당 DB 연결 1개. 유지하던 연결이 끊겨 있으면(예: 페일오버) 새 연결로 한 번 재시도한다(모든 KMS DB
  작업은 멱등이므로 결과가 불명확한 커밋도 다시 실행해도 안전). 새 연결도 실패하면 503이고 다음 요청에서 다시 연결한다.
  연결 풀을 쓰지 않는 이유: 풀의 지수 백오프 재연결이 RTO를 늘린다.
- 처리량(`python -m faceauth.kms.bench`, 2026-10-08, 이 호스트): 100,000명 × 4 템플릿(2048바이트) = 400,000건을
  3.35초에 복호화 (약 119,000건/초, 배치 8192, 동시 요청 4, uvicorn worker 4). 400만 템플릿 기준 약 34초.
  encrypt(등록 경로, 32 스레드)는 약 665회/초. search 기본값(배치 1000, 동시 8)으로 25,000명 × 4 = 100,000건:
  약 88,000~105,000건/초(2회), 400만 건 약 38~46초. 각 수치는 공용 호스트에서 1회 측정값이다.

---

## 10. 개발 전용 항목 (운영 전환 시 교체 필수)

- 자체 서명 개발 CA와 모든 개인 키·토큰·마스터 키가 `secrets/`의 파일이다(컨테이너가 읽도록 0644, 디렉터리 0700).
  운영: HSM/KMS 기반 CA, 비밀 저장소, 마스터 키는 HSM 안에서만 사용.
- 3개 DB 노드가 한 호스트에 있다. `docker kill`로 노드 장애를 흉내 낸다. 운영: 노드를 서로 다른 호스트·랙(가용 영역)에 두고
  디스크·네트워크 분할 장애도 시험해야 한다.
- 펜싱은 docker 명령(재시작 정책, kill, `standby.signal`)이다. 운영: 전원 차단(IPMI/클라우드 API) 등 STONITH.
- 장애 감지는 docker 호스트의 단일 `watch` 프로세스다. 운영: 별도 감시 노드 여러 대의 다수결(또는 Patroni 등 검증된 HA 관리자).
- 내부 네트워크의 DB 연결에 TLS가 없다(`sslmode=prefer`이나 서버 SSL 미설정). 운영: `ssl = on` + 서버 인증서,
  클라이언트 `sslmode=verify-full`.
- WAL 아카이빙·PITR이 없다. 백업은 일 단위 `pg_dump` 예시뿐이다.
- 마스터 키 교체 도구가 없다(§8).
- Docker 내장 DNS가 멈춘 컨테이너 이름 조회에 최대 10초를 쓰는 경우가 관찰되었다. DB·KMS 서비스에는
  `dns_opt: [timeout:1, attempts:1]`을 넣었다. 다중 호스트 DSN을 쓰는 다른 서비스에도 같은 설정이 필요하다.

---

## 11. 애플리케이션 개발자 주의사항 (DB를 쓰는 모든 서비스)

1. **COMMIT을 취소하지 않는다.** 동기 복제 대기 중 취소되면 Postgres는 `WARNING: canceling wait for synchronous
   replication ...`과 함께 성공을 반환한다. 이 커밋은 standby에 없을 수 있다. 쓰기 트랜잭션에 `statement_timeout`을
   걸지 말고, 응답 시간 제한이 필요하면 연결을 끊는다(결과 불명 = 실패로 처리 = 거절).
2. 쓰기 DSN: `host=db,db-standby1,db-standby2 ... target_session_attrs=read-write connect_timeout=3 keepalives_idle=5
   keepalives_interval=2 keepalives_count=3`. 연결 오류는 재연결로 처리하고, 그 요청은 실패(fail closed)로 끝낸다.
3. compose 서비스에 `dns_opt: ["timeout:1", "attempts:1"]`.
4. search: 페일오버 후 인덱스를 재구성한다(§5.3). 자동화하려면 폴링마다 `applied_event_id` 행이 여전히 같은
   `template_id`로 존재하는지 확인하고, 아니면 재구성한다.
5. KMS 키 생성 응답 직후 primary가 죽어도 안전하다. 암호문을 저장하는 이후의 커밋이 같은 WAL 스트림에서 더 뒤에 있으므로,
   그 커밋이 응답되었다면 키도 이미 standby에 있다.
