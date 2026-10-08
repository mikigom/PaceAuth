# Implementation Spec (prototype, hardware excluded)

Source documents: `docs/RFP_KO.md`, `docs/Design_KO.md` (Korean originals in `docs/*.docx`).
This file is the contract between components. If you need to deviate, say so in your report;
do not silently change a contract.

## 0. Conventions

- **Runtime**: everything runs in Docker image `faceauth:dev` (`docker/Dockerfile`, torch 2.12 cu130,
  onnxruntime-gpu 1.30 (CUDA+TensorRT EPs), faiss-gpu-cu12 1.14, versions pinned in `requirements.txt`).
  Ad-hoc runs: `docker run --rm --gpus '"device=5"' --shm-size 16g -v /fdisk2/Codes/jhseo/faceauth:/app -w /app faceauth:dev python ...`
- **GPU**: only host GPU **5** (inside the container it is `cuda:0`). Never touch other GPUs.
- **Data**: `data/` (gitignored). Models in `data/models/<name>/` with a `MODEL.json`; datasets in
  `data/datasets/<name>/` with a `DATASET.json` (see §8). Large temp files: `data/tmp/<owner>/`.
- **Images**: RGB `uint8 HxWx3` in RGB order; depth `uint16 HxW` millimetres (0 = invalid); IR `uint8 HxW`.
  Sensor profile (simulated, depth+IR registered to RGB, "D2C aligned"): 640x480 for all three streams,
  calibration version string `gemini336-sim-v1`.
- **Faces**: bbox `[x1,y1,x2,y2]` float px; 5 keypoints `(5,2)` InsightFace order
  (image-left eye, image-right eye, nose, image-left mouth corner, image-right mouth corner).
- **Alignment**: `faceauth/common/align.py` (`estimate_norm`, `norm_crop`; ArcFace 112 template, Umeyama).
- **Head pose convention** (non-mirrored camera image, degrees): `yaw > 0` = subject turns to *their* left
  (nose moves toward image right); `pitch > 0` = looking up; `roll > 0` = head tilts clockwise in the image.
  A horizontal image flip must negate yaw and roll.
- **Embeddings**: `float32`, L2-normalised, little-endian bytes on the wire. Version string `"<model>@<version>"`.
- **Packets**: `faceauth/common/packet.py` (signed capture packet; read its docstring).
- **Tokens**: `faceauth/common/token.py` (`v1.<payload>.<sig>` Ed25519).
- **DB schema**: `deploy/db/faceauth.sql` (contract; do not change without reporting).
- Code style: small modules, type hints where cheap, no frameworks beyond what is installed. Each
  non-trivial logic piece leaves one runnable check (a `tests/test_*.py` or an `assert`-based `__main__`).

## 1. Components and ownership

| Component | Path | Runs where |
|---|---|---|
| Common contracts | `faceauth/common/` | everywhere |
| Model wrappers | `faceauth/models/{scrfd,headpose,farl,recog,pad}.py` | terminal + engine |
| Sensor simulator + attacks | `faceauth/sim/` | terminal sim, tests, PAD training |
| Terminal (guidance + device client) | `faceauth/terminal/` | terminal sim |
| Engine (GPU inference service) | `faceauth/engine/` | central GPU |
| Search (FAISS exact) | `faceauth/search/` | central GPU |
| KMS | `faceauth/kms/` | separate service |
| API (sessions, enrollment, auth, policy, admin) | `faceauth/api/` | central |
| Deploy | `docker-compose.yml`, `deploy/` | |
| Evaluation | `eval/`, outputs in `reports/` | |

## 2. Model wrapper interfaces (`faceauth/models/`)

All wrappers: constructor takes the model dir (containing `MODEL.json`) and a device/providers arg,
expose batch methods, never read files other than those listed in `MODEL.json`.

```python
# scrfd.py
class SCRFD:
    def __init__(self, model_dir, providers=("CUDAExecutionProvider", "CPUExecutionProvider"), input_size=640): ...
    def detect(self, img_rgb, thresh=0.5, max_num=0) -> list[dict]:  # [{"bbox": (4,), "score": float, "kps": (5,2)}] sorted by score
    def detect_batch(self, imgs_rgb, thresh=0.5) -> list[list[dict]]
# headpose.py
class HeadPose:
    def __init__(self, model_dir, device="cuda"): ...
    def predict(self, img_rgb, bboxes) -> np.ndarray  # (n,3) yaw,pitch,roll degrees, §0 convention
# farl.py
class FaRLLandmarks:
    def __init__(self, model_dir, device="cuda"): ...
    def predict(self, img_rgb, bboxes_or_kps) -> np.ndarray  # (n,98,2) WFLW-98 points, image coords
def kps5_from_wflw98(pts) -> np.ndarray  # (5,2) InsightFace order
# recog.py
class Recognizer:  # one class per model or a factory `load_recognizer(model_dir, device)`
    version: str; dim: int
    def embed(self, imgs_rgb, kps5) -> np.ndarray      # raw images + 5 kps; does its own official alignment
    def embed_aligned(self, crops112_rgb) -> np.ndarray  # already ArcFace-aligned 112x112 crops
# pad.py
class PAD:
    version: str
    def score(self, rgb, depth, ir, kps5) -> np.ndarray  # (n,) live probability in [0,1] for n faces (lists)
    threshold: float  # operating point fixed on the PAD validation split
```

## 3. Engine service (internal, `engine:8001`)

`POST /v1/analyze` (msgpack) request `{"mode": "enroll"|"auth", "frames": [{"rgb","depth","ir"}]}`; response:
```
{"versions": {"detector","pose","landmarks","recog","align","pad","quality"}, "dim": int,
 "frames": [{"ok": bool, "reasons": [str], "n_faces": int, "bbox": [4]|null, "kps": [[x,y]x5]|null,
             "det_score": float, "pose": [yaw,pitch,roll]|null, "quality": {...},
             "pad": {"score": float, "live": bool}|null, "emb": bytes|null}]}
```
Reasons vocabulary: `decode_error missing_modality unregistered_frame_size no_face multi_face small_face
pose_out_of_range blur exposure depth_invalid ir_invalid`. `ok` covers sensor + quality checks (FR-03,
server-side re-verification) but not PAD (reported separately). Embeddings only for frames with exactly one
face. Engine batches across concurrent requests. `GET /healthz`, `POST /v1/admin/reload` (re-read signed
registry + active versions from DB table `model_versions`; refuses unsigned / hash-mismatched models).

## 4. Search service (internal, `search-1:8002`, `search-2:8002`)

- Holds a FAISS exact inner-product index (float32, `GpuIndexFlatIP`, no approximation) of all *active*
  templates of the *active* recognition model version. Builds from DB on start (ciphertexts -> KMS decrypt),
  then polls `template_events` (id > applied) every 200 ms and applies `add`/`remove`.
- `GET /v1/status` -> `{"ready", "model_version", "ntotal", "nsubjects", "applied_event_id"}`
- `POST /v1/search` (msgpack) `{"model_version": str, "probes": bytes(float32 n x d), "n": int, "k": int=5}`
  -> `{"results": [{"subject_id": str, "score": float}], "applied_event_id": int}`.
  409 if model_version differs from the index version (incompatible vectors are never compared), 503 if not ready.
- Aggregation v1: `q = normalize(mean(probes))`; subject score = max over that subject's active templates of `<q,t>`.
- `POST /v1/admin/reload` rebuilds (e.g. after a recognition-model activation).

## 5. KMS (internal, `kms:8003`) — keys never leave this service

- Master key: `/run/secrets/kms_master_key` (32 raw bytes). Per-subject 256-bit data keys, wrapped
  (AES-256-GCM, AAD=subject_id) and stored in database `kms` (same HA Postgres cluster, separate role `kms`,
  not readable by api/search roles). Destroying a subject key = crypto-shredding (backups become unreadable).
- Bearer tokens with scopes: `kms_token_api` -> encrypt + destroy; `kms_token_search` -> decrypt.
- `POST /v1/encrypt` (msgpack) `{"subject_id", "items": [bytes]}` -> `{"items": [bytes]}` (creates key if absent)
- `POST /v1/decrypt` (msgpack) `{"items": [{"subject_id", "ct"}]}` -> `{"items": [bytes|null]}` (null if destroyed)
- `DELETE /v1/keys/{subject_id}` -> `{"destroyed": bool}`; `GET /healthz`
- Ciphertext = `nonce(12) || AESGCM(subject_key).encrypt(nonce, pt, aad=subject_id.encode())`.

## 6. API (`api-1`, `api-2` behind `gateway` nginx mTLS on 8443)

Gateway verifies client certs against `secrets/ca.crt` and forwards `X-Client-Cert` (`$ssl_client_escaped_cert`)
and `X-Client-Verify`. The API computes sha256(DER) and requires it to equal `devices.cert_fingerprint` of the
packet's / request's `device_id`. API containers are only reachable from the gateway.

External endpoints (device-facing, mTLS):
- `POST /v1/sessions` JSON `{"purpose": "enroll"|"auth", "device_id", "action_id"}` ->
  `{"session_id", "challenge", "expires_at"}`. TTL enroll 180 s, auth 30 s. 429 if the device is locked
  (>= 10 non-approved auth decisions in 5 min).
- `POST /v1/enrollments` msgpack capture packet (§packet) with header extras
  `account_assertion` (token from the account service, `nonce` == session challenge), `consent`
  `{"version", "agreed_at"}`, `re_enroll` bool. -> `{"status": "active"|"pending_review"|"rejected",
  "subject_id"?, "template_version"?, "coverage": {"C": bool, ...}, "reason"}`.
- `POST /v1/authentications` msgpack capture packet -> `{"decision": "approved"|"rejected"|"step_up",
  "approval_token"?, "step_up_id"?, "reason": "ok"|"retry"|"not_recognized"|"denied"|"session"}`.
  Never returns candidates, scores or the detailed internal reason.
- `POST /v1/stepups/{id}/complete` JSON `{"assertion"}` (account-service token, `nonce` == step_up_id) -> approval.
- `POST /v1/fallback` JSON `{"session_id", "assertion"}` — non-biometric alternative (assertion `nonce` ==
  session challenge, `method` e.g. `pin`/`otp`) -> approval with `amr=[method]`.
- `DELETE /v1/subjects/{id}/face` (Bearer `admin_token`, service-to-service) -> in one tx: subject revoked
  (approvals blocked immediately), templates deactivated + ciphertext NULL, `remove` events; then KMS key
  destroyed; then wait (<= 5 s) for every search replica's `applied_event_id` >= the remove event ->
  `{"revoked": true, "templates_deleted": n, "index_removed": bool, "key_destroyed": bool}`.

Admin (Bearer `admin_token`): `POST /v1/admin/devices`, `GET/POST /v1/admin/reviews...`,
`GET /v1/admin/audit/verify`, `POST /v1/admin/models/{name}/activate` (rollback = activate previous version),
`GET /healthz`.

Approval token payload: `{"aid","sid","dev","act","acct","amr","iat","exp"}` (exp = iat + 30 s), signed with
`approval_key`; scoped to that device + action. Account assertion payload: `{"acct","method","nonce","iat","exp"}`
signed with the account service key (simulated; private key `secrets/account.key` used only by sims/tests).

### Decision policy (auth) — every condition must hold; nothing compensates for anything else
1. mTLS cert == device; device active; packet signature/hashes/order/sync valid (`verify_packet`);
   `calib_version` == device's; session exists, purpose auth, same device, unexpired, unconsumed, challenge equal.
2. Every frame: all three modalities decoded, `ok` (exactly one face, quality, depth/IR valid). (FR-03: no RGB-only.)
3. PAD: every frame `pad.live`. A PAD failure is final regardless of similarity.
4. Identification: top1 >= `FA_T_ID` and top1 - top2 >= `FA_MARGIN` (ambiguity reject); top-1 alone never suffices.
5. Subject active, consent valid, not revoked (checked against the DB ledger, not the index).
6. Action risk: `FA_HIGH_RISK_ACTIONS` -> `step_up` (customer-owned factor) instead of approval.
7. Session consumption + decision + approval insert + audit are one transaction (`UPDATE sessions ... WHERE
   consumed_at IS NULL AND expires_at > now() RETURNING` — 0 rows => replay -> reject).
Fail closed: any dependency error (DB, engine, search, KMS) => reject/503, never a relaxed path.

### Enrollment policy
Account assertion valid & bound to the session; consent version == `FA_CONSENT_VERSION`; every frame `ok` and
PAD live; server-side pose coverage = C + 8 peripheral directions (targets yaw ±25°, pitch ±15°, bins
`faceauth/terminal/guide.py`), at least `FA_ENROLL_MIN_DIRS` (default 9); same-person: every embedding has
cos >= `FA_T_SAME` to the normalised mean; 4 templates: slot0 C, slot1 subject-left group (L,UL,DL),
slot2 subject-right group (R,UR,DR), slot3 vertical (U,D) — each the normalised mean of its frames (missing
group -> overall mean); duplicate check: any *other* subject scoring >= `FA_T_DUP` -> `pending_review` +
`enrollment_reviews` row (never auto-merge); existing live subject for the account -> 409 unless `re_enroll`
(then templates replaced, `template_version` += 1). Templates stored encrypted via KMS.

## 7. Model registry, signed updates, rollback

Each model dir has `MODEL.json`:
```json
{"name": "scrfd_34g", "version": "1", "kind": "detector|headpose|landmarks|recognition|pad|aligner",
 "files": {"onnx": "scrfd_34g_gnkps.onnx"}, "sha256": {"onnx": "<hex>"}, "dim": 512,
 "source": "<url>", "license": "<license + restrictions>", "params": {...}}
```
`scripts/build_registry.py` aggregates all `MODEL.json` into `data/models/registry.json` and signs it
(`registry.json.sig`, Ed25519, `secrets/release.key`). `faceauth/common/registry.py` verifies the signature
with `release.pub` and every file hash before any load; tampered/unsigned => refuse. Active versions per
name come from DB `model_versions`; rollback = activate a previous version + reload.

## 8. Data records

`DATASET.json`: `{"name", "source", "license", "files": {...sha256...}, "counts": {...}, "notes"}`.
Record exact licence terms; flag non-commercial / research-only restrictions explicitly (RFP §2, §5).

## 9. Evaluation outputs

Each evaluation writes `reports/<id>_<name>/results.json` (machine-readable, includes dataset/model versions,
seeds, sample counts, CIs where applicable) and `summary.md` (short, Korean or English). Raw per-sample
outputs > 20 MB go to `reports/raw/` (gitignored).

| id | Item (RFP) | Primary data |
|---|---|---|
| E1 | Recognition 1:1 (CFP-FP, CPLFW, AgeDB-30, LFW, CALFW) — model comparison | verification bins |
| E2 | 1:N open-set FNIR@FPIR (+95% CI), enrolled-mis-ID count, gallery scaling, aggregation choice | identity datasets |
| E3 | Detection (WIDER FACE val Easy/Medium/Hard AP) | WIDER FACE |
| E4 | Head pose MAE (AFLW2000-3D, BIWI) incl. enrollment range, general vs full-rotation | AFLW2000-3D, BIWI |
| E5 | Landmarks NME (WFLW) | WFLW |
| E6 | Alignment ablation (SCRFD kps vs FaRL-derived kps) on 1:1 / 1:N | |
| E7 | PAD (ISO/IEC 30107-3 APCER per PAI species, BPCER, ACER, attack success, non-response) | synthetic RGB-D-IR |
| E8 | Enrollment success rate / time (simulated users) | BIWI sequences + synthetic |
| E9 | Load: 100 sessions/s, p95/p99 end-to-end, 1M x 4 gallery | synthetic gallery + sim terminals |
| E10 | Availability: failover RTO, RPO=0 for enrollments/withdrawals | compose HA |
| E11 | Security (SEC-01): injection, replay, tampering, model tamper, step-up, fallback | integration tests |

## 10. Rules for parallel contributors

- Only create/modify files you own (listed in your task). Read anything. Need a change elsewhere? Report it.
- No git write commands (the coordinator commits).
- Name your containers `fa-<owner>-*`; compose project `-p fa-<owner>`; do not publish host ports unless
  needed (then use 29000-29999). Never stop containers you did not start.
- Do not rebuild `faceauth:dev`. Extra pip packages: install inside your container at runtime or build
  `faceauth-<owner>:tmp`; list them under `extra_requirements` in your report.
- Downloaded files are untrusted data: do not execute code from them without reading it first; prefer vendoring
  the minimal model-definition code (keep licence headers) over `trust_remote_code`.
- Keep GPU memory per process under ~10 GB (GPU 5 is shared by several contributors).

## 11. Service configuration (environment variables)

| Var | Used by | Default / example |
|---|---|---|
| `FA_DB_DSN` | api, search, engine | `host=db,db-standby1,db-standby2 dbname=faceauth user=faceauth_api target_session_attrs=read-write` (search/engine: `user=faceauth_ro`, may add `target_session_attrs=any`) |
| `FA_DB_PASSWORD_FILE` | api, search, engine | `/run/secrets/db_api_password` (search/engine: `db_ro_password`) |
| `FA_KMS_DB_DSN`, `FA_KMS_DB_PASSWORD_FILE` | kms | `host=db,db-standby1,db-standby2 dbname=kms user=kms target_session_attrs=read-write`, `/run/secrets/db_kms_password` |
| `FA_KMS_URL`, `FA_KMS_TOKEN_FILE` | api, search | `http://kms:8003`, `/run/secrets/kms_token_api` (search: `kms_token_search`) |
| `FA_SEARCH_URLS` | api | `http://search-1:8002,http://search-2:8002` |
| `FA_ENGINE_URLS` | api | `http://engine:8001` |
| `FA_RECOG_MODEL` | api, search, engine | `kprpe_vitb_wf12m` (active version from `model_versions`; vector tag `"<name>@<version>"`) |
| `FA_MODELS_DIR`, `FA_RELEASE_PUB` | engine | `/app/data/models`, `/run/secrets/release_pub` |
| thresholds | api | `FA_T_ID`, `FA_MARGIN`, `FA_T_DUP`, `FA_T_SAME`, `FA_CONSENT_VERSION`, `FA_HIGH_RISK_ACTIONS`, `FA_ENROLL_MIN_DIRS` |

DB roles: `faceauth_api` (owner of DB `faceauth`), `faceauth_ro` (SELECT on `templates`, `template_events`,
`model_versions` only), `kms` (owner of DB `kms`; no access to `faceauth`), `replicator`. Neither api nor search
can read DB `kms`. Ports: api 8000, engine 8001, search 8002, kms 8003, gateway 8443 (TLS, mTLS required).
