"""KMS: token scopes, round trip, subject/AAD binding, crypto-shredding, fail closed.

DB-backed tests need the platform Postgres (role kms) and use a throwaway schema in DB kms:
    docker run --rm --network <project>_internal -u $(id -u):$(id -g) -e HOME=/tmp -v $PWD:/app -w /app \
        faceauth:dev pytest -q tests/test_kms.py
Env: FA_TEST_KMS_DSN (default: the platform multi-host DSN), FA_TEST_KMS_PASSWORD_FILE (default secrets/db_kms_password).
Without that database they are skipped; the offline tests (scopes, 401, fail closed, AAD) always run.
"""
import os
import pathlib
import secrets
import uuid

import msgpack
import psycopg
import pytest
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from fastapi.testclient import TestClient

from faceauth.kms.main import DEFAULT_DSN, create_app, seal, unseal

ROOT = pathlib.Path(__file__).resolve().parent.parent
MASTER, API, SEARCH = os.urandom(32), b"api-" + secrets.token_bytes(8).hex().encode(), b"search-token"


def hdr(tok: bytes) -> dict:
    return {"Authorization": "Bearer " + tok.decode(), "Content-Type": "application/msgpack"}


def post(client, path, tok, obj):
    return client.post(path, content=msgpack.packb(obj), headers=hdr(tok))


def items(r) -> list:
    assert r.status_code == 200, r.text
    return msgpack.unpackb(r.content, raw=False)["items"]


def enc(client, sid, pts):
    return items(post(client, "/v1/encrypt", API, {"subject_id": sid, "items": pts}))


def dec(client, pairs):
    return items(post(client, "/v1/decrypt", SEARCH, {"items": [{"subject_id": s, "ct": c} for s, c in pairs]}))


# ---------------------------------------------------------------- offline (no database)
@pytest.fixture
def offline():  # points at a closed port: every DB access fails
    return TestClient(create_app(MASTER, API, SEARCH, "host=127.0.0.1 port=1 dbname=kms user=kms connect_timeout=1"))


def test_missing_or_unknown_token_is_401(offline):
    body = msgpack.packb({"subject_id": str(uuid.uuid4()), "items": [b"x"]})
    for h in ({}, {"Authorization": "Bearer nope"}, {"Authorization": API.decode()}, {"Authorization": "Bearer "}):
        assert offline.post("/v1/encrypt", content=body, headers=h).status_code == 401
        assert offline.post("/v1/decrypt", content=body, headers=h).status_code == 401
    assert offline.delete(f"/v1/keys/{uuid.uuid4()}", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_scopes_are_enforced(offline):
    sid = str(uuid.uuid4())
    assert post(offline, "/v1/encrypt", SEARCH, {"subject_id": sid, "items": [b"x"]}).status_code == 403
    assert post(offline, "/v1/decrypt", API, {"items": [{"subject_id": sid, "ct": b"0" * 40}]}).status_code == 403
    assert offline.delete(f"/v1/keys/{sid}", headers=hdr(SEARCH)).status_code == 403


def test_database_down_fails_closed(offline):
    sid = str(uuid.uuid4())
    assert post(offline, "/v1/encrypt", API, {"subject_id": sid, "items": [b"x"]}).status_code == 503
    assert post(offline, "/v1/decrypt", SEARCH, {"items": [{"subject_id": sid, "ct": b"0" * 40}]}).status_code == 503
    assert offline.delete(f"/v1/keys/{sid}", headers=hdr(API)).status_code == 503
    assert offline.get("/healthz").status_code == 503


def test_bad_input_is_400(offline):
    h = hdr(API)
    assert offline.post("/v1/encrypt", content=b"\xc1not msgpack", headers=h).status_code == 400
    assert post(offline, "/v1/encrypt", API, {"subject_id": "not-a-uuid", "items": [b"x"]}).status_code == 400
    assert post(offline, "/v1/encrypt", API, {"subject_id": str(uuid.uuid4()), "items": ["str"]}).status_code == 400
    assert post(offline, "/v1/decrypt", SEARCH, {"items": [{"subject_id": str(uuid.uuid4())}]}).status_code == 400


def test_bad_config_is_refused():
    with pytest.raises(ValueError):
        create_app(b"short", API, SEARCH, "")
    with pytest.raises(ValueError):
        create_app(MASTER, API, API, "")
    with pytest.raises(ValueError):
        create_app(MASTER, b"", SEARCH, "")


def test_seal_binds_aad():
    a = AESGCM(os.urandom(32))
    blob = seal(a, b"template", b"subject-1")
    assert unseal(a, blob, b"subject-1") == b"template"
    with pytest.raises(InvalidTag):
        unseal(a, blob, b"subject-2")
    with pytest.raises(InvalidTag):
        unseal(AESGCM(os.urandom(32)), blob, b"subject-1")


# ---------------------------------------------------------------- with the platform database
@pytest.fixture(scope="module")
def db():
    dsn = os.environ.get("FA_TEST_KMS_DSN", DEFAULT_DSN)
    try:
        pw = pathlib.Path(os.environ.get("FA_TEST_KMS_PASSWORD_FILE", ROOT / "secrets/db_kms_password")).read_text().strip()
        admin = psycopg.connect(dsn, password=pw, autocommit=True, connect_timeout=3)
    except (OSError, psycopg.Error) as e:
        pytest.skip(f"kms database not reachable ({type(e).__name__}); see module docstring")
    schema = "kms_test_" + secrets.token_hex(4)
    admin.execute(f"CREATE SCHEMA {schema}")
    admin.execute(f"SET search_path = {schema}")
    admin.execute((ROOT / "deploy/db/kms.sql").read_text())
    client = TestClient(create_app(MASTER, API, SEARCH,
                                   f"{dsn} application_name={schema} options='-c search_path={schema}'", pw))
    yield client, admin
    admin.execute(f"DROP SCHEMA {schema} CASCADE")
    admin.close()


def test_round_trip(db):
    client, _ = db
    sid = str(uuid.uuid4())
    pts = [os.urandom(2048) for _ in range(4)]
    cts = enc(client, sid, pts)
    assert [len(c) for c in cts] == [12 + 2048 + 16] * 4 and len(set(cts)) == 4
    assert dec(client, [(sid, c) for c in cts]) == pts
    # same key for the same subject; the id is canonicalised (AAD identical however it is written)
    (ct,) = enc(client, sid.upper(), [b"x"])
    assert dec(client, [(sid, ct)]) == [b"x"]
    assert client.get("/healthz").status_code == 200


def test_wrong_subject_or_tampering_gives_null(db):
    client, _ = db
    a, b = str(uuid.uuid4()), str(uuid.uuid4())
    (ct,) = enc(client, a, [b"secret"])
    enc(client, b, [b"other"])
    tampered = ct[:-1] + bytes([ct[-1] ^ 1])
    got = dec(client, [(b, ct), (str(uuid.uuid4()), ct), (a, tampered), (a, ct[:5]), (a, ct)])
    assert got == [None, None, None, None, b"secret"]


def test_wrapped_key_is_bound_to_its_subject(db):
    client, admin = db
    a, b = str(uuid.uuid4()), str(uuid.uuid4())
    (ct_a,) = enc(client, a, [b"secret"])
    enc(client, b, [b"other"])
    # someone with write access to DB kms copies a's wrapped key into b's row: unwrapping fails (aad = b)
    admin.execute("UPDATE subject_keys SET wrapped_key = (SELECT wrapped_key FROM subject_keys WHERE subject_id = %s) "
                  "WHERE subject_id = %s", (a, b))
    assert dec(client, [(b, ct_a)]) == [None]
    assert dec(client, [(a, ct_a)]) == [b"secret"]


def test_destroy_is_crypto_shredding(db):
    client, admin = db
    sid = str(uuid.uuid4())
    cts = enc(client, sid, [b"t0", b"t1"])
    r = client.delete(f"/v1/keys/{sid}", headers=hdr(API))
    assert r.status_code == 200 and r.json() == {"destroyed": True}
    assert dec(client, [(sid, c) for c in cts]) == [None, None]
    wrapped, destroyed_at = admin.execute(
        "SELECT wrapped_key, destroyed_at FROM subject_keys WHERE subject_id = %s", (sid,)).fetchone()
    assert wrapped is None and destroyed_at is not None
    assert post(client, "/v1/encrypt", API, {"subject_id": sid, "items": [b"x"]}).status_code == 410
    assert client.delete(f"/v1/keys/{sid}", headers=hdr(API)).json() == {"destroyed": True}  # idempotent
    # a subject withdrawn before it ever had a key can never get one (tombstone)
    new = str(uuid.uuid4())
    assert client.delete(f"/v1/keys/{new}", headers=hdr(API)).status_code == 200
    assert post(client, "/v1/encrypt", API, {"subject_id": new, "items": [b"x"]}).status_code == 410


def test_dead_kept_connection_is_replaced(db):
    client, admin = db  # e.g. the primary died and a standby was promoted while this worker was idle
    sid = str(uuid.uuid4())
    (ct,) = enc(client, sid, [b"before"])
    schema = admin.execute("SELECT current_setting('search_path')").fetchone()[0]
    killed = admin.execute("SELECT count(pg_terminate_backend(pid)) FROM pg_stat_activity WHERE application_name = %s",
                           (schema,)).fetchone()[0]
    assert killed == 1
    assert dec(client, [(sid, ct)]) == [b"before"]  # no 503: one retry on a fresh connection
