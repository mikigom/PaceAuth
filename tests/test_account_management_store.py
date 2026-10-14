"""Account lifecycle and thumbnail storage, using synthetic image fixtures only."""
import concurrent.futures
import sqlite3
import stat
import threading
import time

import cv2
import numpy as np
import pytest
from fastapi import HTTPException

from faceauth.terminal.accounts import Accounts, MAX_PHOTO_BYTES, _password_hash


PASSWORD = "synthetic-password-123"
PHOTO_CONSENT = {"consented_at": "2026-10-14T01:00:00+00:00", "consent_version": "registration-photo-v1"}


@pytest.fixture
def accounts(tmp_path):
    return Accounts(tmp_path / "state" / "accounts.sqlite3")


@pytest.fixture
def jpeg():
    # A color ramp contains no person or other camera input.
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(640, dtype=np.uint16)[None, :] % 256
    image[:, :, 1] = np.arange(480, dtype=np.uint16)[:, None] % 256
    image[:, :, 2] = 90
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def rejects(code, call):
    with pytest.raises(HTTPException) as error:
        call()
    assert error.value.status_code == code


@pytest.mark.parametrize("has_name", [False, True])
def test_migration_preserves_existing_password_session_and_name(tmp_path, has_name):
    path = tmp_path / "legacy.sqlite3"
    salt = b"0123456789abcdef"
    secret = "existing-session-fixture"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE accounts (id TEXT PRIMARY KEY,salt BLOB NOT NULL,password_hash BLOB NOT NULL" +
                   (",display_name TEXT NOT NULL DEFAULT ''" if has_name else "") + ")")
        db.execute("CREATE TABLE sessions (digest TEXT PRIMARY KEY,account_id TEXT NOT NULL,expires REAL NOT NULL)")
        values = ["alice", salt, _password_hash(PASSWORD, salt)] + (["기존 이름"] if has_name else [])
        db.execute("INSERT INTO accounts VALUES (" + ",".join("?" for _ in values) + ")", values)
        db.execute("INSERT INTO sessions VALUES (?,?,?)", (Accounts.digest(secret), "alice", time.time() + 500))
    store = Accounts(path)
    assert store.account(secret) == "alice"
    assert store.account(store.login("alice", PASSWORD)) == "alice"
    profile = store.profile("alice")
    assert profile["display_name"] == ("기존 이름" if has_name else "alice")
    assert profile["enabled"] and not profile["is_admin"]
    assert store.get("alice")["created_at"] > 0
    assert not store.get("alice")["has_photo"]
    assert Accounts(path).get("alice")["created_at"] == store.get("alice")["created_at"]


def test_create_never_grants_admin_or_switches_session(accounts):
    alice = accounts.create("alice", PASSWORD, "앨리스")
    bob = accounts.create("bobby", PASSWORD)
    assert alice["display_name"] == "앨리스"
    assert alice["enabled"] and not alice["is_admin"] and not bob["is_admin"]
    with accounts.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
    assert accounts.profile(None) == {"authenticated": False, "account_id": None, "display_name": None,
                                      "is_admin": False, "enabled": False}
    rejects(409, lambda: accounts.create("alice", PASSWORD))
    rejects(422, lambda: accounts.create("x", PASSWORD))
    rejects(422, lambda: accounts.create("other", "short"))
    assert PASSWORD.encode() not in accounts.path.read_bytes()
    admin = accounts.bootstrap_admin("alice")
    assert admin["is_admin"] and admin["enabled"]
    assert not accounts.get("bobby")["is_admin"]
    rejects(404, lambda: accounts.bootstrap_admin("missing"))


def test_disable_invalidates_sessions_names_and_login_and_reenable_requires_new_login(accounts):
    old_secret = accounts.signup("alice", PASSWORD, "앨리스")
    second_secret = accounts.login("alice", PASSWORD)
    accounts.create("bobby", PASSWORD, "밥")
    assert accounts.names(["alice", "bobby", "missing"]) == {"alice": "앨리스", "bobby": "밥"}
    assert accounts.set_enabled("alice", False)["enabled"] is False
    assert accounts.account(old_secret) is None
    assert accounts.account(second_secret) is None
    assert not accounts.profile("alice")["authenticated"]
    assert accounts.names(["alice", "bobby", "missing"]) == {"bobby": "밥"}
    rejects(403, lambda: accounts.login("alice", PASSWORD))
    rejects(401, lambda: accounts.login("alice", "wrong-password"))
    rejects(403, lambda: accounts.issue("alice"))
    accounts.set_enabled("alice", True)
    assert accounts.account(old_secret) is None
    assert accounts.account(accounts.login("alice", PASSWORD)) == "alice"


def test_account_session_resolution_always_checks_current_account(accounts):
    secret = accounts.signup("alice", PASSWORD)
    with accounts.connect() as db:
        db.execute("UPDATE accounts SET enabled=0 WHERE id='alice'")
    assert accounts.account(secret) is None
    with accounts.connect() as db:
        db.execute("DELETE FROM accounts WHERE id='alice'")
    assert accounts.account(secret) is None


def test_last_enabled_admin_cannot_be_disabled_or_deleted(accounts):
    accounts.create("admin-one", PASSWORD)
    accounts.bootstrap_admin("admin-one")
    rejects(409, lambda: accounts.set_enabled("admin-one", False))
    rejects(409, lambda: accounts.delete("admin-one"))
    accounts.create("admin-two", PASSWORD)
    accounts.bootstrap_admin("admin-two")
    accounts.set_enabled("admin-one", False)
    rejects(409, lambda: accounts.set_enabled("admin-two", False))
    rejects(409, lambda: accounts.delete("admin-two"))
    accounts.delete("admin-one")
    assert accounts.get("admin-two")["is_admin"]


def test_parallel_admin_disables_preserve_one_enabled_admin(accounts):
    for account_id in ("admin-one", "admin-two"):
        accounts.create(account_id, PASSWORD)
        accounts.bootstrap_admin(account_id)
    other_store = Accounts(accounts.path)
    barrier = threading.Barrier(2)

    def disable(store, account_id):
        barrier.wait(timeout=5)
        try:
            store.set_enabled(account_id, False)
            return 200
        except HTTPException as error:
            return error.status_code

    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(disable, accounts, "admin-one"), pool.submit(disable, other_store, "admin-two")]
        assert sorted(f.result(timeout=6) for f in futures) == [200, 409]
    assert len([row for row in accounts.list_accounts() if row["is_admin"] and row["enabled"]]) == 1


def test_delete_clears_password_sessions_photo_and_allows_fresh_identity(accounts, jpeg):
    secret = accounts.signup("alice", PASSWORD)
    accounts.store_photo("alice", jpeg, **PHOTO_CONSENT)
    assert accounts.delete("alice") is True
    assert accounts.get("alice") is None and accounts.account(secret) is None
    assert accounts.list_accounts() == []
    with accounts.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM registration_photos").fetchone()[0] == 0
        assert db.execute("PRAGMA secure_delete").fetchone()[0] == 1
    rejects(404, lambda: accounts.delete("alice"))
    rejects(404, lambda: accounts.photo("alice"))
    new_password = "new-synthetic-password"
    accounts.create("alice", new_password)
    rejects(401, lambda: accounts.login("alice", PASSWORD))
    assert accounts.account(accounts.login("alice", new_password)) == "alice"
    assert accounts.photo("alice") is None


def test_photo_is_small_metadata_free_encrypted_and_survives_restart(accounts, jpeg):
    accounts.create("alice", PASSWORD)
    private_metadata = b"synthetic-private-exif-fixture"
    app1 = b"\xff\xe1" + (len(private_metadata) + 2).to_bytes(2, "big") + private_metadata
    jpeg_with_metadata = jpeg[:2] + app1 + jpeg[2:]
    result = accounts.store_photo("alice", jpeg_with_metadata, **PHOTO_CONSENT)
    assert result["has_photo"] and result["photo_version"] == 1
    assert result["photo_consent_version"] == PHOTO_CONSENT["consent_version"]
    assert result["photo_consented_at"] == PHOTO_CONSENT["consented_at"]
    assert result["photo_updated_at"] > 0
    plaintext = accounts.photo("alice")
    assert private_metadata not in plaintext
    assert cv2.imdecode(np.frombuffer(plaintext, np.uint8), cv2.IMREAD_COLOR).shape == (240, 320, 3)
    assert plaintext not in accounts.path.read_bytes()
    assert jpeg not in accounts.path.read_bytes()
    assert private_metadata not in accounts.path.read_bytes()
    with accounts.connect() as db:
        row = db.execute("SELECT nonce,ciphertext FROM registration_photos WHERE account_id='alice'").fetchone()
        assert len(row["nonce"]) == 12
        assert row["ciphertext"] != plaintext and not row["ciphertext"].startswith(b"\xff\xd8")
    assert stat.S_IMODE(accounts.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(accounts.photo_key_path.stat().st_mode) == 0o600
    assert Accounts(accounts.path).photo("alice") == plaintext


def test_photo_replacement_uses_new_nonce_and_deletion_updates_version(accounts, jpeg):
    accounts.create("alice", PASSWORD)
    assert accounts.photo("alice") is None
    accounts.store_photo("alice", jpeg, **PHOTO_CONSENT)
    with accounts.connect() as db:
        before = db.execute("SELECT nonce,ciphertext FROM registration_photos").fetchone()
    assert accounts.store_photo("alice", jpeg, **PHOTO_CONSENT)["photo_version"] == 2
    with accounts.connect() as db:
        after = db.execute("SELECT nonce,ciphertext FROM registration_photos").fetchone()
    assert before["nonce"] != after["nonce"] and before["ciphertext"] != after["ciphertext"]
    deleted = accounts.delete_photo("alice")
    assert not deleted["has_photo"] and deleted["photo_version"] == 3
    assert deleted["photo_consent_version"] is None and deleted["photo_updated_at"] is None
    assert accounts.photo("alice") is None
    assert accounts.delete_photo("alice")["photo_version"] == 3


def test_encrypted_photo_is_bound_to_account_and_tampering_is_rejected(accounts, jpeg):
    for name in ("alice", "bobby"):
        accounts.create(name, PASSWORD)
    accounts.store_photo("alice", jpeg, **PHOTO_CONSENT)
    with accounts.connect() as db:
        db.execute("INSERT INTO registration_photos (account_id,nonce,ciphertext) "
                   "SELECT 'bobby',nonce,ciphertext FROM registration_photos WHERE account_id='alice'")
    rejects(500, lambda: accounts.photo("bobby"))
    assert accounts.photo("alice") is not None
    with accounts.connect() as db:
        db.execute("UPDATE registration_photos SET ciphertext=? WHERE account_id='alice'", (b"damaged",))
    rejects(500, lambda: accounts.photo("alice"))


def test_photo_requires_explicit_well_formed_consent(accounts, jpeg):
    accounts.create("alice", PASSWORD)
    with pytest.raises(TypeError):
        accounts.store_photo("alice", jpeg)
    for consented_at in (None, "not-a-date", "2026-10-14T01:00:00"):
        rejects(422, lambda: accounts.store_photo("alice", jpeg, consented_at=consented_at))
    rejects(422, lambda: accounts.store_photo("alice", jpeg, consented_at=PHOTO_CONSENT["consented_at"],
                                              consent_version=""))
    assert accounts.photo("alice") is None


def test_rejects_invalid_or_oversized_photos_before_decoder_allocation(accounts, jpeg, monkeypatch):
    accounts.create("alice", PASSWORD)
    rejects(422, lambda: accounts.store_photo("alice", b"not-a-jpeg", **PHOTO_CONSENT))
    rejects(422, lambda: accounts.store_photo("alice", b"\xff\xd8\xff\xe1\x00", **PHOTO_CONSENT))
    rejects(413, lambda: accounts.store_photo("alice", b"x" * (MAX_PHOTO_BYTES + 1), **PHOTO_CONSENT))
    # SOF dimensions alone are enough to reject this 25-megapixel input without
    # asking OpenCV to allocate any image buffer.
    fake_huge = b"\xff\xd8\xff\xc0\x00\x08\x08\x13\x88\x13\x88\x01\xff\xd9"
    monkeypatch.setattr(cv2, "imdecode", lambda *_: pytest.fail("Oversized JPEG reached decoder"))
    rejects(413, lambda: accounts.store_photo("alice", fake_huge, **PHOTO_CONSENT))
    assert not accounts.get("alice")["has_photo"]


def test_missing_photo_key_does_not_silently_replace_existing_key(accounts, jpeg):
    accounts.create("alice", PASSWORD)
    accounts.store_photo("alice", jpeg, **PHOTO_CONSENT)
    accounts.photo_key_path.unlink()
    with pytest.raises(RuntimeError, match="key is missing"):
        Accounts(accounts.path)
    assert not accounts.photo_key_path.exists()


def test_name_update_and_session_limit_preserve_store_contract(accounts):
    secrets = [accounts.signup("alice", PASSWORD)]
    assert accounts.rename("alice", "  새 이름  ")["display_name"] == "새 이름"
    assert accounts.names(["alice"]) == {"alice": "새 이름"}
    rejects(422, lambda: accounts.rename("alice", "bad\nname"))
    rejects(404, lambda: accounts.rename("missing", "이름"))
    secrets.extend(accounts.issue("alice") for _ in range(11))
    assert accounts.account(secrets[0]) is None
    assert sum(accounts.account(secret) == "alice" for secret in secrets) == 10
    accounts.revoke(secrets[-1])
    assert accounts.account(secrets[-1]) is None
    assert secrets[-2].encode() not in accounts.path.read_bytes()
