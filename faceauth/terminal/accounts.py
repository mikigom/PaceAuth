"""Durable pilot accounts, sessions, and encrypted registration thumbnails.

Role checks belong to the HTTP layer. ``bootstrap_admin`` is an explicit local
deployment operation; ordinary account creation never grants administrator rights.
"""
import datetime
import fcntl
import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import threading
import time
import unicodedata
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from fastapi import HTTPException


SESSION_SECONDS = 7200
MAX_PHOTO_BYTES = 2 * 1024 * 1024
MAX_PHOTO_PIXELS = 4_000_000
PHOTO_EDGE = 320
_PHOTO_AAD = b"paceauth-registration-photo-v1:"
_ACCOUNT_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{2,63}$")


def _password_hash(password, salt):
    return hashlib.scrypt(password.encode(), salt=salt, n=16384, r=8, p=1, dklen=32)


def _display_name(value, account_id=None):
    value = unicodedata.normalize("NFC", (value or account_id or "").strip())
    if not value or len(value) > 80 or any(unicodedata.category(c).startswith("C") for c in value):
        raise HTTPException(422, "표시 이름은 제어 문자 없이 1~80자로 입력해 주세요.")
    return value


def _jpeg_dimensions(data):
    """Read JPEG dimensions before decoding to bound decoder allocations."""
    if not isinstance(data, bytes) or not data.startswith(b"\xff\xd8"):
        raise HTTPException(422, "등록 사진은 JPEG 형식이어야 합니다.")
    position = 2
    sof = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
    while position < len(data):
        if data[position] != 0xFF:
            break
        while position < len(data) and data[position] == 0xFF:
            position += 1
        if position >= len(data):
            break
        marker = data[position]
        position += 1
        if marker in {0xDA, 0xD9, 0x00}:
            break
        if marker == 0x01 or 0xD0 <= marker <= 0xD7:
            continue
        if position + 2 > len(data):
            break
        length = int.from_bytes(data[position:position + 2], "big")
        if length < 2 or position + length > len(data):
            break
        if marker in sof:
            if length < 8:
                break
            height = int.from_bytes(data[position + 3:position + 5], "big")
            width = int.from_bytes(data[position + 5:position + 7], "big")
            if width and height:
                return width, height
            break
        position += length
    raise HTTPException(422, "등록 사진을 읽을 수 없습니다.")


def _thumbnail(data):
    if not isinstance(data, bytes):
        raise HTTPException(422, "등록 사진은 JPEG 형식이어야 합니다.")
    if len(data) > MAX_PHOTO_BYTES:
        raise HTTPException(413, "등록 사진은 2 MB 이하여야 합니다.")
    width, height = _jpeg_dimensions(data)
    if width * height > MAX_PHOTO_PIXELS:
        raise HTTPException(413, "등록 사진의 해상도가 너무 큽니다.")
    import cv2
    import numpy as np

    # Explicitly ignore orientation metadata; camera previews already use the
    # agreed orientation. Re-encoding removes EXIF and all other input metadata.
    try:
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8),
                             cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if image is None or image.shape[:2] != (height, width):
            raise HTTPException(422, "등록 사진을 읽을 수 없습니다.")
        if max(width, height) > PHOTO_EDGE:
            ratio = PHOTO_EDGE / max(width, height)
            image = cv2.resize(image, (max(1, round(width * ratio)), max(1, round(height * ratio))),
                               interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise HTTPException(422, "등록 사진을 처리할 수 없습니다.")
        return encoded.tobytes()
    except cv2.error:
        raise HTTPException(422, "등록 사진을 읽을 수 없습니다.") from None


class Accounts:
    """SQLite account store. ``created_at`` is Unix seconds; photos stay encrypted."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("CREATE TABLE IF NOT EXISTS accounts ("
                       "id TEXT PRIMARY KEY, salt BLOB NOT NULL, password_hash BLOB NOT NULL, "
                       "display_name TEXT NOT NULL DEFAULT '')")
            db.execute("CREATE TABLE IF NOT EXISTS sessions ("
                       "digest TEXT PRIMARY KEY, account_id TEXT NOT NULL, expires REAL NOT NULL)")
            columns = {row[1] for row in db.execute("PRAGMA table_info(accounts)")}
            for name, definition in (
                ("display_name", "TEXT NOT NULL DEFAULT ''"),
                ("is_admin", "INTEGER NOT NULL DEFAULT 0"),
                ("enabled", "INTEGER NOT NULL DEFAULT 1"),
                ("created_at", "REAL NOT NULL DEFAULT 0"),
                ("photo_version", "INTEGER NOT NULL DEFAULT 0"),
            ):
                if name not in columns:
                    db.execute(f"ALTER TABLE accounts ADD COLUMN {name} {definition}")
            db.execute("UPDATE accounts SET display_name=id WHERE display_name=''")
            db.execute("UPDATE accounts SET created_at=? WHERE created_at=0", (time.time(),))
            db.execute("CREATE TABLE IF NOT EXISTS registration_photos ("
                       "account_id TEXT PRIMARY KEY REFERENCES accounts(id) ON DELETE CASCADE, "
                       "nonce BLOB NOT NULL, ciphertext BLOB NOT NULL, "
                       "consent_version TEXT NOT NULL DEFAULT '', consented_at TEXT NOT NULL DEFAULT '', "
                       "updated_at REAL NOT NULL DEFAULT 0)")
            photo_columns = {row[1] for row in db.execute("PRAGMA table_info(registration_photos)")}
            for name, definition in (("consent_version", "TEXT NOT NULL DEFAULT ''"),
                                     ("consented_at", "TEXT NOT NULL DEFAULT ''"),
                                     ("updated_at", "REAL NOT NULL DEFAULT 0")):
                if name not in photo_columns:
                    db.execute(f"ALTER TABLE registration_photos ADD COLUMN {name} {definition}")
            db.execute("CREATE INDEX IF NOT EXISTS sessions_account ON sessions(account_id)")
            has_photos = db.execute("SELECT 1 FROM registration_photos LIMIT 1").fetchone() is not None
        self.path.chmod(0o600)
        self.photo_key_path = self.path.with_name(self.path.name + ".photos.key")
        self._photo_cipher = AESGCM(self._photo_key(has_photos))
        self.lock = threading.Lock()
        self.attempts = OrderedDict()

    def _photo_key(self, has_photos):
        if has_photos and not self.photo_key_path.exists():
            raise RuntimeError("Registration photo encryption key is missing")
        fd = os.open(self.photo_key_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            os.fchmod(fd, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX)
            key = os.read(fd, 33)
            if not key and not has_photos:
                key = secrets.token_bytes(32)
                os.write(fd, key)
                os.fsync(fd)
            if len(key) != 32:
                raise RuntimeError("Registration photo encryption key is invalid")
            return key
        finally:
            os.close(fd)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA secure_delete=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _select():
        return ("SELECT a.id AS account_id,a.display_name,a.enabled,a.is_admin,a.created_at,a.photo_version,"
                "(p.account_id IS NOT NULL) AS has_photo,p.updated_at AS photo_updated_at,"
                "p.consent_version AS photo_consent_version,p.consented_at AS photo_consented_at "
                "FROM accounts a LEFT JOIN registration_photos p ON p.account_id=a.id ")

    @staticmethod
    def _metadata(row):
        if row is None:
            return None
        value = dict(row)
        for key in ("enabled", "is_admin", "has_photo"):
            value[key] = bool(value[key])
        return value

    def _required(self, db, account_id):
        row = db.execute(self._select() + "WHERE a.id=?", (account_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "계정을 찾을 수 없습니다.")
        return self._metadata(row)

    @staticmethod
    def _protect_last_admin(db, account):
        if account["is_admin"] and not db.execute(
            "SELECT 1 FROM accounts WHERE is_admin=1 AND enabled=1 AND id<>? LIMIT 1",
            (account["account_id"],),
        ).fetchone():
            raise HTTPException(409, "마지막 활성 관리자 계정은 비활성화하거나 삭제할 수 없습니다.")

    def rate_limit(self, source):
        now = time.monotonic()
        with self.lock:
            recent = [t for t in self.attempts.pop(source, []) if now - t < 300]
            self.attempts[source] = recent
            while len(self.attempts) > 4096:
                self.attempts.popitem(last=False)
            if len(recent) >= 10:
                raise HTTPException(429, "로그인 시도가 많습니다. 잠시 후 다시 시도해 주세요.")
            recent.append(now)

    def create(self, account_id, password, display_name=None):
        """Create an ordinary account without logging the current browser into it."""
        if not isinstance(account_id, str) or not _ACCOUNT_ID.fullmatch(account_id):
            raise HTTPException(422, "계정 ID는 영문 소문자와 숫자, _, ., -로 3~64자 입력해 주세요.")
        if not isinstance(password, str) or not 10 <= len(password) <= 128:
            raise HTTPException(422, "비밀번호는 10~128자로 입력해 주세요.")
        name = _display_name(display_name, account_id)
        salt = secrets.token_bytes(16)
        hashed = _password_hash(password, salt)
        try:
            with self.connect() as db:
                db.execute("INSERT INTO accounts (id,salt,password_hash,display_name,created_at) "
                           "VALUES (?,?,?,?,?)", (account_id, salt, hashed, name, time.time()))
                return self._required(db, account_id)
        except sqlite3.IntegrityError:
            raise HTTPException(409, "이미 사용 중인 테스트 계정입니다.") from None

    def signup(self, account_id, password, display_name=None):
        self.create(account_id, password, display_name)
        return self.issue(account_id)

    def bootstrap_admin(self, account_id):
        """Explicit local deployment only: grant an existing account administrator rights."""
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._required(db, account_id)
            db.execute("UPDATE accounts SET is_admin=1,enabled=1 WHERE id=?", (account_id,))
            return self._required(db, account_id)

    def get(self, account_id):
        with self.connect() as db:
            return self._metadata(db.execute(self._select() + "WHERE a.id=?", (account_id,)).fetchone())

    def list_accounts(self):
        with self.connect() as db:
            return [self._metadata(row) for row in db.execute(self._select() + "ORDER BY a.created_at,a.id")]

    def names(self, account_ids):
        ids = sorted(set(account_ids))
        names = {}
        with self.connect() as db:
            for offset in range(0, len(ids), 500):
                batch = ids[offset:offset + 500]
                names.update(db.execute("SELECT id,display_name FROM accounts WHERE enabled=1 AND id IN (" +
                                        ",".join("?" for _ in batch) + ")", batch).fetchall())
        return names

    def profile(self, account_id):
        row = self.get(account_id) if account_id else None
        return {"authenticated": bool(row and row["enabled"]),
                "account_id": row["account_id"] if row else None,
                "display_name": row["display_name"] if row else None,
                "is_admin": bool(row and row["is_admin"]), "enabled": bool(row and row["enabled"])}

    def rename(self, account_id, display_name):
        name = _display_name(display_name)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._required(db, account_id)
            db.execute("UPDATE accounts SET display_name=? WHERE id=?", (name, account_id))
        return self.profile(account_id)

    def set_enabled(self, account_id, enabled, *, allow_last_admin=False):
        if not isinstance(enabled, bool):
            raise HTTPException(422, "활성 상태를 확인해 주세요.")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            account = self._required(db, account_id)
            if not enabled:
                if not allow_last_admin:
                    self._protect_last_admin(db, account)
                db.execute("DELETE FROM sessions WHERE account_id=?", (account_id,))
            db.execute("UPDATE accounts SET enabled=? WHERE id=?", (int(enabled), account_id))
            return self._required(db, account_id)

    def delete(self, account_id, *, allow_last_admin=False):
        """Delete this local identity, its sessions, password, and encrypted photo.

        The caller must separately revoke the backend biometric templates before
        deleting the local identity; this store has no backend credentials.
        """
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            account = self._required(db, account_id)
            if not allow_last_admin:
                self._protect_last_admin(db, account)
            db.execute("DELETE FROM sessions WHERE account_id=?", (account_id,))
            db.execute("DELETE FROM registration_photos WHERE account_id=?", (account_id,))
            db.execute("DELETE FROM accounts WHERE id=?", (account_id,))
        return True

    def login(self, account_id, password):
        with self.connect() as db:
            row = db.execute("SELECT salt,password_hash,enabled FROM accounts WHERE id=?", (account_id,)).fetchone()
        salt, expected = (row["salt"], row["password_hash"]) if row else (b"\0" * 16, b"\0" * 32)
        valid = hmac.compare_digest(_password_hash(password, salt), expected)
        if not row or not valid:
            raise HTTPException(401, "계정 또는 비밀번호가 맞지 않습니다.")
        if not row["enabled"]:
            raise HTTPException(403, "비활성화된 계정입니다. 관리자에게 문의해 주세요.")
        return self.issue(account_id)

    def issue(self, account_id):
        secret = secrets.token_urlsafe(32)
        now = time.time()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            account = self._required(db, account_id)
            if not account["enabled"]:
                raise HTTPException(403, "비활성화된 계정입니다. 관리자에게 문의해 주세요.")
            db.execute("DELETE FROM sessions WHERE expires<=?", (now,))
            db.execute("INSERT INTO sessions VALUES (?,?,?)", (self.digest(secret), account_id, now + SESSION_SECONDS))
            db.execute("DELETE FROM sessions WHERE account_id=? AND digest NOT IN "
                       "(SELECT digest FROM sessions WHERE account_id=? ORDER BY expires DESC LIMIT 10)",
                       (account_id, account_id))
        return secret

    @staticmethod
    def digest(secret):
        return hashlib.sha256(secret.encode()).hexdigest()

    def account(self, secret):
        if not secret or len(secret) > 128:
            return None
        with self.connect() as db:
            row = db.execute("SELECT s.account_id FROM sessions s JOIN accounts a ON a.id=s.account_id "
                             "WHERE s.digest=? AND s.expires>? AND a.enabled=1",
                             (self.digest(secret), time.time())).fetchone()
        return row[0] if row else None

    def revoke(self, secret):
        with self.connect() as db:
            db.execute("DELETE FROM sessions WHERE digest=?", (self.digest(secret),))

    def store_photo(self, account_id, jpeg, *, consented_at, consent_version="registration-photo-v1"):
        """Store an encrypted, metadata-free thumbnail; plaintext remains in memory."""
        try:
            consent_time = datetime.datetime.fromisoformat(consented_at)
            if consent_time.tzinfo is None or not consent_version or len(consent_version) > 80:
                raise ValueError
        except (ValueError, TypeError):
            raise HTTPException(422, "등록 사진 저장 동의를 확인해 주세요.") from None
        thumbnail = _thumbnail(jpeg)
        nonce = secrets.token_bytes(12)
        ciphertext = self._photo_cipher.encrypt(nonce, thumbnail, _PHOTO_AAD + account_id.encode())
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._required(db, account_id)
            db.execute("INSERT INTO registration_photos "
                       "(account_id,nonce,ciphertext,consent_version,consented_at,updated_at) VALUES (?,?,?,?,?,?) "
                       "ON CONFLICT(account_id) DO UPDATE SET nonce=excluded.nonce,ciphertext=excluded.ciphertext,"
                       "consent_version=excluded.consent_version,consented_at=excluded.consented_at,"
                       "updated_at=excluded.updated_at",
                       (account_id, nonce, ciphertext, consent_version, consented_at, time.time()))
            db.execute("UPDATE accounts SET photo_version=photo_version+1 WHERE id=?", (account_id,))
            return self._required(db, account_id)

    def photo(self, account_id):
        """Return plaintext JPEG bytes to an already-authorized route, or None."""
        with self.connect() as db:
            self._required(db, account_id)
            row = db.execute("SELECT nonce,ciphertext FROM registration_photos WHERE account_id=?", (account_id,)).fetchone()
        if row is None:
            return None
        try:
            return self._photo_cipher.decrypt(row["nonce"], row["ciphertext"], _PHOTO_AAD + account_id.encode())
        except InvalidTag:
            raise HTTPException(500, "등록 사진을 읽을 수 없습니다.") from None

    def delete_photo(self, account_id):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._required(db, account_id)
            result = db.execute("DELETE FROM registration_photos WHERE account_id=?", (account_id,))
            if result.rowcount:
                db.execute("UPDATE accounts SET photo_version=photo_version+1 WHERE id=?", (account_id,))
            return self._required(db, account_id)
