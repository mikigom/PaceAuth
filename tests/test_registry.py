"""Signed model registry: build + verify, and refusal on any tampering (docs/IMPLEMENTATION.md §7).

The model_versions test needs the platform Postgres (role faceauth_api; throwaway schema), as in tests/test_kms.py;
env FA_TEST_DB_DSN / FA_TEST_DB_PASSWORD_FILE. Without it that test is skipped.
"""
import hashlib
import json
import os
import pathlib
import secrets
import subprocess
import sys

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from faceauth.common.registry import RegistryError, active_entries, load_registry

ROOT = pathlib.Path(__file__).resolve().parent.parent
BUILD = [sys.executable, str(ROOT / "scripts/build_registry.py")]


def keypair(d: pathlib.Path, name: str):
    k = ed25519.Ed25519PrivateKey.generate()
    (d / f"{name}.key").write_bytes(k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                    serialization.NoEncryption()))
    (d / f"{name}.pub").write_bytes(k.public_key().public_bytes(serialization.Encoding.PEM,
                                                                serialization.PublicFormat.SubjectPublicKeyInfo))
    return d / f"{name}.key", d / f"{name}.pub"


def add_model(models: pathlib.Path, dirname: str, name: str, version, files: dict, **extra) -> None:
    d = models / dirname
    d.mkdir(parents=True)
    for fname, data in files.values():
        (d / fname).write_bytes(data)
    (d / "MODEL.json").write_text(json.dumps({
        "name": name, "version": version, "kind": "recognition", "dim": 512, "license": "test", **extra,
        "files": {role: f for role, (f, _) in files.items()},
        "sha256": {role: hashlib.sha256(data).hexdigest() for role, (_, data) in files.items()}}))


@pytest.fixture
def reg(tmp_path):
    models = tmp_path / "models"
    add_model(models, "recog_a", "recog_a", 1, {"onnx": ("a.onnx", os.urandom(4096))})
    add_model(models, "recog_a_v2", "recog_a", "2", {"onnx": ("a2.onnx", os.urandom(4096))})
    add_model(models, "pad", "pad", "1", {"weights": ("w.pt", os.urandom(1000)), "config": ("cfg.yaml", b"x: 1\n")})
    key, pub = keypair(tmp_path, "release")
    r = subprocess.run(BUILD + ["--models", str(models), "--key", str(key)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return models, pub, tmp_path


def test_valid_registry_loads(reg):
    models, pub, _ = reg
    entries = load_registry(models, pub)
    assert sorted((e["name"], e["version"]) for e in entries) == [("pad", "1"), ("recog_a", "1"), ("recog_a", "2")]
    pad = next(e for e in entries if e["name"] == "pad")
    assert pad["primary"] == "weights" and pad["path"] == str(models / "pad") and set(pad["files"]) == {"weights", "config"}


@pytest.mark.parametrize("victim", ["recog_a/a.onnx", "pad/cfg.yaml"])
def test_tampered_model_file_is_refused(reg, victim):
    models, pub, _ = reg
    with open(models / victim, "ab") as f:
        f.write(b"\0")
    with pytest.raises(RegistryError, match="hash mismatch"):
        load_registry(models, pub)


def test_missing_model_file_is_refused(reg):
    models, pub, _ = reg
    (models / "pad/w.pt").unlink()
    with pytest.raises(RegistryError):
        load_registry(models, pub)


def test_tampered_model_json_is_refused(reg):
    models, pub, _ = reg
    mj = models / "recog_a/MODEL.json"
    m = json.loads(mj.read_text())
    m["params"] = {"threshold": 0.0}  # weights untouched, metadata changed
    mj.write_text(json.dumps(m))
    with pytest.raises(RegistryError, match="MODEL.json"):
        load_registry(models, pub)


def test_tampered_registry_is_refused(reg):
    models, pub, _ = reg
    raw = (models / "registry.json").read_text()
    (models / "registry.json").write_text(raw.replace('"version": "2"', '"version": "3"'))
    with pytest.raises(RegistryError, match="signature"):
        load_registry(models, pub)


def test_wrong_key_or_missing_signature_is_refused(reg):
    models, _, tmp = reg
    _, other_pub = keypair(tmp, "other")
    with pytest.raises(RegistryError, match="signature"):
        load_registry(models, other_pub)
    (models / "registry.json.sig").unlink()
    with pytest.raises(RegistryError):
        load_registry(models, other_pub)


def test_build_refuses_hash_mismatch_and_duplicates(tmp_path):
    models = tmp_path / "models"
    add_model(models, "good", "good", "1", {"onnx": ("g.onnx", b"good")})
    add_model(models, "bad", "bad", "1", {"onnx": ("b.onnx", b"bad")})
    add_model(models, "dup1", "dup", "1", {"onnx": ("d.onnx", b"d1")})
    add_model(models, "dup2", "dup", "1", {"onnx": ("d.onnx", b"d2")})
    add_model(models, "pending", "pending", "1", {"weights": ("model.pt", b"w")})
    mj = json.loads((models / "pending/MODEL.json").read_text())
    mj["sha256"]["weights"] = None  # file supplied, hash not yet recorded
    (models / "pending/MODEL.json").write_text(json.dumps(mj))
    (models / "bad/b.onnx").write_bytes(b"changed after MODEL.json was written")
    key, pub = keypair(tmp_path, "release")
    r = subprocess.run(BUILD + ["--models", str(models), "--key", str(key)], capture_output=True, text=True)
    assert r.returncode == 1 and "bad" in r.stderr and "duplicate dup@1" in r.stderr and "sha256 missing" in r.stderr
    assert not (models / "registry.json").exists()
    r = subprocess.run(BUILD + ["--models", str(models), "--key", str(key), "--skip-invalid"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert [e["name"] for e in load_registry(models, pub)] == ["good"]


def test_path_escape_is_refused(tmp_path):
    models = tmp_path / "models"
    (models / "evil").mkdir(parents=True)
    (models / "evil/MODEL.json").write_text(json.dumps({"name": "evil", "version": "1", "kind": "pad",
                                                        "files": {"onnx": "../../etc/passwd"}, "sha256": {"onnx": "0" * 64}}))
    key, _ = keypair(tmp_path, "release")
    r = subprocess.run(BUILD + ["--models", str(models), "--key", str(key)], capture_output=True, text=True)
    assert r.returncode == 1 and "bad file name" in r.stderr


class Rows:  # minimal stand-in for a psycopg connection
    def __init__(self, rows):
        self.rows = rows

    def execute(self, *_):
        return self

    def fetchall(self):
        return self.rows


def test_active_entries_checks_db_against_registry(reg):
    models, pub, _ = reg
    entries = load_registry(models, pub)
    sha = {(e["name"], e["version"]): e["sha256"][e["primary"]] for e in entries}
    got = active_entries(entries, Rows([("recog_a", "2", sha["recog_a", "2"]), ("pad", "1", sha["pad", "1"])]))
    assert got["recog_a"]["version"] == "2" and got["pad"]["dir"] == "pad"
    with pytest.raises(RegistryError, match="not in the signed registry"):
        active_entries(entries, Rows([("recog_a", "9", "00")]))
    with pytest.raises(RegistryError, match="differs"):
        active_entries(entries, Rows([("recog_a", "1", sha["recog_a", "2"])]))


def test_build_registers_and_activates_model_versions(reg):
    psycopg = pytest.importorskip("psycopg")
    models, pub, tmp = reg
    dsn = os.environ.get("FA_TEST_DB_DSN", "host=db,db-standby1,db-standby2 dbname=faceauth user=faceauth_api "
                                           "target_session_attrs=read-write connect_timeout=3")
    pwf = os.environ.get("FA_TEST_DB_PASSWORD_FILE", str(ROOT / "secrets/db_api_password"))
    try:
        conn = psycopg.connect(dsn, password=pathlib.Path(pwf).read_text().strip(), autocommit=True)
    except (OSError, psycopg.Error) as e:
        pytest.skip(f"faceauth database not reachable ({type(e).__name__})")
    schema = "reg_test_" + secrets.token_hex(4)
    conn.execute(f"CREATE SCHEMA {schema}")
    try:
        conn.execute(f"SET search_path = {schema}")
        conn.execute((ROOT / "deploy/db/faceauth.sql").read_text())
        args = BUILD + ["--models", str(models), "--key", str(tmp / "release.key"),
                        "--dsn", f"{dsn} options='-c search_path={schema}'", "--password-file", pwf]
        for _ in range(2):  # idempotent
            r = subprocess.run(args, capture_output=True, text=True)
            assert r.returncode == 0, r.stderr
        rows = conn.execute("SELECT name, version, active FROM model_versions ORDER BY 1, 2").fetchall()
        assert rows == [("pad", "1", True), ("recog_a", "1", True), ("recog_a", "2", False)]
        assert conn.execute("SELECT count(*) FROM audit_log WHERE event = 'model_activated'").fetchone()[0] == 2
        assert conn.execute("SELECT audit_verify()").fetchone()[0] is None
        assert set(active_entries(load_registry(models, pub), conn)) == {"pad", "recog_a"}
        # same name+version with different bytes must get a new version, never a silent re-hash
        add = models / "recog_a_v2/a2.onnx"
        add.write_bytes(b"retrained")
        mj = json.loads((models / "recog_a_v2/MODEL.json").read_text())
        mj["sha256"]["onnx"] = hashlib.sha256(b"retrained").hexdigest()
        (models / "recog_a_v2/MODEL.json").write_text(json.dumps(mj))
        r = subprocess.run(args, capture_output=True, text=True)
        assert r.returncode != 0 and "bump the version" in r.stderr
    finally:
        conn.execute(f"DROP SCHEMA {schema} CASCADE")
        conn.close()
