"""Create development secrets under ./secrets (idempotent: existing files are never overwritten).

    python scripts/init_secrets.py [--devices N]

Dev only: a real deployment uses an HSM/KMS-backed CA, device keys in secure hardware (Design §7).
secrets/ is 0700 on the host; files are 0644 so non-root container users (postgres) can read the bind mounts.
"""
import argparse
import datetime
import os
import pathlib
import secrets

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

S = pathlib.Path(__file__).resolve().parent.parent / "secrets"
PEM = serialization.Encoding.PEM


def write(path: pathlib.Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(data)
        path.chmod(0o644)


def priv_pem(k) -> bytes:
    return k.private_bytes(PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())


def pub_pem(k) -> bytes:
    return k.public_key().public_bytes(PEM, serialization.PublicFormat.SubjectPublicKeyInfo)


def ed25519_pair(priv: pathlib.Path, pub: pathlib.Path) -> None:
    if not priv.exists():
        k = ed25519.Ed25519PrivateKey.generate()
        write(priv, priv_pem(k))
        write(pub, pub_pem(k))


def _cert(subject_cn, issuer_name, pub, sign_key, *, ca=False, eku=None, sans=()):
    now = datetime.datetime.now(datetime.timezone.utc)
    b = (x509.CertificateBuilder()
         .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, subject_cn)]))
         .issuer_name(issuer_name).public_key(pub).serial_number(x509.random_serial_number())
         .not_valid_before(now - datetime.timedelta(days=1)).not_valid_after(now + datetime.timedelta(days=825))
         .add_extension(x509.BasicConstraints(ca=ca, path_length=0 if ca else None), critical=True))
    if eku:
        b = b.add_extension(x509.ExtendedKeyUsage([eku]), critical=False)
    if sans:
        b = b.add_extension(x509.SubjectAlternativeName([x509.DNSName(s) for s in sans]), critical=False)
    return b.sign(sign_key, hashes.SHA256())


def ca():
    if not (S / "ca.key").exists():
        key = ec.generate_private_key(ec.SECP256R1())
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "faceauth dev CA")])
        write(S / "ca.key", priv_pem(key))
        write(S / "ca.crt", _cert("faceauth dev CA", name, key.public_key(), key, ca=True).public_bytes(PEM))
    key = serialization.load_pem_private_key((S / "ca.key").read_bytes(), None)
    return key, x509.load_pem_x509_certificate((S / "ca.crt").read_bytes())


def issue(prefix: pathlib.Path, cn: str, client: bool, ca_key, ca_cert, sans=()) -> None:
    if prefix.with_suffix(".crt").exists():
        return
    key = ec.generate_private_key(ec.SECP256R1())
    eku = ExtendedKeyUsageOID.CLIENT_AUTH if client else ExtendedKeyUsageOID.SERVER_AUTH
    cert = _cert(cn, ca_cert.subject, key.public_key(), ca_key, eku=eku, sans=sans)
    write(prefix.with_suffix(".key"), priv_pem(key))
    write(prefix.with_suffix(".crt"), cert.public_bytes(PEM))


def make_device(device_id: str, ca_key=None, ca_cert=None) -> pathlib.Path:
    """mTLS client cert + Ed25519 packet-signing key for one simulated terminal."""
    if ca_key is None:
        ca_key, ca_cert = ca()
    d = S / "devices"
    issue(d / device_id, device_id, True, ca_key, ca_cert)
    ed25519_pair(d / f"{device_id}.ed25519", d / f"{device_id}.ed25519.pub")
    return d


def main(n_devices: int) -> None:
    S.mkdir(exist_ok=True)
    S.chmod(0o700)
    ca_key, ca_cert = ca()
    issue(S / "gateway", "gateway", False, ca_key, ca_cert, sans=("gateway", "localhost"))
    for i in range(1, n_devices + 1):
        make_device(f"dev-{i:03d}", ca_key, ca_cert)
    for name in ("release", "approval", "account"):
        ed25519_pair(S / f"{name}.key", S / f"{name}.pub")
    write(S / "kms_master.key", secrets.token_bytes(32))
    for name in ("kms_token_api", "kms_token_search", "admin_token", "db_postgres_password",
                 "db_api_password", "db_ro_password", "db_kms_password", "db_repl_password"):
        write(S / name, secrets.token_urlsafe(32).encode())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", type=int, default=4)
    main(ap.parse_args().devices)
