"""Signed model registry (docs/IMPLEMENTATION.md §7).

data/models/registry.json (built and signed by scripts/build_registry.py) lists every model version with the
sha256 of each file and of its MODEL.json. Nothing is returned before the Ed25519 signature (release.pub) and
every one of those hashes verify; any failure raises RegistryError (fail closed).

    entries = load_registry("/app/data/models", "/run/secrets/release_pub")   # [entry, ...], entry["path"] = model dir
    active = active_entries(entries, conn)                                    # {name: entry} per model_versions
"""
import hashlib
import json
import pathlib

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import load_pem_public_key


class RegistryError(Exception):
    pass


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def model_file(model_dir: pathlib.Path, name: str) -> pathlib.Path:
    """A file listed in MODEL.json: relative, inside the model dir."""
    p = pathlib.PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or not p.parts:
        raise RegistryError(f"bad file name {name!r}")
    return model_dir / p


def load_registry(models_dir, release_pub) -> list[dict]:
    d = pathlib.Path(models_dir)
    try:
        raw = (d / "registry.json").read_bytes()
        sig = (d / "registry.json.sig").read_bytes()
        pub = load_pem_public_key(pathlib.Path(release_pub).read_bytes())
    except (OSError, ValueError) as e:
        raise RegistryError(f"registry or release key unreadable: {e}") from e
    if not isinstance(pub, Ed25519PublicKey):
        raise RegistryError("release key is not Ed25519")
    try:
        pub.verify(sig, raw)
    except InvalidSignature:
        raise RegistryError("registry signature invalid") from None
    entries = json.loads(raw)["models"]
    for e in entries:
        tag, mdir = f"{e['name']}@{e['version']}", model_file(d, e["dir"])
        try:
            if sha256_file(mdir / "MODEL.json") != e["model_json_sha256"]:
                raise RegistryError(f"{tag}: MODEL.json differs from the signed registry")
            for role, name in e["files"].items():
                if sha256_file(model_file(mdir, name)) != e["sha256"][role]:
                    raise RegistryError(f"{tag}: {role} file {name} hash mismatch")
        except OSError as err:
            raise RegistryError(f"{tag}: {err}") from err
        e["path"] = str(mdir)
    return entries


def active_versions(conn) -> dict[str, tuple[str, str]]:
    """{name: (version, sha256)} of the active rows of model_versions."""
    return {n: (v, h) for n, v, h in conn.execute("SELECT name, version, sha256 FROM model_versions WHERE active").fetchall()}


def active_entries(entries: list[dict], conn) -> dict[str, dict]:
    """Verified registry entry of each active model version; refuses versions that are unsigned or re-hashed."""
    by_key = {(e["name"], e["version"]): e for e in entries}
    out = {}
    for name, (version, sha) in active_versions(conn).items():
        e = by_key.get((name, version))
        if e is None:
            raise RegistryError(f"active {name}@{version} is not in the signed registry")
        if e["sha256"][e["primary"]] != sha:
            raise RegistryError(f"active {name}@{version}: model_versions.sha256 differs from the signed registry")
        out[name] = e
    return out
