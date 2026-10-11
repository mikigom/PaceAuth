"""Aggregate data/models/*/MODEL.json into data/models/registry.json and sign it (docs/IMPLEMENTATION.md §7).

    python scripts/build_registry.py [--models data/models] [--key secrets/release.key] [--skip-invalid]
        [--dsn "host=db,db-standby1,db-standby2 dbname=faceauth user=faceauth_api target_session_attrs=read-write"
         --password-file secrets/db_api_password]

Every listed file must exist and match the sha256 in its MODEL.json, else exit 1 (--skip-invalid leaves such
models out of the registry instead; they then cannot be loaded). The registry also pins each MODEL.json's hash.
--dsn: insert new (name, version, sha256) rows into model_versions (an existing row with a different hash is an
error: bump the version), and activate the lowest version of any name that has no active version (audited).
"""
import argparse
import collections
import datetime
import json
import os
import pathlib
import re
import sys

from cryptography.hazmat.primitives.serialization import load_pem_private_key

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from faceauth.common.registry import RegistryError, model_file, sha256_file  # noqa: E402


def entry(mj: pathlib.Path) -> dict:
    m = json.loads(mj.read_text())
    missing = [k for k in ("name", "version", "kind", "files", "sha256") if not m.get(k)]
    if missing:
        raise ValueError(f"missing {missing}")
    if set(m["files"]) != set(m["sha256"]):
        raise ValueError("files and sha256 list different roles")
    for role, name in m["files"].items():
        if not re.fullmatch(r"[0-9a-fA-F]{64}", str(m["sha256"][role])):
            raise ValueError(f"{role}: sha256 missing or malformed (fill it in once the file is verified)")
        if sha256_file(model_file(mj.parent, name)) != m["sha256"][role].lower():
            raise ValueError(f"{role} file {name}: sha256 differs from MODEL.json")
    return {**m, "version": str(m["version"]), "sha256": {r: h.lower() for r, h in m["sha256"].items()},
            "dir": mj.parent.name, "primary": next(iter(m["files"])), "model_json_sha256": sha256_file(mj)}


def build(models: pathlib.Path, skip_invalid: bool) -> list[dict]:
    entries, errors = [], []
    for mj in sorted(models.glob("*/MODEL.json")):
        try:
            entries.append(entry(mj))
        except (OSError, ValueError, RegistryError) as e:
            errors.append(f"{mj}: {e}")
    n = collections.Counter((e["name"], e["version"]) for e in entries)
    errors += [f"{e['dir']}: duplicate {e['name']}@{e['version']}" for e in entries if n[e["name"], e["version"]] > 1]
    for err in errors:
        print(("skipped: " if skip_invalid else "error: ") + err, file=sys.stderr)
    if errors and not skip_invalid:
        sys.exit(1)
    return [e for e in entries if n[e["name"], e["version"]] == 1]


def sign(models: pathlib.Path, entries: list[dict], key_path: str) -> None:
    key = load_pem_private_key(pathlib.Path(key_path).read_bytes(), None)
    raw = (json.dumps({"format": 1, "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                       "models": entries}, indent=1, sort_keys=True) + "\n").encode()
    for name, data in (("registry.json", raw), ("registry.json.sig", key.sign(raw))):
        tmp = models / (name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, models / name)


def natural(v: str):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", v)]


def register(dsn: str, password: str, entries: list[dict]) -> None:
    import psycopg
    from psycopg.types.json import Jsonb
    with psycopg.connect(dsn, password=password) as conn:  # one transaction
        for e in entries:
            sha = e["sha256"][e["primary"]]
            row = conn.execute("SELECT sha256 FROM model_versions WHERE name = %s AND version = %s",
                               (e["name"], e["version"])).fetchone()
            if row and row[0] != sha:
                sys.exit(f"error: {e['name']}@{e['version']} is registered with another sha256; bump the version")
            if not row:
                conn.execute("INSERT INTO model_versions (name, version, sha256) VALUES (%s, %s, %s)",
                             (e["name"], e["version"], sha))
        activated = []
        for name in sorted({e["name"] for e in entries}):
            if not conn.execute("SELECT 1 FROM model_versions WHERE name = %s AND active", (name,)).fetchone():
                v = min((e["version"] for e in entries if e["name"] == name), key=natural)
                conn.execute("UPDATE model_versions SET active = true, activated_at = now() "
                             "WHERE name = %s AND version = %s", (name, v))
                activated.append({"name": name, "version": v})
        for a in activated:  # audit rows last: the chain lock is held until commit
            conn.execute("INSERT INTO audit_log (actor, event, detail) VALUES ('build_registry', 'model_activated', %s)",
                         (Jsonb(a),))
    print(f"model_versions: {len(entries)} entries, activated {activated}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="data/models", type=pathlib.Path)
    ap.add_argument("--key", default="secrets/release.key")
    ap.add_argument("--skip-invalid", action="store_true")
    ap.add_argument("--dsn")
    ap.add_argument("--password-file", default="secrets/db_api_password")
    a = ap.parse_args()
    entries = build(a.models, a.skip_invalid)
    sign(a.models, entries, a.key)
    print(f"signed {a.models / 'registry.json'}: " + ", ".join(f"{e['name']}@{e['version']}" for e in entries),
          file=sys.stderr)
    if a.dsn:
        register(a.dsn, pathlib.Path(a.password_file).read_text().strip(), entries)


if __name__ == "__main__":
    main()
