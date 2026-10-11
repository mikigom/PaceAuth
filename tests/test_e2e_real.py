"""E11 security / FR end-to-end tests against the REAL stack (gateway mTLS -> API -> real engine/PAD -> search -> DB/KMS).

    docker compose run --rm -u $(id -u):$(id -g) -e HOME=/tmp -e FA_E2E=1 tester python -m pytest -q tests/test_e2e_real.py

Subjects are CelebA identities rendered by SensorSim (gemini336-sim-v1); the enrolled subject's templates are computed by
the engine and stored through the enrollment write path (scripts/loadgen.insert_user). Every case records the device
answer and the internal reason (sessions.reason) into reports/E11_security/results.json.
"""
import json
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
pytestmark = pytest.mark.skipif(os.environ.get("FA_E2E") != "1", reason="needs the running stack (FA_E2E=1)")

DSN = "host=db,db-standby1,db-standby2 dbname=faceauth user=faceauth_api target_session_attrs=read-write"
RESULTS = {}


def record(case, **kw):
    RESULTS[case] = kw
    out = ROOT / "reports/E11_security"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(RESULTS, indent=1, default=str))


@pytest.fixture(scope="module")
def world():
    import cv2
    import httpx
    import psycopg
    import loadgen
    from init_secrets import make_device
    from faceauth.models.scrfd import SCRFD
    from faceauth.sim.sensor import SensorSim
    from faceauth.terminal.client import SECRETS, DeviceClient

    det, sim = SCRFD(ROOT / "data/models/scrfd_34g"), SensorSim(seed=11)
    people = loadgen.identities(seed=1234, n_users=3, n_imp=0, per=12)
    def photo(name):
        img = cv2.cvtColor(cv2.imread(str(loadgen.CELEBA / "img_celeba" / name)), cv2.COLOR_BGR2RGB)
        d = det.detect(img)
        if not d:
            return None
        return img, max(d, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))["kps"]
    faces = {cid: [p for p in map(photo, names) if p is not None] for cid, names, _ in people}
    (a, b, c) = list(faces)
    http = httpx.Client(timeout=120)
    def render(kind, img_kps, seed, n=3):
        from faceauth.sim.capture import sequence
        from faceauth.common.packet import encode_frame
        frames, _ = sequence(sim, kind, img_kps[0], img_kps[1], n=n, seed=seed, distance_mm=600)
        return [encode_frame(f["rgb"], f["depth"], f["ir"]) for f in frames]
    def usable(frames):  # engine view in auth mode (pose window too): every frame ok, single face, live
        an = loadgen.analyze(http, frames, mode="auth")
        return all(f["ok"] and f["pad"] and f["pad"]["live"] for f in an["frames"])
    a_live = [(p, render("live", p, 100 + i)) for i, p in enumerate(faces[a])]
    good = [(p, fr) for p, fr in a_live if usable(fr)]
    assert len(good) >= 5, f"subject {a}: only {len(good)} usable photos"
    pw = (ROOT / "secrets/db_api_password").read_text().strip()
    conn = psycopg.connect(DSN, password=pw, autocommit=True)
    kms = httpx.Client(base_url="http://kms:8003", headers={"Authorization": "Bearer " + (ROOT / "secrets/kms_token_api").read_text().strip()})
    mv = conn.execute("SELECT name || '@' || version FROM model_versions WHERE name = 'kprpe_vitb_wf12m' AND active").fetchone()[0]
    admin = {"Authorization": "Bearer " + (SECRETS / "admin_token").read_text().strip()}
    def withdraw_leftovers():   # the direct insert skips enrollment's duplicate check: never leave a second copy of A
        for (old,) in conn.execute("SELECT id FROM subjects WHERE account_id LIKE %s AND status <> 'revoked'",
                                   (f"e11-{a}-%",)).fetchall():
            dev0.http.delete(f"/v1/subjects/{old}/face", headers=admin, extensions={"sni_hostname": "gateway"}).raise_for_status()
    tpl, align = loadgen.templates_from(http, [fr for _, fr in good[:4]], mv)
    account = f"e11-{a}-{uuid.uuid4().hex[:6]}"
    def new_device():
        dev_id = f"e11-{uuid.uuid4().hex[:8]}"
        make_device(dev_id)
        dev = DeviceClient("https://gateway:8443", dev_id)
        body = {"device_id": dev_id, "public_key_pem": (SECRETS / "devices" / f"{dev_id}.ed25519.pub").read_text(),
                "cert_pem": (SECRETS / "devices" / f"{dev_id}.crt").read_text(), "calib_version": "gemini336-sim-v1"}
        r = dev.http.post("/v1/admin/devices", json=body, extensions={"sni_hostname": "gateway"},
                          headers={"Authorization": "Bearer " + (SECRETS / "admin_token").read_text().strip()})
        assert r.status_code in (200, 201), r.text
        return dev
    dev0 = new_device()
    withdraw_leftovers()
    sid = loadgen.insert_user(conn, kms, mv, align, account, tpl)
    last = conn.execute("SELECT max(id) FROM template_events").fetchone()[0]
    deadline = time.time() + 60                                      # both replicas indexed the new subject
    while time.time() < deadline:
        st = [http.get(f"http://search-{i}:8002/v1/status").json()["applied_event_id"] for i in (1, 2)]
        if min(st) >= last:
            break
        time.sleep(0.5)
    yield dict(new_device=new_device, conn=conn, http=http, render=render, sid=sid, account=account, probe=good[4][0],
               probe_frames=good[4][1], spare=[p for p, _ in good[5:]] or [good[4][0]], b=faces[b][0], c=faces[c][0],
               sim=sim, secrets=SECRETS)
    withdraw_leftovers()
    conn.close()


@pytest.fixture
def dev(world):
    """A fresh registered terminal per test: the per-device lockout (10 rejections / 5 min) must not leak between cases."""
    return world["new_device"]()


def auth(w, dev, frames, ts=None, key=None, action="door-open", mutate=None):
    """-> (device answer, internal reason, session, packet). key: sign with another Ed25519 key; mutate(frames) after
    signing."""
    import msgpack
    from faceauth.common.packet import build_packet
    from faceauth.terminal.client import synced_ts
    s = dev.create_session("auth", action)
    start = int(time.time() * 1000)
    ts = ts or synced_ts([1, 4, 8], start)
    pkt = build_packet(key or dev.key, session_id=s["session_id"], challenge=s["challenge"], device_id=dev.device_id,
                       calib_version=dev.calib_version, frames=frames, ts_ms=ts, captured_at_ms=start)
    if mutate:
        p = msgpack.unpackb(pkt, raw=False)
        p["frames"] = mutate(p["frames"])
        pkt = msgpack.packb(p)
    r = dev.http.post("/v1/authentications", content=pkt, headers={"Content-Type": "application/msgpack"},
                      extensions={"sni_hostname": "gateway"})
    out = r.json() if r.headers.get("content-type", "").startswith("application/json") else msgpack.unpackb(r.content, raw=False)
    why = w["conn"].execute("SELECT reason FROM sessions WHERE id = %s", (s["session_id"],)).fetchone()[0]
    return out, why, s, pkt


def test_genuine_approved(world, dev):
    out, why, _, _ = auth(world, dev, world["probe_frames"])
    record("genuine", decision=out["decision"], reason=out["reason"], internal=why)
    assert out["decision"] == "approved" and "approval_token" in out


@pytest.mark.parametrize("kind", ["print", "replay", "mask_rigid", "mask_silicone"])
def test_presentation_attacks_of_enrolled_face_rejected(world, dev, kind):
    """The attack shows the ENROLLED person's face: only PAD (or the sensor checks) can stop it."""
    res = []
    for seed in range(5):
        out, why, _, _ = auth(world, dev, world["render"](kind, world["probe"], 500 + seed))
        res.append((out["decision"], why))
    record(f"attack_{kind}", n=len(res), approved=sum(d == "approved" for d, _ in res),
           internal=sorted({w for _, w in res}))
    assert all(d != "approved" for d, _ in res)


def test_impostor_not_recognized(world, dev):
    out, why, _, _ = auth(world, dev, world["render"]("live", world["b"], 7))
    record("impostor", decision=out["decision"], reason=out["reason"], internal=why)
    assert out["decision"] == "rejected" and why in ("no_match", "ambiguous")


def test_multi_face_rejected(world, dev):
    from faceauth.common.packet import encode_frame
    from faceauth.sim.capture import multi_face
    frames = []
    for seed in range(3):
        o = multi_face(world["sim"], [world["probe"], world["c"]], seed=seed)
        frames.append(encode_frame(o["rgb"], o["depth"], o["ir"]))
    out, why, _, _ = auth(world, dev, frames)
    record("multi_face", decision=out["decision"], reason=out["reason"], internal=why)
    assert out["decision"] == "rejected" and "multi_face" in why


@pytest.mark.parametrize("missing", ["depth", "ir", "depth+ir"])
def test_missing_modality_never_falls_back_to_rgb(world, dev, missing):
    from faceauth.sim.capture import drop
    frames = [drop(f, *missing.split("+")) for f in world["probe_frames"]]
    out, why, _, _ = auth(world, dev, frames)
    record(f"missing_{missing}", decision=out["decision"], reason=out["reason"], internal=why)
    assert out["decision"] == "rejected"


@pytest.mark.parametrize("fill", [None, "zero", "luma"])
def test_rgb_only_injection_rejected(world, dev, fill):
    from faceauth.common.packet import decode_frame
    from faceauth.sim.capture import rgb_only_replay
    frames = [rgb_only_replay(decode_frame(f)[0], fill) for f in world["probe_frames"]]
    out, why, _, _ = auth(world, dev, frames)
    record(f"rgb_injection_{fill or 'absent'}", decision=out["decision"], internal=why)
    assert out["decision"] == "rejected"


def test_unsynchronised_frames_rejected(world, dev):
    from faceauth.sim.capture import desync
    from faceauth.terminal.client import synced_ts
    out, why, _, _ = auth(world, dev, world["probe_frames"], ts=desync(synced_ts([1, 4, 8], int(time.time() * 1000))))
    record("desync", decision=out["decision"], internal=why)
    assert out["decision"] == "rejected"


def test_injection_without_device_key_rejected(world, dev):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    out, why, _, _ = auth(world, dev, world["probe_frames"], key=Ed25519PrivateKey.generate())
    record("attacker_key", decision=out["decision"], internal=why)
    assert out["decision"] == "rejected"


def test_frame_tampered_after_signing_rejected(world, dev):
    def swap(frames):
        frames[1]["rgb"] = frames[0]["rgb"]
        return frames
    out, why, _, _ = auth(world, dev, world["probe_frames"], mutate=swap)
    record("tampered_frame", decision=out["decision"], internal=why)
    assert out["decision"] == "rejected"


def test_replayed_packet_rejected(world, dev):
    out, _, _, pkt = auth(world, dev, world["probe_frames"])
    assert out["decision"] == "approved"
    r = dev.http.post("/v1/authentications", content=pkt, headers={"Content-Type": "application/msgpack"},
                               extensions={"sni_hostname": "gateway"})
    again = r.json()
    record("packet_replay", first=out["decision"], replay=again)
    assert again["decision"] == "rejected" and again["reason"] == "session"


def test_perfect_digital_injection_is_a_known_limit(world, dev):
    """Perfect synthetic live frames injected behind the sensor and signed with the (stolen) device key look live to
    PAD; only sensor-origin attestation (frames signed inside the camera module, Design §7, hardware phase) stops them.
    Recorded, not asserted."""
    out, why, _, _ = auth(world, dev, world["render"]("live", world["spare"][0], 900))
    record("perfect_injection_with_device_key", decision=out["decision"], internal=why,
           note="expected approved: PAD cannot detect a perfect digital injection; needs sensor attestation")


def test_high_risk_action_requires_step_up(world, dev):
    from faceauth.terminal.client import make_account_assertion
    out, why, _, _ = auth(world, dev, world["probe_frames"], action="transfer_high")
    assert out["decision"] == "step_up", out
    done = dev.complete_stepup(out["step_up_id"], make_account_assertion(world["account"], out["step_up_id"], "otp"))
    record("step_up", first=out["decision"], completed=done["decision"])
    assert done["decision"] == "approved"


def test_model_tamper_refused_on_reload(world, dev):
    """A model_versions row whose hash differs from the signed registry: the engine refuses and keeps serving."""
    c, http = world["conn"], world["http"]
    admin = {"Authorization": "Bearer " + (world["secrets"] / "admin_token").read_text().strip()}
    name, ver, sha = c.execute("SELECT name, version, sha256 FROM model_versions WHERE name = 'pad_rgbdir' AND active").fetchone()
    try:
        c.execute("UPDATE model_versions SET sha256 = %s WHERE name = %s AND version = %s", ("0" * 64, name, ver))
        r = http.post("http://engine:8001/v1/admin/reload", headers=admin)
        still = http.get("http://engine:8001/healthz").json()
    finally:
        c.execute("UPDATE model_versions SET sha256 = %s WHERE name = %s AND version = %s", (sha, name, ver))
    record("model_tamper_reload", status=r.status_code, detail=r.text[:200], still_serving=still["versions"])
    assert r.status_code == 503 and still["ok"]
    assert http.post("http://engine:8001/v1/admin/reload", headers=admin).status_code == 200   # restored: loads again


def test_withdrawal_blocks_and_shreds(world, dev):
    from faceauth.terminal.client import SECRETS
    c = world["conn"]
    r = dev.http.delete(f"/v1/subjects/{world['sid']}/face", extensions={"sni_hostname": "gateway"},
                        headers={"Authorization": "Bearer " + (SECRETS / "admin_token").read_text().strip()})
    gone = r.json()
    out, why, _, _ = auth(world, dev, world["probe_frames"])
    left = c.execute("SELECT count(*) FROM templates WHERE subject_id = %s AND (active OR ciphertext IS NOT NULL)",
                     (world["sid"],)).fetchone()[0]
    record("withdrawal", delete=gone, auth_after=out["decision"], internal=why, templates_left=left)
    assert gone["revoked"] and gone["key_destroyed"] and gone["index_removed"] and left == 0
    assert out["decision"] == "rejected"


def test_device_lockout_after_repeated_rejections(world, dev):
    """SEC-01 retry limit: after 10 rejected sessions within 5 minutes the terminal cannot open new sessions."""
    import httpx
    statuses = []
    for i in range(12):
        try:
            out, _, _, _ = auth(world, dev, world["render"]("live", world["b"], 70 + i))   # impostor: rejected
            statuses.append(out["decision"])
        except httpx.HTTPStatusError as e:
            statuses.append(e.response.status_code)
    record("device_lockout", sequence=statuses)
    assert statuses[:10] == ["rejected"] * 10 and statuses[10:] == [429, 429]
