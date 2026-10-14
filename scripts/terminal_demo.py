"""Terminal CLI: rotation-guided enrollment and 1:N authentication against a running stack.

    # inside the compose networks (tester service); --register = admin registration of the device first.
    # Until data/models/registry.json is built and signed, add --unsigned-models (dev only).
    python scripts/terminal_demo.py --gateway https://gateway:8443 --device dev-001 --account acct-0001 \
        --register --source biwi --enroll-seq 01 --auth-seq 01
    python scripts/terminal_demo.py ... --source sim --images face_*.jpg --auth-images probe.jpg
    # Real Femto Bolt. Stop any camera-preview service before taking ownership of USB.
    python scripts/terminal_demo.py ... --source orbbec --fps 30 --register
    # terminal pipeline only (no server): per-frame analysis + latency
    python scripts/terminal_demo.py --smoke img1.jpg img2.jpg ...

Frames: SensorSim (faceauth.sim.sensor, imported lazily) renders RGB-D-IR for subjects taken from BIWI
sequences (faceauth.sim.biwi: real head rotation) or a face image list. --source orbbec uses real,
calibrated RGB-D-IR frames and their device timestamps. Raw frames stay in memory.
Account ownership / step-up assertions remain an explicit CLI demo using secrets/account.key.
Exit code: 0 approved, 1 enrollment failed, 2 authentication not approved.
"""
import argparse
from contextlib import ExitStack, closing
import datetime
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from cryptography.hazmat.primitives import serialization  # noqa: E402

from faceauth.common import token  # noqa: E402
from faceauth.terminal.client import SECRETS, DeviceClient, make_account_assertion, synced_ts  # noqa: E402
from faceauth.terminal.guide import ORDER, Guide  # noqa: E402
from faceauth.terminal.pipeline import Pipeline, select_auth, wire_frame  # noqa: E402

FPS = 30.0


def read_rgb(path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"cannot read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _render(sim, pipe, photo, n, seed, distance_mm) -> list:
    """n simulated sensor frames of a live presentation of `photo` (its own head pose is kept);
    [None] * n when the photo has no face or the simulator cannot fit a head to it."""
    from faceauth.sim.capture import sequence
    dets = pipe.detector.detect(photo)
    if not dets:
        return [None] * n
    d = max(dets, key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]))
    try:
        return sequence(sim, "live", photo, d["kps"], n=n, seed=seed, distance_mm=distance_mm)[0]
    except ValueError:
        return [None] * n


def frame_source(args, phase: str, pipe):
    """Yield (index, frame) for phase 'enroll' | 'auth'; Orbbec frames include real capture metadata.
    Simulation indices use a 30 fps clock; its SensorSim/BIWI imports stay on the simulation branch."""
    if args.source == "orbbec":
        # The caller has already obtained the session challenge. A new pipeline
        # prevents buffered frames from a previous session entering this capture.
        sensor = args.sensor
        deadline = time.monotonic() + (args.timeout if phase == "enroll" else args.auth_timeout)
        try:
            sensor.start()
            if sensor.calib_version != args.calib_version:
                raise RuntimeError("Camera calibration changed after device registration; register the current calibration")
            idx = 0
            while time.monotonic() < deadline and (phase != "auth" or idx < args.auth_max_frames):
                remaining_ms = max(1, int((deadline - time.monotonic()) * 1000))
                frame = sensor.read(timeout_ms=min(args.frame_timeout_ms, remaining_ms))
                if time.monotonic() >= deadline:
                    break
                yield idx, frame
                idx += 1
        finally:
            sensor.stop()
        return
    from faceauth.sim.sensor import SensorSim
    sim = SensorSim(seed=args.seed)
    if args.source == "biwi":
        from faceauth.sim.biwi import iter_frames
        seq = args.enroll_seq if phase == "enroll" else args.auth_seq or args.enroll_seq
        first = None
        for f in iter_frames(args.biwi_root, seqs=[seq], step=args.step):
            first = f["frame"] if first is None else first
            yield f["frame"] - first, _render(sim, pipe, f["rgb"], 1, args.seed + f["frame"], args.distance_mm)[0]
    else:
        idx = 0
        for k, path in enumerate(args.images if phase == "enroll" else args.auth_images or args.images):
            for f in _render(sim, pipe, read_rgb(path), args.repeat, args.seed + k, args.distance_mm):
                yield idx, f
                idx += 1


def capture_ref(args, idx, encoded, frame):
    """Keep only encoded images and packet metadata in the guide/auth selection."""
    metadata = None
    if args.source == "orbbec":
        if frame["calib_version"] != args.calib_version:
            raise RuntimeError("Capture calibration differs from the registered device")
        metadata = {"ts_ms": dict(frame["ts_ms"]), "captured_at_ms": int(frame["captured_at_ms"])}
    return idx, encoded, metadata


def packet_capture(args, refs, start_ms):
    """Return frames, timestamps and first capture time; real captures never use synced_ts()."""
    if args.source == "orbbec":
        timestamps = [dict(ref[2]["ts_ms"]) for ref in refs]
        captured_at_ms = refs[0][2]["captured_at_ms"]
    else:
        timestamps = synced_ts([ref[0] for ref in refs], start_ms)
        captured_at_ms = start_ms
    return [ref[1] for ref in refs], timestamps, captured_at_ms


def shown(out: dict) -> dict:
    """Server answer for printing, bearer approval token elided."""
    return {k: (v[:12] + "..." if k == "approval_token" else v) for k, v in out.items()}


def coverage_line(cov: dict) -> str:
    return " ".join(f"{d}:{n}/{req}" for d, (n, req) in ((d, cov[d]) for d in ORDER))


def enroll(args, client: DeviceClient, pipe: Pipeline) -> bool:
    session = client.create_session("enroll", "enroll")
    # account ownership is proven before capture; the assertion must outlive the enrollment session (TTL 180 s)
    assertion = make_account_assertion(args.account, session["challenge"], "password", ttl_s=180)
    guide, last, start_ms, t0 = Guide(timeout_s=args.timeout), None, int(time.time() * 1000), time.perf_counter()
    with closing(frame_source(args, "enroll", pipe)) as source:
        for idx, f in source:
            if f is None:  # nobody the simulator could see
                s = guide.feed(None, False, t=idx / FPS, reasons=("no_face",))
            else:  # analyse what will be sent; real capture time drives the guide
                enc, rgb = wire_frame(f["rgb"], f["depth"], f["ir"])
                capture_t = f["captured_at_ms"] / 1000 if args.source == "orbbec" else idx / FPS
                s = pipe.feed(guide, rgb, capture_ref(args, idx, enc, f), t=capture_t)[1]
            if (s["prompt"], s["coverage"]) != last:
                last = (s["prompt"], s["coverage"])
                print(f"[{idx:4d}] {s['prompt']:<28s} {coverage_line(s['coverage'])}")
            if s["done"] or s["timed_out"]:
                break
    if not guide.done:
        print("enrollment capture incomplete:", coverage_line(guide.coverage()))
        return False
    print(f"capture done in {time.perf_counter() - t0:.1f}s wall")
    kept = guide.frames()
    frames, ts, captured_at_ms = packet_capture(args, [ref for _, ref in kept], start_ms)
    consent = {"version": args.consent_version, "agreed_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    real_capture = {"captured_at_ms": captured_at_ms} if args.source == "orbbec" else {}
    out = client.enroll(session, frames, ts, [d for d, _ in kept], assertion, consent,
                        re_enroll=args.re_enroll, **real_capture)
    print("enroll ->", out)
    return out.get("status") == "active"


def authenticate(args, client: DeviceClient, pipe: Pipeline) -> str:
    session = client.create_session("auth", args.action)
    start_ms, t0, results, seen, sel = int(time.time() * 1000), time.perf_counter(), [], [], []
    with closing(frame_source(args, "auth", pipe)) as source:
        for idx, f in source:
            if idx >= args.auth_max_frames:
                break
            if f is not None:
                enc, rgb = wire_frame(f["rgb"], f["depth"], f["ir"])
                results.append(pipe.analyze(rgb))
                seen.append(capture_ref(args, idx, enc, f))
                sel = select_auth(results)
                if sel:
                    break
    if not sel:
        print("authentication capture failed: fewer than 3 usable frames")
        return "capture_failed"
    frames, ts, captured_at_ms = packet_capture(args, [seen[i] for i in sel], start_ms)
    out = client.authenticate(session, frames, ts, captured_at_ms=captured_at_ms)
    print(f"auth -> {out.get('decision')} ({out.get('reason')}) in {1000 * (time.perf_counter() - t0):.0f} ms "
          f"from first frame; frames {[seen[i][0] for i in sel]}")
    sid = session["session_id"]
    if out.get("decision") == "step_up":  # high-risk action: customer-owned factor (simulated OTP)
        out = client.complete_stepup(out["step_up_id"], make_account_assertion(args.account, out["step_up_id"], "otp"))
        print("step-up ->", shown(out))
    elif out.get("decision") != "approved" and args.fallback:  # non-biometric alternative on a fresh session
        fb = client.create_session("auth", args.action)
        out, sid = client.fallback(fb, make_account_assertion(args.account, fb["challenge"], "pin")), fb["session_id"]
        print("fallback ->", shown(out))
    if not out.get("approval_token"):
        return out.get("decision") or "rejected"
    # relying-party check: approval signature + scope to this device, action and session
    p = token.verify(serialization.load_pem_public_key((SECRETS / "approval.pub").read_bytes()), out["approval_token"])
    if (p.get("dev"), p.get("act"), p.get("sid")) != (args.device, args.action, sid):
        raise SystemExit(f"approval token scope mismatch: {p}")
    print("approval token verified:", {k: p.get(k) for k in ("amr", "iat", "exp")})
    return "approved"


def letterbox(rgb, w=640, h=480) -> np.ndarray:
    """Fit into the 640x480 sensor frame (aspect kept, black padding)."""
    s = min(w / rgb.shape[1], h / rgb.shape[0])
    small = cv2.resize(rgb, (round(rgb.shape[1] * s), round(rgb.shape[0] * s)), interpolation=cv2.INTER_AREA)
    out = np.zeros((h, w, 3), np.uint8)
    out[:small.shape[0], :small.shape[1]] = small
    return out


def register(args, client: DeviceClient) -> None:
    """Admin step (Bearer admin_token, through the gateway): register this device's keys and calibration."""
    d = SECRETS / "devices"
    body = {"device_id": args.device, "public_key_pem": (d / f"{args.device}.ed25519.pub").read_text(),
            "cert_pem": (d / f"{args.device}.crt").read_text(), "calib_version": client.calib_version}
    r = client.http.post("/v1/admin/devices", json=body, extensions={"sni_hostname": client.server_name},
                         headers={"Authorization": "Bearer " + (SECRETS / "admin_token").read_text().strip()})
    r.raise_for_status()
    print("registered", r.json())


def smoke(args) -> None:
    pipe = Pipeline.load(args.models, args.torch_device, verify=not args.unsigned_models)
    imgs = [letterbox(read_rgb(p)) for p in args.smoke]
    for img in imgs:  # untimed warm-up pass (CUDA init, cuDNN autotune) so every stage is warm
        pipe.analyze(img)
    times, full, signs = [], [], []
    for p, img in zip(args.smoke, imgs):
        t = time.perf_counter()
        r = pipe.analyze(img)
        times.append(1000 * (time.perf_counter() - t))
        pose = "-" if r["pose"] is None else "yaw %+.1f pitch %+.1f roll %+.1f" % tuple(r["pose"])
        q = " ".join(f"{k}={v:.2f}" for k, v in r["quality"].items())
        print(f"{Path(p).name}: ok={r['ok']} {r['reasons']} faces={r['n_faces']} {pose} {q} "
              f"score={r['score']:.2f} {times[-1]:.1f} ms")
        if r["pose"] is not None:
            full.append(times[-1])
            k = r["kps"]
            if abs(r["pose"][0]) > 10:  # §0: yaw > 0 <=> nose right of the eye midpoint in the image
                signs.append((r["pose"][0] > 0) == (k[2, 0] > (k[0, 0] + k[1, 0]) / 2))
    for name, t in (("all frames", np.array(times)), ("full path (det+pose+landmarks)", np.array(full))):
        if len(t):
            print(f"latency {name}, n={len(t)} (640x480): mean {t.mean():.1f} ms, p50 {np.median(t):.1f}, "
                  f"p95 {np.percentile(t, 95):.1f}, max {t.max():.1f}")
    print(f"yaw sign agrees with SCRFD nose/eye geometry on {sum(signs)}/{len(signs)} faces with |yaw| > 10")


def run(args):
    """Run one CLI transaction, always releasing USB and HTTP resources."""
    pipe = Pipeline.load(args.models, args.torch_device, verify=not args.unsigned_models)
    with ExitStack() as resources:
        client_options = {}
        if args.source == "orbbec":
            from faceauth.terminal.orbbec import FemtoBoltSensor
            args.sensor = FemtoBoltSensor(serial=args.serial, fps=args.fps)
            resources.callback(args.sensor.stop)
            args.sensor.start()
            args.calib_version = args.sensor.calib_version
            args.sensor.stop()  # each actual capture starts after its server challenge exists
            client_options["calib_version"] = args.calib_version
        client = DeviceClient(args.gateway, args.device, **client_options)
        resources.callback(client.close)
        if args.register:
            register(args, client)
        if not args.skip_enroll and not enroll(args, client, pipe):
            return 1
        return 0 if authenticate(args, client, pipe) == "approved" else 2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gateway", default="https://gateway:8443")
    ap.add_argument("--device", default="dev-001")
    ap.add_argument("--account", default="acct-0001")
    ap.add_argument("--action", default="door-open", help="action_id of the authentication session")
    ap.add_argument("--source", choices=("sim", "biwi", "orbbec"))
    ap.add_argument("--serial", help="Femto Bolt serial number (required only when multiple cameras are attached)")
    ap.add_argument("--fps", type=int, choices=(5, 15, 30), default=30, help="Femto Bolt stream FPS; 30 requires USB3 and GPU depth processing")
    ap.add_argument("--frame-timeout-ms", type=int, default=10000, help="maximum wait for a real synchronized frame")
    ap.add_argument("--auth-timeout", type=float, default=20.0, help="real authentication capture deadline in seconds")
    ap.add_argument("--images", nargs="*", default=[], help="face images of one person for --source sim")
    ap.add_argument("--auth-images", nargs="*", default=[], help="authentication images (default: --images)")
    ap.add_argument("--repeat", type=int, default=3, help="sensor frames rendered per image (--source sim)")
    ap.add_argument("--biwi-root", default="data/datasets/biwi")
    ap.add_argument("--enroll-seq", default="01", help="BIWI sequence for enrollment")
    ap.add_argument("--auth-seq", help="BIWI sequence for authentication (default: --enroll-seq)")
    ap.add_argument("--step", type=int, default=3, help="use every n-th BIWI frame")
    ap.add_argument("--distance-mm", type=float, default=600.0, help="simulated subject distance")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--register", action="store_true", help="register the device first (uses secrets/admin_token)")
    ap.add_argument("--unsigned-models", action="store_true",
                    help="DEV ONLY: skip the signed model registry check (no data/models/registry.json yet)")
    ap.add_argument("--consent-version", default="v1")
    ap.add_argument("--re-enroll", action="store_true")
    ap.add_argument("--fallback", action="store_true", help="try the non-biometric fallback after a rejection")
    ap.add_argument("--timeout", type=float, default=60.0, help="enrollment guidance timeout (sensor seconds)")
    ap.add_argument("--auth-max-frames", type=int, default=30)
    ap.add_argument("--skip-enroll", action="store_true")
    ap.add_argument("--models", default="data/models")
    ap.add_argument("--torch-device", default="cuda")
    ap.add_argument("--smoke", nargs="*", help="run the terminal pipeline on these images (no server)")
    args = ap.parse_args()
    if args.smoke:
        return smoke(args)
    if not args.source:
        ap.error("--source is required (or --smoke)")
    if args.source == "sim" and not args.images:
        ap.error("--source sim needs --images")
    if min(args.timeout, args.auth_timeout, args.auth_max_frames, args.frame_timeout_ms) <= 0:
        ap.error("capture timeouts and --auth-max-frames must be positive")
    sys.exit(run(args))


if __name__ == "__main__":
    main()
