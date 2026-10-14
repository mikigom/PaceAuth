"""Probe actual RGB/depth/IR capture without saving biometric image data.

    python scripts/probe_orbbec.py --frames 10

Exit 0 means the required synchronized modalities were captured. Exit 1 means
the sensor is unavailable or incomplete; simulation is never substituted.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from faceauth.terminal.orbbec import FemtoBoltSensor, SensorError, discover_devices  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial")
    parser.add_argument("--info", action="store_true", help="list devices without starting image streams")
    parser.add_argument("--fps", type=int, default=15, choices=(5, 15, 30))
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--timeout-ms", type=int, default=10000)
    args = parser.parse_args()
    if args.frames < 1 or args.timeout_ms < 1:
        parser.error("--frames and --timeout-ms must be positive")
    sensor = FemtoBoltSensor(serial=args.serial, fps=args.fps)
    try:
        if args.info:
            devices = discover_devices()
            print(json.dumps({"event": "discovery", "devices": devices}), flush=True)
            return 0 if any(d["pid"] == 0x066B and (not args.serial or d["serial"] == args.serial) for d in devices) else 1
        sensor.start()
        first_arrival = last_arrival = None
        max_skew_ms = 0
        for index in range(args.frames):
            frame = sensor.read(timeout_ms=args.timeout_ms)
            last_arrival = time.monotonic()
            if first_arrival is None:
                first_arrival = last_arrival
            max_skew_ms = max(max_skew_ms, sensor.info["sync_skew_ms"])
            print(json.dumps({"event": "frame", "ts_ms": frame["ts_ms"],
                              "shape": list(frame["rgb"].shape), "sync_skew_ms": sensor.info["sync_skew_ms"],
                              "depth_valid_fraction": sensor.info["depth_valid_fraction"]}), flush=True)
        span = last_arrival - first_arrival
        print(json.dumps({"event": "ready", **sensor.info, "captured_frames": index + 1,
                          "capture_span_s": round(span, 3),
                          "observed_fps": round(index / span, 2) if span > 0 else None,
                          "max_sync_skew_ms": max_skew_ms}), flush=True)
        return 0
    except SensorError as exc:
        print(json.dumps({"event": "unavailable", **sensor.info, "error": str(exc)}), flush=True)
        return 1
    finally:
        sensor.stop()


if __name__ == "__main__":
    raise SystemExit(main())
