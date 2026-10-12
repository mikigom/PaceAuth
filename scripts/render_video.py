"""Render ui/video.html frame by frame with headless Chromium and encode an H.264 MP4 (1920x1080).

    docker build -f docker/video.Dockerfile -t faceauth-video:dev .
    docker run --rm -u $(id -u):$(id -g) -e HOME=/tmp -v $PWD:/app -w /app faceauth-video:dev \
        python3 scripts/render_video.py [--stills 2,30,75]

The page draws the scene for any time t (window.renderAt), so every frame is exact regardless of render speed.
Inputs: data/demo/timeline.js + frames (scripts/demo_record.py). Output: data/demo/faceauth_demo.mp4 (not in git).
"""
import argparse
import subprocess
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/demo/faceauth_demo.mp4")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stills", help="comma-separated times (s): write PNGs to data/demo/stills/ instead of a video")
    a = ap.parse_args()
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--allow-file-access-from-files"])
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto((ROOT / "ui/video.html").as_uri())
        page.wait_for_function("window.READY === true || window.DURATION === undefined", timeout=60_000)
        if errors:
            raise SystemExit(f"page errors: {errors}")
        dur = page.evaluate("window.DURATION")
        print(f"duration {dur:.1f} s, scenes (start, length): {page.evaluate('window.SCENE_TIMES')}", flush=True)
        if a.stills:
            out = ROOT / "data/demo/stills"
            out.mkdir(parents=True, exist_ok=True)
            for t in (float(x) for x in a.stills.split(",")):
                page.evaluate("t => window.renderAt(t)", t)
                page.screenshot(path=str(out / f"t{t:06.2f}.png"))
            print("stills ->", out)
            return
        ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(a.fps),
                               "-c:v", "mjpeg", "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(ROOT / a.out)],
                              stdin=subprocess.PIPE)
        n = int(dur * a.fps)
        for i in range(n):
            page.evaluate("t => window.renderAt(t)", i / a.fps)
            ff.stdin.write(page.screenshot(type="jpeg", quality=93))
            if i % (a.fps * 10) == 0:
                print(f"{i / a.fps:.0f}/{dur:.0f} s", flush=True)
        ff.stdin.close()
        if ff.wait():
            raise SystemExit("ffmpeg failed")
        if errors:
            raise SystemExit(f"page errors: {errors}")
        print("video ->", ROOT / a.out, f"{n} frames")


if __name__ == "__main__":
    main()
