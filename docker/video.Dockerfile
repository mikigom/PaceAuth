# Demo-video rendering (scripts/render_video.py): headless Chromium via Playwright, ffmpeg, Korean fonts. No GPU.
# Python playwright is pinned to the base image's browser build (/ms-playwright).
FROM mcr.microsoft.com/playwright:v1.61.0-noble
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip ffmpeg fonts-noto-cjk \
 && rm -rf /var/lib/apt/lists/* \
 && pip3 install --break-system-packages --no-cache-dir playwright==1.61.0
