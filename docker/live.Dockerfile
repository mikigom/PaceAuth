FROM python:3.11-slim-bookworm
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 HOME=/tmp XDG_CACHE_HOME=/tmp/.cache \
    MESA_SHADER_CACHE_DIR=/tmp/mesa
RUN apt-get update && apt-get install -y --no-install-recommends \
    libusb-1.0-0 libudev1 libegl1 libgl1 libgl1-mesa-dri \
    && rm -rf /var/lib/apt/lists/*
COPY docker/live.requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt \
    && pip install --no-deps pyorbbecsdk2==2.1.2
WORKDIR /app
COPY faceauth /app/faceauth
COPY ui /app/ui
COPY scripts/probe_orbbec.py /app/scripts/probe_orbbec.py
# The SDK initializes its native logger before Python can redirect it.
WORKDIR /tmp
CMD ["uvicorn", "faceauth.terminal.live:app", "--host", "0.0.0.0", "--port", "8080", "--no-access-log", "--no-server-header"]
