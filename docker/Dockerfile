# One image for every service; only engine/search get a GPU (device 5) via compose.
FROM pytorch/pytorch:2.12.0-cuda13.0-cudnn9-devel
ENV PIP_BREAK_SYSTEM_PACKAGES=1 PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 openssl curl postgresql-client \
 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
WORKDIR /app
COPY faceauth /app/faceauth
COPY eval /app/eval
COPY scripts /app/scripts
COPY tests /app/tests
