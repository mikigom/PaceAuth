# Operator CLI: real sensor capture plus the existing signed GPU model pipeline.
FROM faceauth:dev
RUN apt-get update && apt-get install -y --no-install-recommends \
    libusb-1.0-0 libudev1 libegl1 libgl1-mesa-dri \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-deps pyorbbecsdk2==2.1.2 \
    && python -c "import pyorbbecsdk"
COPY faceauth/terminal/orbbec.py /app/faceauth/terminal/orbbec.py
COPY scripts/terminal_demo.py scripts/probe_orbbec.py /app/scripts/
ENV HOME=/tmp XDG_CACHE_HOME=/tmp/.cache MESA_SHADER_CACHE_DIR=/tmp/mesa \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LP_NUM_THREADS=4
WORKDIR /tmp
CMD ["python", "/app/scripts/terminal_demo.py", "--help"]
