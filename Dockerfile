# GHOST System v2 — application image (software-only run modes)
#
# Build context is the repo root (see docker-compose.yml). Only the v2/ package is
# copied to /app/v2, with PYTHONPATH=/app so `import v2.*` resolves, exactly like a
# host checkout. This image runs everything that does NOT need USB hardware:
# tests, the software demos, the cognitive replay, and dataset comparison.
# Hardware serial/Wi-Fi modes must run on the host (USB passthrough is not
# portable in Docker) — see the README.
FROM python:3.12-slim

# scipy/numpy wheels are prebuilt, but keep build tools for any source fallback.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

# Install deps first so the layer caches across code edits.
COPY v2/requirements.txt /app/v2/requirements.txt
RUN pip install --no-cache-dir -r /app/v2/requirements.txt

# Copy the self-contained v2 package (includes example_csi.csv + datasets/).
COPY v2 /app/v2

# In-container, the cognitive services live at these hostnames (compose network).
ENV GHOST_OLLAMA_HOST=ollama \
    GHOST_CHROMA_HOST=chromadb

# Default: prove the pipeline end to end on the bundled capture.
CMD ["python", "v2/run_phase2_demo.py", "--max-frames", "800", "--calib", "200"]
