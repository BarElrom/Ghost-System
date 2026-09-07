

import os
from dataclasses import dataclass

# --- Self-contained paths -------------------------------------------------
# v2 is designed to run entirely from within this directory. The demo capture
# (example_csi.csv) ships under v2/; we still fall back to a repo-root copy so
# older checkouts keep working. Override with GHOST_V2_DATASET.
_V2_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_V2_DIR)


def _resolve_default_dataset() -> str:
    for cand in (os.path.join(_V2_DIR, "example_csi.csv"),
                 os.path.join(_REPO_ROOT, "example_csi.csv")):
        if os.path.exists(cand):
            return cand
    return os.path.join(_V2_DIR, "example_csi.csv")


DEFAULT_DATASET_PATH = os.getenv("GHOST_V2_DATASET", _resolve_default_dataset())

NUM_SUBCARRIERS = 64
NUM_RECEIVERS = 3
# CSI frame rate per receiver. Dropped from 100 → 50 Hz (Note 2): at 100 Hz the
# ESP32 + USB link dropped packets; 50 Hz halves the load and still resolves
# indoor motion (bandpass highcut 4 Hz ≪ Nyquist 25 Hz). Window *durations*
# below are expressed in seconds, so sample counts scale with this value.
SAMPLE_RATE_HZ = 50

NODE_IDS = {"RX1": 1, "RX2": 2, "RX3": 3}
NODE_ID_TO_NAME = {v: k for k, v in NODE_IDS.items()}

UDP_PORT = int(os.getenv("GHOST_V2_UDP_PORT", "5005"))

NODE_NET_MAP = {
    "RX1": (os.getenv("GHOST_V2_RX1_IP", "127.0.0.1"), UDP_PORT),
    "RX2": (os.getenv("GHOST_V2_RX2_IP", "127.0.0.1"), UDP_PORT),
    "RX3": (os.getenv("GHOST_V2_RX3_IP", "127.0.0.1"), UDP_PORT),
}

TX_MAC = os.getenv("GHOST_V2_TX_MAC", "1a:00:00:00:00:00")
WIFI_CHANNEL = 6
DEFAULT_RSSI = -45
DEFAULT_NOISE_FLOOR = -95

CALIBRATION_SAMPLES = int(os.getenv("GHOST_V2_CALIBRATION_SAMPLES", "200"))
CALIBRATION_MODE = os.getenv("GHOST_V2_CALIBRATION_MODE", "preamble")

BANDPASS_LOWCUT = 0.1
BANDPASS_HIGHCUT = 4.0
BUTTER_ORDER = 4
HAMPEL_WINDOW = 3
HAMPEL_THRESHOLD = 3.0

# Phase sanitization (Note 6): the ESP32 adds a random per-packet carrier-phase
# offset (CFO) and a sampling-time offset (STO) that shows up as a linear phase
# ramp across subcarriers. We fit and remove that line per frame so phase is
# comparable across frames — making static subtraction phase-coherent and the
# phase-variance feature meaningful. Toggle off to inspect raw ESP32 phase.
PHASE_SANITIZE = os.getenv("GHOST_V2_PHASE_SANITIZE", "1") == "1"

# Adaptive static-reflection removal (Note 7): after calibration fixes an initial
# empty-room baseline, blend each cleaned window's static estimate back into it
# with an exponential moving average — H_static ← (1-α)·H_static + α·H_window.
# This tracks slow drift and newly-static clutter (furniture moved into the scene)
# instead of trusting a one-shot baseline forever. α is per-window: the memory is
# ≈1/α windows, so 0.02 ≈ 50-window (tens of seconds) memory. 0 disables it
# (fixed baseline, the pre-Phase-4 behavior). A person who stops moving will fade
# into the baseline after ~1/α windows — that is the intended clutter tradeoff.
STATIC_EWMA_ALPHA = float(os.getenv("GHOST_V2_STATIC_EWMA_ALPHA", "0.02"))

NODE_POSITIONS = {
    "RX1": (-1.0, 0.0),
    "RX2": (1.0, 0.0),
    "RX3": (2.0, 0.0),
}

MAX_PLAUSIBLE_SPEED_M_S = 3.0

# Kalman smoothing of the localizer output (Note 8). A constant-velocity filter
# over [x, y] turns the noisy per-window position into a smooth track and reads
# velocity straight from its state (steadier than the frame-to-frame difference).
# PROCESS_VAR ~ how much the target can accelerate between windows; MEASUREMENT_VAR
# ~ localizer position noise (m²). Both are starting points — tune per deployment.
KALMAN_ENABLED = os.getenv("GHOST_V2_KALMAN", "1") == "1"
KALMAN_PROCESS_VAR = float(os.getenv("GHOST_V2_KALMAN_PROCESS_VAR", "0.3"))
KALMAN_MEASUREMENT_VAR = float(os.getenv("GHOST_V2_KALMAN_MEASUREMENT_VAR", "0.25"))

POS_ENERGY_SCALES = {
    "plan": {
        "y_depth_divisor": 25.0,
        "y_max": 6.0,
        "y_min": 0.5,
        "confidence_divisor": 80.0,
    },
    "embedded_wifi": {
        "y_depth_divisor": 600.0,
        "y_max": 6.0,
        "y_min": 0.5,
        "confidence_divisor": 2000.0,
    },
    # Per-dataset starting points — RE-TUNE with the DEBUG E_total logs so
    # confidence isn't pinned at 1.0 (energy magnitudes differ per source).
    "csi_bench": {
        "y_depth_divisor": 600.0,
        "y_max": 6.0,
        "y_min": 0.5,
        "confidence_divisor": 2000.0,
    },
    "intel_resp": {
        "y_depth_divisor": 600.0,
        "y_max": 6.0,
        "y_min": 0.5,
        "confidence_divisor": 2000.0,
    },
    "giz": {
        "y_depth_divisor": 600.0,
        "y_max": 6.0,
        "y_min": 0.5,
        "confidence_divisor": 2000.0,
    },
}

DEFAULT_POS_SCALE = os.getenv("GHOST_V2_POS_SCALE", "plan")


# --- Cognitive-layer service settings (Ollama / ChromaDB) -----------------
# Kept in v2 so the pipeline is self-contained (no dependency on the root
# config.py). Override via env vars; inside Docker the compose file sets
# GHOST_OLLAMA_HOST=ollama and GHOST_CHROMA_HOST=chromadb.
@dataclass(frozen=True)
class OllamaSettings:
    """Connection settings for the Ollama LLM server (the AI layer, --llm)."""

    host: str = os.getenv("GHOST_OLLAMA_HOST", "localhost")
    port: int = int(os.getenv("GHOST_OLLAMA_PORT", "11434"))
    model: str = os.getenv("GHOST_OLLAMA_MODEL", "llama3.2:3b")
    timeout: float = float(os.getenv("GHOST_OLLAMA_TIMEOUT", "30.0"))

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass(frozen=True)
class ChromaSettings:
    """Connection settings for the ChromaDB vector store (memory bank)."""

    host: str = os.getenv("GHOST_CHROMA_HOST", "localhost")
    port: int = int(os.getenv("GHOST_CHROMA_PORT", "8000"))
    collection_name: str = os.getenv("GHOST_CHROMA_COLLECTION", "spatial_fingerprints")


OLLAMA_SETTINGS = OllamaSettings()
CHROMA_SETTINGS = ChromaSettings()
