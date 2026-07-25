"""
GHOST System v2 — Configuration

Additive configuration for the injection/loopback pipeline. This file grows
as later v2 steps land; for now it carries the constants needed by the
Physical & Transport layer (Plan section 2).

Override via environment variables where noted.
"""

import os

# ---------------------------------------------------------------------------
# Core signal contract (fixed — adapters resample datasets to match)
# ---------------------------------------------------------------------------
NUM_SUBCARRIERS = 64        # LLTF subcarriers carried per frame
NUM_RECEIVERS = 3           # RX1 / RX2 / RX3
SAMPLE_RATE_HZ = 100        # injector pacing target and DSP fs

# ---------------------------------------------------------------------------
# Node identity
# ---------------------------------------------------------------------------
# Logical receiver name <-> compact integer id carried in the wire frame.
NODE_IDS = {"RX1": 1, "RX2": 2, "RX3": 3}
NODE_ID_TO_NAME = {v: k for k, v in NODE_IDS.items()}

# ---------------------------------------------------------------------------
# UDP injection transport (host -> ESP32 over Wi-Fi)
# ---------------------------------------------------------------------------
# On real hardware each ESP32 has its own IP; for local software testing all
# three map to localhost and are disambiguated by the node id inside the frame.
UDP_PORT = int(os.getenv("GHOST_V2_UDP_PORT", "5005"))

NODE_NET_MAP = {
    "RX1": (os.getenv("GHOST_V2_RX1_IP", "127.0.0.1"), UDP_PORT),
    "RX2": (os.getenv("GHOST_V2_RX2_IP", "127.0.0.1"), UDP_PORT),
    "RX3": (os.getenv("GHOST_V2_RX3_IP", "127.0.0.1"), UDP_PORT),
}

# ---------------------------------------------------------------------------
# CSI_DATA line metadata (placeholders the ESP32 firmware / mock stamp into
# the emitted serial line). Only id, timestamp, and the I/Q array are faithful;
# the rest are constant fillers the ghost gateway does not depend on.
# ---------------------------------------------------------------------------
TX_MAC = os.getenv("GHOST_V2_TX_MAC", "1a:00:00:00:00:00")
WIFI_CHANNEL = 6
DEFAULT_RSSI = -45
DEFAULT_NOISE_FLOOR = -95

# ---------------------------------------------------------------------------
# Signal cleaning — static-baseline calibration + stage-2 temporal filtering
# (Plan section 7). Calibration mode:
#   "preamble"      : build the static baseline from the empty-room preamble
#                     the injector streams first (Plan section 11, preferred).
#   "temporal_mean" : fall back to the per-window temporal mean when no
#                     empty-room calibration is available.
# ---------------------------------------------------------------------------
CALIBRATION_SAMPLES = int(os.getenv("GHOST_V2_CALIBRATION_SAMPLES", "200"))
CALIBRATION_MODE = os.getenv("GHOST_V2_CALIBRATION_MODE", "preamble")

# Butterworth band-pass (human-motion band) + Hampel outlier rejection.
BANDPASS_LOWCUT = 0.1     # Hz — removes residual static / drift
BANDPASS_HIGHCUT = 4.0    # Hz — removes high-freq electronic noise
BUTTER_ORDER = 4
HAMPEL_WINDOW = 3         # half-window; total = 2*w + 1
HAMPEL_THRESHOLD = 3.0    # outlier threshold in sigmas
