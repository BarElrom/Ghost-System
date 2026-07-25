# GHOST System — Implementation Guide

**Wi-Fi CSI-Based Human Presence and Activity Sensing System**
**Authors:** Bar Elrom, Yuval Lerfeld
**Advisor:** Mr. Ilya Zeldner — Braude College of Engineering

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Summary](#2-architecture-summary)
3. [Prerequisites and Environment Setup](#3-prerequisites-and-environment-setup)
4. [Layer 1 — Physical Sensing Layer (Firmware)](#4-layer-1--physical-sensing-layer-firmware)
5. [Layer 2 — Processing Layer (Signal Processing Pipeline)](#5-layer-2--processing-layer-signal-processing-pipeline)
6. [Layer 3 — Cognitive Layer (Memory, Reasoning, UI)](#6-layer-3--cognitive-layer-memory-reasoning-ui)
7. [Integration and End-to-End Pipeline](#7-integration-and-end-to-end-pipeline)
8. [Testing and Validation](#8-testing-and-validation)
9. [File Inventory](#9-file-inventory)

---

## 1. Project Overview

The GHOST system uses standard 2.4 GHz Wi-Fi signals to detect human presence, motion, and basic physiological activity (breathing) in indoor environments without cameras or wearable devices. A single ESP32 transmitter broadcasts packets while three spatially separated ESP32 receivers capture Channel State Information (CSI). The CSI data flows through a signal processing pipeline that extracts motion features, which are then interpreted by an AI cognitive layer (Llama 3) to produce structured decisions about presence, location, and activity.

### Data Flow (End-to-End)

```
ESP32 Tx (csi_send)
  | ESP-NOW packets @ 100 Hz, channel 6, 2.4 GHz
  v
ESP32 Rx x3 (csi_recv)
  | CSI_DATA CSV lines via USB serial @ 921600 baud
  v
Gateway (gateway.py)                  -- parse, sync, amplitude matrix
  | Raw Amplitude Matrix [3 x 64 x T]
  v
Signal Cleaner (signal_cleaner.py)    -- Hampel + Butterworth bandpass
  | Cleaned Motion Matrix [3 x 64 x T]
  v
Feature Extractor (feature_extractor.py)  -- PCA + STFT
  | State Vector (6 features)
  v
Memory Bank (ChromaDB)                -- cosine similarity lookup
  | Candidate zones + scores
  v
Agentic Core (Llama 3)               -- reasoning + decision
  | Final Decision JSON
  v
Dashboard                             -- floorplan + live indicators
```

---

## 2. Architecture Summary

| Layer | Component | File(s) | Status |
|-------|-----------|---------|--------|
| 1 — Physical Sensing | ESP32 Transmitter firmware | `esp-csi/examples/get-started/csi_send/` | Complete |
| 1 — Physical Sensing | ESP32 Receiver firmware | `esp-csi/examples/get-started/csi_recv/` | Complete |
| 2 — Processing | Gateway (Data Ingestor) | `gateway.py` | Complete |
| 2 — Processing | Signal Cleaner (DSP) | `signal_cleaner.py` | Complete |
| 2 — Processing | Feature Extractor | `feature_extractor.py` | Complete |
| 3 — Cognitive | Memory Bank (ChromaDB) | `memory_bank.py`, `config.py`, `docker-compose.yml` | Complete |
| 3 — Cognitive | Agentic Core (Llama 3) | `agentic_core.py`, `config.py`, `docker-compose.yml` | Complete |
| 3 — Cognitive | Dashboard (Frontend) | — | Not implemented |

---

## 3. Prerequisites and Environment Setup

### Step 3.1 — Install Python Environment

Create a virtual environment and install all dependencies.

```bash
cd /path/to/TheGhostSystem
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 3.2 — Start Docker Services (ChromaDB + Ollama)

The Memory Bank (ChromaDB) and Agentic Core (Ollama) both run as Docker services.

```bash
# Start both containers (ChromaDB on port 8000, Ollama on port 11434):
docker compose up -d

# Verify ChromaDB:
curl http://localhost:8000/api/v1/heartbeat

# Verify Ollama:
curl http://localhost:11434/api/tags

# Stop when done:
docker compose down
```

ChromaDB uses a named volume (`ghost_chroma_data`) for persistence. Ollama uses `ghost_ollama_data` to cache downloaded models. Data survives container restarts. To reset all stored fingerprints, use `python enrollment.py --clear` or remove volumes with `docker compose down -v`.

**GPU note:** Docker on macOS does not support GPU passthrough; Ollama runs on CPU only. For GPU acceleration on macOS, run Ollama natively (`brew install ollama && ollama serve`) and point config to `localhost:11434`. On Linux with NVIDIA GPUs, uncomment the `deploy:` section in `docker-compose.yml`.

### Step 3.3 — Install ESP-IDF Toolchain (for firmware flashing)

The ESP32 firmware requires the Espressif IoT Development Framework (ESP-IDF).

1. Follow the official ESP-IDF installation guide for your platform: https://docs.espressif.com/projects/esp-idf/en/latest/esp32/get-started/
2. The project uses ESP-IDF v5.x (check `esp-csi/` for the exact version).
3. After installation, source the ESP-IDF environment:
   ```bash
   . $HOME/esp/esp-idf/export.sh
   ```

### Step 3.3 — Hardware Requirements

| Component | Quantity | Role |
|-----------|----------|------|
| ESP32-DevKitC (or equivalent) | 4 | 1 transmitter + 3 receivers |
| USB cables (USB-A to Micro-USB) | 4 | Power + serial data |
| Host computer (macOS/Linux) | 1 | Runs processing + cognitive layers |

**Physical deployment:**
- Place 3 receivers at approximately 1-meter spacing in the target room.
- The transmitter can be co-located or placed centrally.
- All devices operate on Wi-Fi channel 6 (2.4 GHz, wavelength ~12.5 cm).
- The system operates in 2D (X, Y) planar localization only.

---

## 4. Layer 1 — Physical Sensing Layer (Firmware)

### Step 4.1 — Flash the Transmitter Firmware (`csi_send`)

The transmitter broadcasts ESP-NOW null data packets at ~100 Hz on Wi-Fi channel 6.

```bash
cd esp-csi/examples/get-started/csi_send

# Set the ESP32 target
idf.py set-target esp32

# Configure (optional — defaults are correct for GHOST)
idf.py menuconfig

# Build and flash
idf.py build
idf.py -p /dev/cu.usbserial-XXXX flash monitor
```

**Transmitter behavior:**
- Initializes Wi-Fi radio in AP/monitor-compatible mode.
- Sets fixed Wi-Fi channel = 6.
- Constructs 802.11 null data packets with destination MAC `1a:00:00:00:00:00`.
- Transmits one packet every ~10 ms (100 Hz probing rate).
- Each packet carries an incrementing sequence number.

**Key firmware configuration:**
- Wi-Fi channel: 6
- Bandwidth: HT40 (40 MHz)
- PHY mode: 802.11n
- Packet type: ESP-NOW

### Step 4.2 — Flash the Receiver Firmware (`csi_recv`)

Each receiver listens for the transmitter's packets, extracts CSI, and outputs CSV data over USB serial.

```bash
cd esp-csi/examples/get-started/csi_recv

idf.py set-target esp32
idf.py build
idf.py -p /dev/cu.usbserial-XXXX flash monitor
```

Repeat for all 3 receiver ESP32 boards, each connected to a different USB port.

**Receiver behavior:**
- Enables promiscuous mode to capture all packets.
- For each received packet from the transmitter MAC, extracts CSI across 64 OFDM subcarriers.
- Outputs a CSV line per packet over UART at 921600 baud.

**CSV output format per line:**

```
CSI_DATA,<seq_id>,<mac>,<rssi>,<rate>,<sig_mode>,<mcs>,<bandwidth>,
<smoothing>,<not_sounding>,<aggregation>,<stbc>,<fec_coding>,<sgi>,
<noise_floor>,<ampdu_cnt>,<channel>,<secondary_channel>,
<local_timestamp>,<ant>,<sig_len>,<rx_format>,<len>,<first_word>,
"[I0,Q0,I1,Q1,...,I63,Q63]"
```

- The CSI data array contains 128 signed integers (64 I/Q pairs for 64 LLTF subcarriers).
- Each pair `(I[k], Q[k])` represents the in-phase and quadrature components for subcarrier `k`.

### Step 4.3 — Verify Serial Output

After flashing, verify that each receiver outputs CSI data:

```bash
# On macOS, list connected ESP32 serial ports:
ls /dev/cu.usbserial* /dev/cu.SLAB* /dev/cu.wchusbserial* /dev/cu.usbmodem* 2>/dev/null

# Monitor one receiver's output:
screen /dev/cu.usbserial-XXXX 921600
```

You should see continuous `CSI_DATA,...` lines. Press `Ctrl+A` then `K` to exit screen.

### Step 4.4 — Capture Raw CSI to CSV (Validation)

Use `gateway_csv_test.py` to record raw CSI from one receiver into a CSV file:

```bash
python gateway_csv_test.py -p /dev/cu.usbserial-XXXX -d 30 -o example_csi.csv
```

- `-p`: serial port path (auto-detects if omitted)
- `-d`: capture duration in seconds
- `-o`: output CSV file

**Output format (no header):**
- Columns 0-22: ESP32 metadata fields
- Column 23: elapsed time (seconds, float)
- Column 24: `first_word` flag
- Column 25: raw CSI I/Q array string `"[I0,Q0,I1,Q1,...,I63,Q63]"`

---

## 5. Layer 2 — Processing Layer (Signal Processing Pipeline)

### Step 5.1 — Gateway / Data Ingestor (`gateway.py`)

**File:** `gateway.py` (477 lines)
**Purpose:** Receive, synchronize, and convert raw CSI packets from 3 ESP32 receivers into a structured amplitude matrix.

**Input:** Raw serial CSV lines from ESP32 `csi_recv` firmware.
**Output:** Raw Amplitude Matrix — shape `[3 receivers x 64 subcarriers x T time samples]`.

#### 5.1.1 — Key Classes

**`CSIPacket` (dataclass):**
Holds one parsed CSI measurement.

| Field | Type | Description |
|-------|------|-------------|
| `receiver_index` | `int` | Receiver ID (0, 1, or 2) |
| `seq_id` | `int` | Firmware sequence number |
| `mac` | `str` | Source MAC address |
| `rssi` | `int` | Signal strength (dBm) |
| `channel` | `int` | Wi-Fi channel |
| `timestamp` | `int` | Firmware timestamp (microseconds) |
| `amplitude` | `np.ndarray` | Shape `(64,)`, float32 — per-subcarrier amplitude |
| `raw_iq` | `np.ndarray` | Shape `(64, 2)`, int16 — raw I and Q values |

**`CSIParser`:**
Stateless parser that converts one CSV line into a `CSIPacket`.

- Filters out non-`CSI_DATA` lines, firmware log messages, and malformed data.
- Extracts the CSI array by locating `[` and `]` brackets in the line.
- Parses I/Q pairs and computes amplitude: `amplitude[k] = sqrt(I[k]^2 + Q[k]^2)`.
- Uses only the first 128 values (64 LLTF subcarrier pairs), ignoring HT-LTF data.

**`AmplitudeMatrix`:**
Thread-safe ring buffer of shape `[3 x 64 x max_time]`.

- Pre-allocated numpy array (default `max_time=1000`, ~10 seconds at 100 Hz).
- Per-receiver write heads track position independently.
- `append(receiver_index, amplitude)`: writes one `(64,)` sample.
- `get_latest(n)`: returns the last `n` samples across all receivers as `(3, 64, n)`. Zero-filled if insufficient data.
- All reads and writes are protected by `threading.Lock`.

**`SerialReader`:**
One instance per receiver, runs in a dedicated daemon thread.

- Opens the serial port at 921600 baud.
- Reads lines, passes them to `CSIParser`, stores amplitude in the shared matrix.
- Tracks sequence continuity — logs warnings on packet loss.
- Maintains per-reader stats: `received`, `lost`, `errors`.

**`Gateway`:**
Top-level orchestrator.

- Auto-detects ESP32 USB serial ports on macOS by scanning glob patterns:
  - `/dev/cu.usbserial*` (CP2102 / FTDI)
  - `/dev/cu.SLAB*` (Silicon Labs CP210x)
  - `/dev/cu.wchusbserial*` (CH340)
  - `/dev/cu.usbmodem*` (Native USB CDC)
- Filters out Bluetooth and debug ports.
- Creates 3 `SerialReader` threads, one per receiver.
- Provides `get_matrix()` for downstream consumers.

#### 5.1.2 — Usage

```python
from gateway import Gateway

gw = Gateway()                          # auto-detect ports
# gw = Gateway(ports=["/dev/cu.usbserial-0001", ...])  # or explicit
gw.start()

matrix = gw.get_matrix()
data = matrix.get_latest(200)           # shape (3, 64, 200) — last 2 seconds
print(gw.get_stats())                   # per-receiver packet/loss counts

gw.stop()
```

Standalone test:
```bash
python gateway.py
# Prints live stats every 2 seconds. Ctrl+C to stop.
```

---

### Step 5.2 — Signal Cleaner / DSP Preprocessing (`signal_cleaner.py`)

**File:** `signal_cleaner.py` (150 lines)
**Purpose:** Remove hardware artifacts, static reflections, and irrelevant noise from the raw amplitude matrix. Preserves the human-motion frequency band (0.1–4.0 Hz).

**Input:** Raw Amplitude Matrix — shape `[3 x 64 x T]`.
**Output:** Cleaned Motion Matrix — same shape.

#### 5.2.1 — Processing Steps

**Hampel Filter (outlier removal):**
For each subcarrier time-series independently:
1. Slide a window of `2 * hampel_window + 1` samples (default window half-size = 3).
2. Compute the local median and MAD (Median Absolute Deviation).
3. If a sample deviates from the median by more than `3 * 1.4826 * MAD`, replace it with the median.
4. The constant `1.4826` converts MAD to an estimate of standard deviation for Gaussian data.

**Butterworth Bandpass Filter:**
- 4th-order Butterworth bandpass: `[0.1 Hz, 4.0 Hz]`.
- Applied using zero-phase filtering (`scipy.signal.sosfiltfilt`).
- Removes DC/static components (walls, furniture reflections) below 0.1 Hz.
- Suppresses high-frequency electronic noise above 4.0 Hz.
- Preserves: breathing (0.1–0.5 Hz), gestures (0.5–2.0 Hz), body movement (2.0–4.0 Hz).
- Requires at least 25 time samples (due to `sosfiltfilt` padding). If fewer samples are available, only the Hampel filter is applied.

**SOS coefficients** are pre-computed once during `__init__` and reused for all signals.

#### 5.2.2 — Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fs` | 100.0 Hz | Sampling frequency (ESP32 packet rate) |
| `lowcut` | 0.1 Hz | Bandpass lower cutoff |
| `highcut` | 4.0 Hz | Bandpass upper cutoff |
| `butter_order` | 4 | Butterworth filter order |
| `hampel_window` | 3 | Hampel half-window size |
| `hampel_threshold` | 3.0 | Outlier detection threshold (multiples of sigma) |

#### 5.2.3 — Usage

```python
from signal_cleaner import SignalCleaner

cleaner = SignalCleaner(fs=100.0)
cleaned = cleaner.clean(raw_matrix)     # (3, 64, T) -> (3, 64, T)
```

Offline test (from CSV):
```bash
python csi_clean_test.py -i example_csi.csv -o cleaned_csi.csv
```

---

### Step 5.3 — Feature Extractor ("The Translator") (`feature_extractor.py`)

**File:** `feature_extractor.py` (219 lines)
**Purpose:** Convert cleaned CSI signals into a compact 6-element state vector (fingerprint) via PCA dimensionality reduction and STFT spectral analysis.

**Input:** Cleaned Motion Matrix — shape `[3 x 64 x T]` (typically last 2 seconds = 200 samples).
**Output:** Current State Vector — shape `(6,)`.

#### 5.3.1 — Processing Steps

**Step A — PCA per receiver (64 subcarriers -> 1 principal component):**

For each receiver independently:
1. Take the `(64, T)` subcarrier matrix.
2. Center each subcarrier by subtracting its mean across time.
3. Compute the economy SVD of the centered matrix.
4. Extract the first principal component: `pc1[t] = S[0] * Vt[0, t]`.
5. This captures the dominant variance pattern shared across all 64 subcarriers.

Result: 3 time-series of length `T`, one per receiver.

**Step B — Per-receiver variance:**

Compute `np.var(pc1)` for each receiver's principal component signal. These three values (features 4–6) capture how much motion-induced energy is seen at each spatial receiver position.

**Step C — Combine across receivers:**

Average the 3 PC1 signals element-wise into a single combined signal of length `T`.

**Step D — STFT spectral analysis:**

- If `T >= stft_nperseg` (128): apply `scipy.signal.stft` with the configured window and overlap, then compute the mean power spectrum across STFT time frames.
- If `T < 128`: fall back to a simple FFT (`np.fft.rfft`).

Result: `(frequencies, power_spectrum)` arrays.

**Step E — Extract scalar features from the spectrum:**

| Index | Feature | Computation |
|-------|---------|-------------|
| 0 | `breathing_frequency` | Peak frequency in the 0.1–0.5 Hz band |
| 1 | `total_energy` | Sum of power across all frequency bins |
| 2 | `doppler_mean` | Spectral centroid (power-weighted mean frequency) |
| 3 | `variance_rx1` | Variance of Rx0's first principal component |
| 4 | `variance_rx2` | Variance of Rx1's first principal component |
| 5 | `variance_rx3` | Variance of Rx2's first principal component |

#### 5.3.2 — Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fs` | 100.0 Hz | Sampling frequency |
| `window_samples` | 200 | Expected input buffer length (2 sec * 100 Hz) |
| `stft_nperseg` | 128 | STFT segment length in samples |
| `stft_noverlap` | 96 | STFT overlap in samples |

#### 5.3.3 — Usage

```python
from feature_extractor import FeatureExtractor

extractor = FeatureExtractor(fs=100.0)
state_vector = extractor.extract(cleaned_matrix)    # (6,) ndarray
state_dict = extractor.extract_dict(cleaned_matrix) # dict with named keys
```

---

### Step 5.4 — Live End-to-End Pipeline Test (`feature_extractor_test.py`)

Connects all Layer 2 components for a live test with real ESP32 hardware.

```bash
python feature_extractor_test.py -d 60 -o features.csv
```

**Pipeline flow:**
1. Starts the Gateway (auto-detects 3 ESP32 serial ports).
2. Waits 3 seconds for the buffer to fill (warmup).
3. Every 0.5 seconds:
   - Pulls the latest 200 samples (2 seconds) from the amplitude matrix.
   - Runs `SignalCleaner.clean()`.
   - Runs `FeatureExtractor.extract()`.
   - Prints the 6-element state vector to the console.
   - Appends the state vector to a list.
4. On completion (or Ctrl+C), stops the Gateway and writes all state vectors to a CSV file.

**Output CSV columns:** `time_sec, breathing_frequency, total_energy, doppler_mean, variance_rx1, variance_rx2, variance_rx3`

---

### Step 5.5 — 3D CSI Visualization (`csiTest.py`)

Visualizes CSI amplitude as a 3D surface plot (time x subcarrier x amplitude).

```bash
# First capture or clean CSI data:
python gateway_csv_test.py -d 30 -o example_csi.csv
python csi_clean_test.py -i example_csi.csv -o cleaned_csi.csv

# Then visualize:
python csiTest.py
```

- Reads `cleaned_csi.csv` (or modify the filename in the script).
- Parses I/Q or amplitude data from column 25.
- Renders a `matplotlib` 3D surface plot with time on X-axis, subcarrier index on Y-axis, and amplitude on Z-axis.
- Static multipath appears as stable "peaks"; human motion appears as dynamic distortions.

---

## 6. Layer 3 — Cognitive Layer (Memory, Reasoning, UI)

### Step 6.1 — Memory Bank (ChromaDB Vector Database)

**File:** `memory_bank.py` (Layer 3, Component 1)
**Config:** `config.py`
**Infrastructure:** `docker-compose.yml`
**Status:** Complete

**Purpose:** Store spatial/behavioral fingerprints and match the current state vector against known patterns using cosine similarity. ChromaDB runs as a Docker container; the Python client communicates over HTTP.

#### 6.1.1 — Starting ChromaDB

```bash
# Start the ChromaDB container (port 8000, persistent volume):
docker compose up -d

# Verify:
curl http://localhost:8000/api/v1/heartbeat

# Stop:
docker compose down

# Stop and delete all data:
docker compose down -v
```

The `docker-compose.yml` uses the official `chromadb/chroma` image with:
- Named volume `ghost_chroma_data` for persistence
- `IS_PERSISTENT=TRUE` for on-disk storage
- `ANONYMIZED_TELEMETRY=FALSE`
- `restart: unless-stopped`

#### 6.1.2 — Configuration (`config.py`)

Connection settings are defined as a frozen dataclass with environment variable overrides:

```python
from config import CHROMA_SETTINGS

# Defaults: host="localhost", port=8000, collection="spatial_fingerprints"
# Override via: GHOST_CHROMA_HOST, GHOST_CHROMA_PORT, GHOST_CHROMA_COLLECTION
```

#### 6.1.3 — Database Schema

Each stored fingerprint consists of a 6-element embedding (the state vector) plus metadata:

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Auto-generated: `{zone}_{activity}_{uuid8}` |
| `embedding` | float[6] | State vector from FeatureExtractor |
| `metadata.zone` | string | Zone label (e.g., "Kitchen") |
| `metadata.activity` | string | Activity label (e.g., "Static") |
| `metadata.label` | string | Human-readable description |
| `metadata.timestamp` | string | ISO 8601 enrollment time |

The collection uses `"hnsw:space": "cosine"` for cosine similarity search.

#### 6.1.4 — MemoryBank Class (`memory_bank.py`)

```python
from memory_bank import MemoryBank

bank = MemoryBank()  # connects to ChromaDB at localhost:8000

# Enroll a single fingerprint:
fid = bank.enroll(state_vector, zone="Kitchen", activity="Static")

# Enroll a batch (e.g., 60 vectors from a 30-second capture):
ids = bank.enroll_batch(vectors, zones, activities)

# Query — returns ranked candidates by cosine similarity:
candidates = bank.query(state_vector, n_results=3)
# [Candidate(zone="Kitchen", activity="Static", score=0.95, ...)]

# Query with zone filter:
candidates = bank.query(state_vector, zone_filter="Bedroom")

# Utilities:
bank.count()          # number of stored fingerprints
bank.get_all_zones()  # ["Bedroom", "Hall", "Kitchen"]
bank.health_check()   # True if ChromaDB is reachable
bank.clear()          # delete all fingerprints
```

The `Candidate` dataclass contains: `zone`, `activity`, `label`, `score` (0.0–1.0), `fingerprint_id`.

ChromaDB returns cosine **distance** (0 = identical). The module converts to **similarity**: `score = 1.0 - distance`.

#### 6.1.5 — Fingerprint Enrollment Workflow (`enrollment.py`)

**Live enrollment** (with ESP32 hardware connected):

```bash
# Enroll 30 seconds of "Static in Kitchen":
python enrollment.py --zone Kitchen --activity Static --duration 30

# Enroll "Walking in Hall" for 60 seconds:
python enrollment.py --zone Hall --activity Walking --duration 60

# Enroll "Empty" baseline (no person):
python enrollment.py --zone Empty --activity None --duration 30
```

**Offline enrollment** (from a previously captured CSV):

```bash
# First capture features:
python feature_extractor_test.py -d 30 -o kitchen_static.csv

# Then enroll from the CSV:
python enrollment.py --from-csv kitchen_static.csv --zone Kitchen --activity Static
```

**Utility commands:**

```bash
python enrollment.py --list-zones    # show enrolled zones
python enrollment.py --count         # show fingerprint count
python enrollment.py --clear         # delete all fingerprints
```

**Recommended enrollment procedure:**

```
For each zone Z in {Kitchen, Hall, Bedroom, ...}:
    1. Place a person in zone Z
    2. For each activity A in {Static, Walking, Breathing}:
        a. python enrollment.py --zone Z --activity A --duration 30
    3. Also enroll an "Empty" baseline with no person present:
        python enrollment.py --zone Empty --activity None --duration 30
```

---

### Step 6.2 — Agentic Core (Llama 3 Reasoning Engine via Ollama)

**File:** `agentic_core.py` (Layer 3, Component 2)
**Config:** `config.py` (`OllamaSettings`)
**Infrastructure:** `docker-compose.yml` (Ollama service)
**Status:** Complete

**Purpose:** Fuse sensor evidence, memory similarity scores, and temporal context into a final structured decision about human presence, location, and activity using a local Llama 3 model running in Ollama.

#### 6.2.1 — Model Setup (Ollama)

Ollama runs as a Docker service alongside ChromaDB. The model is auto-pulled on first use.

```bash
# Start services (Ollama on port 11434):
docker compose up -d

# Verify Ollama is running:
curl http://localhost:11434/api/tags

# The model (llama3.2:3b, ~2 GB) is pulled automatically on first use.
# To change the model, set the environment variable:
#   GHOST_OLLAMA_MODEL=llama3:8b
```

**Configuration** (`config.py`):
```python
from config import OLLAMA_SETTINGS

# Defaults: host="localhost", port=11434, model="llama3.2:3b", timeout=30.0
# Override via: GHOST_OLLAMA_HOST, GHOST_OLLAMA_PORT, GHOST_OLLAMA_MODEL, GHOST_OLLAMA_TIMEOUT
```

#### 6.2.2 — AgenticCore Class (`agentic_core.py`)

```python
from agentic_core import AgenticCore, Decision
from memory_bank import MemoryBank
import numpy as np

core = AgenticCore()             # connects to Ollama, auto-pulls model
bank = MemoryBank()

# Run reasoning:
state_vector = np.array([0.3, 1500.0, 0.8, 0.012, 0.008, 0.015])
candidates = bank.query(state_vector, n_results=3)
decision = core.reason(state_vector, candidates)

print(decision.to_dict())
# {"Target_Detected": true, "Location": "Kitchen", "Activity": "Static",
#  "Confidence": 0.92, "Timestamp": 1700000000000}

# With temporal continuity (pass previous decision):
decision2 = core.reason(new_vector, new_candidates, previous_decision=decision)

# Health check:
core.health_check()  # True if Ollama is reachable
```

**Decision dataclass** fields:
- `target_detected` (bool) — whether a human is detected
- `location` (str) — inferred zone name
- `activity` (str) — inferred activity class
- `confidence` (float) — 0.0–1.0 confidence score
- `timestamp_ms` (int) — Unix timestamp in milliseconds
- `to_dict()` — returns the Phase A JSON schema

#### 6.2.3 — Prompt Design

The LLM receives a **system message** (you are a Wi-Fi sensing engine, respond with JSON only) and a **user message** with four sections:

- **SENSOR DATA**: The 6 feature values with labels and units
- **MEMORY MATCHES**: Ranked candidates with zone, activity, and cosine similarity scores
- **PREVIOUS STATE**: Last location and activity (or "first inference")
- **ANALYSIS RULES**: Domain heuristics (breathing frequency range, energy thresholds, variance asymmetry)

Ollama's `format="json"` parameter constrains the model to output valid JSON.

#### 6.2.4 — Response Parsing

The parser uses a multi-stage approach:
1. **Primary**: Direct `json.loads()` (Ollama `format="json"` produces valid JSON)
2. **Secondary**: Regex `\{[^{}]*\}` extraction if JSON is wrapped in text
3. **Validation**: Checks required fields, clamps confidence to [0, 1]
4. **Fallback**: Returns `Decision(target_detected=False, location="Unknown", ...)` on any failure

#### 6.2.5 — Output Schema

```json
{
    "Target_Detected": true,
    "Location": "Kitchen",
    "Activity": "Breathing/Static",
    "Confidence": 0.92,
    "Timestamp": 1700000000000
}
```

| Field | Type | Description |
|-------|------|-------------|
| `Target_Detected` | bool | Whether a human is detected |
| `Location` | string | Inferred zone (from memory bank zones) |
| `Activity` | string | Inferred activity class |
| `Confidence` | float | 0.0–1.0 confidence score |
| `Timestamp` | int | Unix timestamp in milliseconds |

#### 6.2.6 — Testing

```bash
docker compose up -d
python agentic_core_test.py
# First run pulls the model (~2 GB). All 11 test groups should pass.
```

---

### Step 6.3 — Dashboard (Frontend Visualization)

**Purpose:** Present system decisions and live signals on an interactive floorplan.

#### 6.3.1 — UI Components

| Component | Description |
|-----------|-------------|
| Floorplan canvas | Building layout with labeled rooms/zones |
| Red dot | Detected human position on floorplan |
| Green dot | Scanner/transmitter position |
| Distance label | Estimated distance from scanner to detected target |
| Breathing waveform | Live plot of the first principal component signal |
| Status log panel | Connection status, Tx/Rx activity indicators, last update time |
| Compass | Current map orientation / point-of-view angle |
| Control buttons | On/Off, Scan, Cancel |

#### 6.3.2 — Implementation Approach

The dashboard can be implemented as:

**Option A — Python (Matplotlib/Tkinter/PyQt5):**
- Fastest to prototype.
- Use `matplotlib.animation` for live updates.
- Suitable for a laptop-based demonstration.

**Option B — Web-based (Flask/FastAPI + HTML/JS):**
- Backend serves state vectors and decisions via a REST API or WebSocket.
- Frontend renders the floorplan using HTML5 Canvas or a JS framework.
- Suitable for tablet deployment.

**Option C — Streamlit:**
- Rapid prototyping with `st.plotly_chart` or `st.pydeck_chart`.
- Less control over real-time updates.

#### 6.3.3 — Dashboard Data Inputs

The dashboard consumes:
1. **Final Decision JSON** from the Agentic Core — for presence, location, activity indicators.
2. **Live PC1 signal** from the Feature Extractor — for the breathing waveform plot.
3. **Gateway stats** — for connection status and packet loss indicators.

---

## 7. Integration and End-to-End Pipeline

### Step 7.1 — Main Pipeline Orchestrator

A main script ties all layers together in a continuous loop.

**Pseudocode:**

```python
# 1. Initialize all components
gateway = Gateway()
gateway.start()
matrix = gateway.get_matrix()

cleaner = SignalCleaner(fs=100.0)
extractor = FeatureExtractor(fs=100.0)

memory = MemoryBank()
core = AgenticCore()

previous_decision = None

# 2. Wait for buffer warmup
time.sleep(3.0)

# 3. Main sensing loop
while running:
    # Layer 2: Signal processing
    raw = matrix.get_latest(200)            # (3, 64, 200) — last 2 seconds
    cleaned = cleaner.clean(raw)
    state_vector = extractor.extract(cleaned)

    # Layer 3: Memory lookup
    candidates = memory.query(state_vector, n_results=3)

    # Layer 3: LLM reasoning
    decision = core.reason(state_vector, candidates, previous_decision)

    # Layer 3: Update dashboard
    dashboard.update(decision, state_vector)

    # Update state
    previous_decision = decision

    time.sleep(0.5)  # extract every 0.5 seconds

# 4. Cleanup
gateway.stop()
```

### Step 7.2 — Configuration Constants (System-Wide)

| Constant | Value | Description |
|----------|-------|-------------|
| Wi-Fi channel | 6 | 2.4 GHz channel |
| Bandwidth | HT40 (40 MHz) | Wi-Fi bandwidth mode |
| Packet rate | ~100 Hz | ESP32 transmission rate |
| Baud rate | 921600 | Serial communication speed |
| Subcarriers | 64 | LLTF OFDM subcarriers |
| Receivers | 3 | Number of ESP32 receivers |
| Buffer depth | 1000 | Ring buffer (~10 sec at 100 Hz) |
| Extraction window | 2.0 sec (200 samples) | Data window per extraction |
| Extraction interval | 0.5 sec | How often features are extracted |
| Bandpass range | 0.1–4.0 Hz | Human motion frequency band |
| Breathing band | 0.1–0.5 Hz | Breathing detection range |
| STFT segment | 128 samples | STFT window length |
| STFT overlap | 96 samples | STFT window overlap |
| Inter-receiver spacing | ~1 meter | Physical spacing between receivers |

---

## 8. Testing and Validation

### Step 8.1 — Layer 1 Tests

| Test | How | Expected Result |
|------|-----|-----------------|
| Serial output verification | `screen /dev/cu.usbserial-XXXX 921600` | Continuous `CSI_DATA,...` lines |
| Packet rate measurement | `gateway_csv_test.py -d 10` | ~100 packets/sec per receiver |
| All 3 receivers active | `python gateway.py` | Stats show 3 active receivers |
| Packet loss rate | Check `lost` in `gw.get_stats()` | < 5% under normal conditions |

### Step 8.2 — Layer 2 Tests

| Test | How | Expected Result |
|------|-----|-----------------|
| CSI parsing correctness | Inspect `CSIPacket` fields | 64 I/Q pairs, valid RSSI/channel |
| Amplitude computation | Compare `sqrt(I^2+Q^2)` manually | Values match |
| Signal cleaning | `csi_clean_test.py` | Reduced noise, preserved motion band |
| PCA dimensionality reduction | Check extractor output | 1 PC per receiver, captures dominant variance |
| STFT output | Inspect spectral features | Breathing frequency in [0.1, 0.5] Hz range |
| Static vs. dynamic distinction | Compare empty room vs. person present | Clear energy/variance difference |
| End-to-end pipeline | `feature_extractor_test.py -d 30` | 6-element state vectors exported to CSV |

### Step 8.3 — Layer 3 Tests

| Test | How | Expected Result |
|------|-----|-----------------|
| ChromaDB storage/retrieval | Store and query known fingerprints | Correct zone returned with high similarity |
| LLM prompt → JSON output | Feed known state vectors | Valid JSON with expected fields |
| Empty room detection | No person in sensing area | `Target_Detected: false` |
| Presence detection | Person standing in room | `Target_Detected: true`, correct location |
| Activity classification | Person walking vs. standing | Different `Activity` values |
| Dashboard rendering | Launch dashboard with live data | Floorplan with indicators updates in real time |

### Step 8.4 — System-Level Tests

| Test | How | Expected Result |
|------|-----|-----------------|
| End-to-end latency | Measure time from CSI capture to dashboard update | Suitable for live monitoring |
| Continuous operation | Run system for 30+ minutes | No crashes, memory leaks, or drift |
| Wi-Fi interference resilience | Run with other Wi-Fi devices active | Motion detection still functional |
| Offline replay | Load saved CSV through pipeline | Reproducible results |

---

## 9. File Inventory

### Root Directory

| File | Layer | Purpose |
|------|-------|---------|
| `gateway.py` | L2.1 | Data Ingestor — serial parsing, amplitude matrix |
| `signal_cleaner.py` | L2.2 | DSP Preprocessing — Hampel + Butterworth bandpass |
| `feature_extractor.py` | L2.3 | Feature Extraction — PCA + STFT → 6-element state vector |
| `config.py` | L3 | Service connection settings (ChromaSettings, OllamaSettings) |
| `memory_bank.py` | L3.1 | Memory Bank — ChromaDB vector store (enroll, query, batch) |
| `agentic_core.py` | L3.2 | Agentic Core — Llama 3 reasoning engine via Ollama |
| `enrollment.py` | L3 Tool | CLI tool for fingerprint enrollment (live + CSV modes) |
| `feature_extractor_test.py` | Test | Live end-to-end pipeline test with CSV export |
| `memory_bank_test.py` | Test | MemoryBank integration tests (requires running ChromaDB) |
| `agentic_core_test.py` | Test | AgenticCore integration tests (requires running Ollama) |
| `gateway_csv_test.py` | Test | Capture raw CSI from one receiver to CSV |
| `csi_clean_test.py` | Test | Offline signal cleaning test |
| `csiTest.py` | Viz | 3D CSI amplitude surface plot |
| `mock.ipynb` | Explore | Jupyter notebook for early data exploration |

### Infrastructure

| File | Description |
|------|-------------|
| `docker-compose.yml` | ChromaDB + Ollama container definitions (ports 8000, 11434) |
| `requirements.txt` | All Python dependencies |
| `.gitignore` | Root-level gitignore |

### Documentation

| File | Description |
|------|-------------|
| `docs/gateway.md` | Detailed technical documentation for the Gateway component |
| `docs/ghost_system.md` | This implementation guide |

### Firmware (External Library)

| Directory | Description |
|-----------|-------------|
| `esp-csi/examples/get-started/csi_send/` | ESP32 transmitter firmware |
| `esp-csi/examples/get-started/csi_recv/` | ESP32 receiver firmware |
| `esp-csi/examples/get-started/tools/` | Python parsing utilities |
| `esp-csi/docs/` | Educational materials on OFDM, CSI, wireless sensing |