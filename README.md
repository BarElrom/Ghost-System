# GHOST System v2 — Hardware-Injection Mode

v2 runs the whole Wi-Fi CSI sensing pipeline **without working antennas**: it
replays public-dataset CSI *into* ESP32 receivers and reads it back **as if it
were live** — or runs the exact same pipeline **fully in software** with no
boards at all.

The pipeline itself lives **under `v2/`** — code, the demo capture
(`example_csi.csv`), the comparison `datasets/`, `requirements.txt`, firmware, and
tests. This README and the Docker stack (`Dockerfile`, `docker-compose.yml`) sit
at the repo root. This README is the single source of truth for how to run it.

---

## Contents
- [1. What it does](#1-what-it-does-end-to-end)
- [2. Directory map](#2-directory-map)
- [3. Quick start](#3-quick-start)
- [4. Run with Docker](#4-run-with-docker-easiest)
- [5. Run on the host (Python venv)](#5-run-on-the-host-python-venv)
- [6. All run modes](#6-all-run-modes)
  - [6.1 Tests](#61-tests-no-hardware)
  - [6.2 Software-only signal demo](#62-software-only-signal-demo)
  - [6.3 Cognitive pipeline (position + velocity + JSON)](#63-cognitive-pipeline-position--velocity--json)
  - [6.4 Plots & multi-dataset comparison](#64-plots--multi-dataset-comparison)
  - [6.5 Hardware: serial loopback](#65-hardware-serial-loopback-recommended)
  - [6.6 Hardware: Wi-Fi loopback](#66-hardware-wi-fi-loopback-alternative)
- [7. Configuration (env vars)](#7-configuration-env-vars)
- [8. Status & known limitations](#8-status--known-limitations)

---

## 1. What it does (end to end)

```
dataset CSI ──▶ Injector ──▶ ESP32 receiver(s) ──▶ Ghost pipeline
                (Service 1)   (loopback firmware)   (Service 2)

Ghost pipeline:  CSI_DATA  ──▶ gateway_v2         (complex I/Q matrix)
                          ──▶ signal_cleaner_v2   (static subtraction + band-pass)
                          ──▶ feature_extractor_v2(motion + per-node energy + phase)
                          ──▶ localizer           (x/y, velocity, confidence, anomalies)
                          ──▶ agentic_core_v2     (LLM reasoning → coordinates/velocity JSON)
```

Three ways to get data into the pipeline:
- **Software only (no hardware):** a mock ESP32 in-process — proves the whole chain.
- **Serial loopback (recommended hardware path):** inject over the USB cable itself.
- **Wi-Fi loopback:** inject over UDP/Wi-Fi (needs boards + host on one plain 2.4 GHz
  network; won't work on WPA-Enterprise / 5 GHz-only office Wi-Fi).

---

## 2. Directory map

```
.                                 # repo root
├── README.md                     # this file
├── Dockerfile                    # app image (software-only run modes)
├── docker-compose.yml            # ghost app + ollama + chromadb
├── .dockerignore
│
└── v2/                           # the GHOST v2 pipeline package
├── requirements.txt              # Python deps (self-contained)
├── example_csi.csv               # bundled real ESP32 capture (demo dataset)
├── datasets/                     # Intel-respiratory / CSI-Bench clips for the comparison
├── config_v2.py                  # constants + Ollama/Chroma settings + dataset path
├── datasets.py                   # dataset profiles for the comparison runner
│
├── transport/                    # the wire contract
│   ├── frame.py                  # UDP frame codec (G2 format, complex int16 I/Q)
│   ├── udp_sender.py             # injector-side UDP sender
│   └── mock_esp32.py             # software stand-in for the ESP32 (UDP → CSI_DATA)
│
├── ghost/                        # Service 2 — the reader/processing pipeline
│   ├── gateway_v2.py             # complex I/Q buffer + pluggable sources (serial/UDP)
│   ├── sources.py                # ListSource / UDPSource (hardware-free sources)
│   ├── frame_manager.py          # cross-receiver frame alignment
│   ├── signal_cleaner_v2.py      # calibration + static subtraction + band-pass
│   ├── feature_extractor_v2.py   # PCA/FFT features + per-node energy + phase features
│   ├── localizer.py              # x/y position, velocity, confidence, anomaly flags
│   ├── kalman.py                 # constant-velocity smoothing of the position track
│   ├── classifier.py             # Level-B activity/presence/fall label logic
│   ├── agentic_core_v2.py        # Ollama reasoning → coordinates/velocity JSON
│   └── main_v2.py                # end-to-end orchestrator (replay + live serial)
│
├── injector/                     # Service 1 — dataset replay
│   ├── injector.py               # calibration preamble + fan-out + pacing (UDP/serial)
│   └── adapters/
│       ├── base.py               # adapter contract + subcarrier normalization
│       ├── embedded_wifi.py      # ESP32-CSV dataset adapter (parses example_csi.csv)
│       ├── csi_bench.py          # CSI-Bench .mat amplitude/phase adapter
│       └── intel_resp.py         # Intel respiratory CSV adapter
│
├── firmware/                     # ESP32 loopback firmware (ESP-IDF v6.x)
│   ├── csi_inject_serial/        # USB-serial loopback  ← recommended
│   └── csi_inject_recv/          # Wi-Fi (UDP) loopback
│
├── tests/                        # 12 dependency-free test suites (~320 checks)
│
├── run_phase2_demo.py            # software-only signal-pipeline demo on a dataset
├── run_serial_demo.py            # signal pipeline over 3 real boards via USB serial
├── run_serial_cognitive_demo.py  # full cognitive pipeline over 3 real boards
├── run_datasets_compare.py       # compare several datasets through the pipeline
├── viz_decisions.py              # plot replay decisions (trajectory/features)
├── export_slides.py              # export slide-ready before/after figures
├── demo_anomaly_correction.py    # agentic guardrail demo (LLM flags an impossible jump)
├── make_sample_datasets.py       # generate placeholder datasets for the comparison
├── send_test.py                  # Wi-Fi: send test frames to a board IP
└── serial_test.py                # Serial: send INJ frames to a board, read CSI back
```

---

## 3. Quick start

The fastest zero-hardware smoke test:

```bash
# Docker (no Python setup needed — Docker Desktop must be running):
docker compose run --rm ghost

# — or — host Python:
pip install -r v2/requirements.txt
python v2/run_phase2_demo.py
```

Both replay the bundled `example_csi.csv` through the full software pipeline and
print a signal-cleaning report + extracted features.

---

## 4. Run with Docker (easiest)

Docker gives you the pipeline **and** the AI layer (Ollama) + memory bank
(ChromaDB) with no local Python/Ollama install. All commands run from the repo
root. **Docker Desktop must be running first.**

**Software pipeline demo (default command):**
```bash
docker compose run --rm ghost
```

**Run the tests:**
```bash
docker compose run --rm ghost sh -c 'for t in v2/tests/test_*.py; do python "$t"; done'
```

**Any run script** (override the container command):
```bash
docker compose run --rm ghost python v2/run_datasets_compare.py
docker compose run --rm ghost python v2/ghost/main_v2.py --replay v2/example_csi.csv \
    --pos-scale embedded_wifi --window 200 --breath-window 600 --step 150
```

**With the AI layer (`--llm`):** start the model service, pull a model once, then
pass `--llm`. The app container reaches Ollama at hostname `ollama` automatically.
```bash
docker compose up -d ollama chromadb
docker compose exec ollama ollama pull llama3.2:3b      # one-time (~2 GB)
docker compose run --rm ghost python v2/ghost/main_v2.py --replay v2/example_csi.csv \
    --pos-scale embedded_wifi --llm
```

**Generated plots/CSVs:** the `ghost` service mounts `./_out` → `/app/v2/_out`, so
write outputs there to get them on the host, e.g. `--plot-out v2/_out/decisions.png`.

**What Docker can't do:** the serial and Wi-Fi *hardware* modes need direct USB /
board access, which isn't portable in Docker — run those on the host ([§6.5](#65-hardware-serial-loopback-recommended)–[§6.6](#66-hardware-wi-fi-loopback-alternative)).

**Services:**
| Service | Image | Port | Purpose |
|---------|-------|:----:|---------|
| `ghost` | built from `Dockerfile` | — | the v2 pipeline (software run modes) |
| `ollama` | `ollama/ollama` | 11434 | AI reasoning layer (`--llm`) |
| `chromadb` | `chromadb/chroma` | 8000 | memory bank (spatial fingerprints) |

Tear down with `docker compose down` (add `-v` to also drop the model/db volumes).

---

## 5. Run on the host (Python venv)

```bash
cd /Users/barelrom/PycharmProjects/ML/TheGhostSystem
python3 -m venv .venv
./.venv/bin/pip install -r v2/requirements.txt   # numpy, scipy, pyserial, filterpy, ...
```

Run any tool with `./.venv/bin/python v2/<script>.py`. Scripts resolve the bundled
`v2/example_csi.csv` automatically, so they work from **any** working directory.

> **Two-terminal rule (only if you also flash firmware).** ESP-IDF and the project
> venv both want to own `python`. Keep them in separate terminals:
> - **IDF terminal** — where you `source .../export.sh`; use only for `idf.py`.
> - **venv terminal** — a fresh terminal; run Python tools with **`./.venv/bin/python`**.
>
> If you see `ModuleNotFoundError: numpy` right after sourcing IDF, you're using
> IDF's Python — prefix with `./.venv/bin/python`.

---

## 6. All run modes

Below, `PY` means `./.venv/bin/python` (host) or `python` inside a `docker compose
run --rm ghost ...` invocation.

### 6.1 Tests (no hardware)

12 plain-script suites (~320 checks), no pytest. Each prints `PASS`/`FAIL` and
exits non-zero on failure.

```bash
# all at once
for t in v2/tests/test_*.py; do $PY "$t"; done
# or individually
$PY v2/tests/test_gateway_v2.py
```

| Suite | Covers |
|-------|--------|
| `test_frame` | wire codec: complex I/Q encode/decode, clipping, bad-input rejection |
| `test_transport_roundtrip` | UDP → mock ESP32 → CSI_DATA → parser (format compatibility) |
| `test_gateway_v2` | complex ring buffer, parser, serial/UDP/List sources end to end |
| `test_frame_manager` | cross-receiver frame alignment / coverage |
| `test_signal_cleaner_v2` | calibration, complex subtraction, amplitude/phase, band-pass |
| `test_feature_extractor_v2` | PCA/FFT features, per-node energy, phase variance |
| `test_injector` | dataset adapter, calibration preamble, fan-out, injector→gateway |
| `test_csi_bench` | CSI-Bench `.mat` adapter: amplitude/phase → I/Q, subcarrier resample |
| `test_localizer` | centroid, depth/confidence clamps, velocity, anomaly flags |
| `test_kalman` | constant-velocity smoothing of the position track |
| `test_classifier` | Level-B activity/presence/fall label logic |
| `test_agentic_core_v2` | LLM contract (mocked Ollama): JSON parse, validation, fallbacks |

### 6.2 Software-only signal demo

Runs the chain on the real capture using a software mock ESP32 — proves the
signal pipeline (stops at features) with no boards:

```bash
$PY v2/run_phase2_demo.py --path v2/example_csi.csv --max-frames 800 --calib 200
```
Prints a before/after amplitude comparison (proving the static baseline was
removed), a phase-coherence diagnostic, and the extracted feature set.
Options: `--max-frames`, `--calib`, `--rate`, `--dataset`, plus `-v`/`-vv`.

### 6.3 Cognitive pipeline (position + velocity + JSON)

The **full** chain (features → localizer → Kalman → AI layer) runs via
`v2/ghost/main_v2.py`, emitting the coordinates/velocity JSON schema. Two modes.

**Offline replay (no hardware):**
```bash
$PY v2/ghost/main_v2.py --replay v2/example_csi.csv \
    --pos-scale embedded_wifi --window 200 --breath-window 600 --step 150
```
Emits one decision per window:
```
[..] pos=(+0.56, 2.61) m  v=0.05 m/s  conf=1.00  breath=0.500Hz  energies={RX1:753, RX2:676, RX3:602}
```

**Live over three boards (USB serial):**
```bash
$PY v2/ghost/main_v2.py --serial --ports /dev/cu.usbserial-3 /dev/cu.usbserial-0001 /dev/cu.usbserial-4
```
Calibrates from the first `--calib` frames, then streams a decision every
`--interval` seconds until Ctrl+C.

**Two window lengths (by design).** Localization uses the short `--window`
(~2 s / 100 samples @ 50 Hz) for responsiveness; breathing needs the long
`--breath-window` (~10 s / 500 samples) because frequency resolution is `fs/T`
(set `--breath-window 0` to disable).

**Key flags:** `--llm` (route through Ollama — start it first, [§4](#4-run-with-docker-easiest)),
`--metric` (treat positions as metric; default is illustrative/non-metric),
`--no-kalman`, `--pos-scale NAME`, `--plot`/`--plot-out`/`--show`.

### 6.4 Plots & multi-dataset comparison

**Export slide-ready figures** (one before/after story per PNG — great for decks):
```bash
$PY v2/export_slides.py --outdir slides/example          # bundled example_csi.csv
# a dataset with real motion (more dynamic trajectory + a walking clip):
$PY v2/export_slides.py --dataset intel_resp \
    --path v2/datasets/intel_resp/CSI_complex1_aligned_fixedPiOffset_20MHz_OneSittingOneWalking.csv \
    --pos-scale intel_resp --window 200 --breath-window 1000 --step 200 \
    --frames 2000 --calib 200 --outdir slides/intel_walking
```
Writes six standalone PNGs: `01_signal_cleaning` (raw vs cleaned amplitude),
`02_kalman_before_after` (localizer track vs Kalman track + velocity),
`03_breathing_spectrum` (band spectrum + detected peak, or an honest "no peak"),
`04_feature_timeline`, `05_trajectory`, and `06_agentic_before_after` (add `--llm`
to make the agentic panel show real before/after instead of a passthrough).

**Demonstrate the agentic guardrail** (LLM catches an impossible jump):
```bash
docker compose up -d ollama && docker compose exec ollama ollama pull llama3.1:8b   # once
$PY v2/demo_anomaly_correction.py --inject-index 5 --outdir slides/anomaly
```
Injects one physically-impossible window (a teleport at >3 m/s) into the raw
localizer stream, then reasons over it twice — deterministic passthrough vs LLM.
The figure shows the LLM raising `unrealistic_speed` and dropping confidence to 0.3
**while keeping x/y/velocity unchanged**, next to the raw localizer reporting the
bad point with no warning. Needs a capable model — `llama3.2:3b` echoes the input
and will NOT flag it, so this defaults to `--model llama3.1:8b`.

**Plot one replay** (all panels in a single dashboard):
```bash
$PY v2/viz_decisions.py --path v2/example_csi.csv --pos-scale embedded_wifi --out decisions.png
```

**Compare several datasets** (skips any whose file is missing):
```bash
$PY v2/run_datasets_compare.py                          # all available
$PY v2/run_datasets_compare.py --only embedded_wifi     # a subset
$PY v2/run_datasets_compare.py --llm --show             # AI layer + interactive
```
Only `embedded_wifi` ships with the repo. To add CSI-Bench / Intel-respiratory
cases, drop their files under `v2/datasets/...` (paths are in `datasets.py`).
`make_sample_datasets.py` can generate placeholder files to exercise the runner.

### 6.5 Hardware: serial loopback (recommended)

No Wi-Fi, no IPs, no network. Each board is one USB cable; port order = RX1/RX2/RX3.

**Source ESP-IDF (IDF terminal, once):**
```bash
source /Users/barelrom/.espressif/v6.0.1/esp-idf/export.sh
```

**Flash the serial firmware to each board** (identical firmware; only `-p` differs):
```bash
cd v2/firmware/csi_inject_serial
idf.py set-target esp32          # first time only
ls /dev/cu.usbserial*            # find the three ports
idf.py -p /dev/cu.usbserial-3    flash
idf.py -p /dev/cu.usbserial-0001 flash
idf.py -p /dev/cu.usbserial-4    flash
```
> Close any `screen` / `idf.py monitor` before flashing or running the Python tools
> (`lsof /dev/cu.usbserial-3` shows what's holding a port).

**Smoke-test one board (venv terminal):**
```bash
$PY v2/serial_test.py /dev/cu.usbserial-3
```
Expect `DONE: sent 10, received 10 CSI_DATA lines back`.

**Signal pipeline over all 3 boards:**
```bash
$PY v2/run_serial_demo.py --ports /dev/cu.usbserial-3 /dev/cu.usbserial-0001 /dev/cu.usbserial-4
```
Options: `--calib`, `--frames`, `--rate`, `--baud`, `--path`.

**Full cognitive pipeline over all 3 boards** (adds localizer + AI layer):
```bash
$PY v2/run_serial_cognitive_demo.py \
    --ports /dev/cu.usbserial-0001 /dev/cu.usbserial-3 /dev/cu.usbserial-5 \
    --pos-scale embedded_wifi          # add --llm to route through Ollama
```
Ports auto-detect if `--ports` is omitted.

### 6.6 Hardware: Wi-Fi loopback (alternative)

Only if the boards + host share a plain 2.4 GHz network. Flash
`v2/firmware/csi_inject_recv` (set SSID/password/NODE_ID via `idf.py menuconfig`),
then point the injector at the boards' IPs:
```bash
export GHOST_V2_RX1_IP=10.0.0.51 GHOST_V2_RX2_IP=10.0.0.52 GHOST_V2_RX3_IP=10.0.0.53
$PY -m v2.injector.injector --dataset embedded_wifi --path v2/example_csi.csv --rate 100
```
Single-board bring-up:
```bash
$PY v2/send_test.py 10.0.0.51 --node RX1 --count 20
```
Won't work on WPA-Enterprise or 5 GHz-only networks — that's why serial is preferred.

---

## 7. Configuration (env vars)

All defaults live in `config_v2.py`; override without editing code:

| Variable | Default | Meaning |
|----------|---------|---------|
| `GHOST_V2_DATASET` | bundled `v2/example_csi.csv` | default dataset path for the runners |
| `GHOST_V2_UDP_PORT` | `5005` | Wi-Fi injection UDP port |
| `GHOST_V2_RX{1,2,3}_IP` | `127.0.0.1` | board IPs for Wi-Fi loopback |
| `GHOST_V2_CALIBRATION_SAMPLES` | `200` | calibration preamble length |
| `GHOST_V2_PHASE_SANITIZE` | `1` | remove per-packet CFO/STO phase ramp |
| `GHOST_V2_STATIC_EWMA_ALPHA` | `0.02` | adaptive static-baseline memory (0 = fixed) |
| `GHOST_V2_KALMAN` | `1` | enable position-track Kalman smoothing |
| `GHOST_V2_POS_SCALE` | `plan` | default `POS_ENERGY_SCALES` entry |
| `GHOST_OLLAMA_HOST` / `_PORT` / `_MODEL` | `localhost` / `11434` / `llama3.2:3b` | AI layer connection |
| `GHOST_CHROMA_HOST` / `_PORT` | `localhost` / `8000` | memory bank connection |

Inside Docker, the compose file sets `GHOST_OLLAMA_HOST=ollama` and
`GHOST_CHROMA_HOST=chromadb` for you.

---

## 8. Status & known limitations

**Working:** wire transport, complex gateway, frame alignment, signal cleaner,
feature extractor, localizer (x/y/velocity/anomaly), Kalman smoothing, Level-B
classifier, AI layer (coordinates/velocity JSON via Ollama, with a deterministic
no-model passthrough), end-to-end orchestrator (`main_v2.py`, replay + live
serial), injector + Embedded-WiFi / CSI-Bench / Intel-respiratory adapters, both
firmwares (serial verified on real 3-board hardware end to end), 12 passing test
suites, software + hardware demos, multi-dataset comparison, and the Docker stack.

**Not yet built:** dashboard; remaining dataset adapters (Gi-z). The AI layer's
`--llm` path is covered by mocked tests; live-model verification against Ollama is
a manual step.

**Model capability matters for reasoning.** `llama3.2:3b` is fine for the label
passthrough but **cannot apply the sanity rules** — in JSON mode it echoes the
input, so it will not flag an implausible-speed window (verified). Reasoning that
must *act* on the numbers (e.g. the anomaly guardrail in
`demo_anomaly_correction.py`) needs a stronger model such as `llama3.1:8b`
(`GHOST_OLLAMA_MODEL=llama3.1:8b`, or `--model`).

**Position is illustrative, not survey-grade.** The localizer's x/y come from an
energy-weighted centroid + a backscatter depth heuristic whose constants are
per-dataset (`POS_ENERGY_SCALES`), and they are only metric with genuine
multi-antenna diversity — single-link datasets are flagged `non_metric`.

**Findings baked into the code:**
- **Phase incoherence (real-data finding).** Raw ESP32 CSI has a random per-packet
  phase, so the complex static baseline partially cancels; the band-pass still
  recovers motion, but complex subtraction needs phase sanitization (CSI-ratio)
  before phase features are fully trustworthy.
- **Breathing needs a long window, and reports 0.0 when absent.** Frequency
  resolution is `fs/T`; resolving 0.1–0.5 Hz needs a ~10 s (~500-frame @ 50 Hz)
  window, not the 2 s motion window. The estimator now detrends + Hann-windows
  the signal and accepts only a *prominent interior* spectral peak, so a peakless
  1/f drift honestly returns `0.0 Hz` instead of the old phantom 0.1 Hz floor.
  On the fanned single-link datasets a still person's respiration is generally
  not recoverable, so "present-still vs empty" cannot be separated by motion or
  breathing alone (e.g. the Intel `sitting` clip classifies as `empty_room`).
- **Console baud.** The serial firmware runs at 115200; raising it (an open item)
  allows higher `--rate` and longer windows.
</content>
