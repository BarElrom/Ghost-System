# GHOST System v2 — Hardware-Injection (Loopback) Mode

**Wi-Fi CSI Human Sensing — Dataset Injection Architecture**
**Authors:** Bar Elrom, Yuval Lerfeld · **Advisor:** Mr. Ilya Zeldner — Braude College of Engineering
**Status:** Design (no code yet) · **Scope:** everything new lives under `v2/`. **Nothing in the existing root code is modified or overridden.** Where behavior must differ, we **duplicate** the file into `v2/` and edit the copy.

---

## 0. Why v2 exists

The RF hardware (antennas + live signals) is unreliable, so we cannot capture real CSI right now. Instead we **inject known CSI (from public datasets) into the real ESP32 receivers** and let the ghost system read it back **as if it were live**. This validates the entire software+firmware pipeline before the antennas are fixed, and gives us a repeatable, presentable demo driven by curated datasets.

This is **Option B (firmware loopback)**: the data physically passes through the real ESP32 devices, so the demo shows real hardware in the loop — not a pure software mock.

---

## 1. Decisions locked in

| # | Question | Decision |
|---|----------|----------|
| 1 | Injection style | **Option B — firmware loopback.** External injector service pushes dataset CSI into the ESP32 receivers; ghost reads from the receivers. |
| 2 | Transport | **UDP** on the injection side. See §2 for the USB caveat: UDP runs over **Wi-Fi into the ESP32**; the ESP32 → host link stays **USB serial**. |
| 3 | Buffer data model | **Full complex I/Q** carried through the whole pipeline (amplitude & phase derived downstream). |
| 4 | Calibration | **Yes — explicit static-baseline calibration.** Injector streams a calibration preamble; ghost computes `H_static`. Fallback = temporal-mean estimate. See §7.1 and §11. |
| 5 | LLM output schema | **Coordinates / velocity schema** (`x`, `y`, `velocity_m_s`, `signal_confidence`, `anomaly_flags`, `raw_node_energies`). |
| 6 | LLM runtime | **Ollama in Docker**, same as today (containerized). No LM Studio. |

**Hard rule:** all new/changed code goes in `v2/`. Root files (`gateway.py`, `signal_cleaner.py`, …) are **read-only references** we copy from, never edit.

---

## 2. Physical & transport architecture (the important part)

### 2.1 The USB reality
A classic ESP32 (ESP32-WROOM, `set-target esp32`) exposes USB through a **CP2102 / CH340 UART bridge**. That link is **serial only** — it cannot carry IP packets. Therefore:

- **UDP into the ESP32 → over Wi-Fi.** The ESP32's own 2.4 GHz radio (station mode) receives injected frames.
- **ESP32 → host → over USB serial.** The modified firmware emits standard `CSI_DATA,...` CSV lines at 921600 baud, which the ghost gateway reads exactly like real capture.

(If we ever move to an **ESP32-S2/S3** with native USB, USB-CDC networking becomes possible and injection could go over the cable. Not our current target — noted for the future.)

### 2.2 End-to-end data path

```
┌──────────────────────────────────────────────────────────────────────────┐
│  SERVICE 1 — INJECTOR (host process, v2/injector/)                         │
│  reads dataset ──▶ adapter normalizes ──▶ pace @100 Hz ──▶ build I/Q frame │
└──────────────────────────────────────────────────────────────────────────┘
        │  UDP unicast over Wi-Fi  (one datagram per node per frame)
        │  payload: node_id + int16 I/Q[64]  (+ frame_seq, phase flag)
        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  ESP32 Rx ×3  (MODIFIED firmware v2/firmware/csi_inject_recv)              │
│  Wi-Fi STA receives UDP ──▶ format as CSI_DATA CSV ──▶ print over UART      │
└──────────────────────────────────────────────────────────────────────────┘
        │  USB serial, 921600 baud, "CSI_DATA,seq,mac,...,\"[I0,Q0,...]\""
        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  SERVICE 2 — GHOST v2 (host process, v2/ghost/)                            │
│  gateway_v2 (complex I/Q buffer) ──▶ signal_cleaner_v2 (static-subtract +   │
│  bandpass) ──▶ feature_extractor_v2 (+phase) ──▶ localizer (x,y,v) ──▶       │
│  agentic_core_v2 (Ollama, coordinates/velocity JSON) ──▶ dashboard          │
└──────────────────────────────────────────────────────────────────────────┘
```

**Two services, as requested:** the *injector* and the *ghost system* are separate processes that only meet at the ESP32 hardware. Neither imports the other.

### 2.3 Networking specifics
- Host runs a Wi-Fi AP (or shares a LAN) that the 3 ESP32s join as stations. A config maps `RX1/RX2/RX3 → (ip, udp_port)`.
- Injector opens one UDP socket, sends a datagram to each node's `(ip, port)` per frame. `node_id` is embedded so a node ignores frames not addressed to it (belt-and-suspenders if broadcast is used).
- Pace target: **100 Hz** per node (10 ms period) to match the real system's assumed sample rate. Injector sleeps/backpressures to hold rate.

---

## 3. The two services in detail

### 3.1 Service 1 — Injector (`v2/injector/`)
Responsibilities:
1. Load a dataset via the matching **adapter** (§10).
2. Normalize every frame to the pipeline contract: **64 subcarriers, complex I/Q, 100 Hz**.
3. Emit a **calibration preamble** first (§11), then the active recording.
4. Reconstruct I/Q from amplitude+phase when needed: `I = A·cos φ`, `Q = A·sin φ`; quantize to **int16** (not int8 — see §16).
5. Pace at 100 Hz and send UDP to each node.
6. Provide 3 node streams. If the dataset has ≥3 antennas/links, map them to RX1/2/3; if single-link, synthesize node diversity in the adapter (§10.1) and **flag position output as non-metric**.

CLI shape (illustrative): `python -m v2.injector --dataset embedded_wifi --path ./data/... --rate 100 --calib 200`.

### 3.2 Service 2 — Ghost v2 (`v2/ghost/`)
Same 3-layer shape as today, but:
- **gateway_v2** reads USB serial and stores **complex I/Q** (not amplitude-only).
- **signal_cleaner_v2** does complex static subtraction + Hampel + bandpass.
- **feature_extractor_v2** is complex-aware and adds phase features.
- **localizer** computes position/velocity/anomaly.
- **agentic_core_v2** emits the coordinates/velocity JSON.

---

## 4. ESP32 firmware changes (`v2/firmware/csi_inject_recv/`)

Duplicate `esp-csi/examples/get-started/csi_recv` into `v2/firmware/csi_inject_recv` and change its role from *sniffer* to *injection formatter*:

**Remove / disable:** promiscuous mode, real CSI callback, ESP-NOW receive path (we are not sensing real RF).

**Add:**
1. **Wi-Fi station connect** to the host AP (SSID/PW via `menuconfig`/`sdkconfig.defaults`).
2. **UDP listener** on a configured port.
3. On each datagram: parse `node_id + int16 I/Q[64]` (+ `frame_seq`). Ignore if `node_id` ≠ this device's configured id.
4. **Format as the exact `CSI_DATA` CSV** the ghost gateway already understands:
   ```
   CSI_DATA,<seq>,<mac>,<rssi>,<rate>,...,<local_timestamp>,...,"[I0,Q0,I1,Q1,...,I63,Q63]"
   ```
   Metadata fields (rssi/rate/channel/…) can be constant placeholders; only `seq`, `timestamp`, and the I/Q array must be faithful.
5. **Print over UART** at 921600 baud.

Result: from the ghost gateway's perspective, the byte stream on the USB port is indistinguishable from a real capture. **The ghost readout code path barely changes** — only the buffer's dtype (complex) changes, because we now keep phase.

Per-node identity: each of the 3 flashed boards gets a distinct `NODE_ID` (RX1/RX2/RX3) via `sdkconfig` so one firmware image + three configs.

---

## 5. Data model change — full complex I/Q

Today `AmplitudeMatrix` stores `sqrt(I²+Q²)` and discards phase. v2 keeps both.

- **`gateway_v2.py` buffer:** shape `[R × 64 × T]` of **`complex64`** (`H = I + jQ`), or equivalently `[R × 64 × T × 2]` real. Complex is cleaner for the math below.
- Amplitude `A = |H|` and phase `φ = ∠H` become **derived** quantities computed *after* cleaning, not at ingest.
- `CSIPacket` in v2 keeps `csi = i_vals + 1j*q_vals` (complex64, shape `(64,)`).

Everything downstream (`signal_cleaner_v2`, `feature_extractor_v2`) accepts a complex `[R×64×T]` array.

---

## 6. Geometry & node configuration

Physical layout (matters only when real antennas return; for replay it is baked into the dataset, but we still use `NODE_POSITIONS` for the localizer):

- **Linear bistatic array**, units in a line parallel to the wall.
- **Tx** at center `(0, 0)`, forward-facing.
- **RX1** `(-1.0, 0.0)` — 1 m left · **RX2** `(1.0, 0.0)` — 1 m right · **RX3** `(2.0, 0.0)` — 2 m right.
- **Elevation 45°** (front of all panels raised toward the wall) so the cone hits the upper body.
- **Azimuth:** Rx angled slightly inward so reception cones intersect the Tx cone inside the room.
- **RF shielding:** metal plates vertically in the ground between Tx and neighboring Rx to block line-of-sight (kills direct-path leakage).

In software this reduces to `NODE_POSITIONS = {RX1:(-1,0), RX2:(1,0), RX3:(2,0)}` in `config_v2.py`. Elevation/azimuth/shielding are documented for the physical build but do not affect replayed data.

---

## 7. Signal-processing math (complete)

CSI is a complex number per subcarrier: in-phase **I** and quadrature **Q**. The raw received channel is the sum of the static wall reflection and the dynamic human reflection:

```
H_raw(t) = H_static + H_dynamic(t)
```

### 7.1 Static background calibration (room empty, N samples)
Averaged independently per subcarrier k:

```
I_static[k] = (1/N) · Σ_{n=1..N} I_raw[k](t_n)
Q_static[k] = (1/N) · Σ_{n=1..N} Q_raw[k](t_n)
```

Defaults: `N = CALIBRATION_SAMPLES = 200` (≈2 s at 100 Hz; firmware reference uses 1000). Calibration source in replay mode: §11.

### 7.2 Real-time background subtraction (per incoming packet)
```
I_dynamic[k](t) = I_raw[k](t) − I_static[k]
Q_dynamic[k](t) = Q_raw[k](t) − Q_static[k]
```

### 7.3 Cleaned amplitude and phase
```
A_clean[k](t) = sqrt( I_dynamic[k](t)² + Q_dynamic[k](t)² )
φ_clean[k](t) = atan2( Q_dynamic[k](t), I_dynamic[k](t) )
```

> **Bug fixed from the shared snippet:** the Python draft wrote `sqrt(dyn_i*2 + dyn_q*2)` (multiply-by-2). The correct form is **`dyn_i² + dyn_q²`** (square). Same fix applies to the velocity distance below.

### 7.4 Stage-2 temporal filtering (kept from v1, applied after subtraction)
The v1 cleaning still adds value on top of static subtraction — they are **complementary**, not competing:
- **Hampel filter** (window `2·3+1`, threshold `3·1.4826·MAD`) removes impulsive outliers per subcarrier time-series.
- **Butterworth band-pass** 0.1–4.0 Hz (4th order, zero-phase `sosfiltfilt`) keeps the human-motion band and removes residual drift + high-freq noise.
  - Breathing 0.1–0.5 Hz · gestures 0.5–2 Hz · body motion 2–4 Hz.
  - Needs ≥25 samples; below that, Hampel only.

**Pipeline order in v2:** `H_raw` → (7.2) complex static subtract → derive `A_clean`, `φ_clean` → (7.4) Hampel + band-pass on the derived series → features.

---

## 8. Position / velocity estimation (`v2/ghost/localizer.py`)

Per-node dynamic energy after cleaning:

```
E_node = Σ_{k=0..63} A_clean[k]          (sum of clean amplitudes for that receiver)
E_total = Σ_nodes E_node
```

**X (lateral) — energy-weighted centroid over node positions:**
```
x_est = Σ_nodes ( X_node · E_node ) / E_total        (0 if E_total = 0)
```

**Y (depth into room) — inverse-square backscatter heuristic:**
```
y_est = max( 0.5, 6.0 − E_total / 25.0 )
```

**Confidence:**
```
confidence = min( 1.0, E_total / 80.0 )
```

**Velocity** between consecutive estimates (dt seconds apart):
```
dx = x − x_prev ;  dy = y − y_prev
distance_moved = sqrt( dx² + dy² )          # NOT dx*2 + dy*2 — fixed
velocity = distance_moved / dt              (0 if dt ≤ 0)
```

**Anomaly flags:**
```
unrealistic_speed            = velocity > 3.0            # implausible indoors
multipath_reflection_suspect = velocity > 3.0 OR confidence < 0.2
```

> **Calibration warning (must document in the demo):** the constants `25.0`, `80.0`, `6.0`, `3.0` are tuned to one energy scale. Across the four datasets (different amplitude ranges, int16 quantization), these need **per-dataset re-tuning** or the position output is arbitrary. The X centroid is coarse for a 3-node line and collapses toward 0 for symmetric targets. **Position is illustrative, not survey-grade** — and it is only meaningful when the 3 node streams carry genuine spatial diversity (§10.1).

---

## 9. LLM layer — coordinates/velocity schema (`v2/ghost/agentic_core_v2.py`)

Duplicate `agentic_core.py`. Keep the Ollama client + Docker runtime. **Change the output contract** to the coordinates/velocity JSON the injector-era design uses. The localizer output is fed to the model as structured input; the model returns:

```json
{
  "timestamp": 1700000000.123,
  "coordinates": { "x_meters": 0.72, "y_meters": 3.10 },
  "velocity_m_s": 0.45,
  "signal_confidence": 0.61,
  "anomaly_flags": {
    "unrealistic_speed": false,
    "multipath_reflection_suspected": false
  },
  "raw_node_energies": { "RX1": 12.4, "RX2": 33.1, "RX3": 20.7 }
}
```

- System prompt: "You are a Wi-Fi CSI localization/reasoning engine … respond with ONLY this JSON."
- Ollama `format="json"`, low temperature, same multi-stage parse+fallback as v1.
- Memory bank (ChromaDB) stays available for zone/person matching but is optional for this schema; if kept, matches become additional prompt context.

---

## 10. Dataset adapters (`v2/injector/adapters/`)

Every adapter converts its native format into the common injector frame: **64 subcarriers, complex I/Q (from A,φ if needed), 100 Hz, 3 node streams, optional empty-room preamble.**

| Dataset | Native format | Adapter job |
|---------|---------------|-------------|
| **Embedded WiFi Sensing** (ESP32) | ESP32 CSV-like | **Start here** — closest to our format. Verify subcarrier count & rate; near drop-in. |
| **CSI-Bench** (`.mat`) | amplitude + phase, many subcarriers | Rebuild I/Q = A·(cosφ, sinφ); resample subcarriers → 64; resolve sample rate; label fall vs walk. |
| **Intel Respiratory** (Zenodo) | CSI, respiratory | Resample to 64/100 Hz; best for the "still person breathing behind a wall" demo. |
| **Gi-z / CSI-Data** | aggregated, mixed | Per-sample format varies; pick specific person-ID / activity samples for edge-case demos. |

Core assumptions are **fixed at 64 subcarriers / 100 Hz**; all variety is absorbed in the adapters (resample there), so `gateway_v2 / signal_cleaner_v2 / feature_extractor_v2` never change per dataset.

### 10.1 The 3-node problem (design note)
Most public CSI datasets are **single-link** (one Tx–Rx pair). Our system needs 3 receiver streams:
- If a dataset provides ≥3 antennas/links → map directly to RX1/2/3 (real spatial diversity → position is meaningful).
- If single-link → the adapter **synthesizes** node diversity (per-node amplitude/phase offsets) so the pipeline runs, but **position/velocity are then non-metric** and must be labeled as such in the demo. Prefer multi-antenna samples whenever localization is being shown.

---

## 11. Calibration in replay mode

The firmware/ghost expects an **empty-room baseline**, but recordings often contain a person throughout. Strategy:

1. **Preferred — injector calibration preamble.** The injector first streams `CALIBRATION_SAMPLES` frames of a *static/empty* signal (a labeled empty segment if the dataset has one, otherwise a synthetic constant frame equal to the temporal mean of a quiet window). The ghost computes `H_static` (§7.1) during this preamble, then flips to operational mode when the active recording begins. A control marker (special `frame_seq` range, or a UDP control message) signals the transition.
2. **Fallback — temporal-mean estimate.** If no calibration preamble is available, estimate `H_static` as the running/temporal mean of the first N operational frames (this is effectively what the v1 band-pass already does by removing sub-0.1 Hz content).

Config: `CALIBRATION_SAMPLES` (default 200), `CALIBRATION_MODE = preamble | temporal_mean`.

---

## 12. v2 directory structure & file inventory

```
v2/
├── GHOST_V2_PLAN.md               # this document
├── config_v2.py                   # node positions, UDP map, calib, geometry, Ollama
├── injector/
│   ├── injector.py                # Service 1: dataset replay → UDP → ESP32
│   └── adapters/
│       ├── base.py                # common frame contract + resample helpers
│       ├── embedded_wifi.py       # ESP32 CSV  (build first)
│       ├── csi_bench.py           # .mat amplitude/phase
│       ├── intel_resp.py          # Intel respiratory
│       └── giz_csidata.py         # Gi-z aggregated samples
├── firmware/
│   └── csi_inject_recv/           # duplicated csi_recv, converted to UDP→UART loopback
├── ghost/
│   ├── gateway_v2.py              # complex I/Q buffer + serial (USB) source
│   ├── signal_cleaner_v2.py       # complex static subtract + Hampel + bandpass
│   ├── feature_extractor_v2.py    # complex-aware, +phase features
│   ├── localizer.py               # x/y, velocity, energies, anomaly flags
│   ├── agentic_core_v2.py         # Ollama, coordinates/velocity JSON schema
│   └── main_v2.py                 # Service 2 orchestrator (end-to-end loop)
└── docker-compose.v2.yml          # reuse existing Ollama container (or reference root)
```

**Duplication is intentional and allowed** — v2 files are copies of their root counterparts with the v2 changes, so root behavior is untouched.

### 12.1 What each v2 file changes vs its root original
| v2 file | Copied from | Key change |
|---------|-------------|-----------|
| `gateway_v2.py` | `gateway.py` | Buffer stores `complex64` I/Q, not amplitude; `CSIPacket.csi` complex |
| `signal_cleaner_v2.py` | `signal_cleaner.py` | Add §7.1–7.2 complex static subtraction stage before Hampel/bandpass |
| `feature_extractor_v2.py` | `feature_extractor.py` | Accept complex input; derive A & φ; add phase-based features |
| `localizer.py` | *(new)* | §8 position/velocity/energy/anomaly |
| `agentic_core_v2.py` | `agentic_core.py` | §9 coordinates/velocity output schema |
| `config_v2.py` | `config.py` | `NODE_POSITIONS`, UDP node map, calibration + geometry constants |
| `main_v2.py` | *(new — v1 has only pseudocode)* | Wires gateway→cleaner→extractor→localizer→core |
| `injector/*`, `firmware/*` | *(new)* | Service 1 + loopback firmware |

---

## 13. Configuration additions (`config_v2.py`)

| Constant | Default | Purpose |
|----------|---------|---------|
| `NUM_SUBCARRIERS` | 64 | LLTF subcarriers (fixed contract) |
| `NUM_RECEIVERS` | 3 | RX1/RX2/RX3 |
| `SAMPLE_RATE_HZ` | 100 | Injector pace + DSP `fs` |
| `NODE_POSITIONS` | `{RX1:(-1,0), RX2:(1,0), RX3:(2,0)}` | Localizer geometry |
| `NODE_NET_MAP` | `{RX1:(ip,port), ...}` | UDP targets for injector |
| `UDP_PORT` | 5005 | ESP32 injection listen port |
| `CALIBRATION_SAMPLES` | 200 | Static baseline length |
| `CALIBRATION_MODE` | `preamble` | `preamble` \| `temporal_mean` |
| `IQ_QUANT` | `int16` | Wire quantization (not int8) |
| `BANDPASS` | 0.1–4.0 Hz, order 4 | Stage-2 filter |
| `POS_ENERGY_SCALES` | per-dataset | Re-tune §8 magic constants |
| Ollama settings | reuse `OLLAMA_SETTINGS` | Docker runtime |

---

## 14. Implementation steps (phased)

**Phase 0 — Scaffolding.** Create `v2/` tree; copy the four root files into their `_v2` counterparts unchanged; confirm imports resolve in isolation.

**Phase 1 — Complex pipeline (offline, no hardware).**
1. `gateway_v2`: complex I/Q buffer.
2. `signal_cleaner_v2`: add static subtraction (§7.1–7.2), keep Hampel+bandpass (§7.4).
3. `feature_extractor_v2`: complex-aware + phase features.
4. Feed a **saved CSV** (existing `example` capture or a dataset dump) straight through — no ESP32, no UDP — to prove the math end-to-end.

**Phase 2 — Injector + first dataset.**
5. `adapters/embedded_wifi.py` (closest format) + `injector.py`.
6. Emit calibration preamble then active frames; verify frame contract (64ch, int16 I/Q, 100 Hz).
7. Loop injector → (temporarily) a UDP echo/mock ESP32 in software → gateway_v2, to validate the wire format **before flashing firmware**.

**Phase 3 — Firmware loopback.**
8. Build `csi_inject_recv`: Wi-Fi STA + UDP listener + CSI_DATA UART emitter.
9. Flash 3 boards with `NODE_ID` = RX1/RX2/RX3.
10. Injector → Wi-Fi → ESP32 → USB serial → gateway_v2. Confirm ghost sees `CSI_DATA` lines.

**Phase 4 — Localizer + LLM + demo.**
11. `localizer.py` (§8) + `agentic_core_v2.py` (§9).
12. Per-dataset tuning of `POS_ENERGY_SCALES`.
13. Add the other adapters (`csi_bench`, `intel_resp`, `giz_csidata`) for the specific demo scenarios (fall vs walk, still breathing, person-ID).
14. `main_v2.py` orchestrator + dashboard hookup.

---

## 15. Testing & validation

| Level | Test | Expected |
|-------|------|----------|
| Wire format | Injector → software UDP mock → gateway_v2 | Frames parse; 64 complex I/Q per node |
| Firmware | Injector → ESP32 → `screen /dev/cu.* 921600` | Continuous `CSI_DATA,...` lines |
| Calibration | Empty preamble → check `H_static` | Non-zero baseline, stable |
| Static subtraction | Person frames after calib | `A_clean` rises vs empty ≈ 0 |
| Phase | Intel respiratory sample | Periodic φ change in breathing band |
| Localizer | Multi-antenna sample, known position | `x_est` tracks lateral movement |
| Anomaly | Inject impossible jump | `unrealistic_speed = true` |
| LLM | Feed localizer JSON | Valid coordinates/velocity JSON out |
| Reproducibility | Replay same dataset twice | Identical decisions |

---

## 16. Bugs fixed from the shared snippets (do not carry over)

1. **Amplitude/energy:** `math.sqrt(dyn_i*2 + dyn_q*2)` → **`dyn_i**2 + dyn_q**2`** (square, not ×2). Corrupts every amplitude if left.
2. **Velocity distance:** `math.sqrt(dx*2 + dy*2)` → **`dx**2 + dy**2`**.
3. **Dunder methods:** `def _init_` → `__init__`; `if _name_ == "_main_"` → `if __name__ == "__main__"`. Single-underscore versions never run.
4. **Wire quantization:** the draft UDP payload used **int8** — too lossy for phase/breathing. Use **int16** (matches real ESP32 gain-compensated output).
5. **Position constants** (`25.0/80.0/6.0/3.0`) are per-scale magic numbers → must be re-tuned per dataset (`POS_ENERGY_SCALES`).

---

## 17. Open items to confirm as we build

- **§4 firmware written, not yet flashed.** `v2/firmware/csi_inject_recv` (STA + UDP listener + `CSI_DATA` UART emitter) is complete and its output format is verified parser-compatible in software, but it has **not been compiled or flashed** (no ESP-IDF/boards in the dev env). Phase 3 bring-up (build → flash 3 boards with distinct `NODE_ID` → confirm serial `CSI_DATA` → run injector at real IPs) is the next hardware step. The firmware is STA-only, so it works for both Wi-Fi topologies unchanged.
- **Wi-Fi topology:** host SoftAP vs shared router LAN for the 3 ESP32s (affects `NODE_NET_MAP`).
- **Metadata placeholders** in the emitted CSI_DATA line — which fields the ghost gateway actually reads (id, timestamp, I/Q) vs which can be constant.
- **Per-dataset sample rate & subcarrier count** (drives the adapter resamplers) — confirm from each dataset's docs.
- **Multi-antenna availability** per demo scenario (drives whether localization is metric or illustrative — §10.1).
- **Dashboard** target (Python vs web) — still unbuilt in v1; v2 `main_v2.py` will expose the coordinates/velocity stream for it.
- **Raw-CSI phase incoherence breaks complex static subtraction (found during Phase 2, on real `example_csi.csv`).** Raw ESP32 CSI carries a **random phase offset per packet** (carrier/sampling frequency offset + packet timing). Measured on 200 real frames: per-frame amplitude `|H|` ≈ 13.5 but the complex mean `|E[H]|` ≈ 1.8 — a **coherence ratio ≈ 0.11**. Consequence: the §7.1 baseline `H_static = mean(H)` largely **cancels itself**, so §7.2 complex subtraction removes almost nothing (demo: `|H_raw|` 11.2 → `|H_dynamic|` 10.5). The amplitude band-pass still recovers a motion signal (the demo works end to end), but the *complex-domain* subtraction from the firmware snippet is unreliable as-is. **Options to fix (a v2.1 task):** (a) work in the **amplitude domain** for the static baseline (subtract mean amplitude, like v1) — robust but discards phase; (b) **phase-sanitize** first — CSI-ratio / conjugate-multiply between two receivers or two antennas (cancels the common per-packet offset), or per-packet linear-phase detrending, *then* do complex subtraction. The `run_phase2_demo.py` prints a live coherence diagnostic and warns when it is low. This does not block the pipeline but should be resolved before trusting phase-based features on real captures.
- **Breathing needs a long window (found during Phase 1).** `feature_extractor_v2` computes the spectrum as a full-window rFFT, so frequency resolution is `fs/T`. At `fs=100 Hz`, the 2 s motion window (200 samples) has only 0.5 Hz resolution and **cannot resolve breathing** (0.1-0.5 Hz). Resolving breathing needs a **~10 s window (~1000 samples)**. Implication: `main_v2` should pull a **longer window for the breathing/respiration path** (e.g. Intel dataset) than for the fast-motion/localization path. The v1 STFT (`nperseg=128`) had the same limit hidden — v2 switched to full-window rFFT to make resolution explicit and window-driven.

---

*Everything above is additive and isolated under `v2/`. The existing root pipeline continues to run unchanged for live-hardware mode when the antennas are repaired.*
```