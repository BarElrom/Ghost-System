# Gateway / Data Ingestor — Technical Documentation

**Layer 2, Component 1 of the GHOST System**
**File:** `gateway.py`
**Author:** Bar Elrom, Yuval Lerfeld
**Advisor:** Mr. Ilya Zeldner

---

## 1. Purpose

The Gateway is the entry point of the Processing Layer (Layer 2). It receives raw CSI (Channel State Information) data from 3 ESP32 receivers connected via USB serial, parses the firmware CSV output, computes per-subcarrier amplitude, tracks packet loss, and stores the results in a thread-safe rolling buffer.

**Input:** Raw serial CSV lines from ESP32 `csi_recv` firmware
**Output:** Raw Amplitude Matrix — shape `[3 receivers x 64 subcarriers x time]`

This matrix is consumed by downstream components: the Signal Cleaner (DSP preprocessing) and the Feature Extractor.

---

## 2. Architecture

```
USB Serial x3 (921600 baud)
     |
     v
+------------------------------+
|  Gateway (orchestrator)      |
|                              |
|  +------------+ x3           |
|  | SerialReader|---(thread)-->|
|  +------------+              |
|         | raw CSV line       |
|         v                    |
|  +------------+              |
|  | CSIParser   | (stateless) |
|  +------------+              |
|         | CSIPacket          |
|         v                    |
|  +----------------+          |
|  | AmplitudeMatrix | (shared)|
|  | [3x64xT] ring  |         |
|  +----------------+          |
|         |                    |
|         v                    |
|    downstream consumers      |
+------------------------------+
```

**Threading model:**
- Main thread spawns 3 daemon threads (one per receiver)
- Each thread reads its serial port, parses lines, and writes to its receiver slice in the shared matrix
- The `AmplitudeMatrix` is protected by a `threading.Lock`
- Contention is minimal since each thread writes to a different receiver index

---

## 3. ESP32 Firmware Output Format

The `csi_recv` firmware (from `esp-csi/examples/get-started/csi_recv`) outputs CSV lines at 921600 baud over USB serial.

### 3.1 CSV Header

Standard ESP32 (non-C5/C6):

```
type,id,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,not_sounding,aggregation,stbc,fec_coding,sgi,noise_floor,ampdu_cnt,channel,secondary_channel,local_timestamp,ant,sig_len,rx_format,len,first_word,data
```

### 3.2 CSV Data Line Example

```
CSI_DATA,0,1a:00:00:00:00:00,-45,8,1,7,1,0,0,0,0,0,0,-95,1,6,0,1234567890,0,52,1,128,0,"[100,102,98,-50,...]"
```

### 3.3 Column Mapping

| Index | Field | Description |
|-------|-------|-------------|
| 0 | type | Always `CSI_DATA` |
| 1 | id | Sequence number (rx_id from ESP-NOW payload) |
| 2 | mac | Source MAC address |
| 3 | rssi | Received Signal Strength (dBm) |
| 4 | rate | Data rate index |
| 5 | sig_mode | Signal mode (0=non-HT, 1=HT, 2=VHT) |
| 6 | mcs | Modulation and Coding Scheme |
| 7 | bandwidth | 0=20MHz, 1=40MHz |
| 8-13 | smoothing...sgi | Various 802.11 flags |
| 14 | noise_floor | RF noise floor (dBm) |
| 15 | ampdu_cnt | AMPDU count |
| 16 | channel | Wi-Fi channel number |
| 17 | secondary_channel | Secondary channel offset |
| 18 | local_timestamp | Firmware timestamp (microseconds) |
| 19 | ant | Antenna index |
| 20 | sig_len | Signal length (bytes) |
| 21 | rx_format | Receive format (sig_mode repeated) |
| 22 | len | Number of CSI data values |
| 23 | first_word | First word invalid flag |
| 24 | data | CSI array: `"[I0,Q0,I1,Q1,...,I63,Q63,...]"` |

### 3.4 CSI Data Array

- Enclosed in quotes and brackets: `"[val1,val2,...,valN]"`
- Values are signed integers (int8, gain-compensated to int16 by firmware)
- Consecutive pairs represent I (in-phase) and Q (quadrature) for each subcarrier
- Total values depends on packet type: 128 (LLTF only) or 384 (LLTF + HT-LTF)
- **The gateway uses only the first 128 values = 64 I/Q pairs = LLTF subcarriers**

### 3.5 Amplitude Computation

For each subcarrier k:

```
amplitude[k] = sqrt(I[k]^2 + Q[k]^2)
```

Where I[k] and Q[k] are extracted from positions 2k and 2k+1 in the raw integer array.

---

## 4. Classes Reference

### 4.1 CSIPacket

Dataclass holding one parsed CSI measurement.

| Field | Type | Description |
|-------|------|-------------|
| `receiver_index` | `int` | Receiver ID (0, 1, or 2) |
| `seq_id` | `int` | Firmware sequence number |
| `mac` | `str` | Source MAC address |
| `rssi` | `int` | Signal strength (dBm) |
| `channel` | `int` | Wi-Fi channel |
| `timestamp` | `int` | Firmware timestamp (microseconds) |
| `amplitude` | `np.ndarray` | Shape (64,), float32 — computed amplitude per subcarrier |
| `raw_iq` | `np.ndarray` | Shape (64, 2), int16 — raw I and Q values |

### 4.2 CSIParser

Stateless parser. Converts one CSV line into a `CSIPacket`.

**Method:**

```python
parser = CSIParser()
packet = parser.parse(line: str, receiver_index: int) -> CSIPacket | None
```

- Returns `None` for non-CSI lines, malformed data, or arrays with fewer than 128 values
- Uses fast `str.split()` (comma or whitespace) instead of regex for CSI array parsing — critical for keeping up with 100Hz packet rate
- Extracts metadata from known CSV column positions

### 4.3 AmplitudeMatrix

Thread-safe ring buffer. Shape: `[3 x 64 x max_time]`.

**Constructor:**

```python
matrix = AmplitudeMatrix(max_time=1000)
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `append(receiver_index, amplitude)` | `None` | Write one (64,) sample, advance ring head |
| `get_latest(n)` | `np.ndarray (3, 64, n)` | Last n samples across all receivers (zero-filled if insufficient data) |
| `get_receiver_count(receiver_index)` | `int` | Total packets ever stored for this receiver |

**Ring buffer behavior:**
- Pre-allocated numpy array, no dynamic resizing
- Old data is silently overwritten when the buffer wraps
- Per-receiver write heads track position independently
- All reads and writes are protected by a threading lock

### 4.4 SerialReader

Reads one serial port in a dedicated thread.

**Constructor:**

```python
reader = SerialReader(
    port="/dev/cu.usbserial-0001",
    receiver_index=0,
    matrix=amplitude_matrix,
    baud_rate=921600,
)
```

**Methods:**

| Method | Description |
|--------|-------------|
| `run()` | Main loop: open serial, read lines, parse, store amplitude, check sequence |
| `stop()` | Set stop flag (loop exits on next iteration) |

**Sequence continuity tracking:**

Each reader tracks the previous sequence number. When a gap is detected:

```
WARNING  Rx0: lost 3 packets (seq 142 -> 146)
```

**Statistics dict (`reader.stats`):**

| Key | Description |
|-----|-------------|
| `received` | Successfully parsed CSI packets |
| `lost` | Estimated lost packets (from sequence gaps) |
| `errors` | Serial/parse errors |

### 4.5 Gateway

Top-level orchestrator.

**Constructor:**

```python
gw = Gateway(
    ports=None,          # None = auto-detect, or list of paths
    baud_rate=921600,
    buffer_size=1000,
)
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `detect_ports()` | `list[str]` | Scan macOS `/dev/cu.*` for ESP32 serial ports |
| `start()` | `None` | Detect ports, create readers, start threads |
| `stop()` | `None` | Stop all readers, join threads |
| `get_matrix()` | `AmplitudeMatrix` | Reference to shared matrix for downstream use |
| `get_stats()` | `dict` | Per-receiver packet/loss/error counts |
| `is_alive()` | `bool` | True if all reader threads are running |

---

## 5. Serial Port Auto-Detection

On macOS, ESP32 USB-serial adapters appear under `/dev/cu.*`. The gateway scans these glob patterns:

| Pattern | Chip |
|---------|------|
| `/dev/cu.usbserial*` | CP2102 / FTDI |
| `/dev/cu.SLAB*` | Silicon Labs CP210x |
| `/dev/cu.wchusbserial*` | CH340 |
| `/dev/cu.usbmodem*` | Native USB CDC |

Bluetooth and debug ports are filtered out. Ports are sorted alphabetically for deterministic receiver assignment (Rx0, Rx1, Rx2).

To override auto-detection, pass explicit port paths:

```python
gw = Gateway(ports=["/dev/cu.usbserial-0001", "/dev/cu.usbserial-0002", "/dev/cu.usbserial-0003"])
```

---

## 6. Usage

### 6.1 As a Module (imported by downstream pipeline)

```python
from gateway import Gateway

# Start the gateway
gw = Gateway()
gw.start()

# Access the shared amplitude matrix
matrix = gw.get_matrix()

# Get the last 200 samples: shape (3, 64, 200)
data = matrix.get_latest(200)

# Check per-receiver counts
for rx in range(3):
    print(f"Rx{rx}: {matrix.get_receiver_count(rx)} packets")

# Check stats
print(gw.get_stats())

# Stop when done
gw.stop()
```

### 6.2 Standalone (testing/debugging)

```bash
cd /Users/barelrom/PycharmProjects/ML/TheGhostSystem
python gateway.py
```

Prints live stats every 2 seconds and amplitude samples every 10 seconds. Ctrl+C to stop.

Example output:

```
2026-05-15 14:30:00  ghost.gateway  INFO  Detected serial ports: ['/dev/cu.usbserial-0001', '/dev/cu.usbserial-0002', '/dev/cu.usbserial-0003']
2026-05-15 14:30:00  ghost.gateway  INFO  Rx0: opening /dev/cu.usbserial-0001 @ 921600 baud
2026-05-15 14:30:00  ghost.gateway  INFO  Rx1: opening /dev/cu.usbserial-0002 @ 921600 baud
2026-05-15 14:30:00  ghost.gateway  INFO  Rx2: opening /dev/cu.usbserial-0003 @ 921600 baud
2026-05-15 14:30:00  ghost.gateway  INFO  Gateway started with 3 receiver(s)

Gateway running. Press Ctrl+C to stop.

--- stats (t=2s) ---
  Rx0 (/dev/cu.usbserial-0001): 187 pkts, 2 lost, 0 errors
  Rx1 (/dev/cu.usbserial-0002): 193 pkts, 0 lost, 0 errors
  Rx2 (/dev/cu.usbserial-0003): 190 pkts, 1 lost, 0 errors
```

---

## 7. Configuration Constants

| Constant | Default | Description |
|----------|---------|-------------|
| `DEFAULT_BAUD_RATE` | 921600 | Serial baud rate (matches `csi_recv` sdkconfig) |
| `DEFAULT_BUFFER_SIZE` | 1000 | Ring buffer depth (~10 sec at 100 Hz) |
| `NUM_SUBCARRIERS` | 64 | LLTF subcarrier count |
| `NUM_RECEIVERS` | 3 | Number of ESP32 receivers |
| `SERIAL_TIMEOUT` | 1.0 | Serial readline timeout (seconds) |

---

## 8. Error Handling

| Scenario | Behavior |
|----------|----------|
| No serial ports found | Logs error, gateway does not start |
| Fewer than 3 ports found | Logs warning, starts with available ports |
| Serial port fails to open | Logs error, that reader thread exits |
| Serial disconnects mid-run | Catches `SerialException`, reader exits, `is_alive()` returns False |
| Malformed CSV line | Parser returns None, line silently skipped |
| CSI array too short | Parser returns None, logged at DEBUG level |
| Non-CSI_DATA line (firmware logs, headers) | Parser returns None, silently skipped |
| Ring buffer full | Oldest data overwritten (expected behavior) |

---

## 9. Data Flow Summary

```
ESP32 Tx (csi_send)
    | ESP-NOW packets @ 100Hz, channel 6
    v
ESP32 Rx x3 (csi_recv)
    | CSI_DATA CSV lines via UART
    v
USB Serial (921600 baud)
    | /dev/cu.usbserial-*
    v
+-------------------+
| SerialReader x3   |  (daemon threads)
| readline + parse  |
+-------------------+
    | CSIPacket
    v
+-------------------+
| AmplitudeMatrix   |  (thread-safe ring buffer)
| [3 x 64 x 1000]  |
+-------------------+
    |
    v
Downstream: Signal Cleaner -> Feature Extractor -> Cognitive Layer
```

---

## 10. Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | any | Array operations, amplitude computation |
| `pyserial` | >= 3.5 | Serial port communication |

Install:

```bash
pip install pyserial
```

---

## 11. File Structure After Implementation

```
TheGhostSystem/
  gateway.py          <-- this component
  docs/
    gateway.md        <-- this document
  esp-csi/            <-- ESP32 firmware (csi_recv, csi_send)
  csiTest.py          <-- earlier prototype (3D visualization)
  example_csi.csv     <-- sample data
```

---

## 12. Relation to System Architecture

From the GHOST system architecture (Phase A document):

```
Layer 1: Physical Sensing Layer
    ESP32 Tx + 3x ESP32 Rx
        |
        v
Layer 2: Processing Layer
    [Gateway/Ingestor]  <-- THIS COMPONENT
        |
        v
    Signal Cleaner (DSP preprocessing)
        |
        v
    Feature Extractor (PCA + STFT)
        |
        v
Layer 3: Cognitive Layer
    Memory Bank (ChromaDB) + Agentic Core (Llama 3)
        |
        v
    Dashboard (visualization)
```

The Gateway produces the **Raw Amplitude Matrix** `[3 x 64 x T]` that the Signal Cleaner consumes for outlier removal, static component suppression, and bandpass filtering.