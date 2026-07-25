# csi_inject_recv — ESP32 CSI Injection Loopback Firmware

GHOST v2, Option B (firmware loopback). This firmware turns each ESP32 receiver
into a **UDP → UART loopback**: it does not sense real RF. It joins Wi-Fi as a
station, receives GHOST `G2` injection frames over UDP, and re-emits them as
standard `CSI_DATA` serial lines that the ghost gateway already understands.

```
injector (host) --UDP/Wi-Fi--> ESP32 (this firmware) --USB serial--> ghost gateway_v2
```

> **Status: written to spec, not yet compiled/flashed.** It requires ESP-IDF and
> physical boards, which the development environment does not have. Build and
> flash it on your hardware (below), then run the Phase 3 hardware test.

---

## What it does

1. Connects to Wi-Fi (SSID/password from `menuconfig`).
2. Binds a UDP socket on `CONFIG_GHOST_UDP_PORT` (default 5005).
3. For each datagram: validates the `G2` header, checks `node_id == CONFIG_GHOST_NODE_ID`,
   decodes the interleaved big-endian int16 I/Q, and prints a `CSI_DATA,...,"[I0,Q0,...]"`
   line over the UART at 921600 baud.

The emitted line is byte-compatible with `v2/transport/mock_esp32.py`
`frame_to_csi_line()` and the root `CSIParser` (only `id`, `local_timestamp`,
and the I/Q array are faithful; other metadata are constant fillers).

The wire frame is defined in `v2/transport/frame.py` (verified offsets):

| Offset | Field | |
|---|---|---|
| 0 | magic `"G2"` | 2 bytes |
| 2 | version | 1 |
| 3 | node_id | 1 (1=RX1, 2=RX2, 3=RX3) |
| 4 | flags | 1 (bit0 = calibration, informational) |
| 5 | reserved | 1 |
| 6 | frame_seq | uint32 BE |
| 10 | num_sub | uint16 BE |
| 12 | I/Q | num_sub×2 int16 BE, `I0,Q0,I1,Q1,...` |

---

## Wi-Fi topology (either works — the ESP32 is always a station)

- **Shared router:** point SSID/password at your 2.4 GHz router; the host running
  the injector must be on the same network.
- **Host SoftAP:** create a hotspot on the machine running the injector and point
  SSID/password at it. (macOS Internet Sharing, or a small `hostapd`/AP setup.)

Either way the ESP32 firmware is identical — it just joins the given SSID.

---

## Build & flash (per board)

Requires ESP-IDF v5.x installed and sourced (`. $HOME/esp/esp-idf/export.sh`).

```bash
cd v2/firmware/csi_inject_recv
idf.py set-target esp32

# Configure SSID / password / port / NODE_ID:
idf.py menuconfig      # → "GHOST v2 Injection Firmware"

# Flash board #1 as RX1 (set NODE_ID=1 in menuconfig), then:
idf.py -p /dev/cu.usbserial-XXXX flash monitor
```

Repeat for the other two boards, setting **`CONFIG_GHOST_NODE_ID = 2` and `= 3`**
in `menuconfig` before flashing each. (One image, three node ids.)

On boot each board logs its assigned IP:

```
csi_inject_recv: got IP 192.168.1.51 — node RX1 ready
csi_inject_recv: UDP loopback listening on :5005 for node RX2
```

---

## Point the injector at the boards

Record each board's IP and set them for the injector (host side):

```bash
export GHOST_V2_RX1_IP=192.168.1.51
export GHOST_V2_RX2_IP=192.168.1.52
export GHOST_V2_RX3_IP=192.168.1.53
export GHOST_V2_UDP_PORT=5005
```

Then run the injector against a dataset (real hardware in the loop):

```bash
python -m v2.injector.injector --dataset embedded_wifi --path example_csi.csv --rate 100
```

Or override all three to one host/port for a single-board bring-up:

```bash
python -m v2.injector.injector --path example_csi.csv --host 192.168.1.51 --port 5005
```

---

## Phase 3 hardware bring-up checklist

1. Flash one board, note its IP from the monitor log.
2. `screen /dev/cu.usbserial-XXXX 921600` on the host — confirm the board boots
   and prints the "UDP loopback listening" line.
3. Send a single frame from the host and confirm a `CSI_DATA,...` line appears:
   ```bash
   python -c "import numpy as np; from v2.transport.udp_sender import UDPSender; \
   s=UDPSender(net_map={'RX1':('192.168.1.51',5005)}); \
   s.send('RX1',0,(np.arange(64)+1j*np.arange(64)).astype('complex64')); s.close()"
   ```
4. Point the ghost `GatewayV2.from_serial([...])` at the board's USB port and
   confirm the complex matrix fills.
5. Repeat for all three boards, then run the full injector → 3×ESP32 → ghost chain.

---

## Notes / limits

- **Throughput:** one line is ~800 bytes (128 values). At 100 Hz that's ~80 KB/s,
  near the 921600-baud UART's usable ceiling. If lines are truncated, lower the
  injector `--rate` or raise the baud on both firmware and gateway.
- **num_sub:** capped at 64 (128 I/Q values) in the emitter buffer.
- **Timestamp:** uses `esp_timer_get_time()` (real microseconds), unlike the mock
  which derives it from the sequence number. The gateway does not depend on it.
