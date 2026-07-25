"""
GHOST System — Gateway / Data Ingestor
Layer 2, Component 1: Receive, synchronize, and convert raw CSI packets
from ESP32 receivers into a structured amplitude matrix.

Output: Raw Amplitude Matrix, shape [3 receivers x 64 subcarriers x time]

Usage:
    # As a module:
    from gateway import Gateway
    gw = Gateway()
    gw.start()
    matrix = gw.get_matrix()
    latest = matrix.get_latest(100)  # shape (3, 64, 100)
    gw.stop()

    # Standalone:
    python gateway.py
"""

import glob
import time
import logging
import threading
from dataclasses import dataclass

import numpy as np
import serial
import serial.tools.list_ports

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
DEFAULT_BAUD_RATE = 921600
DEFAULT_BUFFER_SIZE = 1000  # ~10 sec at 100 Hz
NUM_SUBCARRIERS = 64        # LLTF subcarriers
NUM_RECEIVERS = 3
SERIAL_TIMEOUT = 1.0        # seconds, for readline
RECONNECT_DELAY = 2.0       # seconds to wait before reconnecting
MAX_RECONNECT_ATTEMPTS = 10 # max consecutive reconnect attempts before giving up

logger = logging.getLogger("ghost.gateway")


# ---------------------------------------------------------------------------
# CSIPacket — parsed representation of one CSI_DATA line
# ---------------------------------------------------------------------------
@dataclass
class CSIPacket:
    """One parsed CSI measurement from an ESP32 receiver."""
    receiver_index: int
    seq_id: int
    mac: str
    rssi: int
    channel: int
    timestamp: int           # local_timestamp from firmware (microseconds)
    amplitude: np.ndarray    # shape (64,), float32
    raw_iq: np.ndarray       # shape (64, 2), int — raw I and Q per subcarrier


# ---------------------------------------------------------------------------
# CSIParser — stateless parser for ESP32 CSV lines
# ---------------------------------------------------------------------------

# Column indices for standard ESP32 (non-C5/C6) csi_recv firmware output:
#   type,id,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,not_sounding,
#   aggregation,stbc,fec_coding,sgi,noise_floor,ampdu_cnt,channel,
#   secondary_channel,local_timestamp,ant,sig_len,rx_format,len,first_word,data
COL_TYPE = 0
COL_ID = 1
COL_MAC = 2
COL_RSSI = 3
COL_CHANNEL = 16
COL_TIMESTAMP = 18
COL_LEN = 22
# data is the last field (index 24), but may contain commas inside the
# quoted bracket array, so we extract it from the tail of the line.


class CSIParser:
    """Stateless parser: raw CSV line -> CSIPacket or None."""

    def parse(self, line: str, receiver_index: int) -> CSIPacket | None:
        """Parse one serial line into a CSIPacket.

        Returns None for non-CSI lines, malformed data, or lines with
        too few I/Q values.
        """
        line = line.strip()
        if not line.startswith("CSI_DATA"):
            return None

        # --- extract the CSI data array from the tail of the line ---
        # The data field looks like: ",[val1,val2,...]" or ,"[val1,val2,...]"
        # Find the first '[' and last ']' to isolate the array portion.
        bracket_start = line.find("[")
        bracket_end = line.rfind("]")
        if bracket_start == -1 or bracket_end == -1 or bracket_end <= bracket_start:
            logger.debug("Rx%d: no CSI array brackets found", receiver_index)
            return None

        csi_str = line[bracket_start:bracket_end + 1]

        # The metadata CSV is everything before the CSI data field.
        # Split the portion before the bracket on commas.
        meta_part = line[:bracket_start]
        # Remove trailing comma and quote: ...,"[ → strip trailing ,"
        meta_part = meta_part.rstrip(',"')
        fields = meta_part.split(",")

        if len(fields) < 19:
            logger.debug("Rx%d: too few CSV fields (%d)", receiver_index, len(fields))
            return None

        try:
            seq_id = int(fields[COL_ID])
            mac = fields[COL_MAC]
            rssi = int(fields[COL_RSSI])
            channel = int(fields[COL_CHANNEL])
            timestamp = int(fields[COL_TIMESTAMP])
        except (ValueError, IndexError) as e:
            logger.debug("Rx%d: metadata parse error: %s", receiver_index, e)
            return None

        # --- parse I/Q integers ---
        # Strip brackets, then split on comma or whitespace.
        inner = csi_str[1:-1]  # remove '[' and ']'
        if "," in inner:
            parts = inner.split(",")
        else:
            parts = inner.split()

        if len(parts) < 2 * NUM_SUBCARRIERS:
            logger.debug(
                "Rx%d: CSI array too short (%d values, need %d)",
                receiver_index, len(parts), 2 * NUM_SUBCARRIERS,
            )
            return None

        # Take first 128 integers = 64 I/Q pairs (LLTF subcarriers only).
        vals = np.array(parts[: 2 * NUM_SUBCARRIERS], dtype=np.float32)
        i_vals = vals[0::2]
        q_vals = vals[1::2]

        amplitude = np.sqrt(i_vals ** 2 + q_vals ** 2)
        raw_iq = np.stack([i_vals, q_vals], axis=1).astype(np.int16)  # (64, 2)

        return CSIPacket(
            receiver_index=receiver_index,
            seq_id=seq_id,
            mac=mac,
            rssi=rssi,
            channel=channel,
            timestamp=timestamp,
            amplitude=amplitude,
            raw_iq=raw_iq,
        )


# ---------------------------------------------------------------------------
# AmplitudeMatrix — thread-safe rolling buffer [3 x 64 x T]
# ---------------------------------------------------------------------------
class AmplitudeMatrix:
    """Thread-safe ring buffer storing amplitude data from all receivers.

    Shape: [NUM_RECEIVERS x NUM_SUBCARRIERS x max_time]
    """

    def __init__(self, max_time: int = DEFAULT_BUFFER_SIZE):
        self.max_time = max_time
        self.buffer = np.zeros(
            (NUM_RECEIVERS, NUM_SUBCARRIERS, max_time), dtype=np.float32
        )
        # Per-receiver write head (absolute count, mod max_time for index).
        self._count = np.zeros(NUM_RECEIVERS, dtype=np.int64)
        self._lock = threading.Lock()

    def append(self, receiver_index: int, amplitude: np.ndarray) -> None:
        """Write one (64,) amplitude sample for a given receiver."""
        with self._lock:
            idx = int(self._count[receiver_index] % self.max_time)
            self.buffer[receiver_index, :, idx] = amplitude
            self._count[receiver_index] += 1

    def get_latest(self, n: int) -> np.ndarray:
        """Return the last n samples across all receivers.

        Returns shape (3, 64, n). If a receiver has fewer than n samples,
        the missing positions are zero-filled.
        """
        with self._lock:
            n = min(n, self.max_time)
            result = np.zeros(
                (NUM_RECEIVERS, NUM_SUBCARRIERS, n), dtype=np.float32
            )
            for rx in range(NUM_RECEIVERS):
                total = int(self._count[rx])
                available = min(total, n, self.max_time)
                if available == 0:
                    continue
                head = int(self._count[rx] % self.max_time)
                if head >= available:
                    result[rx, :, n - available:] = (
                        self.buffer[rx, :, head - available : head]
                    )
                else:
                    # Wraps around the ring boundary.
                    tail_len = available - head
                    result[rx, :, n - available : n - available + tail_len] = (
                        self.buffer[rx, :, self.max_time - tail_len :]
                    )
                    result[rx, :, n - head:] = self.buffer[rx, :, :head]
            return result

    def get_receiver_count(self, receiver_index: int) -> int:
        """Total number of samples ever stored for this receiver."""
        with self._lock:
            return int(self._count[receiver_index])


# ---------------------------------------------------------------------------
# SerialReader — one per receiver, runs in its own thread
# ---------------------------------------------------------------------------
class SerialReader:
    """Reads CSI lines from one serial port and feeds them into the matrix."""

    def __init__(
        self,
        port: str,
        receiver_index: int,
        matrix: AmplitudeMatrix,
        baud_rate: int = DEFAULT_BAUD_RATE,
    ):
        self.port = port
        self.receiver_index = receiver_index
        self.baud_rate = baud_rate
        self._matrix = matrix
        self._parser = CSIParser()
        self._running = False
        self._prev_seq: int | None = None
        self.stats = {"received": 0, "lost": 0, "errors": 0}

    def _open_serial(self) -> serial.Serial | None:
        """Try to open the serial port, return None on failure."""
        try:
            ser = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=SERIAL_TIMEOUT,
            )
            return ser
        except (serial.SerialException, OSError) as e:
            logger.error("Rx%d: failed to open %s — %s", self.receiver_index, self.port, e)
            return None

    def run(self) -> None:
        """Main loop — open serial port and process lines until stopped.

        On serial errors, automatically reconnects up to
        MAX_RECONNECT_ATTEMPTS consecutive times before giving up.
        """
        self._running = True
        reconnect_attempts = 0

        while self._running:
            logger.info(
                "Rx%d: opening %s @ %d baud",
                self.receiver_index, self.port, self.baud_rate,
            )

            ser = self._open_serial()
            if ser is None:
                reconnect_attempts += 1
                if reconnect_attempts >= MAX_RECONNECT_ATTEMPTS:
                    logger.error(
                        "Rx%d: giving up after %d failed reconnect attempts",
                        self.receiver_index, reconnect_attempts,
                    )
                    break
                logger.warning(
                    "Rx%d: retrying in %.1fs (%d/%d)...",
                    self.receiver_index, RECONNECT_DELAY,
                    reconnect_attempts, MAX_RECONNECT_ATTEMPTS,
                )
                time.sleep(RECONNECT_DELAY)
                continue

            # Successfully connected — reset counter.
            reconnect_attempts = 0

            try:
                while self._running:
                    try:
                        raw = ser.readline()
                    except (serial.SerialException, OSError) as e:
                        logger.error("Rx%d: serial read error — %s", self.receiver_index, e)
                        self.stats["errors"] += 1
                        break  # break inner loop to reconnect

                    if not raw:
                        continue  # timeout, no data

                    try:
                        line = raw.decode("utf-8", errors="replace")
                    except UnicodeDecodeError:
                        self.stats["errors"] += 1
                        continue

                    packet = self._parser.parse(line, self.receiver_index)
                    if packet is None:
                        continue

                    # --- sequence continuity check ---
                    if self._prev_seq is not None:
                        expected = self._prev_seq + 1
                        if packet.seq_id != expected:
                            gap = packet.seq_id - self._prev_seq - 1
                            if gap > 0:
                                self.stats["lost"] += gap
                    self._prev_seq = packet.seq_id

                    # --- store amplitude ---
                    self._matrix.append(self.receiver_index, packet.amplitude)
                    self.stats["received"] += 1

            except Exception as e:
                logger.error("Rx%d: unexpected error — %s: %s",
                             self.receiver_index, type(e).__name__, e)
                self.stats["errors"] += 1

            finally:
                ser.close()
                logger.info("Rx%d: serial port closed", self.receiver_index)

            # If we're still running, the inner loop broke due to error — reconnect.
            if self._running:
                reconnect_attempts += 1
                if reconnect_attempts >= MAX_RECONNECT_ATTEMPTS:
                    logger.error(
                        "Rx%d: giving up after %d consecutive errors",
                        self.receiver_index, reconnect_attempts,
                    )
                    break
                logger.warning(
                    "Rx%d: reconnecting in %.1fs (%d/%d)...",
                    self.receiver_index, RECONNECT_DELAY,
                    reconnect_attempts, MAX_RECONNECT_ATTEMPTS,
                )
                time.sleep(RECONNECT_DELAY)

        self._running = False

    def stop(self) -> None:
        """Signal the reader loop to exit."""
        self._running = False


# ---------------------------------------------------------------------------
# Gateway — top-level orchestrator
# ---------------------------------------------------------------------------
class Gateway:
    """Manages serial readers and the shared amplitude matrix.

    Args:
        ports: Explicit list of serial port paths. If None, auto-detect.
        baud_rate: Serial baud rate (default 921600).
        buffer_size: Ring buffer depth in packets (default 1000).
    """

    # Glob patterns for ESP32 USB-serial adapters on macOS.
    PORT_PATTERNS = [
        "/dev/cu.usbserial*",
        "/dev/cu.SLAB*",
        "/dev/cu.wchusbserial*",
        "/dev/cu.usbmodem*",
    ]

    def __init__(
        self,
        ports: list[str] | None = None,
        baud_rate: int = DEFAULT_BAUD_RATE,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
    ):
        self._baud_rate = baud_rate
        self._matrix = AmplitudeMatrix(max_time=buffer_size)
        self._readers: list[SerialReader] = []
        self._threads: list[threading.Thread] = []
        self._ports = ports

    # ---- port detection ----

    def detect_ports(self) -> list[str]:
        """Scan for ESP32 USB-serial ports on macOS.

        Returns a sorted list of port paths. Logs a warning if the count
        is not exactly NUM_RECEIVERS.
        """
        found: set[str] = set()
        for pattern in self.PORT_PATTERNS:
            found.update(glob.glob(pattern))

        # Filter out Bluetooth and debug ports.
        filtered = sorted(
            p for p in found
            if "Bluetooth" not in p and "debug" not in p.lower()
        )

        if len(filtered) == 0:
            logger.error("No ESP32 serial ports detected")
        elif len(filtered) < NUM_RECEIVERS:
            logger.warning(
                "Found %d serial ports (expected %d): %s",
                len(filtered), NUM_RECEIVERS, filtered,
            )
        else:
            logger.info("Detected serial ports: %s", filtered)

        return filtered

    # ---- lifecycle ----

    def start(self) -> None:
        """Detect ports, create readers, start threads."""
        ports = self._ports if self._ports else self.detect_ports()
        if not ports:
            logger.error("No serial ports available — gateway not started")
            return

        for idx, port in enumerate(ports[:NUM_RECEIVERS]):
            reader = SerialReader(
                port=port,
                receiver_index=idx,
                matrix=self._matrix,
                baud_rate=self._baud_rate,
            )
            thread = threading.Thread(
                target=reader.run,
                name=f"Rx{idx}-{port}",
                daemon=True,
            )
            self._readers.append(reader)
            self._threads.append(thread)

        for t in self._threads:
            t.start()

        logger.info("Gateway started with %d receiver(s)", len(self._readers))

    def stop(self) -> None:
        """Stop all readers and wait for threads to finish."""
        for reader in self._readers:
            reader.stop()
        for thread in self._threads:
            thread.join(timeout=3.0)
        logger.info("Gateway stopped")

    # ---- accessors ----

    def get_matrix(self) -> AmplitudeMatrix:
        """Return the shared amplitude matrix for downstream consumers."""
        return self._matrix

    def get_stats(self) -> dict:
        """Per-receiver packet and loss statistics."""
        return {
            f"Rx{r.receiver_index} ({r.port})": dict(r.stats)
            for r in self._readers
        }

    def is_alive(self) -> bool:
        """True if all reader threads are still running."""
        return all(t.is_alive() for t in self._threads)


# ---------------------------------------------------------------------------
# Standalone entry point — run for live testing / debugging
# ---------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    gw = Gateway()
    gw.start()

    if not gw._readers:
        print("No receivers connected. Exiting.")
        return

    print("\nGateway running. Press Ctrl+C to stop.\n")

    try:
        cycle = 0
        while True:
            time.sleep(2)
            cycle += 1
            stats = gw.get_stats()

            matrix = gw.get_matrix()

            print(f"--- stats (t={cycle * 2}s) ---")
            for label, s in stats.items():
                total = s['received'] + s['lost']
                loss_pct = (s['lost'] / total * 100) if total > 0 else 0
                print(f"  {label}: {s['received']} pkts, {s['lost']} lost ({loss_pct:.1f}%), {s['errors']} errors")

            # Show latest amplitude snapshot from each active receiver.
            latest = matrix.get_latest(1)  # shape (3, 64, 1)
            for rx in range(len(gw._readers)):
                count = matrix.get_receiver_count(rx)
                if count > 0:
                    amp = latest[rx, :, 0]
                    print(
                        f"  Rx{rx} amplitude [first 8 subcarriers]: "
                        f"{amp[:8].round(1)}"
                    )
            print(f"  Matrix shape: {matrix.buffer.shape}  "
                  f"(filled: {[matrix.get_receiver_count(i) for i in range(NUM_RECEIVERS)]})")

            if not gw.is_alive():
                print("WARNING: one or more reader threads have died")

    except KeyboardInterrupt:
        print("\nStopping...")
        gw.stop()
        print("Done.")


if __name__ == "__main__":
    main()