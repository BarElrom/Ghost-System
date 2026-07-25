"""
GHOST v2 — Gateway: turn incoming CSI lines into a complex I/Q buffer.

Two differences from the root gateway.py (which is left untouched):

  * It keeps the FULL COMPLEX signal (I + jQ), not just amplitude, so phase
    survives all the way through the pipeline.
  * The data source is pluggable: the same gateway reads from a real ESP32 over
    USB serial (SerialSource) or from injected UDP frames (see sources.py).

Pipeline position:  CSI line  ->  [gateway_v2]  ->  complex matrix [3 x 64 x T]
"""

import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

try:
    import serial  # pyserial — only needed by SerialSource (real hardware)
except ImportError:  # pragma: no cover
    serial = None

from v2.config_v2 import NUM_RECEIVERS, NUM_SUBCARRIERS

DEFAULT_BAUD_RATE = 921600
DEFAULT_BUFFER_SIZE = 1000        # samples kept per receiver (~10 s at 100 Hz)
SERIAL_READ_TIMEOUT = 1.0         # seconds

logger = logging.getLogger("ghost.v2.gateway")


# ---------------------------------------------------------------------------
# One parsed CSI reading
# ---------------------------------------------------------------------------
@dataclass
class CSIPacket:
    """A single CSI reading from one receiver, carrying complex subcarriers."""

    receiver_index: int
    seq_id: int
    mac: str
    rssi: int
    channel: int
    timestamp: int                 # firmware microsecond timestamp
    csi: np.ndarray                # complex64, shape (64,) — I + jQ per subcarrier

    @property
    def amplitude(self) -> np.ndarray:
        """Per-subcarrier magnitude |csi| = sqrt(I^2 + Q^2)."""
        return np.abs(self.csi).astype(np.float32)

    @property
    def phase(self) -> np.ndarray:
        """Per-subcarrier phase angle(csi) = atan2(Q, I), in radians."""
        return np.angle(self.csi).astype(np.float32)


# ---------------------------------------------------------------------------
# Parser: one CSI_DATA text line -> CSIPacket
# ---------------------------------------------------------------------------
# Column positions in the ESP32 CSI_DATA line (see firmware / mock output).
_COL_SEQ = 1
_COL_MAC = 2
_COL_RSSI = 3
_COL_CHANNEL = 16
_COL_TIMESTAMP = 18


class CSIParser:
    """Parses a raw ``CSI_DATA,...,"[I0,Q0,...]"`` line into a CSIPacket."""

    def parse(self, line: str, receiver_index: int) -> CSIPacket | None:
        """Return a CSIPacket, or None if the line is not valid CSI."""
        line = line.strip()
        if not line.startswith("CSI_DATA"):
            return None

        # The I/Q array is the bracketed section at the tail of the line.
        open_bracket = line.find("[")
        close_bracket = line.rfind("]")
        if open_bracket == -1 or close_bracket <= open_bracket:
            return None

        metadata = line[:open_bracket].rstrip(',"').split(",")
        if len(metadata) < 19:
            return None
        try:
            seq_id = int(metadata[_COL_SEQ])
            mac = metadata[_COL_MAC]
            rssi = int(metadata[_COL_RSSI])
            channel = int(metadata[_COL_CHANNEL])
            timestamp = int(metadata[_COL_TIMESTAMP])
        except (ValueError, IndexError):
            return None

        # Parse the interleaved I,Q integers into a complex vector.
        body = line[open_bracket + 1:close_bracket]
        tokens = body.split(",") if "," in body else body.split()
        if len(tokens) < 2 * NUM_SUBCARRIERS:
            return None
        values = np.array(tokens[: 2 * NUM_SUBCARRIERS], dtype=np.float32)
        csi = (values[0::2] + 1j * values[1::2]).astype(np.complex64)

        return CSIPacket(receiver_index, seq_id, mac, rssi, channel, timestamp, csi)


# ---------------------------------------------------------------------------
# Rolling complex buffer, shared across receiver threads
# ---------------------------------------------------------------------------
class ComplexMatrix:
    """Thread-safe rolling buffer of complex CSI, shape [receivers x 64 x time]."""

    def __init__(self, max_time: int = DEFAULT_BUFFER_SIZE):
        self.max_time = max_time
        self._buffer = np.zeros((NUM_RECEIVERS, NUM_SUBCARRIERS, max_time), dtype=np.complex64)
        self._written = np.zeros(NUM_RECEIVERS, dtype=np.int64)   # total ever written per rx
        self._lock = threading.Lock()

    def append(self, receiver_index: int, csi: np.ndarray) -> None:
        """Store one (64,) complex sample for a receiver, overwriting the oldest."""
        with self._lock:
            slot = int(self._written[receiver_index] % self.max_time)
            self._buffer[receiver_index, :, slot] = csi
            self._written[receiver_index] += 1

    def get_latest(self, n: int) -> np.ndarray:
        """Return the newest n samples per receiver, shape (3, 64, n), zero-padded."""
        with self._lock:
            n = min(n, self.max_time)
            out = np.zeros((NUM_RECEIVERS, NUM_SUBCARRIERS, n), dtype=np.complex64)
            for rx in range(NUM_RECEIVERS):
                have = min(int(self._written[rx]), n)
                if have == 0:
                    continue
                # Ring indices of the `have` most recent samples, oldest -> newest.
                recent = (np.arange(self._written[rx] - have, self._written[rx])
                          % self.max_time)
                # Index rx first, then columns, so the result stays (64, have)
                # rather than numpy moving the fancy-index axis to the front.
                out[rx, :, n - have:] = self._buffer[rx][:, recent]
            return out

    def get_receiver_count(self, receiver_index: int) -> int:
        """Total samples ever written for a receiver."""
        with self._lock:
            return int(self._written[receiver_index])


# ---------------------------------------------------------------------------
# Packet sources — where CSI lines come from
# ---------------------------------------------------------------------------
class PacketSource:
    """Base source: hands out (receiver_index, CSI line) pairs on demand."""

    def poll(self, timeout: float) -> tuple[int, str] | None:
        """Return the next (receiver_index, line), or None if none within timeout."""
        raise NotImplementedError

    def eof(self) -> bool:
        """True when a finite source is exhausted (streaming sources: always False)."""
        return False

    def stop(self) -> None:
        """Ask the source to stop (no-op by default)."""

    def close(self) -> None:
        """Release any resources (no-op by default)."""


class SerialSource(PacketSource):
    """Reads CSI lines from one ESP32 over USB serial (the real-hardware source)."""

    def __init__(self, port: str, receiver_index: int, baud_rate: int = DEFAULT_BAUD_RATE):
        if serial is None:
            raise RuntimeError("pyserial not installed; SerialSource unavailable")
        self.port = port
        self.receiver_index = receiver_index
        self.baud_rate = baud_rate
        self._serial = None

    def _open_once(self) -> bool:
        """Open the serial port lazily; return False if it cannot be opened."""
        if self._serial is not None:
            return True
        try:
            self._serial = serial.Serial(self.port, self.baud_rate, timeout=SERIAL_READ_TIMEOUT)
            logger.info("Rx%d: opened %s @ %d baud", self.receiver_index, self.port, self.baud_rate)
            return True
        except (serial.SerialException, OSError) as e:
            logger.error("Rx%d: cannot open %s — %s", self.receiver_index, self.port, e)
            return False

    def poll(self, timeout: float) -> tuple[int, str] | None:
        """Read one line from the serial port (None on timeout / error)."""
        if not self._open_once():
            time.sleep(timeout)
            return None
        try:
            raw = self._serial.readline()
        except (serial.SerialException, OSError) as e:
            logger.error("Rx%d: serial read error — %s", self.receiver_index, e)
            return None
        if not raw:
            return None
        return self.receiver_index, raw.decode("utf-8", errors="replace")

    def close(self) -> None:
        """Close the serial port."""
        if self._serial is not None:
            self._serial.close()
            self._serial = None


# ---------------------------------------------------------------------------
# Gateway: run a consumer thread per source into the shared matrix
# ---------------------------------------------------------------------------
class GatewayV2:
    """Reads packet sources into one shared complex matrix, tracking loss stats."""

    def __init__(self, sources: list[PacketSource], buffer_size: int = DEFAULT_BUFFER_SIZE):
        self._sources = list(sources)
        self._matrix = ComplexMatrix(max_time=buffer_size)
        self._parser = CSIParser()
        self._threads: list[threading.Thread] = []
        self._running = False
        self._last_seq: dict[int, int] = {}
        self._stats = {rx: {"received": 0, "lost": 0} for rx in range(NUM_RECEIVERS)}
        self._stats_lock = threading.Lock()

    @classmethod
    def from_serial(cls, ports: list[str], baud_rate: int = DEFAULT_BAUD_RATE,
                    buffer_size: int = DEFAULT_BUFFER_SIZE) -> "GatewayV2":
        """Build a gateway that reads the given serial ports as RX0, RX1, RX2."""
        sources = [SerialSource(p, i, baud_rate) for i, p in enumerate(ports[:NUM_RECEIVERS])]
        return cls(sources, buffer_size=buffer_size)

    def start(self) -> None:
        """Start one consumer thread per source."""
        self._running = True
        for i, source in enumerate(self._sources):
            thread = threading.Thread(target=self._consume, args=(source,),
                                      name=f"v2-source-{i}", daemon=True)
            self._threads.append(thread)
            thread.start()
        logger.info("GatewayV2 started with %d source(s)", len(self._sources))

    def _consume(self, source: PacketSource) -> None:
        """Pull lines from one source, parse them, and store them in the matrix."""
        while self._running and not source.eof():
            item = source.poll(0.2)
            if item is None:
                continue
            receiver_index, line = item
            packet = self._parser.parse(line, receiver_index)
            if packet is None:
                continue
            self._record_sequence(receiver_index, packet.seq_id)
            self._matrix.append(receiver_index, packet.csi)
            with self._stats_lock:
                self._stats[receiver_index]["received"] += 1

    def _record_sequence(self, receiver_index: int, seq_id: int) -> None:
        """Count dropped packets from gaps in the per-receiver sequence number."""
        previous = self._last_seq.get(receiver_index)
        if previous is not None and seq_id > previous + 1:
            with self._stats_lock:
                self._stats[receiver_index]["lost"] += seq_id - previous - 1
        self._last_seq[receiver_index] = seq_id

    def stop(self) -> None:
        """Stop all consumer threads and close the sources."""
        self._running = False
        for source in self._sources:
            source.stop()
        for thread in self._threads:
            thread.join(timeout=3.0)
        for source in self._sources:
            source.close()
        logger.info("GatewayV2 stopped")

    def wait(self, timeout: float = 5.0) -> bool:
        """Wait for consumer threads to finish (used with finite replay sources)."""
        deadline = time.time() + timeout
        for thread in self._threads:
            thread.join(timeout=max(0.0, deadline - time.time()))
        return not any(t.is_alive() for t in self._threads)

    def get_matrix(self) -> ComplexMatrix:
        """The shared complex CSI matrix downstream stages read from."""
        return self._matrix

    def get_stats(self) -> dict:
        """Per-receiver {received, lost} counters."""
        with self._stats_lock:
            return {f"Rx{rx}": dict(s) for rx, s in self._stats.items()}

    def is_alive(self) -> bool:
        """True while any consumer thread is running."""
        return any(t.is_alive() for t in self._threads)
