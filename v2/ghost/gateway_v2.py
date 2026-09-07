
import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

try:
    import serial
except ImportError:
    serial = None

from v2.config_v2 import NUM_RECEIVERS, NUM_SUBCARRIERS
from v2.transport.frame import HEADER_SIZE, MAGIC, decode_frame, frame_to_csi_line

DEFAULT_BAUD_RATE = 921600
DEFAULT_BUFFER_SIZE = 1000
SERIAL_READ_TIMEOUT = 1.0

logger = logging.getLogger("ghost.v2.gateway")


@dataclass
class CSIPacket:
    """A single CSI reading from one receiver, carrying complex subcarriers."""

    receiver_index: int
    seq_id: int
    mac: str
    rssi: int
    channel: int
    timestamp: int
    csi: np.ndarray

    @property
    def amplitude(self) -> np.ndarray:
        """Per-subcarrier magnitude |csi| = sqrt(I^2 + Q^2)."""
        return np.abs(self.csi).astype(np.float32)

    @property
    def phase(self) -> np.ndarray:
        """Per-subcarrier phase angle(csi) = atan2(Q, I), in radians."""
        return np.angle(self.csi).astype(np.float32)


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

        body = line[open_bracket + 1:close_bracket]
        tokens = body.split(",") if "," in body else body.split()
        if len(tokens) < 2 * NUM_SUBCARRIERS:
            return None
        values = np.array(tokens[: 2 * NUM_SUBCARRIERS], dtype=np.float32)
        csi = (values[0::2] + 1j * values[1::2]).astype(np.complex64)

        return CSIPacket(receiver_index, seq_id, mac, rssi, channel, timestamp, csi)


class ComplexMatrix:
    """Thread-safe rolling buffer of complex CSI, shape [receivers x 64 x time].

    Alongside each CSI sample it keeps two parallel time buffers (Note 1):
      * ``_ts_device`` — the ESP32 ``local_timestamp`` (µs, ``esp_timer_get_time``)
        carried in the CSI line; the *capture* instant on the board.
      * ``_ts_host``   — ``time.monotonic_ns()`` stamped when the host ingested
        the sample; the *arrival* instant.
      * ``_seq``       — the per-receiver sequence id.
    These let the FrameManager align the N receivers onto one timeline instead of
    treating same-index samples across receivers as simultaneous (they are not).
    """

    def __init__(self, max_time: int = DEFAULT_BUFFER_SIZE):
        self.max_time = max_time
        self._buffer = np.zeros((NUM_RECEIVERS, NUM_SUBCARRIERS, max_time), dtype=np.complex64)
        self._ts_device = np.zeros((NUM_RECEIVERS, max_time), dtype=np.int64)
        self._ts_host = np.zeros((NUM_RECEIVERS, max_time), dtype=np.int64)
        self._seq = np.full((NUM_RECEIVERS, max_time), -1, dtype=np.int64)
        self._written = np.zeros(NUM_RECEIVERS, dtype=np.int64)
        self._lock = threading.Lock()

    def append(self, receiver_index: int, csi: np.ndarray,
               device_ts: int = 0, host_ts: int = 0, seq_id: int = -1) -> None:
        """Store one (64,) complex sample + its timestamps, overwriting the oldest."""
        with self._lock:
            slot = int(self._written[receiver_index] % self.max_time)
            self._buffer[receiver_index, :, slot] = csi
            self._ts_device[receiver_index, slot] = device_ts
            self._ts_host[receiver_index, slot] = host_ts
            self._seq[receiver_index, slot] = seq_id
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
                recent = (np.arange(self._written[rx] - have, self._written[rx])
                          % self.max_time)
                out[rx, :, n - have:] = self._buffer[rx][:, recent]
            return out

    def get_streams(self, n: int) -> list[dict]:
        """Per-receiver ordered snapshot of the newest ``n`` samples + timestamps.

        Returns one dict per receiver, oldest-first, with only the samples that
        actually exist (no zero-padding — the FrameManager needs real timestamps):

            {"csi": [64, k] complex, "device_ts": [k], "host_ts": [k], "seq": [k]}

        where ``k = min(n, samples_written)``. This is the raw material the
        FrameManager joins across receivers onto a common timeline.
        """
        with self._lock:
            n = min(n, self.max_time)
            streams = []
            for rx in range(NUM_RECEIVERS):
                have = min(int(self._written[rx]), n)
                if have == 0:
                    streams.append({
                        "csi": np.zeros((NUM_SUBCARRIERS, 0), dtype=np.complex64),
                        "device_ts": np.zeros(0, dtype=np.int64),
                        "host_ts": np.zeros(0, dtype=np.int64),
                        "seq": np.zeros(0, dtype=np.int64),
                    })
                    continue
                idx = (np.arange(self._written[rx] - have, self._written[rx])
                       % self.max_time)
                streams.append({
                    "csi": self._buffer[rx][:, idx].copy(),
                    "device_ts": self._ts_device[rx][idx].copy(),
                    "host_ts": self._ts_host[rx][idx].copy(),
                    "seq": self._seq[rx][idx].copy(),
                })
            return streams

    def get_receiver_count(self, receiver_index: int) -> int:
        """Total samples ever written for a receiver."""
        with self._lock:
            return int(self._written[receiver_index])


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


class BinarySerialSource(PacketSource):
    """Reads binary G2 frames from one ESP32 over USB serial (Note 3).

    The updated firmware emits length-delimited G2 frames (``transport/frame.py``)
    instead of ASCII ``CSI_DATA`` lines — ≈4× less UART traffic, the real fix for
    the packet loss the string format caused. This source resynchronizes on the
    ``"G2"`` magic, reads exactly one frame, and hands it to the gateway as a
    ``CSI_DATA`` line (``frame_to_csi_line``) so every downstream stage is
    unchanged. Routing is by USB port, so the frame's ``node_id`` is ignored —
    ``receiver_index`` comes from which cable this is.
    """

    def __init__(self, port: str, receiver_index: int, baud_rate: int = DEFAULT_BAUD_RATE):
        if serial is None:
            raise RuntimeError("pyserial not installed; BinarySerialSource unavailable")
        self.port = port
        self.receiver_index = receiver_index
        self.baud_rate = baud_rate
        self._serial = None
        self._buf = bytearray()

    def _open_once(self) -> bool:
        """Open the serial port lazily; return False if it cannot be opened."""
        if self._serial is not None:
            return True
        try:
            self._serial = serial.Serial(self.port, self.baud_rate, timeout=SERIAL_READ_TIMEOUT)
            logger.info("Rx%d: opened %s @ %d baud (binary G2)",
                        self.receiver_index, self.port, self.baud_rate)
            return True
        except (serial.SerialException, OSError) as e:
            logger.error("Rx%d: cannot open %s — %s", self.receiver_index, self.port, e)
            return False

    def poll(self, timeout: float) -> tuple[int, str] | None:
        """Read one whole G2 frame and present it as a CSI_DATA line."""
        if not self._open_once():
            time.sleep(timeout)
            return None
        frame = self._read_frame(timeout)
        if frame is None:
            return None
        return self.receiver_index, frame_to_csi_line(frame)

    def _read_frame(self, timeout: float):
        """Resync on MAGIC, read one full G2 frame, decode it (None on timeout)."""
        deadline = time.monotonic() + timeout
        try:
            idx = self._buf.find(MAGIC)
            while idx == -1:
                # keep only a trailing byte, in case MAGIC straddles two reads
                if len(self._buf) > 1:
                    del self._buf[:-1]
                if not self._fill(deadline):
                    return None
                idx = self._buf.find(MAGIC)
            del self._buf[:idx]

            while len(self._buf) < HEADER_SIZE:
                if not self._fill(deadline):
                    return None
            num_sub = int.from_bytes(self._buf[10:12], "big")
            if not 0 < num_sub <= NUM_SUBCARRIERS * 64:  # corrupt header guard
                del self._buf[:2]  # drop this magic, resync past it
                return None
            total = HEADER_SIZE + num_sub * 4

            while len(self._buf) < total:
                if not self._fill(deadline):
                    return None
            raw = bytes(self._buf[:total])
            frame = decode_frame(raw)  # may raise ValueError on bad magic/version
            del self._buf[:total]      # consume only once the frame is valid
            return frame
        except (serial.SerialException, OSError) as e:
            logger.error("Rx%d: serial read error — %s", self.receiver_index, e)
            return None
        except ValueError as e:
            logger.warning("Rx%d: dropping malformed frame — %s", self.receiver_index, e)
            del self._buf[:2]  # step past this magic so we resync forward
            return None

    def _fill(self, deadline: float) -> bool:
        """Pull available bytes into the buffer; False once past the deadline."""
        if time.monotonic() >= deadline:
            return False
        want = self._serial.in_waiting or 1
        chunk = self._serial.read(want)
        if chunk:
            self._buf.extend(chunk)
            return True
        return time.monotonic() < deadline

    def close(self) -> None:
        """Close the serial port."""
        if self._serial is not None:
            self._serial.close()
            self._serial = None


class GatewayV2:
    """Reads packet sources into one shared complex matrix, tracking loss stats."""

    def __init__(self, sources: list[PacketSource], buffer_size: int = DEFAULT_BUFFER_SIZE,
                 align_on: str = "seq"):
        from v2.ghost.frame_manager import FrameManager
        self._sources = list(sources)
        self._matrix = ComplexMatrix(max_time=buffer_size)
        self._manager = FrameManager(num_receivers=NUM_RECEIVERS, align_on=align_on)
        self._parser = CSIParser()
        self._threads: list[threading.Thread] = []
        self._running = False
        self._last_seq: dict[int, int] = {}
        self._stats = {rx: {"received": 0, "lost": 0} for rx in range(NUM_RECEIVERS)}
        self._stats_lock = threading.Lock()

    @classmethod
    def from_serial(cls, ports: list[str], baud_rate: int = DEFAULT_BAUD_RATE,
                    buffer_size: int = DEFAULT_BUFFER_SIZE,
                    align_on: str = "seq", binary: bool = True) -> "GatewayV2":
        """Build a gateway that reads the given serial ports as RX0, RX1, RX2.

        ``binary=True`` (default) expects the G2-binary firmware; pass
        ``binary=False`` for boards still flashed with the ASCII CSI_DATA firmware.
        """
        src_cls = BinarySerialSource if binary else SerialSource
        sources = [src_cls(p, i, baud_rate) for i, p in enumerate(ports[:NUM_RECEIVERS])]
        return cls(sources, buffer_size=buffer_size, align_on=align_on)

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
            host_ts = time.monotonic_ns()
            receiver_index, line = item
            packet = self._parser.parse(line, receiver_index)
            if packet is None:
                continue
            self._record_sequence(receiver_index, packet.seq_id)
            self._matrix.append(receiver_index, packet.csi,
                                device_ts=packet.timestamp, host_ts=host_ts,
                                seq_id=packet.seq_id)
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

    def get_aligned_window(self, n: int, return_window: bool = False):
        """Cross-receiver-aligned newest-``n`` window (Note 5), drop-in for
        ``get_matrix().get_latest(n)`` but with genuinely simultaneous columns.

        Uses the gateway's FrameManager to join the per-receiver streams by
        sequence id (the same TX frame seen by every RX). Returns the ``[N×64×T]``
        complex matrix, or the full ``AlignedWindow`` when ``return_window`` is set.
        """
        streams = self._matrix.get_streams(n)
        window = self._manager.align(streams)
        return window if return_window else window.matrix

    def get_stats(self) -> dict:
        """Per-receiver {received, lost} counters."""
        with self._stats_lock:
            return {f"Rx{rx}": dict(s) for rx, s in self._stats.items()}

    def is_alive(self) -> bool:
        """True while any consumer thread is running."""
        return any(t.is_alive() for t in self._threads)
