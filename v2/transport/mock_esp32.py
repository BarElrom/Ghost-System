"""
v2 transport — software mock of the ESP32 loopback firmware.

Stands in for the real `csi_inject_recv` firmware (Plan section 4) so the whole
transport contract can be validated before flashing hardware. Its job mirrors
the firmware exactly:

    UDP frame (over the wire)  ->  decode  ->  emit a CSI_DATA serial line

The emitted line matches the standard ESP32 `csi_recv` CSV format, so the
existing ghost gateway / CSIParser can consume it unchanged. Only id, timestamp
and the I/Q array are faithful; all other metadata fields are constant fillers.

Run as a standalone listener (prints CSI_DATA lines it receives):

    python -m v2.transport.mock_esp32
"""

import logging
import socket
import threading

import numpy as np

from v2.config_v2 import (
    DEFAULT_NOISE_FLOOR,
    DEFAULT_RSSI,
    NODE_ID_TO_NAME,
    SAMPLE_RATE_HZ,
    TX_MAC,
    UDP_PORT,
    WIFI_CHANNEL,
)
from v2.transport.frame import Frame, decode_frame

logger = logging.getLogger("ghost.v2.mock_esp32")

# Microseconds per frame at the configured rate (deterministic timestamp).
_US_PER_FRAME = 1_000_000 // SAMPLE_RATE_HZ


def frame_to_csi_line(frame: Frame) -> str:
    """Format a decoded frame as a standard ESP32 CSI_DATA CSV line.

    Column layout (indices 0-24) matches esp-csi `csi_recv` output:
        type,id,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,not_sounding,
        aggregation,stbc,fec_coding,sgi,noise_floor,ampdu_cnt,channel,
        secondary_channel,local_timestamp,ant,sig_len,rx_format,len,first_word,
        "[I0,Q0,...,I63,Q63]"
    """
    i = np.clip(np.rint(frame.iq.real), -32768, 32767).astype(np.int16)
    q = np.clip(np.rint(frame.iq.imag), -32768, 32767).astype(np.int16)

    interleaved = np.empty(frame.num_sub * 2, dtype=np.int16)
    interleaved[0::2] = i
    interleaved[1::2] = q
    csi_array = "[" + ",".join(str(int(v)) for v in interleaved) + "]"

    n_vals = frame.num_sub * 2
    timestamp = frame.frame_seq * _US_PER_FRAME

    fields = [
        "CSI_DATA",          # 0  type
        str(frame.frame_seq),  # 1  id (sequence)
        TX_MAC,              # 2  mac
        str(DEFAULT_RSSI),   # 3  rssi
        "11",                # 4  rate
        "1",                 # 5  sig_mode
        "7",                 # 6  mcs
        "1",                 # 7  bandwidth (40 MHz)
        "0",                 # 8  smoothing
        "1",                 # 9  not_sounding
        "0",                 # 10 aggregation
        "0",                 # 11 stbc
        "0",                 # 12 fec_coding
        "0",                 # 13 sgi
        str(DEFAULT_NOISE_FLOOR),  # 14 noise_floor
        "0",                 # 15 ampdu_cnt
        str(WIFI_CHANNEL),   # 16 channel
        "0",                 # 17 secondary_channel
        str(timestamp),      # 18 local_timestamp
        "0",                 # 19 ant
        str(n_vals),         # 20 sig_len
        "1",                 # 21 rx_format
        str(n_vals),         # 22 len
        "0",                 # 23 first_word
    ]
    return ",".join(fields) + ',"' + csi_array + '"'


class MockESP32:
    """UDP listener that decodes injection frames and formats CSI_DATA lines.

    Args:
        bind_host: interface to bind (default localhost for testing).
        bind_port: UDP port (0 = ephemeral, read back via `.port`).
    """

    def __init__(self, bind_host: str = "127.0.0.1", bind_port: int = UDP_PORT):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
        except OSError:
            pass
        self._sock.bind((bind_host, bind_port))
        self._running = False

    @property
    def port(self) -> int:
        """Actual bound UDP port (useful when binding to port 0)."""
        return self._sock.getsockname()[1]

    def receive(self, timeout: float | None = 1.0):
        """Receive and decode one frame.

        Returns (node_name, Frame, addr), or None on timeout.
        """
        self._sock.settimeout(timeout)
        try:
            data, addr = self._sock.recvfrom(2048)
        except socket.timeout:
            return None
        frame = decode_frame(data)
        node_name = NODE_ID_TO_NAME.get(frame.node_id, f"RX?{frame.node_id}")
        return node_name, frame, addr

    def serve(self, sink, stop_event: "threading.Event | None" = None) -> None:
        """Loop: receive frames and push formatted CSI_DATA lines to `sink`.

        Args:
            sink: callable(node_name: str, line: str) -> None.
            stop_event: optional threading.Event; loop exits when set.
        """
        self._running = True
        self._sock.settimeout(0.5)
        while self._running and (stop_event is None or not stop_event.is_set()):
            try:
                data, _addr = self._sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                frame = decode_frame(data)
            except ValueError as e:
                logger.warning("dropping malformed frame: %s", e)
                continue
            node_name = NODE_ID_TO_NAME.get(frame.node_id, f"RX?{frame.node_id}")
            sink(node_name, frame_to_csi_line(frame))
        self._running = False

    def stop(self) -> None:
        self._running = False

    def close(self) -> None:
        self._sock.close()


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    mock = MockESP32()
    print(f"Mock ESP32 listening on UDP {mock.port}. Ctrl+C to stop.")

    def sink(node_name: str, line: str) -> None:
        print(f"[{node_name}] {line}")

    try:
        mock.serve(sink)
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        mock.close()


if __name__ == "__main__":
    _main()
