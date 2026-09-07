"""
v2 ghost — simulation packet sources (hardware-free).

These let the gateway_v2 pipeline run end to end without any ESP32 hardware:

  - ListSource : replay a fixed list of (receiver_index, CSI_DATA line) items.
                 Fully deterministic — used by tests and offline replay.
  - UDPSource  : bind a UDP socket, receive G2 injection frames, and present
                 them as CSI_DATA lines. This collapses "ESP32 + USB serial"
                 into one software hop so the whole injector -> ghost chain can
                 be exercised on one machine.

The production path uses SerialSource (in gateway_v2.py). These sources import
the transport codec + the mock's line formatter, so gateway_v2 core stays free
of any simulation/mock dependency.
"""

import logging
import socket

from v2.config_v2 import NODE_ID_TO_NAME, NODE_IDS, UDP_PORT
from v2.ghost.gateway_v2 import PacketSource
from v2.transport.frame import decode_frame
from v2.transport.mock_esp32 import frame_to_csi_line

logger = logging.getLogger("ghost.v2.sources")


class ListSource(PacketSource):
    """Replay a fixed list of (receiver_index, line) items, then signal eof."""

    def __init__(self, items: list[tuple[int, str]]):
        self._items = list(items)
        self._pos = 0

    def poll(self, timeout: float) -> tuple[int, str] | None:
        if self._pos >= len(self._items):
            return None
        item = self._items[self._pos]
        self._pos += 1
        return item

    def eof(self) -> bool:
        return self._pos >= len(self._items)


class UDPSource(PacketSource):
    """Receive G2 injection frames over UDP and yield CSI_DATA lines.

    Maps the frame's node_id to a receiver_index. Node ids are 1-based
    (RX1/2/3); receiver_index is 0-based (0/1/2).
    """

    def __init__(self, bind_host: str = "0.0.0.0", bind_port: int = UDP_PORT):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
        except OSError:
            pass
        self._sock.bind((bind_host, bind_port))
        self._node_to_rx = {
            name: idx for idx, name in enumerate(sorted(NODE_IDS, key=NODE_IDS.get))
        }

    @property
    def port(self) -> int:
        return self._sock.getsockname()[1]

    def poll(self, timeout: float) -> tuple[int, str] | None:
        self._sock.settimeout(timeout)
        try:
            data, _addr = self._sock.recvfrom(2048)
        except socket.timeout:
            return None
        except OSError:
            return None
        try:
            frame = decode_frame(data)
        except ValueError as e:
            logger.warning("dropping malformed frame: %s", e)
            return None
        node_name = NODE_ID_TO_NAME.get(frame.node_id)
        if node_name is None or node_name not in self._node_to_rx:
            logger.warning("unknown node_id %s; dropping", frame.node_id)
            return None
        rx = self._node_to_rx[node_name]
        return rx, frame_to_csi_line(frame)

    def close(self) -> None:
        self._sock.close()
