"""
v2 transport — injector-side UDP sender.

Thin wrapper that serializes a frame and sends it to the configured network
target for a given receiver node. The injector service (Service 1) uses this
to push dataset CSI into the ESP32 receivers over Wi-Fi.
"""

import logging
import socket

import numpy as np

from v2.config_v2 import NODE_IDS, NODE_NET_MAP
from v2.transport.frame import encode_frame

logger = logging.getLogger("ghost.v2.udp_sender")


class UDPSender:
    """Sends injection frames to ESP32 receivers over UDP.

    Args:
        net_map: mapping of node name -> (ip, port). Defaults to
            config_v2.NODE_NET_MAP. Pass a custom map for local testing.
    """

    def __init__(self, net_map: dict[str, tuple[str, int]] | None = None):
        self._net_map = dict(net_map) if net_map is not None else dict(NODE_NET_MAP)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(
        self,
        node_name: str,
        frame_seq: int,
        iq: np.ndarray,
        calibration: bool = False,
    ) -> int:
        """Encode and send one frame to the named node.

        Returns the number of bytes sent.

        Raises:
            KeyError: if node_name is unknown.
        """
        if node_name not in self._net_map:
            raise KeyError(f"unknown node {node_name!r}; known: {list(self._net_map)}")
        node_id = NODE_IDS[node_name]
        payload = encode_frame(node_id, frame_seq, iq, calibration=calibration)
        return self._sock.sendto(payload, self._net_map[node_name])

    def close(self) -> None:
        self._sock.close()

    def __enter__(self) -> "UDPSender":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
