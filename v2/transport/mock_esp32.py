
import logging
import socket
import threading

from v2.config_v2 import NODE_ID_TO_NAME, UDP_PORT
from v2.transport.frame import Frame, decode_frame, frame_to_csi_line

logger = logging.getLogger("ghost.v2.mock_esp32")

# ``frame_to_csi_line`` now lives in the codec module (transport/frame.py) — it is
# the inverse of CSIParser.parse and is shared by the hardware binary source too.
# Re-exported here so existing ``from v2.transport.mock_esp32 import
# frame_to_csi_line`` imports keep working.
__all__ = ["MockESP32", "frame_to_csi_line"]


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
