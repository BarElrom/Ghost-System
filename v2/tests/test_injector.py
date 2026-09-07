"""Phase 2 tests — injector + Embedded WiFi adapter.

Covers:
  - EmbeddedWiFiAdapter parsing (ESP32 CSV -> complex 64) + subcarrier normalize
  - Injector calibration preamble + operational stream (decoded raw off the wire)
  - 3-node fan-out: synthesized (single-link) and direct (>=3 streams)
  - Full injector -> UDPSource -> gateway_v2 complex arrival

Run:  python v2/tests/test_injector.py
"""

import os
import sys
import tempfile
import time

from _harness import Harness

import numpy as np

from v2.injector.adapters.base import InMemoryAdapter, Snapshot
from v2.injector.adapters.embedded_wifi import EmbeddedWiFiAdapter
from v2.injector.injector import Injector, _NODE_NAMES, _DEFAULT_GAINS
from v2.transport.frame import Frame
from v2.transport.mock_esp32 import MockESP32, frame_to_csi_line
from v2.ghost.gateway_v2 import GatewayV2
from v2.ghost.sources import UDPSource


def _full(val: complex, n: int = 64) -> np.ndarray:
    return np.full(n, val, dtype=np.complex64)


def _wait_until(cond, timeout: float = 3.0, interval: float = 0.02) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(interval)
    return cond()


def _recv_n(mock: MockESP32, n: int, timeout: float = 2.0):
    """Receive n frames, grouped by node name -> list[Frame] sorted by seq."""
    by_node: dict[str, list] = {}
    for _ in range(n):
        got = mock.receive(timeout)
        if got is None:
            break
        node, frame, _addr = got
        by_node.setdefault(node, []).append(frame)
    for node in by_node:
        by_node[node].sort(key=lambda fr: fr.frame_seq)
    return by_node


def test_embedded_wifi_adapter_parse(h: Harness) -> None:
    i = np.arange(64, dtype=np.float32) - 5.0
    q = (np.arange(64, dtype=np.float32) + 2.0)
    iq = (i + 1j * q).astype(np.complex64)
    line = frame_to_csi_line(Frame(node_id=1, frame_seq=0, iq=iq))

    path = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False).name
    try:
        with open(path, "w") as f:
            f.write("header,line,with,no,brackets\n")
            f.write(line + "\n")
            f.write(line + "\n")
        snaps = list(EmbeddedWiFiAdapter(path, iq_order="iq").snapshots())
        h.expect("parsed 2 frames (header skipped)", len(snaps) == 2, str(len(snaps)))
        got = snaps[0].iq_by_stream[0]
        h.expect("64 subcarriers", got.shape == (64,), str(got.shape))
        h.expect("I recovered", np.allclose(got.real, i, atol=1e-3))
        h.expect("Q recovered", np.allclose(got.imag, q, atol=1e-3))
    finally:
        os.unlink(path)


def test_embedded_wifi_normalization(h: Harness) -> None:
    long_nums = ",".join(str(v) for v in range(256))
    short_nums = ",".join(str(v) for v in range(80))
    path = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False).name
    try:
        with open(path, "w") as f:
            f.write(f'x,"[{long_nums}]"\n')
            f.write(f'x,"[{short_nums}]"\n')
        snaps = list(EmbeddedWiFiAdapter(path).snapshots())
        h.expect("both lines parsed", len(snaps) == 2)
        h.expect("long line normalized to 64", snaps[0].iq_by_stream[0].shape == (64,))
        h.expect("short line normalized to 64", snaps[1].iq_by_stream[0].shape == (64,))
        h.expect("truncation keeps head", snaps[0].iq_by_stream[0][0] == (0 + 1j))
    finally:
        os.unlink(path)


def test_injector_preamble_and_operational(h: Harness) -> None:
    mock = MockESP32(bind_host="127.0.0.1", bind_port=0)
    try:
        B, M = 500 + 100j, 1000 + 300j
        adapter = InMemoryAdapter([Snapshot({0: _full(B)}), Snapshot({0: _full(M)})], num_streams=1)
        inj = Injector(
            adapter,
            net_map={n: ("127.0.0.1", mock.port) for n in _NODE_NAMES},
            rate_hz=0, calibration_samples=2, calib_window=1,
        )
        inj.run()
        inj.close()

        by_node = _recv_n(mock, 12)
        h.expect("all 3 nodes received", set(by_node) == {"RX1", "RX2", "RX3"}, str(set(by_node)))
        h.expect("4 frames per node", all(len(v) == 4 for v in by_node.values()))

        rx1 = by_node.get("RX1", [])
        if len(rx1) == 4:
            flags = [fr.calibration for fr in rx1]
            h.expect("calib flags = [T,T,F,F]", flags == [True, True, False, False], str(flags))
            h.expect("calib value == baseline B", np.allclose(rx1[0].iq.real, 500) and np.allclose(rx1[0].iq.imag, 100))
            h.expect("last operational == M", np.allclose(rx1[3].iq.real, 1000) and np.allclose(rx1[3].iq.imag, 300))
        h.expect("single-link marked synthesized", inj.synthesized_nodes is True)
    finally:
        mock.close()


def test_injector_direct_map_three_streams(h: Harness) -> None:
    mock = MockESP32(bind_host="127.0.0.1", bind_port=0)
    try:
        snap = Snapshot({0: _full(10), 1: _full(20), 2: _full(30)})
        adapter = InMemoryAdapter([snap], num_streams=3)
        inj = Injector(
            adapter,
            net_map={n: ("127.0.0.1", mock.port) for n in _NODE_NAMES},
            rate_hz=0, calibration_samples=1, calib_window=1,
        )
        h.expect("3-stream dataset not synthesized", inj.synthesized_nodes is False)
        inj.run()
        inj.close()

        by_node = _recv_n(mock, 6)
        ok = (
            np.allclose(by_node["RX1"][1].iq.real, 10)
            and np.allclose(by_node["RX2"][1].iq.real, 20)
            and np.allclose(by_node["RX3"][1].iq.real, 30)
        )
        h.expect("streams mapped directly RX1/2/3 = 10/20/30", ok)
    finally:
        mock.close()


def test_injector_to_gateway_v2(h: Harness) -> None:
    udp = UDPSource(bind_host="127.0.0.1", bind_port=0)
    gw = GatewayV2(sources=[udp])
    gw.start()
    try:
        B, M = 500 + 0j, 1000 + 300j
        adapter = InMemoryAdapter([Snapshot({0: _full(B)}), Snapshot({0: _full(M)})], num_streams=1)
        inj = Injector(
            adapter,
            net_map={n: ("127.0.0.1", udp.port) for n in _NODE_NAMES},
            rate_hz=0, calibration_samples=2, calib_window=1,
        )
        inj.run()
        inj.close()

        m = gw.get_matrix()
        arrived = _wait_until(
            lambda: all(m.get_receiver_count(rx) >= 4 for rx in range(3)), timeout=3.0
        )
        h.expect("frames reached all gateway receivers", arrived,
                 f"counts={[m.get_receiver_count(r) for r in range(3)]}")

        latest = gw.get_matrix().get_latest(1)
        h.expect("RX1 last real == 1000", np.isclose(latest[0, 0, 0].real, 1000, atol=1.0))
        h.expect("RX1 last imag == 300 (phase kept)", np.isclose(latest[0, 0, 0].imag, 300, atol=1.0))
        exp = np.complex64(M) * _DEFAULT_GAINS["RX2"]
        h.expect("RX2 last matches M*gain", np.isclose(latest[1, 0, 0].real, exp.real, atol=1.5)
                 and np.isclose(latest[1, 0, 0].imag, exp.imag, atol=1.5))
        h.expect("RX3 also received", gw.get_matrix().get_receiver_count(2) >= 4)
    finally:
        gw.stop()


def main() -> int:
    h = Harness("injector + embedded_wifi")
    h.case("adapter_parse", test_embedded_wifi_adapter_parse)
    h.case("adapter_normalization", test_embedded_wifi_normalization)
    h.case("injector_preamble_and_operational", test_injector_preamble_and_operational)
    h.case("injector_direct_map", test_injector_direct_map_three_streams)
    h.case("injector_to_gateway_v2", test_injector_to_gateway_v2)
    return h.done()


if __name__ == "__main__":
    sys.exit(main())
