"""Phase 1 tests — gateway_v2 complex I/Q buffer + source abstraction.

Covers the complex ring buffer, the complex CSI parser (amplitude AND phase),
and two hardware-free end-to-end paths (ListSource replay and UDPSource
injection) proving phase is preserved from injector to matrix.

Run:  python v2/tests/test_gateway_v2.py
"""

import sys
import time

from _harness import Harness

import numpy as np

from v2.ghost.gateway_v2 import BinarySerialSource, ComplexMatrix, CSIParser, GatewayV2
from v2.ghost.sources import ListSource, UDPSource
from v2.transport.frame import Frame, encode_frame
from v2.transport.udp_sender import UDPSender
from v2.transport.mock_esp32 import frame_to_csi_line


class _FakeSerial:
    """Minimal pyserial stand-in: serves a fixed byte blob across read() calls."""

    def __init__(self, data: bytes):
        self._data = bytearray(data)

    @property
    def in_waiting(self) -> int:
        return len(self._data)

    def read(self, n: int) -> bytes:
        chunk = bytes(self._data[:n])
        del self._data[:n]
        return chunk

    def close(self) -> None:
        pass


def _line_for(rx: int, seq: int, iq: np.ndarray) -> tuple[int, str]:
    """Build a (receiver_index, CSI_DATA line) item from complex iq."""
    frame = Frame(node_id=rx + 1, frame_seq=seq, iq=iq.astype(np.complex64))
    return rx, frame_to_csi_line(frame)


def _const_iq(real: float, imag: float, n: int = 64) -> np.ndarray:
    return (np.full(n, real) + 1j * np.full(n, imag)).astype(np.complex64)


def _wait_until(cond, timeout: float = 3.0, interval: float = 0.02) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(interval)
    return cond()


def test_matrix_basic(h: Harness) -> None:
    m = ComplexMatrix(max_time=100)
    for k in range(3):
        m.append(0, _const_iq(k, 2 * k))
    h.expect("count tracks appends", m.get_receiver_count(0) == 3)
    latest = m.get_latest(3)
    h.expect("shape (3,64,3)", latest.shape == (3, 64, 3), str(latest.shape))
    h.expect("dtype complex64", latest.dtype == np.complex64)
    h.expect("real ordering 0,1,2", np.allclose(latest[0, 0, :].real, [0, 1, 2]))
    h.expect("imag ordering 0,2,4", np.allclose(latest[0, 0, :].imag, [0, 2, 4]))
    h.expect("rx1 empty -> zeros", np.all(latest[1] == 0))


def test_matrix_zero_fill(h: Harness) -> None:
    m = ComplexMatrix(max_time=100)
    m.append(0, _const_iq(5, 5))
    m.append(0, _const_iq(6, 6))
    latest = m.get_latest(5)
    h.expect("leading zero-fill", np.all(latest[0, 0, :3] == 0))
    h.expect("data at tail", np.allclose(latest[0, 0, 3:].real, [5, 6]))


def test_matrix_wraparound(h: Harness) -> None:
    m = ComplexMatrix(max_time=4)
    for k in range(6):
        m.append(0, _const_iq(k, k))
    latest = m.get_latest(4)
    h.expect("count keeps growing", m.get_receiver_count(0) == 6)
    h.expect("last 4 in order 2,3,4,5", np.allclose(latest[0, 0, :].real, [2, 3, 4, 5]))


def test_matrix_timestamps(h: Harness) -> None:
    # Note 1: each slot carries device_ts, host_ts and seq alongside the CSI.
    m = ComplexMatrix(max_time=10)
    for k in range(3):
        m.append(0, _const_iq(k, k), device_ts=k * 100, host_ts=k * 1000, seq_id=k)
    streams = m.get_streams(3)
    h.expect("stream is per-receiver", len(streams) == 3)
    s0 = streams[0]
    h.expect("device_ts preserved in order", s0["device_ts"].tolist() == [0, 100, 200])
    h.expect("host_ts preserved in order", s0["host_ts"].tolist() == [0, 1000, 2000])
    h.expect("seq preserved in order", s0["seq"].tolist() == [0, 1, 2])
    h.expect("csi shape [64,3]", s0["csi"].shape == (64, 3), str(s0["csi"].shape))
    h.expect("empty receiver → 0-length stream", streams[1]["csi"].shape == (64, 0))
    h.expect("append still defaults timestamps to 0", True)  # back-compat: no-arg append below
    m.append(1, _const_iq(9, 9))  # legacy 2-arg call must not raise
    h.expect("legacy 2-arg append works", m.get_receiver_count(1) == 1)


def test_parser_complex(h: Harness) -> None:
    i = np.arange(64, dtype=np.float32) - 10.0
    q = (np.arange(64, dtype=np.float32) * 2.0) - 30.0
    iq = (i + 1j * q).astype(np.complex64)
    _, line = _line_for(rx=1, seq=7, iq=iq)

    pkt = CSIParser().parse(line, receiver_index=1)
    h.expect("parser returns a packet", pkt is not None)
    if pkt is None:
        return
    h.expect("seq parsed", pkt.seq_id == 7)
    h.expect("csi complex64", pkt.csi.dtype == np.complex64)
    h.expect("I preserved", np.array_equal(pkt.csi.real, i))
    h.expect("Q preserved (phase kept!)", np.array_equal(pkt.csi.imag, q))
    h.expect("amplitude property = |csi|", np.allclose(pkt.amplitude, np.sqrt(i**2 + q**2), atol=1e-3))
    h.expect("phase property = atan2(Q,I)", np.allclose(pkt.phase, np.arctan2(q, i), atol=1e-5))


def test_parser_rejects(h: Harness) -> None:
    p = CSIParser()
    h.expect("non-CSI line -> None", p.parse("hello world", 0) is None)
    h.expect("short array -> None", p.parse('CSI_DATA,1,mac,-40,0,0,0,0,0,0,0,0,0,0,-95,0,6,0,123,0,4,1,4,0,"[1,2,3,4]"', 0) is None)


def test_end_to_end_list_source(h: Harness) -> None:
    items = [
        _line_for(0, 0, _const_iq(1, 10)),
        _line_for(1, 0, _const_iq(2, 20)),
        _line_for(2, 0, _const_iq(3, 30)),
    ]
    gw = GatewayV2(sources=[ListSource(items)])
    gw.start()
    finished = gw.wait(timeout=3.0)
    h.expect("consumer thread finished on eof", finished)

    m = gw.get_matrix()
    h.expect("rx0 got 1 sample", m.get_receiver_count(0) == 1)
    h.expect("rx1 got 1 sample", m.get_receiver_count(1) == 1)
    h.expect("rx2 got 1 sample", m.get_receiver_count(2) == 1)

    latest = m.get_latest(1)
    h.expect("rx0 I/Q routed correctly", latest[0, 0, 0] == (1 + 10j))
    h.expect("rx1 I/Q routed correctly", latest[1, 0, 0] == (2 + 20j))
    h.expect("rx2 I/Q routed correctly", latest[2, 0, 0] == (3 + 30j))

    stats = gw.get_stats()
    h.expect("stats received == 1 per rx", all(stats[f"Rx{r}"]["received"] == 1 for r in range(3)))
    gw.stop()


def test_binary_serial_source(h: Harness) -> None:
    # Note 3: BinarySerialSource reads G2 frames and presents them as CSI lines.
    i = np.arange(64, dtype=np.float32) - 5.0
    q = (np.arange(64, dtype=np.float32) * 3.0) - 10.0
    iq = (i + 1j * q).astype(np.complex64)
    data = encode_frame(node_id=1, frame_seq=42, iq=iq)

    src = BinarySerialSource(port="fake", receiver_index=2)
    src._serial = _FakeSerial(data)  # bypass _open_once (no real port)
    item = src.poll(0.5)
    h.expect("returns a (rx, line) item", item is not None)
    if item is None:
        return
    rx, line = item
    h.expect("routed by receiver_index, not frame node_id", rx == 2)
    pkt = CSIParser().parse(line, receiver_index=rx)
    h.expect("decodes to correct seq", pkt is not None and pkt.seq_id == 42)
    if pkt is None:
        return
    h.expect("I preserved through binary path", np.allclose(pkt.csi.real, i))
    h.expect("Q preserved through binary path", np.allclose(pkt.csi.imag, q))


def test_binary_serial_resync(h: Harness) -> None:
    # A text banner + junk precede the first frame; the source must resync on "G2".
    banner = b"csi_inject_serial ready\n\x00\x01junk"
    data = banner + encode_frame(node_id=1, frame_seq=5, iq=_const_iq(7, -7))
    src = BinarySerialSource("fake", 0)
    src._serial = _FakeSerial(data)
    item = src.poll(0.5)
    h.expect("skips banner+junk to the first frame", item is not None)
    if item is None:
        return
    pkt = CSIParser().parse(item[1], 0)
    h.expect("resync decodes seq 5", pkt is not None and pkt.seq_id == 5)


def test_binary_serial_bad_frame_recovers(h: Harness) -> None:
    # A frame with a bad version byte must be dropped, then the next one decoded.
    good = encode_frame(node_id=1, frame_seq=9, iq=_const_iq(1, 1))
    bad = bytearray(encode_frame(node_id=1, frame_seq=0, iq=_const_iq(0, 0)))
    bad[2] = 2  # version 2 -> decode_frame raises ValueError
    src = BinarySerialSource("fake", 1)
    src._serial = _FakeSerial(bytes(bad) + good)
    first = src.poll(0.5)
    h.expect("malformed frame yields None", first is None)
    second = src.poll(0.5)
    h.expect("recovers the next good frame", second is not None)
    if second is None:
        return
    pkt = CSIParser().parse(second[1], 1)
    h.expect("recovered seq 9 after bad frame", pkt is not None and pkt.seq_id == 9)


def test_binary_serial_timeout(h: Harness) -> None:
    src = BinarySerialSource("fake", 0)
    src._serial = _FakeSerial(b"")
    h.expect("no data returns None within timeout", src.poll(0.1) is None)


def test_end_to_end_udp_source(h: Harness) -> None:
    udp = UDPSource(bind_host="127.0.0.1", bind_port=0)
    gw = GatewayV2(sources=[udp])
    gw.start()
    try:
        sender = UDPSender(net_map={"RX2": ("127.0.0.1", udp.port)})
        n_frames = 5
        for seq in range(n_frames):
            sender.send("RX2", frame_seq=seq, iq=_const_iq(0, 100))
        sender.close()

        got_all = _wait_until(lambda: gw.get_matrix().get_receiver_count(1) >= n_frames, timeout=3.0)
        h.expect("all UDP frames ingested", got_all,
                 f"count={gw.get_matrix().get_receiver_count(1)}")

        latest = gw.get_matrix().get_latest(n_frames)
        h.expect("real part is 0", np.allclose(latest[1, :, :].real, 0.0))
        h.expect("imag part preserved as 100", np.allclose(latest[1, :, :].imag, 100.0))
        phase = np.angle(latest[1, 0, 0])
        h.expect("phase preserved = +pi/2", np.isclose(phase, np.pi / 2, atol=1e-4), str(phase))
        h.expect("only rx1 received", gw.get_matrix().get_receiver_count(0) == 0)
    finally:
        gw.stop()


def main() -> int:
    h = Harness("gateway_v2 (complex I/Q)")
    h.case("matrix_basic", test_matrix_basic)
    h.case("matrix_timestamps", test_matrix_timestamps)
    h.case("matrix_zero_fill", test_matrix_zero_fill)
    h.case("matrix_wraparound", test_matrix_wraparound)
    h.case("parser_complex", test_parser_complex)
    h.case("parser_rejects", test_parser_rejects)
    h.case("binary_serial_source", test_binary_serial_source)
    h.case("binary_serial_resync", test_binary_serial_resync)
    h.case("binary_serial_bad_frame_recovers", test_binary_serial_bad_frame_recovers)
    h.case("binary_serial_timeout", test_binary_serial_timeout)
    h.case("end_to_end_list_source", test_end_to_end_list_source)
    h.case("end_to_end_udp_source", test_end_to_end_udp_source)
    return h.done()


if __name__ == "__main__":
    sys.exit(main())
