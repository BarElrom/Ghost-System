"""Integration test for the v2 transport contract (Plan section 2).

Proves the full injection path in software, end to end:

    UDPSender  --UDP-->  MockESP32  -->  CSI_DATA line  -->  root CSIParser

The final stage feeds the mock's output through the EXISTING root
gateway.CSIParser (unmodified), proving the emitted line is byte-compatible
with what the ghost system already consumes.

Run:  python v2/tests/test_transport_roundtrip.py
"""

import sys

from _harness import Harness  # noqa: E402  (path bootstrap happens in _harness)

import numpy as np

from v2.transport.udp_sender import UDPSender
from v2.transport.mock_esp32 import MockESP32, frame_to_csi_line

# Existing root parser — imported read-only to prove format compatibility.
from gateway import CSIParser


def _make_iq() -> np.ndarray:
    i = np.arange(64, dtype=np.float32) - 20.0
    q = (np.arange(64, dtype=np.float32) * 3.0) - 90.0
    return (i + 1j * q).astype(np.complex64)


def test_udp_roundtrip_and_parse(h: Harness) -> None:
    mock = MockESP32(bind_host="127.0.0.1", bind_port=0)  # ephemeral port
    try:
        # Point the sender at the mock's actual bound port.
        net_map = {"RX2": ("127.0.0.1", mock.port)}
        sender = UDPSender(net_map=net_map)

        iq = _make_iq()
        sender.send("RX2", frame_seq=42, iq=iq)

        got = mock.receive(timeout=2.0)
        h.expect("frame received (not timeout)", got is not None)
        if got is None:
            return
        node_name, frame, _addr = got

        h.expect("node routed to RX2", node_name == "RX2", node_name)
        h.expect("seq preserved over UDP", frame.frame_seq == 42)
        h.expect("I preserved over UDP", np.array_equal(frame.iq.real, iq.real))
        h.expect("Q preserved over UDP", np.array_equal(frame.iq.imag, iq.imag))

        # --- format as a CSI_DATA line and parse with the ROOT parser ---
        line = frame_to_csi_line(frame)
        h.expect("line starts with CSI_DATA", line.startswith("CSI_DATA,"))

        parser = CSIParser()
        pkt = parser.parse(line, receiver_index=1)
        h.expect("root CSIParser accepts the line", pkt is not None)
        if pkt is None:
            return

        h.expect("parsed seq_id matches", pkt.seq_id == 42, str(pkt.seq_id))
        h.expect("parsed receiver_index respected", pkt.receiver_index == 1)

        # raw_iq from the root parser must equal our injected I/Q (int16).
        exp_i = iq.real.astype(np.int16)
        exp_q = iq.imag.astype(np.int16)
        h.expect("parsed I matches injected", np.array_equal(pkt.raw_iq[:, 0], exp_i))
        h.expect("parsed Q matches injected", np.array_equal(pkt.raw_iq[:, 1], exp_q))

        # amplitude the root parser computes must equal sqrt(I^2+Q^2).
        exp_amp = np.sqrt(iq.real ** 2 + iq.imag ** 2)
        h.expect(
            "parsed amplitude matches sqrt(I^2+Q^2)",
            np.allclose(pkt.amplitude, exp_amp, atol=1e-3),
        )
        sender.close()
    finally:
        mock.close()


def test_calibration_frame_formats(h: Harness) -> None:
    mock = MockESP32(bind_host="127.0.0.1", bind_port=0)
    try:
        sender = UDPSender(net_map={"RX1": ("127.0.0.1", mock.port)})
        sender.send("RX1", frame_seq=0, iq=np.zeros(64, dtype=np.complex64), calibration=True)
        got = mock.receive(timeout=2.0)
        h.expect("calibration frame received", got is not None)
        if got is None:
            return
        node_name, frame, _ = got
        h.expect("calibration flag survives transport", frame.calibration is True)
        h.expect("calibration routed to RX1", node_name == "RX1")
        # Still produces a parseable CSI_DATA line.
        pkt = CSIParser().parse(frame_to_csi_line(frame), receiver_index=0)
        h.expect("calibration line parses", pkt is not None)
        sender.close()
    finally:
        mock.close()


def test_unknown_node_rejected(h: Harness) -> None:
    sender = UDPSender(net_map={"RX1": ("127.0.0.1", 5005)})
    h.expect_raises(
        "sending to unknown node raises KeyError",
        KeyError,
        lambda: sender.send("RX9", 0, np.zeros(64, dtype=np.complex64)),
    )
    sender.close()


def main() -> int:
    h = Harness("transport roundtrip")
    h.case("udp_roundtrip_and_parse", test_udp_roundtrip_and_parse)
    h.case("calibration_frame_formats", test_calibration_frame_formats)
    h.case("unknown_node_rejected", test_unknown_node_rejected)
    return h.done()


if __name__ == "__main__":
    sys.exit(main())
