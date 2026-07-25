"""Unit tests for the v2 wire frame codec (v2/transport/frame.py).

Run:  python v2/tests/test_frame.py
"""

import sys

from _harness import Harness  # noqa: E402  (path bootstrap happens in _harness)

import numpy as np

from v2.transport import frame as F


def test_header_size(h: Harness) -> None:
    h.expect("header size is 12 bytes", F.HEADER_SIZE == 12, str(F.HEADER_SIZE))
    h.expect("frame_size(64) == 268", F.frame_size(64) == 268, str(F.frame_size(64)))


def test_roundtrip(h: Harness) -> None:
    i = np.arange(64, dtype=np.float32) - 20.0
    q = (np.arange(64, dtype=np.float32) * 3.0) - 90.0
    iq = (i + 1j * q).astype(np.complex64)

    data = F.encode_frame(node_id=2, frame_seq=42, iq=iq, calibration=False)
    h.expect("encoded length matches frame_size", len(data) == F.frame_size(64), str(len(data)))

    fr = F.decode_frame(data)
    h.expect("node_id preserved", fr.node_id == 2)
    h.expect("frame_seq preserved", fr.frame_seq == 42)
    h.expect("num_sub preserved", fr.num_sub == 64)
    h.expect("calibration flag false", fr.calibration is False)
    h.expect("I values preserved", np.array_equal(fr.iq.real, i))
    h.expect("Q values preserved", np.array_equal(fr.iq.imag, q))
    h.expect("decoded dtype complex64", fr.iq.dtype == np.complex64)


def test_calibration_flag(h: Harness) -> None:
    iq = np.zeros(64, dtype=np.complex64)
    data = F.encode_frame(node_id=1, frame_seq=0, iq=iq, calibration=True)
    fr = F.decode_frame(data)
    h.expect("calibration flag true", fr.calibration is True)


def test_int16_clipping(h: Harness) -> None:
    iq = np.array([40000 + 0j, -40000 + 0j], dtype=np.complex64)
    fr = F.decode_frame(F.encode_frame(3, 1, iq))
    h.expect("positive clip to 32767", fr.iq.real[0] == 32767, str(fr.iq.real[0]))
    h.expect("negative clip to -32768", fr.iq.real[1] == -32768, str(fr.iq.real[1]))


def test_rounding(h: Harness) -> None:
    iq = np.array([1.4 + 1.6j], dtype=np.complex64)
    fr = F.decode_frame(F.encode_frame(1, 1, iq))
    h.expect("real rounds 1.4 -> 1", fr.iq.real[0] == 1.0, str(fr.iq.real[0]))
    h.expect("imag rounds 1.6 -> 2", fr.iq.imag[0] == 2.0, str(fr.iq.imag[0]))


def test_encode_requires_complex(h: Harness) -> None:
    h.expect_raises(
        "encode rejects real array",
        TypeError,
        lambda: F.encode_frame(1, 0, np.arange(64, dtype=np.float32)),
    )


def test_decode_bad_magic(h: Harness) -> None:
    good = F.encode_frame(1, 0, np.zeros(64, dtype=np.complex64))
    bad = b"XX" + good[2:]
    h.expect_raises("decode rejects bad magic", ValueError, lambda: F.decode_frame(bad))


def test_decode_truncated(h: Harness) -> None:
    good = F.encode_frame(1, 0, np.zeros(64, dtype=np.complex64))
    h.expect_raises(
        "decode rejects truncated payload",
        ValueError,
        lambda: F.decode_frame(good[:-4]),
    )
    h.expect_raises(
        "decode rejects sub-header input",
        ValueError,
        lambda: F.decode_frame(b"G2"),
    )


def main() -> int:
    h = Harness("frame codec")
    h.case("header_size", test_header_size)
    h.case("roundtrip", test_roundtrip)
    h.case("calibration_flag", test_calibration_flag)
    h.case("int16_clipping", test_int16_clipping)
    h.case("rounding", test_rounding)
    h.case("encode_requires_complex", test_encode_requires_complex)
    h.case("decode_bad_magic", test_decode_bad_magic)
    h.case("decode_truncated", test_decode_truncated)
    return h.done()


if __name__ == "__main__":
    sys.exit(main())
