"""
v2 transport — wire frame codec.

Defines the binary UDP payload the injector sends to each ESP32 receiver.
The frame carries complex CSI as interleaved int16 I/Q (int16, not int8 — see
Plan section 16: int8 is too lossy for phase/breathing).

Wire layout (big-endian / network order):

    Offset  Size  Field       Notes
    ------  ----  ----------  -------------------------------------------
    0       2     magic       b"G2"
    2       1     version     currently 1
    3       1     node_id     1=RX1, 2=RX2, 3=RX3
    4       1     flags       bit0 = calibration frame
    5       1     reserved    0
    6       4     frame_seq   uint32, per-node monotonic sequence
    10      2     num_sub     uint16, subcarriers in payload (64)
    12      ...   iq          num_sub * 2 * int16, interleaved I0,Q0,I1,Q1,...

Total size = 12 + num_sub * 4 bytes (268 bytes for 64 subcarriers), well
within a single UDP datagram / Wi-Fi MTU.

This module depends only on numpy + struct so it can be shared by the injector,
the software mock, and (as a reference) the firmware format.
"""

import struct
from dataclasses import dataclass

import numpy as np

# --- constants ---
MAGIC = b"G2"
VERSION = 1

# magic(2s) version(B) node_id(B) flags(B) reserved(B) frame_seq(I) num_sub(H)
HEADER_FMT = ">2sBBBBIH"
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # 12

FLAG_CALIBRATION = 0x01

INT16_MIN = -32768
INT16_MAX = 32767

# Big-endian int16 dtype used for the I/Q payload (endianness-independent).
_IQ_DTYPE = np.dtype(">i2")


@dataclass
class Frame:
    """One decoded injection frame for a single receiver node."""

    node_id: int
    frame_seq: int
    iq: np.ndarray            # complex64, shape (num_sub,)
    calibration: bool = False

    @property
    def num_sub(self) -> int:
        return int(self.iq.shape[0])


def encode_frame(
    node_id: int,
    frame_seq: int,
    iq: np.ndarray,
    calibration: bool = False,
) -> bytes:
    """Serialize one frame to bytes.

    Args:
        node_id: 1/2/3 for RX1/RX2/RX3.
        frame_seq: per-node sequence number (wrapped to uint32).
        iq: complex array of shape (num_sub,). Real=I, Imag=Q. Values are
            rounded to nearest int and clipped to the int16 range.
        calibration: mark this frame as part of the empty-room preamble.

    Raises:
        TypeError: if iq is not complex.
    """
    iq = np.asarray(iq)
    if not np.iscomplexobj(iq):
        raise TypeError("iq must be a complex array (real=I, imag=Q)")
    if iq.ndim != 1:
        raise ValueError(f"iq must be 1-D, got {iq.ndim}-D")

    num_sub = int(iq.shape[0])
    flags = FLAG_CALIBRATION if calibration else 0

    header = struct.pack(
        HEADER_FMT,
        MAGIC,
        VERSION,
        node_id & 0xFF,
        flags,
        0,                      # reserved
        frame_seq & 0xFFFFFFFF,
        num_sub,
    )

    i = np.clip(np.rint(iq.real), INT16_MIN, INT16_MAX)
    q = np.clip(np.rint(iq.imag), INT16_MIN, INT16_MAX)

    interleaved = np.empty(num_sub * 2, dtype=_IQ_DTYPE)
    interleaved[0::2] = i
    interleaved[1::2] = q

    return header + interleaved.tobytes()


def decode_frame(data: bytes) -> Frame:
    """Deserialize bytes into a Frame.

    Raises:
        ValueError: on bad magic, unsupported version, or truncated payload.
    """
    if len(data) < HEADER_SIZE:
        raise ValueError(
            f"frame too short for header: {len(data)} < {HEADER_SIZE}"
        )

    magic, version, node_id, flags, _reserved, frame_seq, num_sub = struct.unpack(
        HEADER_FMT, data[:HEADER_SIZE]
    )

    if magic != MAGIC:
        raise ValueError(f"bad magic {magic!r} (expected {MAGIC!r})")
    if version != VERSION:
        raise ValueError(f"unsupported version {version} (expected {VERSION})")

    expected = HEADER_SIZE + num_sub * 2 * 2  # 2 int16 per subcarrier
    if len(data) < expected:
        raise ValueError(
            f"payload too short: {len(data)} < {expected} "
            f"(num_sub={num_sub})"
        )

    raw = np.frombuffer(data[HEADER_SIZE:expected], dtype=_IQ_DTYPE).astype(np.float32)
    i = raw[0::2]
    q = raw[1::2]
    iq = (i + 1j * q).astype(np.complex64)

    return Frame(
        node_id=int(node_id),
        frame_seq=int(frame_seq),
        iq=iq,
        calibration=bool(flags & FLAG_CALIBRATION),
    )


def frame_size(num_sub: int) -> int:
    """Total wire size in bytes for a frame with the given subcarrier count."""
    return HEADER_SIZE + num_sub * 4
