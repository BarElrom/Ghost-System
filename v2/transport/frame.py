

import struct
from dataclasses import dataclass

import numpy as np

from v2.config_v2 import (
    DEFAULT_NOISE_FLOOR,
    DEFAULT_RSSI,
    SAMPLE_RATE_HZ,
    TX_MAC,
    WIFI_CHANNEL,
)

MAGIC = b"G2"
VERSION = 1

HEADER_FMT = ">2sBBBBIH"
HEADER_SIZE = struct.calcsize(HEADER_FMT)

FLAG_CALIBRATION = 0x01

INT16_MIN = -32768
INT16_MAX = 32767

_IQ_DTYPE = np.dtype(">i2")


@dataclass
class Frame:
    """One decoded injection frame for a single receiver node."""

    node_id: int
    frame_seq: int
    iq: np.ndarray
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
        0,
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

    expected = HEADER_SIZE + num_sub * 2 * 2
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


_US_PER_FRAME = 1_000_000 // SAMPLE_RATE_HZ


def frame_to_csi_line(frame: Frame) -> str:
    """Format a decoded Frame as a standard ESP32 CSI_DATA CSV line.

    This is the inverse of ``CSIParser.parse`` — the exact line a real ESP32
    would emit for this frame — so both the software mock and the binary-serial
    hardware source can present decoded frames to the gateway unchanged.

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
        "CSI_DATA",
        str(frame.frame_seq),
        TX_MAC,
        str(DEFAULT_RSSI),
        "11", "1", "7", "1", "0", "1", "0", "0", "0", "0",
        str(DEFAULT_NOISE_FLOOR),
        "0",
        str(WIFI_CHANNEL),
        "0",
        str(timestamp),
        "0",
        str(n_vals),
        "1",
        str(n_vals),
        "0",
    ]
    return ",".join(fields) + ',"' + csi_array + '"'
