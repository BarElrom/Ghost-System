
import logging

import numpy as np

from v2.config_v2 import NUM_SUBCARRIERS
from v2.injector.adapters.base import DatasetAdapter, Snapshot, normalize_subcarriers

logger = logging.getLogger("ghost.v2.adapter.embedded_wifi")


class EmbeddedWiFiAdapter(DatasetAdapter):
    """Parse an ESP32 CSI CSV file into single-link Snapshots.

    Args:
        path: CSV file path.
        iq_order: "iq" (real first) or "qi" (imag first). See module note.
        max_frames: optional cap on number of parsed frames.
    """

    num_streams = 1
    name = "embedded_wifi"

    def __init__(self, path: str, iq_order: str = "iq", max_frames: int | None = None):
        if iq_order not in ("iq", "qi"):
            raise ValueError("iq_order must be 'iq' or 'qi'")
        self.path = path
        self.iq_order = iq_order
        self.max_frames = max_frames

    def snapshots(self):
        count = 0
        with open(self.path, "r") as f:
            for line in f:
                iq = self._parse_line(line)
                if iq is None:
                    continue
                yield Snapshot(iq_by_stream={0: iq})
                count += 1
                if self.max_frames is not None and count >= self.max_frames:
                    break
        logger.info("%s: parsed %d frames from %s", self.name, count, self.path)

    def _parse_line(self, line: str) -> np.ndarray | None:
        bs = line.find("[")
        be = line.rfind("]")
        if bs == -1 or be == -1 or be <= bs:
            return None
        inner = line[bs + 1:be].strip()
        if not inner:
            return None
        parts = inner.split(",") if "," in inner else inner.split()
        try:
            vals = np.array([float(p) for p in parts if p.strip() != ""], dtype=np.float32)
        except ValueError:
            return None
        n_pairs = vals.shape[0] // 2
        if n_pairs == 0:
            return None
        vals = vals[: 2 * n_pairs]
        a = vals[0::2]
        b = vals[1::2]
        if self.iq_order == "iq":
            i_vals, q_vals = a, b
        else:
            i_vals, q_vals = b, a
        iq = (i_vals + 1j * q_vals).astype(np.complex64)
        return normalize_subcarriers(iq, NUM_SUBCARRIERS)
