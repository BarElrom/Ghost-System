"""
GHOST v2 — FrameManager: align the N receiver streams onto one timeline.

The gateway ingests each receiver independently, so same-index samples across
receivers are NOT the same instant (USB/Wi-Fi arrival jitter, per-board clocks,
lost packets). The localizer, however, treats a column of the [N x 64 x T]
matrix as one simultaneous snapshot. This module closes that gap (Note 5):

  * join the receivers by *sequential position* (seq_id — the same TX frame seen
    by every RX) and/or *receive time* (a common host clock), and
  * emit them in *linear running time* — an ordered, gap-aware window whose
    columns really are cross-receiver simultaneous.

The join is a ``pandas.merge_asof`` (nearest-key within a tolerance), which is
purpose-built for "line these streams up by timestamp" work (Note 4: use
existing packages rather than hand-rolling).

Time keys:
  * ``seq``       — exact match on the per-frame sequence id. Physically correct
                    when all receivers observe the same transmitter (default).
  * ``host_ts``   — ``time.monotonic_ns()`` at host ingest; the only clock shared
                    across separate boards. Use when seq isn't shared/reliable.
  * ``device_ts`` — ESP ``esp_timer_get_time`` (µs); per-board, only comparable
                    across receivers in the software-mock/replay path.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from v2.config_v2 import NUM_RECEIVERS, NUM_SUBCARRIERS, SAMPLE_RATE_HZ

logger = logging.getLogger("ghost.v2.frame_manager")

_TIME_UNIT_SECONDS = {"seq": None, "device_ts": 1e-6, "host_ts": 1e-9}


@dataclass
class AlignedWindow:
    """One time-aligned window across all receivers.

    Attributes:
        matrix: [N x 64 x T] complex CSI; column t is simultaneous across rx.
        time_s: [T] monotonic time in seconds, zero-based at the first column.
        seq: [T] the reference sequence id per column (-1 if seq wasn't the key).
        coverage: per-receiver fraction of reference ticks that found a match.
        dropped: reference ticks discarded because some receiver had no match.
    """

    matrix: np.ndarray
    time_s: np.ndarray
    seq: np.ndarray
    coverage: dict
    dropped: int

    @property
    def num_frames(self) -> int:
        return int(self.matrix.shape[2])


class FrameManager:
    """Joins per-receiver CSI streams into cross-receiver-aligned windows.

    Args:
        num_receivers: expected receiver count (defaults to config NUM_RECEIVERS).
        sample_rate_hz: nominal frame rate, used to size the join tolerance and
            to derive time from ``seq``.
        align_on: "seq" | "host_ts" | "device_ts" (see module docstring).
        tolerance_ratio: match tolerance as a fraction of the frame period. Only
            used for the timestamp keys ("seq" is matched exactly).
    """

    def __init__(
        self,
        num_receivers: int = NUM_RECEIVERS,
        sample_rate_hz: float = float(SAMPLE_RATE_HZ),
        align_on: str = "seq",
        tolerance_ratio: float = 0.5,
    ):
        if align_on not in _TIME_UNIT_SECONDS:
            raise ValueError(
                f"align_on must be one of {sorted(_TIME_UNIT_SECONDS)}, got {align_on!r}"
            )
        self.num_receivers = int(num_receivers)
        self.sample_rate_hz = float(sample_rate_hz)
        self.align_on = align_on
        self.tolerance_ratio = float(tolerance_ratio)

    def align(self, streams: list[dict]) -> AlignedWindow:
        """Align the per-receiver streams (from ``ComplexMatrix.get_streams``).

        Returns an AlignedWindow whose ``matrix`` columns are cross-receiver
        simultaneous. Ticks where any receiver lacks a match are dropped.
        """
        if len(streams) != self.num_receivers:
            raise ValueError(
                f"expected {self.num_receivers} streams, got {len(streams)}"
            )

        counts = [int(s["seq"].shape[0]) for s in streams]
        if min(counts) == 0:
            logger.warning("align: a receiver has 0 samples %s → empty window", counts)
            return self._empty_window()

        # Reference = the receiver with the most samples (most complete timeline).
        ref_rx = int(np.argmax(counts))
        key = self.align_on

        # Reference frame carries the join key + the reference seq (for the output
        # seq axis) + its own row index into the csi buffer.
        ref = streams[ref_rx]
        merged = pd.DataFrame({
            key: ref[key].astype(np.int64),
            "seq": ref["seq"].astype(np.int64),
            f"row_{ref_rx}": np.arange(counts[ref_rx], dtype=np.int64),
        }).sort_values(key, kind="stable").reset_index(drop=True)

        # "seq" is an exact identity across receivers (same TX frame) → inner
        # merge, which naturally drops any tick a receiver lost. Timestamp keys
        # are approximate → nearest-match within tolerance via merge_asof, which
        # leaves NaN (later dropped) when no sample is close enough.
        tolerance = None if key == "seq" else self._tolerance_ticks()
        for rx in range(self.num_receivers):
            if rx == ref_rx:
                continue
            right = pd.DataFrame({
                key: streams[rx][key].astype(np.int64),
                f"row_{rx}": np.arange(counts[rx], dtype=np.int64),
            }).sort_values(key, kind="stable").reset_index(drop=True)
            if key == "seq":
                merged = merged.merge(right, on=key, how="inner")
            else:
                merged = pd.merge_asof(merged, right, on=key,
                                       direction="nearest", tolerance=tolerance)

        row_cols = [f"row_{rx}" for rx in range(self.num_receivers)]
        complete = merged.dropna(subset=row_cols)
        dropped = int(counts[ref_rx] - len(complete))
        if complete.empty:
            logger.warning("align: no tick had all %d receivers (key=%s, tol=%s)",
                           self.num_receivers, key, tolerance)
            return self._empty_window()

        matrix = self._gather(streams, complete, row_cols)
        time_s = self._time_seconds(complete[key].to_numpy())
        seq = (complete["seq"].to_numpy().astype(np.int64)
               if "seq" in complete else np.full(len(complete), -1, dtype=np.int64))
        coverage = {rx: float(complete[f"row_{rx}"].notna().mean())
                    for rx in range(self.num_receivers)}

        logger.info(
            "align: %d aligned ticks (dropped %d incomplete) key=%s ref=rx%d span=%.3fs",
            len(complete), dropped, key, ref_rx, float(time_s[-1] - time_s[0]) if len(time_s) else 0.0,
        )
        return AlignedWindow(matrix, time_s, seq, coverage, dropped)

    def _gather(self, streams: list[dict], complete: pd.DataFrame,
                row_cols: list[str]) -> np.ndarray:
        """Assemble the [N x 64 x T] matrix from the matched per-receiver rows."""
        t = len(complete)
        matrix = np.zeros((self.num_receivers, NUM_SUBCARRIERS, t), dtype=np.complex64)
        for rx in range(self.num_receivers):
            rows = complete[row_cols[rx]].to_numpy().astype(np.int64)
            matrix[rx] = streams[rx]["csi"][:, rows]
        return matrix

    def _tolerance_ticks(self) -> int:
        """Join tolerance in the key's integer units (µs for device, ns for host)."""
        period_s = 1.0 / self.sample_rate_hz
        unit = _TIME_UNIT_SECONDS[self.align_on]  # seconds per integer tick
        return int(self.tolerance_ratio * period_s / unit)

    def _time_seconds(self, key_values: np.ndarray) -> np.ndarray:
        """Convert the reference key column to zero-based seconds."""
        key_values = key_values.astype(np.float64)
        if self.align_on == "seq":
            base = key_values - key_values[0]
            return (base / self.sample_rate_hz).astype(np.float64)
        scale = _TIME_UNIT_SECONDS[self.align_on]
        return ((key_values - key_values[0]) * scale).astype(np.float64)

    def _empty_window(self) -> AlignedWindow:
        return AlignedWindow(
            matrix=np.zeros((self.num_receivers, NUM_SUBCARRIERS, 0), dtype=np.complex64),
            time_s=np.zeros(0, dtype=np.float64),
            seq=np.zeros(0, dtype=np.int64),
            coverage={rx: 0.0 for rx in range(self.num_receivers)},
            dropped=0,
        )
