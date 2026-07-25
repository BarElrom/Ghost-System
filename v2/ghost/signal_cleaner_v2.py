"""
GHOST v2 — Signal cleaner: remove the static wall, keep the moving person.

Steps (all in the complex domain so phase is preserved):

  1. Calibrate an empty-room baseline  H_static = average(H) over N frames.
  2. Subtract it from every frame       H_dynamic = H_raw - H_static.
  3. Derive the human signal            amplitude = |H_dynamic|, phase = angle(H_dynamic).
  4. Band-pass the amplitude (0.1-4 Hz) to keep breathing/motion, drop drift/noise.

Static subtraction and the band-pass complement each other: subtraction removes
the wall coherently in I/Q, the band-pass cleans up what's left.

Input:  complex matrix [3 x 64 x T] from gateway_v2.
Output: CleanedCSI.
"""

import logging
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt

from v2.config_v2 import (
    BANDPASS_HIGHCUT,
    BANDPASS_LOWCUT,
    BUTTER_ORDER,
    CALIBRATION_MODE,
    CALIBRATION_SAMPLES,
    HAMPEL_THRESHOLD,
    HAMPEL_WINDOW,
    NUM_RECEIVERS,
    NUM_SUBCARRIERS,
    SAMPLE_RATE_HZ,
)

logger = logging.getLogger("ghost.v2.signal_cleaner")

# sosfiltfilt needs at least this many samples to pad its edges (order-4 band-pass).
_MIN_SAMPLES_FOR_BANDPASS = 25


@dataclass
class CleanedCSI:
    """The result of cleaning one window of CSI."""

    dynamic: np.ndarray      # complex64 [3 x 64 x T] — person signal (H_raw - H_static)
    amplitude: np.ndarray    # float32   [3 x 64 x T] — |dynamic|, band-passed
    phase: np.ndarray        # float32   [3 x 64 x T] — angle(dynamic)
    static: np.ndarray       # complex64 [3 x 64]     — baseline that was subtracted
    calibrated: bool         # True if an explicit calibration baseline was used


def _active_columns(matrix: np.ndarray) -> np.ndarray:
    """Boolean [R x T] marking time columns that hold real data (not zero-filled)."""
    return np.any(matrix != 0, axis=1)


def _column_mean(matrix: np.ndarray) -> np.ndarray:
    """Per-(receiver, subcarrier) mean over active columns, shape [R x 64] complex."""
    receiver, subcarrier, time = matrix.shape
    mean = np.zeros((receiver, subcarrier), dtype=np.complex64)
    active = _active_columns(matrix)
    for rx in range(receiver):
        if active[rx].any():
            mean[rx] = matrix[rx][:, active[rx]].mean(axis=1).astype(np.complex64)
    return mean


class SignalCleanerV2:
    """Calibrates a static baseline, then subtracts it and band-passes the motion."""

    def __init__(
        self,
        sample_rate_hz: float = float(SAMPLE_RATE_HZ),
        lowcut: float = BANDPASS_LOWCUT,
        highcut: float = BANDPASS_HIGHCUT,
        butter_order: int = BUTTER_ORDER,
        hampel_window: int = HAMPEL_WINDOW,
        hampel_threshold: float = HAMPEL_THRESHOLD,
        calibration_samples: int = CALIBRATION_SAMPLES,
        calibration_mode: str = CALIBRATION_MODE,
    ):
        self.sample_rate_hz = sample_rate_hz
        self.hampel_window = hampel_window
        self.hampel_threshold = hampel_threshold
        self.calibration_samples = calibration_samples
        self.calibration_mode = calibration_mode

        nyquist = sample_rate_hz / 2.0
        self._bandpass = butter(butter_order, [lowcut / nyquist, highcut / nyquist],
                                btype="band", output="sos")

        self._static: np.ndarray | None = None                     # [R x 64] baseline
        self._sum = np.zeros((NUM_RECEIVERS, NUM_SUBCARRIERS), dtype=np.complex128)
        self._count = np.zeros(NUM_RECEIVERS, dtype=np.int64)
        self._frames_seen = 0

    # ---- calibration --------------------------------------------------
    @property
    def calibrated(self) -> bool:
        """True once a static baseline has been established."""
        return self._static is not None

    @property
    def baseline(self) -> np.ndarray | None:
        """The static baseline [R x 64], or None before calibration."""
        return self._static

    def reset_calibration(self) -> None:
        """Discard the baseline and start calibration over."""
        self._static = None
        self._sum[:] = 0
        self._count[:] = 0
        self._frames_seen = 0

    def set_baseline(self, static: np.ndarray) -> None:
        """Set the baseline [R x 64] directly (e.g. a known empty-room average)."""
        static = np.asarray(static)
        if static.shape != (NUM_RECEIVERS, NUM_SUBCARRIERS):
            raise ValueError(f"baseline must be {(NUM_RECEIVERS, NUM_SUBCARRIERS)}, got {static.shape}")
        self._static = static.astype(np.complex64)

    def add_calibration(self, frames: np.ndarray) -> bool:
        """Accumulate empty-room frames; return True when the baseline is finalized."""
        frames = np.asarray(frames)
        if frames.ndim == 2:
            frames = frames[:, :, None]
        if frames.ndim != 3:
            raise ValueError(f"expected [R,64] or [R,64,T], got {frames.shape}")

        active = _active_columns(frames)
        for rx in range(NUM_RECEIVERS):
            if active[rx].any():
                self._sum[rx] += frames[rx][:, active[rx]].sum(axis=1)
                self._count[rx] += int(active[rx].sum())
        self._frames_seen += frames.shape[2]

        if not self.calibrated and self._frames_seen >= self.calibration_samples:
            self._static = np.divide(
                self._sum, np.maximum(self._count, 1)[:, None]
            ).astype(np.complex64)
            logger.info("Calibration complete over %d frames", self._frames_seen)
            return True
        return False

    def calibrate_from(self, matrix: np.ndarray) -> None:
        """One-shot calibration: baseline = temporal mean of an empty-room window."""
        self.reset_calibration()
        self._static = _column_mean(np.asarray(matrix))

    # ---- cleaning -----------------------------------------------------
    def clean(self, matrix: np.ndarray, apply_temporal_filter: bool = True) -> CleanedCSI:
        """Subtract the baseline, derive amplitude/phase, and band-pass the motion."""
        matrix = np.asarray(matrix)
        if matrix.ndim != 3:
            raise ValueError(f"expected [R,64,T] complex, got {matrix.shape}")
        if not np.iscomplexobj(matrix):
            raise TypeError("matrix must be complex (from gateway_v2)")

        baseline, calibrated = self._pick_baseline(matrix)

        # Subtract the wall, but keep zero-filled columns at zero (not -baseline).
        dynamic = (matrix - baseline[:, :, None]).astype(np.complex64)
        keep = _active_columns(matrix)[:, None, :]
        dynamic = np.where(keep, dynamic, 0).astype(np.complex64)

        amplitude = np.abs(dynamic).astype(np.float32)
        phase = np.angle(dynamic).astype(np.float32)
        if apply_temporal_filter:
            amplitude = self._bandpass_amplitude(amplitude)

        return CleanedCSI(dynamic, amplitude, phase, baseline.astype(np.complex64), calibrated)

    def _pick_baseline(self, matrix: np.ndarray) -> tuple[np.ndarray, bool]:
        """Choose the baseline: the calibrated one, the window mean, or zeros."""
        if self.calibrated:
            return self._static, True
        if self.calibration_mode == "temporal_mean":
            return _column_mean(matrix), False
        logger.warning("clean() before calibration (mode=%s); subtracting zero", self.calibration_mode)
        return np.zeros(matrix.shape[:2], dtype=np.complex64), False

    # ---- stage-2 temporal filter --------------------------------------
    def _bandpass_amplitude(self, amplitude: np.ndarray) -> np.ndarray:
        """Hampel-despike then band-pass each subcarrier's amplitude over time."""
        receiver, subcarrier, time = amplitude.shape
        out = np.empty_like(amplitude)
        can_bandpass = time >= _MIN_SAMPLES_FOR_BANDPASS
        if not can_bandpass:
            logger.warning("Only %d samples (<%d); Hampel only, band-pass skipped", time, _MIN_SAMPLES_FOR_BANDPASS)
        for rx in range(receiver):
            for sub in range(subcarrier):
                signal = amplitude[rx, sub].astype(np.float64)
                if not np.any(signal):
                    out[rx, sub] = 0.0
                    continue
                signal = self._hampel_despike(signal)
                if can_bandpass:
                    signal = sosfiltfilt(self._bandpass, signal)
                out[rx, sub] = signal
        return out

    def _hampel_despike(self, signal: np.ndarray) -> np.ndarray:
        """Replace outliers (> threshold*MAD from the local median) with the median."""
        k = self.hampel_window
        cleaned = signal.copy()
        for i in range(len(signal)):
            window = signal[max(0, i - k): i + k + 1]
            median = np.median(window)
            sigma = 1.4826 * np.median(np.abs(window - median))   # MAD -> std estimate
            if sigma > 0 and abs(signal[i] - median) > self.hampel_threshold * sigma:
                cleaned[i] = median
        return cleaned
