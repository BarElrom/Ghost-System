"""
GHOST System — Signal Cleaner / DSP Preprocessing
Layer 2, Component 2: Remove hardware artifacts, static reflections, and
irrelevant noise from the raw amplitude matrix.

Preserves the human-motion frequency band (0.1–4.0 Hz) which captures
breathing, gestures, and body movement.

Input:  Raw Amplitude Matrix, shape [3 receivers x 64 subcarriers x time]
Output: Cleaned Motion Matrix, same shape

Usage:
    from signal_cleaner import SignalCleaner
    cleaner = SignalCleaner()
    cleaned = cleaner.clean(raw_matrix)  # (3, 64, T) -> (3, 64, T)
"""

import logging

import numpy as np
from scipy.signal import butter, sosfiltfilt

logger = logging.getLogger("ghost.signal_cleaner")

# Minimum number of time samples required for sosfiltfilt.
# sosfiltfilt default padlen = 3 * max(len(sos sections)) * 2,
# for order 4 bandpass (4 SOS sections) that's 3 * 4 * 2 = 24.
_MIN_SAMPLES_FOR_BANDPASS = 25


class SignalCleaner:
    """DSP preprocessing: Hampel outlier removal + Butterworth bandpass.

    Args:
        fs: Sampling frequency in Hz (ESP32 packet rate, ~100 Hz).
        lowcut: Bandpass lower cutoff in Hz.
        highcut: Bandpass upper cutoff in Hz.
        butter_order: Butterworth filter order.
        hampel_window: Half-window size for the Hampel filter.
            Total window = 2 * hampel_window + 1.
        hampel_threshold: Number of standard deviations for outlier detection.
    """

    def __init__(
        self,
        fs: float = 100.0,
        lowcut: float = 0.1,
        highcut: float = 4.0,
        butter_order: int = 4,
        hampel_window: int = 3,
        hampel_threshold: float = 3.0,
    ):
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.butter_order = butter_order
        self.hampel_window = hampel_window
        self.hampel_threshold = hampel_threshold

        # Pre-compute Butterworth SOS coefficients (reused for every signal).
        nyq = fs / 2.0
        low = lowcut / nyq
        high = highcut / nyq
        self._sos = butter(butter_order, [low, high], btype="band", output="sos")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def clean(self, matrix: np.ndarray) -> np.ndarray:
        """Clean the raw amplitude matrix.

        Args:
            matrix: shape (R, 64, T) where R is number of receivers.

        Returns:
            Cleaned matrix, same shape and dtype as input.
        """
        if matrix.ndim != 3:
            raise ValueError(f"Expected 3-D array (receivers, subcarriers, time), got {matrix.ndim}-D")

        n_receivers, n_subcarriers, n_time = matrix.shape
        out = np.empty_like(matrix)

        can_bandpass = n_time >= _MIN_SAMPLES_FOR_BANDPASS
        if not can_bandpass:
            logger.warning(
                "Only %d time samples (need >= %d for bandpass). "
                "Skipping Butterworth filter, applying Hampel only.",
                n_time, _MIN_SAMPLES_FOR_BANDPASS,
            )

        for rx in range(n_receivers):
            for sub in range(n_subcarriers):
                signal = matrix[rx, sub, :].astype(np.float64)

                # Skip all-zero signals (empty buffer slots).
                if np.all(signal == 0.0):
                    out[rx, sub, :] = 0.0
                    continue

                signal = self._hampel_filter(signal)

                if can_bandpass:
                    signal = self._bandpass_filter(signal)

                out[rx, sub, :] = signal

        logger.info(
            "Cleaned matrix: %d receivers x %d subcarriers x %d samples",
            n_receivers, n_subcarriers, n_time,
        )
        return out

    # ------------------------------------------------------------------
    # Hampel filter
    # ------------------------------------------------------------------
    def _hampel_filter(self, signal: np.ndarray) -> np.ndarray:
        """Replace outliers with the local median.

        For each sample, a window of 2*hampel_window+1 neighbours is examined.
        If the sample deviates from the local median by more than
        hampel_threshold * 1.4826 * MAD, it is replaced with the median.

        The constant 1.4826 converts MAD to an estimate of σ for Gaussian data.
        """
        n = len(signal)
        k = self.hampel_window
        threshold = self.hampel_threshold
        filtered = signal.copy()

        for i in range(n):
            lo = max(0, i - k)
            hi = min(n, i + k + 1)
            window = signal[lo:hi]

            median = np.median(window)
            mad = np.median(np.abs(window - median))
            sigma = 1.4826 * mad

            if sigma > 0.0 and np.abs(signal[i] - median) > threshold * sigma:
                filtered[i] = median

        return filtered

    # ------------------------------------------------------------------
    # Butterworth bandpass
    # ------------------------------------------------------------------
    def _bandpass_filter(self, signal: np.ndarray) -> np.ndarray:
        """Zero-phase Butterworth bandpass filter."""
        return sosfiltfilt(self._sos, signal)