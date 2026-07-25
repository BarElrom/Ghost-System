"""
GHOST System — Feature Extractor ("The Translator")
Layer 2, Component 3: Convert cleaned CSI signals into compact motion
descriptors via PCA dimensionality reduction and STFT spectral analysis.

Input:  Cleaned Motion Matrix, shape [3 receivers x 64 subcarriers x T time]
        (typically the last 2 seconds = ~200 samples at 100 Hz)

Output: Current State Vector (Fingerprint), shape (6,)
        [breathing_frequency, total_energy, doppler_mean,
         variance_rx1, variance_rx2, variance_rx3]

Usage:
    from feature_extractor import FeatureExtractor
    extractor = FeatureExtractor()
    state = extractor.extract(cleaned_matrix)       # (6,) ndarray
    info  = extractor.extract_dict(cleaned_matrix)   # dict with named keys
"""

import logging

import numpy as np
from scipy.signal import stft as scipy_stft

logger = logging.getLogger("ghost.feature_extractor")

NUM_RECEIVERS = 3
NUM_FEATURES = 6

# Breathing band limits (Hz).  6–30 breaths/min → 0.1–0.5 Hz.
_BREATHING_LO = 0.1
_BREATHING_HI = 0.5

# Feature vector index names (for extract_dict).
FEATURE_NAMES = [
    "breathing_frequency",
    "total_energy",
    "doppler_mean",
    "variance_rx1",
    "variance_rx2",
    "variance_rx3",
]


class FeatureExtractor:
    """PCA + STFT feature extraction from cleaned CSI data.

    Args:
        fs: Sampling frequency in Hz.
        window_samples: Expected number of time samples in the input buffer
            (2 seconds * fs).  Used only for documentation / logging.
        stft_nperseg: STFT segment length in samples.
        stft_noverlap: STFT overlap in samples.
    """

    def __init__(
        self,
        fs: float = 100.0,
        window_samples: int = 200,
        stft_nperseg: int = 128,
        stft_noverlap: int = 96,
    ):
        self.fs = fs
        self.window_samples = window_samples
        self.stft_nperseg = stft_nperseg
        self.stft_noverlap = stft_noverlap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def extract(self, cleaned_matrix: np.ndarray) -> np.ndarray:
        """Extract the 6-element state vector.

        Args:
            cleaned_matrix: shape (R, 64, T).  R <= 3 receivers.

        Returns:
            np.ndarray of shape (6,), float64.
        """
        if cleaned_matrix.ndim != 3:
            raise ValueError(
                f"Expected 3-D array (receivers, subcarriers, time), "
                f"got {cleaned_matrix.ndim}-D"
            )

        n_rx, n_sub, n_time = cleaned_matrix.shape

        # Pad to 3 receivers if fewer are present.
        if n_rx < NUM_RECEIVERS:
            logger.warning(
                "Input has %d receivers, expected %d. "
                "Padding missing receivers with zeros.",
                n_rx, NUM_RECEIVERS,
            )
            pad = np.zeros(
                (NUM_RECEIVERS - n_rx, n_sub, n_time),
                dtype=cleaned_matrix.dtype,
            )
            cleaned_matrix = np.concatenate([cleaned_matrix, pad], axis=0)
            n_rx = NUM_RECEIVERS

        # ---- Step 1: PCA per receiver  (64 subcarriers → 1 PC) ----------
        pc1_signals = []  # list of (T,) arrays
        for rx in range(NUM_RECEIVERS):
            pc1 = self._pca_reduce(cleaned_matrix[rx])  # (64, T) → (T,)
            pc1_signals.append(pc1)

        # ---- Per-receiver variance (features 4-6) -----------------------
        variances = [float(np.var(pc)) for pc in pc1_signals]

        # ---- Step 2: Combine across receivers ---------------------------
        combined = np.mean(np.stack(pc1_signals, axis=0), axis=0)  # (T,)

        # ---- Step 3: STFT -----------------------------------------------
        freqs, power_spectrum = self._compute_spectrum(combined, n_time)

        # ---- Step 4: Extract scalar features from spectrum ---------------
        breathing_freq = self._breathing_frequency(freqs, power_spectrum)
        total_energy = self._total_energy(power_spectrum)
        doppler_mean = self._doppler_mean(freqs, power_spectrum)

        state = np.array(
            [breathing_freq, total_energy, doppler_mean,
             variances[0], variances[1], variances[2]],
            dtype=np.float64,
        )

        logger.info(
            "State vector: breath=%.3f Hz  energy=%.2f  doppler=%.3f  "
            "var=[%.4f, %.4f, %.4f]",
            *state,
        )
        return state

    def extract_dict(self, cleaned_matrix: np.ndarray) -> dict:
        """Same as extract() but returns a named dictionary."""
        vec = self.extract(cleaned_matrix)
        return dict(zip(FEATURE_NAMES, vec.tolist()))

    # ------------------------------------------------------------------
    # PCA — reduce 64 subcarriers to 1 principal component
    # ------------------------------------------------------------------
    @staticmethod
    def _pca_reduce(data: np.ndarray) -> np.ndarray:
        """Project (n_sub, T) onto the first principal component → (T,).

        Centers each subcarrier, computes the leading left singular vector,
        and projects the centered data onto it.
        """
        # data shape: (64, T)
        mean = data.mean(axis=1, keepdims=True)  # (64, 1)
        centered = data - mean                    # (64, T)

        # Economy SVD: U (64, k), S (k,), Vt (k, T)  where k = min(64, T)
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)

        # First principal component time-series = first row of Vt, scaled.
        # pc1[t] = S[0] * Vt[0, t]  preserves variance magnitude.
        pc1 = S[0] * Vt[0, :]  # (T,)
        return pc1

    # ------------------------------------------------------------------
    # Spectrum computation (STFT or FFT fallback)
    # ------------------------------------------------------------------
    def _compute_spectrum(
        self, signal: np.ndarray, n_time: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (frequencies, mean_power_spectrum).

        Uses STFT when the signal is long enough, otherwise falls back to
        a simple FFT.
        """
        if n_time >= self.stft_nperseg:
            freqs, _, Zxx = scipy_stft(
                signal,
                fs=self.fs,
                nperseg=self.stft_nperseg,
                noverlap=self.stft_noverlap,
            )
            # Mean power spectrum across STFT time frames.
            power = np.mean(np.abs(Zxx) ** 2, axis=1)  # (n_freq,)
        else:
            logger.warning(
                "Signal length %d < stft_nperseg %d. "
                "Falling back to FFT.",
                n_time, self.stft_nperseg,
            )
            spectrum = np.fft.rfft(signal)
            power = np.abs(spectrum) ** 2
            freqs = np.fft.rfftfreq(n_time, d=1.0 / self.fs)

        return freqs, power

    # ------------------------------------------------------------------
    # Scalar feature helpers
    # ------------------------------------------------------------------
    def _breathing_frequency(
        self, freqs: np.ndarray, power: np.ndarray
    ) -> float:
        """Peak frequency in the breathing band (0.1–0.5 Hz)."""
        mask = (freqs >= _BREATHING_LO) & (freqs <= _BREATHING_HI)
        if not np.any(mask) or np.all(power[mask] == 0.0):
            return 0.0
        band_power = power[mask]
        band_freqs = freqs[mask]
        return float(band_freqs[np.argmax(band_power)])

    @staticmethod
    def _total_energy(power: np.ndarray) -> float:
        """Sum of power across all frequency bins."""
        return float(np.sum(power))

    @staticmethod
    def _doppler_mean(freqs: np.ndarray, power: np.ndarray) -> float:
        """Spectral centroid — power-weighted mean frequency."""
        total = np.sum(power)
        if total == 0.0:
            return 0.0
        return float(np.sum(freqs * power) / total)