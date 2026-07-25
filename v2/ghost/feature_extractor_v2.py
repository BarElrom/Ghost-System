"""
GHOST System v2 — Feature Extractor (complex-aware)

Phase 1 of the v2 plan. Adapted from the root feature_extractor.py (untouched)
to consume the CleanedCSI produced by signal_cleaner_v2 and to add the pieces
the injection pipeline needs (Plan sections 3.4 and 8):

  Kept from v1 (spectral motion descriptors on the combined PC1 signal):
    - breathing_frequency : peak frequency in 0.1-0.5 Hz
    - total_energy        : sum of power across the spectrum
    - doppler_mean        : spectral centroid
    - variance_rx[]       : per-receiver PC1 variance (motion energy per node)

  New in v2:
    - node_energies       : per-receiver mean per-frame amplitude sum
                            (E_node for the localizer, Plan section 8)
    - phase_variance_rx[] : per-receiver phase motion (only possible now that
                            phase is preserved end to end)

Input:  CleanedCSI (amplitude/phase [R x 64 x T]) from signal_cleaner_v2.
Output: FeatureSet.
"""

import logging
from dataclasses import dataclass

import numpy as np

from v2.config_v2 import NODE_IDS, NUM_RECEIVERS, SAMPLE_RATE_HZ

logger = logging.getLogger("ghost.v2.feature_extractor")

# Breathing band (Hz): 6-30 breaths/min.
_BREATHING_LO = 0.1
_BREATHING_HI = 0.5

# Receiver index -> node name, ordered by node id (RX1, RX2, RX3).
_NODE_NAMES = sorted(NODE_IDS, key=NODE_IDS.get)


@dataclass
class FeatureSet:
    """Motion descriptors + per-node energies for one cleaned window."""

    breathing_frequency: float
    total_energy: float
    doppler_mean: float
    variance_rx: list          # length R — PC1 variance per receiver
    node_energies: dict        # {"RX1": float, ...} — for the localizer
    phase_variance_rx: list    # length R — phase motion per receiver

    def vector6(self) -> np.ndarray:
        """v1-compatible 6-vector [breath, energy, doppler, var0, var1, var2]."""
        v = list(self.variance_rx[:3]) + [0.0] * max(0, 3 - len(self.variance_rx))
        return np.array(
            [self.breathing_frequency, self.total_energy, self.doppler_mean,
             v[0], v[1], v[2]],
            dtype=np.float64,
        )

    def to_dict(self) -> dict:
        return {
            "breathing_frequency": self.breathing_frequency,
            "total_energy": self.total_energy,
            "doppler_mean": self.doppler_mean,
            "variance_rx": list(self.variance_rx),
            "node_energies": dict(self.node_energies),
            "phase_variance_rx": list(self.phase_variance_rx),
        }


class FeatureExtractorV2:
    """PCA + STFT feature extraction with per-node energy and phase features.

    The spectrum is the full-window rFFT of the combined PC1 signal, giving a
    frequency resolution of sample_rate_hz/T. Note this means breathing
    (0.1-0.5 Hz) can only be resolved when the window is long enough: at 100 Hz a
    2 s window (200 samples) has 0.5 Hz resolution and CANNOT resolve breathing —
    a ~10 s window (~1000 samples) is required. See Plan open items.

    Args:
        sample_rate_hz: CSI frame rate (Hz).
    """

    def __init__(self, sample_rate_hz: float = float(SAMPLE_RATE_HZ)):
        self.sample_rate_hz = sample_rate_hz

    # ------------------------------------------------------------------
    def extract(self, cleaned) -> FeatureSet:
        """Extract a FeatureSet from a CleanedCSI (or object with .amplitude/.phase)."""
        amplitude = np.asarray(cleaned.amplitude)
        phase = np.asarray(cleaned.phase)
        if amplitude.ndim != 3:
            raise ValueError(f"expected amplitude [R,64,T], got {amplitude.shape}")

        n_rx, n_sub, n_time = amplitude.shape

        # ---- per-receiver PCA (64 subcarriers -> 1 PC) ----
        pc1_signals = [self._pca_reduce(amplitude[rx]) for rx in range(n_rx)]
        variance_rx = [float(np.var(pc)) for pc in pc1_signals]

        # ---- combine PC1 across receivers ----
        combined = np.mean(np.stack(pc1_signals, axis=0), axis=0)  # (T,)

        # ---- spectrum of combined signal ----
        freqs, power = self._compute_spectrum(combined, n_time)
        breathing_frequency = self._breathing_frequency(freqs, power)
        total_energy = float(np.sum(power))
        doppler_mean = self._doppler_mean(freqs, power)

        # ---- per-node energy (E_node for localizer) ----
        node_energies = {}
        for rx in range(n_rx):
            per_frame_sum = amplitude[rx].sum(axis=0)     # (T,) sum over subcarriers
            name = _NODE_NAMES[rx] if rx < len(_NODE_NAMES) else f"RX{rx + 1}"
            node_energies[name] = float(np.mean(per_frame_sum))

        # ---- per-receiver phase motion ----
        phase_variance_rx = [
            self._phase_variance(amplitude[rx], phase[rx]) for rx in range(n_rx)
        ]

        features = FeatureSet(
            breathing_frequency=breathing_frequency,
            total_energy=total_energy,
            doppler_mean=doppler_mean,
            variance_rx=variance_rx,
            node_energies=node_energies,
            phase_variance_rx=phase_variance_rx,
        )
        logger.info(
            "Features: breath=%.3f energy=%.2f doppler=%.3f var=%s energies=%s",
            breathing_frequency, total_energy, doppler_mean,
            [round(v, 4) for v in variance_rx],
            {k: round(v, 2) for k, v in node_energies.items()},
        )
        return features

    # ------------------------------------------------------------------
    @staticmethod
    def _pca_reduce(data: np.ndarray) -> np.ndarray:
        """Project (n_sub, T) onto its first principal component -> (T,)."""
        mean = data.mean(axis=1, keepdims=True)
        centered = data - mean
        if not np.any(centered):
            return np.zeros(data.shape[1], dtype=np.float64)
        _U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        return S[0] * Vt[0, :]

    def _compute_spectrum(self, signal: np.ndarray, n_time: int):
        """Full-window power spectrum via rFFT (resolution = fs / n_time)."""
        if n_time < 2:
            return np.zeros(1), np.zeros(1)
        spectrum = np.fft.rfft(signal)
        power = np.abs(spectrum) ** 2
        freqs = np.fft.rfftfreq(n_time, d=1.0 / self.sample_rate_hz)
        return freqs, power

    @staticmethod
    def _breathing_frequency(freqs: np.ndarray, power: np.ndarray) -> float:
        """Peak frequency inside the 0.1-0.5 Hz breathing band (0 if none)."""
        mask = (freqs >= _BREATHING_LO) & (freqs <= _BREATHING_HI)
        if not np.any(mask) or np.all(power[mask] == 0.0):
            return 0.0
        band_power = power[mask]
        band_freqs = freqs[mask]
        return float(band_freqs[np.argmax(band_power)])

    @staticmethod
    def _doppler_mean(freqs: np.ndarray, power: np.ndarray) -> float:
        """Spectral centroid: the power-weighted average frequency."""
        total = np.sum(power)
        if total == 0.0:
            return 0.0
        return float(np.sum(freqs * power) / total)

    @staticmethod
    def _phase_variance(amp_rx: np.ndarray, phase_rx: np.ndarray) -> float:
        """Variance of the dominant subcarrier's unwrapped phase over time."""
        if phase_rx.shape[1] < 2:
            return 0.0
        mean_amp = amp_rx.mean(axis=1)          # (64,)
        if not np.any(mean_amp):
            return 0.0
        dominant = int(np.argmax(mean_amp))
        ph = np.unwrap(phase_rx[dominant, :].astype(np.float64))
        return float(np.var(ph))
