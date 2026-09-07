
import logging
from dataclasses import dataclass

import numpy as np

from v2.config_v2 import NODE_IDS, NUM_RECEIVERS, SAMPLE_RATE_HZ

logger = logging.getLogger("ghost.v2.feature_extractor")

_BREATHING_LO = 0.1
_BREATHING_HI = 0.5
# A real respiration peak must stand this many times above the in-band power
# median to count; otherwise the band is just 1/f drift and we report 0.0 Hz.
_BREATHING_PROMINENCE = 3.0

_NODE_NAMES = sorted(NODE_IDS, key=NODE_IDS.get)


@dataclass
class FeatureSet:
    """Motion descriptors + per-node energies for one cleaned window."""

    breathing_frequency: float
    total_energy: float
    doppler_mean: float
    variance_rx: list
    node_energies: dict
    phase_variance_rx: list

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
    (0.1-0.5 Hz) can only be resolved when the window is long enough: at 50 Hz a
    2 s window (100 samples) has 0.5 Hz resolution and CANNOT resolve breathing —
    a ~10 s window (~500 samples) is required. See Plan open items.

    Args:
        sample_rate_hz: CSI frame rate (Hz).
    """

    def __init__(self, sample_rate_hz: float = float(SAMPLE_RATE_HZ)):
        self.sample_rate_hz = sample_rate_hz

    def extract(self, cleaned) -> FeatureSet:
        """Extract a FeatureSet from a CleanedCSI (or object with .amplitude/.phase)."""
        amplitude = np.asarray(cleaned.amplitude)
        phase = np.asarray(cleaned.phase)
        if amplitude.ndim != 3:
            raise ValueError(f"expected amplitude [R,64,T], got {amplitude.shape}")

        n_rx, n_sub, n_time = amplitude.shape

        pc1_signals = [self._pca_reduce(amplitude[rx]) for rx in range(n_rx)]
        variance_rx = [float(np.var(pc)) for pc in pc1_signals]

        combined = np.mean(np.stack(pc1_signals, axis=0), axis=0)

        freqs, power = self._compute_spectrum(combined, n_time)
        breathing_frequency = self._breathing_frequency(combined, n_time)
        total_energy = float(np.sum(power))
        doppler_mean = self._doppler_mean(freqs, power)

        dynamic_mag = np.abs(np.asarray(cleaned.dynamic))
        node_energies = {}
        for rx in range(n_rx):
            per_frame_sum = dynamic_mag[rx].sum(axis=0)
            name = _NODE_NAMES[rx] if rx < len(_NODE_NAMES) else f"RX{rx + 1}"
            node_energies[name] = float(np.mean(per_frame_sum))
            logger.debug("  node_energy %s: mean_t Σ_k|H_dyn| = %.3f (per-frame Σ range [%.1f,%.1f])",
                         name, node_energies[name], float(per_frame_sum.min()),
                         float(per_frame_sum.max()))

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

    def breathing_spectrum(self, cleaned) -> tuple:
        """(freqs, power, peak_hz) for the breathing analysis of a cleaned window.

        Returns the exact detrended + Hann-windowed power spectrum that
        ``_breathing_frequency`` decides on, plus the detected peak (0.0 if the
        band is peakless drift). Intended for plots/diagnostics — the pipeline
        itself only needs the scalar ``breathing_frequency``.
        """
        amplitude = np.asarray(cleaned.amplitude)
        n_rx, _, n_time = amplitude.shape
        pcs = [self._pca_reduce(amplitude[rx]) for rx in range(n_rx)]
        combined = np.mean(np.stack(pcs, axis=0), axis=0)
        if n_time < 4:
            return np.zeros(1), np.zeros(1), 0.0
        t = np.arange(n_time)
        detrended = combined - np.polyval(np.polyfit(t, combined, 1), t)
        power = np.abs(np.fft.rfft(detrended * np.hanning(n_time))) ** 2
        freqs = np.fft.rfftfreq(n_time, d=1.0 / self.sample_rate_hz)
        return freqs, power, self._breathing_frequency(combined, n_time)

    @staticmethod
    def _pca_reduce(data: np.ndarray) -> np.ndarray:
        """Project (n_sub, T) onto its first principal component -> (T,)."""
        mean = data.mean(axis=1, keepdims=True)
        centered = data - mean
        if not np.any(centered):
            return np.zeros(data.shape[1], dtype=np.float64)
        _U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        if logger.isEnabledFor(logging.DEBUG):
            evr = float(S[0] ** 2 / np.sum(S ** 2)) if S.size else 0.0
            logger.debug("  PCA: singular values S[:3]=%s → PC1 explains %.1f%% of variance",
                         np.round(S[:3], 3).tolist(), 100.0 * evr)
        return S[0] * Vt[0, :]

    def _compute_spectrum(self, signal: np.ndarray, n_time: int):
        """Full-window power spectrum via rFFT (resolution = fs / n_time)."""
        if n_time < 2:
            return np.zeros(1), np.zeros(1)
        spectrum = np.fft.rfft(signal)
        power = np.abs(spectrum) ** 2
        freqs = np.fft.rfftfreq(n_time, d=1.0 / self.sample_rate_hz)
        if logger.isEnabledFor(logging.DEBUG):
            res = self.sample_rate_hz / n_time
            pk = int(np.argmax(power[1:])) + 1 if power.size > 1 else 0
            logger.debug(
                "  spectrum: rFFT over T=%d → resolution fs/T=%.4f Hz; peak @ %.4f Hz "
                "(power=%.3g)%s",
                n_time, res, float(freqs[pk]), float(power[pk]),
                "  ⚠res>0.1Hz: cannot resolve breathing" if res > _BREATHING_LO else "",
            )
        return freqs, power

    def _breathing_frequency(self, signal: np.ndarray, n_time: int) -> float:
        """Dominant respiration frequency in the 0.1-0.5 Hz band, or 0.0 if none.

        Raw CSI carries a strong 1/f drift whose tail leaks past the cleaner's
        0.1 Hz high-pass and piles up at the band's low edge — a plain argmax
        then always returns 0.1 Hz, a phantom "breathing" reading that appears
        even for an empty room. To report breathing only when it is really there
        we (1) remove the linear trend and apply a Hann window to suppress that
        leakage, then (2) take the strongest *interior local maximum* in the band
        and require it to stand a factor above the in-band median. A monotonic
        (peakless) spectrum has no qualifying local max, so we honestly return
        0.0 rather than the low-edge artifact.
        """
        if n_time < 4:
            return 0.0
        t = np.arange(n_time)
        detrended = signal - np.polyval(np.polyfit(t, signal, 1), t)
        windowed = detrended * np.hanning(n_time)
        power = np.abs(np.fft.rfft(windowed)) ** 2
        freqs = np.fft.rfftfreq(n_time, d=1.0 / self.sample_rate_hz)

        band = np.where((freqs >= _BREATHING_LO) & (freqs <= _BREATHING_HI))[0]
        if band.size == 0 or not np.any(power[band]):
            logger.debug("  breathing: no bins / no power in %.1f-%.1f Hz band → 0.0 Hz",
                         _BREATHING_LO, _BREATHING_HI)
            return 0.0
        # Interior local maxima only — this excludes the low-edge drift pile-up
        # that a monotonic 1/f tail would otherwise win.
        peaks = [i for i in band
                 if 0 < i < power.size - 1
                 and power[i] > power[i - 1] and power[i] >= power[i + 1]]
        if not peaks:
            logger.debug("  breathing: no local peak in %.1f-%.1f Hz (monotonic drift) → 0.0 Hz",
                         _BREATHING_LO, _BREATHING_HI)
            return 0.0
        best = max(peaks, key=lambda i: power[i])
        floor = float(np.median(power[band]))
        if floor > 0.0 and power[best] < _BREATHING_PROMINENCE * floor:
            logger.debug("  breathing: peak @ %.4f Hz not prominent (%.3g < %.1f×%.3g median) → 0.0 Hz",
                         float(freqs[best]), power[best], _BREATHING_PROMINENCE, floor)
            return 0.0
        peak = float(freqs[best])
        logger.debug("  breathing: local peak @ %.4f Hz (power=%.3g, %.1f× band median)",
                     peak, power[best], power[best] / floor if floor else float("inf"))
        return peak

    @staticmethod
    def _doppler_mean(freqs: np.ndarray, power: np.ndarray) -> float:
        """Spectral centroid: the power-weighted average frequency."""
        total = np.sum(power)
        if total == 0.0:
            return 0.0
        centroid = float(np.sum(freqs * power) / total)
        logger.debug("  doppler: spectral centroid Σ(f·P)/ΣP = %.4g / %.4g = %.4f Hz",
                     float(np.sum(freqs * power)), float(total), centroid)
        return centroid

    @staticmethod
    def _phase_variance(amp_rx: np.ndarray, phase_rx: np.ndarray) -> float:
        """Variance of the dominant subcarrier's unwrapped phase over time."""
        if phase_rx.shape[1] < 2:
            return 0.0
        mean_amp = amp_rx.mean(axis=1)
        if not np.any(mean_amp):
            return 0.0
        dominant = int(np.argmax(mean_amp))
        ph = np.unwrap(phase_rx[dominant, :].astype(np.float64))
        variance = float(np.var(ph))
        logger.debug("  phase_var: dominant subcarrier=%d (mean|amp|=%.3f) → unwrapped phase var=%.3f",
                     dominant, float(mean_amp[dominant]), variance)
        return variance
