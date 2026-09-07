
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
    PHASE_SANITIZE,
    SAMPLE_RATE_HZ,
    STATIC_EWMA_ALPHA,
)

logger = logging.getLogger("ghost.v2.signal_cleaner")

_MIN_SAMPLES_FOR_BANDPASS = 25


@dataclass
class CleanedCSI:
    """The result of cleaning one window of CSI."""

    dynamic: np.ndarray
    amplitude: np.ndarray
    phase: np.ndarray
    static: np.ndarray
    calibrated: bool


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


def _sanitize_phase(matrix: np.ndarray) -> np.ndarray:
    """Remove per-frame CFO/SFO by de-rotating a fitted linear phase ramp.

    ESP32 CSI carries a random per-packet carrier-phase offset (CFO, a constant
    across subcarriers) and a sampling-time offset (STO, a linear ramp across
    subcarriers): ``H[k] = H_true[k]·exp(j·(a·k + b))``. We estimate ``(a, b)``
    per (receiver, frame) with a least-squares fit of the unwrapped phase vs the
    subcarrier index and multiply the ramp out. Magnitude is untouched — only the
    phase reference changes — so amplitude features are unaffected, but phase
    becomes comparable across frames (coherent static subtraction, meaningful
    phase variance). Zero-filled (inactive) columns fit to 0 and stay 0.

    Args:
        matrix: complex ``[R, 64, T]``.

    Returns:
        complex64 ``[R, 64, T]`` with the per-frame linear phase removed.
    """
    receiver, subcarrier, time = matrix.shape
    if subcarrier < 2:  # need >= 2 subcarriers to fit a line
        return matrix.astype(np.complex64)
    cols = matrix.transpose(1, 0, 2).reshape(subcarrier, receiver * time)  # [K, N]
    k = np.arange(subcarrier, dtype=np.float64)
    phase = np.unwrap(np.angle(cols), axis=0)  # [K, N]

    kc = k - k.mean()
    denom = float(np.sum(kc ** 2))  # constant > 0 for K >= 2
    phase_mean = phase.mean(axis=0)  # [N]
    slope = (kc[:, None] * (phase - phase_mean[None, :])).sum(axis=0) / denom  # [N]
    intercept = phase_mean - slope * k.mean()  # [N]

    correction = np.exp(-1j * (slope[None, :] * k[:, None] + intercept[None, :]))  # [K, N]
    fixed = (cols * correction).reshape(subcarrier, receiver, time).transpose(1, 0, 2)
    return fixed.astype(np.complex64)


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
        phase_sanitize: bool | None = None,
        static_ewma_alpha: float | None = None,
    ):
        self.sample_rate_hz = sample_rate_hz
        self.hampel_window = hampel_window
        self.hampel_threshold = hampel_threshold
        self.calibration_samples = calibration_samples
        self.calibration_mode = calibration_mode
        self._phase_sanitize = PHASE_SANITIZE if phase_sanitize is None else bool(phase_sanitize)
        self._static_ewma_alpha = (
            STATIC_EWMA_ALPHA if static_ewma_alpha is None else float(static_ewma_alpha)
        )
        self._lowcut = lowcut
        self._highcut = highcut

        nyquist = sample_rate_hz / 2.0
        self._bandpass = butter(butter_order, [lowcut / nyquist, highcut / nyquist],
                                btype="band", output="sos")

        self._static: np.ndarray | None = None
        self._sum = np.zeros((NUM_RECEIVERS, NUM_SUBCARRIERS), dtype=np.complex128)
        self._count = np.zeros(NUM_RECEIVERS, dtype=np.int64)
        self._frames_seen = 0

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

        if self._phase_sanitize:
            frames = _sanitize_phase(frames)

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
        matrix = np.asarray(matrix)
        if self._phase_sanitize:
            matrix = _sanitize_phase(matrix)
        self._static = _column_mean(matrix)
        if logger.isEnabledFor(logging.INFO):
            active = _active_columns(matrix)
            for rx in range(self._static.shape[0]):
                logger.info(
                    "calibrate rx=%d: H_static |mean|=%.3f over %d active frames "
                    "(empty-room baseline)",
                    rx, float(np.abs(self._static[rx]).mean()), int(active[rx].sum()),
                )

    def clean(self, matrix: np.ndarray, apply_temporal_filter: bool = True,
              adapt: bool = True) -> CleanedCSI:
        """Subtract the baseline, derive amplitude/phase, and band-pass the motion.

        When ``adapt`` is set and a calibrated baseline exists, the baseline is
        nudged toward this window's static estimate afterwards (EWMA, Note 7) so
        it tracks slow drift / new clutter. This window is cleaned with the
        *pre-update* baseline, so a single ``clean`` call is unaffected by α; pass
        ``adapt=False`` to skip the update (e.g. an auxiliary breathing window).
        """
        matrix = np.asarray(matrix)
        if matrix.ndim != 3:
            raise ValueError(f"expected [R,64,T] complex, got {matrix.shape}")
        if not np.iscomplexobj(matrix):
            raise TypeError("matrix must be complex (from gateway_v2)")

        if self._phase_sanitize:
            matrix = _sanitize_phase(matrix)
            logger.debug("clean: applied per-frame CFO/SFO phase sanitization")

        baseline, calibrated = self._pick_baseline(matrix)

        dynamic = (matrix - baseline[:, :, None]).astype(np.complex64)
        keep = _active_columns(matrix)[:, None, :]
        n_zeroed = int(dynamic.size - np.count_nonzero(np.broadcast_to(keep, dynamic.shape)))
        dynamic = np.where(keep, dynamic, 0).astype(np.complex64)
        logger.debug("clean: baseline=%s calibrated=%s zero-forced cells=%d",
                     "H_static" if calibrated else self.calibration_mode, calibrated, n_zeroed)

        # Per-rx DSP diagnostics: how much did the static subtraction remove, and
        # how phase-coherent is the raw CSI (low coherence => weak complex subtraction).
        if logger.isEnabledFor(logging.INFO):
            active2d = keep[:, 0, :]
            for rx in range(matrix.shape[0]):
                col = active2d[rx]
                if not col.any():
                    continue
                raw_rx = matrix[rx][:, col]
                dyn_rx = dynamic[rx][:, col]
                raw_mag = float(np.abs(raw_rx).mean())
                dyn_mag = float(np.abs(dyn_rx).mean())
                removed = 100.0 * (1.0 - dyn_mag / (raw_mag + 1e-12))
                coherence = float(np.abs(raw_rx.mean(axis=1)).mean() / (raw_mag + 1e-12))
                logger.info(
                    "clean rx=%d T=%d: |H_raw|=%.3f → |H_dyn|=%.3f (−%.0f%% static)  "
                    "phase-coherence=%.3f%s",
                    rx, raw_rx.shape[1], raw_mag, dyn_mag, removed, coherence,
                    "  ⚠low" if coherence < 0.3 else "",
                )

        amplitude = np.abs(dynamic).astype(np.float32)
        phase = np.angle(dynamic).astype(np.float32)
        if apply_temporal_filter:
            amplitude = self._bandpass_amplitude(amplitude)

        result = CleanedCSI(dynamic, amplitude, phase, baseline.astype(np.complex64), calibrated)

        # Adapt the baseline AFTER cleaning this window, so result.static is the
        # baseline actually subtracted and the current window is unaffected by α.
        if adapt and calibrated and self._static_ewma_alpha > 0.0:
            self._update_baseline(matrix)

        return result

    def _update_baseline(self, matrix: np.ndarray) -> None:
        """EWMA the calibrated baseline toward this window's static estimate.

        The window's static component is its per-(rx, subcarrier) complex mean
        over active columns; motion averages out, static clutter persists. Only
        receivers with data in this window are updated.
        """
        current = _column_mean(matrix)  # [R, 64] over active columns
        active = _active_columns(matrix).any(axis=1)  # [R]
        a = self._static_ewma_alpha
        for rx in range(NUM_RECEIVERS):
            if active[rx]:
                self._static[rx] = (
                    (1.0 - a) * self._static[rx] + a * current[rx]
                ).astype(np.complex64)
        logger.debug("clean: EWMA baseline update α=%.3f on rx=%s",
                     a, [rx for rx in range(NUM_RECEIVERS) if active[rx]])

    def _pick_baseline(self, matrix: np.ndarray) -> tuple[np.ndarray, bool]:
        """Choose the baseline: the calibrated one, the window mean, or zeros."""
        if self.calibrated:
            return self._static, True
        if self.calibration_mode == "temporal_mean":
            return _column_mean(matrix), False
        logger.warning("clean() before calibration (mode=%s); subtracting zero", self.calibration_mode)
        return np.zeros(matrix.shape[:2], dtype=np.complex64), False

    def _bandpass_amplitude(self, amplitude: np.ndarray) -> np.ndarray:
        """Hampel-despike then band-pass each subcarrier's amplitude over time."""
        receiver, subcarrier, time = amplitude.shape
        out = np.empty_like(amplitude)
        can_bandpass = time >= _MIN_SAMPLES_FOR_BANDPASS
        if not can_bandpass:
            logger.warning("Only %d samples (<%d); Hampel only, band-pass skipped", time, _MIN_SAMPLES_FOR_BANDPASS)
        n_active_sub = 0
        n_hampel_replaced = 0
        for rx in range(receiver):
            for sub in range(subcarrier):
                signal = amplitude[rx, sub].astype(np.float64)
                if not np.any(signal):
                    out[rx, sub] = 0.0
                    continue
                n_active_sub += 1
                despiked = self._hampel_despike(signal)
                n_hampel_replaced += int(np.count_nonzero(despiked != signal))
                signal = despiked
                if can_bandpass:
                    signal = sosfiltfilt(self._bandpass, signal)
                out[rx, sub] = signal
        logger.debug(
            "bandpass: %d/%d active subcarriers, band=[%.2f,%.2f]Hz (sosfiltfilt, zero-phase) %s; "
            "Hampel (thr=%.1f·MAD) replaced %d outlier sample(s)",
            n_active_sub, receiver * subcarrier, self._lowcut, self._highcut,
            "applied" if can_bandpass else "SKIPPED", self.hampel_threshold, n_hampel_replaced,
        )
        return out

    def _hampel_despike(self, signal: np.ndarray) -> np.ndarray:
        """Replace outliers (> threshold*MAD from the local median) with the median."""
        k = self.hampel_window
        cleaned = signal.copy()
        for i in range(len(signal)):
            window = signal[max(0, i - k): i + k + 1]
            median = np.median(window)
            sigma = 1.4826 * np.median(np.abs(window - median))
            if sigma > 0 and abs(signal[i] - median) > self.hampel_threshold * sigma:
                cleaned[i] = median
        return cleaned
