"""Phase 1 tests — signal_cleaner_v2 (complex static subtraction + stage-2).

Validates the injection-mode math (Plan section 7): calibration baseline,
complex background subtraction, derived amplitude/phase, unfilled-column
handling, temporal-mean fallback, and the stage-2 band-pass.

Run:  python v2/tests/test_signal_cleaner_v2.py
"""

import sys

from _harness import Harness  # noqa: E402  (path bootstrap happens in _harness)

import numpy as np

from v2.config_v2 import NUM_RECEIVERS as R, NUM_SUBCARRIERS as S
from v2.ghost.signal_cleaner_v2 import SignalCleanerV2, CleanedCSI


def _const(val: complex, T: int) -> np.ndarray:
    return np.full((R, S, T), val, dtype=np.complex64)


# --- calibration -----------------------------------------------------------
def test_calibrate_from(h: Harness) -> None:
    B = 10 + 5j
    c = SignalCleanerV2()
    c.calibrate_from(_const(B, 50))
    h.expect("calibrated flag set", c.calibrated)
    h.expect("baseline shape [R,64]", c.baseline.shape == (R, S), str(c.baseline.shape))
    h.expect("baseline == empty-room constant", np.allclose(c.baseline, B))


def test_add_calibration_streaming(h: Harness) -> None:
    B = -3 + 8j
    c = SignalCleanerV2(calibration_samples=100)
    done1 = c.add_calibration(_const(B, 40))
    done2 = c.add_calibration(_const(B, 40))
    done3 = c.add_calibration(_const(B, 40))  # 120 >= 100 -> completes here
    h.expect("not done at 40 frames", done1 is False)
    h.expect("not done at 80 frames", done2 is False)
    h.expect("done at 120 frames", done3 is True)
    h.expect("streamed baseline correct", np.allclose(c.baseline, B, atol=1e-3))


# --- subtraction math ------------------------------------------------------
def test_static_subtraction_recovers_dynamic(h: Harness) -> None:
    B = 10 + 5j
    T = 12
    # dynamic varies over time and subcarrier
    t = np.arange(T)
    sub = np.arange(S)
    d = (t[None, None, :] - 3.0) + 1j * (sub[None, :, None] * 0.1)
    d = np.broadcast_to(d, (R, S, T)).astype(np.complex64)
    matrix = (_const(B, T) + d).astype(np.complex64)

    c = SignalCleanerV2()
    c.set_baseline(np.full((R, S), B, dtype=np.complex64))
    res = c.clean(matrix, apply_temporal_filter=False)
    h.expect("returns CleanedCSI", isinstance(res, CleanedCSI))
    h.expect("baseline subtracted -> dynamic == d", np.allclose(res.dynamic, d, atol=1e-2))
    h.expect("static recorded in result", np.allclose(res.static, B))


def test_empty_room_amplitude_zero(h: Harness) -> None:
    B = 10 + 5j
    c = SignalCleanerV2()
    c.calibrate_from(_const(B, 40))
    res = c.clean(_const(B, 40))  # operational window identical to baseline
    h.expect("empty-room dynamic ~ 0", np.allclose(res.dynamic, 0, atol=1e-3))
    h.expect("empty-room amplitude ~ 0", np.allclose(res.amplitude, 0, atol=1e-3))


def test_amplitude_and_phase_math(h: Harness) -> None:
    # baseline zero, dynamic = 3 + 4j constant -> A=5, phase=atan2(4,3)
    T = 10  # < 25 -> no band-pass; constant series -> Hampel no-op
    c = SignalCleanerV2()
    c.set_baseline(np.zeros((R, S), dtype=np.complex64))
    res = c.clean(_const(3 + 4j, T))
    h.expect("A_clean = |H_dyn| = 5", np.allclose(res.amplitude, 5.0, atol=1e-4))
    h.expect("phi_clean = atan2(4,3)", np.allclose(res.phase, np.arctan2(4.0, 3.0), atol=1e-5))


# --- temporal-mean fallback ------------------------------------------------
def test_temporal_mean_mode(h: Harness) -> None:
    B = 6 + 0j
    T = 64
    t = np.arange(T)
    osc = np.sin(2 * np.pi * 1.0 * t / 100.0)  # zero-mean
    d = np.broadcast_to(osc[None, None, :], (R, S, T)).astype(np.complex64)
    matrix = (_const(B, T) + d).astype(np.complex64)

    c = SignalCleanerV2(calibration_mode="temporal_mean")
    h.expect("starts uncalibrated", not c.calibrated)
    res = c.clean(matrix, apply_temporal_filter=False)
    h.expect("result marked not calibrated", res.calibrated is False)
    # temporal mean == B, so dynamic is the zero-mean oscillation
    per_bin_mean = res.dynamic.mean(axis=2)
    h.expect("dynamic is zero-mean over time", np.allclose(per_bin_mean, 0, atol=1e-3))


# --- unfilled column handling ----------------------------------------------
def test_zero_column_not_turned_into_negative_static(h: Harness) -> None:
    B = 10 + 5j
    d_val = 7 + 1j
    T = 6
    matrix = np.full((R, S, T), B + d_val, dtype=np.complex64)
    matrix[:, :, 2] = 0  # unfilled columns
    matrix[:, :, 4] = 0

    c = SignalCleanerV2()
    c.set_baseline(np.full((R, S), B, dtype=np.complex64))
    res = c.clean(matrix, apply_temporal_filter=False)
    h.expect("unfilled col 2 stays 0 (not -B)", np.allclose(res.dynamic[:, :, 2], 0))
    h.expect("unfilled col 4 stays 0 (not -B)", np.allclose(res.dynamic[:, :, 4], 0))
    h.expect("filled col recovers d_val", np.allclose(res.dynamic[:, :, 0], d_val, atol=1e-2))


# --- stage-2 band-pass -----------------------------------------------------
def test_bandpass_removes_dc_keeps_oscillation(h: Harness) -> None:
    T = 256
    t = np.arange(T)
    amp_series = 5.0 + np.sin(2 * np.pi * 1.0 * t / 100.0)  # DC 5 + 1 Hz, always > 0
    dyn = np.broadcast_to(amp_series[None, None, :], (R, S, T)).astype(np.complex64)

    c = SignalCleanerV2(sample_rate_hz=100.0)
    c.set_baseline(np.zeros((R, S), dtype=np.complex64))
    res = c.clean(dyn, apply_temporal_filter=True)
    mean_after = float(np.mean(res.amplitude))
    std_after = float(np.std(res.amplitude))
    h.expect("DC removed (mean ~ 0)", abs(mean_after) < 0.5, f"mean={mean_after:.3f}")
    h.expect("1 Hz oscillation retained (std > 0.3)", std_after > 0.3, f"std={std_after:.3f}")


# --- error handling --------------------------------------------------------
def test_errors(h: Harness) -> None:
    c = SignalCleanerV2()
    c.set_baseline(np.zeros((R, S), dtype=np.complex64))
    h.expect_raises(
        "clean rejects real matrix", TypeError,
        lambda: c.clean(np.zeros((R, S, 10), dtype=np.float32)),
    )
    h.expect_raises(
        "clean rejects wrong ndim", ValueError,
        lambda: c.clean(np.zeros((R, S), dtype=np.complex64)),
    )
    h.expect_raises(
        "set_baseline rejects wrong shape", ValueError,
        lambda: c.set_baseline(np.zeros((R, S + 1), dtype=np.complex64)),
    )


def main() -> int:
    h = Harness("signal_cleaner_v2 (complex)")
    h.case("calibrate_from", test_calibrate_from)
    h.case("add_calibration_streaming", test_add_calibration_streaming)
    h.case("static_subtraction_recovers_dynamic", test_static_subtraction_recovers_dynamic)
    h.case("empty_room_amplitude_zero", test_empty_room_amplitude_zero)
    h.case("amplitude_and_phase_math", test_amplitude_and_phase_math)
    h.case("temporal_mean_mode", test_temporal_mean_mode)
    h.case("zero_column_handling", test_zero_column_not_turned_into_negative_static)
    h.case("bandpass_removes_dc", test_bandpass_removes_dc_keeps_oscillation)
    h.case("errors", test_errors)
    return h.done()


if __name__ == "__main__":
    sys.exit(main())
