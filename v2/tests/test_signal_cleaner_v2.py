"""Phase 1 tests — signal_cleaner_v2 (complex static subtraction + stage-2).

Validates the injection-mode math (Plan section 7): calibration baseline,
complex background subtraction, derived amplitude/phase, unfilled-column
handling, temporal-mean fallback, and the stage-2 band-pass.

Run:  python v2/tests/test_signal_cleaner_v2.py
"""

import sys

from _harness import Harness

import numpy as np

from v2.config_v2 import NUM_RECEIVERS as R, NUM_SUBCARRIERS as S
from v2.ghost.signal_cleaner_v2 import SignalCleanerV2, CleanedCSI, _sanitize_phase


def _const(val: complex, T: int) -> np.ndarray:
    return np.full((R, S, T), val, dtype=np.complex64)


# These tests assert on the exact complex subtraction algebra with complex
# baselines, so they opt out of phase sanitization (which re-references phase
# per frame). Sanitization is exercised on its own in test_phase_sanitize_*.
def _cleaner(**kw) -> SignalCleanerV2:
    kw.setdefault("phase_sanitize", False)
    return SignalCleanerV2(**kw)


def test_calibrate_from(h: Harness) -> None:
    B = 10 + 5j
    c = _cleaner()
    c.calibrate_from(_const(B, 50))
    h.expect("calibrated flag set", c.calibrated)
    h.expect("baseline shape [R,64]", c.baseline.shape == (R, S), str(c.baseline.shape))
    h.expect("baseline == empty-room constant", np.allclose(c.baseline, B))


def test_add_calibration_streaming(h: Harness) -> None:
    B = -3 + 8j
    c = _cleaner(calibration_samples=100)
    done1 = c.add_calibration(_const(B, 40))
    done2 = c.add_calibration(_const(B, 40))
    done3 = c.add_calibration(_const(B, 40))
    h.expect("not done at 40 frames", done1 is False)
    h.expect("not done at 80 frames", done2 is False)
    h.expect("done at 120 frames", done3 is True)
    h.expect("streamed baseline correct", np.allclose(c.baseline, B, atol=1e-3))


def test_static_subtraction_recovers_dynamic(h: Harness) -> None:
    B = 10 + 5j
    T = 12
    t = np.arange(T)
    sub = np.arange(S)
    d = (t[None, None, :] - 3.0) + 1j * (sub[None, :, None] * 0.1)
    d = np.broadcast_to(d, (R, S, T)).astype(np.complex64)
    matrix = (_const(B, T) + d).astype(np.complex64)

    c = _cleaner()
    c.set_baseline(np.full((R, S), B, dtype=np.complex64))
    res = c.clean(matrix, apply_temporal_filter=False)
    h.expect("returns CleanedCSI", isinstance(res, CleanedCSI))
    h.expect("baseline subtracted -> dynamic == d", np.allclose(res.dynamic, d, atol=1e-2))
    h.expect("static recorded in result", np.allclose(res.static, B))


def test_empty_room_amplitude_zero(h: Harness) -> None:
    B = 10 + 5j
    c = SignalCleanerV2()
    c.calibrate_from(_const(B, 40))
    res = c.clean(_const(B, 40))
    h.expect("empty-room dynamic ~ 0", np.allclose(res.dynamic, 0, atol=1e-3))
    h.expect("empty-room amplitude ~ 0", np.allclose(res.amplitude, 0, atol=1e-3))


def test_amplitude_and_phase_math(h: Harness) -> None:
    T = 10
    c = _cleaner()
    c.set_baseline(np.zeros((R, S), dtype=np.complex64))
    res = c.clean(_const(3 + 4j, T))
    h.expect("A_clean = |H_dyn| = 5", np.allclose(res.amplitude, 5.0, atol=1e-4))
    h.expect("phi_clean = atan2(4,3)", np.allclose(res.phase, np.arctan2(4.0, 3.0), atol=1e-5))


def test_temporal_mean_mode(h: Harness) -> None:
    B = 6 + 0j
    T = 64
    t = np.arange(T)
    osc = np.sin(2 * np.pi * 1.0 * t / 100.0)
    d = np.broadcast_to(osc[None, None, :], (R, S, T)).astype(np.complex64)
    matrix = (_const(B, T) + d).astype(np.complex64)

    c = SignalCleanerV2(calibration_mode="temporal_mean")
    h.expect("starts uncalibrated", not c.calibrated)
    res = c.clean(matrix, apply_temporal_filter=False)
    h.expect("result marked not calibrated", res.calibrated is False)
    per_bin_mean = res.dynamic.mean(axis=2)
    h.expect("dynamic is zero-mean over time", np.allclose(per_bin_mean, 0, atol=1e-3))


def test_zero_column_not_turned_into_negative_static(h: Harness) -> None:
    B = 10 + 5j
    d_val = 7 + 1j
    T = 6
    matrix = np.full((R, S, T), B + d_val, dtype=np.complex64)
    matrix[:, :, 2] = 0
    matrix[:, :, 4] = 0

    c = _cleaner()
    c.set_baseline(np.full((R, S), B, dtype=np.complex64))
    res = c.clean(matrix, apply_temporal_filter=False)
    h.expect("unfilled col 2 stays 0 (not -B)", np.allclose(res.dynamic[:, :, 2], 0))
    h.expect("unfilled col 4 stays 0 (not -B)", np.allclose(res.dynamic[:, :, 4], 0))
    h.expect("filled col recovers d_val", np.allclose(res.dynamic[:, :, 0], d_val, atol=1e-2))


def test_bandpass_removes_dc_keeps_oscillation(h: Harness) -> None:
    T = 256
    t = np.arange(T)
    amp_series = 5.0 + np.sin(2 * np.pi * 1.0 * t / 100.0)
    dyn = np.broadcast_to(amp_series[None, None, :], (R, S, T)).astype(np.complex64)

    c = SignalCleanerV2(sample_rate_hz=100.0)
    c.set_baseline(np.zeros((R, S), dtype=np.complex64))
    res = c.clean(dyn, apply_temporal_filter=True)
    mean_after = float(np.mean(res.amplitude))
    std_after = float(np.std(res.amplitude))
    h.expect("DC removed (mean ~ 0)", abs(mean_after) < 0.5, f"mean={mean_after:.3f}")
    h.expect("1 Hz oscillation retained (std > 0.3)", std_after > 0.3, f"std={std_after:.3f}")


def test_phase_sanitize_removes_ramp(h: Harness) -> None:
    # H_true has flat phase; each frame is corrupted by a random linear ramp
    # (STO slope + CFO offset). Sanitization must de-rotate it back to flat.
    rng = np.random.default_rng(0)
    T = 8
    k = np.arange(S)
    m = np.zeros((R, S, T), dtype=np.complex64)
    for rx in range(R):
        for t in range(T):
            a = rng.uniform(-0.15, 0.15)
            b = rng.uniform(-np.pi, np.pi)
            m[rx, :, t] = np.exp(1j * (a * k + b))
    out = _sanitize_phase(m)
    h.expect("magnitude preserved (=1)", np.allclose(np.abs(out), 1.0, atol=1e-4))
    h.expect("per-frame ramp removed -> flat phase ~0",
             np.allclose(np.angle(out), 0.0, atol=1e-3))


def test_phase_sanitize_preserves_amplitude(h: Harness) -> None:
    # It is a pure phase rotation: |out| == |in| for arbitrary complex input.
    rng = np.random.default_rng(1)
    m = (rng.standard_normal((R, S, 5)) + 1j * rng.standard_normal((R, S, 5))).astype(np.complex64)
    out = _sanitize_phase(m)
    h.expect("magnitude unchanged elementwise", np.allclose(np.abs(out), np.abs(m), atol=1e-4))


def test_phase_sanitize_coherent_static_subtraction(h: Harness) -> None:
    # The payoff: a fixed static scene observed with random per-frame CFO/STO.
    # After sanitization every frame collapses to the same phase reference, so the
    # calibrated baseline cancels it; without it, the random phase survives.
    rng = np.random.default_rng(2)
    T = 60
    k = np.arange(S)
    h_static = np.zeros((R, S), dtype=np.complex64)
    for rx in range(R):
        mag = 1.0 + 0.2 * k / S + 0.3 * (rx + 1)
        phase = 0.05 * k + 0.4 * np.sin(0.2 * k)  # smooth, unwrap-able
        h_static[rx] = mag * np.exp(1j * phase)

    def observe(n: int) -> np.ndarray:
        m = np.zeros((R, S, n), dtype=np.complex64)
        for rx in range(R):
            for t in range(n):
                a = rng.uniform(-0.1, 0.1)      # per-frame STO slope
                b = rng.uniform(-np.pi, np.pi)  # per-frame CFO offset
                m[rx, :, t] = h_static[rx] * np.exp(1j * (a * k + b))
        return m

    calib, op = observe(T), observe(T)

    on = SignalCleanerV2(phase_sanitize=True)
    on.calibrate_from(calib)
    resid_on = float(np.mean(np.abs(on.clean(op, apply_temporal_filter=False).dynamic)))

    off = SignalCleanerV2(phase_sanitize=False)
    off.calibrate_from(calib)
    resid_off = float(np.mean(np.abs(off.clean(op, apply_temporal_filter=False).dynamic)))

    h.expect("sanitized static residual ~ 0", resid_on < 0.05, f"resid_on={resid_on:.4f}")
    h.expect("sanitization beats raw static subtraction by >5x",
             resid_on * 5 < resid_off, f"on={resid_on:.4f} off={resid_off:.4f}")


def test_phase_sanitize_toggle_effect(h: Harness) -> None:
    # A constant-across-subcarriers frame: raw phase = atan2(4,3); sanitized, the
    # constant offset is removed so phase -> 0. Amplitude (=5) is identical either way.
    base = np.zeros((R, S), dtype=np.complex64)
    off = SignalCleanerV2(phase_sanitize=False)
    off.set_baseline(base)
    res_off = off.clean(_const(3 + 4j, 6))
    h.expect("raw keeps phase atan2(4,3)", np.allclose(res_off.phase, np.arctan2(4.0, 3.0), atol=1e-5))

    on = SignalCleanerV2(phase_sanitize=True)
    on.set_baseline(base)
    res_on = on.clean(_const(3 + 4j, 6))
    h.expect("sanitized rotates constant phase to 0", np.allclose(res_on.phase, 0.0, atol=1e-4))
    h.expect("amplitude identical (=5) both ways", np.allclose(res_on.amplitude, 5.0, atol=1e-4))


def test_static_ewma_absorbs_new_clutter(h: Harness) -> None:
    # After calibration on empty room B, a new static object C appears. It shows
    # up as dynamic at first, then the adaptive baseline absorbs it over time.
    B, C = 10 + 5j, 4 - 2j
    c = SignalCleanerV2(phase_sanitize=False, static_ewma_alpha=0.3)
    c.calibrate_from(_const(B, 30))
    scene = _const(B + C, 8)

    first = float(np.mean(np.abs(c.clean(scene, apply_temporal_filter=False).dynamic)))
    last = first
    for _ in range(30):
        last = float(np.mean(np.abs(c.clean(scene, apply_temporal_filter=False).dynamic)))

    h.expect("new clutter initially appears as dynamic ~|C|", first > 0.9 * abs(C),
             f"first={first:.3f} |C|={abs(C):.3f}")
    h.expect("adaptive baseline absorbs the static clutter", last < 0.05 * abs(C),
             f"last={last:.4f}")


def test_static_ewma_disabled_keeps_fixed_baseline(h: Harness) -> None:
    B, C = 10 + 5j, 4 - 2j
    c = SignalCleanerV2(phase_sanitize=False, static_ewma_alpha=0.0)
    c.calibrate_from(_const(B, 30))
    scene = _const(B + C, 8)
    dn = 0.0
    for _ in range(10):
        dn = float(np.mean(np.abs(c.clean(scene, apply_temporal_filter=False).dynamic)))
    h.expect("α=0 keeps dynamic pinned at |C| (no drift)", abs(dn - abs(C)) < 1e-3,
             f"dn={dn:.4f} |C|={abs(C):.4f}")
    h.expect("α=0 baseline unchanged", np.allclose(c.baseline, B, atol=1e-4))


def test_static_ewma_current_window_uses_prewindow_baseline(h: Harness) -> None:
    # A single clean() must be identical with or without adaptation, and
    # res.static must be the baseline actually subtracted (not the updated one).
    B, C = 10 + 5j, 4 - 2j
    scene = _const(B + C, 8)

    fixed = SignalCleanerV2(phase_sanitize=False, static_ewma_alpha=0.0)
    fixed.calibrate_from(_const(B, 30))
    r_fixed = fixed.clean(scene, apply_temporal_filter=False)

    adaptive = SignalCleanerV2(phase_sanitize=False, static_ewma_alpha=0.3)
    adaptive.calibrate_from(_const(B, 30))
    r_adaptive = adaptive.clean(scene, apply_temporal_filter=False)

    h.expect("first clean identical regardless of α",
             np.allclose(r_fixed.dynamic, r_adaptive.dynamic, atol=1e-5))
    h.expect("res.static is the pre-update baseline B", np.allclose(r_adaptive.static, B, atol=1e-4))
    h.expect("internal baseline advanced past B", not np.allclose(adaptive.baseline, B, atol=1e-3))


def test_static_ewma_adapt_false_skips_update(h: Harness) -> None:
    B, C = 10 + 5j, 4 - 2j
    c = SignalCleanerV2(phase_sanitize=False, static_ewma_alpha=0.3)
    c.calibrate_from(_const(B, 30))
    c.clean(_const(B + C, 8), apply_temporal_filter=False, adapt=False)
    h.expect("adapt=False leaves the baseline unchanged", np.allclose(c.baseline, B, atol=1e-4))


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
    h.case("phase_sanitize_removes_ramp", test_phase_sanitize_removes_ramp)
    h.case("phase_sanitize_preserves_amplitude", test_phase_sanitize_preserves_amplitude)
    h.case("phase_sanitize_coherent_static_subtraction", test_phase_sanitize_coherent_static_subtraction)
    h.case("phase_sanitize_toggle_effect", test_phase_sanitize_toggle_effect)
    h.case("static_ewma_absorbs_new_clutter", test_static_ewma_absorbs_new_clutter)
    h.case("static_ewma_disabled_keeps_fixed_baseline", test_static_ewma_disabled_keeps_fixed_baseline)
    h.case("static_ewma_current_window_uses_prewindow_baseline", test_static_ewma_current_window_uses_prewindow_baseline)
    h.case("static_ewma_adapt_false_skips_update", test_static_ewma_adapt_false_skips_update)
    h.case("errors", test_errors)
    return h.done()


if __name__ == "__main__":
    sys.exit(main())
