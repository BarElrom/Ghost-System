"""Phase 1 tests — feature_extractor_v2 (complex-aware features).

Validates the v1-style spectral features plus the v2 additions (per-node
energies for the localizer, and phase-variance features).

Run:  python v2/tests/test_feature_extractor_v2.py
"""

import sys

from _harness import Harness

import numpy as np

from v2.config_v2 import NUM_RECEIVERS as R, NUM_SUBCARRIERS as S
from v2.ghost.feature_extractor_v2 import FeatureExtractorV2, FeatureSet


class _Cleaned:
    """Minimal stand-in for CleanedCSI carrying amplitude/phase/dynamic."""

    def __init__(self, amplitude, phase, dynamic):
        self.amplitude = amplitude
        self.phase = phase
        self.dynamic = dynamic


def _cleaned(amplitude, phase=None):
    amplitude = amplitude.astype(np.float32)
    if phase is None:
        phase = np.zeros_like(amplitude)
    phase = phase.astype(np.float32)
    dynamic = (amplitude * np.exp(1j * phase)).astype(np.complex64)
    return _Cleaned(amplitude, phase, dynamic)


def test_empty_input(h: Harness) -> None:
    ex = FeatureExtractorV2()
    fs = ex.extract(_cleaned(np.zeros((R, S, 200))))
    h.expect("returns FeatureSet", isinstance(fs, FeatureSet))
    h.expect("breathing 0", fs.breathing_frequency == 0.0)
    h.expect("energy 0", fs.total_energy == 0.0)
    h.expect("variances 0", all(v == 0.0 for v in fs.variance_rx))
    h.expect("node energies 0", all(v == 0.0 for v in fs.node_energies.values()))
    h.expect("phase variances 0", all(v == 0.0 for v in fs.phase_variance_rx))


def test_breathing_frequency(h: Harness) -> None:
    fs_hz = 100.0
    T = 1000
    t = np.arange(T)
    breath = np.sin(2 * np.pi * 0.3 * t / fs_hz)
    amp = np.broadcast_to(breath[None, None, :], (R, S, T)).copy()
    amp = amp + 2.0

    ex = FeatureExtractorV2(sample_rate_hz=fs_hz)
    fs = ex.extract(_cleaned(amp))
    h.expect("breathing freq ~ 0.3 Hz", abs(fs.breathing_frequency - 0.3) < 0.06,
             f"{fs.breathing_frequency:.3f}")
    h.expect("total energy > 0", fs.total_energy > 0)


def test_breathing_rejects_drift_floor(h: Harness) -> None:
    """A monotonic 1/f-style drift (no periodic peak) must report 0.0 Hz.

    Regression guard for the old argmax behavior, which pinned the reading to
    the band's low edge (~0.1 Hz) — a phantom breathing signal — whenever the
    spectrum just decayed with frequency.
    """
    fs_hz = 50.0
    T = 1000
    t = np.arange(T, dtype=np.float64) / T
    # Smooth monotonic drift (survives linear detrending, no periodic component):
    # strong low-frequency power but no in-band spectral peak.
    drift = 3.0 * t ** 2 + t ** 3
    amp = np.broadcast_to(drift[None, None, :], (R, S, T)).copy() + 5.0

    ex = FeatureExtractorV2(sample_rate_hz=fs_hz)
    fs = ex.extract(_cleaned(amp))
    h.expect("no phantom breathing floor", fs.breathing_frequency == 0.0,
             f"{fs.breathing_frequency:.3f}")


def test_node_energies_scale_with_amplitude(h: Harness) -> None:
    T = 50
    amp = np.zeros((R, S, T))
    amp[0] = 1.0
    amp[1] = 2.0
    amp[2] = 3.0
    ex = FeatureExtractorV2()
    fs = ex.extract(_cleaned(amp))
    h.expect("RX1 energy = 64", np.isclose(fs.node_energies["RX1"], 64.0), str(fs.node_energies["RX1"]))
    h.expect("RX2 energy = 128", np.isclose(fs.node_energies["RX2"], 128.0), str(fs.node_energies["RX2"]))
    h.expect("RX3 energy = 192", np.isclose(fs.node_energies["RX3"], 192.0), str(fs.node_energies["RX3"]))
    h.expect("node energy keys are RX1/2/3", set(fs.node_energies) == {"RX1", "RX2", "RX3"})


def test_variance_reflects_motion(h: Harness) -> None:
    T = 200
    t = np.arange(T)
    amp = np.zeros((R, S, T))
    amp[0] = np.broadcast_to((np.sin(2 * np.pi * 1.0 * t / 100.0) + 2.0)[None, :], (S, T))
    amp[1] = 2.0
    amp[2] = 2.0
    ex = FeatureExtractorV2()
    fs = ex.extract(_cleaned(amp))
    h.expect("moving receiver has high variance", fs.variance_rx[0] > 1e-3, str(fs.variance_rx[0]))
    h.expect("still receiver ~0 variance", fs.variance_rx[1] < 1e-6, str(fs.variance_rx[1]))


def test_phase_variance(h: Harness) -> None:
    T = 200
    t = np.arange(T)
    amp = np.full((R, S, T), 3.0)
    phase = np.zeros((R, S, T))
    phase[0, 0, :] = 0.02 * t
    ex = FeatureExtractorV2()
    fs = ex.extract(_cleaned(amp, phase))
    h.expect("moving-phase receiver has variance", fs.phase_variance_rx[0] > 1e-2,
             str(fs.phase_variance_rx[0]))
    h.expect("static-phase receiver ~0", fs.phase_variance_rx[1] < 1e-9,
             str(fs.phase_variance_rx[1]))


def test_vector6_and_dict(h: Harness) -> None:
    T = 200
    amp = np.full((R, S, T), 2.0)
    ex = FeatureExtractorV2()
    fs = ex.extract(_cleaned(amp))
    v = fs.vector6()
    h.expect("vector6 shape (6,)", v.shape == (6,), str(v.shape))
    h.expect("vector6[0] == breathing", v[0] == fs.breathing_frequency)
    h.expect("vector6[3:6] == variances", np.allclose(v[3:6], fs.variance_rx[:3]))
    d = fs.to_dict()
    h.expect("to_dict has all keys",
             set(d) == {"breathing_frequency", "total_energy", "doppler_mean",
                        "variance_rx", "node_energies", "phase_variance_rx"})


def test_bad_shape(h: Harness) -> None:
    ex = FeatureExtractorV2()
    h.expect_raises(
        "extract rejects 2-D amplitude", ValueError,
        lambda: ex.extract(_cleaned(np.zeros((R, S)))),
    )


def main() -> int:
    h = Harness("feature_extractor_v2")
    h.case("empty_input", test_empty_input)
    h.case("breathing_frequency", test_breathing_frequency)
    h.case("breathing_rejects_drift_floor", test_breathing_rejects_drift_floor)
    h.case("node_energies", test_node_energies_scale_with_amplitude)
    h.case("variance_reflects_motion", test_variance_reflects_motion)
    h.case("phase_variance", test_phase_variance)
    h.case("vector6_and_dict", test_vector6_and_dict)
    h.case("bad_shape", test_bad_shape)
    return h.done()


if __name__ == "__main__":
    sys.exit(main())
