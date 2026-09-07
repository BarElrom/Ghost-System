"""Tests for the CSI-Bench .mat adapter.

Uses synthetic .mat files (scipy.io.savemat) so the adapter logic is validated
without the real Kaggle dataset: amplitude+phase reconstruction, complex-key
path, subcarrier resampling, orientation auto-detection, and error handling.

Run:  python v2/tests/test_csi_bench.py
"""

import os
import sys
import tempfile

from _harness import Harness

import numpy as np
from scipy.io import savemat

from v2.config_v2 import NUM_SUBCARRIERS as S
from v2.injector.adapters.csi_bench import CSIBenchAdapter
from v2.injector.injector import build_adapter


def _write_mat(**vars) -> str:
    path = tempfile.NamedTemporaryFile(suffix=".mat", delete=False).name
    savemat(path, vars)
    return path


def _amp_phase(T, n_sub):
    """Deterministic amplitude/phase grids of shape (T, n_sub)."""
    t = np.arange(T)[:, None]
    k = np.arange(n_sub)[None, :]
    amp = 5.0 + (t % 7) + 0.1 * k
    phase = np.sin(0.05 * t + 0.1 * k)
    return amp.astype(np.float64), phase.astype(np.float64)


def test_amp_phase_reconstruction(h: Harness) -> None:
    T = 200
    amp, phase = _amp_phase(T, S)
    path = _write_mat(amplitude=amp, phase=phase)
    try:
        snaps = list(CSIBenchAdapter(path).snapshots())
        h.expect("one snapshot per time sample", len(snaps) == T, str(len(snaps)))
        iq0 = snaps[0].iq_by_stream[0]
        h.expect("64 subcarriers", iq0.shape == (S,), str(iq0.shape))
        h.expect("I = A cos(phase)", np.allclose(iq0.real, amp[0] * np.cos(phase[0]), atol=1e-2))
        h.expect("Q = A sin(phase)", np.allclose(iq0.imag, amp[0] * np.sin(phase[0]), atol=1e-2))
    finally:
        os.unlink(path)


def test_complex_key(h: Harness) -> None:
    T = 120
    amp, phase = _amp_phase(T, S)
    csi = (amp * np.exp(1j * phase)).astype(np.complex64)
    path = _write_mat(csi=csi)
    try:
        snaps = list(CSIBenchAdapter(path).snapshots())
        h.expect("frames from complex key", len(snaps) == T)
        h.expect("complex values preserved",
                 np.allclose(snaps[5].iq_by_stream[0], csi[5], atol=1e-2))
    finally:
        os.unlink(path)


def test_subcarrier_resample(h: Harness) -> None:
    T, n_in = 150, 30
    amp, phase = _amp_phase(T, n_in)
    path = _write_mat(amplitude=amp, phase=phase)
    try:
        snaps = list(CSIBenchAdapter(path).snapshots())
        h.expect("resampled to 64 subcarriers", snaps[0].iq_by_stream[0].shape == (S,))
        h.expect("frame count unchanged", len(snaps) == T)
    finally:
        os.unlink(path)


def test_orientation_autodetect(h: Harness) -> None:
    T, n_sub = 300, S
    amp, phase = _amp_phase(T, n_sub)
    path = _write_mat(amplitude=amp.T, phase=phase.T)
    try:
        snaps = list(CSIBenchAdapter(path).snapshots())
        h.expect("auto-oriented to T frames", len(snaps) == T, str(len(snaps)))
        h.expect("each frame has 64 subcarriers", snaps[0].iq_by_stream[0].shape == (S,))
    finally:
        os.unlink(path)


def test_multiantenna_collapse(h: Harness) -> None:
    T, ant = 100, 3
    amp, phase = _amp_phase(T, S)
    amp3 = np.stack([amp, amp + 1, amp + 2], axis=2)
    phase3 = np.stack([phase, phase, phase], axis=2)
    path = _write_mat(amplitude=amp3, phase=phase3)
    try:
        snaps = list(CSIBenchAdapter(path).snapshots())
        h.expect("3D collapsed to T frames", len(snaps) == T, str(len(snaps)))
        h.expect("took antenna 0",
                 np.allclose(snaps[0].iq_by_stream[0].real, amp[0] * np.cos(phase[0]), atol=1e-2))
    finally:
        os.unlink(path)


def test_missing_keys_errors(h: Harness) -> None:
    path = _write_mat(something_else=np.zeros((10, 10)))
    try:
        h.expect_raises("missing amp/phase raises", ValueError,
                        lambda: list(CSIBenchAdapter(path).snapshots()))
    finally:
        os.unlink(path)


def test_registered_in_build_adapter(h: Harness) -> None:
    amp, phase = _amp_phase(200, S)
    path = _write_mat(amplitude=amp, phase=phase)
    try:
        adapter = build_adapter("csi_bench", path)
        h.expect("build_adapter knows csi_bench", adapter.name == "csi_bench")
        h.expect("it produces snapshots", len(list(adapter.snapshots())) == 200)
    finally:
        os.unlink(path)


def main() -> int:
    h = Harness("csi_bench adapter")
    h.case("amp_phase_reconstruction", test_amp_phase_reconstruction)
    h.case("complex_key", test_complex_key)
    h.case("subcarrier_resample", test_subcarrier_resample)
    h.case("orientation_autodetect", test_orientation_autodetect)
    h.case("multiantenna_collapse", test_multiantenna_collapse)
    h.case("missing_keys_errors", test_missing_keys_errors)
    h.case("registered_in_build_adapter", test_registered_in_build_adapter)
    return h.done()


if __name__ == "__main__":
    sys.exit(main())
