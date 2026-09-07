"""
Intel Wi-Fi CSI Respiratory Sensing adapter (Zenodo dataset #3).

Presence / breathing-focused CSI. Zenodo layouts vary, so this adapter is
deliberately flexible and self-describing:

  * ``.mat`` files  → reuse the CSI-Bench loader (amplitude+phase or complex CSI)
  * ``.csv`` files  → each row is one time step of CSI values (amplitude, or
                      interleaved I/Q if the row length is 2x subcarriers)

If it can't figure out the layout, run the inspector to see what's inside:

    ./.venv/bin/python -m v2.injector.adapters.intel_resp <file>

then pass the right key/column hints (or tell the maintainer the printed shapes
so the parser can be pinned to this dataset's exact format).
"""

import logging

import numpy as np

from v2.config_v2 import NUM_SUBCARRIERS
from v2.injector.adapters.base import DatasetAdapter, Snapshot, normalize_subcarriers
from v2.injector.adapters.csi_bench import (
    _AMP_KEYS,
    _COMPLEX_KEYS,
    _PHASE_KEYS,
    _find_key,
    load_mat,
)

logger = logging.getLogger("ghost.v2.adapter.intel_resp")


class IntelRespAdapter(DatasetAdapter):
    """Yields single-link Snapshots from an Intel respiratory CSI file (.mat/.csv).

    Args:
        path: dataset file (.mat or .csv).
        amp_key / phase_key / complex_key: override .mat variable names.
        max_frames: optional cap on frames.
        label: optional ground-truth label attached to every Snapshot.
    """

    num_streams = 1
    name = "intel_resp"

    def __init__(self, path, amp_key=None, phase_key=None, complex_key=None,
                 max_frames=None, label=None):
        self.path = path
        self.amp_key = amp_key
        self.phase_key = phase_key
        self.complex_key = complex_key
        self.max_frames = max_frames
        self.label = label

    def snapshots(self):
        csi = self._extract_complex()  # [time, subcarrier] complex
        count = 0
        for t in range(csi.shape[0]):
            iq = normalize_subcarriers(csi[t], NUM_SUBCARRIERS)
            yield Snapshot(iq_by_stream={0: iq}, label=self.label)
            count += 1
            if self.max_frames is not None and count >= self.max_frames:
                break
        logger.info("%s: %d frames from %s", self.name, count, self.path)

    def _extract_complex(self) -> np.ndarray:
        if str(self.path).lower().endswith(".csv"):
            return self._from_csv()
        return self._from_mat()

    def _from_mat(self) -> np.ndarray:
        data = load_mat(self.path)
        ckey = self.complex_key or _find_key(data, _COMPLEX_KEYS)
        if ckey and np.iscomplexobj(np.asarray(data[ckey])):
            csi = np.asarray(data[ckey])
        else:
            akey = self.amp_key or _find_key(data, _AMP_KEYS)
            pkey = self.phase_key or _find_key(data, _PHASE_KEYS)
            if akey is None:
                raise ValueError(
                    f"{self.path}: no amplitude/complex variable found. "
                    f"Run: python -m v2.injector.adapters.intel_resp {self.path}"
                )
            amp = np.asarray(data[akey], dtype=np.float64)
            if pkey is not None:
                pha = np.asarray(data[pkey], dtype=np.float64)
                csi = amp * np.exp(1j * pha)
            else:  # amplitude only — presence/breathing still works on |H|
                csi = amp.astype(np.complex128)
        return _orient_time_subcarrier(csi)

    def _from_csv(self) -> np.ndarray:
        import csv

        with open(self.path, newline="") as fh:
            reader = csv.reader(fh)
            first = next(reader)
            has_header = any(not _is_number(tok) for tok in first)
            if has_header:
                header = [h.strip().lower() for h in first]
                real_idx = [i for i, h in enumerate(header) if h.startswith("real")]
                imag_idx = [i for i, h in enumerate(header) if h.startswith("imag")]
                data = np.array([[float(x) for x in row] for row in reader if row])
            else:
                real_idx = imag_idx = []
                data = np.array([[float(x) for x in first]] +
                                [[float(x) for x in row] for row in reader if row])

        if real_idx and imag_idx and len(real_idx) == len(imag_idx):
            # Named real-k / imag-k columns (e.g. the Intel respiratory format).
            csi = data[:, real_idx] + 1j * data[:, imag_idx]
        elif data.shape[1] % 2 == 0 and data.shape[1] >= 2 * 8:
            # Unnamed but even width → interleaved I/Q across the whole row.
            csi = data[:, 0::2] + 1j * data[:, 1::2]
        else:  # amplitude-only rows
            csi = data.astype(np.complex128)
        return csi


def _is_number(tok: str) -> bool:
    try:
        float(tok)
        return True
    except (TypeError, ValueError):
        return False


def _orient_time_subcarrier(csi: np.ndarray) -> np.ndarray:
    """Collapse >2D and orient as [time, subcarrier] (longer axis = time)."""
    csi = np.asarray(csi)
    while csi.ndim > 2:
        csi = csi[..., 0] if csi.shape[-1] < csi.shape[0] else csi.reshape(csi.shape[0], -1)
    if csi.ndim == 1:
        csi = csi[:, None]
    if csi.shape[0] < csi.shape[1]:
        csi = csi.T
    return csi


def describe(path: str) -> None:
    """Print what's inside the file so the right keys/columns can be chosen."""
    if str(path).lower().endswith(".csv"):
        import csv
        with open(path, newline="") as fh:
            reader = csv.reader(fh)
            first = next(reader)
            has_header = any(not _is_number(t) for t in first)
            n_rows = sum(1 for _ in reader) + (0 if has_header else 1)
        cols = first if has_header else [f"col{i}" for i in range(len(first))]
        real = [h for h in cols if str(h).strip().lower().startswith("real")]
        print(f"{path}: CSV cols={len(first)} rows={n_rows} header={has_header} "
              f"real/imag pairs={len(real)}")
    else:
        from v2.injector.adapters.csi_bench import describe_mat
        describe_mat(path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("usage: python -m v2.injector.adapters.intel_resp <file>")
        raise SystemExit(2)
    describe(sys.argv[1])
