

import logging

import numpy as np

from v2.config_v2 import NUM_SUBCARRIERS
from v2.injector.adapters.base import DatasetAdapter, Snapshot, normalize_subcarriers

logger = logging.getLogger("ghost.v2.adapter.csi_bench")

_AMP_KEYS = ["amplitude", "amp", "csi_amp", "csi_amplitude", "abs", "mag"]
_PHASE_KEYS = ["phase", "csi_phase", "pha", "angle", "ang"]
_COMPLEX_KEYS = ["csi", "csi_complex", "csi_data", "h", "csi_matrix", "data"]


def load_mat(path: str) -> dict:
    """Load a .mat into {name: ndarray}. Handles v5 (scipy) and v7.3 (h5py)."""
    try:
        from scipy.io import loadmat
        raw = loadmat(path)
        return {k: np.asarray(v) for k, v in raw.items() if not k.startswith("__")}
    except NotImplementedError:
        try:
            import h5py
        except ImportError as e:
            raise RuntimeError(
                "This .mat is v7.3 (HDF5); install h5py to read it: pip install h5py"
            ) from e
        out = {}
        with h5py.File(path, "r") as f:
            for key in f.keys():
                out[key] = np.asarray(f[key]).T
        return out


def describe_mat(path: str) -> dict:
    """Print the variables (name, shape, dtype) in a .mat — use to pick keys."""
    data = load_mat(path)
    print(f"{path}:")
    for name, arr in data.items():
        arr = np.asarray(arr)
        kind = "complex" if np.iscomplexobj(arr) else str(arr.dtype)
        print(f"  {name:20s} shape={arr.shape} dtype={kind}")
    return data


def _find_key(data: dict, candidates: list) -> str | None:
    """Case-insensitive lookup of the first candidate name present in data."""
    lower = {k.lower(): k for k in data}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


class CSIBenchAdapter(DatasetAdapter):
    """Yields single-link Snapshots from a CSI-Bench .mat file.

    Args:
        path: .mat file path.
        amp_key / phase_key: override the amplitude/phase variable names.
        complex_key: override the complex-CSI variable name (used if it exists
            and is complex; takes precedence over amp/phase).
        time_axis: 0 or 1 to force which axis is time (default: auto = longer axis).
        max_frames: optional cap on frames.
        label: optional activity label attached to every Snapshot.
    """

    num_streams = 1
    name = "csi_bench"

    def __init__(
        self,
        path: str,
        amp_key: str | None = None,
        phase_key: str | None = None,
        complex_key: str | None = None,
        time_axis: int | None = None,
        max_frames: int | None = None,
        label: str | None = None,
    ):
        self.path = path
        self.amp_key = amp_key
        self.phase_key = phase_key
        self.complex_key = complex_key
        self.time_axis = time_axis
        self.max_frames = max_frames
        self.label = label

    def snapshots(self):
        csi = self._extract_complex()
        n_time = csi.shape[0]
        count = 0
        for t in range(n_time):
            iq = normalize_subcarriers(csi[t], NUM_SUBCARRIERS)
            yield Snapshot(iq_by_stream={0: iq}, label=self.label)
            count += 1
            if self.max_frames is not None and count >= self.max_frames:
                break
        logger.info("%s: %d frames from %s", self.name, count, self.path)

    def _extract_complex(self) -> np.ndarray:
        """Load the file and return complex CSI oriented as [time, subcarrier]."""
        data = load_mat(self.path)
        if not data:
            raise ValueError(f"no variables found in {self.path}")

        ckey = self.complex_key or _find_key(data, _COMPLEX_KEYS)
        if ckey is not None and np.iscomplexobj(np.asarray(data[ckey])):
            csi = np.asarray(data[ckey])
        else:
            akey = self.amp_key or _find_key(data, _AMP_KEYS)
            pkey = self.phase_key or _find_key(data, _PHASE_KEYS)
            if akey is None or pkey is None:
                raise ValueError(
                    f"could not find complex CSI or amplitude+phase in {list(data)}; "
                    f"pass amp_key/phase_key (or complex_key). "
                    f"Run `python -m v2.injector.adapters.csi_bench {self.path}` to inspect."
                )
            amp = np.asarray(data[akey], dtype=np.float64)
            phase = np.asarray(data[pkey], dtype=np.float64)
            if amp.shape != phase.shape:
                raise ValueError(f"amplitude {amp.shape} and phase {phase.shape} shapes differ")
            csi = amp * np.exp(1j * phase)

        csi = self._to_2d(csi)
        csi = self._orient_time_first(csi)
        return csi.astype(np.complex64)

    def _to_2d(self, csi: np.ndarray) -> np.ndarray:
        """Collapse >2D arrays to [A, B] by taking the first index of the
        smallest axis (typically the antenna/link dimension)."""
        csi = np.asarray(csi)
        while csi.ndim > 2:
            drop = int(np.argmin(csi.shape))
            logger.warning(
                "%s: array is %dD %s; taking index 0 of axis %d (assumed antenna/link)",
                self.name, csi.ndim, csi.shape, drop,
            )
            csi = np.take(csi, 0, axis=drop)
        if csi.ndim == 1:
            csi = csi[:, None]
        return csi

    def _orient_time_first(self, csi: np.ndarray) -> np.ndarray:
        """Return [time, subcarrier]. Auto: the longer axis is time."""
        if self.time_axis == 1:
            return csi.T
        if self.time_axis == 0:
            return csi
        return csi if csi.shape[0] >= csi.shape[1] else csi.T


def _main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Inspect a CSI-Bench .mat file")
    ap.add_argument("path", help=".mat file")
    args = ap.parse_args()
    describe_mat(args.path)


if __name__ == "__main__":
    _main()
