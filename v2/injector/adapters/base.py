"""
Dataset adapter contract.

A Snapshot is one time step of CSI. iq_by_stream maps a receiver-stream index
(0-based) to a complex64 (NUM_SUBCARRIERS,) array. Single-link datasets provide
just {0: iq}; the injector fans that out to the three physical nodes.
"""

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class Snapshot:
    """One time step of CSI, keyed by receiver-stream index."""

    iq_by_stream: dict          # {stream_index: complex64 (64,)}
    label: str | None = None


class DatasetAdapter:
    """Base class. Subclasses yield time-ordered Snapshots.

    Attributes:
        num_streams: number of independent receiver streams the dataset carries.
            1 = single-link (injector synthesizes node diversity, position is
            non-metric — Plan section 10.1). >= 3 maps directly to RX1/2/3.
        name: short adapter name.
    """

    num_streams: int = 1
    name: str = "adapter"

    def snapshots(self) -> Iterable[Snapshot]:
        raise NotImplementedError


class InMemoryAdapter(DatasetAdapter):
    """Adapter over an in-memory list of Snapshots (tests / programmatic use)."""

    def __init__(self, snaps, num_streams: int = 1, name: str = "memory"):
        self._snaps = list(snaps)
        self.num_streams = num_streams
        self.name = name

    def snapshots(self) -> Iterable[Snapshot]:
        return iter(self._snaps)


def normalize_subcarriers(iq: np.ndarray, target: int) -> np.ndarray:
    """Resize a complex subcarrier vector to `target` length.

    - equal length: unchanged
    - longer: truncate to the first `target` (ESP32 LLTF is the first 64)
    - shorter: linear-interpolate real and imag onto `target` points
    """
    iq = np.asarray(iq)
    n = iq.shape[0]
    if n == target:
        return iq.astype(np.complex64)
    if n > target:
        return iq[:target].astype(np.complex64)
    xp = np.linspace(0.0, 1.0, n)
    x = np.linspace(0.0, 1.0, target)
    re = np.interp(x, xp, iq.real)
    im = np.interp(x, xp, iq.imag)
    return (re + 1j * im).astype(np.complex64)
