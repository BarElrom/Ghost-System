"""Tiny formatting helpers for the ghost.v2 diagnostic logs.

Keeps 64-length subcarrier vectors and complex matrices from flooding the
console: everything collapses to compact magnitude statistics.
"""

import numpy as np


def mag(a) -> str:
    """Compact magnitude stats for a real/complex array: μ, range, shape."""
    a = np.asarray(a)
    m = np.abs(a)
    if m.size == 0:
        return f"empty shape={a.shape}"
    return f"μ={m.mean():.3f} [{m.min():.3f},{m.max():.3f}] shape={a.shape}"


def vec(a, p: int = 3) -> str:
    """Short fixed-precision list for small vectors (energies, variances)."""
    return "[" + ", ".join(f"{float(x):.{p}f}" for x in np.asarray(a).ravel()) + "]"