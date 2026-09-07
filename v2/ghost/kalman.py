"""
GHOST v2 — constant-velocity Kalman smoother for localizer output (Note 8).

The localizer emits one ``(x, y)`` per window; consecutive windows jitter, and its
velocity is a raw frame-to-frame difference that spikes on that jitter (tripping
the ``unrealistic_speed`` flag). This filter models the target as moving with a
roughly constant velocity and fuses the noisy measurements into a smooth track,
reading velocity straight from the state.

State is ``[x, vx, y, vy]`` (position + velocity per axis); the measurement is the
localizer's ``(x, y)``. Built on ``filterpy`` (Note 4: use existing packages).
"""

import logging

import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag

from v2.config_v2 import KALMAN_MEASUREMENT_VAR, KALMAN_PROCESS_VAR

logger = logging.getLogger("ghost.v2.kalman")

_INITIAL_VAR = 500.0  # large initial covariance: trust the first measurement little


class PositionKalman:
    """Constant-velocity Kalman filter over a 2-D position stream.

    Args:
        process_var: acceleration process noise (m²/s⁴-ish); larger = trust the
            measurements more / track faster maneuvers.
        measurement_var: localizer position-noise variance (m²); larger = smoother
            but laggier.
    """

    def __init__(
        self,
        process_var: float = KALMAN_PROCESS_VAR,
        measurement_var: float = KALMAN_MEASUREMENT_VAR,
    ):
        self.process_var = float(process_var)
        self.measurement_var = float(measurement_var)
        self._kf = KalmanFilter(dim_x=4, dim_z=2)
        # measure x (state 0) and y (state 2); velocities are hidden.
        self._kf.H = np.array([[1.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 1.0, 0.0]])
        self._kf.R = np.eye(2) * self.measurement_var
        self._kf.P = np.eye(4) * _INITIAL_VAR
        self._initialized = False

    def reset(self) -> None:
        """Forget all state; the next update re-seeds from its measurement."""
        self._initialized = False
        self._kf.P = np.eye(4) * _INITIAL_VAR

    def update(self, x: float, y: float, dt: float) -> tuple[float, float, float]:
        """Fuse one measurement; return the smoothed ``(x, y, speed)``.

        The first call (or any call with ``dt <= 0``) seeds the state to the
        measurement and reports zero speed — there is no time base for velocity
        yet. Subsequent calls run the predict/update cycle with a time-varying
        transition built from ``dt``.
        """
        x, y = float(x), float(y)
        if not self._initialized or dt <= 0.0:
            self._kf.x = np.array([x, 0.0, y, 0.0])
            self._initialized = True
            logger.debug("kalman: seed state to (%.3f, %.3f), v=0", x, y)
            return x, y, 0.0

        transition = np.array([[1.0, dt], [0.0, 1.0]])
        self._kf.F = block_diag(transition, transition)
        q = Q_discrete_white_noise(dim=2, dt=dt, var=self.process_var)
        self._kf.Q = block_diag(q, q)

        self._kf.predict()
        self._kf.update(np.array([x, y]))

        sx, svx, sy, svy = (float(v) for v in self._kf.x)
        speed = float(np.hypot(svx, svy))
        logger.debug("kalman: meas=(%.3f,%.3f) → smooth=(%.3f,%.3f) v=%.3f (dt=%.3f)",
                     x, y, sx, sy, speed, dt)
        return sx, sy, speed
