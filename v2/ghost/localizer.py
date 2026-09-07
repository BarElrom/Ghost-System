
import logging
from dataclasses import dataclass, field

from v2.config_v2 import (
    DEFAULT_POS_SCALE,
    MAX_PLAUSIBLE_SPEED_M_S,
    NODE_POSITIONS,
    POS_ENERGY_SCALES,
)

logger = logging.getLogger("ghost.v2.localizer")


@dataclass
class LocationEstimate:
    """One localizer output for a single cleaned window."""

    x_meters: float
    y_meters: float
    velocity_m_s: float
    confidence: float
    node_energies: dict
    unrealistic_speed: bool = False
    multipath_reflection_suspect: bool = False
    non_metric: bool = False

    def to_dict(self) -> dict:
        """Coordinates/velocity schema (Plan section 9), minus the timestamp,
        which the orchestrator / LLM layer stamps."""
        return {
            "coordinates": {
                "x_meters": self.x_meters,
                "y_meters": self.y_meters,
            },
            "velocity_m_s": self.velocity_m_s,
            "signal_confidence": self.confidence,
            "anomaly_flags": {
                "unrealistic_speed": self.unrealistic_speed,
                "multipath_reflection_suspected": self.multipath_reflection_suspect,
                "non_metric_position": self.non_metric,
            },
            "raw_node_energies": dict(self.node_energies),
        }


class Localizer:
    """Energy-weighted position + velocity estimator (Plan section 8).

    Holds the previous estimate internally so velocity can be computed across
    successive calls to estimate().

    Args:
        node_positions: {node_name: (x, y)} in metres. Defaults to config
            NODE_POSITIONS.
        scale: name of a POS_ENERGY_SCALES entry, OR a dict with the four scale
            keys. Defaults to config DEFAULT_POS_SCALE. See config_v2 for why
            this must be tuned per dataset.
        max_speed: anomaly threshold in m/s.
        non_metric: mark estimates as non-metric (single-link / synthesized node
            diversity — Plan section 10.1).
    """

    def __init__(
        self,
        node_positions: dict = None,
        scale=None,
        max_speed: float = float(MAX_PLAUSIBLE_SPEED_M_S),
        non_metric: bool = False,
    ):
        self.node_positions = dict(node_positions or NODE_POSITIONS)
        self.scale = self._resolve_scale(scale)
        self.max_speed = float(max_speed)
        self.non_metric = bool(non_metric)
        self._prev = None

    @staticmethod
    def _resolve_scale(scale) -> dict:
        """Accept a scale-set name, an explicit dict, or None (config default)."""
        if scale is None:
            scale = DEFAULT_POS_SCALE
        if isinstance(scale, str):
            if scale not in POS_ENERGY_SCALES:
                raise KeyError(
                    f"unknown POS_ENERGY_SCALES entry {scale!r}; "
                    f"known: {sorted(POS_ENERGY_SCALES)}"
                )
            return dict(POS_ENERGY_SCALES[scale])
        required = {"y_depth_divisor", "y_max", "y_min", "confidence_divisor"}
        missing = required - set(scale)
        if missing:
            raise ValueError(f"scale dict missing keys: {sorted(missing)}")
        return dict(scale)

    def reset(self) -> None:
        """Forget the previous estimate (next velocity will be 0)."""
        self._prev = None

    def estimate(self, feature_set, dt: float = 0.0) -> LocationEstimate:
        """Localize one window.

        Args:
            feature_set: object with a .node_energies dict (FeatureSet), or a
                plain dict of node energies.
            dt: seconds since the previous estimate, for velocity. Use 0 (or the
                first call) to suppress velocity.
        """
        energies = self._energies(feature_set)
        e_total = float(sum(energies.values()))

        logger.debug("estimate: energies=%s E_total=%.3f scale=%s dt=%.3fs",
                     {k: round(v, 2) for k, v in energies.items()}, e_total, self.scale, dt)

        x_est = self._lateral_centroid(energies, e_total)
        y_est = self._depth(e_total)
        conf_div = self.scale["confidence_divisor"]
        confidence = min(1.0, e_total / conf_div) if conf_div > 0 else 0.0
        logger.debug("  confidence: min(1, E_total/conf_div) = min(1, %.3f/%.1f) = %.3f%s",
                     e_total, conf_div, confidence,
                     "  (saturated at 1.0)" if conf_div > 0 and e_total >= conf_div else "")

        velocity = self._velocity(x_est, y_est, dt)
        self._prev = (x_est, y_est)

        unrealistic = velocity > self.max_speed
        multipath = unrealistic or confidence < 0.2

        est = LocationEstimate(
            x_meters=x_est,
            y_meters=y_est,
            velocity_m_s=velocity,
            confidence=confidence,
            node_energies=energies,
            unrealistic_speed=unrealistic,
            multipath_reflection_suspect=multipath,
            non_metric=self.non_metric,
        )
        logger.info(
            "Estimate: x=%.2f y=%.2f v=%.2f conf=%.2f E_total=%.1f%s%s",
            x_est, y_est, velocity, confidence, e_total,
            " [unrealistic_speed]" if unrealistic else "",
            " [multipath?]" if multipath and not unrealistic else "",
        )
        return est

    @staticmethod
    def _energies(feature_set) -> dict:
        raw = getattr(feature_set, "node_energies", feature_set)
        if not isinstance(raw, dict):
            raise TypeError("feature_set must expose a node_energies dict")
        return {k: float(v) for k, v in raw.items()}

    def _lateral_centroid(self, energies: dict, e_total: float) -> float:
        """Energy-weighted mean of node X positions (0 if no energy)."""
        if e_total <= 0.0:
            return 0.0
        weighted = 0.0
        terms = []
        for name, energy in energies.items():
            pos = self.node_positions.get(name)
            if pos is None:
                continue
            weighted += pos[0] * energy
            terms.append(f"{name}: X={pos[0]:+.1f}·E={energy:.1f}={pos[0] * energy:+.1f}")
        x = weighted / e_total
        logger.debug("  x (centroid): Σ(X·E)/E_total = (%s) / %.1f = %+.3f m",
                     " + ".join(terms), e_total, x)
        return x

    def _depth(self, e_total: float) -> float:
        """Inverse-backscatter depth heuristic, clamped to [y_min, y_max]."""
        s = self.scale
        raw = s["y_max"] - e_total / s["y_depth_divisor"]
        clamped = max(s["y_min"], min(s["y_max"], raw))
        logger.debug("  y (depth): y_max − E_total/y_depth_div = %.1f − %.1f/%.1f = %.3f → %.3f m%s",
                     s["y_max"], e_total, s["y_depth_divisor"], raw, clamped,
                     "  (clamped)" if clamped != raw else "")
        return clamped

    def _velocity(self, x: float, y: float, dt: float) -> float:
        """Speed between this and the previous estimate (0 on first call / dt<=0).

        Uses sqrt(dx**2 + dy**2) — the Plan section 16 fix for the ``dx*2 + dy*2``
        bug in the shared snippet.
        """
        if self._prev is None or dt <= 0.0:
            logger.debug("  velocity: 0.0 (first estimate or dt<=0)")
            return 0.0
        dx = x - self._prev[0]
        dy = y - self._prev[1]
        distance = (dx ** 2 + dy ** 2) ** 0.5
        v = distance / dt
        logger.debug("  velocity: √(dx²+dy²)/dt = √(%+.3f²+%+.3f²)/%.3f = %.3f/%.3f = %.3f m/s",
                     dx, dy, dt, distance, dt, v)
        return v