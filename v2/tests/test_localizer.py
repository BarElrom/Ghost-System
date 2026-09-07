"""Phase 4 tests — localizer (position / velocity / anomaly).

Validates the Plan section 8 estimator: energy-weighted lateral centroid, the
depth heuristic and its clamps, confidence mapping, velocity across successive
estimates (with the section-16 square fix), and the anomaly flags.

Run:  python v2/tests/test_localizer.py
"""

import sys

from _harness import Harness

from v2.config_v2 import NODE_POSITIONS
from v2.ghost.localizer import Localizer, LocationEstimate


_SCALE = {"y_depth_divisor": 10.0, "y_max": 6.0, "y_min": 0.5, "confidence_divisor": 100.0}


def _loc(**kw) -> Localizer:
    kw.setdefault("scale", _SCALE)
    return Localizer(**kw)


def test_empty_room(h: Harness) -> None:
    est = _loc().estimate({"RX1": 0.0, "RX2": 0.0, "RX3": 0.0})
    h.expect("returns LocationEstimate", isinstance(est, LocationEstimate))
    h.expect("x = 0 when no energy", est.x_meters == 0.0)
    h.expect("y clamps to y_max", est.y_meters == 6.0)
    h.expect("confidence 0", est.confidence == 0.0)
    h.expect("velocity 0 on first call", est.velocity_m_s == 0.0)
    h.expect("low confidence -> multipath flag", est.multipath_reflection_suspect is True)
    h.expect("no unrealistic speed", est.unrealistic_speed is False)


def test_lateral_centroid(h: Harness) -> None:
    est = _loc().estimate({"RX1": 0.0, "RX2": 0.0, "RX3": 50.0})
    h.expect("centroid at RX3 x=+2", abs(est.x_meters - 2.0) < 1e-9)

    est = _loc().estimate({"RX1": 30.0, "RX2": 0.0, "RX3": 0.0})
    h.expect("centroid at RX1 x=-1", abs(est.x_meters - (-1.0)) < 1e-9)


def test_symmetric_target_collapses(h: Harness) -> None:
    est = _loc().estimate({"RX1": 20.0, "RX2": 20.0, "RX3": 0.0})
    h.expect("symmetric energy -> x ~ 0", abs(est.x_meters) < 1e-9)


def test_depth_and_confidence(h: Harness) -> None:
    est = _loc().estimate({"RX1": 10.0, "RX2": 10.0, "RX3": 0.0})
    h.expect("depth 6 - E/div", abs(est.y_meters - 4.0) < 1e-9)
    h.expect("confidence E/div", abs(est.confidence - 0.2) < 1e-9)

    est = _loc().estimate({"RX1": 500.0, "RX2": 500.0, "RX3": 500.0})
    h.expect("depth clamps to y_min", est.y_meters == 0.5)
    h.expect("confidence clamps to 1.0", est.confidence == 1.0)


def test_velocity_uses_squares(h: Harness) -> None:
    loc = _loc()
    loc.estimate({"RX1": 10.0, "RX2": 10.0, "RX3": 0.0}, dt=0.0)
    est = loc.estimate({"RX1": 0.0, "RX2": 0.0, "RX3": 20.0}, dt=1.0)
    h.expect("velocity = sqrt(dx^2+dy^2)/dt", abs(est.velocity_m_s - 2.0) < 1e-9)


def test_velocity_zero_when_dt_nonpositive(h: Harness) -> None:
    loc = _loc()
    loc.estimate({"RX1": 10.0, "RX2": 10.0, "RX3": 0.0}, dt=1.0)
    est = loc.estimate({"RX1": 0.0, "RX2": 0.0, "RX3": 20.0}, dt=0.0)
    h.expect("velocity 0 when dt=0", est.velocity_m_s == 0.0)


def test_unrealistic_speed_flag(h: Harness) -> None:
    loc = _loc()
    loc.estimate({"RX1": 30.0, "RX2": 0.0, "RX3": 0.0}, dt=1.0)
    est = loc.estimate({"RX1": 0.0, "RX2": 0.0, "RX3": 30.0}, dt=0.1)
    h.expect("unrealistic speed flagged", est.unrealistic_speed is True)
    h.expect("unrealistic -> multipath suspect", est.multipath_reflection_suspect is True)


def test_reset_clears_previous(h: Harness) -> None:
    loc = _loc()
    loc.estimate({"RX1": 30.0, "RX2": 0.0, "RX3": 0.0}, dt=1.0)
    loc.reset()
    est = loc.estimate({"RX1": 0.0, "RX2": 0.0, "RX3": 30.0}, dt=1.0)
    h.expect("velocity 0 after reset", est.velocity_m_s == 0.0)


def test_accepts_feature_set_object(h: Harness) -> None:
    class _FS:
        node_energies = {"RX1": 10.0, "RX2": 10.0, "RX3": 0.0}

    est = _loc().estimate(_FS())
    h.expect("reads .node_energies", abs(est.y_meters - 4.0) < 1e-9)


def test_non_metric_flag_and_to_dict(h: Harness) -> None:
    est = _loc(non_metric=True).estimate({"RX1": 10.0, "RX2": 10.0, "RX3": 0.0})
    d = est.to_dict()
    h.expect("non_metric propagates", est.non_metric is True)
    h.expect("to_dict has coordinates", "coordinates" in d and "x_meters" in d["coordinates"])
    h.expect("to_dict velocity", "velocity_m_s" in d)
    h.expect("to_dict confidence", d["signal_confidence"] == est.confidence)
    h.expect("to_dict anomaly non_metric", d["anomaly_flags"]["non_metric_position"] is True)
    h.expect("to_dict raw energies", d["raw_node_energies"]["RX1"] == 10.0)


def test_default_config_positions(h: Harness) -> None:
    loc = Localizer(scale=_SCALE)
    est = loc.estimate({"RX1": 0.0, "RX2": 0.0, "RX3": 50.0})
    h.expect("uses config NODE_POSITIONS", abs(est.x_meters - NODE_POSITIONS["RX3"][0]) < 1e-9)


def test_bad_scale_name_raises(h: Harness) -> None:
    h.expect_raises("unknown scale name raises", KeyError,
                    lambda: Localizer(scale="does_not_exist"))


def test_bad_feature_input_raises(h: Harness) -> None:
    h.expect_raises("non-dict energies raise", TypeError,
                    lambda: _loc().estimate(object()))


def main() -> int:
    h = Harness("localizer")
    h.case("empty_room", test_empty_room)
    h.case("lateral_centroid", test_lateral_centroid)
    h.case("symmetric_target_collapses", test_symmetric_target_collapses)
    h.case("depth_and_confidence", test_depth_and_confidence)
    h.case("velocity_uses_squares", test_velocity_uses_squares)
    h.case("velocity_zero_when_dt_nonpositive", test_velocity_zero_when_dt_nonpositive)
    h.case("unrealistic_speed_flag", test_unrealistic_speed_flag)
    h.case("reset_clears_previous", test_reset_clears_previous)
    h.case("accepts_feature_set_object", test_accepts_feature_set_object)
    h.case("non_metric_flag_and_to_dict", test_non_metric_flag_and_to_dict)
    h.case("default_config_positions", test_default_config_positions)
    h.case("bad_scale_name_raises", test_bad_scale_name_raises)
    h.case("bad_feature_input_raises", test_bad_feature_input_raises)
    return h.done()


if __name__ == "__main__":
    sys.exit(main())