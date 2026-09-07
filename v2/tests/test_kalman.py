"""Phase 5 tests — PositionKalman constant-velocity smoother (Note 8).

Covers seeding, constant-velocity speed estimation, jitter reduction on a static
target, and reset — the behaviour the pipeline relies on to stabilize the track
and read a steadier velocity than the localizer's frame-to-frame difference.

Run:  python v2/tests/test_kalman.py
"""

import sys

from _harness import Harness

import numpy as np

from v2.ghost.kalman import PositionKalman


def test_first_update_seeds(h: Harness) -> None:
    kf = PositionKalman()
    x, y, speed = kf.update(2.0, 3.0, dt=0.0)
    h.expect("first update returns the measurement x", x == 2.0)
    h.expect("first update returns the measurement y", y == 3.0)
    h.expect("first update reports zero speed", speed == 0.0)


def test_dt_zero_reseeds(h: Harness) -> None:
    kf = PositionKalman()
    kf.update(0.0, 0.0, dt=1.0)
    x, y, speed = kf.update(5.0, 5.0, dt=0.0)  # dt<=0 -> reseed, no velocity
    h.expect("dt<=0 reseeds to the measurement", (x, y) == (5.0, 5.0))
    h.expect("dt<=0 reports zero speed", speed == 0.0)


def test_constant_velocity_speed(h: Harness) -> None:
    # Noise-free constant-velocity motion: the filter's speed must converge to
    # the true |v|.
    kf = PositionKalman()
    vx, vy, dt = 0.5, 0.3, 1.0
    true_speed = float(np.hypot(vx, vy))
    speed = 0.0
    for step in range(20):
        speed = kf.update(vx * step * dt, vy * step * dt, dt if step else 0.0)[2]
    h.expect("velocity converges to the true speed",
             abs(speed - true_speed) < 0.05, f"speed={speed:.3f} true={true_speed:.3f}")


def test_smoothing_reduces_jitter(h: Harness) -> None:
    # Static target at (2, 3) observed with Gaussian noise. Smoothed positions
    # should sit far closer to the truth than the raw measurements. Uses a
    # smoothing-favorable process var (the mechanism); the shipped default trades
    # some smoothing for responsiveness to real motion.
    rng = np.random.default_rng(7)
    truth = np.array([2.0, 3.0])
    kf = PositionKalman(process_var=0.05, measurement_var=0.25)
    raw_err, smooth_err = [], []
    for step in range(60):
        meas = truth + rng.normal(0.0, 0.4, size=2)
        sx, sy, _ = kf.update(meas[0], meas[1], 1.0 if step else 0.0)
        if step >= 20:  # measure at steady state
            raw_err.append(float(np.sum((meas - truth) ** 2)))
            smooth_err.append(float((sx - truth[0]) ** 2 + (sy - truth[1]) ** 2))
    raw_mse, smooth_mse = float(np.mean(raw_err)), float(np.mean(smooth_err))
    h.expect("Kalman cuts position MSE vs raw measurements",
             smooth_mse < 0.5 * raw_mse, f"smooth={smooth_mse:.4f} raw={raw_mse:.4f}")


def test_reset(h: Harness) -> None:
    kf = PositionKalman()
    kf.update(0.0, 0.0, 0.0)
    kf.update(1.0, 1.0, 1.0)
    kf.reset()
    x, y, speed = kf.update(9.0, 9.0, 1.0)  # first call after reset -> reseed
    h.expect("reset makes the next update reseed", (x, y, speed) == (9.0, 9.0, 0.0))


def main() -> int:
    h = Harness("kalman (constant-velocity smoother)")
    h.case("first_update_seeds", test_first_update_seeds)
    h.case("dt_zero_reseeds", test_dt_zero_reseeds)
    h.case("constant_velocity_speed", test_constant_velocity_speed)
    h.case("smoothing_reduces_jitter", test_smoothing_reduces_jitter)
    h.case("reset", test_reset)
    return h.done()


if __name__ == "__main__":
    sys.exit(main())
