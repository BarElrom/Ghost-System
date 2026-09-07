
import argparse
import logging
import os
import sys
import threading
import time

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

logger = logging.getLogger("ghost.v2.pipeline")


def _configure_logging(level_name: str) -> None:
    """Route the ghost.v2 diagnostic logs to the console at the chosen level.

    Third-party libraries stay at WARNING; only the ghost.v2.* loggers follow
    ``level_name`` so the math is visible without httpx/urllib3 noise.
    """
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s.%(msecs)03d %(name)-24s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("ghost.v2").setLevel(getattr(logging, level_name))


def add_log_flags(ap: argparse.ArgumentParser) -> None:
    """Attach --log-level and -v/-vv to any runner's parser (shared across demos)."""
    ap.add_argument("--log-level", default="WARNING",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                    help="ghost.v2 diagnostic verbosity (INFO = per-stage summaries, "
                         "DEBUG = full per-term math)")
    ap.add_argument("-v", "--verbose", action="count", default=0,
                    help="-v = INFO, -vv = DEBUG (overrides --log-level)")


def apply_log_flags(args) -> None:
    """Resolve the log flags from parsed args and configure logging once."""
    level = args.log_level
    if args.verbose >= 2:
        level = "DEBUG"
    elif args.verbose == 1:
        level = "INFO"
    _configure_logging(level)

from dataclasses import replace

from v2.config_v2 import (
    CALIBRATION_SAMPLES,
    DEFAULT_POS_SCALE,
    KALMAN_ENABLED,
    SAMPLE_RATE_HZ,
)
from v2.ghost.signal_cleaner_v2 import SignalCleanerV2
from v2.ghost.feature_extractor_v2 import FeatureExtractorV2
from v2.ghost.localizer import Localizer
from v2.ghost.kalman import PositionKalman
from v2.ghost.agentic_core_v2 import AgenticCoreV2, DecisionV2


class GhostV2Pipeline:
    """Cleaner -> extractor -> localizer -> agentic core, as one object.

    Args:
        pos_scale: POS_ENERGY_SCALES name for the localizer (per-dataset tuning).
        non_metric: mark positions as illustrative (single-link datasets).
        use_llm: route decisions through Ollama; False = deterministic passthrough.
        sample_rate_hz: DSP frame rate.
        use_kalman: smooth the position track with a constant-velocity Kalman
            filter before reasoning (default: config KALMAN_ENABLED).
    """

    def __init__(self, pos_scale: str = DEFAULT_POS_SCALE, non_metric: bool = False,
                 use_llm: bool = False, sample_rate_hz: float = float(SAMPLE_RATE_HZ),
                 label_space=None, classifier_thresholds=None,
                 use_kalman: bool | None = None):
        self.cleaner = SignalCleanerV2()
        self.extractor = FeatureExtractorV2(sample_rate_hz=sample_rate_hz)
        self.localizer = Localizer(scale=pos_scale, non_metric=non_metric)
        self.core = AgenticCoreV2(use_llm=use_llm, label_space=label_space,
                                  classifier_thresholds=classifier_thresholds)
        enabled = KALMAN_ENABLED if use_kalman is None else bool(use_kalman)
        self.kalman = PositionKalman() if enabled else None
        self.previous_decision: DecisionV2 | None = None

    def calibrate(self, calib_slice: np.ndarray) -> None:
        """Build the empty-room baseline H_static from a calibration window."""
        self.cleaner.calibrate_from(calib_slice)

    def process(self, op_slice: np.ndarray, dt: float = 0.0,
                breath_slice: np.ndarray | None = None) -> tuple:
        """Run one operational window end to end.

        Args:
            op_slice: complex [R x 64 x T_short] window for localization.
            dt: seconds since the previous window (for velocity).
            breath_slice: optional longer complex window; its breathing_frequency
                replaces the short window's (Plan section 17 dual-window note).

        Returns:
            ``(decision, features, raw_estimate, smoothed_estimate)`` — the raw
            localizer estimate (before Kalman) and the Kalman-smoothed estimate
            that the agentic core reasoned over. When Kalman is disabled the two
            are the same object.
        """
        logger.debug(
            "── window: op_slice=%s dt=%.3fs breath_slice=%s",
            tuple(op_slice.shape), dt,
            None if breath_slice is None else tuple(breath_slice.shape),
        )
        cleaned = self.cleaner.clean(op_slice)
        features = self.extractor.extract(cleaned)

        if breath_slice is not None:
            # adapt=False: the breathing window is auxiliary; don't let it also
            # nudge the adaptive baseline (the op window already did, Note 7).
            breath_cleaned = self.cleaner.clean(breath_slice, adapt=False)
            breath_features = self.extractor.extract(breath_cleaned)
            features.breathing_frequency = breath_features.breathing_frequency

        raw_estimate = self.localizer.estimate(features, dt=dt)
        smoothed_estimate = self._apply_kalman(raw_estimate, dt)
        decision = self.core.reason(smoothed_estimate, features, self.previous_decision)
        self.previous_decision = decision
        return decision, features, raw_estimate, smoothed_estimate

    def _apply_kalman(self, estimate, dt: float):
        """Return a Kalman-smoothed copy of the estimate (or it unchanged)."""
        if self.kalman is None:
            return estimate
        xs, ys, speed = self.kalman.update(estimate.x_meters, estimate.y_meters, dt)
        unrealistic = speed > self.localizer.max_speed
        return replace(
            estimate,
            x_meters=xs, y_meters=ys, velocity_m_s=speed,
            unrealistic_speed=unrealistic,
            multipath_reflection_suspect=unrealistic or estimate.confidence < 0.2,
        )


def _print_decision(decision: DecisionV2, features) -> None:
    d = decision.to_dict()
    c = d["coordinates"]
    flags = d["anomaly_flags"]
    tags = [k for k, v in flags.items() if v]
    print(
        f"[{d['timestamp']:.2f}] "
        f"pos=({c['x_meters']:+.2f}, {c['y_meters']:.2f}) m  "
        f"v={d['velocity_m_s']:.2f} m/s  conf={d['signal_confidence']:.2f}  "
        f"breath={features.breathing_frequency:.3f}Hz  "
        f"energies={{" + ", ".join(f'{k}:{v:.0f}' for k, v in d['raw_node_energies'].items()) + "}"
        + (f"  ANOMALY[{','.join(tags)}]" if tags else "")
    )


def replay_to_window(args, verbose: bool = True) -> tuple:
    """Replay a dataset through the injector + UDP mock into one aligned window.

    Returns ``(window, meta)`` where ``window`` is the complex ``[R, 64, T]``
    cross-receiver matrix (calibration preamble + operational frames, ready to
    split at ``args.calib``) and ``meta`` carries per-node frame counts. On any
    setup failure ``window`` is ``None`` and ``meta["error"]`` explains why.

    This is the shared front half of the replay path: ``collect_replay`` runs the
    pipeline over the window it returns, and the slide exporter reuses the same
    window for the DSP-stage plots — so both see identical data from one run.
    """
    from v2.ghost.gateway_v2 import GatewayV2
    from v2.ghost.sources import UDPSource
    from v2.injector.injector import Injector, build_adapter, _NODE_NAMES

    if not os.path.exists(args.path):
        return None, {"error": f"dataset not found: {args.path}"}

    adapter = build_adapter(args.dataset, args.path)
    buffer_size = args.calib + args.frames + 50

    udp = UDPSource(bind_host="127.0.0.1", bind_port=0)
    gw = GatewayV2(sources=[udp], buffer_size=buffer_size)
    gw.start()

    inj = Injector(adapter, net_map={n: ("127.0.0.1", udp.port) for n in _NODE_NAMES},
                   rate_hz=args.rate, calibration_samples=args.calib)
    if verbose:
        print(f"Replaying '{args.path}' ({args.calib} calib + up to {args.frames} frames)...")
    t = threading.Thread(target=lambda: inj.run(max_frames=args.frames), daemon=True)
    t.start()
    t.join()

    prev, stable = -1, 0
    for _ in range(300):
        time.sleep(0.03)
        got = min(gw.get_matrix().get_receiver_count(r) for r in range(3))
        stable = stable + 1 if got == prev else 0
        prev = got
        if stable >= 3 and got > 0:
            break

    counts = [gw.get_matrix().get_receiver_count(r) for r in range(3)]
    total = min(counts)
    if verbose:
        print(f"Frames received per node: {counts}")
    if total <= args.calib:
        gw.stop()
        return None, {"error": "not enough frames to separate calibration from operational",
                      "counts": counts}

    aligned = gw.get_aligned_window(total, return_window=True)
    window = aligned.matrix
    total = aligned.num_frames
    if verbose and aligned.dropped:
        print(f"Aligned {total} cross-receiver frames "
              f"(dropped {aligned.dropped} incomplete; coverage={aligned.coverage})")
    gw.stop()
    if total <= args.calib:
        return None, {"error": "not enough aligned frames to separate calibration "
                               "from operational", "counts": counts, "aligned": total}

    return window, {"counts": counts, "total": total, "path": args.path}


def pipeline_over_window(window, args, label_space=None,
                         classifier_thresholds=None) -> tuple:
    """Run the sliding-window cognitive pipeline over an already-aligned window.

    Returns ``(results, meta)`` — the tail half of ``collect_replay``. ``window``
    is a complex ``[R, 64, T]`` matrix (as from ``replay_to_window``); the first
    ``args.calib`` frames calibrate the cleaner, the rest are localized.
    """
    pipe = GhostV2Pipeline(pos_scale=args.pos_scale, non_metric=not args.metric,
                           use_llm=args.llm, label_space=label_space,
                           classifier_thresholds=classifier_thresholds,
                           use_kalman=not getattr(args, "no_kalman", False))
    pipe.calibrate(window[:, :, : args.calib])
    op = window[:, :, args.calib:]

    win = args.window
    step = max(1, args.step)
    dt = step / float(SAMPLE_RATE_HZ)
    n_op = op.shape[2]

    results = []
    for i, start in enumerate(range(0, max(1, n_op - win + 1), step)):
        op_slice = op[:, :, start:start + win]
        breath_slice = None
        if args.breath_window and start + args.breath_window <= n_op:
            breath_slice = op[:, :, start:start + args.breath_window]
        decision, features, raw_est, smoothed_est = pipe.process(
            op_slice, dt=dt if i > 0 else 0.0, breath_slice=breath_slice)
        results.append((decision, features, raw_est, smoothed_est))

    meta = {"window": win, "step": step, "dt": dt, "n_op": n_op,
            "llm": bool(args.llm), "non_metric": not args.metric, "path": args.path,
            "pos_scale": args.pos_scale, "kalman": not getattr(args, "no_kalman", False)}
    return results, meta


def collect_replay(args, verbose: bool = True, label_space=None,
                   classifier_thresholds=None) -> tuple:
    """Replay a dataset through the injector + UDP mock and collect decisions.

    Runs the full pipeline over a sliding window and returns
    ``(results, meta)`` where ``results`` is a list of
    ``(DecisionV2, FeatureSet, raw_estimate, smoothed_estimate)`` tuples — one
    per window — and ``meta`` carries the run parameters. On any setup failure
    ``results`` is ``None`` and ``meta["error"]`` explains why.

    Used by ``run_replay`` (prints), ``viz_decisions`` (plots) and
    ``run_datasets_compare``. Thin wrapper over ``replay_to_window`` +
    ``pipeline_over_window``.
    """
    window, wmeta = replay_to_window(args, verbose=verbose)
    if window is None:
        return None, wmeta

    results, meta = pipeline_over_window(window, args, label_space=label_space,
                                         classifier_thresholds=classifier_thresholds)
    meta["counts"] = wmeta["counts"]
    return results, meta


def run_replay(args) -> int:
    """Replay a dataset through the full pipeline and print one decision per window."""
    results, meta = collect_replay(args)
    if results is None:
        print(f"  ERROR: {meta['error']}")
        return 1

    print(f"\n=== Decisions (window={meta['window']}, step={meta['step']}, "
          f"dt={meta['dt']:.2f}s, llm={'on' if meta['llm'] else 'off'}) ===")
    for decision, features, *_ in results:
        _print_decision(decision, features)

    print(f"\nReplay complete — {len(results)} decision(s) emitted.")

    if getattr(args, "plot", False):
        import matplotlib
        if not getattr(args, "show", False):
            matplotlib.use("Agg")  # headless-safe: only write the PNG
        from v2.viz_decisions import _plot
        _plot(results, meta, args.plot_out, getattr(args, "show", False))
    return 0


def run_serial(args) -> int:
    """Continuously localize from three ESP32 boards read over USB serial."""
    from v2.ghost.gateway_v2 import GatewayV2

    gw = GatewayV2.from_serial(args.ports, baud_rate=args.baud,
                               buffer_size=max(2000, args.breath_window + 200),
                               binary=not args.string_serial)
    gw.start()
    print(f"Reading {args.ports} @ {args.baud} baud — warming up "
          f"{args.calib} calibration frames...")

    pipe = GhostV2Pipeline(pos_scale=args.pos_scale, non_metric=not args.metric,
                           use_llm=args.llm,
                           use_kalman=not getattr(args, "no_kalman", False))

    deadline = time.time() + args.warmup_timeout
    while time.time() < deadline:
        if min(gw.get_matrix().get_receiver_count(r) for r in range(3)) >= args.calib:
            break
        time.sleep(0.1)
    counts = [gw.get_matrix().get_receiver_count(r) for r in range(3)]
    if min(counts) < args.calib:
        print(f"  ERROR: only {counts} frames after warmup; is the injector running?")
        gw.stop()
        return 1
    pipe.calibrate(gw.get_aligned_window(args.calib))
    print("Calibrated. Streaming decisions (Ctrl+C to stop)...\n")

    interval = args.interval
    try:
        while True:
            time.sleep(interval)
            need = max(args.window, args.breath_window)
            latest = gw.get_aligned_window(need)
            op_slice = latest[:, :, -args.window:]
            breath_slice = latest if args.breath_window else None
            decision, features, *_ = pipe.process(op_slice, dt=interval,
                                                  breath_slice=breath_slice)
            _print_decision(decision, features)
    except KeyboardInterrupt:
        print("\nstopping...")
    finally:
        gw.stop()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="GHOST v2 end-to-end cognitive pipeline")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--replay", dest="path", metavar="DATASET_PATH",
                      help="offline replay of a dataset file (software mock ESP32)")
    mode.add_argument("--serial", action="store_true",
                      help="live mode: read three ESP32 boards over USB serial")

    ap.add_argument("--ports", nargs=3, metavar=("RX1", "RX2", "RX3"),
                    help="serial ports for --serial mode (RX1 RX2 RX3 order)")
    ap.add_argument("--dataset", default="embedded_wifi", help="adapter name (replay)")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--string-serial", action="store_true", dest="string_serial",
                    help="read legacy ASCII CSI_DATA firmware instead of G2 binary "
                         "(default: binary)")
    ap.add_argument("--calib", type=int, default=int(CALIBRATION_SAMPLES),
                    help="calibration preamble length (frames)")
    ap.add_argument("--frames", type=int, default=800, help="operational frames (replay)")
    ap.add_argument("--rate", type=float, default=0.0, help="replay pacing Hz (0 = fast)")
    ap.add_argument("--window", type=int, default=100,
                    help="localization window (samples; 100 = 2 s @ 50 Hz)")
    ap.add_argument("--breath-window", type=int, default=500, dest="breath_window",
                    help="breathing window (samples; 500 = 10 s @ 50 Hz). 0 disables.")
    ap.add_argument("--step", type=int, default=50,
                    help="replay window hop (samples between decisions; 50 = 1 s @ 50 Hz)")
    ap.add_argument("--interval", type=float, default=0.5,
                    help="seconds between decisions in --serial mode")
    ap.add_argument("--warmup-timeout", type=float, default=30.0, dest="warmup_timeout")
    ap.add_argument("--pos-scale", default=DEFAULT_POS_SCALE, dest="pos_scale",
                    help="POS_ENERGY_SCALES entry for the localizer")
    ap.add_argument("--metric", action="store_true",
                    help="treat positions as metric (default: illustrative/non-metric)")
    ap.add_argument("--no-kalman", action="store_true", dest="no_kalman",
                    help="disable constant-velocity Kalman smoothing of the position track")
    ap.add_argument("--llm", action="store_true",
                    help="route decisions through Ollama (needs docker compose up -d)")
    ap.add_argument("--plot", action="store_true",
                    help="plot the decisions after a --replay run (saves a PNG)")
    ap.add_argument("--plot-out", default="ghost_v2_decisions.png", dest="plot_out",
                    help="PNG path for --plot")
    ap.add_argument("--show", action="store_true",
                    help="with --plot, also open an interactive window")
    add_log_flags(ap)
    args = ap.parse_args()
    apply_log_flags(args)

    if args.serial:
        if not args.ports:
            ap.error("--serial requires --ports RX1 RX2 RX3")
        return run_serial(args)
    if args.path:
        return run_replay(args)
    ap.error("choose a mode: --replay DATASET_PATH or --serial --ports ...")


if __name__ == "__main__":
    sys.exit(main())