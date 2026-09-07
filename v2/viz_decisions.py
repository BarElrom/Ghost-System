"""
GHOST v2 — visualize the cognitive pipeline's decisions.

Runs the same offline replay as ``main_v2.py --replay`` (dataset -> mock ESP32 ->
gateway -> cleaner -> features -> localizer -> agentic core) and plots the
resulting per-window decisions:

    1. Trajectory  — the (x, y) localizations in the room, in time order,
                     with the three receiver positions marked.
    2. Position    — x(t) and y(t) over the decision timeline.
    3. Motion      — velocity and signal-confidence over time.
    4. Node energy — per-receiver energy and breathing frequency over time.

Each point is ONE localization of the tracked object for one ~2 s window;
consecutive points are ``step / sample_rate`` seconds apart.

Usage (venv terminal):
    ./.venv/bin/python v2/viz_decisions.py --path example_csi.csv \
        --pos-scale embedded_wifi --window 100 --breath-window 300 --step 75
    # add --llm to route through Ollama; --show to open a window instead of only saving.

NOTE: with single-link datasets (e.g. example_csi.csv) node diversity is
synthesized, so positions are NON-METRIC (illustrative) — the plot says so.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib

from v2.config_v2 import DEFAULT_DATASET_PATH, NODE_POSITIONS, SAMPLE_RATE_HZ
from v2.ghost.main_v2 import add_log_flags, apply_log_flags, collect_replay


def _records(results) -> dict:
    """Flatten the (decision, features, raw_est, kalman_est) list into arrays.

    Three position/velocity series per window: ``raw_*`` (localizer, before
    smoothing), ``kal_*`` (Kalman-smoothed, fed to the agentic core), and the
    plain keys (the decision — after agentic reasoning).
    """
    rec = {"x": [], "y": [], "v": [], "conf": [], "breath": [], "energies": {},
           "raw_x": [], "raw_y": [], "raw_v": [],
           "kal_x": [], "kal_y": [], "kal_v": []}
    for decision, features, raw, kalman in results:
        d = decision.to_dict()
        rec["x"].append(d["coordinates"]["x_meters"])
        rec["y"].append(d["coordinates"]["y_meters"])
        rec["v"].append(d["velocity_m_s"])
        rec["conf"].append(d["signal_confidence"])
        rec["breath"].append(float(features.breathing_frequency))
        for name, val in d["raw_node_energies"].items():
            rec["energies"].setdefault(name, []).append(val)
        rec["raw_x"].append(float(raw.x_meters))
        rec["raw_y"].append(float(raw.y_meters))
        rec["raw_v"].append(float(raw.velocity_m_s))
        rec["kal_x"].append(float(kalman.x_meters))
        rec["kal_y"].append(float(kalman.y_meters))
        rec["kal_v"].append(float(kalman.velocity_m_s))
    return rec


def _plot(results, meta, out_path: str, show: bool) -> None:
    import matplotlib.pyplot as plt

    rec = _records(results)
    n = len(rec["x"])
    t = [i * meta["dt"] for i in range(n)]  # logical time (s) from window step

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    metric_note = "  (NON-METRIC — synthesized node diversity)" if meta["non_metric"] else ""
    kalman_on = meta.get("kalman", True)
    fig.suptitle(
        f"GHOST v2 decisions — {os.path.basename(meta['path'])}  "
        f"[{n} windows, {meta['dt']:.2f}s apart, llm={'on' if meta['llm'] else 'off'}, "
        f"kalman={'on' if kalman_on else 'off'}]{metric_note}",
        fontsize=12, fontweight="bold",
    )

    # 1) Trajectory: localizer (raw) -> Kalman (smoothed) -> decision (after
    #    agentic reasoning). This is the before/after-smoothing comparison.
    ax = axes[0][0]
    ax.plot(rec["raw_x"], rec["raw_y"], "x", color="#b8b8b8", markersize=6,
            zorder=1, label="localizer (raw)")
    ax.plot(rec["kal_x"], rec["kal_y"], "-", color="#2b7de0", linewidth=2,
            alpha=0.85, zorder=2, label="Kalman (smoothed)")
    sc = ax.scatter(rec["x"], rec["y"], c=t, cmap="viridis", s=80, zorder=3,
                    edgecolor="black", linewidth=0.5, label="decision (after agentic)")
    for name, (nx, ny) in NODE_POSITIONS.items():
        ax.plot(nx, ny, "^", color="crimson", markersize=12, zorder=4)
        ax.annotate(name, (nx, ny), textcoords="offset points", xytext=(6, 6),
                    color="crimson", fontweight="bold")
    ax.annotate("start", (rec["raw_x"][0], rec["raw_y"][0]), textcoords="offset points",
                xytext=(8, -12), fontsize=9)
    ax.set_title("1. Trajectory: localizer → Kalman → decision")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)  — depth")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("decision time (s)")

    # 2) Position components over time.
    ax = axes[0][1]
    ax.plot(t, rec["x"], "o-", label="x (m)", color="#2b7de0")
    ax.plot(t, rec["y"], "s-", label="y (m)", color="#0f9e8f")
    ax.set_title("2. Position vs time")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("metres")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 3) Velocity before/after smoothing (raw localizer vs Kalman) + confidence.
    ax = axes[1][0]
    ax.plot(t, rec["raw_v"], "o-", color="#e0a6a6", alpha=0.85,
            label="velocity (raw localizer)")
    ax.plot(t, rec["kal_v"], "s-", color="#d5602e", label="velocity (Kalman)")
    ax.set_ylabel("velocity (m/s)")
    ax.set_xlabel("time (s)")
    ax.set_title("3. Velocity: raw vs Kalman  & confidence")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(t, rec["conf"], "--", color="#7a6cf0", alpha=0.7, label="confidence")
    ax2.set_ylabel("confidence", color="#7a6cf0")
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis="y", labelcolor="#7a6cf0")

    # 4) Per-node energies + breathing (twin axes).
    ax = axes[1][1]
    for name, vals in rec["energies"].items():
        ax.plot(t, vals, "o-", label=name)
    ax.set_ylabel("node energy  Σ|H_dyn|")
    ax.set_xlabel("time (s)")
    ax.set_title("4. Per-receiver energy & breathing")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    ax3 = ax.twinx()
    ax3.plot(t, rec["breath"], "d--", color="#c98a1e", label="breathing")
    ax3.set_ylabel("breathing (Hz)", color="#c98a1e")
    ax3.tick_params(axis="y", labelcolor="#c98a1e")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=130)
    print(f"saved plot → {out_path}")
    if show:
        plt.show()


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot GHOST v2 replay decisions")
    ap.add_argument("--path", default=DEFAULT_DATASET_PATH, help="dataset file")
    ap.add_argument("--dataset", default="embedded_wifi", help="adapter name")
    ap.add_argument("--calib", type=int, default=200, help="calibration preamble frames")
    ap.add_argument("--frames", type=int, default=800, help="operational frames")
    ap.add_argument("--rate", type=float, default=0.0, help="replay pacing Hz (0 = fast)")
    ap.add_argument("--window", type=int, default=100, help="localization window (samples; 100 = 2 s @ 50 Hz)")
    ap.add_argument("--breath-window", type=int, default=300, dest="breath_window",
                    help="breathing window (samples; 300 = 6 s @ 50 Hz). 0 disables.")
    ap.add_argument("--step", type=int, default=75, help="window hop between decisions (75 = 1.5 s @ 50 Hz)")
    ap.add_argument("--pos-scale", default="embedded_wifi", dest="pos_scale",
                    help="POS_ENERGY_SCALES entry for the localizer")
    ap.add_argument("--metric", action="store_true",
                    help="treat positions as metric (default: illustrative/non-metric)")
    ap.add_argument("--llm", action="store_true", help="route decisions through Ollama")
    ap.add_argument("--no-kalman", action="store_true", dest="no_kalman",
                    help="disable Kalman smoothing (plot raw localizer as the track)")
    ap.add_argument("--out", default="ghost_v2_decisions.png", help="output PNG path")
    ap.add_argument("--show", action="store_true", help="open an interactive window too")
    add_log_flags(ap)
    args = ap.parse_args()
    apply_log_flags(args)

    if not args.show:
        matplotlib.use("Agg")  # headless-safe; only save the file

    results, meta = collect_replay(args)
    if results is None:
        print(f"  ERROR: {meta['error']}")
        return 1
    if not results:
        print("  ERROR: no decisions produced (window/step too large for the stream?)")
        return 1

    print(f"collected {len(results)} decision(s); sample_rate={SAMPLE_RATE_HZ} Hz")
    _plot(results, meta, args.out, args.show)
    return 0


if __name__ == "__main__":
    sys.exit(main())
