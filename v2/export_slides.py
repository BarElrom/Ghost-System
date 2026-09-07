"""
GHOST v2 — export slide-ready figures from one replay run.

Runs the offline pipeline once (dataset -> mock ESP32 -> gateway -> cleaner ->
features -> localizer -> Kalman -> agentic core) and writes a set of individual,
presentation-quality PNGs — each isolating ONE before/after story you can drop
straight onto a slide:

    01_signal_cleaning.png     static subtraction + band-pass (raw vs cleaned)
    02_kalman_before_after.png localizer track vs Kalman-smoothed track + velocity
    03_breathing_spectrum.png  breathing-band spectrum with the detected peak
                               (or an honest "no peak" for empty/still)
    04_feature_timeline.png    motion / doppler / breathing over time
    05_trajectory.png          final localization trajectory, coloured by time
    06_agentic_before_after.png position/velocity fed INTO vs OUT OF the AI layer

Each figure is a standalone file (not a dashboard) so it composes cleanly in a
deck. One injector run feeds every plot, so they all describe the same data.

Usage (venv terminal):
    ./.venv/bin/python v2/export_slides.py                     # bundled example_csi.csv
    ./.venv/bin/python v2/export_slides.py --dataset intel_resp \
        --path datasets/intel_resp/CSI_complex1_aligned_fixedPiOffset_20MHz_OneSittingOneWalking.csv \
        --pos-scale intel_resp --window 200 --breath-window 1000 --step 200 \
        --frames 2000 --calib 200 --outdir slides_walking
    # add --llm to route decisions through Ollama (start it first: docker compose up -d)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib

from v2.config_v2 import DEFAULT_DATASET_PATH, NODE_POSITIONS, SAMPLE_RATE_HZ
from v2.ghost.signal_cleaner_v2 import SignalCleanerV2
from v2.ghost.feature_extractor_v2 import FeatureExtractorV2
from v2.ghost.main_v2 import (add_log_flags, apply_log_flags, pipeline_over_window,
                              replay_to_window)
from v2.viz_decisions import _records

# Consistent, readable-from-the-back-of-the-room styling for every figure.
_STYLE = {
    "figure.dpi": 150,
    "font.size": 13,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "savefig.bbox": "tight",
}
_RAW = "#b0b0b0"
_ACCENT = "#2b7de0"
_ACCENT2 = "#d5602e"
_GREEN = "#0f9e8f"
_PURPLE = "#7a6cf0"


def _save(fig, outdir: str, name: str, saved: list) -> None:
    import matplotlib.pyplot as plt
    path = os.path.join(outdir, name)
    fig.savefig(path)
    plt.close(fig)
    saved.append(path)
    print(f"  saved {path}")


def _tag(meta: dict) -> str:
    """Common subtitle: dataset + non-metric caveat."""
    note = "  (positions NON-METRIC — synthesized node diversity)" if meta["non_metric"] else ""
    return f"{os.path.basename(meta['path'])}{note}"


def fig_signal_cleaning(window, args, outdir, saved) -> None:
    """Raw vs cleaned amplitude for RX1 — the static-subtraction + band-pass step."""
    import matplotlib.pyplot as plt

    cleaner = SignalCleanerV2()
    cleaner.calibrate_from(window[:, :, : args.calib])
    op = window[:, :, args.calib:]
    cleaned = cleaner.clean(op)

    raw_amp = np.abs(op)                      # [R, 64, T]  (still carries the static baseline)
    clean_amp = np.asarray(cleaned.amplitude)  # [R, 64, T]  (baseline removed + band-passed)
    t = np.arange(op.shape[2]) / float(SAMPLE_RATE_HZ)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig.suptitle("Signal cleaning — static subtraction + band-pass (RX1)",
                 fontsize=17, fontweight="bold")

    ax1.plot(t, raw_amp[0].mean(axis=0), color=_RAW, linewidth=1.5)
    ax1.set_title("Before: raw amplitude (dominated by the static room baseline)")
    ax1.set_ylabel("mean |H|  (a.u.)")

    ax2.plot(t, clean_amp[0].mean(axis=0), color=_ACCENT, linewidth=1.5)
    ax2.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
    ax2.set_title("After: baseline removed + 0.1–4 Hz band-pass (motion revealed)")
    ax2.set_ylabel("mean |H_dyn|  (a.u.)")
    ax2.set_xlabel("time (s)")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, outdir, "01_signal_cleaning.png", saved)


def fig_kalman(results, meta, outdir, saved) -> None:
    """Before/after Kalman: raw localizer track vs smoothed track + velocity."""
    import matplotlib.pyplot as plt

    rec = _records(results)
    n = len(rec["x"])
    t = [i * meta["dt"] for i in range(n)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Kalman smoothing — before vs after   [{_tag(meta)}]",
                 fontsize=16, fontweight="bold")

    # (a) trajectory
    ax1.plot(rec["raw_x"], rec["raw_y"], "x", color=_RAW, markersize=7,
             label="localizer (raw)")
    ax1.plot(rec["kal_x"], rec["kal_y"], "-o", color=_ACCENT, linewidth=2,
             markersize=4, alpha=0.9, label="Kalman (smoothed)")
    for name, (nx, ny) in NODE_POSITIONS.items():
        ax1.plot(nx, ny, "^", color="crimson", markersize=13)
        ax1.annotate(name, (nx, ny), textcoords="offset points", xytext=(6, 6),
                     color="crimson", fontweight="bold")
    ax1.set_title("Trajectory")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m) — depth")
    ax1.legend(loc="best")

    # (b) velocity
    ax2.plot(t, rec["raw_v"], "o-", color="#e0a6a6", alpha=0.9,
             label="velocity (raw localizer)")
    ax2.plot(t, rec["kal_v"], "s-", color=_ACCENT2, label="velocity (Kalman)")
    ax2.set_title("Velocity")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("velocity (m/s)")
    ax2.legend(loc="best")

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, outdir, "02_kalman_before_after.png", saved)


def fig_breathing_spectrum(window, args, meta, outdir, saved) -> None:
    """Breathing-band power spectrum with the detected peak (or honest 'no peak')."""
    import matplotlib.pyplot as plt

    cleaner = SignalCleanerV2()
    cleaner.calibrate_from(window[:, :, : args.calib])
    op = window[:, :, args.calib:]
    # Use the longest available window for the best frequency resolution (fs/T).
    n = op.shape[2]
    if args.breath_window and args.breath_window <= n:
        n = args.breath_window
    cleaned = cleaner.clean(op[:, :, :n])
    ex = FeatureExtractorV2(sample_rate_hz=float(SAMPLE_RATE_HZ))
    freqs, power, peak = ex.breathing_spectrum(cleaned)

    fig, ax = plt.subplots(figsize=(11, 6))
    show = freqs <= 1.0
    ax.plot(freqs[show], power[show], color=_ACCENT, linewidth=1.8)
    ax.axvspan(0.1, 0.5, color=_GREEN, alpha=0.12, label="breathing band 0.1–0.5 Hz")
    if peak > 0.0:
        pk_power = float(power[np.argmin(np.abs(freqs - peak))])
        ax.plot(peak, pk_power, "v", color=_ACCENT2, markersize=14, zorder=5)
        ax.annotate(f"detected peak\n{peak:.3f} Hz  ({peak * 60:.0f} breaths/min)",
                    (peak, pk_power), textcoords="offset points", xytext=(12, -6),
                    color=_ACCENT2, fontweight="bold")
        title = f"Breathing detection — peak @ {peak:.3f} Hz"
    else:
        ax.text(0.5, 0.82, "no peak ≥ 3× band median → 0.0 Hz\n(old code returned a phantom 0.1 Hz here)",
                transform=ax.transAxes, ha="center", color=_ACCENT2, fontweight="bold",
                bbox=dict(boxstyle="round", fc="white", ec=_ACCENT2, alpha=0.9))
        title = "Breathing detection — no prominent peak (reported 0.0 Hz)"
    ax.set_title(f"{title}\n{_tag(meta)}", fontsize=15)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("power  (detrended + Hann)")
    ax.set_xlim(0, 1.0)
    ax.legend(loc="upper right")

    fig.tight_layout()
    _save(fig, outdir, "03_breathing_spectrum.png", saved)


def fig_feature_timeline(results, meta, outdir, saved) -> None:
    """Motion / doppler / breathing over the decision timeline."""
    import matplotlib.pyplot as plt

    n = len(results)
    t = [i * meta["dt"] for i in range(n)]
    motion = [float(np.mean(f.variance_rx)) if f.variance_rx else 0.0 for _, f, *_ in results]
    doppler = [float(f.doppler_mean) for _, f, *_ in results]
    breath = [float(f.breathing_frequency) for _, f, *_ in results]

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.suptitle(f"Feature timeline   [{_tag(meta)}]", fontsize=16, fontweight="bold")
    ax.plot(t, motion, "o-", color=_ACCENT, label="motion  mean(var_rx)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("motion (a.u.)", color=_ACCENT)
    ax.tick_params(axis="y", labelcolor=_ACCENT)

    ax2 = ax.twinx()
    ax2.plot(t, doppler, "s--", color=_ACCENT2, alpha=0.85, label="doppler mean (Hz)")
    ax2.plot(t, breath, "d--", color=_GREEN, alpha=0.85, label="breathing (Hz)")
    ax2.set_ylabel("frequency (Hz)")

    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [ln.get_label() for ln in lines], loc="upper right")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, outdir, "04_feature_timeline.png", saved)


def fig_trajectory(results, meta, outdir, saved) -> None:
    """Final localization trajectory, coloured by time, with the receivers."""
    import matplotlib.pyplot as plt

    rec = _records(results)
    n = len(rec["x"])
    t = [i * meta["dt"] for i in range(n)]

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.plot(rec["x"], rec["y"], "-", color="#cfcfcf", linewidth=1, zorder=1)
    sc = ax.scatter(rec["x"], rec["y"], c=t, cmap="viridis", s=90, zorder=3,
                    edgecolor="black", linewidth=0.5)
    for name, (nx, ny) in NODE_POSITIONS.items():
        ax.plot(nx, ny, "^", color="crimson", markersize=14, zorder=4)
        ax.annotate(name, (nx, ny), textcoords="offset points", xytext=(6, 6),
                    color="crimson", fontweight="bold")
    ax.annotate("start", (rec["x"][0], rec["y"][0]), textcoords="offset points",
                xytext=(8, -12))
    ax.set_title(f"Localization trajectory\n{_tag(meta)}", fontsize=15)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m) — depth")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("decision time (s)")
    fig.tight_layout()
    _save(fig, outdir, "05_trajectory.png", saved)


def fig_agentic(results, meta, outdir, saved) -> None:
    """Position/velocity fed INTO the AI layer (Kalman) vs OUT of it (decision)."""
    import matplotlib.pyplot as plt

    rec = _records(results)
    n = len(rec["x"])
    t = [i * meta["dt"] for i in range(n)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    passthrough = "" if meta["llm"] else "  (LLM off → passthrough: in == out)"
    fig.suptitle(f"Agentic layer — before vs after{passthrough}\n{_tag(meta)}",
                 fontsize=15, fontweight="bold")

    ax1.plot(t, rec["kal_x"], "o--", color=_RAW, label="x in (Kalman)")
    ax1.plot(t, rec["x"], "o-", color=_ACCENT, label="x out (decision)")
    ax1.plot(t, rec["kal_y"], "s--", color="#b9d6c9", label="y in (Kalman)")
    ax1.plot(t, rec["y"], "s-", color=_GREEN, label="y out (decision)")
    ax1.set_title("Position")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("metres")
    ax1.legend(loc="best", fontsize=9)

    ax2.plot(t, rec["kal_v"], "o--", color=_RAW, label="v in (Kalman)")
    ax2.plot(t, rec["v"], "o-", color=_ACCENT2, label="v out (decision)")
    ax2.set_title("Velocity")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("velocity (m/s)")
    ax2.legend(loc="best")

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, outdir, "06_agentic_before_after.png", saved)


def main() -> int:
    ap = argparse.ArgumentParser(description="Export GHOST v2 slide figures from a replay")
    ap.add_argument("--path", default=DEFAULT_DATASET_PATH, help="dataset file")
    ap.add_argument("--dataset", default="embedded_wifi", help="adapter name")
    ap.add_argument("--calib", type=int, default=200, help="calibration preamble frames")
    ap.add_argument("--frames", type=int, default=800, help="operational frames")
    ap.add_argument("--rate", type=float, default=0.0, help="replay pacing Hz (0 = fast)")
    ap.add_argument("--window", type=int, default=100, help="localization window (samples)")
    ap.add_argument("--breath-window", type=int, default=500, dest="breath_window",
                    help="breathing window (samples; 500 = 10 s @ 50 Hz). 0 disables.")
    ap.add_argument("--step", type=int, default=75, help="window hop between decisions")
    ap.add_argument("--pos-scale", default="embedded_wifi", dest="pos_scale",
                    help="POS_ENERGY_SCALES entry for the localizer")
    ap.add_argument("--metric", action="store_true",
                    help="treat positions as metric (default: illustrative/non-metric)")
    ap.add_argument("--no-kalman", action="store_true", dest="no_kalman",
                    help="disable Kalman smoothing")
    ap.add_argument("--llm", action="store_true", help="route decisions through Ollama")
    ap.add_argument("--outdir", default="slides", help="directory for the exported PNGs")
    add_log_flags(ap)
    args = ap.parse_args()
    apply_log_flags(args)

    matplotlib.use("Agg")  # headless-safe: only write files
    import matplotlib.pyplot as plt
    plt.rcParams.update(_STYLE)

    window, wmeta = replay_to_window(args)
    if window is None:
        print(f"  ERROR: {wmeta.get('error', 'replay failed')}")
        return 1

    results, meta = pipeline_over_window(window, args)
    if not results:
        print("  ERROR: no decisions produced (window/step too large for the stream?)")
        return 1
    print(f"collected {len(results)} decision(s); rendering figures...")

    os.makedirs(args.outdir, exist_ok=True)
    saved: list = []
    fig_signal_cleaning(window, args, args.outdir, saved)
    fig_kalman(results, meta, args.outdir, saved)
    fig_breathing_spectrum(window, args, meta, args.outdir, saved)
    fig_feature_timeline(results, meta, args.outdir, saved)
    fig_trajectory(results, meta, args.outdir, saved)
    fig_agentic(results, meta, args.outdir, saved)

    print(f"\nExported {len(saved)} figure(s) → {os.path.abspath(args.outdir)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
