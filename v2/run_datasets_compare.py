"""
GHOST v2 — run several datasets through the pipeline and compare the cases.

For each dataset profile that has a file present, this runs the full offline
pipeline (dataset -> cleaner -> features -> localizer -> agentic core, with the
Level-B activity classifier), then draws a comparison the audience can read:

    1. Small-multiple trajectories — one localization cloud per dataset.
    2. Feature heatmap           — datasets x {motion, breathing, doppler,
                                    peak velocity, energy spread}: where the
                                    use-cases actually separate.
    3. Expected vs detected label — the classification per dataset.

Positions are non-metric for single-link datasets, so the *features + labels*
carry the comparison, not the x/y. Missing datasets are skipped with a note.

Two figures are produced, on shared axes so they overlay 1:1:
    * <out>.png                — final results, AFTER the agentic layer (plot 1).
    * <out>_before_agentic.png — the same view built from the deterministic
                                 localizer→Kalman estimate that was fed INTO the
                                 agentic layer, i.e. BEFORE any LLM edit (plot 2).
Compare the two to see what the agentic layer changed. Without --llm the agentic
layer is a passthrough, so the two figures are identical.

Usage:
    ./.venv/bin/python v2/run_datasets_compare.py                 # all available
    ./.venv/bin/python v2/run_datasets_compare.py --only embedded_wifi csi_bench_fall
    ./.venv/bin/python v2/run_datasets_compare.py --llm --show
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from types import SimpleNamespace

import numpy as np
import matplotlib

from v2.config_v2 import NODE_POSITIONS
from v2.datasets import DATASET_PROFILES, DatasetProfile
from v2.ghost.classifier import _thr, classify_activity
from v2.ghost.main_v2 import add_log_flags, apply_log_flags, collect_replay

# Scale-COMPARABLE metrics only. Raw motion/energy scale with each dataset's CSI
# amplitude (ESP32 ~tens, Intel ~thousands), so they can't be compared across
# datasets directly. These four can:
#   motion_level   — where this dataset's motion sits in its own still→moving
#                    band (0 = still/empty, 1 = clearly moving). Ties to the label.
#   breathing/doppler — physical units (Hz), already comparable.
#   energy_balance — RX energy imbalance as a fraction of total (dimensionless).
_FEATURES = ["motion_level", "breathing", "doppler", "energy_balance"]


def _summarize(profile: DatasetProfile, results, stage: str = "final") -> dict:
    """Aggregate one dataset's per-window decisions into comparison metrics.

    ``stage`` selects which position/velocity the trajectory + peak-velocity are
    read from:
      * ``"final"``       — the DecisionV2 after the agentic layer (plot 1).
      * ``"pre_agentic"`` — the deterministic localizer→Kalman estimate that was
                            fed INTO the agentic layer, i.e. before any LLM edit
                            (plot 2).
    The motion features (motion/breathing/doppler/energy) are computed before
    localization, so they are identical either way; only x/y/velocity — and any
    label that keys off peak velocity — can move between the two stages.
    """
    xs, ys, vels, motions, breaths, dopplers, spreads, totals, labels = \
        [], [], [], [], [], [], [], [], []
    for decision, features, raw_est, smoothed_est in results:
        src = decision if stage == "final" else smoothed_est
        xs.append(float(src.x_meters))
        ys.append(float(src.y_meters))
        vels.append(float(src.velocity_m_s))
        var = list(getattr(features, "variance_rx", []) or [])
        motions.append(float(np.mean(var)) if var else 0.0)
        breaths.append(float(features.breathing_frequency))
        dopplers.append(float(features.doppler_mean))
        energies = list((features.node_energies or {}).values())
        spreads.append(float(max(energies) - min(energies)) if energies else 0.0)
        totals.append(float(sum(energies)) if energies else 0.0)
        if decision.activity_label is not None:
            labels.append(decision.activity_label)

    mean_motion = float(np.mean(motions)) if motions else 0.0
    mean_doppler = float(np.mean(dopplers)) if dopplers else 0.0
    mean_breath = float(np.mean(breaths)) if breaths else 0.0
    mean_spread = float(np.mean(spreads)) if spreads else 0.0
    mean_total = float(np.mean(totals)) if totals else 0.0
    peak_vel = float(np.max(vels)) if vels else 0.0

    # Clip-level label: classify on the aggregate (robust to per-window noise
    # when only a few windows fit; the whole-clip mean separates the cases).
    agg_feat = SimpleNamespace(variance_rx=[mean_motion] * 3, doppler_mean=mean_doppler,
                               breathing_frequency=mean_breath, node_energies={})
    agg_est = SimpleNamespace(velocity_m_s=peak_vel)
    detected, _ = classify_activity(agg_feat, agg_est, profile.label_space(),
                                    thresholds=profile.thresholds())

    # Scale-comparable metrics (see _FEATURES note).
    t = _thr(profile.thresholds())
    still, moving = t["still"], t["moving"]
    motion_level = 0.0 if moving <= still else max(0.0, min(1.0, (mean_motion - still) / (moving - still)))
    energy_balance = mean_spread / mean_total if mean_total > 0 else 0.0

    return {
        "name": profile.name,
        "stage": stage,
        "use_case": profile.use_case,
        "expected": profile.expected_label or "(unknown)",
        "detected": detected,
        "n": len(results),
        "xs": xs, "ys": ys,
        # comparable metrics for the heatmap:
        "motion_level": motion_level,
        "breathing": mean_breath,
        "doppler": mean_doppler,
        "energy_balance": energy_balance,
        # raw values (kept for the console line / debugging):
        "motion_raw": mean_motion,
    }


def _shared_limits(summaries: list):
    """(xlim, ylim) padded to cover the x/y of every summary in the list."""
    all_x = [x for s in summaries for x in s["xs"]] or [0]
    all_y = [y for s in summaries for y in s["ys"]] or [0]
    xpad = max(0.5, (max(all_x) - min(all_x)) * 0.2)
    ypad = max(0.5, (max(all_y) - min(all_y)) * 0.2)
    return ((min(all_x) - xpad, max(all_x) + xpad),
            (min(all_y) - ypad, max(all_y) + ypad))


def _suffix_path(path: str, suffix: str) -> str:
    """Insert ``suffix`` before the extension: a.png -> a<suffix>.png."""
    root, ext = os.path.splitext(path)
    return f"{root}{suffix}{ext}"


def _plot(summaries: list, out_path: str, show: bool, title_suffix: str = "",
          xlim=None, ylim=None) -> None:
    import matplotlib.pyplot as plt

    n = len(summaries)
    fig = plt.figure(figsize=(max(11, 4 * n), 10))
    gs = fig.add_gridspec(3, n, height_ratios=[1.2, 1.0, 0.5], hspace=0.45, wspace=0.3)
    palette = plt.get_cmap("tab10")

    title = "GHOST v2 — dataset / use-case comparison"
    if title_suffix:
        title += f"  ·  {title_suffix}"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Row 1: trajectory small-multiples (shared axes for fair comparison — when
    # xlim/ylim are passed in they are common to both the before/after figures so
    # the position shift is directly readable).
    if xlim is None or ylim is None:
        xlim, ylim = _shared_limits(summaries)
    for i, s in enumerate(summaries):
        ax = fig.add_subplot(gs[0, i])
        t = list(range(s["n"]))
        ax.scatter(s["xs"], s["ys"], c=t, cmap="viridis", s=45, edgecolor="k", linewidth=0.4, zorder=3)
        ax.plot(s["xs"], s["ys"], "-", color="gray", alpha=0.4, zorder=2)
        for name, (nx, ny) in NODE_POSITIONS.items():
            ax.plot(nx, ny, "^", color="crimson", markersize=8, zorder=4)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(f"{s['name']}\n[{s['use_case']}]", fontsize=10)
        ax.set_xlabel("x (m)")
        if i == 0:
            ax.set_ylabel("y (m)")
        ax.grid(True, alpha=0.3)

    # Row 2: feature heatmap (columns normalized across datasets so cases separate).
    ax = fig.add_subplot(gs[1, :])
    mat = np.array([[s[f] for f in _FEATURES] for s in summaries], dtype=float)  # [N, F]
    norm = mat.copy()
    for j in range(mat.shape[1]):
        col = mat[:, j]
        lo, hi = col.min(), col.max()
        norm[:, j] = 0.5 if hi <= lo else (col - lo) / (hi - lo)
    im = ax.imshow(norm, aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(_FEATURES)))
    ax.set_xticklabels(_FEATURES, rotation=15)
    ax.set_yticks(range(n))
    ax.set_yticklabels([s["name"] for s in summaries])
    ax.set_title("Feature comparison — scale-comparable metrics "
                 "(motion_level 0-1, breathing/doppler Hz, energy_balance fraction; "
                 "color = per-column relative)", fontsize=10)
    for i in range(n):
        for j in range(len(_FEATURES)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    color="white" if norm[i, j] < 0.6 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)

    # Row 3: expected vs detected label per dataset.
    ax = fig.add_subplot(gs[2, :])
    ax.axis("off")
    rows = [["dataset", "use-case", "expected", "detected", "match"]]
    for s in summaries:
        match = "—" if s["expected"] == "(unknown)" else ("✓" if s["detected"] == s["expected"] else "✗")
        rows.append([s["name"], s["use_case"], s["expected"], s["detected"], match])
    table = ax.table(cellText=rows, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    for c in range(len(rows[0])):
        table[0, c].set_facecolor("#222")
        table[0, c].set_text_props(color="white", fontweight="bold")
    ax.set_title("Classification: expected vs detected", fontsize=11, pad=10)

    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"saved comparison → {out_path}")
    if show:
        plt.show()


def _write_csv(summaries: list, csv_path: str) -> None:
    cols = ["name", "use_case", "expected", "detected", "n"] + _FEATURES
    lines = [",".join(cols)]
    for s in summaries:
        lines.append(",".join(str(s[c]) for c in cols))
    with open(csv_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"saved summary → {csv_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare GHOST v2 across datasets / use-cases")
    ap.add_argument("--only", nargs="+", metavar="NAME",
                    help="restrict to these profile names (default: all available)")
    ap.add_argument("--llm", action="store_true", help="route decisions through Ollama")
    ap.add_argument("--out", default="ghost_v2_comparison.png",
                    help="output PNG for plot 1 (final); plot 2 gets a "
                         "'_before_agentic' suffix")
    ap.add_argument("--csv", default="ghost_v2_comparison.csv",
                    help="output summary CSV (plot 2 gets a '_before_agentic' suffix)")
    ap.add_argument("--show", action="store_true", help="open an interactive window too")
    add_log_flags(ap)
    args = ap.parse_args()
    apply_log_flags(args)

    if not args.show:
        matplotlib.use("Agg")

    wanted = args.only or list(DATASET_PROFILES)
    summaries_final, summaries_pre = [], []
    for name in wanted:
        profile = DATASET_PROFILES.get(name)
        if profile is None:
            print(f"  skip {name}: unknown profile")
            continue
        if not profile.exists():
            print(f"  skip {profile.name}: file not found ({profile.path})")
            continue
        print(f"→ running {profile.name} [{profile.use_case}] from {profile.path}")
        results, meta = collect_replay(profile.to_args(llm=args.llm), verbose=False,
                                       label_space=profile.label_space(),
                                       classifier_thresholds=profile.thresholds())
        if not results:
            print(f"  skip {profile.name}: {meta.get('error', 'no decisions')}")
            continue
        s = _summarize(profile, results, stage="final")
        s_pre = _summarize(profile, results, stage="pre_agentic")
        summaries_final.append(s)
        summaries_pre.append(s_pre)
        print(f"   {s['n']} windows · detected={s['detected']} "
              f"(expected {s['expected']}) · motion_raw={s['motion_raw']:.1f} "
              f"motion_level={s['motion_level']:.2f} breath={s['breathing']:.3f} "
              f"doppler={s['doppler']:.3f}")

    if not summaries_final:
        print("\nNo datasets ran. Drop files under datasets/ (see v2/datasets.py) and retry.")
        return 1

    print(f"\ncompared {len(summaries_final)} dataset(s).")

    # Shared axis limits across BOTH figures so the before/after position shift is
    # read on one common frame.
    xlim, ylim = _shared_limits(summaries_final + summaries_pre)

    # Plot 1 — final results (after the agentic layer).
    _write_csv(summaries_final, args.csv)
    _plot(summaries_final, args.out, args.show,
          title_suffix="final results (after agentic layer)", xlim=xlim, ylim=ylim)

    # Plot 2 — same view, but built from the pre-agentic (localizer→Kalman)
    # estimate that was fed into the agentic layer.
    pre_out = _suffix_path(args.out, "_before_agentic")
    pre_csv = _suffix_path(args.csv, "_before_agentic")
    _write_csv(summaries_pre, pre_csv)
    _plot(summaries_pre, pre_out, args.show,
          title_suffix="before agentic layer (deterministic localizer→Kalman)",
          xlim=xlim, ylim=ylim)

    if not args.llm:
        print("note: without --llm the agentic layer is a deterministic passthrough, "
              "so plot 2 (before) will match plot 1 (after). Run with --llm to see the "
              "before/after difference.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
