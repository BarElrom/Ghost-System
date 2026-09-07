"""
GHOST v2 — anomaly-correction demo: the agentic layer as a guardrail.

Runs a normal replay, then INJECTS one physically-impossible window into the raw
localizer estimate stream (a teleport at an indoor-impossible speed). Each window
is then reasoned twice, over the SAME estimates:

    * deterministic  — DecisionV2.from_estimate(): the raw localizer, passed
                       through blindly (this is the offline / no-LLM path).
    * agentic (LLM)  — AgenticCoreV2.reason(): the LLM sanity-checks the estimate.

At the injected window the LLM raises anomaly_flags.unrealistic_speed and drops
signal_confidence, while KEEPING x/y/velocity unchanged — it flags the bad point,
it does not silently rewrite it. The deterministic path reports the bad point with
no warning. The figure makes that difference the whole story.

Needs a capable model — llama3.2:3b will NOT apply the rule (it echoes the input);
this defaults to llama3.1:8b. Start Ollama first (docker compose up -d) and make
sure the model is pulled (ollama pull llama3.1:8b).

Usage:
    ./.venv/bin/python v2/demo_anomaly_correction.py                 # bundled example_csi.csv
    ./.venv/bin/python v2/demo_anomaly_correction.py --model llama3.1:8b \
        --inject-index 5 --inject-velocity 8.4 --inject-x 6.0 --outdir slides/anomaly
"""

import argparse
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib

from v2.config_v2 import DEFAULT_DATASET_PATH, OLLAMA_SETTINGS, SAMPLE_RATE_HZ
from v2.ghost.agentic_core_v2 import AgenticCoreV2, DecisionV2
from v2.ghost.main_v2 import add_log_flags, apply_log_flags, collect_replay

_RAW = "#c0392b"      # deterministic / raw localizer (blind)
_LLM = "#2b7de0"      # agentic layer (flags the anomaly)
_BAND = "#f1c40f"     # injected-window highlight


def _decide(raws, feats, model: str):
    """Return (deterministic, agentic) DecisionV2 lists over the estimate stream."""
    det = [DecisionV2.from_estimate(r) for r in raws]

    core = AgenticCoreV2(settings=replace(OLLAMA_SETTINGS, model=model), use_llm=True)
    if not core._ensure_client():  # noqa: SLF001 — explicit connectivity check for a clear error
        print(f"  ERROR: Ollama not reachable / model '{model}' unavailable. "
              f"Start it (docker compose up -d) and `ollama pull {model}`.")
        return None, None
    agentic, prev = [], None
    for r, f in zip(raws, feats):
        d = core.reason(r, feature_set=f, previous_decision=prev)
        agentic.append(d)
        prev = d
    return det, agentic


def _plot(det, llm, meta, idx, inject, outdir: str) -> str:
    import matplotlib.pyplot as plt

    n = len(det)
    t = [i * meta["dt"] for i in range(n)]

    def series(ds, key):
        return [d.to_dict()[key] for d in ds]

    def flag(ds):
        return [1 if d.to_dict()["anomaly_flags"]["unrealistic_speed"] else 0 for d in ds]

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(
        f"Agentic layer as a guardrail — injected impossible jump "
        f"(v={inject['v']:.1f} m/s at t={t[idx]:.1f}s)\n"
        f"raw localizer reports it blindly · LLM ({meta['model']}) flags it & drops "
        f"confidence · coordinates unchanged",
        fontsize=14, fontweight="bold",
    )

    # shade the injected window on every panel
    for ax in axes:
        ax.axvspan(t[idx] - meta["dt"] / 2, t[idx] + meta["dt"] / 2,
                   color=_BAND, alpha=0.25, label="injected anomaly")

    # 1) velocity — identical for both (proves coordinates/velocity are NOT rewritten)
    ax = axes[0]
    ax.plot(t, series(det, "velocity_m_s"), "o-", color=_RAW, label="raw localizer (deterministic)")
    ax.plot(t, series(llm, "velocity_m_s"), "x--", color=_LLM, markersize=9,
            label="agentic (LLM)")
    ax.set_ylabel("velocity (m/s)")
    ax.set_title("Velocity — LLM keeps the value (flags, does not rewrite)")
    ax.legend(loc="upper right", fontsize=9)

    # 2) confidence — LLM drops it at the anomaly
    ax = axes[1]
    ax.plot(t, series(det, "signal_confidence"), "o-", color=_RAW, label="raw localizer")
    ax.plot(t, series(llm, "signal_confidence"), "s-", color=_LLM, label="agentic (LLM)")
    ax.set_ylabel("signal_confidence")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Confidence — LLM lowers it on the implausible window")
    ax.legend(loc="lower right", fontsize=9)

    # 3) unrealistic_speed flag — off for raw, on for LLM at the anomaly
    ax = axes[2]
    ax.step(t, flag(det), where="mid", color=_RAW, linewidth=2, label="raw localizer")
    ax.step(t, flag(llm), where="mid", color=_LLM, linewidth=2, label="agentic (LLM)")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["clear", "FLAGGED"])
    ax.set_ylim(-0.2, 1.2)
    ax.set_ylabel("unrealistic_speed")
    ax.set_xlabel("time (s)")
    ax.set_title("Anomaly flag — raw stays silent, LLM raises it")
    ax.legend(loc="center right", fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "anomaly_correction.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="GHOST v2 agentic anomaly-correction demo")
    ap.add_argument("--path", default=DEFAULT_DATASET_PATH, help="dataset file")
    ap.add_argument("--dataset", default="embedded_wifi", help="adapter name")
    ap.add_argument("--calib", type=int, default=200)
    ap.add_argument("--frames", type=int, default=800)
    ap.add_argument("--rate", type=float, default=0.0)
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--breath-window", type=int, default=300, dest="breath_window")
    ap.add_argument("--step", type=int, default=75)
    ap.add_argument("--pos-scale", default="embedded_wifi", dest="pos_scale")
    ap.add_argument("--metric", action="store_true")
    ap.add_argument("--no-kalman", action="store_true", dest="no_kalman")
    ap.add_argument("--model", default="llama3.1:8b",
                    help="Ollama model for the agentic layer (3b is too weak to flag)")
    ap.add_argument("--inject-index", type=int, default=-1, dest="inject_index",
                    help="window to corrupt (default: middle)")
    ap.add_argument("--inject-velocity", type=float, default=8.4, dest="inject_velocity",
                    help="impossible speed to inject (m/s; >3 triggers the rule)")
    ap.add_argument("--inject-x", type=float, default=6.0, dest="inject_x",
                    help="teleport x_meters for the corrupted window")
    ap.add_argument("--outdir", default="slides/anomaly")
    add_log_flags(ap)
    args = ap.parse_args()
    apply_log_flags(args)
    args.llm = False  # collect_replay just supplies the raw estimates; we reason ourselves

    matplotlib.use("Agg")

    results, meta = collect_replay(args)
    if results is None:
        print(f"  ERROR: {meta.get('error', 'replay failed')}")
        return 1
    if not results:
        print("  ERROR: no windows produced (window/step too large?)")
        return 1

    raws = [raw for (_, _, raw, _) in results]
    feats = [f for (_, f, *_) in results]

    idx = args.inject_index if args.inject_index >= 0 else len(raws) // 2
    idx = max(1, min(idx, len(raws) - 1))  # not the first window (needs a "previous")
    # Inject a blind teleport: raw localizer reports it with NO warning.
    raws[idx] = replace(raws[idx], x_meters=args.inject_x,
                        velocity_m_s=args.inject_velocity,
                        unrealistic_speed=False, multipath_reflection_suspect=False)
    inject = {"v": args.inject_velocity, "x": args.inject_x}
    print(f"Injected impossible window at index {idx} "
          f"(v={inject['v']} m/s, x={inject['x']} m); reasoning with '{args.model}'...")

    det, llm = _decide(raws, feats, args.model)
    if det is None:
        return 1

    meta["model"] = args.model
    path = _plot(det, llm, meta, idx, inject, args.outdir)

    d0, l0 = det[idx].to_dict(), llm[idx].to_dict()
    print("\n  injected window — deterministic vs agentic:")
    print(f"    raw localizer : v={d0['velocity_m_s']:.2f}  conf={d0['signal_confidence']:.2f}  "
          f"unrealistic_speed={d0['anomaly_flags']['unrealistic_speed']}")
    print(f"    agentic (LLM) : v={l0['velocity_m_s']:.2f}  conf={l0['signal_confidence']:.2f}  "
          f"unrealistic_speed={l0['anomaly_flags']['unrealistic_speed']}  "
          f"(x/y unchanged: {l0['coordinates']['x_meters']:.2f}, {l0['coordinates']['y_meters']:.2f})")
    print(f"\nsaved → {os.path.abspath(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
