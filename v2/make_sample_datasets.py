"""
Generate SYNTHETIC placeholder datasets so the comparison demo can run on all
profiles before the real downloads arrive.

These are NOT real captures — they are derived from example_csi.csv purely so the
multi-dataset comparison figure and the classifier can be rehearsed end to end.
Replace them with the real files (same paths) when you have them; the runner uses
whatever is on disk.

Usage:
    ./.venv/bin/python v2/make_sample_datasets.py
    ./.venv/bin/python v2/run_datasets_compare.py --show
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from scipy.io import savemat

from v2.datasets import DATASET_PROFILES
from v2.injector.injector import build_adapter


def _real_frames(path: str, n: int) -> np.ndarray:
    """Pull up to n complex CSI frames [time, 64] from example_csi.csv."""
    adapter = build_adapter("embedded_wifi", path)
    frames = []
    for i, snap in enumerate(adapter.snapshots()):
        frames.append(snap.iq_by_stream[0])
        if i + 1 >= n:
            break
    return np.array(frames)


def main() -> int:
    src = DATASET_PROFILES["embedded_wifi"].path
    if not os.path.exists(src):
        print(f"need {src} to derive samples from")
        return 1
    csi = _real_frames(src, 900)
    amp, pha = np.abs(csi), np.angle(csi)
    rng = np.random.default_rng(0)

    targets = {
        # activity: real motion -> should classify "walk"
        "csi_bench_activity": ("amp_phase", amp[:600], pha[:600]),
        # fall: same motion (no velocity spike) -> honestly classifies "not_fall"
        "csi_bench_fall": ("amp_phase", amp[:400], pha[:400]),
        # presence: near-static low-variance -> "empty_room"/"present_still"
        "intel_resp": ("still", None, None),
    }

    for name, (kind, a, p) in targets.items():
        profile = DATASET_PROFILES[name]
        os.makedirs(os.path.dirname(profile.path), exist_ok=True)
        if kind == "still":
            base = amp[:1500].mean()
            a = np.ones((1500, amp.shape[1])) * base + rng.normal(0, 0.01, (1500, amp.shape[1]))
            p = np.zeros_like(a)
        savemat(profile.path, {"amplitude": a, "phase": p})
        print(f"wrote SYNTHETIC {profile.path}  ({a.shape[0]} frames, use_case={profile.use_case})")

    print("\nDone. Now run:  ./.venv/bin/python v2/run_datasets_compare.py --show")
    print("NOTE: these are placeholders derived from example_csi.csv — replace with real data.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
