"""
GHOST v2 — dataset profiles for the multi-case comparison demo.

Each profile bundles everything a run needs: which adapter parses the file,
where the file is, the localizer scale to use, and the use-case (which fixes the
closed label set the classifier / LLM must choose from). The comparison runner
iterates these and skips any whose file is missing, so you can add datasets
incrementally just by dropping files under v2/datasets/.
"""

import argparse
import os
from dataclasses import dataclass

from v2.config_v2 import DEFAULT_DATASET_PATH
from v2.ghost.classifier import LABEL_SPACES

_V2_DIR = os.path.dirname(os.path.abspath(__file__))


def _p(*parts) -> str:
    """Absolute path under the v2 directory (self-contained datasets/ bundle)."""
    return os.path.join(_V2_DIR, *parts)


@dataclass
class DatasetProfile:
    """One dataset clip and how to run it through the pipeline."""

    name: str            # display name (unique)
    adapter: str         # build_adapter() key
    path: str            # dataset file
    use_case: str        # key in LABEL_SPACES: activity | fall | presence | person_id
    pos_scale: str       # POS_ENERGY_SCALES entry
    expected_label: str = None   # ground truth for the clip (if known)
    metric: bool = False
    calib: int = 200
    frames: int = 800
    window: int = 200
    breath_window: int = 600
    step: int = 150
    rate: float = 0.0
    # Per-dataset motion thresholds for the classifier (magnitudes scale with
    # CSI amplitude). None => classifier defaults (embedded_wifi scale).
    motion_still: float = None
    motion_moving: float = None

    def label_space(self) -> list:
        return list(LABEL_SPACES.get(self.use_case, []))

    def thresholds(self) -> dict:
        """Classifier threshold overrides for this dataset (None if defaults)."""
        t = {}
        if self.motion_still is not None:
            t["still"] = self.motion_still
        if self.motion_moving is not None:
            t["moving"] = self.motion_moving
        return t or None

    def exists(self) -> bool:
        return os.path.exists(self.path)

    def to_args(self, llm: bool = False) -> argparse.Namespace:
        """Build the Namespace that collect_replay() consumes."""
        return argparse.Namespace(
            path=self.path, dataset=self.adapter, calib=self.calib,
            frames=self.frames, rate=self.rate, window=self.window,
            breath_window=self.breath_window, step=self.step,
            pos_scale=self.pos_scale, metric=self.metric, llm=llm,
        )


# The demo set. Files that don't exist yet are simply skipped by the runner.
DATASET_PROFILES = {
    # Real today (ships with the repo) — HAR / activity.
    "embedded_wifi": DatasetProfile(
        name="embedded_wifi", adapter="embedded_wifi", path=DEFAULT_DATASET_PATH,
        use_case="activity", pos_scale="embedded_wifi", expected_label=None,
    ),
    # CSI-Bench (#1) — drop a .mat here. Two use-cases as examples.
    "csi_bench_activity": DatasetProfile(
        name="csi_bench_activity", adapter="csi_bench",
        path=_p("datasets", "csi_bench", "activity.mat"),
        use_case="activity", pos_scale="csi_bench", expected_label="walk",
    ),
    "csi_bench_fall": DatasetProfile(
        name="csi_bench_fall", adapter="csi_bench",
        path=_p("datasets", "csi_bench", "fall.mat"),
        use_case="fall", pos_scale="csi_bench", expected_label="fall",
        window=200, breath_window=0, step=100,
    ),
    # Intel Respiratory (#3) — real Zenodo captures, three presence cases.
    "intel_empty": DatasetProfile(
        name="intel_empty", adapter="intel_resp",
        path=_p("datasets", "intel_resp",
                "CSI_complex1_aligned_fixedPiOffset_20MHz_Empty.csv"),
        use_case="presence", pos_scale="intel_resp", expected_label="empty_room",
        window=1000, breath_window=1000, step=400, frames=2000,
        # Tuned to the residual-motion scale this pipeline actually produces on
        # the Intel clips (empty≈210, sitting≈188, walking≈870 aggregate). Empty
        # and walking separate cleanly; sitting's residual motion is *below* the
        # empty room's, so bulk motion alone cannot tell "present but still" from
        # "empty" — that needs a working breathing/phase feature (Plan §17).
        motion_still=400.0, motion_moving=600.0,
    ),
    "intel_sitting": DatasetProfile(
        name="intel_sitting", adapter="intel_resp",
        path=_p("datasets", "intel_resp",
                "CSI_complex1_aligned_fixedPiOffset_20MHz_OneSitting_IFcomputer.csv"),
        use_case="presence", pos_scale="intel_resp", expected_label="present_still",
        window=1000, breath_window=1000, step=400, frames=2000,
        # Tuned to the residual-motion scale this pipeline actually produces on
        # the Intel clips (empty≈210, sitting≈188, walking≈870 aggregate). Empty
        # and walking separate cleanly; sitting's residual motion is *below* the
        # empty room's, so bulk motion alone cannot tell "present but still" from
        # "empty" — that needs a working breathing/phase feature (Plan §17).
        motion_still=400.0, motion_moving=600.0,
    ),
    "intel_walking": DatasetProfile(
        name="intel_walking", adapter="intel_resp",
        path=_p("datasets", "intel_resp",
                "CSI_complex1_aligned_fixedPiOffset_20MHz_OneSittingOneWalking.csv"),
        use_case="presence", pos_scale="intel_resp", expected_label="present_moving",
        window=1000, breath_window=1000, step=400, frames=2000,
        # Tuned to the residual-motion scale this pipeline actually produces on
        # the Intel clips (empty≈210, sitting≈188, walking≈870 aggregate). Empty
        # and walking separate cleanly; sitting's residual motion is *below* the
        # empty room's, so bulk motion alone cannot tell "present but still" from
        # "empty" — that needs a working breathing/phase feature (Plan §17).
        motion_still=400.0, motion_moving=600.0,
    ),
}


def available_profiles() -> list:
    """Profiles whose data file is present, in registry order."""
    return [p for p in DATASET_PROFILES.values() if p.exists()]
