"""Level-B tests — deterministic activity classifier + DecisionV2 label plumbing.

Validates the closed-set classifier (activity / fall / presence / person_id),
confidence bounds, list-vs-name label spaces, and that the optional
classification is additive: DecisionV2.to_dict() is unchanged unless a label
space is active, and AgenticCoreV2 attaches a label when one is configured.

Run:  python v2/tests/test_classifier.py
"""

import sys
from types import SimpleNamespace

from _harness import Harness

from v2.ghost.classifier import LABEL_SPACES, classify_activity
from v2.ghost.agentic_core_v2 import AgenticCoreV2, DecisionV2


def _feat(motion=0.0, breathing=0.0, doppler=0.0, energies=None):
    var = [motion, motion, motion]
    return SimpleNamespace(
        breathing_frequency=breathing, total_energy=0.0, doppler_mean=doppler,
        variance_rx=var, node_energies=energies or {"RX1": 700, "RX2": 640, "RX3": 580},
        phase_variance_rx=[0, 0, 0],
    )


def _est(velocity=0.0, confidence=1.0):
    return SimpleNamespace(
        x_meters=0.5, y_meters=2.6, velocity_m_s=velocity, confidence=confidence,
        node_energies={"RX1": 700, "RX2": 640, "RX3": 580},
        unrealistic_speed=False, multipath_reflection_suspect=False, non_metric=True,
    )


def test_activity(h: Harness) -> None:
    lbl, c = classify_activity(_feat(motion=0.5), _est(velocity=0.0), "activity")
    h.expect("low motion -> still", lbl == "still", lbl)
    h.expect("still confidence in [0,1]", 0.0 <= c <= 1.0, str(c))
    lbl, _ = classify_activity(_feat(motion=12.0, doppler=0.5), _est(velocity=0.02), "activity")
    h.expect("high motion, low doppler -> walk", lbl == "walk", lbl)
    lbl, _ = classify_activity(_feat(motion=12.0, doppler=2.0), _est(), "activity")
    h.expect("high motion + high doppler -> gesture", lbl == "gesture", lbl)


def test_fall(h: Harness) -> None:
    lbl, c = classify_activity(_feat(motion=12.0, doppler=2.0), _est(velocity=1.5), "fall")
    h.expect("velocity spike -> fall", lbl == "fall", lbl)
    h.expect("fall confidence bounded", 0.0 <= c <= 1.0, str(c))
    lbl, _ = classify_activity(_feat(motion=1.0), _est(velocity=0.0), "fall")
    h.expect("calm -> not_fall", lbl == "not_fall", lbl)


def test_presence(h: Harness) -> None:
    # Motion-based 3-way split (defaults: still=3, moving=8).
    lbl, _ = classify_activity(_feat(motion=0.5), _est(), "presence")
    h.expect("very low motion -> empty_room", lbl == "empty_room", lbl)
    lbl, _ = classify_activity(_feat(motion=5.0, breathing=0.3), _est(), "presence")
    h.expect("mid motion -> present_still", lbl == "present_still", lbl)
    lbl, _ = classify_activity(_feat(motion=12.0), _est(), "presence")
    h.expect("high motion -> present_moving", lbl == "present_moving", lbl)


def test_presence_thresholds(h: Harness) -> None:
    # Per-dataset thresholds relabel the same motion values (Intel-scale demo).
    thr = {"still": 1800.0, "moving": 2600.0}
    lbl, _ = classify_activity(_feat(motion=1202.0), _est(), "presence", thresholds=thr)
    h.expect("Intel empty (1202) -> empty_room", lbl == "empty_room", lbl)
    lbl, _ = classify_activity(_feat(motion=2283.0), _est(), "presence", thresholds=thr)
    h.expect("Intel sitting (2283) -> present_still", lbl == "present_still", lbl)
    lbl, _ = classify_activity(_feat(motion=2912.0), _est(), "presence", thresholds=thr)
    h.expect("Intel walking (2912) -> present_moving", lbl == "present_moving", lbl)


def test_person_id_and_spaces(h: Harness) -> None:
    lbl, c = classify_activity(_feat(motion=12.0), _est(), "person_id")
    h.expect("person_id -> unknown (honest)", lbl == "unknown", lbl)
    h.expect("person_id low confidence", c < 0.5, str(c))
    # A label *list* must resolve to the same use-case as its name.
    by_name, _ = classify_activity(_feat(motion=0.5), _est(), "activity")
    by_list, _ = classify_activity(_feat(motion=0.5), _est(), LABEL_SPACES["activity"])
    h.expect("list label-space == name label-space", by_name == by_list, f"{by_name}/{by_list}")


def test_decision_additive(h: Harness) -> None:
    # Without a label space, to_dict() has NO classification block (unchanged schema).
    d = DecisionV2(x_meters=0.5, y_meters=2.6)
    h.expect("no classification key by default", "classification" not in d.to_dict())
    # With a label, the block appears.
    d.activity_label, d.label_confidence, d.label_space = "walk", 0.8, ["still", "walk"]
    out = d.to_dict()
    h.expect("classification block present when labeled", "classification" in out)
    h.expect("label value carried", out["classification"]["activity_label"] == "walk")
    # Existing keys still present and untouched.
    h.expect("coordinates still present", "coordinates" in out)
    h.expect("raw_node_energies still present", "raw_node_energies" in out)


def test_core_attaches_label(h: Harness) -> None:
    # use_llm=False -> deterministic passthrough; label_space -> label attached.
    core = AgenticCoreV2(use_llm=False, label_space="activity")
    d = core.reason(_est(velocity=0.02), _feat(motion=12.0, doppler=0.5))
    h.expect("core attaches activity_label", d.activity_label == "walk", str(d.activity_label))
    h.expect("core sets label_confidence", d.label_confidence is not None)
    # No label space -> unchanged behavior (no label).
    core2 = AgenticCoreV2(use_llm=False)
    d2 = core2.reason(_est(), _feat(motion=12.0))
    h.expect("no label when label_space is None", d2.activity_label is None)


def main() -> int:
    h = Harness("Level-B classifier")
    h.case("activity", test_activity)
    h.case("fall", test_fall)
    h.case("presence", test_presence)
    h.case("presence thresholds", test_presence_thresholds)
    h.case("person_id + label spaces", test_person_id_and_spaces)
    h.case("DecisionV2 additive schema", test_decision_additive)
    h.case("core attaches label", test_core_attaches_label)
    return h.done()


if __name__ == "__main__":
    sys.exit(main())
