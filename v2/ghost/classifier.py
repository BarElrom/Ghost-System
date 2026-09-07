"""
GHOST v2 — deterministic activity classifier (Level B, no-LLM core).

Maps the motion FeatureSet + localizer estimate onto a *closed* label set
(the dataset's use-case). This is the classification that runs when Ollama is
off; the agentic core can optionally override/confirm it with the LLM.

Everything here is pure Python + the feature numbers already computed upstream —
no new dependencies, so it runs inside the dependency-free test suite.

The thresholds are intentionally explicit constants: like POS_ENERGY_SCALES,
they are approximate and MUST be tuned per dataset (feature magnitudes differ).
They are chosen to be sensible on the example_csi.csv feature ranges
(motion-variance ~10-18, velocity ~0-0.06, breathing 0.1-0.5 Hz).
"""

import logging

logger = logging.getLogger("ghost.v2.classifier")

# --- default thresholds (per-dataset overrides expected) --------------------
# Motion thresholds are on mean(variance_rx), whose magnitude scales with the
# dataset's CSI amplitude — so, like POS_ENERGY_SCALES, they MUST be tuned per
# dataset (pass a thresholds dict via the DatasetProfile). Defaults suit the
# embedded_wifi scale (motion ~10-25).
DEFAULT_THRESHOLDS = {
    "still": 3.0,          # motion below this ~ no bulk motion (empty / still)
    "moving": 8.0,         # motion above this ~ sustained body motion
    "gesture_doppler": 1.4,  # doppler_mean (Hz) above this ~ fast limb motion
    "fall_velocity": 1.0,    # localizer velocity (m/s) spike consistent with a fall
    "fall_doppler": 1.6,     # + high doppler burst
}
BREATHING_LO = 0.1
BREATHING_HI = 0.5


def _thr(thresholds) -> dict:
    d = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        d.update(thresholds)
    return d

# The label sets the demo uses, keyed by a short use-case name.
LABEL_SPACES = {
    "activity": ["still", "walk", "sit_down", "stand_up", "gesture"],
    "fall": ["fall", "not_fall"],
    "presence": ["empty_room", "present_still", "present_moving"],
    "person_id": ["person_A", "person_B", "unknown"],
}


def _signals(feature_set, estimate) -> dict:
    """Collapse the FeatureSet + estimate into the few scalars we classify on."""
    var = list(getattr(feature_set, "variance_rx", []) or [])
    motion = sum(var) / len(var) if var else 0.0
    energies = getattr(feature_set, "node_energies", {}) or {}
    e_total = float(sum(energies.values())) if energies else 0.0
    return {
        "motion": float(motion),
        "velocity": float(getattr(estimate, "velocity_m_s", 0.0)),
        "doppler": float(getattr(feature_set, "doppler_mean", 0.0)),
        "breathing": float(getattr(feature_set, "breathing_frequency", 0.0)),
        "e_total": e_total,
    }


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _activity(s: dict, t: dict) -> tuple:
    if s["motion"] < t["still"] and s["velocity"] < 0.05:
        return "still", _clamp01(1.0 - s["motion"] / t["still"])
    if s["motion"] >= t["moving"] and s["doppler"] >= t["gesture_doppler"]:
        return "gesture", _clamp01(s["doppler"] / (2 * t["gesture_doppler"]))
    if s["motion"] >= t["moving"]:
        return "walk", _clamp01(s["motion"] / (2 * t["moving"]))
    # in-between motion — a transition (sit/stand); we can't split the two, so
    # report the more likely bulk-motion label with modest confidence.
    return "walk", 0.4


def _fall(s: dict, t: dict) -> tuple:
    if s["velocity"] >= t["fall_velocity"] or (s["motion"] >= t["moving"] and s["doppler"] >= t["fall_doppler"]):
        score = max(s["velocity"] / (2 * t["fall_velocity"]), s["doppler"] / (2 * t["fall_doppler"]))
        return "fall", _clamp01(score)
    return "not_fall", _clamp01(1.0 - s["velocity"] / t["fall_velocity"])


def _presence(s: dict, t: dict) -> tuple:
    # Motion-based 3-way split (per-dataset thresholds). Empty rooms have the
    # least residual motion after static subtraction; walking the most.
    if s["motion"] >= t["moving"]:
        return "present_moving", _clamp01(s["motion"] / (2 * t["moving"]))
    if s["motion"] < t["still"]:
        return "empty_room", _clamp01(1.0 - s["motion"] / t["still"])
    breathing = BREATHING_LO <= s["breathing"] <= BREATHING_HI
    return "present_still", 0.7 if breathing else 0.5


def _person_id(s: dict, t: dict) -> tuple:
    # Identity needs a per-person gait fingerprint we do not have from a single
    # window of aggregate features — be honest rather than guess.
    return "unknown", 0.2


_DISPATCH = {
    "activity": _activity,
    "fall": _fall,
    "presence": _presence,
    "person_id": _person_id,
}


def classify_activity(feature_set, estimate, label_space, thresholds=None) -> tuple:
    """Return (label, confidence) for the given label set.

    Args:
        feature_set: a FeatureSet (motion descriptors).
        estimate: a LocationEstimate (for velocity).
        label_space: either a use-case name in LABEL_SPACES ("activity",
            "fall", "presence", "person_id") OR an explicit list of labels
            matching one of those sets.
        thresholds: optional per-dataset overrides for DEFAULT_THRESHOLDS
            (motion magnitudes scale with CSI amplitude — tune per dataset).

    Unknown label sets return (first_label, 0.0) so callers never crash.
    """
    use_case = label_space if isinstance(label_space, str) else _match_use_case(label_space)
    s = _signals(feature_set, estimate)
    t = _thr(thresholds)
    fn = _DISPATCH.get(use_case)
    if fn is None:
        labels = LABEL_SPACES.get(use_case) or (label_space if isinstance(label_space, list) else [])
        return (labels[0] if labels else "unknown"), 0.0
    label, conf = fn(s, t)
    logger.debug("classify[%s]: signals=%s thr=%s → %s (%.2f)", use_case,
                 {k: round(v, 3) for k, v in s.items()}, t, label, conf)
    return label, conf


def _match_use_case(labels) -> str:
    """Find which known use-case a list of labels corresponds to."""
    label_set = set(labels or [])
    for name, known in LABEL_SPACES.items():
        if label_set == set(known):
            return name
    return "unknown"
