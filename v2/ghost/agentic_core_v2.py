
import json
import logging
import re
import time
from dataclasses import dataclass, field

logger = logging.getLogger("ghost.v2.agentic_core")

try:
    import ollama as _ollama_lib
except ImportError:
    _ollama_lib = None

from v2.config_v2 import OLLAMA_SETTINGS


_SYSTEM_PROMPT = (
    "You are a Wi-Fi CSI localization and reasoning engine. "
    "You are given a preliminary position/velocity estimate from a deterministic "
    "localizer, the underlying motion features, and the previous estimate. "
    "Sanity-check the estimate (implausible jumps, low confidence, symmetric "
    "ambiguity), then respond with ONLY a JSON object using exactly these keys: "
    "timestamp (float, unix seconds), coordinates (object with x_meters and "
    "y_meters, floats), velocity_m_s (float), signal_confidence (float 0-1), "
    "anomaly_flags (object with unrealistic_speed and multipath_reflection_"
    "suspected, booleans), raw_node_energies (object of node->float). "
    "Do not invent nodes; echo the node energies you were given."
)

_USER_PROMPT_TEMPLATE = """\
LOCALIZER ESTIMATE:
- x_meters: {x:.3f}
- y_meters: {y:.3f}
- velocity_m_s: {v:.3f}
- signal_confidence: {conf:.3f}
- anomaly_flags: unrealistic_speed={unreal}, multipath_reflection_suspected={multi}
- position_is_metric: {metric}

MOTION FEATURES:
- breathing_frequency: {breath:.4f} Hz
- total_energy: {energy:.2f}
- doppler_mean: {doppler:.4f} Hz
- per-node energies: {energies}

PREVIOUS STATE:
{previous_section}

MEMORY MATCHES (optional context):
{memory_section}

ANALYSIS RULES:
- If velocity_m_s > 3.0 the target moved implausibly fast indoors — set
  anomaly_flags.unrealistic_speed=true AND set signal_confidence to at most 0.3.
  Never change x_meters/y_meters or velocity_m_s — only raise the flag and lower
  confidence (the raw estimate is kept; the flag warns downstream it is suspect).
- If position_is_metric is False the x/y are illustrative only (synthesized node
  diversity); keep them but do not raise confidence on their basis.
- A symmetric energy split collapses x toward 0 — do not over-interpret x ~ 0.
- Prefer temporal continuity: large jumps from the previous position are suspect.
- Echo raw_node_energies unchanged.

Respond with ONLY the JSON object described in the system message. Use
timestamp={timestamp}."""


@dataclass
class DecisionV2:
    """A single reasoning output from the v2 Agentic Core (Plan section 9)."""

    x_meters: float = 0.0
    y_meters: float = 0.0
    velocity_m_s: float = 0.0
    signal_confidence: float = 0.0
    unrealistic_speed: bool = False
    multipath_reflection_suspected: bool = False
    node_energies: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: time.time())
    # Optional Level-B classification (added only when a label space is active).
    # Defaults keep the schema byte-identical to the position-only output.
    activity_label: str | None = None
    label_confidence: float | None = None
    label_space: list | None = None

    def to_dict(self) -> dict:
        d = {
            "timestamp": self.timestamp,
            "coordinates": {"x_meters": self.x_meters, "y_meters": self.y_meters},
            "velocity_m_s": self.velocity_m_s,
            "signal_confidence": self.signal_confidence,
            "anomaly_flags": {
                "unrealistic_speed": self.unrealistic_speed,
                "multipath_reflection_suspected": self.multipath_reflection_suspected,
            },
            "raw_node_energies": dict(self.node_energies),
        }
        if self.activity_label is not None:
            d["classification"] = {
                "activity_label": self.activity_label,
                "label_confidence": self.label_confidence,
                "label_space": list(self.label_space or []),
            }
        return d

    @classmethod
    def from_estimate(cls, estimate, timestamp: float | None = None) -> "DecisionV2":
        """Build a DecisionV2 straight from a localizer LocationEstimate.

        This is the fallback path (and the offline path when Ollama is off): the
        deterministic estimate is passed through unchanged.
        """
        return cls(
            x_meters=float(estimate.x_meters),
            y_meters=float(estimate.y_meters),
            velocity_m_s=float(estimate.velocity_m_s),
            signal_confidence=float(estimate.confidence),
            unrealistic_speed=bool(estimate.unrealistic_speed),
            multipath_reflection_suspected=bool(estimate.multipath_reflection_suspect),
            node_energies=dict(estimate.node_energies),
            timestamp=timestamp if timestamp is not None else time.time(),
        )


class AgenticCoreV2:
    """LLM reasoning layer producing coordinates/velocity DecisionV2 objects.

    Args:
        settings: Ollama connection settings (defaults to the v1 OLLAMA_SETTINGS).
        use_llm: when False, reason() skips Ollama entirely and passes the
            localizer estimate straight through DecisionV2.from_estimate. Useful
            for offline/dashboard runs and tests without a model.
        client: an injected Ollama-compatible client (a chat(...) provider),
            mainly for tests. When given, use_llm defaults to True.
    """

    def __init__(self, settings=None, use_llm: bool = True, client=None,
                 label_space=None, classifier_thresholds=None):
        self._settings = settings or OLLAMA_SETTINGS
        self._client = client
        self._model_ready = client is not None
        self.use_llm = use_llm if client is None else True
        # When set (a use-case name or label list), reason() attaches a
        # deterministic activity label to every decision. Default None => the
        # position-only behavior is completely unchanged.
        self.label_space = label_space
        self.classifier_thresholds = classifier_thresholds

    def _finalize(self, decision, feature_set, estimate):
        """Attach a Level-B activity label if a label space is configured."""
        if self.label_space and feature_set is not None:
            from v2.ghost.classifier import classify_activity
            label, conf = classify_activity(feature_set, estimate, self.label_space,
                                            thresholds=self.classifier_thresholds)
            decision.activity_label = label
            decision.label_confidence = conf
            decision.label_space = list(self.label_space) if isinstance(self.label_space, list) else None
        return decision

    def _ensure_client(self) -> bool:
        """Create the Ollama client + pull the model on first use.

        Returns True if a usable client is available, False otherwise (caller
        falls back to the deterministic estimate).
        """
        if self._client is not None:
            return True
        if _ollama_lib is None:
            logger.warning("ollama library not installed; using localizer estimate directly")
            return False
        try:
            self._client = _ollama_lib.Client(host=self._settings.base_url)
            self._ensure_model()
            return True
        except Exception as e:
            logger.error("Ollama unavailable (%s); using localizer estimate directly", e)
            self._client = None
            return False

    def _ensure_model(self) -> None:
        if self._model_ready:
            return
        model = self._settings.model
        try:
            available = [m.model for m in self._client.list().models]
            if any(model == m or model == m.split(":")[0] for m in available):
                self._model_ready = True
                return
        except Exception:
            pass
        logger.info("Pulling model '%s' (first use may take a while)...", model)
        self._client.pull(model)
        self._model_ready = True

    def health_check(self) -> bool:
        """True if the Ollama server is reachable."""
        if not self._ensure_client():
            return False
        try:
            self._client.list()
            return True
        except Exception as e:
            logger.error("Ollama health check failed: %s", e)
            return False

    def reason(self, estimate, feature_set=None, previous_decision=None,
               candidates=None) -> DecisionV2:
        """Reason over a localizer estimate and return a DecisionV2.

        Args:
            estimate: a LocationEstimate from the localizer.
            feature_set: the FeatureSet behind the estimate (motion context).
            previous_decision: last DecisionV2 for temporal continuity.
            candidates: optional memory-bank matches (extra prompt context).

        On any failure — or when use_llm is False / Ollama is unavailable — the
        deterministic estimate is passed through unchanged.
        """
        timestamp = time.time()

        if not self.use_llm or not self._ensure_client():
            logger.info("reason: deterministic passthrough (use_llm=%s, ollama=%s) — "
                        "localizer estimate used unchanged",
                        self.use_llm, self._client is not None)
            return self._finalize(DecisionV2.from_estimate(estimate, timestamp),
                                  feature_set, estimate)

        prompt = self._build_user_prompt(
            estimate, feature_set, previous_decision, candidates, timestamp,
        )
        logger.debug("reason: LLM prompt →\n%s", prompt)
        try:
            t0 = time.time()
            response = self._client.chat(
                model=self._settings.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                format="json",
                options={"temperature": 0.1},
            )
            latency_ms = (time.time() - t0) * 1000.0
            raw_text = response.message.content
            logger.info("reason: Ollama '%s' replied in %.0f ms (temp=0.1, format=json)",
                        self._settings.model, latency_ms)
            logger.debug("Raw LLM response: %s", raw_text)
            decision = self._parse_response(raw_text, estimate, timestamp)
            if logger.isEnabledFor(logging.INFO):
                logger.info(
                    "reason: LLM decision x=%.2f(Δ%+.2f) y=%.2f(Δ%+.2f) v=%.2f conf=%.2f "
                    "vs estimate x=%.2f y=%.2f",
                    decision.x_meters, decision.x_meters - estimate.x_meters,
                    decision.y_meters, decision.y_meters - estimate.y_meters,
                    decision.velocity_m_s, decision.signal_confidence,
                    estimate.x_meters, estimate.y_meters,
                )
            return self._finalize(decision, feature_set, estimate)
        except Exception as e:
            logger.error("LLM reasoning failed (%s); using localizer estimate", e)
            return self._finalize(DecisionV2.from_estimate(estimate, timestamp),
                                  feature_set, estimate)

    @staticmethod
    def _build_user_prompt(estimate, feature_set, previous_decision, candidates,
                           timestamp) -> str:
        energies = estimate.node_energies
        energies_str = ", ".join(f"{k}={v:.2f}" for k, v in energies.items()) or "(none)"

        if feature_set is not None:
            breath = float(getattr(feature_set, "breathing_frequency", 0.0))
            energy = float(getattr(feature_set, "total_energy", 0.0))
            doppler = float(getattr(feature_set, "doppler_mean", 0.0))
        else:
            breath = energy = doppler = 0.0

        if previous_decision is not None:
            previous_section = (
                f"- Last position: x={previous_decision.x_meters:.2f}, "
                f"y={previous_decision.y_meters:.2f}\n"
                f"- Last velocity: {previous_decision.velocity_m_s:.2f} m/s"
            )
        else:
            previous_section = "- First inference (no previous state)"

        if candidates:
            memory_section = "\n".join(
                f"  {i}. zone={getattr(c, 'zone', '?')}, "
                f"activity={getattr(c, 'activity', '?')}, "
                f"score={getattr(c, 'score', 0.0):.3f}"
                for i, c in enumerate(candidates, 1)
            )
        else:
            memory_section = "  (none)"

        return _USER_PROMPT_TEMPLATE.format(
            x=estimate.x_meters, y=estimate.y_meters, v=estimate.velocity_m_s,
            conf=estimate.confidence,
            unreal=str(estimate.unrealistic_speed).lower(),
            multi=str(estimate.multipath_reflection_suspect).lower(),
            metric=str(not estimate.non_metric).lower(),
            breath=breath, energy=energy, doppler=doppler,
            energies=energies_str,
            previous_section=previous_section,
            memory_section=memory_section,
            timestamp=round(timestamp, 3),
        )

    @staticmethod
    def _parse_response(raw_text, estimate, timestamp) -> DecisionV2:
        """Parse LLM JSON into a DecisionV2, falling back to the estimate.

        Direct json.loads first, then a nested-object regex, then validate. Any
        failure returns DecisionV2.from_estimate so the pipeline never stalls.
        """
        if not raw_text or not raw_text.strip():
            logger.warning("Empty LLM response; using localizer estimate")
            return DecisionV2.from_estimate(estimate, timestamp)

        data = None
        try:
            data = json.loads(raw_text.strip())
            logger.debug("parse: direct json.loads succeeded")
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                    logger.debug("parse: recovered JSON via nested-brace regex")
                except json.JSONDecodeError:
                    pass

        if not isinstance(data, dict):
            logger.warning("Could not parse JSON from LLM response; using estimate")
            return DecisionV2.from_estimate(estimate, timestamp)

        return AgenticCoreV2._dict_to_decision(data, estimate, timestamp)

    @staticmethod
    def _dict_to_decision(data, estimate, timestamp) -> DecisionV2:
        """Validate a parsed dict into a DecisionV2 (estimate fills any gap)."""
        try:
            coords = data.get("coordinates") or {}
            x = float(coords.get("x_meters", estimate.x_meters))
            y = float(coords.get("y_meters", estimate.y_meters))

            velocity = float(data.get("velocity_m_s", estimate.velocity_m_s))

            conf_in = float(data.get("signal_confidence", estimate.confidence))
            conf = max(0.0, min(1.0, conf_in))
            if conf != conf_in:
                logger.debug("validate: signal_confidence %.3f clamped to [0,1] → %.3f", conf_in, conf)

            flags = data.get("anomaly_flags") or {}
            unreal = bool(flags.get("unrealistic_speed", estimate.unrealistic_speed))
            multi = bool(flags.get(
                "multipath_reflection_suspected", estimate.multipath_reflection_suspect))

            energies = data.get("raw_node_energies")
            if isinstance(energies, dict) and energies:
                energies = {k: float(v) for k, v in energies.items()}
            else:
                energies = dict(estimate.node_energies)

            ts_raw = data.get("timestamp")
            ts = float(ts_raw) if ts_raw is not None else timestamp

            return DecisionV2(
                x_meters=x, y_meters=y, velocity_m_s=velocity,
                signal_confidence=conf,
                unrealistic_speed=unreal,
                multipath_reflection_suspected=multi,
                node_energies=energies, timestamp=ts,
            )
        except (TypeError, ValueError) as e:
            logger.warning("DecisionV2 validation failed (%s); using estimate", e)
            return DecisionV2.from_estimate(estimate, timestamp)