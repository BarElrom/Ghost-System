"""
GHOST System — Agentic Core (Llama 3 Reasoning Engine via Ollama)
Layer 3, Component 2: Fuse sensor data, memory matches, and temporal
context into structured decisions about human presence, location, and
activity.

The Ollama server must be running (docker compose up -d) before using
this module.  On first use, the configured model is auto-pulled (~2 GB
for llama3.2:3b).

Usage:
    from agentic_core import AgenticCore
    core = AgenticCore()
    decision = core.reason(state_vector, candidates)
    print(decision.to_dict())
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field

import numpy as np
import ollama as _ollama_lib

from config import OLLAMA_SETTINGS, OllamaSettings
from feature_extractor import NUM_FEATURES
from memory_bank import Candidate

logger = logging.getLogger("ghost.agentic_core")

# Feature labels for prompt construction.
_FEATURE_LABELS = [
    ("breathing_frequency", "Hz"),
    ("total_energy", ""),
    ("doppler_mean", "Hz"),
    ("variance_rx1", ""),
    ("variance_rx2", ""),
    ("variance_rx3", ""),
]

_SYSTEM_PROMPT = (
    "You are a Wi-Fi CSI sensing analysis engine. "
    "Given sensor features, memory-bank matches, and the previous state, "
    "determine human presence, location, and activity. "
    "Respond with ONLY a JSON object containing these fields: "
    "Target_Detected (bool), Location (string), Activity (string), "
    "Confidence (float 0-1), Timestamp (int, unix ms)."
)

_USER_PROMPT_TEMPLATE = """\
SENSOR DATA:
{sensor_section}

MEMORY MATCHES (cosine similarity):
{memory_section}

PREVIOUS STATE:
{previous_section}

ANALYSIS RULES:
- breathing_frequency in [0.1, 0.5] Hz with non-trivial energy suggests a breathing human
- High doppler_mean (> 1.0 Hz) suggests active movement
- Low total_energy (< 50) with near-zero variances suggests empty room
- Receiver variance asymmetry indicates spatial position relative to receivers
- Consider transition plausibility from the previous state
- If no memory matches are relevant (low scores), use sensor data alone

Respond with ONLY a JSON object:
{{"Target_Detected": bool, "Location": "zone", "Activity": "activity", "Confidence": 0.0-1.0, "Timestamp": {timestamp}}}"""


# ---------------------------------------------------------------------------
# Decision — structured output
# ---------------------------------------------------------------------------
@dataclass
class Decision:
    """A single reasoning output from the Agentic Core.

    Attributes:
        target_detected: Whether a human is detected.
        location: Inferred zone name.
        activity: Inferred activity class.
        confidence: Confidence score (0.0-1.0).
        timestamp_ms: Unix timestamp in milliseconds.
    """

    target_detected: bool = False
    location: str = "Unknown"
    activity: str = "Unknown"
    confidence: float = 0.0
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    def to_dict(self) -> dict:
        """Return the Phase A JSON schema."""
        return {
            "Target_Detected": self.target_detected,
            "Location": self.location,
            "Activity": self.activity,
            "Confidence": self.confidence,
            "Timestamp": self.timestamp_ms,
        }


# ---------------------------------------------------------------------------
# AgenticCore — LLM reasoning engine
# ---------------------------------------------------------------------------
class AgenticCore:
    """LLM-based reasoning layer that produces structured Decisions.

    Connects to an Ollama server, ensures the configured model is
    available (auto-pulls on first use), and provides a ``reason()``
    method that fuses sensor data with memory matches.

    Args:
        settings: Ollama connection settings.  Uses defaults if None.
    """

    def __init__(self, settings: OllamaSettings | None = None):
        self._settings = settings or OLLAMA_SETTINGS
        self._client = _ollama_lib.Client(host=self._settings.base_url)
        self.ensure_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def health_check(self) -> bool:
        """Verify that the Ollama server is reachable.

        Returns:
            True if the server responds to a list request.
        """
        try:
            self._client.list()
            return True
        except Exception as e:
            logger.error("Ollama health check failed: %s", e)
            return False

    def ensure_model(self) -> None:
        """Pull the configured model if it is not already available.

        This is called automatically during ``__init__``.  The first
        invocation downloads the model (~2 GB for llama3.2:3b).
        """
        model = self._settings.model
        try:
            local_models = self._client.list()
            # The list response contains model objects with a 'model' field.
            available = [m.model for m in local_models.models]
            if any(model == m or model == m.split(":")[0] for m in available):
                logger.info("Model '%s' already available", model)
                return
        except Exception:
            pass

        logger.info("Pulling model '%s' (this may take a while)...", model)
        self._client.pull(model)
        logger.info("Model '%s' pulled successfully", model)

    def reason(
        self,
        state_vector: np.ndarray,
        candidates: list[Candidate],
        previous_decision: "Decision | None" = None,
    ) -> Decision:
        """Run LLM reasoning and return a structured Decision.

        Args:
            state_vector: Shape (NUM_FEATURES,) from FeatureExtractor.
            candidates: Ranked memory-bank candidates from MemoryBank.query().
            previous_decision: The last decision for temporal continuity.

        Returns:
            A Decision dataclass.  On any failure, returns a fallback
            Decision with target_detected=False.
        """
        self._validate_vector(state_vector)

        user_prompt = self._build_user_prompt(
            state_vector, candidates, previous_decision,
        )

        try:
            response = self._client.chat(
                model=self._settings.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                format="json",
                options={"temperature": 0.1},
            )
            raw_text = response.message.content
            logger.debug("Raw LLM response: %s", raw_text)
            return self._parse_response(raw_text)

        except Exception as e:
            logger.error("LLM reasoning failed: %s", e)
            return Decision()

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------
    def _build_user_prompt(
        self,
        state_vector: np.ndarray,
        candidates: list[Candidate],
        previous_decision: "Decision | None",
    ) -> str:
        """Construct the user prompt from sensor data and context."""
        # Sensor section
        sensor_lines = []
        for i, (label, unit) in enumerate(_FEATURE_LABELS):
            val = state_vector[i]
            suffix = f" {unit}" if unit else ""
            sensor_lines.append(f"- {label}: {val:.4f}{suffix}")
        sensor_section = "\n".join(sensor_lines)

        # Memory section
        if candidates:
            mem_lines = []
            for rank, c in enumerate(candidates, 1):
                mem_lines.append(
                    f"  {rank}. zone={c.zone}, activity={c.activity}, "
                    f"score={c.score:.4f}"
                )
            memory_section = "\n".join(mem_lines)
        else:
            memory_section = "  (no matches — memory bank is empty)"

        # Previous state section
        if previous_decision is not None:
            previous_section = (
                f"- Last location: {previous_decision.location}\n"
                f"- Last activity: {previous_decision.activity}"
            )
        else:
            previous_section = "- First inference (no previous state)"

        timestamp = int(time.time() * 1000)

        return _USER_PROMPT_TEMPLATE.format(
            sensor_section=sensor_section,
            memory_section=memory_section,
            previous_section=previous_section,
            timestamp=timestamp,
        )

    # ------------------------------------------------------------------
    # Response parser
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_response(raw_text: str) -> Decision:
        """Parse LLM text output into a Decision.

        Tries direct JSON parsing first, then regex extraction for JSON
        wrapped in markdown or explanatory text.  Returns a fallback
        Decision on any failure.
        """
        if not raw_text or not raw_text.strip():
            logger.warning("Empty LLM response, returning fallback")
            return Decision()

        # Primary: direct JSON parse
        data = None
        try:
            data = json.loads(raw_text.strip())
        except json.JSONDecodeError:
            pass

        # Secondary: extract JSON object from surrounding text
        if data is None:
            match = re.search(r"\{[^{}]*\}", raw_text)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    pass

        if data is None or not isinstance(data, dict):
            logger.warning("Could not parse JSON from LLM response")
            return Decision()

        return AgenticCore._dict_to_decision(data)

    @staticmethod
    def _dict_to_decision(data: dict) -> Decision:
        """Convert a parsed dict into a validated Decision."""
        try:
            target = bool(data.get("Target_Detected", False))
            location = str(data.get("Location", "Unknown"))
            activity = str(data.get("Activity", "Unknown"))

            confidence = float(data.get("Confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))

            timestamp_raw = data.get("Timestamp")
            if timestamp_raw is not None:
                timestamp_ms = int(timestamp_raw)
            else:
                timestamp_ms = int(time.time() * 1000)

            return Decision(
                target_detected=target,
                location=location,
                activity=activity,
                confidence=confidence,
                timestamp_ms=timestamp_ms,
            )
        except (TypeError, ValueError) as e:
            logger.warning("Decision validation failed: %s", e)
            return Decision()

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_vector(state_vector: np.ndarray) -> None:
        """Raise if the vector is not a valid state vector."""
        if not isinstance(state_vector, np.ndarray):
            raise TypeError(
                f"Expected np.ndarray, got {type(state_vector).__name__}"
            )
        if state_vector.shape != (NUM_FEATURES,):
            raise ValueError(
                f"Expected shape ({NUM_FEATURES},), got {state_vector.shape}"
            )
