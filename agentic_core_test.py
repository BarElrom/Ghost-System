"""
agentic_core_test.py — Integration test for the Agentic Core.

Requires a running Ollama server (docker compose up -d).
The configured model is auto-pulled on first use (~2 GB for llama3.2:3b).

Tests prompt construction, response parsing, input validation, and
end-to-end LLM reasoning.

Usage:
    docker compose up -d
    python agentic_core_test.py
"""

import json
import sys
import logging

import numpy as np

from config import OllamaSettings
from agentic_core import AgenticCore, Decision
from memory_bank import Candidate

passed = 0
failed = 0


def report(name: str, ok: bool, detail: str = ""):
    global passed, failed
    if ok:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        msg = f"  FAIL  {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)


def make_vector(*values) -> np.ndarray:
    """Create a (6,) state vector from positional args."""
    return np.array(values, dtype=np.float64)


def make_candidates() -> list[Candidate]:
    """Create sample memory-bank candidates."""
    return [
        Candidate(zone="Kitchen", activity="Static", label="Static in Kitchen",
                  score=0.95, fingerprint_id="kitchen_static_abc12345"),
        Candidate(zone="Hall", activity="Walking", label="Walking in Hall",
                  score=0.72, fingerprint_id="hall_walking_def67890"),
        Candidate(zone="Bedroom", activity="Breathing", label="Breathing in Bedroom",
                  score=0.55, fingerprint_id="bedroom_breathing_111222"),
    ]


def main():
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    print("=" * 60)
    print("  GHOST AgenticCore — Integration Tests")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Test 1: Health check
    # ------------------------------------------------------------------
    print("\n[1] Health Check")
    try:
        core = AgenticCore()
        ok = core.health_check()
        report("Ollama reachable", ok)
    except Exception as e:
        print(f"\n  FATAL: Cannot connect to Ollama — {e}")
        print("  Make sure the container is running: docker compose up -d")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Test 2: Prompt construction
    # ------------------------------------------------------------------
    print("\n[2] Prompt Construction")
    occupied_vec = make_vector(0.30, 1500.0, 0.80, 0.012, 0.008, 0.015)
    candidates = make_candidates()

    prompt = core._build_user_prompt(occupied_vec, candidates, None)
    report("Contains sensor data",
           "breathing_frequency" in prompt and "total_energy" in prompt)
    report("Contains memory matches",
           "Kitchen" in prompt and "0.9500" in prompt)
    report("Contains analysis rules",
           "ANALYSIS RULES" in prompt)
    report("Contains first inference note",
           "First inference" in prompt or "first inference" in prompt)

    # With previous decision
    prev = Decision(target_detected=True, location="Kitchen", activity="Static",
                    confidence=0.9, timestamp_ms=1000000)
    prompt2 = core._build_user_prompt(occupied_vec, candidates, prev)
    report("Contains previous location",
           "Kitchen" in prompt2 and "Last location" in prompt2)

    # ------------------------------------------------------------------
    # Test 3: Response parsing — valid JSON
    # ------------------------------------------------------------------
    print("\n[3] Response Parsing — Valid JSON")
    valid_json = json.dumps({
        "Target_Detected": True,
        "Location": "Kitchen",
        "Activity": "Static",
        "Confidence": 0.92,
        "Timestamp": 1700000000000,
    })
    d = AgenticCore._parse_response(valid_json)
    report("target_detected is True", d.target_detected is True)
    report("location is Kitchen", d.location == "Kitchen")
    report("activity is Static", d.activity == "Static")
    report("confidence is 0.92", abs(d.confidence - 0.92) < 1e-6)
    report("timestamp preserved", d.timestamp_ms == 1700000000000)

    # ------------------------------------------------------------------
    # Test 4: Response parsing — JSON wrapped in markdown
    # ------------------------------------------------------------------
    print("\n[4] Response Parsing — JSON in Markdown")
    markdown_wrapped = (
        'Here is the analysis:\n```json\n'
        '{"Target_Detected": false, "Location": "Empty", '
        '"Activity": "None", "Confidence": 0.1, "Timestamp": 999}\n```'
    )
    d = AgenticCore._parse_response(markdown_wrapped)
    report("target_detected is False", d.target_detected is False)
    report("location is Empty", d.location == "Empty")

    # ------------------------------------------------------------------
    # Test 5: Response parsing — invalid/garbage input
    # ------------------------------------------------------------------
    print("\n[5] Response Parsing — Invalid Input")
    d = AgenticCore._parse_response("this is not json at all")
    report("Fallback: target_detected=False", d.target_detected is False)
    report("Fallback: location=Unknown", d.location == "Unknown")

    d = AgenticCore._parse_response("")
    report("Empty string: fallback", d.target_detected is False)

    d = AgenticCore._parse_response(None)
    report("None input: fallback", d.target_detected is False)

    # ------------------------------------------------------------------
    # Test 6: Response parsing — missing required fields
    # ------------------------------------------------------------------
    print("\n[6] Response Parsing — Missing Fields")
    partial_json = json.dumps({"Target_Detected": True})
    d = AgenticCore._parse_response(partial_json)
    report("Partial JSON: target_detected=True", d.target_detected is True)
    report("Missing location defaults to Unknown", d.location == "Unknown")
    report("Missing confidence defaults to 0.0", d.confidence == 0.0)

    # Confidence clamping
    over_json = json.dumps({
        "Target_Detected": True, "Location": "X", "Activity": "Y",
        "Confidence": 1.5, "Timestamp": 1000,
    })
    d = AgenticCore._parse_response(over_json)
    report("Confidence clamped to 1.0", d.confidence == 1.0)

    under_json = json.dumps({
        "Target_Detected": True, "Location": "X", "Activity": "Y",
        "Confidence": -0.5, "Timestamp": 1000,
    })
    d = AgenticCore._parse_response(under_json)
    report("Confidence clamped to 0.0", d.confidence == 0.0)

    # ------------------------------------------------------------------
    # Test 7: Input validation — wrong vector shape
    # ------------------------------------------------------------------
    print("\n[7] Input Validation — Wrong Shape")
    try:
        bad_vec = np.array([1.0, 2.0, 3.0])
        core.reason(bad_vec, [])
        report("Rejects wrong shape", False, "no exception raised")
    except ValueError:
        report("Rejects wrong shape", True)

    # ------------------------------------------------------------------
    # Test 8: Input validation — wrong type
    # ------------------------------------------------------------------
    print("\n[8] Input Validation — Wrong Type")
    try:
        core.reason([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [])
        report("Rejects non-ndarray", False, "no exception raised")
    except TypeError:
        report("Rejects non-ndarray", True)

    # ------------------------------------------------------------------
    # Test 9: End-to-end reasoning — occupied room
    # ------------------------------------------------------------------
    print("\n[9] End-to-End Reasoning — Occupied Room")
    occupied_vec = make_vector(0.30, 1500.0, 0.80, 0.012, 0.008, 0.015)
    candidates = make_candidates()
    d = core.reason(occupied_vec, candidates)
    report("Returns a Decision", isinstance(d, Decision))
    report("Has valid location (non-empty)", len(d.location) > 0)
    report("Has valid activity (non-empty)", len(d.activity) > 0)
    report("Confidence in [0, 1]", 0.0 <= d.confidence <= 1.0)
    report("Timestamp is positive int", d.timestamp_ms > 0)
    print(f"         Decision: detected={d.target_detected}, "
          f"location={d.location}, activity={d.activity}, "
          f"confidence={d.confidence:.2f}")

    # ------------------------------------------------------------------
    # Test 10: End-to-end reasoning — empty room
    # ------------------------------------------------------------------
    print("\n[10] End-to-End Reasoning — Empty Room")
    empty_vec = make_vector(0.00, 5.0, 0.01, 0.001, 0.001, 0.001)
    empty_candidates = [
        Candidate(zone="Empty", activity="None", label="Empty room",
                  score=0.98, fingerprint_id="empty_none_aaa111"),
    ]
    d = core.reason(empty_vec, empty_candidates)
    report("Returns a Decision", isinstance(d, Decision))
    report("Confidence in [0, 1]", 0.0 <= d.confidence <= 1.0)
    print(f"         Decision: detected={d.target_detected}, "
          f"location={d.location}, activity={d.activity}, "
          f"confidence={d.confidence:.2f}")

    # ------------------------------------------------------------------
    # Test 11: Decision.to_dict() output format
    # ------------------------------------------------------------------
    print("\n[11] Decision.to_dict() Output Format")
    d = Decision(
        target_detected=True, location="Kitchen", activity="Static",
        confidence=0.88, timestamp_ms=1700000000000,
    )
    out = d.to_dict()
    report("Has Target_Detected key", "Target_Detected" in out)
    report("Has Location key", "Location" in out)
    report("Has Activity key", "Activity" in out)
    report("Has Confidence key", "Confidence" in out)
    report("Has Timestamp key", "Timestamp" in out)
    report("Target_Detected is bool", isinstance(out["Target_Detected"], bool))
    report("Location is str", isinstance(out["Location"], str))
    report("Confidence is float", isinstance(out["Confidence"], float))
    report("Timestamp is int", isinstance(out["Timestamp"], int))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    total = passed + failed
    print(f"  Results: {passed}/{total} passed, {failed} failed")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
