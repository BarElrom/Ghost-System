"""Phase 4 tests — agentic_core_v2 (coordinates/velocity reasoning).

Exercises the LLM contract WITHOUT a live Ollama server: a fake client returns
canned chat responses so we can test the prompt path, JSON parsing, validation,
and every fallback branch deterministically. Mirrors the v1 agentic_core tests.

Run:  python v2/tests/test_agentic_core_v2.py
"""

import sys

from _harness import Harness

from v2.ghost.agentic_core_v2 import AgenticCoreV2, DecisionV2
from v2.ghost.localizer import LocationEstimate


class _Msg:
    def __init__(self, content):
        self.message = type("M", (), {"content": content})()


class _FakeClient:
    """Records the last chat() call and returns a scripted response."""

    def __init__(self, content, raise_on_chat=False):
        self._content = content
        self._raise = raise_on_chat
        self.last_messages = None
        self.last_kwargs = None

    def chat(self, model, messages, **kwargs):
        self.last_messages = messages
        self.last_kwargs = kwargs
        if self._raise:
            raise RuntimeError("boom")
        return _Msg(self._content)


def _estimate(**kw) -> LocationEstimate:
    base = dict(
        x_meters=0.5, y_meters=3.0, velocity_m_s=0.4, confidence=0.6,
        node_energies={"RX1": 12.0, "RX2": 33.0, "RX3": 20.0},
        unrealistic_speed=False, multipath_reflection_suspect=False, non_metric=False,
    )
    base.update(kw)
    return LocationEstimate(**base)


class _FS:
    breathing_frequency = 0.3
    total_energy = 1500.0
    doppler_mean = 0.8


def test_use_llm_false_passthrough(h: Harness) -> None:
    core = AgenticCoreV2(use_llm=False)
    d = core.reason(_estimate(), _FS())
    h.expect("returns DecisionV2", isinstance(d, DecisionV2))
    h.expect("x passthrough", d.x_meters == 0.5)
    h.expect("y passthrough", d.y_meters == 3.0)
    h.expect("velocity passthrough", d.velocity_m_s == 0.4)
    h.expect("confidence passthrough", d.signal_confidence == 0.6)
    h.expect("energies passthrough", d.node_energies["RX2"] == 33.0)


def test_valid_llm_json(h: Harness) -> None:
    content = (
        '{"timestamp": 1700000000.5, '
        '"coordinates": {"x_meters": 1.25, "y_meters": 2.5}, '
        '"velocity_m_s": 0.9, "signal_confidence": 0.8, '
        '"anomaly_flags": {"unrealistic_speed": false, '
        '"multipath_reflection_suspected": true}, '
        '"raw_node_energies": {"RX1": 12.0, "RX2": 33.0, "RX3": 20.0}}'
    )
    core = AgenticCoreV2(client=_FakeClient(content))
    d = core.reason(_estimate(), _FS())
    h.expect("parsed x", d.x_meters == 1.25)
    h.expect("parsed y", d.y_meters == 2.5)
    h.expect("parsed velocity", d.velocity_m_s == 0.9)
    h.expect("parsed confidence", d.signal_confidence == 0.8)
    h.expect("parsed multipath flag", d.multipath_reflection_suspected is True)
    h.expect("parsed timestamp", d.timestamp == 1700000000.5)


def test_json_wrapped_in_text(h: Harness) -> None:
    content = (
        "Here is the result:\n```json\n"
        '{"coordinates": {"x_meters": -0.5, "y_meters": 4.0}, '
        '"velocity_m_s": 0.1, "signal_confidence": 0.3, '
        '"anomaly_flags": {"unrealistic_speed": false, '
        '"multipath_reflection_suspected": false}}\n```\nDone.'
    )
    core = AgenticCoreV2(client=_FakeClient(content))
    d = core.reason(_estimate(), _FS())
    h.expect("regex-extracted x", d.x_meters == -0.5)
    h.expect("regex-extracted y", d.y_meters == 4.0)


def test_confidence_clamped(h: Harness) -> None:
    content = ('{"coordinates": {"x_meters": 0.0, "y_meters": 1.0}, '
               '"velocity_m_s": 0.0, "signal_confidence": 5.0, '
               '"anomaly_flags": {}}')
    core = AgenticCoreV2(client=_FakeClient(content))
    d = core.reason(_estimate(), _FS())
    h.expect("confidence clamped to 1.0", d.signal_confidence == 1.0)


def test_missing_fields_fall_back_to_estimate(h: Harness) -> None:
    content = '{"velocity_m_s": 2.2}'
    core = AgenticCoreV2(client=_FakeClient(content))
    d = core.reason(_estimate(x_meters=0.7, y_meters=3.3), _FS())
    h.expect("velocity from llm", d.velocity_m_s == 2.2)
    h.expect("x from estimate", d.x_meters == 0.7)
    h.expect("y from estimate", d.y_meters == 3.3)
    h.expect("energies from estimate", d.node_energies["RX1"] == 12.0)


def test_empty_response_falls_back(h: Harness) -> None:
    core = AgenticCoreV2(client=_FakeClient("   "))
    d = core.reason(_estimate(), _FS())
    h.expect("empty -> estimate x", d.x_meters == 0.5)
    h.expect("empty -> estimate conf", d.signal_confidence == 0.6)


def test_garbage_response_falls_back(h: Harness) -> None:
    core = AgenticCoreV2(client=_FakeClient("not json at all"))
    d = core.reason(_estimate(), _FS())
    h.expect("garbage -> estimate x", d.x_meters == 0.5)


def test_chat_exception_falls_back(h: Harness) -> None:
    core = AgenticCoreV2(client=_FakeClient("{}", raise_on_chat=True))
    d = core.reason(_estimate(velocity_m_s=0.42), _FS())
    h.expect("exception -> estimate velocity", d.velocity_m_s == 0.42)


def test_prompt_contains_context(h: Harness) -> None:
    client = _FakeClient('{"coordinates": {"x_meters": 0, "y_meters": 1}, '
                         '"velocity_m_s": 0, "signal_confidence": 0.5, '
                         '"anomaly_flags": {}}')
    core = AgenticCoreV2(client=client)
    core.reason(_estimate(non_metric=True), _FS())
    user = client.last_messages[1]["content"]
    h.expect("prompt has features", "breathing_frequency" in user)
    h.expect("prompt has energies", "RX2=33.00" in user)
    h.expect("prompt marks non-metric", "position_is_metric: false" in user)
    h.expect("chat asked for json", client.last_kwargs.get("format") == "json")


def test_previous_and_memory_context(h: Harness) -> None:
    client = _FakeClient('{"coordinates": {"x_meters": 0, "y_meters": 1}, '
                         '"velocity_m_s": 0, "signal_confidence": 0.5, '
                         '"anomaly_flags": {}}')
    core = AgenticCoreV2(client=client)
    prev = DecisionV2(x_meters=1.1, y_meters=2.2, velocity_m_s=0.3)

    class _Cand:
        zone, activity, score = "Kitchen", "Static", 0.91

    core.reason(_estimate(), _FS(), previous_decision=prev, candidates=[_Cand()])
    user = client.last_messages[1]["content"]
    h.expect("prompt has previous position", "Last position" in user)
    h.expect("prompt has memory match", "Kitchen" in user)


def test_decision_to_dict_schema(h: Harness) -> None:
    d = DecisionV2.from_estimate(_estimate(), timestamp=1700000000.0)
    out = d.to_dict()
    h.expect("has timestamp", out["timestamp"] == 1700000000.0)
    h.expect("has coordinates.x", out["coordinates"]["x_meters"] == 0.5)
    h.expect("has velocity", out["velocity_m_s"] == 0.4)
    h.expect("has confidence", out["signal_confidence"] == 0.6)
    h.expect("has anomaly flags", set(out["anomaly_flags"]) ==
             {"unrealistic_speed", "multipath_reflection_suspected"})
    h.expect("has raw energies", out["raw_node_energies"]["RX3"] == 20.0)


def main() -> int:
    h = Harness("agentic_core_v2 (coordinates/velocity)")
    h.case("use_llm_false_passthrough", test_use_llm_false_passthrough)
    h.case("valid_llm_json", test_valid_llm_json)
    h.case("json_wrapped_in_text", test_json_wrapped_in_text)
    h.case("confidence_clamped", test_confidence_clamped)
    h.case("missing_fields_fall_back_to_estimate", test_missing_fields_fall_back_to_estimate)
    h.case("empty_response_falls_back", test_empty_response_falls_back)
    h.case("garbage_response_falls_back", test_garbage_response_falls_back)
    h.case("chat_exception_falls_back", test_chat_exception_falls_back)
    h.case("prompt_contains_context", test_prompt_contains_context)
    h.case("previous_and_memory_context", test_previous_and_memory_context)
    h.case("decision_to_dict_schema", test_decision_to_dict_schema)
    return h.done()


if __name__ == "__main__":
    sys.exit(main())