"""The provider adapter, isolated: no SDK, no network, no credentials.

A fake client stands in for the SDK's `client.messages.stream(...)`. The tests
pin the request shape (one decision per turn, cached system prompt, the fixed
tool list), that the full assistant content (thinking blocks included) goes
back into the history, and that refusals and truncations are never parsed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_analyst.agent.planner.anthropic_adapter import AnthropicPlannerModel
from ai_analyst.agent.planner.model import ModelStopped, ModelUnavailable
from ai_analyst.agent.planner.prompt import SYSTEM_PROMPT
from ai_analyst.agent.planner.tools import tool_specs
from ai_analyst.config import Settings
from ai_analyst.contracts.planner import PlannerOutcomeKind
from evals.planner.cases import Q1_OPENING

SRC = str(Path(__file__).resolve().parents[2] / "src")


def block(kind, **fields):
    return SimpleNamespace(type=kind, **fields)


def message(*content, stop_reason="tool_use"):
    return SimpleNamespace(
        content=list(content),
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=120, output_tokens=40, cache_read_input_tokens=100),
    )


class FakeStream:
    def __init__(self, reply):
        self.reply = reply

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.reply


class FakeClient:
    """Answers each request with the next scripted message and records the request."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.messages = self

    def stream(self, **request):
        messages = request["messages"]
        _check_pairing(messages)
        self.requests.append({**request, "messages": [dict(m) for m in messages]})
        return FakeStream(self.replies.pop(0))


def _check_pairing(messages):
    """The API's rule: every tool_use is answered by a tool_result in the next turn."""
    for index, turn in enumerate(messages):
        if turn["role"] != "assistant":
            continue
        uses = {b.id for b in turn["content"] if getattr(b, "type", None) == "tool_use"}
        if not uses or index + 1 >= len(messages):
            continue
        following = messages[index + 1]["content"]
        answered = {r["tool_use_id"] for r in following if isinstance(r, dict)
                    and r.get("type") == "tool_result"} if isinstance(following, list) else set()
        assert uses <= answered, f"unanswered tool_use ids: {uses - answered}"


def tool_call(name, arguments, call_id="toolu_1"):
    return block("tool_use", name=name, input=arguments, id=call_id)


def adapter(replies, max_tokens=4096):
    client = FakeClient(replies)
    return AnthropicPlannerModel("claude-opus-5", max_tokens, client=client), client


def test_the_request_asks_for_one_decision_per_turn():
    model, client = adapter([message(tool_call("list_available_metrics", {}))])
    session = model.open(SYSTEM_PROMPT, tool_specs())
    act = session.start("question")
    request = client.requests[0]
    assert request["model"] == "claude-opus-5"
    assert request["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    assert request["system"][0]["text"] == SYSTEM_PROMPT
    assert request["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert [t["name"] for t in request["tools"]] == [t.name for t in tool_specs()]
    assert act.tool_name == "list_available_metrics"
    assert act.usage.input_tokens == 120 and act.usage.cache_read_input_tokens == 100


def test_the_whole_assistant_turn_goes_back_including_thinking():
    thinking = block("thinking", thinking="", signature="sig")
    first = message(thinking, tool_call("list_available_metrics", {}, "toolu_A"))
    model, client = adapter([first, message(tool_call("list_available_metrics", {}, "toolu_B"))])
    session = model.open(SYSTEM_PROMPT, tool_specs())
    session.start("question")
    session.reply("toolu_A", "<tool_result>...</tool_result>", is_error=False)
    history = client.requests[1]["messages"]
    assert history[1]["role"] == "assistant"
    assert history[1]["content"][0] is thinking
    assert history[2]["content"][0]["type"] == "tool_result"
    assert history[2]["content"][0]["tool_use_id"] == "toolu_A"


def test_a_text_only_turn_is_an_action_without_a_tool():
    model, _ = adapter([message(block("text", text="I think the answer is 42"),
                                stop_reason="end_turn")])
    act = model.open(SYSTEM_PROMPT, tool_specs()).start("q")
    assert act.tool_name is None and act.text_chars > 0


def test_several_tool_calls_are_reported_for_the_planner_to_refuse():
    model, _ = adapter([message(tool_call("a", {}, "1"), tool_call("b", {}, "2"))])
    act = model.open(SYSTEM_PROMPT, tool_specs()).start("q")
    assert act.extra_tool_calls == 1


def test_every_tool_call_of_a_malformed_turn_is_answered(tiny):
    """Two calls in one turn: malformed, and both ids get a result, or the API 400s."""
    from ai_analyst.agent.planner import LLMPlanner, PlanningLoop
    from ai_analyst.contracts.planner import PlannerOutcomeKind as Kind

    replies = [
        message(tool_call("list_available_metrics", {}, "x1"),
                tool_call("inspect_dataset", {}, "x2")),
        message(tool_call("run_analysis_plan",
                          {"plan": Q1_OPENING.model_dump(mode="json")}, "x3")),
    ]
    model, client = adapter(replies)
    ctx = tiny.tool_context()
    loop = PlanningLoop(ctx=ctx, settings=ctx.settings)
    result = loop.run(LLMPlanner(model), "Q1 opening pipeline?", tiny.planner_context(ctx))
    assert result.kind is Kind.FINAL_PLAN
    assert result.malformed_outputs == 1
    answered = {r["tool_use_id"] for r in client.requests[1]["messages"][-1]["content"]}
    assert answered == {"x1", "x2"}


def test_a_refusal_is_never_parsed():
    model, _ = adapter([message(tool_call("run_analysis_plan", {}), stop_reason="refusal")])
    with pytest.raises(ModelStopped) as exc:
        model.open(SYSTEM_PROMPT, tool_specs()).start("q")
    assert exc.value.stop_reason == "refusal"


def test_a_truncated_tool_call_is_retried_once_with_a_larger_cap_then_stops():
    truncated = message(tool_call("run_analysis_plan", {"plan": {}}), stop_reason="max_tokens")
    model, client = adapter([truncated, truncated], max_tokens=1000)
    with pytest.raises(ModelStopped):
        model.open(SYSTEM_PROMPT, tool_specs()).start("q")
    assert [r["max_tokens"] for r in client.requests] == [1000, 2000]


def test_a_truncated_call_that_succeeds_on_retry_is_used():
    good = message(tool_call("list_available_metrics", {}))
    truncated = message(tool_call("run_analysis_plan", {}), stop_reason="max_tokens")
    model, _ = adapter([truncated, good])
    act = model.open(SYSTEM_PROMPT, tool_specs()).start("q")
    assert act.tool_name == "list_available_metrics"


def test_the_adapter_drives_the_real_loop_end_to_end(tiny):
    """A fake provider through the adapter, the planner, the loop, and the gate."""
    from ai_analyst.agent.planner import LLMPlanner, PlanningLoop

    replies = [
        message(tool_call("list_available_metrics", {}, "t1")),
        message(tool_call("run_analysis_plan",
                          {"plan": Q1_OPENING.model_dump(mode="json")}, "t2")),
    ]
    model, client = adapter(replies)
    ctx = tiny.tool_context()
    loop = PlanningLoop(ctx=ctx, settings=ctx.settings)
    result = loop.run(LLMPlanner(model), "What was Q1 opening pipeline?", tiny.planner_context(ctx))
    assert result.kind is PlannerOutcomeKind.FINAL_PLAN
    assert result.turns[0].input_tokens == 120
    # The tool result was returned against the tool call's id.
    assert client.requests[1]["messages"][-1]["content"][0]["tool_use_id"] == "t1"


def test_the_model_is_configuration():
    settings = Settings(planner_model="claude-sonnet-5", planner_max_output_tokens=8000)
    model = AnthropicPlannerModel.from_settings(settings, client=FakeClient([]))
    assert model.name == "claude-sonnet-5" and model.max_tokens == 8000


def test_importing_the_planner_never_imports_a_provider_sdk():
    code = (
        "import sys; import ai_analyst.agent.planner, ai_analyst.agent.planner.loop, "
        "ai_analyst.agent.planner.anthropic_adapter, ai_analyst.agent.planner.evaluation; "
        "print('anthropic' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True, env={"PYTHONPATH": SRC})
    assert out.stdout.strip() == "False"


def test_a_missing_sdk_is_model_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)
    model = AnthropicPlannerModel("claude-opus-5", 1000)
    with pytest.raises(ModelUnavailable):
        model.open(SYSTEM_PROMPT, tool_specs())


def test_the_semantic_engine_does_not_depend_on_the_planner():
    code = (
        "import sys, ai_analyst.semantic.execute, ai_analyst.semantic.gate, "
        "ai_analyst.agent.tools.surface; "
        "print(any(m.startswith('ai_analyst.agent.planner') for m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True, env={"PYTHONPATH": SRC})
    assert out.stdout.strip() == "False"

