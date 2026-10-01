"""The Anthropic adapter: the only module that knows a provider.

It implements `PlannerModel` over the Messages API with a manual tool loop.
The SDK's tool runner is not used on purpose: every tool call here must pass
through the deterministic planning loop, its budgets and its validators, and
the terminal tools must never execute inside the model conversation.

Request shape, per turn:

* the static system prompt and the fixed tool list, with a prompt-cache
  breakpoint after the system prompt so the stable prefix is reused;
* `tool_choice: auto` with parallel tool use disabled, so a turn carries at
  most one decision. A turn with no tool call is malformed output, handled by
  the loop;
* streaming, with the final message read after the stream closes;
* the full assistant content, thinking blocks included, appended to the
  history before the next request, as the API requires.

Stops that must never be parsed:

* `refusal`: `ModelStopped`, which the loop turns into a rejection.
* `max_tokens` with a tool call present: the input may be a silently truncated
  object. Retried once with twice the cap, then `ModelStopped`.

The SDK is imported lazily, so importing the planner package, and running the
default test suite, never needs it or its credentials. Credentials come from
the SDK's own environment resolution; nothing here reads, stores or logs a key.
"""

from __future__ import annotations

import time
from typing import Any

from ai_analyst.agent.planner.model import (
    ModelAction,
    ModelStopped,
    ModelUnavailable,
    ModelUsage,
    ToolSpec,
)


def _tool_param(spec: ToolSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "description": spec.description,
        "input_schema": spec.input_schema,
    }


def _block_type(block: Any) -> str:
    return getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else "")


def action_from_message(message: Any, latency_ms: float | None = None) -> ModelAction:
    """A provider message reduced to one neutral action. Nothing is validated here."""
    tool_blocks = [b for b in message.content if _block_type(b) == "tool_use"]
    text_chars = sum(len(getattr(b, "text", "") or "") for b in message.content
                     if _block_type(b) == "text")
    usage = getattr(message, "usage", None)
    first = tool_blocks[0] if tool_blocks else None
    return ModelAction(
        tool_name=getattr(first, "name", None) if first else None,
        arguments=getattr(first, "input", None) if first else None,
        call_id=getattr(first, "id", None) if first else None,
        stop_reason=str(getattr(message, "stop_reason", "") or ""),
        text_chars=text_chars,
        extra_tool_calls=max(0, len(tool_blocks) - 1),
        usage=ModelUsage(
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", None),
        ),
        latency_ms=latency_ms,
    )


class AnthropicPlannerSession:
    """One planning conversation, holding the provider's native history."""

    def __init__(self, client: Any, model: str, max_tokens: int, system: str,
                 tools: list[ToolSpec]) -> None:
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.system = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        self.tools = [_tool_param(t) for t in tools]
        self.messages: list[dict[str, Any]] = []
        # Every tool call in the last assistant turn. The API rejects a history
        # in which any of them lacks a result, so each one is answered.
        self._pending: list[str] = []

    def start(self, user_message: str) -> ModelAction:
        self.messages.append({"role": "user", "content": user_message})
        return self._turn()

    def reply(self, call_id: str | None, content: str, *, is_error: bool) -> ModelAction:
        pending, self._pending = self._pending, []
        if call_id is None and not pending:
            # The last turn called no tool; answer it as a plain user message.
            self.messages.append({"role": "user", "content": content})
            return self._turn()
        answered = call_id if call_id is not None else pending[0]
        results = [
            {
                "type": "tool_result",
                "tool_use_id": answered,
                "content": content,
                "is_error": is_error,
            }
        ]
        # A turn with several tool calls is malformed; the others were not run.
        results += [
            {
                "type": "tool_result",
                "tool_use_id": other,
                "content": "not executed: call exactly one tool per turn",
                "is_error": True,
            }
            for other in pending
            if other != answered
        ]
        self.messages.append({"role": "user", "content": results})
        return self._turn()

    def _request(self, max_tokens: int):
        started = time.perf_counter()
        with self.client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            system=self.system,
            tools=self.tools,
            tool_choice={"type": "auto", "disable_parallel_tool_use": True},
            messages=self.messages,
        ) as stream:
            message = stream.get_final_message()
        return message, (time.perf_counter() - started) * 1000

    def _turn(self) -> ModelAction:
        errors = _sdk_errors()
        try:
            message, latency = self._request(self.max_tokens)
            if message.stop_reason == "max_tokens" and any(
                _block_type(b) == "tool_use" for b in message.content
            ):
                message, latency = self._request(self.max_tokens * 2)
        except errors as exc:
            raise ModelUnavailable(type(exc).__name__) from exc
        if message.stop_reason == "refusal":
            raise ModelStopped("refusal")
        if message.stop_reason == "max_tokens" and any(
            _block_type(b) == "tool_use" for b in message.content
        ):
            raise ModelStopped("max_tokens")
        # The whole content goes back, thinking blocks included.
        self.messages.append({"role": "assistant", "content": message.content})
        self._pending = [
            getattr(b, "id", None) for b in message.content if _block_type(b) == "tool_use"
        ]
        return action_from_message(message, latency)


def _sdk_errors() -> tuple[type[BaseException], ...]:
    try:
        import anthropic
    except ImportError:  # pragma: no cover - only reachable without the llm extra
        return ()
    return (anthropic.APIError,)


class AnthropicPlannerModel:
    """`PlannerModel` over the Anthropic Messages API."""

    def __init__(self, model: str, max_tokens: int, client: Any | None = None) -> None:
        self.name = model
        self.max_tokens = max_tokens
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise ModelUnavailable(
                    "the 'anthropic' package is not installed; install the llm extra"
                ) from exc
            self._client = anthropic.Anthropic()
        return self._client

    def open(self, system: str, tools: list[ToolSpec]) -> AnthropicPlannerSession:
        return AnthropicPlannerSession(self.client, self.name, self.max_tokens, system, tools)

    @classmethod
    def from_settings(cls, settings, client: Any | None = None) -> AnthropicPlannerModel:
        return cls(settings.planner_model, settings.planner_max_output_tokens, client)
