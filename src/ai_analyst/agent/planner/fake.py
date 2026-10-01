"""A scripted model for network-free tests and evaluation replay.

`ScriptedModel` plays back a fixed list of actions, one per turn, regardless of
what the loop sends it; it records what it was sent so a test can inspect the
conversation. `ScriptedPolicy` chooses each action from the conversation so far,
which is how an evaluation replays a golden plan after the tool calls it needs.

Neither calls a network, reads a credential, or depends on any provider SDK.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from ai_analyst.agent.planner.model import ModelAction, ModelStopped, ModelUsage, ToolSpec


def action(tool: str | None, arguments: Any = None, **kwargs) -> ModelAction:
    """A model action, for scripts. `arguments` may be a Pydantic model or a dict."""
    if hasattr(arguments, "model_dump"):
        arguments = arguments.model_dump(mode="json", exclude_none=True)
    return ModelAction(
        tool_name=tool,
        arguments=arguments if tool is not None else None,
        call_id=kwargs.pop("call_id", f"call_{tool}" if tool else None),
        stop_reason=kwargs.pop("stop_reason", "tool_use" if tool else "end_turn"),
        usage=kwargs.pop("usage", ModelUsage(input_tokens=0, output_tokens=0)),
        **kwargs,
    )


def stop(reason: str) -> ModelAction:
    """A turn that stops before any action can be parsed."""
    return ModelAction(tool_name=None, arguments=None, call_id=None, stop_reason=reason)


@dataclass
class Transcript:
    system: str = ""
    tools: list[ToolSpec] = field(default_factory=list)
    user_messages: list[str] = field(default_factory=list)
    replies: list[tuple[str | None, str, bool]] = field(default_factory=list)


class _Session:
    def __init__(self, next_action: Callable[[Transcript], ModelAction],
                 transcript: Transcript) -> None:
        self._next = next_action
        self.transcript = transcript

    def _emit(self) -> ModelAction:
        chosen = self._next(self.transcript)
        if chosen.tool_name is None and chosen.stop_reason in ("refusal", "max_tokens"):
            raise ModelStopped(chosen.stop_reason)
        return chosen

    def start(self, user_message: str) -> ModelAction:
        self.transcript.user_messages.append(user_message)
        return self._emit()

    def reply(self, call_id: str | None, content: str, *, is_error: bool) -> ModelAction:
        self.transcript.replies.append((call_id, content, is_error))
        return self._emit()


class ScriptedModel:
    """Plays back actions in order. Running out is a test error, not a model choice."""

    name = "scripted"

    def __init__(self, actions: Iterable[ModelAction]) -> None:
        self._actions = list(actions)
        self.transcripts: list[Transcript] = []

    def open(self, system: str, tools: list[ToolSpec]) -> _Session:
        transcript = Transcript(system=system, tools=list(tools))
        self.transcripts.append(transcript)
        queue = iter(self._actions)

        def next_action(_: Transcript) -> ModelAction:
            try:
                return next(queue)
            except StopIteration as exc:
                raise AssertionError("the scripted model ran out of actions") from exc

        return _Session(next_action, transcript)


class ScriptedPolicy:
    """Chooses each action from the transcript so far. For evaluation replay."""

    name = "scripted-policy"

    def __init__(self, policy: Callable[[Transcript], ModelAction]) -> None:
        self._policy = policy
        self.transcripts: list[Transcript] = []

    def open(self, system: str, tools: list[ToolSpec]) -> _Session:
        transcript = Transcript(system=system, tools=list(tools))
        self.transcripts.append(transcript)
        return _Session(self._policy, transcript)
