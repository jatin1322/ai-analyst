"""The model boundary: the only thing a provider adapter implements.

The planning loop never sees an SDK type. It opens a session with a system
prompt and a list of tool specs, sends the first message, and after that only
replies to the model's last action. Each model turn comes back as a
`ModelAction`: the one tool the model chose to call and its *unvalidated*
arguments, plus metadata. Validation happens in the planner, never here.

A session is per planning run and holds the provider's native history, so
provider-specific content (such as thinking blocks, which must be returned
verbatim on the next request) never has to survive a round trip through a
provider-neutral transcript.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolSpec:
    """One tool as the model sees it: a name, a description, a JSON schema."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class ModelUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_input_tokens: int | None = None


@dataclass(frozen=True)
class ModelAction:
    """What the model did in one turn. Nothing here has been validated."""

    tool_name: str | None
    arguments: dict[str, Any] | None
    call_id: str | None
    stop_reason: str
    # Characters of free text the model wrote beside (or instead of) a tool
    # call. The text itself is not kept: it is not an instruction to anyone.
    text_chars: int = 0
    # More than one tool call in a turn is malformed: one decision per turn.
    extra_tool_calls: int = 0
    usage: ModelUsage = field(default_factory=ModelUsage)
    latency_ms: float | None = None


class ModelStopped(RuntimeError):
    """The model stopped in a way that must not be parsed: a refusal or a truncation."""

    def __init__(self, stop_reason: str, message: str = "") -> None:
        super().__init__(message or f"the model stopped with {stop_reason!r}")
        self.stop_reason = stop_reason


class ModelUnavailable(RuntimeError):
    """The provider could not be reached or rejected the request. Nothing was planned."""


class PlannerModelSession(Protocol):
    def start(self, user_message: str) -> ModelAction:
        """Send the first user message; return the model's first action."""
        ...

    def reply(self, call_id: str | None, content: str, *, is_error: bool) -> ModelAction:
        """Answer the model's last action (a tool result, or an error), and continue."""
        ...


class PlannerModel(Protocol):
    """A provider behind the planner. Swapping it changes nothing else."""

    name: str

    def open(self, system: str, tools: list[ToolSpec]) -> PlannerModelSession:
        ...
