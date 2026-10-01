"""The planner: model actions in, typed outcomes out.

`Planner` is the provider-independent interface the planning loop drives. It
proposes one `PlannerOutcome` per step; it never executes a tool, never runs a
plan and never judges its own plan valid. `LLMPlanner` implements it over any
`PlannerModel` by parsing each model action through the tool's strict Pydantic
model.

Parsing fails closed. A turn with no tool call, several tool calls, an unknown
tool, or terminal arguments that do not validate raises `MalformedOutput`; the
loop decides whether the model gets its one bounded retry. Nothing is repaired:
malformed JSON is never patched into shape, and a partial plan is never
executed.

Invalid arguments to an *inspection* tool are not malformed output: the request
is passed on as a `ToolRequest` with its raw arguments, and the loop refuses it
against the inspection budget (13.17).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pydantic import ValidationError

from ai_analyst.agent.context import PlannerContext
from ai_analyst.agent.planner.model import ModelAction, PlannerModel, PlannerModelSession
from ai_analyst.agent.planner.prompt import SYSTEM_PROMPT, first_message
from ai_analyst.agent.planner.tools import (
    TOOLS,
    DeclareUnanswerableArgs,
    RequestClarificationArgs,
    RunAnalysisPlanArgs,
    RunInvestigationArgs,
    tool_specs,
)
from ai_analyst.contracts.planner import (
    ClarificationOutcome,
    FinalPlan,
    PlannerOutcome,
    PlanPath,
    Rejected,
    RejectedReason,
    ToolRequest,
)


@dataclass(frozen=True)
class Observation:
    """What the loop tells the planner after a step: a tool result or an error."""

    call_id: str | None
    content: str
    is_error: bool = False


class MalformedOutput(ValueError):
    """A model turn that does not parse into exactly one valid outcome."""

    def __init__(self, code: str, message: str, call_id: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.call_id = call_id


class Planner(Protocol):
    """Proposes one outcome per step. The deterministic loop decides what happens next."""

    def plan(self, question: str, context: PlannerContext) -> PlannerOutcome:
        ...

    def observe(self, observation: Observation) -> PlannerOutcome:
        ...

    @property
    def last_action(self) -> ModelAction | None:
        ...


def validation_summary(exc: ValidationError) -> str:
    """Where and why the arguments failed. Locations and messages only, never inputs."""
    parts = []
    for error in exc.errors(include_input=False, include_url=False)[:10]:
        location = ".".join(str(x) for x in error["loc"]) or "(root)"
        parts.append(f"{location}: {error['msg']}")
    return "; ".join(parts)


class LLMPlanner:
    """A `Planner` over any `PlannerModel`. One instance per planning run."""

    def __init__(self, model: PlannerModel) -> None:
        self.model = model
        self._session: PlannerModelSession | None = None
        self._last: ModelAction | None = None

    @property
    def last_action(self) -> ModelAction | None:
        return self._last

    def plan(self, question: str, context: PlannerContext) -> PlannerOutcome:
        self._session = self.model.open(SYSTEM_PROMPT, tool_specs())
        return self._parse(self._session.start(first_message(question, context)))

    def observe(self, observation: Observation) -> PlannerOutcome:
        if self._session is None:
            raise RuntimeError("observe() before plan()")
        action = self._session.reply(
            observation.call_id, observation.content, is_error=observation.is_error
        )
        return self._parse(action)

    def _parse(self, action: ModelAction) -> PlannerOutcome:
        self._last = action
        if action.tool_name is None:
            raise MalformedOutput(
                "no_tool_call", "call exactly one tool; plain text is not an action"
            )
        if action.extra_tool_calls:
            raise MalformedOutput(
                "several_tool_calls", "call exactly one tool per turn", action.call_id
            )
        tool = TOOLS.get(action.tool_name)
        if tool is None:
            raise MalformedOutput(
                "unknown_tool", f"there is no tool named {action.tool_name!r}", action.call_id
            )
        arguments = action.arguments if isinstance(action.arguments, dict) else None
        if arguments is None:
            raise MalformedOutput(
                "arguments_not_an_object", "tool arguments must be a JSON object", action.call_id
            )
        if not tool.terminal:
            return ToolRequest(tool=tool.name, arguments=arguments, call_id=action.call_id)
        try:
            args = tool.args.model_validate(arguments)
        except ValidationError as exc:
            raise MalformedOutput(
                "invalid_arguments",
                f"{tool.name} arguments do not validate: {validation_summary(exc)}",
                action.call_id,
            ) from exc
        return _terminal(args)


def _terminal(args) -> PlannerOutcome:
    match args:
        case RunAnalysisPlanArgs(plan=plan, edit=None):
            return FinalPlan(path=PlanPath.SEMANTIC, plan=plan)
        case RunAnalysisPlanArgs(edit=edit):
            return FinalPlan(path=PlanPath.EDIT, edit=edit)
        case RunInvestigationArgs(plan=plan, why_not_semantic=why):
            return FinalPlan(path=PlanPath.INVESTIGATION, investigation=plan, why_not_semantic=why)
        case RequestClarificationArgs(request=request):
            return ClarificationOutcome(request=request)
        case DeclareUnanswerableArgs(reason=reason, concept=concept, detail=detail):
            return Rejected(reason=RejectedReason(reason.value), concept=concept, detail=detail)
    raise MalformedOutput("unknown_terminal", "unrecognised terminal action")  # pragma: no cover
