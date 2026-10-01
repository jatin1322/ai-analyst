"""Planner outcomes and the planning run record (ARCHITECTURE 13.6).

The planner proposes; the deterministic system decides. Every model output is
parsed into exactly one `PlannerOutcome`, and anything that does not parse is
malformed output, which fails closed:

* `TOOL_REQUEST`: inspect the dataset through one of the bounded tools, then
  plan again. Never terminal.
* `FINAL_PLAN`: a typed plan (semantic, investigation, or an edit of the
  session's active plan). It is final only after the gate has passed it; the
  planner cannot declare its own plan valid.
* `CLARIFICATION_REQUEST`: ask rather than guess (13.12).
* `REJECTED`: the question cannot be answered safely, or the planning run
  failed closed. The reason says which.

No field anywhere here carries a computed number, a trust tier, or SQL. The
run record is metadata only: turn counts, tool names, outcome kinds, rejection
codes, token counts and latency. It never holds a prompt, a tool result, or a
value read from the dataset.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai_analyst.contracts.concepts import BusinessConcept
from ai_analyst.contracts.investigation import InvestigationPlan
from ai_analyst.contracts.plan import AnalysisPlan
from ai_analyst.contracts.rejection import PlanValidation
from ai_analyst.contracts.session import CarryForwardReport, PlanEdit
from ai_analyst.contracts.tools import ClarificationRequest


class PlannerOutcomeKind(StrEnum):
    FINAL_PLAN = "final_plan"
    TOOL_REQUEST = "tool_request"
    CLARIFICATION_REQUEST = "clarification_request"
    REJECTED = "rejected"


class PlanPath(StrEnum):
    SEMANTIC = "semantic"
    INVESTIGATION = "investigation"
    # A follow-up: a typed edit of the session's active plan (13.13).
    EDIT = "edit"


class NotSemanticReason(StrEnum):
    """Why an investigation, when the semantic path comes first (13.6)."""

    NO_METRIC_DEFINES_IT = "no_metric_defines_it"
    NEEDS_DERIVED_FEATURE = "needs_derived_feature"
    NEEDS_ASSOCIATION = "needs_association"


class RejectedReason(StrEnum):
    # The planner's own judgement that the question cannot be answered safely.
    CONCEPT_UNAVAILABLE = "concept_unavailable"
    TEMPORAL_VIOLATION = "temporal_violation"
    UNSUPPORTED_ANALYSIS = "unsupported_analysis"
    OUT_OF_COVERAGE = "out_of_coverage"
    # Imposed by the planning loop when the run fails closed.
    MALFORMED_OUTPUT = "malformed_output"
    VALIDATION_FAILED = "validation_failed"
    TURN_BUDGET_EXHAUSTED = "turn_budget_exhausted"
    CONTEXT_BUDGET_EXHAUSTED = "context_budget_exhausted"
    MODEL_REFUSED = "model_refused"
    MODEL_TRUNCATED = "model_truncated"
    MODEL_UNAVAILABLE = "model_unavailable"

    @property
    def chosen_by_model(self) -> bool:
        return self in MODEL_REJECTION_REASONS


MODEL_REJECTION_REASONS = frozenset(
    {
        RejectedReason.CONCEPT_UNAVAILABLE,
        RejectedReason.TEMPORAL_VIOLATION,
        RejectedReason.UNSUPPORTED_ANALYSIS,
        RejectedReason.OUT_OF_COVERAGE,
    }
)


class _Outcome(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ToolRequest(_Outcome):
    """Inspect before deciding. The arguments are validated by the tool's own model."""

    kind: Literal[PlannerOutcomeKind.TOOL_REQUEST] = PlannerOutcomeKind.TOOL_REQUEST
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    call_id: str | None = None


class FinalPlan(_Outcome):
    """A typed plan the gate has passed. Exactly one plan field is set, per path."""

    kind: Literal[PlannerOutcomeKind.FINAL_PLAN] = PlannerOutcomeKind.FINAL_PLAN
    path: PlanPath
    plan: AnalysisPlan | None = None
    investigation: InvestigationPlan | None = None
    # For an edit: the edit itself, and the plan it produced after `apply_edit`.
    edit: PlanEdit | None = None
    carry_forward: CarryForwardReport | None = None
    why_not_semantic: NotSemanticReason | None = None
    # The gate's verdict, attached by the loop. Never supplied by the model.
    validation: PlanValidation | None = None

    @model_validator(mode="after")
    def _one_plan_per_path(self) -> FinalPlan:
        if self.path is PlanPath.INVESTIGATION:
            if self.investigation is None or self.plan is not None or self.edit is not None:
                raise ValueError("an investigation outcome carries exactly one investigation")
            if self.why_not_semantic is None:
                raise ValueError("an investigation must say why the semantic path does not apply")
        elif self.path is PlanPath.SEMANTIC:
            if self.plan is None or self.investigation is not None or self.edit is not None:
                raise ValueError("a semantic outcome carries exactly one analysis plan")
        elif self.edit is None or self.investigation is not None:
            raise ValueError("an edit outcome carries exactly one plan edit")
        return self

    @property
    def executable(self) -> AnalysisPlan | InvestigationPlan | None:
        """The plan the deterministic pipeline runs, once the gate has passed it."""
        return self.investigation if self.path is PlanPath.INVESTIGATION else self.plan


class ClarificationOutcome(_Outcome):
    kind: Literal[PlannerOutcomeKind.CLARIFICATION_REQUEST] = (
        PlannerOutcomeKind.CLARIFICATION_REQUEST
    )
    request: ClarificationRequest


class Rejected(_Outcome):
    kind: Literal[PlannerOutcomeKind.REJECTED] = PlannerOutcomeKind.REJECTED
    reason: RejectedReason
    concept: BusinessConcept | None = None
    detail: str = ""
    # Filled deterministically from the dataset for an unavailable concept.
    available_alternatives: list[str] = Field(default_factory=list)
    # The last gate verdict, when the run ended because repairs ran out.
    validation: PlanValidation | None = None


type PlannerOutcome = Annotated[
    ToolRequest | FinalPlan | ClarificationOutcome | Rejected,
    Field(discriminator="kind"),
]

TERMINAL_KINDS = frozenset(
    {
        PlannerOutcomeKind.FINAL_PLAN,
        PlannerOutcomeKind.CLARIFICATION_REQUEST,
        PlannerOutcomeKind.REJECTED,
    }
)


# --------------------------------------------------------------- run record


class TurnStatus(StrEnum):
    TOOL_EXECUTED = "tool_executed"
    TOOL_REFUSED = "tool_refused"
    MALFORMED = "malformed"
    GATE_REJECTED = "gate_rejected"
    BOUNDARY_REJECTED = "boundary_rejected"
    ACCEPTED = "accepted"
    STOPPED = "stopped"


class TurnRecord(BaseModel):
    """One model call and what the loop did with it. Metadata only."""

    model_config = ConfigDict(frozen=True)

    turn: int
    action: str | None = None
    outcome_kind: PlannerOutcomeKind | None = None
    status: TurnStatus
    # Rejection or error codes, never messages carrying data.
    codes: tuple[str, ...] = ()
    tool_result_type: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float | None = None


class PlanningResult(BaseModel):
    """A finished planning run: one terminal outcome and how it was reached."""

    model_config = ConfigDict(frozen=True)

    outcome: Annotated[
        FinalPlan | ClarificationOutcome | Rejected, Field(discriminator="kind")
    ]
    turns: tuple[TurnRecord, ...] = ()
    tool_calls: int = 0
    repairs: int = 0
    malformed_outputs: int = 0
    context_tokens: int = 0

    @property
    def kind(self) -> PlannerOutcomeKind:
        return self.outcome.kind

    def codes(self) -> set[str]:
        """Every rejection or error code seen during the run, including repaired ones."""
        return {c for t in self.turns for c in t.codes}

    def actions(self) -> list[str]:
        return [t.action for t in self.turns if t.action]
