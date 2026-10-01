"""AnalysisTrace: one deterministic object that explains an answer end to end
(WP8, ARCHITECTURE 13.10, 13.14-13.15, 13.21-13.23).

The interview question this contract answers: for any number the system
reports, why is it what it is, and why can it be trusted? Every section here
is read off an object the deterministic core already produced -- a gate
verdict, a `ResultSet.compilation`, a `TrustAssessment`, a `ConceptBindings` --
and nothing in this module computes a new value or calls a model. Building the
same inputs twice must produce the same trace (CLAUDE.md 1.1, 1.3): no field
here may be filled from a clock, a random id, or model prose.

A trace never carries a tool result's content, only its metadata (name,
status, codes). The planner section is a summary of *how the model behaved*,
not of *what it saw* -- the LLM boundary the loop itself enforces
(ARCHITECTURE 13.23) is preserved here rather than re-opened for explanation.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai_analyst.contracts.answer import ProvenanceReport
from ai_analyst.contracts.binding import BindingStatus, EvidenceKind, GrantKind
from ai_analyst.contracts.investigation import InvestigationPlan
from ai_analyst.contracts.plan import AnalysisPlan, AnalysisStance
from ai_analyst.contracts.planner import PlannerOutcomeKind, PlanPath
from ai_analyst.contracts.rejection import PlanValidation
from ai_analyst.contracts.result import (
    AttributionRead,
    ResolvedSnapshot,
    ResultColumn,
    TrustAssessment,
)
from ai_analyst.contracts.session import CarryForwardReport


class PlannerTrace(BaseModel):
    """How the model behaved reaching this plan. Metadata only, never a tool
    result's content (ARCHITECTURE 13.23): tool names in call order, turn and
    repair counts, the rejection codes seen (including ones later repaired),
    and token totals.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    outcome_kind: PlannerOutcomeKind
    path: PlanPath | None = None
    turns: int
    tool_calls: tuple[str, ...] = ()
    repairs: int = 0
    malformed_outputs: int = 0
    rejection_codes: tuple[str, ...] = ()
    input_tokens: int = 0
    output_tokens: int = 0
    context_tokens: int = 0


class EvidenceUsed(BaseModel):
    """One piece of evidence behind a binding: its kind and where it came from."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: EvidenceKind
    source: str = ""


class BindingTrace(BaseModel):
    """One concept the plan touched: its column(s), status and evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    concept: str
    columns: tuple[str, ...] = ()
    status: BindingStatus
    evidence: tuple[EvidenceUsed, ...] = ()
    caveats: tuple[str, ...] = ()


class GrantTrace(BaseModel):
    """One usage grant the compiled query relied on."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    grant_id: str
    kind: GrantKind
    purposes: tuple[str, ...] = ()
    source: str = ""


class TimeTrace(BaseModel):
    """Every snapshot and attribution read the computation actually used."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    resolved_snapshots: tuple[ResolvedSnapshot, ...] = ()
    attribution_reads: tuple[AttributionRead, ...] = ()
    stance: AnalysisStance
    knowledge_cutoff: date | None = None


class ComputedQuery(BaseModel):
    """One compiled, executed query: its SQL, id, and the shape of its result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query_id: str
    compiled_sql: str | None = None
    columns: tuple[ResultColumn, ...] = ()
    row_count: int = 0


class ComputationTrace(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    queries: tuple[ComputedQuery, ...] = ()


class ProvenanceSection(BaseModel):
    """The scan report when a draft was supplied, otherwise a note that none was."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    report: ProvenanceReport | None = None
    note: str = ""

    @model_validator(mode="after")
    def _one_or_the_other(self) -> ProvenanceSection:
        if self.report is None and not self.note:
            raise ValueError("provenance section needs a report or a note explaining its absence")
        return self


class AnalysisTrace(BaseModel):
    """One deterministic explanation of one executed plan and its results.

    Every field traces to an object the deterministic core already built:
    nothing here is computed, and nothing here is a tool result's content.
    `capture_resolution` and `family_lineage` are reserved extension points
    for later work packages; both are free-form mappings until their own
    milestone defines a schema, and both default to empty so a trace built
    today has nothing to migrate later.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    question: str

    # ------------------------------------------------------------- planner
    planner: PlannerTrace | None = None

    # ---------------------------------------------------------------- plan
    plan_path: PlanPath
    plan: AnalysisPlan | None = None
    investigation: InvestigationPlan | None = None
    carry_forward: CarryForwardReport | None = None

    # ---------------------------------------------------------------- gate
    gate: PlanValidation

    # ------------------------------------------------------------ bindings
    bindings: tuple[BindingTrace, ...] = ()
    grants: tuple[GrantTrace, ...] = ()

    # ----------------------------------------------------------------- time
    time: TimeTrace

    # ----------------------------------------------------------- computation
    computation: ComputationTrace

    # ---------------------------------------------------------------- trust
    trust: TrustAssessment

    # ----------------------------------------------------------- provenance
    provenance: ProvenanceSection

    # ------------------------------------------------------- reserved (later WPs)
    # Filled by a future capture/replay work package. A free-form mapping,
    # reserved rather than typed, until that package defines its own schema.
    capture_resolution: dict[str, Any] = Field(default_factory=dict)
    # Filled by a future metric/feature lineage work package. Same reservation.
    family_lineage: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _one_plan_per_path(self) -> AnalysisTrace:
        if self.plan_path is PlanPath.INVESTIGATION:
            if self.investigation is None or self.plan is not None:
                raise ValueError("an investigation trace carries exactly one investigation plan")
        else:
            if self.plan is None or self.investigation is not None:
                raise ValueError(f"a {self.plan_path.value} trace carries exactly one AnalysisPlan")
        return self

    @property
    def executable(self) -> AnalysisPlan | InvestigationPlan:
        return self.investigation if self.plan_path is PlanPath.INVESTIGATION else self.plan
