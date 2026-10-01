"""The investigation plan (ARCHITECTURE 13.8).

A second plan type, for questions the metric registry does not define. It is
plan-as-data in exactly the sense `AnalysisPlan` is: a closed vocabulary of
populations, variables, groupings and operations, each compiled by
deterministic code over the same concept resolver, the same stance allowlist
and the same monetary boundary.

It is **composable**, not a list of questions. A novel question becomes a
combination of:

* a **population**: a cohort frozen at one snapshot (5.4), observed over a
  window of later or earlier snapshots;
* **variables**: per-opportunity values, each read from a concept, a physical
  column, or a *derived feature* recomputed from snapshot history;
* **groupings**: how the units are split, optionally binned at stated edges;
* one **operation** from a fixed statistical set;
* an optional **comparison** against a reference group;
* **evidence requirements** that decide what may be reported.

It is **not a program**. No field holds an expression, a formula, SQL, code, or
a reference to another plan, and the schema is non-recursive, so an expression
tree is unrepresentable rather than discouraged. The only numbers a plan may
carry are bin edges, a support floor and a limit: structural parameters, never
results.

An investigation result is at most trust tier B, always (13.10). Its definition
is ad hoc even when every input is confirmed.
"""

from __future__ import annotations

import uuid
from datetime import date
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from ai_analyst.contracts.concepts import BusinessConcept
from ai_analyst.contracts.plan import AnalysisStance, Filter, Period, SnapshotSelection
from ai_analyst.contracts.result import SnapshotRule
from ai_analyst.contracts.status import OpportunityStatus

type VariableId = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,31}$")]


def new_plan_id() -> str:
    return f"p{uuid.uuid4().hex[:12]}"


class HypothesisKind(StrEnum):
    """What kind of question is being asked."""

    ASSOCIATION = "association"
    DIFFERENCE = "difference"
    DISTRIBUTION = "distribution"
    COMPOSITION = "composition"
    TREND = "trend"


class AnalysisUnit(StrEnum):
    """What one row of the analysis is.

    Only the opportunity in v1. Snapshot rows of one opportunity are not
    independent observations, and treating them as if they were is the error
    the grain rule exists to prevent.
    """

    OPPORTUNITY = "opportunity"


class DerivedFeature(StrEnum):
    """Per-opportunity features computed from snapshot history over the window.

    Each is recomputed by the engine from the snapshots it can see (5.9). None
    reads a precomputed counter, even where the export carries one.
    """

    # The concept's value at the first or last snapshot of the window.
    VALUE_AT_START = "value_at_start"
    VALUE_AT_END = "value_at_end"
    # How many times the concept's value changed between adjacent snapshots.
    CHANGE_COUNT = "change_count"
    CHANGED = "changed"
    # Close-date movement (expected_close_date).
    PUSH_COUNT = "push_count"
    SLIPPED = "slipped"
    PULLED_IN = "pulled_in"
    # The trace outcome through the window end (5.4). Retrospective only.
    FINAL_STATE = "final_state"

    @property
    def needs_concept(self) -> bool:
        return self in _NEEDS_CONCEPT

    @property
    def fixed_concept(self) -> BusinessConcept | None:
        """The concept a feature is always computed over, when it has one."""
        return _FIXED_CONCEPT.get(self)

    @property
    def is_boolean(self) -> bool:
        return self in (
            DerivedFeature.CHANGED,
            DerivedFeature.SLIPPED,
            DerivedFeature.PULLED_IN,
        )

    @property
    def is_count(self) -> bool:
        return self in (DerivedFeature.CHANGE_COUNT, DerivedFeature.PUSH_COUNT)

    @property
    def reads_outcome(self) -> bool:
        """Whether the feature describes what eventually happened."""
        return self is DerivedFeature.FINAL_STATE


_NEEDS_CONCEPT = frozenset(
    {
        DerivedFeature.VALUE_AT_START,
        DerivedFeature.VALUE_AT_END,
        DerivedFeature.CHANGE_COUNT,
        DerivedFeature.CHANGED,
    }
)
_FIXED_CONCEPT: dict[DerivedFeature, BusinessConcept] = {
    DerivedFeature.PUSH_COUNT: BusinessConcept.EXPECTED_CLOSE_DATE,
    DerivedFeature.SLIPPED: BusinessConcept.EXPECTED_CLOSE_DATE,
    DerivedFeature.PULLED_IN: BusinessConcept.EXPECTED_CLOSE_DATE,
    DerivedFeature.FINAL_STATE: BusinessConcept.OPPORTUNITY_STATUS,
}


class DerivedFeatureRef(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    feature: DerivedFeature
    concept: BusinessConcept | None = None

    @model_validator(mode="after")
    def _concept_matches_feature(self) -> DerivedFeatureRef:
        fixed = self.feature.fixed_concept
        if self.feature.needs_concept and self.concept is None:
            raise ValueError(f"{self.feature.value} needs the concept it is computed over")
        if fixed is not None and self.concept not in (None, fixed):
            raise ValueError(
                f"{self.feature.value} is always computed over {fixed.value}, "
                f"not {self.concept.value}"
            )
        return self

    @property
    def effective_concept(self) -> BusinessConcept:
        return self.concept or self.feature.fixed_concept  # type: ignore[return-value]


class VariableRole(StrEnum):
    OUTCOME = "outcome"
    EXPOSURE = "exposure"
    COVARIATE = "covariate"


class Variable(BaseModel):
    """One per-opportunity value. Exactly one source, never an expression."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: VariableId
    role: VariableRole = VariableRole.COVARIATE
    concept: BusinessConcept | None = None
    column: str | None = None
    derived: DerivedFeatureRef | None = None

    @model_validator(mode="after")
    def _exactly_one_source(self) -> Variable:
        sources = [s for s in (self.concept, self.column, self.derived) if s is not None]
        if len(sources) != 1:
            raise ValueError(
                f"variable {self.id!r} must name exactly one of concept, column or "
                f"derived feature, got {len(sources)}"
            )
        return self


class BinningKind(StrEnum):
    NONE = "none"
    EXPLICIT_EDGES = "explicit_edges"


class Binning(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: BinningKind = BinningKind.NONE
    # Structural parameters, not results: where a numeric variable is cut.
    edges: list[float] = Field(default_factory=list)

    @model_validator(mode="after")
    def _edges_are_coherent(self) -> Binning:
        if self.kind is BinningKind.EXPLICIT_EDGES:
            if not 1 <= len(self.edges) <= 10:
                raise ValueError("explicit binning needs between 1 and 10 edges")
            if any(b <= a for a, b in zip(self.edges, self.edges[1:], strict=False)):
                raise ValueError("bin edges must be strictly increasing")
        elif self.edges:
            raise ValueError("edges are only meaningful for explicit binning")
        return self


class Grouping(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    variable: VariableId
    binning: Binning = Binning()


class Population(BaseModel):
    """Who is analysed and which snapshots are observed.

    Membership is frozen at `cohort` (5.4). The window is the span of snapshots
    over which derived features are computed; both ends default to the cohort
    snapshot, which is the prospective case: nothing after the cohort is read.
    Tracing forward is an explicit `window_end`, and under a prospective stance
    the gate refuses one that passes the knowledge horizon.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    period: Period
    cohort: SnapshotSelection = SnapshotSelection(rule=SnapshotRule.PERIOD_OPEN)
    window_start: SnapshotSelection | None = None
    window_end: SnapshotSelection | None = None
    filters: list[Filter] = Field(default_factory=list)
    status_filter: list[OpportunityStatus] = Field(
        default_factory=lambda: [OpportunityStatus.OPEN]
    )
    # Restrict the cohort to opportunities whose close date falls in the period.
    close_date_in_period: bool = False


class StatisticalOperation(StrEnum):
    COUNT = "count"
    SUM = "sum"
    DISTRIBUTION = "distribution"
    CROSSTAB = "crosstab"
    RATE_BY_GROUP = "rate_by_group"
    DIFFERENCE_IN_RATES = "difference_in_rates"
    RANK_CORRELATION = "rank_correlation"
    TREND = "trend"


class Operation(BaseModel):
    """What is computed, over which variables. Names only; no arithmetic."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: StatisticalOperation
    # SUM, RANK_CORRELATION, TREND: the numeric variable.
    measure: VariableId | None = None
    # RANK_CORRELATION: the second numeric variable.
    against: VariableId | None = None
    # RATE_BY_GROUP, DIFFERENCE_IN_RATES: the variable whose truth is counted.
    outcome: VariableId | None = None
    # For a categorical outcome, the value that counts as true ("won").
    outcome_value: str | None = None


class ComparisonKind(StrEnum):
    VS_REFERENCE_GROUP = "vs_reference_group"


class InvestigationComparison(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: ComparisonKind = ComparisonKind.VS_REFERENCE_GROUP
    # The group label every other group's rate is compared with.
    reference: str


class EvidenceRequirements(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    # A group with fewer units is reported with its counts and no rate.
    min_group_support: int = Field(default=10, ge=1)
    required_concepts: list[BusinessConcept] = Field(default_factory=list)
    disclose_association_not_causation: bool = True


class Hypothesis(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    statement: str = Field(min_length=1)
    kind: HypothesisKind
    outcome: VariableId | None = None
    exposures: list[VariableId] = Field(default_factory=list)


# Limits on the shape of an investigation, enforced by the gate so that a
# violation is a structured rejection rather than a schema error.
MAX_VARIABLES = 6
MAX_GROUPINGS = 2


class InvestigationPlan(BaseModel):
    """What the planner emits for a question no registry metric defines."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    plan_id: str = Field(default_factory=new_plan_id)
    question_restatement: str = Field(min_length=1)
    hypotheses: list[Hypothesis] = Field(min_length=1)
    # Required, with no default: the planner must choose (13.11).
    stance: AnalysisStance
    knowledge_cutoff: date | None = None
    population: Population
    unit: AnalysisUnit = AnalysisUnit.OPPORTUNITY
    variables: list[Variable] = Field(min_length=1)
    grouping: list[Grouping] = Field(default_factory=list)
    operation: Operation
    comparison: InvestigationComparison | None = None
    evidence: EvidenceRequirements = EvidenceRequirements()
    limit: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _unique_variable_ids(self) -> InvestigationPlan:
        ids = [v.id for v in self.variables]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"variable ids used more than once: {duplicates}")
        return self

    def variable(self, variable_id: str) -> Variable | None:
        return next((v for v in self.variables if v.id == variable_id), None)

    @property
    def referenced_variables(self) -> list[str]:
        """Every variable id the plan's other parts point at."""
        refs = [g.variable for g in self.grouping]
        op = self.operation
        refs += [r for r in (op.measure, op.against, op.outcome) if r]
        for h in self.hypotheses:
            refs += ([h.outcome] if h.outcome else []) + list(h.exposures)
        return refs

    @property
    def derived_features(self) -> list[DerivedFeature]:
        return [v.derived.feature for v in self.variables if v.derived is not None]
