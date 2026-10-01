"""Typed returns of the deterministic tool surface (ARCHITECTURE 13.5).

Every tool returns one of these models, never prose. None of them is wired to a
model yet; they are the data a later planner loop will receive. Three rules are
visible in the shapes themselves:

* **Values are withheld with a reason**, not silently omitted. A column the
  stance forbids appears with its classification and `values_withheld_reason`,
  so the agent knows it exists and why it cannot see inside it.
* **Numbers carry a handle.** A distribution or relationship is registered as a
  result, and its `reference` is what a later answer must cite.
* **Nothing is unbounded.** Listings, value tables and samples are capped, and
  say when they were truncated.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai_analyst.contracts.binding import BindingStatus
from ai_analyst.contracts.concepts import BusinessConcept
from ai_analyst.contracts.plan import AnalysisStance
from ai_analyst.contracts.rejection import PlanValidation, RejectionCode
from ai_analyst.contracts.result import TrustTier
from ai_analyst.contracts.session import PlanEdit


class _View(BaseModel):
    model_config = ConfigDict(frozen=True)


class ToolScope(_View):
    """The stance and horizon every tool answered under."""

    stance: AnalysisStance
    horizon: date | None = None


class ConceptSummary(_View):
    concept: BusinessConcept
    status: BindingStatus
    columns: tuple[str, ...] = ()
    usable: bool
    reason: str = ""


class ColumnListing(_View):
    name: str
    information_class: str
    usable: bool
    reason: str = ""


class DatasetView(_View):
    scope: ToolScope
    dataset_id: str
    row_count: int
    snapshot_count: int
    first_snapshot: date | None = None
    last_snapshot: date | None = None
    fiscal_year_start_month: int
    fiscal_calendar_resolved: bool
    concepts: tuple[ConceptSummary, ...]
    columns: tuple[ColumnListing, ...] = ()
    columns_truncated: bool = False
    quarantined_count: int = 0


class ConceptView(_View):
    scope: ToolScope
    concept: BusinessConcept
    display_name: str
    definition: str
    semantic_type: str
    retrospective: bool
    status: BindingStatus
    columns: tuple[str, ...] = ()
    evidence_kinds: tuple[str, ...] = ()
    caveats: tuple[str, ...] = ()
    alternatives: tuple[str, ...] = ()
    note: str = ""
    usable: bool
    rejection: RejectionCode | None = None
    reconstruction_verdict: str | None = None
    grants: tuple[str, ...] = ()


class ColumnSummary(_View):
    """Structural facts about one column, over rows the scope permits."""

    rows_considered: int
    null_count: int
    distinct_count: int
    # Exact extremes as text. No mean: money and unknowns never get one.
    min: str | None = None
    max: str | None = None


class ColumnView(_View):
    scope: ToolScope
    name: str
    origin: str
    dtype: str
    information_class: str
    availability: str
    disposition: str
    quarantine: str | None = None
    usable: bool
    rejection: RejectionCode | None = None
    grants: tuple[str, ...] = ()
    summary: ColumnSummary | None = None
    values_withheld_reason: str = ""


class ValueCount(_View):
    value: str
    count: int


class ValueDistribution(_View):
    scope: ToolScope
    column: str
    rows_considered: int = 0
    null_count: int = 0
    distinct_count: int = 0
    values: tuple[ValueCount, ...] = ()
    truncated: bool = False
    # The registered result a later answer must cite for any of these numbers.
    reference: str | None = None
    values_withheld_reason: str = ""


class PairCount(_View):
    a: str
    b: str
    count: int


class RelationshipEvidence(_View):
    scope: ToolScope
    a: str
    b: str
    support: int = 0
    pairs: tuple[PairCount, ...] = ()
    truncated: bool = False
    reference: str | None = None
    trust_tier: TrustTier = TrustTier.B


class SampleRowsView(_View):
    scope: ToolScope
    columns: tuple[str, ...]
    rows: tuple[tuple[str | None, ...], ...]
    excluded: dict[str, str] = Field(default_factory=dict)
    limit: int


class MetricEntry(_View):
    name: str
    display_name: str
    definition: str
    required_concepts: tuple[str, ...]
    default_snapshot_rule: str
    patterns: tuple[str, ...]


class UnavailableMetric(_View):
    name: str
    missing_concepts: tuple[str, ...]
    reason: str


class MetricCatalog(_View):
    scope: ToolScope
    available: tuple[MetricEntry, ...]
    unavailable: tuple[UnavailableMetric, ...]


class ResultSummary(_View):
    """What a run tool reports back: a handle, a shape and a tier. No cells."""

    reference: str
    query_id: str
    columns: tuple[str, ...]
    row_count: int
    trust_tier: TrustTier
    trust_reasons: tuple[str, ...] = ()


class RunOutcome(_View):
    validation: PlanValidation
    results: tuple[ResultSummary, ...] = ()

    @property
    def executed(self) -> bool:
        return bool(self.results)


class ClarificationReason(StrEnum):
    MISSING_PERIOD = "missing_period"
    AMBIGUOUS_DEFINITION = "ambiguous_definition"
    CONCEPT_UNAVAILABLE = "concept_unavailable"
    AMBIGUOUS_BINDING = "ambiguous_binding"
    STANCE_UNCLEAR = "stance_unclear"
    COVERAGE_GAP = "coverage_gap"
    AMBIGUOUS_REFERENCE = "ambiguous_reference"


class ClarificationOption(_View):
    label: str = Field(min_length=1)
    # The typed change that choosing this option applies, when it maps to one.
    edit: PlanEdit | None = None


class ClarificationRequest(BaseModel):
    """Ask rather than assume (ARCHITECTURE 13.12)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reason: ClarificationReason
    ambiguity_id: str | None = None
    concept: BusinessConcept | None = None
    question: str = Field(min_length=1)
    options: list[ClarificationOption] = Field(default_factory=list)
    default_option: int | None = None
    # Filled deterministically from the dataset, never by the model.
    available_alternatives: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _coherent(self) -> ClarificationRequest:
        if self.options and not 2 <= len(self.options) <= 4:
            raise ValueError("a clarification offers between 2 and 4 options")
        if self.default_option is not None and not 0 <= self.default_option < len(self.options):
            raise ValueError("default_option must index an option")
        if self.reason is ClarificationReason.CONCEPT_UNAVAILABLE and self.concept is None:
            raise ValueError("an unavailable-concept clarification must name the concept")
        return self
