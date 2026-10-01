"""Structured plan rejections (ARCHITECTURE 8.2, Appendix C.7).

The plan gate returns *data*, never prose. A rejection names a code, the spec
it came from, and where in the plan the problem is, so a caller can enumerate
and group rejections instead of parsing sentences. The human-readable message
is carried alongside the code, never instead of it.

This matters more than it looks. "Rejected because the amount concept is
unavailable" is a sentence a later layer would have to regex. `RejectionCode`
is a value it can branch on, and the same rejection can then be rendered as a
clarifying question, an abstention, or a suggestion to confirm a binding,
without anyone re-deriving why the plan failed.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from ai_analyst.contracts.concepts import BusinessConcept


class RejectionCode(StrEnum):
    """Why a plan, or one spec in it, cannot be compiled."""

    # Metric-level
    UNKNOWN_METRIC = "unknown_metric"
    METRIC_UNAVAILABLE = "metric_unavailable"
    METRIC_STANCE_INCOMPATIBLE = "metric_stance_incompatible"
    METRIC_PATTERN_MISMATCH = "metric_pattern_mismatch"

    # Concept-level
    CONCEPT_UNAVAILABLE = "concept_unavailable"
    CONCEPT_NOT_CONFIRMED = "concept_not_confirmed"
    CONCEPT_AMBIGUOUS = "concept_ambiguous"
    CONCEPT_WITHHELD = "concept_withheld"
    # A tenant declaration contradicts a known classification (13.2).
    DECLARATION_CONFLICT = "declaration_conflict"
    # A column readable only through a usage grant, used outside its purposes.
    GRANT_PURPOSE_NOT_PERMITTED = "grant_purpose_not_permitted"

    # Column-level
    UNKNOWN_COLUMN = "unknown_column"
    COLUMN_QUARANTINED = "column_quarantined"
    COLUMN_UNCLASSIFIED = "column_unclassified"
    COLUMN_NOT_KNOWABLE_AT_SNAPSHOT = "column_not_knowable_at_snapshot"

    # Temporal safety
    STANCE_VIOLATION = "stance_violation"
    KNOWLEDGE_CUTOFF_VIOLATION = "knowledge_cutoff_violation"
    RETROSPECTIVE_CONCEPT_IN_PROSPECTIVE = "retrospective_concept_in_prospective"

    # Shape
    SNAPSHOT_UNRESOLVABLE = "snapshot_unresolvable"
    SNAPSHOT_COVERAGE = "snapshot_coverage"
    INVALID_FILTER_VALUE = "invalid_filter_value"
    INVALID_OUTPUT_SHAPE = "invalid_output_shape"
    PERIOD_UNRESOLVABLE = "period_unresolvable"

    # Investigation path (13.8)
    SEMANTIC_PATH_AVAILABLE = "semantic_path_available"
    INEXPRESSIBLE = "inexpressible"
    TOO_MANY_VARIABLES = "too_many_variables"
    TOO_MANY_GROUPINGS = "too_many_groupings"
    UNKNOWN_VARIABLE = "unknown_variable"
    INVALID_OPERATION = "invalid_operation"

    # Measures and comparisons (13.7)
    UNKNOWN_MEASURE_CONCEPT = "unknown_measure_concept"
    MEASURE_CONCEPT_NOT_APPLICABLE = "measure_concept_not_applicable"
    INCOMPATIBLE_COMPARISON = "incompatible_comparison"

    @property
    def is_temporal_safety(self) -> bool:
        """Whether this rejection protects against reading the future."""
        return self in _TEMPORAL_SAFETY


_TEMPORAL_SAFETY: frozenset[RejectionCode] = frozenset(
    {
        RejectionCode.STANCE_VIOLATION,
        RejectionCode.KNOWLEDGE_CUTOFF_VIOLATION,
        RejectionCode.RETROSPECTIVE_CONCEPT_IN_PROSPECTIVE,
        RejectionCode.COLUMN_NOT_KNOWABLE_AT_SNAPSHOT,
    }
)


class PlanRejection(BaseModel):
    """One machine-readable reason a plan cannot be compiled."""

    model_config = ConfigDict(frozen=True)

    code: RejectionCode
    message: str
    spec_id: str | None = None
    # Where in the spec the problem is: "metrics", "dimensions", "filters",
    # "features", "snapshot", "period". Named so a caller can point at it.
    field: str = ""
    concept: BusinessConcept | None = None
    column: str | None = None
    metric: str | None = None
    # What would make this plan compilable, when there is a concrete answer.
    remedy: str = ""

    @property
    def is_temporal_safety(self) -> bool:
        return self.code.is_temporal_safety


class PlanValidation(BaseModel):
    """The plan gate's verdict on one plan.

    A validation with no rejections is the only thing the compiler accepts.
    Warnings do not block: they are disclosures that travel onto the result,
    such as an inferred binding that caps the trust tier at B.
    """

    plan_ok: bool
    rejections: list[PlanRejection] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    # Assumptions the gate resolved on the plan's behalf, such as which
    # snapshot a rule landed on. Carried through onto the result.
    assumptions: list[str] = Field(default_factory=list)

    @property
    def rejected(self) -> bool:
        return not self.plan_ok

    @property
    def codes(self) -> list[RejectionCode]:
        return [r.code for r in self.rejections]

    def has(self, code: RejectionCode) -> bool:
        return any(r.code is code for r in self.rejections)

    def for_spec(self, spec_id: str) -> list[PlanRejection]:
        return [r for r in self.rejections if r.spec_id == spec_id]

    @property
    def temporal_safety_rejections(self) -> list[PlanRejection]:
        return [r for r in self.rejections if r.is_temporal_safety]

    def __bool__(self) -> bool:
        return self.plan_ok
