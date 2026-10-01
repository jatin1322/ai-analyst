"""The AnalysisPlan (ARCHITECTURE Appendix A).

This is the artifact that makes the system auditable. The planner emits a plan;
a deterministic compiler turns it into SQL. Every field is enumerated or
schema-checked, and none accepts free-form arithmetic, so whole categories of
fabrication are unrepresentable rather than merely discouraged.

`OrderSpec` and `ChartHint` are referenced in ARCHITECTURE Appendix A but left
undefined there. They are defined here.
"""

from __future__ import annotations

import uuid
from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai_analyst.contracts.result import SnapshotRule

# Filter values stay raw JSON scalars. Coercion against the column's declared
# type happens in the plan gate, where the schema is available. Declaring a
# `date` member here would let Pydantic's smart union silently keep an ISO date
# string as a `str`.
type FilterValue = str | int | float | bool | None


class PeriodKind(StrEnum):
    FISCAL_QUARTER = "fiscal_quarter"
    FISCAL_YEAR = "fiscal_year"
    MONTH = "month"
    CUSTOM = "custom"
    RELATIVE = "relative"


class RelativePeriod(StrEnum):
    CURRENT = "current"
    PREVIOUS = "previous"
    LAST_N = "last_n"


class Period(BaseModel):
    """A time period. Resolved to concrete dates by the semantic calendar."""

    model_config = ConfigDict(frozen=True)

    kind: PeriodKind
    label: str | None = None
    start: date | None = None
    end: date | None = None
    relative: RelativePeriod | None = None
    n: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _coherent(self) -> Period:
        if self.kind is PeriodKind.CUSTOM and (self.start is None or self.end is None):
            raise ValueError("custom period requires both start and end")
        if self.kind is PeriodKind.RELATIVE and self.relative is None:
            raise ValueError("relative period requires a relative value")
        if self.relative is RelativePeriod.LAST_N and self.n is None:
            raise ValueError("last_n requires n")
        if self.start and self.end and self.start > self.end:
            raise ValueError("period start is after period end")
        return self


class SnapshotSelection(BaseModel):
    """Which snapshot a metric reads. There is no safe default (§5.1)."""

    model_config = ConfigDict(frozen=True)

    rule: SnapshotRule
    explicit_date: date | None = None
    max_drift_days: int = Field(default=10, ge=0)

    @model_validator(mode="after")
    def _exact_needs_date(self) -> SnapshotSelection:
        if self.rule is SnapshotRule.AS_OF_EXACT and self.explicit_date is None:
            raise ValueError("as_of_exact requires explicit_date")
        return self


class FilterOp(StrEnum):
    EQ = "eq"
    NE = "ne"
    IN = "in"
    NOT_IN = "not_in"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    BETWEEN = "between"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"


_NO_VALUE_OPS = {FilterOp.IS_NULL, FilterOp.IS_NOT_NULL}
_SINGLE_VALUE_OPS = {FilterOp.EQ, FilterOp.NE, FilterOp.GT, FilterOp.GTE, FilterOp.LT, FilterOp.LTE}


class FilterValueSource(StrEnum):
    """Where a filter's literal values came from (ARCHITECTURE 13.7).

    The provenance scanner whitelists a numeral in an answer only when it can
    be traced; a user-supplied threshold is traceable to the question.
    """

    USER = "user"
    PROFILE = "profile"
    SESSION = "session"


class Filter(BaseModel):
    """One predicate. The column is validated against the schema by the plan gate."""

    model_config = ConfigDict(frozen=True)

    column: str
    op: FilterOp
    values: list[FilterValue] = Field(default_factory=list)
    value_source: FilterValueSource = FilterValueSource.USER

    @model_validator(mode="after")
    def _arity(self) -> Filter:
        n = len(self.values)
        if self.op in _NO_VALUE_OPS and n != 0:
            raise ValueError(f"{self.op} takes no values, got {n}")
        if self.op in _SINGLE_VALUE_OPS and n != 1:
            raise ValueError(f"{self.op} takes exactly one value, got {n}")
        if self.op is FilterOp.BETWEEN and n != 2:
            raise ValueError(f"between takes exactly two values, got {n}")
        if self.op in {FilterOp.IN, FilterOp.NOT_IN} and n == 0:
            raise ValueError(f"{self.op} requires at least one value")
        return self


class ComparisonKind(StrEnum):
    PERIOD_OVER_PERIOD = "period_over_period"
    YEAR_OVER_YEAR = "year_over_year"
    VS_PERIOD = "vs_period"
    NONE = "none"


class Comparison(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: ComparisonKind = ComparisonKind.NONE
    baseline: Period | None = None

    @model_validator(mode="after")
    def _baseline_required(self) -> Comparison:
        if self.kind is ComparisonKind.VS_PERIOD and self.baseline is None:
            raise ValueError("vs_period requires a baseline period")
        return self


class SortDirection(StrEnum):
    ASC = "asc"
    DESC = "desc"


class OrderSpec(BaseModel):
    """Referenced by ARCHITECTURE Appendix A but undefined there."""

    model_config = ConfigDict(frozen=True)

    column: str
    direction: SortDirection = SortDirection.DESC


class ChartType(StrEnum):
    NONE = "none"
    LINE = "line"
    BAR = "bar"
    WATERFALL = "waterfall"
    TABLE = "table"


class ChartHint(BaseModel):
    """Referenced by ARCHITECTURE Appendix A but undefined there."""

    model_config = ConfigDict(frozen=True)

    chart_type: ChartType = ChartType.NONE
    x: str | None = None
    y: str | None = None
    series: str | None = None
    title: str | None = None


class AnalysisPattern(StrEnum):
    """The six question patterns plus slip risk (ARCHITECTURE §1.2)."""

    POINT_IN_TIME = "point_in_time"
    BRIDGE = "bridge"
    TRANSITION = "transition"
    COHORT_TRACE = "cohort_trace"
    RATE = "rate"
    RANKED_LIST = "ranked_list"
    SLIP_RISK = "slip_risk"


class CreationBasis(StrEnum):
    CREATED_DATE = "created_date"
    FIRST_SEEN = "first_seen"


class SlipBasis(StrEnum):
    PERIOD_MOVE = "period_move"
    ANY_PUSH = "any_push"


class WinRateBasis(StrEnum):
    CLOSED_ONLY = "closed_only"
    ALL_COHORT = "all_cohort"


class RateKey(StrEnum):
    CLOSE_DATE = "close_date"
    OBSERVED_AT = "observed_at"


class Attribution(StrEnum):
    PERIOD_OPEN = "period_open"
    LATEST = "latest"
    AT_CLOSE = "at_close"


class AnalysisStance(StrEnum):
    """Whether an analysis may use hindsight (ARCHITECTURE 5.8, Appendix C.1).

    Prospective is the default, and deliberately so. A question that meant to
    use hindsight and did not gets a narrower answer; a question that meant to
    be as-of-correct and silently used hindsight gets a wrong answer that looks
    right. Only one of those two failures is recoverable by the reader.
    """

    PROSPECTIVE = "prospective"
    RETROSPECTIVE = "retrospective"

    @property
    def permits_retrospective_information(self) -> bool:
        return self is AnalysisStance.RETROSPECTIVE


class AnalysisSpec(BaseModel):
    """One analysis. A plan may contain several."""

    # Unknown fields are errors: a planner cannot smuggle a raw column or an
    # expression through a key the gate would never look at.
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
    pattern: AnalysisPattern
    metrics: list[str] = Field(default_factory=list)
    dimensions: list[str] = Field(default_factory=list)
    period: Period
    snapshot: SnapshotSelection
    filters: list[Filter] = Field(default_factory=list)
    comparison: Comparison = Comparison()

    # Ambiguity resolutions. Defaults are the ones argued for in §5.3.
    creation_basis: CreationBasis = CreationBasis.CREATED_DATE
    slip_basis: SlipBasis = SlipBasis.PERIOD_MOVE
    win_rate_basis: WinRateBasis = WinRateBasis.CLOSED_ONLY
    rate_key: RateKey = RateKey.CLOSE_DATE
    attribution: Attribution = Attribution.PERIOD_OPEN

    # Appendix C.1. The field §5.8 enforcement hangs on. Without it there is
    # nothing to enforce the availability classes against.
    stance: AnalysisStance = AnalysisStance.PROSPECTIVE
    # Appendix C.2. An explicit ceiling on as_of. A prospective analysis already
    # refuses to read past its own snapshot; this makes the intent auditable and
    # lets a retrospective analysis be deliberately truncated.
    knowledge_cutoff: date | None = None
    # Appendix C.3. A per-opportunity attribute carried into a ranked list, not
    # aggregated and not grouped by. Overloading `dimensions` with one would
    # group by a high-cardinality column and yield a row per opportunity.
    features: list[str] = Field(default_factory=list)
    # ARCHITECTURE 13.7. The business concept a monetary or quantity metric is
    # measured in, by name ("amount", "terminal_amount", or a tenant concept
    # added to the ontology later). Resolved per dataset by the concept
    # resolver, which alone turns it into a physical expression. Deliberately a
    # string checked by the gate, not an enum of currently known measures, and
    # never a physical column: `None` means each metric's own measure concept.
    measure_concept: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_]{0,63}$")

    order_by: list[OrderSpec] = Field(default_factory=list)
    limit: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _features_are_not_dimensions(self) -> AnalysisSpec:
        overlap = sorted(set(self.features) & set(self.dimensions))
        if overlap:
            raise ValueError(
                f"{overlap} appear as both a feature and a dimension; a feature is "
                "carried per opportunity, a dimension is grouped by"
            )
        return self

    @model_validator(mode="after")
    def _metrics_required(self) -> AnalysisSpec:
        if self.pattern is not AnalysisPattern.BRIDGE and not self.metrics:
            raise ValueError(f"pattern {self.pattern} requires at least one metric")
        return self


def new_plan_id() -> str:
    return f"p{uuid.uuid4().hex[:12]}"


class AnalysisPlan(BaseModel):
    """What the planner emits. The only thing the compiler will accept."""

    model_config = ConfigDict(extra="forbid")

    # Identity and lineage (ARCHITECTURE 13.13). A follow-up is an edit of a
    # prior plan, and the new plan names the plan it was derived from.
    plan_id: str = Field(default_factory=new_plan_id)
    parent_plan_id: str | None = None
    question_restatement: str = Field(min_length=1)
    specs: list[AnalysisSpec] = Field(min_length=1)
    chart_hint: ChartHint | None = None
    assumptions: list[str] = Field(default_factory=list)
    unresolved_ambiguities: list[str] = Field(default_factory=list)

    @property
    def needs_clarification(self) -> bool:
        return bool(self.unresolved_ambiguities)

    def spec(self, spec_id: str) -> AnalysisSpec:
        for s in self.specs:
            if s.id == spec_id:
                return s
        raise KeyError(f"no spec with id {spec_id!r}")

    @model_validator(mode="after")
    def _unique_spec_ids(self) -> AnalysisPlan:
        ids = [s.id for s in self.specs]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate spec ids: {ids}")
        return self
