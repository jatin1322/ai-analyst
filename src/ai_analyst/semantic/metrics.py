"""The metric registry (ARCHITECTURE 5.5, superseded in part by 12.1).

Every metric declares the **business concepts** it needs, never a physical
column. That is what lets one definition serve a tenant whose amount column is
`new_amount` and one whose amount column is `ARR`: the concept is the contract,
and `ConceptResolver` does the translation per dataset.

The registry is schema-aware in the sense of 5.5, but it gates on concepts
rather than column names (12.1). A metric whose required concepts cannot be
**validly bound** for a dataset is not offered at all, so a plan cannot ask for
it and a model cannot hallucinate that it exists. This converts a class of
hallucination into a class of abstention.

"Validly bound" is strict on purpose. A metric is *built on* its required
concepts, so they are load-bearing and an inferred binding is not enough
(12.6): a name match once bound `close_date_qtr`, a quarter label, to the close
date, and a pipeline metric silently computed on that would be wrong in a way
no invariant would catch. Dimensions and filters are not load-bearing and may
run on an inferred binding, which caps the result at trust tier B.

Nothing in this module computes a number. Each metric carries enough
declarative information for `compiler.py` to build its SQL, and the compiler is
the only thing that emits arithmetic.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from ai_analyst.contracts.concepts import BusinessConcept, SemanticType
from ai_analyst.contracts.plan import AnalysisPattern, AnalysisStance
from ai_analyst.contracts.result import SnapshotRule
from ai_analyst.semantic import bridge


class MetricKind(StrEnum):
    """How a metric is computed, which decides which compiler path builds it."""

    # One aggregate over one snapshot's rows.
    POINT_IN_TIME = "point_in_time"
    # One term of the pipeline bridge, computed by the bridge and read off.
    BRIDGE_TERM = "bridge_term"
    # A ratio with both components exposed.
    RATE = "rate"
    # A cohort frozen at one snapshot and classified by its fate at a later one.
    COHORT = "cohort"


class Aggregation(StrEnum):
    SUM = "sum"
    COUNT_DISTINCT = "count_distinct"
    AVERAGE = "average"


class MetricDefinition(BaseModel):
    """One metric, defined in concepts."""

    model_config = ConfigDict(frozen=True)

    name: str
    display_name: str
    definition: str
    kind: MetricKind
    # Concepts the metric is built on. Load-bearing: each needs a confirmed
    # binding before the metric is offered.
    required_concepts: tuple[BusinessConcept, ...]
    expected_semantic_type: SemanticType
    default_snapshot_rule: SnapshotRule
    permitted_stances: tuple[AnalysisStance, ...] = (
        AnalysisStance.PROSPECTIVE,
        AnalysisStance.RETROSPECTIVE,
    )
    # The ambiguities of 5.3 this metric is sensitive to, surfaced in answers.
    ambiguity_notes: tuple[str, ...] = ()
    patterns: tuple[AnalysisPattern, ...] = (AnalysisPattern.POINT_IN_TIME,)

    # POINT_IN_TIME configuration.
    aggregation: Aggregation | None = None
    measure_concept: BusinessConcept | None = None
    open_only: bool = False
    close_date_in_period: bool = False

    # BRIDGE_TERM configuration.
    bridge_component: str | None = None

    # RATE configuration: filled in by the compiler from the plan's bases.
    rate_numerator: str = ""
    rate_denominator: str = ""

    @property
    def is_monetary(self) -> bool:
        return self.expected_semantic_type is SemanticType.MONEY

    def permits(self, stance: AnalysisStance) -> bool:
        return stance in self.permitted_stances

    def supports(self, pattern: AnalysisPattern) -> bool:
        return pattern in self.patterns


_C = BusinessConcept
_GRAIN = (_C.OPPORTUNITY_ID, _C.SNAPSHOT_DATE)

# Ambiguity notes, written once so two metrics cannot disagree about the same
# default (5.3).
_AMBIGUITY_MEASURE = (
    "The amount measure is whichever column the tenant declared for the amount "
    "concept; a dataset-specific measure can be named explicitly in the plan."
)
_AMBIGUITY_ATTRIBUTION = (
    "Dimension values are attributed as of the opening snapshot of the period, "
    "because that fixes the population at the same moment the cohort is fixed."
)
_AMBIGUITY_SNAPSHOT = (
    "Snapshot dates do not land on period boundaries; the snapshot actually "
    "used and its drift from the boundary are reported with the result."
)
_AMBIGUITY_SLIP = (
    "A slip is a close date moving out of the period into a later one. A move "
    "within the period is a push, not a slip."
)
_AMBIGUITY_CREATED = (
    "'Created in period' uses the created-date concept when the tenant has one, "
    "and first appearance in a snapshot otherwise. The two answer different "
    "questions and the result says which was used."
)
_AMBIGUITY_WIN_RATE = (
    "Win rate is won over won-plus-lost by default, keyed by the period the "
    "close date falls in. Counting still-open deals in the denominator is a "
    "different question and is selected explicitly."
)


def _pipeline_metric(
    name: str, display: str, definition: str, rule: SnapshotRule, notes: tuple[str, ...]
) -> MetricDefinition:
    return MetricDefinition(
        name=name,
        display_name=display,
        definition=definition,
        kind=MetricKind.POINT_IN_TIME,
        required_concepts=(
            *_GRAIN,
            _C.AMOUNT,
            _C.EXPECTED_CLOSE_DATE,
            _C.OPPORTUNITY_STATUS,
        ),
        expected_semantic_type=SemanticType.MONEY,
        default_snapshot_rule=rule,
        aggregation=Aggregation.SUM,
        measure_concept=_C.AMOUNT,
        open_only=True,
        close_date_in_period=True,
        ambiguity_notes=notes,
        patterns=(AnalysisPattern.POINT_IN_TIME, AnalysisPattern.RANKED_LIST),
    )


def _bridge_metric(
    name: str,
    display: str,
    definition: str,
    component: str,
    notes: tuple[str, ...],
) -> MetricDefinition:
    return MetricDefinition(
        name=name,
        display_name=display,
        definition=definition,
        kind=MetricKind.BRIDGE_TERM,
        required_concepts=(
            *_GRAIN,
            _C.AMOUNT,
            _C.EXPECTED_CLOSE_DATE,
            _C.OPPORTUNITY_STATUS,
        ),
        expected_semantic_type=SemanticType.MONEY,
        default_snapshot_rule=SnapshotRule.PERIOD_CLOSE,
        bridge_component=component,
        ambiguity_notes=notes,
        patterns=(AnalysisPattern.POINT_IN_TIME, AnalysisPattern.BRIDGE),
    )


METRICS: dict[str, MetricDefinition] = {
    m.name: m
    for m in [
        MetricDefinition(
            name="deal_count",
            display_name="Deal count",
            definition=(
                "Number of distinct opportunities present in the resolved snapshot. "
                "Counts opportunities, not snapshot rows: an opportunity appearing "
                "in several snapshots is still one deal."
            ),
            kind=MetricKind.POINT_IN_TIME,
            required_concepts=_GRAIN,
            expected_semantic_type=SemanticType.QUANTITY,
            default_snapshot_rule=SnapshotRule.LATEST,
            aggregation=Aggregation.COUNT_DISTINCT,
            measure_concept=_C.OPPORTUNITY_ID,
            ambiguity_notes=(_AMBIGUITY_SNAPSHOT,),
            patterns=(AnalysisPattern.POINT_IN_TIME, AnalysisPattern.RANKED_LIST),
        ),
        _pipeline_metric(
            "opening_pipeline",
            "Opening pipeline",
            "Open pipeline at the period's opening snapshot, counting only "
            "opportunities whose close date falls in the period.",
            SnapshotRule.PERIOD_OPEN,
            (_AMBIGUITY_MEASURE, _AMBIGUITY_SNAPSHOT, _AMBIGUITY_ATTRIBUTION),
        ),
        _pipeline_metric(
            "ending_pipeline",
            "Ending pipeline",
            "Open pipeline at the period's closing snapshot, counting only "
            "opportunities whose close date falls in the period.",
            SnapshotRule.PERIOD_CLOSE,
            (_AMBIGUITY_MEASURE, _AMBIGUITY_SNAPSHOT),
        ),
        _bridge_metric(
            "created_pipeline",
            "Created pipeline",
            "Pipeline that entered the period because the opportunity was newly "
            "created, measured at the closing snapshot.",
            bridge.CREATED,
            (_AMBIGUITY_CREATED, _AMBIGUITY_MEASURE),
        ),
        _bridge_metric(
            "slipped_pipeline",
            "Slipped pipeline",
            "Pipeline that left the period because its close date moved out to a "
            "later period, measured at the opening snapshot's amount.",
            bridge.SLIPPED_OUT,
            (_AMBIGUITY_SLIP, _AMBIGUITY_MEASURE),
        ),
        _bridge_metric(
            "pulled_in_pipeline",
            "Pulled-in pipeline",
            "Pipeline that entered the period without being newly created, "
            "dominated by close dates moving in from a later period.",
            bridge.PULLED_IN,
            (
                "The term reports its composition, because entering the period "
                "without being created covers close-date moves and later arrivals "
                "and the two mean different things.",
                _AMBIGUITY_MEASURE,
            ),
        ),
        _bridge_metric(
            "won_pipeline",
            "Won pipeline",
            "Pipeline that left the period by closing won, valued at the amount it "
            "carried at the opening snapshot.",
            bridge.CLOSED_WON,
            (
                "Won is read from the authoritative status concept, never from a "
                "stage label.",
                _AMBIGUITY_MEASURE,
            ),
        ),
        _bridge_metric(
            "lost_pipeline",
            "Lost pipeline",
            "Pipeline that left the period by closing lost, valued at the amount it "
            "carried at the opening snapshot.",
            bridge.CLOSED_LOST,
            (
                "Lost is read from the authoritative status concept. A deleted or "
                "excluded record is neither lost nor open and is reported "
                "separately as other-removed.",
                _AMBIGUITY_MEASURE,
            ),
        ),
        MetricDefinition(
            name="win_rate",
            display_name="Win rate",
            definition=(
                "Won divided by won plus lost, over opportunities whose close date "
                "falls in the period, evaluated at the resolved snapshot. Both "
                "components are reported alongside the ratio."
            ),
            kind=MetricKind.RATE,
            required_concepts=(*_GRAIN, _C.OPPORTUNITY_STATUS, _C.EXPECTED_CLOSE_DATE),
            expected_semantic_type=SemanticType.QUANTITY,
            default_snapshot_rule=SnapshotRule.PERIOD_CLOSE,
            ambiguity_notes=(_AMBIGUITY_WIN_RATE, _AMBIGUITY_SNAPSHOT),
            patterns=(AnalysisPattern.RATE,),
        ),
        MetricDefinition(
            name="average_deal_size",
            display_name="Average deal size",
            definition=(
                "Total open pipeline divided by the number of open opportunities in "
                "the period. Computed as a DECIMAL sum over a count, never as a "
                "float average, so the result is exact."
            ),
            kind=MetricKind.POINT_IN_TIME,
            required_concepts=(
                *_GRAIN,
                _C.AMOUNT,
                _C.EXPECTED_CLOSE_DATE,
                _C.OPPORTUNITY_STATUS,
            ),
            expected_semantic_type=SemanticType.MONEY,
            default_snapshot_rule=SnapshotRule.PERIOD_OPEN,
            aggregation=Aggregation.AVERAGE,
            measure_concept=_C.AMOUNT,
            open_only=True,
            close_date_in_period=True,
            ambiguity_notes=(
                _AMBIGUITY_MEASURE,
                "An average over a period is an average of that period's open "
                "pipeline at one snapshot, not an average across snapshots.",
            ),
            patterns=(AnalysisPattern.POINT_IN_TIME,),
        ),
        MetricDefinition(
            name="pipeline_coverage",
            display_name="Pipeline coverage",
            definition=(
                "Open pipeline at the period's opening snapshot divided by the "
                "amount that actually closed won in the period. How many dollars of "
                "pipeline stood behind each dollar booked."
            ),
            kind=MetricKind.RATE,
            required_concepts=(
                *_GRAIN,
                _C.AMOUNT,
                _C.EXPECTED_CLOSE_DATE,
                _C.OPPORTUNITY_STATUS,
            ),
            expected_semantic_type=SemanticType.QUANTITY,
            default_snapshot_rule=SnapshotRule.PERIOD_OPEN,
            ambiguity_notes=(
                "There is no quota or target concept in the ontology, so coverage "
                "is measured against realized bookings rather than against a "
                "target. A tenant that wants coverage against quota has to supply "
                "a target first.",
                "The numerator is not a subset of the denominator here, so the "
                "ratio is expected to exceed 1 and is not bounded by it.",
                _AMBIGUITY_MEASURE,
            ),
            patterns=(AnalysisPattern.RATE,),
        ),
        MetricDefinition(
            name="cohort_fate",
            display_name="Cohort fate",
            definition=(
                "Opportunities open at the period's opening snapshot, frozen as a "
                "cohort and classified by their fate at the period's closing "
                "snapshot: won, lost, excluded, still open, unknown or vanished. "
                "Each state reports its opportunity count and the cohort's own "
                "amount, read when the cohort was fixed. The states partition the "
                "cohort exactly."
            ),
            kind=MetricKind.COHORT,
            required_concepts=(*_GRAIN, _C.AMOUNT, _C.OPPORTUNITY_STATUS),
            expected_semantic_type=SemanticType.MONEY,
            default_snapshot_rule=SnapshotRule.PERIOD_OPEN,
            # Following a cohort into later snapshots is hindsight by construction.
            permitted_stances=(AnalysisStance.RETROSPECTIVE,),
            ambiguity_notes=(_AMBIGUITY_MEASURE,),
            patterns=(AnalysisPattern.COHORT_TRACE,),
        ),
    ]
}

METRIC_NAMES: tuple[str, ...] = tuple(METRICS)


def metric(name: str) -> MetricDefinition:
    if name not in METRICS:
        raise KeyError(
            f"no metric named {name!r}; the registry defines {', '.join(METRIC_NAMES)}"
        )
    return METRICS[name]


class MetricAvailability(BaseModel):
    """Whether one metric can be computed on one dataset under one stance."""

    model_config = ConfigDict(frozen=True)

    name: str
    available: bool
    missing_concepts: tuple[BusinessConcept, ...] = Field(default_factory=tuple)
    reason: str = ""

    @property
    def reasons(self) -> tuple[str, ...]:
        return tuple(f"{c.value} concept unavailable" for c in self.missing_concepts)


def metric_availability(
    definition: MetricDefinition, resolver, stance: AnalysisStance
) -> MetricAvailability:
    """Whether a metric may be offered, and if not, exactly which concept blocks it.

    Concepts are checked as load-bearing, so an inferred binding reports the
    metric as unavailable rather than quietly computing on a guess.
    """
    if not definition.permits(stance):
        return MetricAvailability(
            name=definition.name,
            available=False,
            reason=(
                f"{definition.name} is not permitted under a {stance.value} stance"
            ),
        )
    missing = tuple(
        concept
        for concept in definition.required_concepts
        if not resolver.has(concept, load_bearing=True)
    )
    if missing:
        return MetricAvailability(
            name=definition.name,
            available=False,
            missing_concepts=missing,
            reason="; ".join(f"{c.value} concept unavailable" for c in missing),
        )
    return MetricAvailability(name=definition.name, available=True)


def available_metrics(resolver, stance: AnalysisStance) -> list[MetricDefinition]:
    """The catalog for one dataset: every metric whose concepts are bound.

    This is the list a later layer shows a model. A metric absent from it cannot
    be planned, which is the abstention mechanism of 5.5.
    """
    return [
        d
        for d in METRICS.values()
        if metric_availability(d, resolver, stance).available
    ]
