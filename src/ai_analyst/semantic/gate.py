"""The plan gate (ARCHITECTURE 8.2).

Deterministic validation of a plan before any SQL exists. The gate returns
structured rejections, never prose, so a later layer can branch on a code
rather than parse a sentence (`contracts/rejection.py`).

The gate is the second of five anti-hallucination layers. The first is
structural — the plan schema cannot express arithmetic — and this one checks
everything the schema cannot: that the metrics exist, that their concepts are
bound strongly enough, that every column named is one the stance permits, that
the snapshot rule can actually resolve against this dataset's snapshots, and
that the knowledge cutoff is respected.

**The compiler accepts nothing this gate has not passed.** That is enforced in
`compiler.py` by requiring the validation object, not by convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

from ai_analyst.contracts.binding import ColumnPurpose
from ai_analyst.contracts.columns import Availability
from ai_analyst.contracts.concepts import CONCEPTS, BusinessConcept, SemanticType
from ai_analyst.contracts.plan import (
    AnalysisPattern,
    AnalysisPlan,
    AnalysisSpec,
    AnalysisStance,
    Attribution,
    Comparison,
    ComparisonKind,
    CreationBasis,
    Filter,
    FilterOp,
    Period,
    PeriodKind,
)
from ai_analyst.contracts.rejection import PlanRejection, PlanValidation, RejectionCode
from ai_analyst.contracts.result import (
    AttributionRead,
    ResolvedSnapshot,
    SnapshotRule,
    TemporalRelation,
    TrustTier,
)
from ai_analyst.semantic.calendar import (
    CalendarError,
    FiscalCalendarResolution,
    ResolvedPeriod,
    add_months,
)
from ai_analyst.semantic.metrics import METRIC_NAMES, METRICS, MetricKind, metric_availability
from ai_analyst.semantic.resolver import ConceptResolver
from ai_analyst.semantic.snapshots import (
    SnapshotResolution,
    SnapshotResolutionError,
    SnapshotResolver,
)

# Filter operators whose values are compared against a column's own type. A
# date column filtered with a string that is not a date is a plan error, not a
# runtime surprise.
_VALUE_OPS = frozenset(set(FilterOp) - {FilterOp.IS_NULL, FilterOp.IS_NOT_NULL})


@dataclass
class ValidatedSpec:
    """One spec that passed the gate, with everything it resolved."""

    spec: AnalysisSpec
    resolver: ConceptResolver
    period_start: date
    period_end: date
    period_label: str
    snapshot: SnapshotResolution
    opening_as_of: date
    closing_as_of: date
    assumptions: list[str]
    warnings: list[str]
    # The snapshot every dimension and feature is attributed from (13.11 #4).
    attributions: list[AttributionRead] = field(default_factory=list)
    # The baseline of a comparison, itself fully validated (13.7).
    baseline: ValidatedSpec | None = None
    # The period's opening and closing snapshots, when the analysis reads the
    # whole window rather than one snapshot (bridge, trace, transition).
    window: tuple[SnapshotResolution, ...] = ()

    @property
    def trust_tier(self) -> TrustTier:
        return self.resolver.trust_tier

    def snapshots_read(self) -> list[ResolvedSnapshot]:
        """Every snapshot this analysis reads, each once, in reading order.

        A result reports the snapshots it actually used (CLAUDE.md 3.2), which
        for a multi-snapshot analysis is more than the one its rule names: the
        window it spans, the snapshot each attribution reads, and a comparison's
        baseline.
        """
        found: list[ResolvedSnapshot] = [*self.snapshot.to_contract()]
        for resolution in self.window:
            found += resolution.to_contract()
        for read in self.attributions:
            found.append(
                ResolvedSnapshot(
                    rule=_ATTRIBUTION_RULES[Attribution(read.rule)], resolved_as_of=read.read_as_of
                )
            )
        if self.baseline is not None:
            found += self.baseline.snapshots_read()
        unique: list[ResolvedSnapshot] = []
        for item in found:
            if not any(
                u.resolved_as_of == item.resolved_as_of and u.rule is item.rule for u in unique
            ):
                unique.append(item)
        return unique


@dataclass
class GateOutcome:
    """The gate's full result: the verdict plus what each passing spec resolved."""

    validation: PlanValidation
    specs: dict[str, ValidatedSpec]

    @property
    def ok(self) -> bool:
        return self.validation.plan_ok


def _dimension_column(
    name: str, resolver: ConceptResolver, purpose: ColumnPurpose = ColumnPurpose.DIMENSION
) -> str | None:
    """The physical column a plan reference resolves to, or None."""
    column, _ = resolver.reference(name, purpose=purpose, field_name=purpose.value)
    return column


def _check_filter(
    item: Filter, resolver: ConceptResolver, spec_id: str
) -> PlanRejection | None:
    """Whether one filter names a readable column and a coherent value."""
    _, rejection = resolver.reference(
        item.column, purpose=ColumnPurpose.FILTER, field_name="filters"
    )
    if rejection is not None:
        return rejection
    if item.op in _VALUE_OPS and any(v is None for v in item.values):
        return PlanRejection(
            code=RejectionCode.INVALID_FILTER_VALUE,
            message=(
                f"filter on {item.column!r} uses {item.op.value} with a null value; "
                "use is_null or is_not_null to test for absence"
            ),
            spec_id=spec_id,
            field="filters",
            column=item.column,
        )
    return None


# Patterns whose dimensions are grouped values attributed from one snapshot.
ATTRIBUTED_PATTERNS = frozenset(
    {AnalysisPattern.POINT_IN_TIME, AnalysisPattern.RANKED_LIST, AnalysisPattern.RATE}
)


def prospective_horizon(spec: AnalysisSpec, analysis_as_of: date) -> date:
    """What a prospective analysis may know: its own snapshot, or an earlier cutoff."""
    cutoff = spec.knowledge_cutoff
    return analysis_as_of if cutoff is None else min(analysis_as_of, cutoff)


_ATTRIBUTION_RULES = {
    Attribution.PERIOD_OPEN: SnapshotRule.PERIOD_OPEN,
    Attribution.AT_CLOSE: SnapshotRule.PERIOD_CLOSE,
    Attribution.LATEST: SnapshotRule.LATEST,
}


def reads_window(spec: AnalysisSpec) -> bool:
    """Whether the analysis reads the period's opening and closing snapshots."""
    return (
        spec.pattern
        in (AnalysisPattern.BRIDGE, AnalysisPattern.COHORT_TRACE, AnalysisPattern.TRANSITION)
        or "pipeline_coverage" in spec.metrics
        or any(m in METRICS and METRICS[m].kind is MetricKind.BRIDGE_TERM for m in spec.metrics)
    )


def attribution_snapshot(
    rule: Attribution, *, opening: date, closing: date, latest: date
) -> date:
    """The snapshot an attribution rule reads dimension values from (5.3 ambiguity 5)."""
    return {
        Attribution.PERIOD_OPEN: opening,
        Attribution.AT_CLOSE: closing,
        Attribution.LATEST: latest,
    }[rule]


def _column_availability(resolver: ConceptResolver, column: str) -> Availability:
    registry_column = resolver.registry.get(column)
    if registry_column.classification.classified:
        return registry_column.classification.availability
    grant = resolver.bindings.generic_grant_for(column) or next(
        (g for g in resolver.bindings.grants if g.column == column), None
    )
    return grant.availability if grant else Availability.UNKNOWN


def relation_of(read_as_of: date, reference: date, availability: Availability) -> TemporalRelation:
    """Where a read sits in time relative to what the analysis may know."""
    if availability is Availability.FUTURE_CONTAMINATED:
        return TemporalRelation.RETROSPECTIVE_TERMINAL
    if read_as_of > reference:
        return TemporalRelation.LATER
    if read_as_of == reference:
        return TemporalRelation.CONTEMPORANEOUS
    return TemporalRelation.BACKWARD


def attribution_reads(
    spec: AnalysisSpec,
    resolver: ConceptResolver,
    snapshot: SnapshotResolution,
    *,
    opening: date,
    closing: date,
    latest: date,
) -> list[AttributionRead]:
    """Every dimension and feature, with the snapshot it will be read from.

    Multi-snapshot analysis is not banned: a dimension read at an earlier
    snapshot is history, and one read at the analysis snapshot is the state
    then. What this records, so the gate and the compiler can both refuse it
    under a prospective stance, is a read *after* what the analysis may know.
    """
    if spec.pattern not in ATTRIBUTED_PATTERNS or len(snapshot.resolved) != 1:
        return []
    analysis_as_of = snapshot.as_of
    read_as_of = attribution_snapshot(
        spec.attribution, opening=opening, closing=closing, latest=latest
    )
    reference = (
        prospective_horizon(spec, analysis_as_of)
        if spec.stance is AnalysisStance.PROSPECTIVE
        else analysis_as_of
    )
    reads: list[AttributionRead] = []
    for names, purpose in (
        (spec.dimensions, ColumnPurpose.DIMENSION),
        (spec.features, ColumnPurpose.FEATURE),
    ):
        for name in names:
            column = _dimension_column(name, resolver, purpose)
            if column is None:
                continue
            reads.append(
                AttributionRead(
                    field=name,
                    column=column,
                    rule=spec.attribution.value,
                    read_as_of=read_as_of,
                    horizon=reference if spec.stance is AnalysisStance.PROSPECTIVE else None,
                    relation=relation_of(
                        read_as_of, reference, _column_availability(resolver, column)
                    ),
                )
            )
    return reads


def attribution_rejections(
    spec: AnalysisSpec, reads: list[AttributionRead]
) -> list[PlanRejection]:
    """The prospective later-snapshot attribution guard (ARCHITECTURE 13.11 #4)."""
    if spec.stance is not AnalysisStance.PROSPECTIVE:
        return []
    return [
        PlanRejection(
            code=RejectionCode.STANCE_VIOLATION,
            message=(
                f"{read.field!r} ({read.column}) would be attributed from the snapshot of "
                f"{read.read_as_of.isoformat()} under attribution {read.rule!r}, but a "
                f"prospective analysis may read nothing after "
                f"{read.horizon.isoformat()}"  # type: ignore[union-attr]
                + (
                    "; the value is a retrospective outcome"
                    if read.relation is TemporalRelation.RETROSPECTIVE_TERMINAL
                    else ""
                )
            ),
            spec_id=spec.id,
            field="attribution",
            column=read.column,
            remedy="Attribute as of the period's opening snapshot, or use a retrospective stance.",
        )
        for read in reads
        if not read.relation.safe_for_prospective
    ]


def measure_concept_of(spec: AnalysisSpec) -> BusinessConcept | None:
    """The spec's measure concept as an ontology concept, or None if unknown."""
    if spec.measure_concept is None:
        return None
    try:
        return BusinessConcept(spec.measure_concept)
    except ValueError:
        return None


def _check_measure_concept(spec: AnalysisSpec, resolver: ConceptResolver) -> list[PlanRejection]:
    """A measure concept is a concept, measures money or a quantity, and fits the metrics.

    The name is checked against the ontology, never against the physical
    schema: a raw column name is an unknown concept here even when a column of
    that name exists, so a planner cannot route a physical column past concept
    resolution by calling it a measure.
    """
    name = spec.measure_concept
    concept = measure_concept_of(spec)
    if concept is None:
        return [
            PlanRejection(
                code=RejectionCode.UNKNOWN_MEASURE_CONCEPT,
                message=f"{name!r} is not a concept in the ontology",
                spec_id=spec.id,
                field="measure_concept",
                remedy=(
                    "Name a measure concept such as 'amount'; physical columns are not "
                    "accepted."
                ),
            )
        ]
    definition = CONCEPTS[concept]
    if definition.semantic_type not in (SemanticType.MONEY, SemanticType.QUANTITY):
        return [
            PlanRejection(
                code=RejectionCode.MEASURE_CONCEPT_NOT_APPLICABLE,
                message=(
                    f"{concept.value} is a {definition.semantic_type.value} concept, "
                    "not something that can be summed"
                ),
                spec_id=spec.id,
                field="measure_concept",
                concept=concept,
            )
        ]
    out = [
        PlanRejection(
            code=RejectionCode.MEASURE_CONCEPT_NOT_APPLICABLE,
            message=f"metric {m!r} does not aggregate a measure, so it cannot take one",
            spec_id=spec.id,
            field="measure_concept",
            metric=m,
        )
        for m in spec.metrics
        if m in METRICS and METRICS[m].measure_concept is not BusinessConcept.AMOUNT
        and m != "pipeline_coverage"
    ]
    outcome = resolver.try_resolve(
        concept, load_bearing=True, field_name="measure_concept", purpose=ColumnPurpose.MEASURE
    )
    if isinstance(outcome, PlanRejection):
        out.append(outcome)
    return out


def baseline_period(
    spec: AnalysisSpec,
    period: ResolvedPeriod,
    calendar: FiscalCalendarResolution,
    anchor: date,
) -> ResolvedPeriod:
    """The period a comparison measures against. Raises CalendarError if undefined."""
    kind = spec.comparison.kind
    cal = calendar.calendar
    if kind is ComparisonKind.VS_PERIOD:
        return cal.resolve_period(spec.comparison.baseline, anchor)  # type: ignore[arg-type]
    if period.kind in (PeriodKind.FISCAL_QUARTER, PeriodKind.RELATIVE):
        shift = 1 if kind is ComparisonKind.PERIOD_OVER_PERIOD else 4
        return cal.to_resolved(cal.shift_quarter(cal.quarter_of(period.start), -shift))
    if period.kind is PeriodKind.FISCAL_YEAR:
        year = cal.fiscal_year_of(period.start) - 1
        start, end = cal.fiscal_year_bounds(year)
        return ResolvedPeriod(kind=period.kind, start=start, end=end, label=f"FY{year}")
    if period.kind is PeriodKind.MONTH:
        months = 1 if kind is ComparisonKind.PERIOD_OVER_PERIOD else 12
        start = add_months(period.start, -months)
        end = add_months(start, 1) - timedelta(days=1)
        return ResolvedPeriod(kind=period.kind, start=start, end=end, label=f"{start:%Y-%m}")
    raise CalendarError(
        f"a {kind.value} comparison needs a calendar period; a custom range has no "
        "defined previous period, so name the baseline with vs_period"
    )


def _validate_baseline(
    spec: AnalysisSpec,
    period: ResolvedPeriod,
    current: SnapshotResolution,
    resolver: ConceptResolver,
    snapshots: SnapshotResolver,
    calendar: FiscalCalendarResolution,
) -> tuple[list[PlanRejection], ValidatedSpec | None]:
    """Validate a comparison's baseline as a spec of its own, then check it is knowable."""

    def reject(code: RejectionCode, message: str) -> list[PlanRejection]:
        return [PlanRejection(code=code, message=message, spec_id=spec.id, field="comparison")]

    if spec.pattern is not AnalysisPattern.POINT_IN_TIME or any(
        m in METRICS and METRICS[m].kind is MetricKind.BRIDGE_TERM for m in spec.metrics
    ):
        return reject(
            RejectionCode.INCOMPATIBLE_COMPARISON,
            "a comparison is compiled for point-in-time metrics only; a rate or a "
            "bridge term has no single value per period to compare",
        ), None
    try:
        base = baseline_period(spec, period, calendar, snapshots.latest)
    except CalendarError as exc:
        return reject(RejectionCode.PERIOD_UNRESOLVABLE, str(exc)), None
    baseline_spec = spec.model_copy(
        update={
            "period": Period(
                kind=PeriodKind.CUSTOM, start=base.start, end=base.end, label=base.label
            ),
            "comparison": Comparison(),
        }
    )
    rejections, validated, _, _ = validate_spec(baseline_spec, resolver, snapshots, calendar)
    resolver.spec_id = spec.id
    if rejections:
        return [
            r.model_copy(update={"message": f"baseline {base.label}: {r.message}"})
            for r in rejections
        ], None
    if spec.stance is AnalysisStance.PROSPECTIVE:
        horizon = prospective_horizon(spec, current.as_of)
        if validated.snapshot.as_of > horizon:  # type: ignore[union-attr]
            return reject(
                RejectionCode.STANCE_VIOLATION,
                f"the baseline {base.label} resolves to the snapshot of "
                f"{validated.snapshot.as_of.isoformat()}, after the prospective horizon "
                f"of {horizon.isoformat()}",
            ), None
    return [], validated


def validate_spec(
    spec: AnalysisSpec,
    resolver: ConceptResolver,
    snapshots: SnapshotResolver,
    calendar: FiscalCalendarResolution,
) -> tuple[list[PlanRejection], ValidatedSpec | None, list[str], list[str]]:
    """Validate one analysis spec. Returns rejections, and the resolution if clean."""
    rejections: list[PlanRejection] = []
    assumptions: list[str] = []
    warnings: list[str] = []
    resolver.spec_id = spec.id

    # --- period ---------------------------------------------------------
    try:
        period = calendar.calendar.resolve_period(spec.period, snapshots.latest)
    except CalendarError as exc:
        return (
            [
                PlanRejection(
                    code=RejectionCode.PERIOD_UNRESOLVABLE,
                    message=str(exc),
                    spec_id=spec.id,
                    field="period",
                )
            ],
            None,
            assumptions,
            warnings,
        )
    if not calendar.is_resolved:
        assumptions.append(calendar.assumption)

    # --- metrics --------------------------------------------------------
    for name in spec.metrics:
        if name not in METRICS:
            rejections.append(
                PlanRejection(
                    code=RejectionCode.UNKNOWN_METRIC,
                    message=f"no metric named {name!r}",
                    spec_id=spec.id,
                    field="metrics",
                    metric=name,
                    remedy=f"The registry defines: {', '.join(METRIC_NAMES)}.",
                )
            )
            continue
        definition = METRICS[name]
        if not definition.permits(spec.stance):
            rejections.append(
                PlanRejection(
                    code=RejectionCode.METRIC_STANCE_INCOMPATIBLE,
                    message=(
                        f"metric {name!r} is not permitted under a "
                        f"{spec.stance.value} stance"
                    ),
                    spec_id=spec.id,
                    field="metrics",
                    metric=name,
                )
            )
            continue
        if not definition.supports(spec.pattern):
            rejections.append(
                PlanRejection(
                    code=RejectionCode.METRIC_PATTERN_MISMATCH,
                    message=(
                        f"metric {name!r} cannot be computed as a "
                        f"{spec.pattern.value} analysis"
                    ),
                    spec_id=spec.id,
                    field="metrics",
                    metric=name,
                    remedy=(
                        "Supported patterns: "
                        f"{', '.join(p.value for p in definition.patterns)}."
                    ),
                )
            )
            continue

        availability = metric_availability(definition, resolver, spec.stance)
        if not availability.available:
            for concept in availability.missing_concepts:
                outcome = resolver.try_resolve(concept, load_bearing=True, field_name="metrics")
                rejections.append(
                    outcome
                    if isinstance(outcome, PlanRejection)
                    else PlanRejection(
                        code=RejectionCode.CONCEPT_UNAVAILABLE,
                        message=f"{concept.value} concept unavailable",
                        spec_id=spec.id,
                        field="metrics",
                        concept=concept,
                        metric=name,
                    )
                )
            if not availability.missing_concepts:
                rejections.append(
                    PlanRejection(
                        code=RejectionCode.METRIC_UNAVAILABLE,
                        message=availability.reason,
                        spec_id=spec.id,
                        field="metrics",
                        metric=name,
                    )
                )
            continue

        # Resolve for real, so the compilation record names the columns used.
        for concept in definition.required_concepts:
            resolver.resolve(concept, load_bearing=True, field_name="metrics")
        assumptions.extend(definition.ambiguity_notes)

    # --- dimensions and features ----------------------------------------
    for names, purpose in (
        (spec.dimensions, ColumnPurpose.DIMENSION),
        (spec.features, ColumnPurpose.FEATURE),
    ):
        for name in names:
            _, rejection = resolver.reference(
                name, purpose=purpose, field_name=f"{purpose.value}s"
            )
            if rejection is not None:
                rejections.append(rejection)

    # --- filters ---------------------------------------------------------
    for item in spec.filters:
        rejection = _check_filter(item, resolver, spec.id)
        if rejection is not None:
            rejections.append(rejection)

    # --- creation basis (5.3 ambiguity 1) ---------------------------------
    # 'Created in period' defaults to the created-date concept. Without one,
    # first appearance answers a different question, so it is used only when
    # the plan selects it explicitly, never as a silent fallback.
    touches_bridge = spec.pattern is AnalysisPattern.BRIDGE or any(
        m in METRICS and METRICS[m].kind is MetricKind.BRIDGE_TERM for m in spec.metrics
    )
    if (
        touches_bridge
        and spec.creation_basis is CreationBasis.CREATED_DATE
        and not resolver.has(BusinessConcept.CREATED_DATE, load_bearing=False)
    ):
        rejections.append(
            PlanRejection(
                code=RejectionCode.CONCEPT_UNAVAILABLE,
                message=(
                    "created_date concept unavailable, so 'created in period' cannot use "
                    "a creation date on this dataset"
                ),
                spec_id=spec.id,
                field="creation_basis",
                concept=BusinessConcept.CREATED_DATE,
                remedy=(
                    "Select creation_basis 'first_seen' explicitly, which counts an "
                    "opportunity as created at its first snapshot."
                ),
            )
        )

    # --- measure concept (13.7) --------------------------------------------
    if spec.measure_concept is not None:
        rejections.extend(_check_measure_concept(spec, resolver))

    # --- output shape -----------------------------------------------------
    if spec.pattern is AnalysisPattern.RANKED_LIST and spec.limit is None:
        rejections.append(
            PlanRejection(
                code=RejectionCode.INVALID_OUTPUT_SHAPE,
                message="a ranked list needs a limit, or it is not a ranked list",
                spec_id=spec.id,
                field="limit",
            )
        )
    for order in spec.order_by:
        known = set(spec.metrics) | set(spec.dimensions) | set(spec.features)
        if order.column not in known and not resolver.registry.has(order.column):
            rejections.append(
                PlanRejection(
                    code=RejectionCode.INVALID_OUTPUT_SHAPE,
                    message=(
                        f"order_by names {order.column!r}, which is not among the "
                        "metrics, dimensions, or features this spec produces"
                    ),
                    spec_id=spec.id,
                    field="order_by",
                    column=order.column,
                )
            )

    # --- snapshot coverage -------------------------------------------------
    snapshot: SnapshotResolution | None = None
    opening = closing = None
    window: tuple[SnapshotResolution, ...] = ()
    try:
        snapshot = snapshots.resolve(
            spec.snapshot.rule, period=period, explicit_date=spec.snapshot.explicit_date
        )
        open_resolution = snapshots.resolve(SnapshotRule.PERIOD_OPEN, period=period)
        close_resolution = snapshots.resolve(SnapshotRule.PERIOD_CLOSE, period=period)
        opening, closing = open_resolution.as_of, close_resolution.as_of
        if reads_window(spec):
            window = (open_resolution, close_resolution)
    except SnapshotResolutionError as exc:
        rejections.append(
            PlanRejection(
                code=RejectionCode.SNAPSHOT_UNRESOLVABLE,
                message=str(exc),
                spec_id=spec.id,
                field="snapshot",
            )
        )

    if snapshot is not None:
        warnings.extend(snapshot.warnings)
        for resolved in snapshot.resolved:
            assumptions.append(
                f"{spec.snapshot.rule.value} resolved to the snapshot of "
                f"{resolved.isoformat()}."
            )
        if snapshot.drift_days > spec.snapshot.max_drift_days:
            warnings.append(
                f"snapshot drift of {snapshot.drift_days} day(s) exceeds this plan's "
                f"tolerance of {spec.snapshot.max_drift_days} day(s)"
            )

        # --- knowledge cutoff ---------------------------------------------
        cutoff = spec.knowledge_cutoff
        if cutoff is not None:
            late = [d for d in snapshot.resolved if d > cutoff]
            if late:
                rejections.append(
                    PlanRejection(
                        code=RejectionCode.KNOWLEDGE_CUTOFF_VIOLATION,
                        message=(
                            f"the snapshot rule resolved to "
                            f"{', '.join(d.isoformat() for d in late)}, which is after "
                            f"the knowledge cutoff of {cutoff.isoformat()}"
                        ),
                        spec_id=spec.id,
                        field="knowledge_cutoff",
                    )
                )
            assumptions.append(
                f"No row after {cutoff.isoformat()} was read."
            )
        if (
            spec.stance is AnalysisStance.PROSPECTIVE
            and closing is not None
            and spec.pattern is AnalysisPattern.COHORT_TRACE
        ):
            rejections.append(
                PlanRejection(
                    code=RejectionCode.STANCE_VIOLATION,
                    message=(
                        "a cohort trace follows opportunities into snapshots after "
                        "the cohort was fixed, which is hindsight"
                    ),
                    spec_id=spec.id,
                    field="stance",
                    remedy="Use a retrospective stance.",
                )
            )

    reads: list[AttributionRead] = []
    if snapshot is not None and opening is not None and closing is not None:
        reads = attribution_reads(
            spec, resolver, snapshot, opening=opening, closing=closing, latest=snapshots.latest
        )
        rejections.extend(attribution_rejections(spec, reads))
        for read in reads:
            assumptions.append(
                f"{read.field} is attributed as of the snapshot of {read.read_as_of.isoformat()}."
            )

    baseline: ValidatedSpec | None = None
    if snapshot is not None and spec.comparison.kind is not ComparisonKind.NONE:
        baseline_rejections, baseline = _validate_baseline(
            spec, period, snapshot, resolver, snapshots, calendar
        )
        rejections.extend(baseline_rejections)
        if baseline is not None:
            assumptions.append(
                f"Compared with {baseline.period_label} at the snapshot of "
                f"{baseline.snapshot.as_of.isoformat()}."
            )

    if rejections or snapshot is None or opening is None or closing is None:
        return rejections, None, assumptions, warnings

    return (
        rejections,
        ValidatedSpec(
            spec=spec,
            resolver=resolver,
            period_start=period.start,
            period_end=period.end,
            period_label=period.label,
            snapshot=snapshot,
            opening_as_of=opening,
            closing_as_of=closing,
            assumptions=assumptions,
            warnings=warnings,
            attributions=reads,
            baseline=baseline,
            window=window,
        ),
        assumptions,
        warnings,
    )


def validate_plan(
    plan: AnalysisPlan,
    *,
    dataset_id: str,
    registry,
    bindings,
    snapshots: SnapshotResolver,
    calendar: FiscalCalendarResolution,
) -> GateOutcome:
    """Validate a whole plan. The only thing the compiler will accept."""
    rejections: list[PlanRejection] = []
    warnings: list[str] = []
    assumptions: list[str] = []
    validated: dict[str, ValidatedSpec] = {}

    for spec in plan.specs:
        resolver = ConceptResolver(
            dataset_id=dataset_id,
            registry=registry,
            bindings=bindings,
            stance=spec.stance,
            knowledge_cutoff=spec.knowledge_cutoff,
            spec_id=spec.id,
        )
        spec_rejections, resolved, spec_assumptions, spec_warnings = validate_spec(
            spec, resolver, snapshots, calendar
        )
        rejections.extend(spec_rejections)
        warnings.extend(spec_warnings)
        assumptions.extend(spec_assumptions)
        if resolved is not None:
            validated[spec.id] = resolved
            warnings.extend(resolver.trust_reasons)

    return GateOutcome(
        validation=PlanValidation(
            plan_ok=not rejections,
            rejections=rejections,
            warnings=list(dict.fromkeys(warnings)),
            assumptions=list(dict.fromkeys(assumptions)),
        ),
        specs=validated,
    )


def required_concepts_for(plan: AnalysisPlan) -> set[BusinessConcept]:
    """Every concept the plan's metrics are built on."""
    needed: set[BusinessConcept] = set()
    for spec in plan.specs:
        for name in spec.metrics:
            if name in METRICS:
                needed.update(METRICS[name].required_concepts)
    return needed
