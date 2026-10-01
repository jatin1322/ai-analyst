"""The deterministic compiler (ARCHITECTURE 2.1, 8.1).

A validated `AnalysisPlan` in, DuckDB SQL out. No LLM call, no randomness, no
arithmetic that the model chose. Compiling the same plan against the same
dataset twice produces byte-identical SQL, which is what makes a result
reproducible and a regression detectable.

Four rules hold everywhere in this module:

* **The compiler rejects an unvalidated plan.** It takes a `GateOutcome`, not a
  plan, and refuses one that did not pass. The check is structural rather than
  a convention someone has to remember.
* **Permitted-column checks happen before SQL generation.** By the time a
  column name reaches a string here, `ConceptResolver` has already established
  that the stance allows reading it. Under a prospective stance a contaminated
  column is unreachable, not merely discouraged.
* **Monetary expressions are explicitly DECIMAL**, produced by
  `ColumnResolution.measure_sql` and visible in the emitted SQL.
* **A prospective analysis cannot read rows beyond its horizon.** Every query
  carries an `as_of` ceiling, so the guarantee is in the SQL rather than in the
  planner's good behaviour.

There is no table or file reachability: the only relation any compiled query
names is the dataset's own scan expression, supplied by the store.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from ai_analyst.contracts.binding import ColumnPurpose
from ai_analyst.contracts.comparison import ComparisonOperator, CompiledComparison, Operand
from ai_analyst.contracts.concepts import CONCEPTS, BusinessConcept
from ai_analyst.contracts.plan import (
    AnalysisPattern,
    AnalysisPlan,
    AnalysisStance,
    CreationBasis,
    SortDirection,
    WinRateBasis,
)
from ai_analyst.contracts.result import CompilationMetadata, SnapshotRule, ValueKind
from ai_analyst.contracts.status import OpportunityStatus
from ai_analyst.semantic import bridge as bridge_mod
from ai_analyst.semantic.bridge import BridgeSpec, bridge_sql
from ai_analyst.semantic.calendar import ResolvedPeriod
from ai_analyst.semantic.cohort import cohort, trace_sql
from ai_analyst.semantic.comparisons import compile_comparison
from ai_analyst.semantic.gate import (
    GateOutcome,
    ValidatedSpec,
    _dimension_column,
    measure_concept_of,
    prospective_horizon,
)
from ai_analyst.semantic.metrics import METRICS, Aggregation, MetricKind
from ai_analyst.semantic.rate import RateSpec, rate_sql
from ai_analyst.semantic.resolver import ConceptResolver
from ai_analyst.semantic.sql import (
    CompilationError,
    build_with,
    exact_divide,
    filters_sql,
    literal,
    quote_ident,
)
from ai_analyst.semantic.transitions import TransitionField, TransitionSpec, transition_sql


class UnvalidatedPlan(CompilationError):
    """The compiler was handed a plan the gate did not pass."""


@dataclass(frozen=True)
class CompiledQuery:
    """One spec compiled to SQL, with the record of how."""

    spec_id: str
    sql: str
    metadata: CompilationMetadata
    assumptions: tuple[str, ...]
    warnings: tuple[str, ...]
    # The rate spec, when this query is a rate, so the checker can verify it.
    rate: RateSpec | None = None
    is_bridge: bool = False
    # How each output column is rendered, decided here from the metric
    # definitions. Money is a semantic fact, never inferred from a DECIMAL.
    column_kinds: dict[str, ValueKind] = field(default_factory=dict)
    # Typed comparisons compiled into the query (13.7).
    comparisons: tuple[CompiledComparison, ...] = ()


def _status_is(
    resolver: ConceptResolver, state: OpportunityStatus, alias: str | None = None
) -> str:
    status = resolver.resolve(BusinessConcept.OPPORTUNITY_STATUS, field_name="status")
    return f"LOWER(CAST({status.sql(alias)} AS VARCHAR)) = {literal(state.value)}"


def _horizon(stance: AnalysisStance, cutoff: date | None, ceiling: date) -> str:
    """The `as_of` ceiling every compiled query carries.

    Under a prospective stance the ceiling is the analysis snapshot itself: the
    query may not read a row recorded after the moment it claims to describe.
    An explicit knowledge cutoff tightens it further, in either stance.
    """
    limits = [ceiling] if stance is AnalysisStance.PROSPECTIVE else []
    if cutoff is not None:
        limits.append(cutoff)
    if not limits:
        return "TRUE"
    return f"as_of <= {literal(min(limits))}"


def _base_rows(
    scan: str, validated: ValidatedSpec, as_of: date, *, in_period: bool
) -> str:
    """Rows at one snapshot, already narrowed by stance, filters, and period."""
    spec, resolver = validated.spec, validated.resolver
    clauses = [f"as_of = {literal(as_of)}", filters_sql(_resolved_filters(validated))]
    horizon = _horizon(spec.stance, spec.knowledge_cutoff, as_of)
    if horizon != "TRUE":
        clauses.append(horizon)
    if in_period:
        close = resolver.resolve(BusinessConcept.EXPECTED_CLOSE_DATE, field_name="period")
        clauses.append(
            f"{close.sql()} BETWEEN {literal(validated.period_start)} "
            f"AND {literal(validated.period_end)}"
        )
    return f"SELECT * FROM {scan}\nWHERE " + "\n  AND ".join(f"({c})" for c in clauses)


def _resolved_filters(validated: ValidatedSpec):
    """Plan filters with concept-named columns rewritten to physical columns."""
    out = []
    for item in validated.spec.filters:
        column = _dimension_column(item.column, validated.resolver, ColumnPurpose.FILTER)
        out.append(item.model_copy(update={"column": column or item.column}))
    return out


def _dimension_columns(validated: ValidatedSpec) -> list[str]:
    return [
        _dimension_column(d, validated.resolver) or d for d in validated.spec.dimensions
    ]


def effective_measure_concept(
    validated: ValidatedSpec, default: BusinessConcept
) -> BusinessConcept:
    """The spec's measure concept when it names one, else the metric's own.

    A name that is not an ontology concept is refused here as well as at the
    gate, never quietly replaced by the metric's default: a physical column
    name is never a measure.
    """
    name = validated.spec.measure_concept
    if name is None:
        return default
    concept = measure_concept_of(validated.spec)
    if concept is None:
        raise CompilationError(
            f"measure concept {name!r} is not a concept in the ontology; a physical "
            "column is never a measure"
        )
    return concept


def _measure(validated: ValidatedSpec, concept: BusinessConcept) -> str:
    """A measure expression, always through concept resolution.

    There is no path from a physical column name to a measure: the concept is
    resolved by the resolver, and a monetary concept crosses the DECIMAL
    boundary in `ColumnResolution.measure_sql`.
    """
    concept = effective_measure_concept(validated, concept)
    return validated.resolver.resolve(concept, field_name="metrics").measure_sql()


def _attributed(
    base: str, scan: str, validated: ValidatedSpec, *, multi_snapshot: bool = False
) -> str:
    """Replace each attributed column with its value at the attribution snapshot.

    When every read is at the rows' own snapshot the base is returned as it is.
    Otherwise the attributed columns are taken, per opportunity, from the
    snapshot the attribution rule names (5.3 ambiguity 5), and an opportunity
    absent there has no value rather than a value from a different snapshot.
    """
    reads = validated.attributions
    if not reads:
        return base
    as_of = validated.snapshot.as_of
    if not multi_snapshot and all(r.read_as_of == as_of for r in reads):
        return base
    columns = list(dict.fromkeys(r.column for r in reads))
    read_as_of = reads[0].read_as_of
    opp = validated.resolver.resolve(
        BusinessConcept.OPPORTUNITY_ID, load_bearing=False, field_name="attribution"
    ).sql()
    excluded = ", ".join(quote_ident(c) for c in columns)
    replaced = ", ".join(f"a.{quote_ident(c)} AS {quote_ident(c)}" for c in columns)
    return (
        f"SELECT b.* EXCLUDE ({excluded}), {replaced}\n"
        f"FROM (\n{_indent(base)}\n) b\n"
        f"LEFT JOIN (\n"
        f"    SELECT {opp} AS attribution_opp, {excluded} FROM {scan}\n"
        f"    WHERE as_of = {literal(read_as_of)}\n"
        f") a ON a.attribution_opp = b.{opp}"
    )


def check_attribution(validated: ValidatedSpec) -> None:
    """The prospective later-snapshot guard, re-asserted at compilation.

    The gate refuses such a plan; this refuses to compile one even if the
    validated object was built or altered some other way. The horizon is
    recomputed from the spec, not read from the recorded reads, so a tampered
    `relation` cannot talk its way past it.
    """
    spec = validated.spec
    if spec.stance is not AnalysisStance.PROSPECTIVE or not validated.attributions:
        return
    horizon = prospective_horizon(spec, validated.snapshot.as_of)
    for read in validated.attributions:
        if read.read_as_of > horizon:
            raise CompilationError(
                f"prospective attribution guard: {read.field!r} would be read from the "
                f"snapshot of {read.read_as_of.isoformat()}, after the horizon of "
                f"{horizon.isoformat()}"
            )


def _compile_point_in_time(scan: str, validated: ValidatedSpec) -> str:
    """One aggregate per metric over one snapshot's rows, grouped by dimensions."""
    spec, resolver = validated.spec, validated.resolver
    as_of = validated.snapshot.as_of
    definitions = [METRICS[m] for m in spec.metrics]
    in_period = any(d.close_date_in_period for d in definitions)
    base = _base_rows(scan, validated, as_of, in_period=in_period)

    open_only = any(d.open_only for d in definitions)
    if open_only:
        base += f"\n  AND ({_status_is(resolver, OpportunityStatus.OPEN)})"
    base = _attributed(base, scan, validated)

    dimensions = _dimension_columns(validated)
    dim_select = "".join(f"    {quote_ident(d)},\n" for d in dimensions)
    group = ", ".join(quote_ident(d) for d in dimensions)

    selects: list[str] = []
    for definition in definitions:
        alias = quote_ident(definition.name)
        match definition.aggregation:
            case Aggregation.COUNT_DISTINCT:
                column = resolver.resolve(
                    definition.measure_concept, field_name="metrics"
                ).sql()
                selects.append(f"    COUNT(DISTINCT {column}) AS {alias}")
            case Aggregation.SUM:
                measure = _measure(validated, definition.measure_concept)
                selects.append(f"    COALESCE(SUM({measure}), 0) AS {alias}")
            case Aggregation.AVERAGE:
                # Never AVG, and never `/`: both return DOUBLE in DuckDB, and a
                # monetary average must not pass through binary floating point.
                measure = _measure(validated, definition.measure_concept)
                average = exact_divide(f"SUM({measure})", f"COUNT({measure})", 2)
                selects.append(f"    {average} AS {alias}")
            case _:  # pragma: no cover - exhaustive over Aggregation
                raise CompilationError(f"metric {definition.name} has no aggregation")

    body_lines = ["SELECT", dim_select + ",\n".join(selects), "FROM point_in_time_rows"]
    if dimensions:
        body_lines.append(f"GROUP BY {group}")
    body_lines.append(_order_by(validated, dimensions))
    if spec.limit is not None:
        body_lines.append(f"LIMIT {spec.limit}")
    return build_with(
        [("point_in_time_rows", base)],
        "\n".join(line for line in body_lines if line),
    )


def _compile_compared(
    scan: str, validated: ValidatedSpec
) -> tuple[str, tuple[CompiledComparison, ...]]:
    """Current and baseline point-in-time results, joined and compared by type.

    Each side is the ordinary point-in-time query over its own period and
    snapshot. The comparison columns are compiled by `compile_comparison`, so
    a metric is only ever compared with itself, in its own unit.
    """
    baseline = validated.baseline
    assert baseline is not None
    current_sql = _compile_point_in_time(scan, validated)
    baseline_sql = _compile_point_in_time(scan, baseline)
    dimensions = _dimension_columns(validated)
    kinds = _column_kinds(validated)
    selects = [f"COALESCE(c.{quote_ident(d)}, b.{quote_ident(d)}) AS {quote_ident(d)}"
               for d in dimensions]
    comparisons: list[CompiledComparison] = []
    for name in validated.spec.metrics:
        definition = METRICS[name]
        additive = definition.aggregation in (Aggregation.SUM, Aggregation.COUNT_DISTINCT)

        def side(alias: str, name: str = name, additive: bool = additive) -> str:
            # A group absent on one side has no pipeline there, which for a
            # sum or a count is zero. An average of nothing stays absent.
            column = f"{alias}.{quote_ident(name)}"
            return f"COALESCE({column}, 0)" if additive else column

        left = Operand(
            label=f"{name} {validated.period_label}",
            sql=side("c"),
            kind=kinds[name],
            period_label=validated.period_label,
            as_of=validated.snapshot.as_of,
        )
        right = Operand(
            label=f"{name} {baseline.period_label}",
            sql=side("b"),
            kind=kinds[name],
            period_label=baseline.period_label,
            as_of=baseline.snapshot.as_of,
        )
        change = compile_comparison(f"{name}_change", left, right, ComparisonOperator.DIFFERENCE)
        relative = compile_comparison(
            f"{name}_pct_change", left, right, ComparisonOperator.RELATIVE_CHANGE
        )
        comparisons += [change, relative]
        selects += [
            f"{left.sql} AS {quote_ident(name)}",
            f"{right.sql} AS {quote_ident(name + '_baseline')}",
            f"{change.expression_sql} AS {quote_ident(change.name)}",
            f"{relative.expression_sql} AS {quote_ident(relative.name)}",
        ]
    join = (
        " AND ".join(
            f"c.{quote_ident(d)} IS NOT DISTINCT FROM b.{quote_ident(d)}" for d in dimensions
        )
        or "TRUE"
    )
    order = (
        f"\nORDER BY {', '.join(quote_ident(d) for d in dimensions)}" if dimensions else ""
    )
    sql = (
        f"WITH current_period AS (\n{_indent(current_sql)}\n),\n"
        f"baseline_period AS (\n{_indent(baseline_sql)}\n)\n"
        f"SELECT " + ",\n       ".join(selects) + "\n"
        f"FROM current_period c\nFULL OUTER JOIN baseline_period b ON {join}{order}"
    )
    return sql, tuple(comparisons)


def _order_by(validated: ValidatedSpec, dimensions: list[str]) -> str:
    spec = validated.spec
    if spec.order_by:
        parts = [
            f"{quote_ident(o.column)} "
            f"{'DESC' if o.direction is SortDirection.DESC else 'ASC'}"
            for o in spec.order_by
        ]
        return f"ORDER BY {', '.join(parts)}"
    if dimensions:
        return f"ORDER BY {', '.join(quote_ident(d) for d in dimensions)}"
    return ""


def _bridge_spec(validated: ValidatedSpec) -> BridgeSpec:
    spec = validated.spec
    return BridgeSpec(
        period=ResolvedPeriod(
            kind=spec.period.kind,
            start=validated.period_start,
            end=validated.period_end,
            label=validated.period_label,
        ),
        opening_as_of=validated.opening_as_of,
        closing_as_of=validated.closing_as_of,
        filters=tuple(_resolved_filters(validated)),
        creation_basis=spec.creation_basis,
        slip_basis=spec.slip_basis,
        measure_concept=effective_measure_concept(validated, BusinessConcept.AMOUNT),
    )


def _compile_bridge(scan: str, validated: ValidatedSpec) -> tuple[str, BridgeSpec]:
    bridge_spec = _bridge_spec(validated)
    return bridge_sql(scan, bridge_spec, validated.resolver), bridge_spec


def _compile_bridge_terms(scan: str, validated: ValidatedSpec) -> tuple[str, BridgeSpec]:
    """A point-in-time query over named bridge terms.

    The whole bridge is computed and the requested terms are read off it, so a
    metric like `slipped_pipeline` is by construction the same number the full
    bridge reports. Computing it separately is how two views of one quantity
    drift apart.
    """
    inner, bridge_spec = _compile_bridge(scan, validated)
    # One column per requested metric, in the order the plan asked for them, so
    # a bridge-term result is addressed the same way as any other metric result
    # and the provenance scanner needs no special case.
    selects = []
    for name in validated.spec.metrics:
        component = METRICS[name].bridge_component
        selects.append(
            f"    COALESCE(MAX(CASE WHEN component = {literal(component)} "
            f"THEN amount END), 0) AS {quote_ident(name)}"
        )
        selects.append(
            f"    COALESCE(MAX(CASE WHEN component = {literal(component)} "
            f"THEN opportunity_count END), 0) AS {quote_ident(name + '_opportunities')}"
        )
    sql = (
        f"WITH full_bridge AS (\n"
        f"{_indent(inner)}\n"
        f")\n"
        f"SELECT\n" + ",\n".join(selects) + "\n"
        "FROM full_bridge"
    )
    return sql, bridge_spec


def _indent(sql: str, spaces: int = 4) -> str:
    pad = " " * spaces
    return "\n".join(pad + line if line.strip() else line for line in sql.splitlines())


def _compile_rate(scan: str, validated: ValidatedSpec) -> tuple[str, RateSpec]:
    spec, resolver = validated.spec, validated.resolver
    name = spec.metrics[0]
    dimensions = _dimension_columns(validated)

    won = _status_is(resolver, OpportunityStatus.WON)
    lost = _status_is(resolver, OpportunityStatus.LOST)

    if name == "win_rate":
        base = _attributed(
            _base_rows(scan, validated, validated.snapshot.as_of, in_period=True),
            scan,
            validated,
        )
        denominator = (
            f"({won}) OR ({lost})"
            if spec.win_rate_basis is WinRateBasis.CLOSED_ONLY
            else "TRUE"
        )
        rate_spec = RateSpec(
            numerator_predicate=won,
            denominator_predicate=denominator,
            measure_sql=None,
            by=tuple(dimensions),
            numerator_is_subset=True,
            numerator_label="won",
            denominator_label=(
                "closed" if spec.win_rate_basis is WinRateBasis.CLOSED_ONLY else "all cohort"
            ),
        )
    elif name == "pipeline_coverage":
        # Opening pipeline over what actually closed won in the period. Both
        # populations come from one row set, keyed by snapshot, so the two
        # halves cannot be drawn from differently filtered populations.
        measure = _measure(validated, BusinessConcept.AMOUNT)
        close = resolver.resolve(BusinessConcept.EXPECTED_CLOSE_DATE, field_name="period")
        open_status = _status_is(resolver, OpportunityStatus.OPEN)
        in_period = (
            f"{close.sql()} BETWEEN {literal(validated.period_start)} "
            f"AND {literal(validated.period_end)}"
        )
        base = (
            f"SELECT * FROM {scan}\n"
            f"WHERE as_of IN ({literal(validated.opening_as_of)}, "
            f"{literal(validated.closing_as_of)})\n"
            f"  AND ({filters_sql(_resolved_filters(validated))})\n"
            f"  AND ({in_period})"
        )
        base = _attributed(base, scan, validated, multi_snapshot=True)
        rate_spec = RateSpec(
            numerator_predicate=(
                f"as_of = {literal(validated.opening_as_of)} AND ({open_status})"
            ),
            denominator_predicate=(
                f"as_of = {literal(validated.closing_as_of)} AND ({won})"
            ),
            measure_sql=measure,
            by=tuple(dimensions),
            # Pipeline is not a subset of bookings; coverage above 1 is the
            # normal and expected case.
            numerator_is_subset=False,
            numerator_label="opening pipeline",
            denominator_label="closed won",
        )
    else:  # pragma: no cover - guarded by the gate's pattern check
        raise CompilationError(f"metric {name!r} is not a rate")

    return rate_sql(base, rate_spec), rate_spec


def _compile_transition(scan: str, validated: ValidatedSpec) -> str:
    spec = validated.spec
    # Which attribute the transition tracks. Defaults to the close date, the
    # movement every slip question is about.
    field = TransitionField.CLOSE_DATE
    for candidate in TransitionField:
        if candidate.value in spec.dimensions or candidate.value in spec.features:
            field = candidate
            break
    transition_spec = TransitionSpec(
        field=field,
        from_as_of=validated.opening_as_of,
        to_as_of=validated.closing_as_of,
        filters=tuple(_resolved_filters(validated)),
        require_before=True,
        changed_only=True,
    )
    return transition_sql(scan, transition_spec, validated.resolver)


def _compile_cohort_trace(scan: str, validated: ValidatedSpec) -> str:
    spec = validated.spec
    spec_cohort = cohort(
        validated.opening_as_of,
        filters=list(_resolved_filters(validated)),
        extra_predicate=_status_is(validated.resolver, OpportunityStatus.OPEN),
    )
    return trace_sql(
        scan,
        spec_cohort,
        validated.closing_as_of,
        validated.resolver,
        dimensions=tuple(spec.dimensions),
    )


def compile_spec(scan: str, validated: ValidatedSpec) -> CompiledQuery:
    """Compile one validated spec to SQL."""
    spec, resolver = validated.spec, validated.resolver
    check_attribution(validated)
    rate_spec: RateSpec | None = None
    is_bridge = False
    comparisons: tuple[CompiledComparison, ...] = ()

    match spec.pattern:
        case AnalysisPattern.BRIDGE:
            sql, _ = _compile_bridge(scan, validated)
            is_bridge = True
        case AnalysisPattern.RATE:
            sql, rate_spec = _compile_rate(scan, validated)
        case AnalysisPattern.TRANSITION:
            sql = _compile_transition(scan, validated)
        case AnalysisPattern.COHORT_TRACE:
            sql = _compile_cohort_trace(scan, validated)
        case AnalysisPattern.POINT_IN_TIME | AnalysisPattern.RANKED_LIST:
            kinds = {METRICS[m].kind for m in spec.metrics}
            if kinds == {MetricKind.BRIDGE_TERM}:
                sql, _ = _compile_bridge_terms(scan, validated)
            elif MetricKind.BRIDGE_TERM in kinds:
                raise CompilationError(
                    "a bridge-term metric cannot be mixed with a point-in-time "
                    "metric in one spec: they are measured at different snapshots"
                )
            elif validated.baseline is not None:
                check_attribution(validated.baseline)
                sql, comparisons = _compile_compared(scan, validated)
            else:
                sql = _compile_point_in_time(scan, validated)
        case _:
            raise CompilationError(
                f"pattern {spec.pattern.value} has no compiler in this milestone"
            )

    metadata = CompilationMetadata(
        pattern=spec.pattern.value,
        stance=spec.stance.value,
        dataset_id=resolver.dataset_id,
        concept_columns=resolver.concept_columns,
        metrics=tuple(spec.metrics),
        permitted_columns=tuple(sorted(resolver.permitted_columns)),
        knowledge_cutoff=spec.knowledge_cutoff,
        monetary_expressions=resolver.monetary_expressions,
        path="semantic",
        usage_grants=tuple(g.disclosure for g in resolver.grants_used),
        attributions=tuple(validated.attributions),
        comparisons=tuple(c.summary for c in comparisons),
    )
    return CompiledQuery(
        spec_id=spec.id,
        sql=sql,
        metadata=metadata,
        assumptions=tuple(dict.fromkeys(validated.assumptions)),
        warnings=tuple(dict.fromkeys(validated.warnings)),
        rate=rate_spec,
        is_bridge=is_bridge,
        column_kinds=_column_kinds(validated),
        comparisons=comparisons,
    )


def _column_kinds(validated: ValidatedSpec) -> dict[str, ValueKind]:
    """How each output column renders, from definitions rather than storage type."""
    spec = validated.spec
    kinds: dict[str, ValueKind] = {
        "amount": ValueKind.MONEY,
        "cohort_amount": ValueKind.MONEY,
        "opportunity_count": ValueKind.COUNT,
        "ratio": ValueKind.RATIO,
        "component": ValueKind.TEXT,
        "terminal_state": ValueKind.TEXT,
        "from_as_of": ValueKind.DATE,
        "to_as_of": ValueKind.DATE,
        "changed": ValueKind.BOOLEAN,
    }
    measure = measure_concept_of(spec)
    measure_is_money = measure is None or CONCEPTS[measure].is_monetary
    for name in spec.metrics:
        definition = METRICS[name]
        if definition.is_monetary:
            kind = ValueKind.MONEY if measure_is_money else ValueKind.QUANTITY
        else:
            kind = ValueKind.COUNT
        kinds[name] = kind
        kinds[f"{name}_opportunities"] = ValueKind.COUNT
        kinds[f"{name}_baseline"] = kind
        kinds[f"{name}_change"] = kind
        kinds[f"{name}_pct_change"] = ValueKind.RATIO
    if spec.pattern is AnalysisPattern.RATE and spec.metrics:
        money_rate = spec.metrics[0] == "pipeline_coverage"
        component = ValueKind.MONEY if money_rate else ValueKind.COUNT
        kinds["numerator"] = kinds["denominator"] = component
    if spec.pattern is AnalysisPattern.TRANSITION:
        field_kind = ValueKind.TEXT
        for candidate in TransitionField:
            if candidate.value in spec.dimensions or candidate.value in spec.features:
                field_kind = {
                    TransitionField.AMOUNT: ValueKind.MONEY,
                    TransitionField.CLOSE_DATE: ValueKind.DATE,
                }.get(candidate, ValueKind.TEXT)
                break
        else:
            field_kind = ValueKind.DATE
        kinds["before"] = kinds["after"] = field_kind
    return kinds


def compile_plan(scan: str, plan: AnalysisPlan, outcome: GateOutcome) -> list[CompiledQuery]:
    """Compile every spec of a validated plan.

    Refuses outright when the gate rejected the plan. There is no flag to
    override this: an unvalidated plan has not had its columns checked against
    the stance, so compiling one would put the leakage guarantee in the hands
    of whoever called the compiler.
    """
    if not outcome.ok:
        codes = ", ".join(sorted({r.code.value for r in outcome.validation.rejections}))
        raise UnvalidatedPlan(
            f"the plan gate rejected this plan ({codes}); the compiler only "
            "accepts a plan that passed validation"
        )
    return [compile_spec(scan, outcome.specs[spec.id]) for spec in plan.specs]


# Re-exported so callers do not reach into `bridge` for the component names.
BRIDGE_COMPONENTS = bridge_mod.BRIDGE_COMPONENTS
DEFAULT_SNAPSHOT_RULES: dict[str, SnapshotRule] = {
    name: definition.default_snapshot_rule for name, definition in METRICS.items()
}
CREATION_BASES = tuple(CreationBasis)
