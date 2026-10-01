"""The investigation path: gate, compiler and runner (ARCHITECTURE 13.8).

Path B answers questions no registry metric defines, without becoming a way
around path A. Four properties hold, and each is enforced here rather than in
a prompt:

* **Same resolver, same allowlist.** Every variable resolves through
  `ConceptResolver`, so a quarantined, unclassified or stance-forbidden column
  is refused exactly as it is on the semantic path, and a usage grant is
  honoured only within its purposes.
* **The semantic path wins.** A plan a registry metric could answer is
  rejected with `SEMANTIC_PATH_AVAILABLE`, whether or not that metric is
  available here. If the metric is unavailable, the semantic gate's refusal is
  the right answer, and the investigation path must not route around it.
* **Recomputed, never read.** Every derived feature is computed from the
  snapshots in the window. No precomputed movement counter is consulted.
* **Deterministic arithmetic.** Counts, sums, exact-division ratios, DECIMAL
  differences and a rank correlation, all in DuckDB. Money goes through the
  DECIMAL boundary; no `AVG`, no `/` on money.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import duckdb

from ai_analyst.config import Settings
from ai_analyst.contracts.binding import ColumnPurpose
from ai_analyst.contracts.concepts import BusinessConcept
from ai_analyst.contracts.investigation import (
    MAX_GROUPINGS,
    MAX_VARIABLES,
    BinningKind,
    DerivedFeature,
    Grouping,
    HypothesisKind,
    InvestigationPlan,
    StatisticalOperation,
    Variable,
)
from ai_analyst.contracts.plan import AnalysisStance
from ai_analyst.contracts.rejection import PlanRejection, PlanValidation, RejectionCode
from ai_analyst.contracts.result import (
    CompilationMetadata,
    ResolvedSnapshot,
    ResultSet,
    SnapshotRule,
    TrustTier,
    ValueKind,
)
from ai_analyst.contracts.schema import DataType
from ai_analyst.contracts.status import OpportunityStatus
from ai_analyst.data.money import monetary_measure_sql
from ai_analyst.semantic.calendar import CalendarError, FiscalCalendarResolution, ResolvedPeriod
from ai_analyst.semantic.cohort import EXCLUDED, LOST, OPEN, UNKNOWN, VANISHED, WON
from ai_analyst.semantic.compiler import CompiledQuery
from ai_analyst.semantic.execute import AbstentionRequired, execute
from ai_analyst.semantic.rate import RateSpec
from ai_analyst.semantic.resolver import ColumnResolution, ConceptResolver
from ai_analyst.semantic.snapshots import (
    SnapshotResolution,
    SnapshotResolutionError,
    SnapshotResolver,
)
from ai_analyst.semantic.sql import build_with, exact_divide, filters_sql, literal, quote_ident
from ai_analyst.semantic.trust import AnalysisPath, assess

_NUMERIC = frozenset({DataType.DECIMAL, DataType.DOUBLE, DataType.BIGINT, DataType.INTEGER})

ASSOCIATION_DISCLOSURE = (
    "This is an association between variables in this dataset, not evidence that "
    "one causes the other."
)


@dataclass
class ResolvedVariable:
    """One plan variable resolved against the dataset."""

    variable: Variable
    column: str
    resolution: ColumnResolution | None = None
    dtype: DataType = DataType.VARCHAR
    # For a physical-column variable: money by classification or tenant grant.
    monetary: bool = False

    @property
    def derived(self) -> DerivedFeature | None:
        return self.variable.derived.feature if self.variable.derived else None

    @property
    def is_monetary(self) -> bool:
        if self.derived in (DerivedFeature.VALUE_AT_START, DerivedFeature.VALUE_AT_END):
            return bool(self.resolution and self.resolution.is_monetary)
        if self.derived is not None:
            return False
        if self.resolution is None:
            return self.monetary
        return self.resolution.is_monetary

    @property
    def is_boolean(self) -> bool:
        if self.derived is not None:
            return self.derived.is_boolean
        return self.dtype is DataType.BOOLEAN

    @property
    def is_numeric(self) -> bool:
        if self.derived is not None:
            if self.derived.is_count:
                return True
            if self.derived in (DerivedFeature.VALUE_AT_START, DerivedFeature.VALUE_AT_END):
                return self.dtype in _NUMERIC
            return False
        return self.dtype in _NUMERIC

    def value_sql(self, alias: str | None = None) -> str:
        """The variable's value in a snapshot row: money through its boundary."""
        if self.resolution is not None:
            return self.resolution.measure_sql(alias)
        if self.monetary:
            # A physical money column crosses the same DECIMAL boundary as a
            # concept-resolved measure: never summed as a DOUBLE.
            return monetary_measure_sql(self.column, self.dtype, alias=alias)
        ident = quote_ident(self.column)
        return f"{quote_ident(alias)}.{ident}" if alias else ident


@dataclass
class ValidatedInvestigation:
    plan: InvestigationPlan
    resolver: ConceptResolver
    period: ResolvedPeriod
    cohort: SnapshotResolution
    cohort_as_of: date
    window_start: date
    window_end: date
    horizon: date | None
    variables: dict[str, ResolvedVariable]
    filters: list = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # The resolutions of the window's two edges, reported with the result.
    window: tuple[SnapshotResolution, ...] = ()

    def snapshots_read(self) -> list[ResolvedSnapshot]:
        """The cohort snapshot and the window's edges, each once."""
        found: list[ResolvedSnapshot] = []
        for resolution in (self.cohort, *self.window):
            for item in resolution.to_contract():
                if not any(
                    f.resolved_as_of == item.resolved_as_of and f.rule is item.rule for f in found
                ):
                    found.append(item)
        return found


@dataclass
class InvestigationOutcome:
    validation: PlanValidation
    validated: ValidatedInvestigation | None

    @property
    def ok(self) -> bool:
        return self.validation.plan_ok


# ---------------------------------------------------------------- semantic path


def semantic_equivalent(plan: InvestigationPlan) -> str | None:
    """The registry metric an investigation plan duplicates, if any.

    Deliberately conservative in one direction: any plan using a derived
    feature is outside the registry, because no registry metric recomputes
    per-opportunity history. Without one, every value is read at a single
    snapshot, and the three shapes below are exactly registry metrics.
    """
    if plan.derived_features:
        return None
    op = plan.operation
    concept_of = {v.id: v.concept for v in plan.variables}
    if op.kind in (StatisticalOperation.COUNT, StatisticalOperation.DISTRIBUTION):
        return "deal_count"
    if (
        op.kind is StatisticalOperation.SUM
        and concept_of.get(op.measure or "") is BusinessConcept.AMOUNT
        and plan.population.status_filter == [OpportunityStatus.OPEN]
    ):
        return (
            "ending_pipeline"
            if plan.population.cohort.rule is SnapshotRule.PERIOD_CLOSE
            else "opening_pipeline"
        )
    if (
        op.kind in (StatisticalOperation.RATE_BY_GROUP, StatisticalOperation.DIFFERENCE_IN_RATES)
        and concept_of.get(op.outcome or "") is BusinessConcept.OPPORTUNITY_STATUS
        and (op.outcome_value or "").lower() == OpportunityStatus.WON.value
    ):
        return "win_rate"
    return None


# ------------------------------------------------------------------------ gate


def _reject(code: RejectionCode, message: str, plan: InvestigationPlan, **kw) -> PlanRejection:
    return PlanRejection(code=code, message=message, spec_id=plan.plan_id, **kw)


def _purposes(plan: InvestigationPlan) -> dict[str, ColumnPurpose]:
    """What each variable is used for, which decides which grants apply."""
    purposes: dict[str, ColumnPurpose] = {}
    for g in plan.grouping:
        purposes[g.variable] = ColumnPurpose.DIMENSION
    for ref in (plan.operation.measure, plan.operation.against):
        if ref:
            purposes.setdefault(ref, ColumnPurpose.MEASURE)
    return purposes


def _resolve_variable(
    variable: Variable,
    purpose: ColumnPurpose | None,
    resolver: ConceptResolver,
    plan: InvestigationPlan,
) -> ResolvedVariable | PlanRejection:
    if variable.column is not None:
        name = variable.column
        if not resolver.registry.has(name):
            return _reject(
                RejectionCode.UNKNOWN_COLUMN,
                f"variable {variable.id!r} names column {name!r}, which does not exist",
                plan,
                field="variables",
                column=name,
            )
        rejection = resolver.inspect_column(
            name, field_name="variables", purpose=purpose or ColumnPurpose.FEATURE
        )
        if rejection is not None:
            return rejection
        column = resolver.registry.get(name)
        return ResolvedVariable(
            variable=variable,
            column=name,
            dtype=column.dtype,
            monetary=resolver.column_is_monetary(name),
        )

    concept = variable.concept or variable.derived.effective_concept  # type: ignore[union-attr]
    outcome = resolver.try_resolve(
        concept,
        load_bearing=False,
        field_name="variables",
        purpose=purpose if variable.derived is None else None,
    )
    if isinstance(outcome, PlanRejection):
        return outcome
    return ResolvedVariable(
        variable=variable, column=outcome.column, resolution=outcome, dtype=outcome.dtype
    )


def _check_operation(
    plan: InvestigationPlan, variables: dict[str, ResolvedVariable]
) -> list[PlanRejection]:
    """Whether the operation's variables have the shape it needs."""
    op, out = plan.operation, []

    def bad(message: str) -> None:
        out.append(_reject(RejectionCode.INVALID_OPERATION, message, plan, field="operation"))

    def numeric(ref: str | None, role: str) -> None:
        if ref is None:
            bad(f"{op.kind.value} needs a {role} variable")
        elif ref in variables and not variables[ref].is_numeric:
            bad(f"{op.kind.value} needs a numeric {role}; {ref!r} is not numeric")

    groups = len(plan.grouping)
    match op.kind:
        case StatisticalOperation.SUM:
            numeric(op.measure, "measure")
        case StatisticalOperation.DISTRIBUTION if groups != 1:
            bad("a distribution needs exactly one grouping: the variable distributed")
        case StatisticalOperation.CROSSTAB if groups != 2:
            bad("a crosstab needs exactly two groupings")
        case StatisticalOperation.RATE_BY_GROUP | StatisticalOperation.DIFFERENCE_IN_RATES:
            if op.outcome is None:
                bad(f"{op.kind.value} needs an outcome variable")
            elif op.outcome in variables:
                resolved = variables[op.outcome]
                if not resolved.is_boolean and op.outcome_value is None:
                    bad(
                        f"outcome {op.outcome!r} is not boolean, so outcome_value must "
                        "say which value counts as true"
                    )
            if op.kind is StatisticalOperation.DIFFERENCE_IN_RATES:
                if groups != 1:
                    bad("a difference in rates needs exactly one grouping")
                if plan.comparison is None:
                    bad("a difference in rates needs a reference group")
        case StatisticalOperation.RANK_CORRELATION:
            numeric(op.measure, "measure")
            numeric(op.against, "second variable")
            if groups:
                bad("a rank correlation is computed over the whole population, ungrouped")
        case StatisticalOperation.TREND:
            if groups > 1:
                bad("a trend takes at most one grouping")
            if op.measure is not None and op.measure in variables:
                trend = variables[op.measure]
                if trend.variable.derived is not None or not trend.is_numeric:
                    bad("a trend measure must be a numeric concept or column read per snapshot")
    for g in plan.grouping:
        if (
            g.binning.kind is BinningKind.EXPLICIT_EDGES
            and g.variable in variables
            and not variables[g.variable].is_numeric
        ):
                bad(f"grouping {g.variable!r} is binned at edges but is not numeric")
    return out


def validate_investigation(
    plan: InvestigationPlan,
    *,
    dataset_id: str,
    registry,
    bindings,
    snapshots: SnapshotResolver,
    calendar: FiscalCalendarResolution,
) -> InvestigationOutcome:
    """Deterministic validation of an investigation plan. The compiler's only input."""
    resolver = ConceptResolver(
        dataset_id=dataset_id,
        registry=registry,
        bindings=bindings,
        stance=plan.stance,
        knowledge_cutoff=plan.knowledge_cutoff,
        spec_id=plan.plan_id,
    )
    rejections: list[PlanRejection] = []
    assumptions: list[str] = []
    warnings: list[str] = []

    # --- the semantic path wins ---------------------------------------------
    metric = semantic_equivalent(plan)
    if metric is not None:
        rejections.append(
            _reject(
                RejectionCode.SEMANTIC_PATH_AVAILABLE,
                f"this investigation computes what the registry metric {metric!r} "
                "defines; it must be asked as an analysis plan",
                plan,
                field="operation",
                metric=metric,
                remedy=f"Submit an AnalysisPlan using {metric!r}.",
            )
        )

    # --- shape ---------------------------------------------------------------
    if len(plan.variables) > MAX_VARIABLES:
        rejections.append(
            _reject(
                RejectionCode.TOO_MANY_VARIABLES,
                f"{len(plan.variables)} variables; an investigation takes at most "
                f"{MAX_VARIABLES}",
                plan,
                field="variables",
            )
        )
    if len(plan.grouping) > MAX_GROUPINGS:
        rejections.append(
            _reject(
                RejectionCode.TOO_MANY_GROUPINGS,
                f"{len(plan.grouping)} groupings; an investigation takes at most "
                f"{MAX_GROUPINGS}",
                plan,
                field="grouping",
            )
        )
    known = {v.id for v in plan.variables}
    for ref in dict.fromkeys(plan.referenced_variables):
        if ref not in known:
            rejections.append(
                _reject(
                    RejectionCode.UNKNOWN_VARIABLE,
                    f"{ref!r} is referenced but not defined as a variable",
                    plan,
                    field="variables",
                )
            )

    # --- period and snapshots --------------------------------------------------
    try:
        period = calendar.calendar.resolve_period(plan.population.period, snapshots.latest)
    except CalendarError as exc:
        rejections.append(
            _reject(RejectionCode.PERIOD_UNRESOLVABLE, str(exc), plan, field="period")
        )
        return InvestigationOutcome(PlanValidation(plan_ok=False, rejections=rejections), None)
    if not calendar.is_resolved:
        assumptions.append(calendar.assumption)

    pop = plan.population
    try:
        cohort = snapshots.resolve(
            pop.cohort.rule, period=period, explicit_date=pop.cohort.explicit_date
        )
        cohort_as_of = cohort.as_of

        def edge(selection) -> SnapshotResolution:
            if selection is None:
                return cohort
            return snapshots.resolve(
                selection.rule, period=period, explicit_date=selection.explicit_date
            )

        window = (edge(pop.window_start), edge(pop.window_end))
        start, end = window[0].as_of, window[1].as_of
    except SnapshotResolutionError as exc:
        rejections.append(
            _reject(RejectionCode.SNAPSHOT_UNRESOLVABLE, str(exc), plan, field="population")
        )
        return InvestigationOutcome(PlanValidation(plan_ok=False, rejections=rejections), None)

    warnings.extend(cohort.warnings)
    assumptions.append(f"The cohort is frozen at the snapshot of {cohort_as_of.isoformat()}.")
    assumptions.append(
        f"Derived features are computed over snapshots from {start.isoformat()} "
        f"to {end.isoformat()}."
    )
    if not start <= cohort_as_of <= end:
        rejections.append(
            _reject(
                RejectionCode.SNAPSHOT_COVERAGE,
                "the observation window must contain the cohort snapshot",
                plan,
                field="population",
            )
        )

    # --- temporal safety -------------------------------------------------------
    horizon: date | None
    if plan.stance is AnalysisStance.PROSPECTIVE:
        horizon = plan.knowledge_cutoff or cohort_as_of
        if end > horizon:
            code = (
                RejectionCode.KNOWLEDGE_CUTOFF_VIOLATION
                if plan.knowledge_cutoff
                else RejectionCode.STANCE_VIOLATION
            )
            rejections.append(
                _reject(
                    code,
                    f"the window ends at {end.isoformat()}, after the knowledge horizon "
                    f"of {horizon.isoformat()}; a prospective investigation cannot read "
                    "later snapshots",
                    plan,
                    field="population",
                    remedy="End the window at the cohort snapshot, or use a retrospective stance.",
                )
            )
        for feature in plan.derived_features:
            if feature.reads_outcome:
                rejections.append(
                    _reject(
                        RejectionCode.STANCE_VIOLATION,
                        f"{feature.value} describes what eventually happened and cannot "
                        "be read under a prospective stance",
                        plan,
                        field="variables",
                    )
                )
    else:
        horizon = plan.knowledge_cutoff
        if horizon is not None and end > horizon:
            rejections.append(
                _reject(
                    RejectionCode.KNOWLEDGE_CUTOFF_VIOLATION,
                    f"the window ends at {end.isoformat()}, after the knowledge cutoff "
                    f"of {horizon.isoformat()}",
                    plan,
                    field="knowledge_cutoff",
                )
            )
    if horizon is not None:
        assumptions.append(f"No row after {horizon.isoformat()} was read.")

    # --- variables, filters, required concepts ------------------------------
    purposes = _purposes(plan)
    variables: dict[str, ResolvedVariable] = {}
    for variable in plan.variables:
        outcome = _resolve_variable(variable, purposes.get(variable.id), resolver, plan)
        if isinstance(outcome, PlanRejection):
            rejections.append(outcome)
        else:
            variables[variable.id] = outcome

    resolved_filters = []
    for item in pop.filters:
        column, rejection = resolver.reference(
            item.column, purpose=ColumnPurpose.FILTER, field_name="filters"
        )
        if rejection is not None:
            rejections.append(rejection)
        else:
            resolved_filters.append(item.model_copy(update={"column": column}))

    needed = list(plan.evidence.required_concepts) + [BusinessConcept.OPPORTUNITY_ID]
    if pop.status_filter:
        needed.append(BusinessConcept.OPPORTUNITY_STATUS)
    if pop.close_date_in_period:
        needed.append(BusinessConcept.EXPECTED_CLOSE_DATE)
    for concept in dict.fromkeys(needed):
        outcome = resolver.try_resolve(
            concept, load_bearing=False, field_name="evidence", purpose=ColumnPurpose.FILTER
        )
        if isinstance(outcome, PlanRejection):
            rejections.append(outcome)

    rejections.extend(_check_operation(plan, variables))

    if any(h.kind is HypothesisKind.ASSOCIATION for h in plan.hypotheses) or (
        plan.evidence.disclose_association_not_causation
    ):
        assumptions.append(ASSOCIATION_DISCLOSURE)
    assumptions.append(
        f"A group with fewer than {plan.evidence.min_group_support} opportunities "
        "is reported with its counts and no rate."
    )

    validation = PlanValidation(
        plan_ok=not rejections,
        rejections=rejections,
        warnings=list(dict.fromkeys(warnings)),
        assumptions=list(dict.fromkeys(assumptions)),
    )
    if rejections:
        return InvestigationOutcome(validation, None)
    return InvestigationOutcome(
        validation,
        ValidatedInvestigation(
            plan=plan,
            resolver=resolver,
            period=period,
            cohort=cohort,
            cohort_as_of=cohort_as_of,
            window_start=start,
            window_end=end,
            window=window,
            horizon=horizon,
            variables=variables,
            filters=resolved_filters,
            assumptions=validation.assumptions,
            warnings=validation.warnings,
        ),
    )


# -------------------------------------------------------------------- compiler


def _status_is(status_sql: str, state: str) -> str:
    return f"LOWER(CAST({status_sql} AS VARCHAR)) = {literal(state)}"


def _derived_sql(rv: ResolvedVariable, v: ValidatedInvestigation, opp: str) -> str:
    """One derived feature, per opportunity, over the window's snapshots."""
    feature = rv.derived
    value = rv.value_sql()
    ws, we = literal(v.window_start), literal(v.window_end)
    ps, pe = literal(v.period.start), literal(v.period.end)

    def at(as_of: str) -> str:
        return f"SELECT {opp} AS opp_id, {value} AS v FROM window_rows WHERE as_of = {as_of}"

    def sequenced(predicate: str, *, flag: bool) -> str:
        aggregate = f"COUNT(*) FILTER (WHERE rn > 1 AND {predicate})"
        return (
            f"SELECT opp_id, {aggregate}{' > 0' if flag else ''} AS v FROM (\n"
            f"    SELECT {opp} AS opp_id, {value} AS v, LAG({value}) OVER w AS prev,\n"
            f"           ROW_NUMBER() OVER w AS rn\n"
            f"    FROM window_rows WINDOW w AS (PARTITION BY {opp} ORDER BY as_of)\n"
            f") GROUP BY opp_id"
        )

    match feature:
        case DerivedFeature.VALUE_AT_START:
            return at(ws)
        case DerivedFeature.VALUE_AT_END:
            return at(we)
        case DerivedFeature.CHANGE_COUNT:
            return sequenced("v IS DISTINCT FROM prev", flag=False)
        case DerivedFeature.CHANGED:
            return sequenced("v IS DISTINCT FROM prev", flag=True)
        case DerivedFeature.PUSH_COUNT:
            return sequenced("v > prev", flag=False)
        case DerivedFeature.SLIPPED | DerivedFeature.PULLED_IN:
            moved = (
                f"s.v BETWEEN {ps} AND {pe} AND e.v > {pe}"
                if feature is DerivedFeature.SLIPPED
                else f"s.v > {pe} AND e.v BETWEEN {ps} AND {pe}"
            )
            return (
                f"SELECT s.opp_id, COALESCE({moved}, FALSE) AS v\n"
                f"FROM ({at(ws)}) s\nLEFT JOIN ({at(we)}) e USING (opp_id)"
            )
        case DerivedFeature.FINAL_STATE:
            s = rv.resolution.sql() if rv.resolution else quote_ident(rv.column)

            def ever(state: str) -> str:
                return f"BOOL_OR({_status_is(s, state)})"

            return (
                f"SELECT {opp} AS opp_id, CASE\n"
                f"    WHEN {ever(WON)} THEN {literal(WON)}\n"
                f"    WHEN {ever(LOST)} THEN {literal(LOST)}\n"
                f"    WHEN {ever(EXCLUDED)} THEN {literal(EXCLUDED)}\n"
                f"    WHEN MAX(as_of) >= {we} AND {ever(OPEN)} THEN {literal(OPEN)}\n"
                f"    WHEN MAX(as_of) >= {we} THEN {literal(UNKNOWN)}\n"
                f"    ELSE {literal(VANISHED)} END AS v\n"
                f"FROM window_rows WHERE as_of >= {literal(v.cohort_as_of)} GROUP BY 1"
            )
    raise AssertionError(f"unhandled derived feature {feature}")  # pragma: no cover


def _number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else repr(float(value))


def _label_sql(g: Grouping, rv: ResolvedVariable) -> str:
    """The group label for one unit: its value, or its bin at the stated edges."""
    column = quote_ident(f"v_{g.variable}")
    if g.binning.kind is BinningKind.NONE:
        return f"COALESCE(CAST({column} AS VARCHAR), '(null)')"
    edges = g.binning.edges
    branches = [f"WHEN {column} IS NULL THEN '(null)'"]
    branches.append(f"WHEN {column} < {edges[0]} THEN '< {_number(edges[0])}'")
    for low, high in zip(edges, edges[1:], strict=False):
        branches.append(
            f"WHEN {column} < {high} THEN '[{_number(low)}, {_number(high)})'"
        )
    return f"CASE {' '.join(branches)} ELSE '>= {_number(edges[-1])}' END"


def compile_investigation(scan: str, validated: ValidatedInvestigation) -> CompiledQuery:
    """Compile a validated investigation into one read-only DuckDB query."""
    plan, resolver = validated.plan, validated.resolver
    op = plan.operation
    opp = resolver.resolve(
        BusinessConcept.OPPORTUNITY_ID, load_bearing=False, field_name="population"
    ).sql()
    pop = plan.population

    clauses = [f"as_of = {literal(validated.cohort_as_of)}", filters_sql(validated.filters)]
    if validated.horizon is not None:
        clauses.append(f"as_of <= {literal(validated.horizon)}")
    if pop.status_filter:
        status = resolver.resolve(
            BusinessConcept.OPPORTUNITY_STATUS, load_bearing=False, field_name="population",
            purpose=ColumnPurpose.FILTER,
        ).sql()
        states = ", ".join(literal(s.value) for s in pop.status_filter)
        clauses.append(f"LOWER(CAST({status} AS VARCHAR)) IN ({states})")
    if pop.close_date_in_period:
        close = resolver.resolve(
            BusinessConcept.EXPECTED_CLOSE_DATE, load_bearing=False, field_name="population",
            purpose=ColumnPurpose.FILTER,
        ).sql()
        clauses.append(
            f"{close} BETWEEN {literal(validated.period.start)} "
            f"AND {literal(validated.period.end)}"
        )
    window = [
        f"as_of BETWEEN {literal(validated.window_start)} AND {literal(validated.window_end)}",
        f"{opp} IN (SELECT {opp} FROM cohort)",
    ]
    if validated.horizon is not None:
        window.append(f"as_of <= {literal(validated.horizon)}")

    ctes: list[tuple[str, str]] = [
        ("cohort", f"SELECT * FROM {scan}\nWHERE " + "\n  AND ".join(f"({c})" for c in clauses)),
        ("window_rows", f"SELECT * FROM {scan}\nWHERE " + "\n  AND ".join(window)),
    ]
    selects = [f"c.{opp} AS opp_id"]
    joins: list[str] = []
    for vid, rv in validated.variables.items():
        column = quote_ident(f"v_{vid}")
        if rv.derived is None:
            selects.append(f"{rv.value_sql('c')} AS {column}")
            continue
        cte = f"f_{vid}"
        ctes.append((cte, _derived_sql(rv, validated, opp)))
        joins.append(f"LEFT JOIN {cte} ON {cte}.opp_id = c.{opp}")
        if rv.derived.is_count:
            selects.append(f"COALESCE({cte}.v, 0) AS {column}")
        elif rv.derived.is_boolean:
            selects.append(f"COALESCE({cte}.v, FALSE) AS {column}")
        else:
            selects.append(f"{cte}.v AS {column}")
    ctes.append(
        (
            "units",
            "SELECT " + ",\n       ".join(selects) + "\nFROM cohort c" + "".join(
                f"\n{j}" for j in joins
            ),
        )
    )

    labels = [
        (g.variable, _label_sql(g, validated.variables[g.variable])) for g in plan.grouping
    ]
    label_select = "".join(f",\n       {sql} AS {quote_ident(name)}" for name, sql in labels)
    ctes.append(("labelled", f"SELECT *{label_select}\nFROM units"))
    group_cols = [quote_ident(name) for name, _ in labels]
    group_list = ", ".join(group_cols)
    head = "".join(f"{c}, " for c in group_cols)
    group_by = f"\nGROUP BY {group_list}" if group_cols else ""
    order_by = f"\nORDER BY {group_list}" if group_cols else ""
    support = plan.evidence.min_group_support

    kinds: dict[str, ValueKind] = {name: ValueKind.TEXT for name, _ in labels}
    kinds.update(units=ValueKind.COUNT)
    rate: RateSpec | None = None

    def measure(ref: str) -> tuple[str, ValueKind]:
        rv = validated.variables[ref]
        kind = ValueKind.MONEY if rv.is_monetary else ValueKind.QUANTITY
        return quote_ident(f"v_{ref}"), kind

    match op.kind:
        case StatisticalOperation.COUNT | StatisticalOperation.CROSSTAB:
            body = f"SELECT {head}COUNT(*) AS units\nFROM labelled{group_by}{order_by}"
        case StatisticalOperation.SUM:
            column, kind = measure(op.measure)  # type: ignore[arg-type]
            kinds["total"] = kind
            body = (
                f"SELECT {head}COALESCE(SUM({column}), 0) AS total, COUNT(*) AS units\n"
                f"FROM labelled{group_by}{order_by}"
            )
        case StatisticalOperation.DISTRIBUTION:
            kinds["share"] = ValueKind.RATIO
            share = exact_divide("COUNT(*)", "SUM(COUNT(*)) OVER ()", 6)
            body = (
                f"SELECT {head}COUNT(*) AS units, {share} AS share\n"
                f"FROM labelled{group_by}{order_by}"
            )
        case StatisticalOperation.RATE_BY_GROUP | StatisticalOperation.DIFFERENCE_IN_RATES:
            rv = validated.variables[op.outcome]  # type: ignore[index]
            outcome_col = quote_ident(f"v_{op.outcome}")
            wanted = (op.outcome_value or "true").lower()
            truth = (
                outcome_col
                if rv.is_boolean and op.outcome_value is None
                else f"LOWER(CAST({outcome_col} AS VARCHAR)) = {literal(wanted)}"
            )
            numerator = f"COUNT(*) FILTER (WHERE {truth})"
            ratio = exact_divide(numerator, "COUNT(*)", 6)
            rates = (
                f"SELECT {head}{numerator} AS numerator, COUNT(*) AS denominator,\n"
                f"       CASE WHEN COUNT(*) >= {support} THEN {ratio} END AS ratio,\n"
                f"       COUNT(*) < {support} AS below_support\n"
                f"FROM labelled{group_by}"
            )
            kinds.update(
                numerator=ValueKind.COUNT,
                denominator=ValueKind.COUNT,
                ratio=ValueKind.RATIO,
                below_support=ValueKind.BOOLEAN,
            )
            rate = RateSpec(numerator_predicate="TRUE", denominator_predicate="TRUE")
            if op.kind is StatisticalOperation.RATE_BY_GROUP:
                body = rates + order_by
            else:
                ctes.append(("rates", rates))
                reference = literal(plan.comparison.reference)  # type: ignore[union-attr]
                kinds["difference_vs_reference"] = ValueKind.RATIO
                body = (
                    f"SELECT r.*, r.ratio - ref.ratio AS difference_vs_reference\n"
                    f"FROM rates r\n"
                    f"LEFT JOIN (SELECT ratio FROM rates WHERE {group_cols[0]} = {reference}) ref "
                    f"ON TRUE{order_by}"
                )
        case StatisticalOperation.RANK_CORRELATION:
            x, _ = measure(op.measure)  # type: ignore[arg-type]
            y, _ = measure(op.against)  # type: ignore[arg-type]

            def rank(col: str) -> str:
                return (
                    f"(RANK() OVER (ORDER BY {col}) + "
                    f"(COUNT(*) OVER (PARTITION BY {col}) - 1) / 2.0)"
                )

            ctes.append(
                (
                    "ranked",
                    f"SELECT {rank(x)} AS rx, {rank(y)} AS ry FROM labelled\n"
                    f"WHERE {x} IS NOT NULL AND {y} IS NOT NULL",
                )
            )
            kinds["rank_correlation"] = ValueKind.QUANTITY
            # Not money, and not arithmetic anyone chose: Spearman's rho as the
            # Pearson correlation of average ranks, rounded to six places so the
            # same data gives the same digits.
            body = (
                f"SELECT COUNT(*) AS units,\n"
                f"       CASE WHEN COUNT(*) >= {support} "
                f"THEN CAST(ROUND(corr(rx, ry), 6) AS DECIMAL(10,6)) END AS rank_correlation\n"
                f"FROM ranked"
            )
        case StatisticalOperation.TREND:
            kinds["as_of"] = ValueKind.DATE
            total = ""
            if op.measure is not None:
                rv = validated.variables[op.measure]
                total = f",\n       COALESCE(SUM({rv.value_sql('w')}), 0) AS total"
                kinds["total"] = ValueKind.MONEY if rv.is_monetary else ValueKind.QUANTITY
            label_cols = "".join(f", l.{c}" for c in group_cols)
            body = (
                f"SELECT w.as_of{label_cols}, COUNT(DISTINCT w.{opp}) AS units{total}\n"
                f"FROM window_rows w\nJOIN labelled l ON l.opp_id = w.{opp}\n"
                f"GROUP BY w.as_of{label_cols}\nORDER BY w.as_of{label_cols}"
            )
        case _:  # pragma: no cover - exhaustive over StatisticalOperation
            raise AssertionError(f"unhandled operation {op.kind}")

    if plan.limit is not None:
        body += f"\nLIMIT {plan.limit}"

    metadata = CompilationMetadata(
        pattern=f"investigation:{op.kind.value}",
        stance=plan.stance.value,
        dataset_id=resolver.dataset_id,
        concept_columns=resolver.concept_columns,
        permitted_columns=tuple(sorted(resolver.permitted_columns)),
        knowledge_cutoff=plan.knowledge_cutoff,
        monetary_expressions=resolver.monetary_expressions,
        path="investigation",
        usage_grants=tuple(g.disclosure for g in resolver.grants_used),
    )
    return CompiledQuery(
        spec_id=plan.plan_id,
        sql=build_with(ctes, body),
        metadata=metadata,
        assumptions=tuple(validated.assumptions),
        warnings=tuple(validated.warnings),
        rate=rate,
        column_kinds=kinds,
    )


def run_investigation(
    conn: duckdb.DuckDBPyConnection,
    scan: str,
    outcome: InvestigationOutcome,
    *,
    dataset_id: str,
    calendar: FiscalCalendarResolution | None = None,
    settings: Settings | None = None,
) -> ResultSet:
    """Compile and execute a validated investigation. At most tier B, always."""
    if not outcome.ok or outcome.validated is None:
        from ai_analyst.semantic.compiler import UnvalidatedPlan

        codes = ", ".join(sorted({r.code.value for r in outcome.validation.rejections}))
        raise UnvalidatedPlan(
            f"the investigation gate rejected this plan ({codes}); only a validated "
            "investigation compiles"
        )
    validated = outcome.validated
    query = compile_investigation(scan, validated)
    trust = assess(
        path=AnalysisPath.INVESTIGATION,
        resolver=validated.resolver,
        snapshot=validated.cohort,
        max_drift_days=validated.plan.population.cohort.max_drift_days,
        calendar=calendar,
        period_kind=validated.plan.population.period.kind,
    )
    if trust.tier is TrustTier.C:
        raise AbstentionRequired(trust)
    return execute(
        conn,
        query,
        dataset_id=dataset_id,
        trust=trust,
        resolved_snapshots=validated.snapshots_read(),
        settings=settings,
    )
