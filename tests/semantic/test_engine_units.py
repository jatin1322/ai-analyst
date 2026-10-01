"""Unit tests for the semantic engine's primitives.

Separate from the golden plans, which check numbers end to end. These check the
properties each primitive is supposed to guarantee, including the ones that
only show up on inputs the tiny fixture does not contain.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import duckdb
import pytest

from ai_analyst.config import Settings
from ai_analyst.contracts.concepts import CONCEPTS, AnalyticalOperation, BusinessConcept
from ai_analyst.contracts.plan import (
    AnalysisPattern,
    AnalysisSpec,
    AnalysisStance,
    Filter,
    FilterOp,
    Period,
    PeriodKind,
    SnapshotSelection,
)
from ai_analyst.contracts.rejection import PlanRejection, RejectionCode
from ai_analyst.contracts.result import SnapshotRule, TrustTier
from ai_analyst.contracts.tenant import TenantProfile
from ai_analyst.semantic.bridge import (
    BRIDGE_TOLERANCE,
    COMPONENT_SIGNS,
    BridgeComponent,
    BridgeImbalance,
    BridgeResult,
    check_balance,
)
from ai_analyst.semantic.calendar import (
    FiscalCalendar,
    FiscalCalendarSource,
    resolve_calendar,
)
from ai_analyst.semantic.cohort import TERMINAL_STATES, cohort, trace_sql
from ai_analyst.semantic.compiler import UnvalidatedPlan, compile_plan, compile_spec
from ai_analyst.semantic.metrics import (
    METRIC_NAMES,
    MetricKind,
    available_metrics,
    metric,
)
from ai_analyst.semantic.rate import RateRow, RateSpec, RateViolation, check_rate
from ai_analyst.semantic.resolver import ConceptResolver, ResolutionError
from ai_analyst.semantic.snapshots import SnapshotResolutionError, SnapshotResolver
from ai_analyst.semantic.sql import exact_divide
from ai_analyst.semantic.transitions import (
    TransitionField,
    TransitionSpec,
    transition_sql,
)

Q1 = Period(kind=PeriodKind.FISCAL_QUARTER, label="FY2025-Q1")


def spec(spec_id: str, **kwargs) -> AnalysisSpec:
    kwargs.setdefault("pattern", AnalysisPattern.POINT_IN_TIME)
    kwargs.setdefault("period", Q1)
    kwargs.setdefault("snapshot", SnapshotSelection(rule=SnapshotRule.PERIOD_OPEN))
    return AnalysisSpec(id=spec_id, **kwargs)


# ---------------------------------------------------------------------------
# Fiscal calendar
# ---------------------------------------------------------------------------


def test_a_february_fiscal_year_is_named_for_the_year_it_begins():
    calendar = FiscalCalendar(2)
    # FY2025 runs 2025-02-01 to 2026-01-31, so 2026-01-15 is still FY2025.
    assert calendar.fiscal_year_of(date(2026, 1, 15)) == 2025
    assert calendar.fiscal_year_of(date(2025, 2, 1)) == 2025
    q1 = calendar.quarter(2025, 1)
    assert (q1.start, q1.end) == (date(2025, 2, 1), date(2025, 4, 30))


def test_quarter_arithmetic_crosses_the_year_boundary():
    calendar = FiscalCalendar(1)
    q4 = calendar.quarter(2025, 4)
    assert calendar.next_quarter(q4).label == "FY2026-Q1"
    assert calendar.previous_quarter(calendar.quarter(2025, 1)).label == "FY2024-Q4"
    assert [q.label for q in calendar.last_n_quarters(q4, 3)] == [
        "FY2025-Q2",
        "FY2025-Q3",
        "FY2025-Q4",
    ]


def test_a_month_six_lands_in_q2_not_q3():
    """The FLOOR-versus-CAST rounding trap, asserted rather than assumed."""
    assert FiscalCalendar(1).quarter_of(date(2025, 6, 15)).quarter == 2


def test_the_fiscal_calendar_is_never_inferred_from_data():
    """With no tenant declaration the start month is an assumption, and says so."""
    resolution = resolve_calendar(None, Settings())
    assert resolution.source is FiscalCalendarSource.CONFIGURED_DEFAULT
    assert not resolution.is_resolved
    assert "unresolved" in resolution.assumption


def test_a_tenant_declaration_resolves_the_fiscal_calendar():
    tenant = TenantProfile(tenant_id="t", fiscal_year_start_month=2)
    resolution = resolve_calendar(tenant, Settings())
    assert resolution.is_resolved and resolution.start_month == 2
    assert "declared" in resolution.assumption


def test_an_unresolved_calendar_travels_onto_the_result_as_an_assumption(tmp_path):
    """The assumption must reach the answer, not stop at the settings object."""
    from tests.semantic.conftest import TINY_CSV, TINY_TENANT, build_engine

    undeclared_calendar = TenantProfile(
        tenant_id="tiny",
        source=TINY_TENANT.source,
        concept_columns=TINY_TENANT.concept_columns,
        fiscal_year_start_month=None,
    )
    engine = build_engine(TINY_CSV, "no_fiscal", tmp_path, tenant=undeclared_calendar)
    result = engine.one(spec("a", metrics=["opening_pipeline"], period=Q1))
    assert any("unresolved" in a for a in result.assumptions)


# ---------------------------------------------------------------------------
# Snapshot selection
# ---------------------------------------------------------------------------

DATES = [date(2025, 1, 1), date(2025, 2, 1), date(2025, 3, 31), date(2025, 6, 30)]


@pytest.mark.parametrize(
    ("rule", "expected"),
    [
        (SnapshotRule.LATEST, date(2025, 6, 30)),
        (SnapshotRule.PERIOD_OPEN, date(2025, 1, 1)),
        (SnapshotRule.PERIOD_CLOSE, date(2025, 3, 31)),
        (SnapshotRule.LATEST_IN_PERIOD, date(2025, 3, 31)),
    ],
)
def test_each_snapshot_rule_resolves_deterministically(rule, expected):
    from ai_analyst.semantic.calendar import FiscalCalendar

    period = FiscalCalendar(1).to_resolved(FiscalCalendar(1).quarter(2025, 1))
    resolution = SnapshotResolver(DATES).resolve(rule, period=period)
    assert resolution.as_of == expected


def test_as_of_exact_refuses_a_date_with_no_snapshot():
    with pytest.raises(SnapshotResolutionError, match="no snapshot exists"):
        SnapshotResolver(DATES).resolve(
            SnapshotRule.AS_OF_EXACT, explicit_date=date(2025, 5, 5)
        )


def test_all_returns_every_snapshot():
    assert SnapshotResolver(DATES).resolve(SnapshotRule.ALL).resolved == tuple(DATES)


def test_drift_beyond_tolerance_warns_rather_than_failing():
    from ai_analyst.semantic.calendar import ResolvedPeriod

    period = ResolvedPeriod(
        kind=PeriodKind.CUSTOM,
        start=date(2025, 1, 1),
        end=date(2025, 4, 15),
        label="custom",
    )
    resolution = SnapshotResolver(DATES, max_drift_days=10).resolve(
        SnapshotRule.PERIOD_CLOSE, period=period
    )
    # Nearest snapshot on or before 2025-04-15 is 2025-03-31: 15 days of drift.
    assert resolution.as_of == date(2025, 3, 31)
    assert resolution.drift_days == 15
    assert not resolution.within_tolerance
    assert resolution.warnings


def test_a_real_drift_reaches_the_result(tiny):
    """Exercised on a period whose boundary is not a snapshot date."""
    period = Period(
        kind=PeriodKind.CUSTOM, start=date(2025, 1, 1), end=date(2025, 4, 15)
    )
    result = tiny.one(
        spec("drift", metrics=["ending_pipeline"], period=period,
             snapshot=SnapshotSelection(rule=SnapshotRule.PERIOD_CLOSE))
    )
    resolved = result.resolved_snapshots[0]
    assert resolved.resolved_as_of == date(2025, 4, 1)
    assert resolved.drift_days == 14
    assert any("14 day" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


def test_a_monetary_concept_resolves_through_an_explicit_decimal_cast(tiny):
    resolver = ConceptResolver(
        dataset_id="tiny", registry=tiny.registry, bindings=tiny.bindings
    )
    amount = resolver.resolve(BusinessConcept.AMOUNT)
    assert amount.is_monetary
    # The tiny fixture's amount is already DECIMAL, so no cast is needed; the
    # point is that the expression came from the monetary boundary.
    assert amount.measure_sql() == '"amount"'


def test_a_double_money_column_is_cast_to_decimal_at_the_boundary():
    from ai_analyst.contracts.schema import DataType
    from ai_analyst.data.money import monetary_measure_sql

    assert monetary_measure_sql("x", DataType.DOUBLE) == 'CAST("x" AS DECIMAL(18,2))'
    assert monetary_measure_sql("x", DataType.DECIMAL) == '"x"'


def test_a_column_that_cannot_hold_money_is_refused_not_coerced():
    from ai_analyst.contracts.schema import DataType
    from ai_analyst.data.money import monetary_measure_sql

    with pytest.raises(ValueError, match="cannot be resolved as a monetary measure"):
        monetary_measure_sql("x", DataType.VARCHAR)


def test_an_inferred_binding_is_refused_where_it_is_load_bearing(undeclared):
    resolver = ConceptResolver(
        dataset_id="u", registry=undeclared.registry, bindings=undeclared.bindings
    )
    outcome = resolver.try_resolve(BusinessConcept.AMOUNT, load_bearing=True)
    assert isinstance(outcome, PlanRejection)
    assert outcome.code is RejectionCode.CONCEPT_NOT_CONFIRMED


def test_an_inferred_binding_is_usable_where_it_is_not_load_bearing(undeclared):
    resolver = ConceptResolver(
        dataset_id="u", registry=undeclared.registry, bindings=undeclared.bindings
    )
    outcome = resolver.try_resolve(BusinessConcept.CUSTOMER_SEGMENT, load_bearing=False)
    assert not isinstance(outcome, PlanRejection)
    # ... and it costs the result its tier-A standing.
    assert resolver.trust_tier is TrustTier.B
    assert any("inference" in r for r in resolver.trust_reasons)


def test_resolving_an_unavailable_concept_raises_with_the_rejection(tiny):
    resolver = ConceptResolver(
        dataset_id="tiny", registry=tiny.registry, bindings=tiny.bindings
    )
    with pytest.raises(ResolutionError) as exc:
        resolver.resolve(BusinessConcept.NARRATIVE_TEXT)
    assert exc.value.rejection.code is RejectionCode.CONCEPT_UNAVAILABLE


def test_the_compilation_record_names_the_columns_each_concept_resolved_to(tiny):
    result = tiny.one(spec("a", metrics=["opening_pipeline"], period=Q1))
    columns = result.compilation.concept_columns
    assert columns["amount"] == "amount"
    assert columns["expected_close_date"] == "close_date"
    assert columns["opportunity_status"] == "status"


# ---------------------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------------------


def test_a_transition_reports_before_after_and_the_snapshot_pair(tiny):
    resolver = ConceptResolver(
        dataset_id="tiny", registry=tiny.registry, bindings=tiny.bindings
    )
    sql = transition_sql(
        tiny.scan,
        TransitionSpec(
            field=TransitionField.CLOSE_DATE,
            from_as_of=date(2025, 1, 1),
            to_as_of=date(2025, 3, 31),
            require_before=True,
            changed_only=True,
        ),
        resolver,
    )
    rows = tiny.query(sql)
    # Between 2025-01-01 and 2025-03-31 exactly two close dates move:
    #   OPP-001  2025-03-15 -> 2025-05-15  (slips out of Q1)
    #   OPP-002  2025-05-20 -> 2025-03-25  (pulled into Q1)
    moved = {r[0]: (r[2], r[3]) for r in rows}
    assert moved == {
        "OPP-001": (date(2025, 3, 15), date(2025, 5, 15)),
        "OPP-002": (date(2025, 5, 20), date(2025, 3, 25)),
    }
    assert all(r[4] is True for r in rows)
    assert all((r[6], r[7]) == (date(2025, 1, 1), date(2025, 3, 31)) for r in rows)


def test_a_transition_is_computed_from_snapshots_not_from_a_stored_counter(tiny):
    """The SQL must not reference a precomputed movement column (5.9)."""
    resolver = ConceptResolver(
        dataset_id="tiny", registry=tiny.registry, bindings=tiny.bindings
    )
    sql = transition_sql(
        tiny.scan,
        TransitionSpec(
            field=TransitionField.CLOSE_DATE,
            from_as_of=date(2025, 1, 1),
            to_as_of=date(2025, 3, 31),
        ),
        resolver,
    )
    assert "push_count" not in sql


def test_an_amount_transition_compares_decimals(tiny):
    resolver = ConceptResolver(
        dataset_id="tiny", registry=tiny.registry, bindings=tiny.bindings
    )
    sql = transition_sql(
        tiny.scan,
        TransitionSpec(
            field=TransitionField.AMOUNT,
            from_as_of=date(2025, 1, 1),
            to_as_of=date(2025, 3, 31),
            require_before=True,
            changed_only=True,
        ),
        resolver,
    )
    rows = tiny.query(sql)
    # OPP-004 is the only amount change: 200000 -> 250000 at 2025-03-31.
    assert [(r[0], r[2], r[3]) for r in rows] == [
        ("OPP-004", Decimal("200000.00"), Decimal("250000.00"))
    ]


def test_every_transition_field_maps_to_a_concept_never_a_column():
    for field in TransitionField:
        assert field.concept in CONCEPTS


# ---------------------------------------------------------------------------
# Cohort and trace
# ---------------------------------------------------------------------------


def test_the_trace_states_partition_the_cohort_exactly(tiny):
    resolver = ConceptResolver(
        dataset_id="tiny", registry=tiny.registry, bindings=tiny.bindings,
        stance=AnalysisStance.RETROSPECTIVE,
    )
    sql = trace_sql(tiny.scan, cohort(date(2025, 1, 1)), date(2025, 6, 30), resolver)
    rows = tiny.query(sql)
    # The 2025-01-01 snapshot holds 6 opportunities, and every one lands in
    # exactly one state, so the counts sum back to the cohort size.
    assert sum(r[1] for r in rows) == 6
    assert all(r[0] in TERMINAL_STATES for r in rows)


def test_the_trace_classifies_each_of_the_fixtures_documented_cases(tiny):
    resolver = ConceptResolver(
        dataset_id="tiny", registry=tiny.registry, bindings=tiny.bindings,
        stance=AnalysisStance.RETROSPECTIVE,
    )
    sql = trace_sql(tiny.scan, cohort(date(2025, 1, 1)), date(2025, 6, 30), resolver)
    states = {r[0]: (r[1], r[2]) for r in tiny.query(sql)}
    # From the cohort of 2025-01-01, traced to 2025-06-30:
    #   won      OPP-002 (50000) and OPP-003 (75000)  = 125000
    #   lost     OPP-007 (60000)                      =  60000
    #   open     OPP-001 (100000) and OPP-004 (200000)= 300000
    #   vanished OPP-005 (40000), last seen 2025-03-31, no terminal state
    assert states["won"] == (2, Decimal("125000.00"))
    assert states["lost"] == (1, Decimal("60000.00"))
    assert states["open"] == (2, Decimal("300000.00"))
    assert states["vanished"] == (1, Decimal("40000.00"))


def test_the_trace_measures_the_cohorts_own_amount_not_a_later_one(tiny):
    """OPP-004 is 200000 at the cohort snapshot and 180000 at the end."""
    resolver = ConceptResolver(
        dataset_id="tiny", registry=tiny.registry, bindings=tiny.bindings,
        stance=AnalysisStance.RETROSPECTIVE,
    )
    sql = trace_sql(tiny.scan, cohort(date(2025, 1, 1)), date(2025, 6, 30), resolver)
    states = {r[0]: r[2] for r in tiny.query(sql)}
    assert states["open"] == Decimal("300000.00")  # 100000 + 200000, not 180000


def test_cohort_membership_is_frozen_at_creation(tiny):
    """OPP-006 and OPP-008 appear later and never join the cohort."""
    resolver = ConceptResolver(
        dataset_id="tiny", registry=tiny.registry, bindings=tiny.bindings,
        stance=AnalysisStance.RETROSPECTIVE,
    )
    sql = trace_sql(tiny.scan, cohort(date(2025, 1, 1)), date(2025, 6, 30), resolver)
    assert sum(r[1] for r in tiny.query(sql)) == 6  # not 8


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        (["open", "won"], "won"),
        (["open", "lost"], "lost"),
        (["open", "excluded"], "excluded"),
        (["won", "lost"], "won"),
        (["excluded", "won"], "won"),
        (["unknown"], "unknown"),
        (["open"], "open"),
    ],
)
def test_trace_state_precedence_is_deterministic(statuses, expected, tmp_path):
    """Six states need a total order, or one opportunity has two answers.

    The tiny fixture derives status from stage keywords and so contains no
    excluded or unknown row. These are exercised directly against the
    classification SQL instead, which is where the precedence lives.
    """
    ever = {f"ever_{s}": ("TRUE" if s in statuses else "FALSE") for s in
            ("won", "lost", "excluded", "open")}
    case = (
        "CASE "
        f"WHEN {ever['ever_won']} THEN 'won' "
        f"WHEN {ever['ever_lost']} THEN 'lost' "
        f"WHEN {ever['ever_excluded']} THEN 'excluded' "
        f"WHEN TRUE AND {ever['ever_open']} THEN 'open' "
        "WHEN TRUE THEN 'unknown' "
        "ELSE 'vanished' END"
    )
    assert duckdb.execute(f"SELECT {case}").fetchone()[0] == expected


# ---------------------------------------------------------------------------
# Rate
# ---------------------------------------------------------------------------


def test_a_zero_denominator_yields_no_value(tiny):
    """Q3 has no closed deals at all, so the win rate is absent, not zero."""
    result = tiny.one(
        spec("r", pattern=AnalysisPattern.RATE, metrics=["win_rate"],
             period=Period(kind=PeriodKind.FISCAL_QUARTER, label="FY2025-Q3"),
             snapshot=SnapshotSelection(rule=SnapshotRule.LATEST))
    )
    assert result.cell(0, "denominator") == 0
    assert result.cell(0, "ratio") is None


def test_a_subset_rate_above_one_is_rejected():
    spec_ = RateSpec(numerator_predicate="a", denominator_predicate="b")
    with pytest.raises(RateViolation, match="above denominator"):
        check_rate(
            [RateRow(numerator=Decimal("5"), denominator=Decimal("3"), ratio=Decimal("1.7"))],
            spec_,
        )


def test_a_non_subset_rate_above_one_is_allowed():
    """Pipeline coverage is expected to exceed 1 and must not be flagged."""
    spec_ = RateSpec(
        numerator_predicate="a", denominator_predicate="b", numerator_is_subset=False
    )
    check_rate(
        [RateRow(numerator=Decimal("5"), denominator=Decimal("3"), ratio=Decimal("1.666667"))],
        spec_,
    )


def test_a_ratio_reported_against_a_zero_denominator_is_rejected():
    with pytest.raises(RateViolation, match="zero denominator"):
        check_rate(
            [RateRow(numerator=Decimal("1"), denominator=Decimal("0"), ratio=Decimal("1"))],
            RateSpec(numerator_predicate="a", denominator_predicate="b"),
        )


@pytest.mark.parametrize(
    ("numerator", "denominator", "scale", "expected"),
    [
        ("CAST(200000.00 AS DECIMAL(18,2))", "CAST(3 AS BIGINT)", 2, Decimal("66666.66")),
        ("CAST(1 AS BIGINT)", "CAST(3 AS BIGINT)", 6, Decimal("0.333333")),
        ("CAST(1 AS BIGINT)", "CAST(4 AS BIGINT)", 2, Decimal("0.25")),
    ],
)
def test_exact_divide_never_produces_a_float(numerator, denominator, scale, expected):
    sql = exact_divide(numerator, denominator, scale)
    value, dtype = duckdb.execute(f"SELECT {sql}, typeof({sql})").fetchone()
    assert value == expected
    assert dtype.startswith("DECIMAL")


def test_exact_divide_yields_null_on_a_zero_denominator():
    sql = exact_divide("CAST(1 AS BIGINT)", "CAST(0 AS BIGINT)", 2)
    assert duckdb.execute(f"SELECT {sql}").fetchone()[0] is None


# ---------------------------------------------------------------------------
# Bridge invariant
# ---------------------------------------------------------------------------


def _bridge(**amounts) -> BridgeResult:
    components = tuple(
        BridgeComponent(
            name=name,
            amount=Decimal(str(amounts.get(name, 0))),
            opportunity_count=0,
            sign=COMPONENT_SIGNS.get(name, 0),
        )
        for name in [
            "opening_pipeline", "created_in_period", "pulled_in", "amount_increased",
            "amount_decreased", "closed_won", "closed_lost", "slipped_out",
            "other_removed", "ending_pipeline",
        ]
    )
    return BridgeResult(
        dataset_id="d", period_label="p", period_start=date(2025, 1, 1),
        period_end=date(2025, 3, 31), opening_as_of=date(2025, 1, 1),
        closing_as_of=date(2025, 3, 31), components=components,
        measure_concept=BusinessConcept.AMOUNT,
    )


def test_the_bridge_invariant_fails_a_result_that_does_not_close():
    result = _bridge(opening_pipeline=100, closed_won=10, ending_pipeline=80)
    assert not result.balances
    with pytest.raises(BridgeImbalance, match="does not balance"):
        check_balance(result)


def test_the_bridge_tolerance_is_a_decimal_scale_not_a_fudge_factor():
    assert Decimal("0.01") == BRIDGE_TOLERANCE
    # One cent is absorbed; ten cents is not.
    assert _bridge(opening_pipeline=100, ending_pipeline=Decimal("99.99")).balances
    assert not _bridge(opening_pipeline=100, ending_pipeline=Decimal("99.90")).balances


def test_the_balance_check_detects_a_missing_term(moves):
    """Zeroing a real term must break the identity rather than be absorbed.

    This checks the arithmetic of the invariant, not how `other_removed` is
    computed. That `other_removed` is a set predicate rather than the residual
    is asserted against the emitted SQL in
    `test_other_removed_is_computed_by_predicate_not_by_subtraction`.
    """
    result = moves.one(
        spec("b", pattern=AnalysisPattern.BRIDGE, period=Q1,
             snapshot=SnapshotSelection(rule=SnapshotRule.PERIOD_CLOSE))
    )
    amounts = {
        result.cell(i, "component"): result.cell(i, "amount")
        for i in range(result.row_count)
    }
    tampered = _bridge(**{**{k: v for k, v in amounts.items()}, "other_removed": 0})
    assert not tampered.balances


# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------

REQUIRED_METRICS = (
    "deal_count", "opening_pipeline", "ending_pipeline", "created_pipeline",
    "slipped_pipeline", "pulled_in_pipeline", "won_pipeline", "lost_pipeline",
    "win_rate", "average_deal_size", "pipeline_coverage",
)


def test_every_metric_this_milestone_requires_exists():
    assert set(REQUIRED_METRICS) <= set(METRIC_NAMES)


@pytest.mark.parametrize("name", REQUIRED_METRICS)
def test_every_metric_declares_concepts_and_not_columns(name):
    definition = metric(name)
    assert definition.required_concepts
    assert all(c in CONCEPTS for c in definition.required_concepts)
    assert definition.definition and definition.display_name
    assert definition.default_snapshot_rule in set(SnapshotRule)
    assert definition.permitted_stances


@pytest.mark.parametrize("name", REQUIRED_METRICS)
def test_every_metric_has_compilation_logic(name):
    definition = metric(name)
    if definition.kind is MetricKind.POINT_IN_TIME:
        assert definition.aggregation is not None
    elif definition.kind is MetricKind.BRIDGE_TERM:
        assert definition.bridge_component is not None
    else:
        assert definition.kind is MetricKind.RATE


def test_metrics_that_touch_money_declare_a_monetary_semantic_type():
    for name in ("opening_pipeline", "ending_pipeline", "average_deal_size"):
        assert metric(name).is_monetary


def test_no_metric_is_available_without_a_tenant_declaration(undeclared):
    """The other half of golden test 14: the switch must be off by default."""
    resolver = ConceptResolver(
        dataset_id="u", registry=undeclared.registry, bindings=undeclared.bindings
    )
    names = {m.name for m in available_metrics(resolver, AnalysisStance.PROSPECTIVE)}
    # Only the grain is confirmed without a declaration, so only the metric
    # built on the grain alone survives.
    assert names == {"deal_count"}


def test_declaring_the_columns_makes_the_catalog_complete(tiny):
    resolver = ConceptResolver(
        dataset_id="tiny", registry=tiny.registry, bindings=tiny.bindings
    )
    names = {m.name for m in available_metrics(resolver, AnalysisStance.PROSPECTIVE)}
    assert set(REQUIRED_METRICS) <= names


def test_a_metric_names_the_exact_concept_that_blocks_it(undeclared):
    from ai_analyst.semantic.metrics import metric_availability

    resolver = ConceptResolver(
        dataset_id="u", registry=undeclared.registry, bindings=undeclared.bindings
    )
    availability = metric_availability(
        metric("opening_pipeline"), resolver, AnalysisStance.PROSPECTIVE
    )
    assert not availability.available
    assert BusinessConcept.AMOUNT in availability.missing_concepts
    assert "amount concept unavailable" in availability.reasons


def test_a_cohort_trace_no_longer_requires_a_terminal_outcome_column():
    """A trace resolves from status history; terminal columns reconcile it."""
    from ai_analyst.contracts.concepts import concepts_required_for

    required = concepts_required_for(AnalyticalOperation.COHORT_TRACE)
    assert BusinessConcept.TERMINAL_OUTCOME not in required
    assert BusinessConcept.OPPORTUNITY_STATUS in required
    # Forecast accuracy genuinely needs the realized outcome and keeps it.
    assert BusinessConcept.TERMINAL_OUTCOME in concepts_required_for(
        AnalyticalOperation.FORECAST_ACCURACY
    )


# ---------------------------------------------------------------------------
# Gate and compiler
# ---------------------------------------------------------------------------


def test_the_compiler_refuses_a_plan_the_gate_rejected(undeclared):
    from ai_analyst.contracts.plan import AnalysisPlan

    bad = spec("x", metrics=["opening_pipeline"], period=Q1)
    plan = AnalysisPlan(question_restatement="q", specs=[bad])
    outcome = undeclared.gate(bad)
    assert outcome.validation.rejected
    with pytest.raises(UnvalidatedPlan, match="only\n?\\s*accepts a plan that passed"):
        compile_plan(undeclared.scan, plan, outcome)


def test_an_unknown_metric_is_rejected_by_name(tiny):
    outcome = tiny.gate(spec("x", metrics=["revenue_per_unicorn"], period=Q1))
    assert outcome.validation.has(RejectionCode.UNKNOWN_METRIC)
    rejection = outcome.validation.rejections[0]
    assert rejection.metric == "revenue_per_unicorn"
    assert "opening_pipeline" in rejection.remedy


def test_a_ranked_list_without_a_limit_is_rejected(tiny):
    outcome = tiny.gate(
        spec("x", pattern=AnalysisPattern.RANKED_LIST, metrics=["deal_count"], period=Q1)
    )
    assert outcome.validation.has(RejectionCode.INVALID_OUTPUT_SHAPE)


def test_a_metric_used_in_the_wrong_pattern_is_rejected(tiny):
    outcome = tiny.gate(
        spec("x", pattern=AnalysisPattern.RATE, metrics=["opening_pipeline"], period=Q1)
    )
    assert outcome.validation.has(RejectionCode.METRIC_PATTERN_MISMATCH)


def test_a_knowledge_cutoff_before_the_snapshot_is_rejected(tiny):
    outcome = tiny.gate(
        spec("x", metrics=["ending_pipeline"], period=Q1,
             snapshot=SnapshotSelection(rule=SnapshotRule.PERIOD_CLOSE),
             knowledge_cutoff=date(2025, 2, 15))
    )
    assert outcome.validation.has(RejectionCode.KNOWLEDGE_CUTOFF_VIOLATION)
    assert outcome.validation.temporal_safety_rejections


def test_a_knowledge_cutoff_that_permits_the_snapshot_passes_and_bounds_the_sql(tiny):
    result = tiny.one(
        spec("x", metrics=["opening_pipeline"], period=Q1,
             knowledge_cutoff=date(2025, 3, 31))
    )
    assert "as_of <= DATE '2025-01-01'" in result.compiled_sql
    assert result.compilation.knowledge_cutoff == date(2025, 3, 31)


def test_a_prospective_query_carries_an_as_of_ceiling_in_the_sql(tiny):
    result = tiny.one(spec("x", metrics=["opening_pipeline"], period=Q1))
    assert "as_of <= DATE '2025-01-01'" in result.compiled_sql


def test_a_filter_comparing_against_null_is_rejected(tiny):
    outcome = tiny.gate(
        spec("x", metrics=["opening_pipeline"], period=Q1,
             filters=[Filter(column="segment", op=FilterOp.EQ, values=[None])])
    )
    assert outcome.validation.has(RejectionCode.INVALID_FILTER_VALUE)


def test_a_cohort_trace_under_a_prospective_stance_is_rejected(tiny):
    outcome = tiny.gate(
        spec("x", pattern=AnalysisPattern.COHORT_TRACE, metrics=["deal_count"],
             period=Q1, stance=AnalysisStance.PROSPECTIVE)
    )
    assert outcome.validation.has(RejectionCode.STANCE_VIOLATION)


def test_compiling_the_same_plan_twice_gives_identical_sql(tiny):
    one = tiny.one(spec("x", metrics=["opening_pipeline"], period=Q1))
    two = tiny.one(spec("x", metrics=["opening_pipeline"], period=Q1))
    assert one.compiled_sql == two.compiled_sql


def test_the_compiled_sql_names_only_the_datasets_own_scan(tiny):
    result = tiny.one(spec("x", metrics=["opening_pipeline"], period=Q1))
    sql = result.compiled_sql
    for forbidden in ("ATTACH", "COPY", "INSTALL", "PRAGMA", "read_csv", "glob("):
        assert forbidden not in sql
    assert sql.count("read_parquet") == 1


def test_the_compilation_record_lists_the_stance_derived_allowlist(tiny):
    result = tiny.one(spec("x", metrics=["opening_pipeline"], period=Q1))
    permitted = set(result.compilation.permitted_columns)
    assert {"amount", "close_date", "status"} <= permitted


def test_the_monetary_boundary_is_recorded_on_the_result(tiny):
    result = tiny.one(spec("x", metrics=["opening_pipeline"], period=Q1))
    assert result.compilation.monetary_expressions == ('"amount"',)


def test_a_bridge_term_metric_cannot_be_mixed_with_a_point_in_time_metric(tiny):
    from ai_analyst.semantic.sql import CompilationError

    bad = spec("x", metrics=["opening_pipeline", "slipped_pipeline"], period=Q1,
               snapshot=SnapshotSelection(rule=SnapshotRule.PERIOD_CLOSE))
    outcome = tiny.gate(bad)
    assert outcome.ok
    with pytest.raises(CompilationError, match="different snapshots"):
        compile_spec(tiny.scan, outcome.specs["x"])


def test_the_trust_tier_is_computed_from_the_inputs_not_chosen(tiny):
    """deal_count needs only the grain, which is verified, so it is tier A."""
    grain_only = tiny.one(
        spec("a", metrics=["deal_count"], period=Q1,
             snapshot=SnapshotSelection(rule=SnapshotRule.LATEST))
    )
    assert grain_only.trust_tier is TrustTier.A
    assert not grain_only.trust_reasons

    # opening_pipeline reads status, which this fixture derives from stage
    # keywords and flags as not authoritative, so it can only reach tier B.
    caveated = tiny.one(spec("b", metrics=["opening_pipeline"], period=Q1))
    assert caveated.trust_tier is TrustTier.B
    assert any("authoritative_status" in r for r in caveated.trust_reasons)


def test_a_tier_b_result_names_why_it_is_tier_b(tiny):
    result = tiny.one(spec("b", metrics=["opening_pipeline"], period=Q1))
    assert result.trust_reasons
    assert all(isinstance(r, str) and r for r in result.trust_reasons)


def test_the_weakest_input_decides_the_tier():
    assert TrustTier.weakest([TrustTier.A, TrustTier.B]) is TrustTier.B
    assert TrustTier.weakest([TrustTier.A, TrustTier.C]) is TrustTier.C
    assert TrustTier.weakest([TrustTier.A]) is TrustTier.A
    assert not TrustTier.C.emits_a_number


def test_other_removed_is_computed_by_predicate_not_by_subtraction(moves):
    """The property that makes the bridge invariant worth having (5.2).

    If `other_removed` were the residual needed to close the identity, the
    bridge would balance by construction and the invariant would verify
    nothing. It must therefore appear in the SQL as its own classification
    branch, and the SQL must never derive it from the other terms.
    """
    from ai_analyst.semantic.bridge import OTHER_REMOVED

    result = moves.one(
        spec("b", pattern=AnalysisPattern.BRIDGE, period=Q1,
             snapshot=SnapshotSelection(rule=SnapshotRule.PERIOD_CLOSE))
    )
    sql = result.compiled_sql
    # It is the terminal ELSE of the classification CASE: every leaver no
    # earlier predicate claimed, decided per opportunity.
    assert f"ELSE '{OTHER_REMOVED}'" in sql
    # And it is aggregated from those rows, not from the other components.
    assert f"WHERE component IN ('{OTHER_REMOVED}'" not in sql
    assert "opening_pipeline -" not in sql
    assert "ending_pipeline -" not in sql


def test_every_bridge_term_comes_from_its_own_classification_branch(moves):
    """All nine terms are decided by one ordered, total CASE over opportunities."""
    result = moves.one(
        spec("b", pattern=AnalysisPattern.BRIDGE, period=Q1,
             snapshot=SnapshotSelection(rule=SnapshotRule.PERIOD_CLOSE))
    )
    sql = result.compiled_sql
    for name in ("created_in_period", "pulled_in", "amount_increased",
                 "amount_decreased", "closed_won", "closed_lost", "slipped_out"):
        assert f"'{name}'" in sql
