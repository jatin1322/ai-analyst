"""The investigation path (ARCHITECTURE 13.8): contracts, gate, compiler.

Every expected value is hand-computed from `tests/fixtures/tiny/snapshots.csv`,
with the arithmetic shown. None was produced by running the compiler.

The base population used throughout: Q1 2025, cohort frozen at the
2025-01-01 snapshot, open opportunities only. Six opportunities:

    opp      amount   segment      forecast (01-01, 02-01, 03-31)   close (same)
    OPP-001  100000   Enterprise   Commit, Commit, Commit            03-15, 03-15, 05-15
    OPP-002   50000   Mid-Market   Best Case, Best Case, Commit      05-20, 05-20, 03-25
    OPP-003   75000   Mid-Market   Pipeline x3                       06-10 x3
    OPP-004  200000   Enterprise   Pipeline x3                       06-20 x3
    OPP-005   40000   SMB          Pipeline x3                       03-20 x3
    OPP-007   60000   SMB          Commit, Commit, Omitted           02-28 x3
"""

from __future__ import annotations

import csv
from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from ai_analyst.contracts.concepts import BusinessConcept as C
from ai_analyst.contracts.investigation import (
    Binning,
    BinningKind,
    DerivedFeature,
    DerivedFeatureRef,
    EvidenceRequirements,
    Grouping,
    Hypothesis,
    HypothesisKind,
    InvestigationComparison,
    InvestigationPlan,
    Operation,
    Population,
    StatisticalOperation,
    Variable,
)
from ai_analyst.contracts.plan import (
    AnalysisStance,
    Filter,
    FilterOp,
    Period,
    PeriodKind,
    SnapshotSelection,
)
from ai_analyst.contracts.rejection import RejectionCode
from ai_analyst.contracts.result import SnapshotRule, TrustFactorKind, TrustTier
from ai_analyst.semantic.compiler import UnvalidatedPlan
from ai_analyst.semantic.investigation import (
    run_investigation,
    semantic_equivalent,
    validate_investigation,
)
from tests.semantic.conftest import TINY_CSV, build_engine

Q1 = Period(kind=PeriodKind.FISCAL_QUARTER, label="FY2025-Q1")
RETRO, PROSP = AnalysisStance.RETROSPECTIVE, AnalysisStance.PROSPECTIVE
Op = StatisticalOperation
F = DerivedFeature


def derived(vid: str, feature: DerivedFeature, concept: C | None = None) -> Variable:
    return Variable(id=vid, derived=DerivedFeatureRef(feature=feature, concept=concept))


def concept(vid: str, c: C) -> Variable:
    return Variable(id=vid, concept=c)


def plan(
    *,
    variables,
    operation,
    stance=RETRO,
    grouping=(),
    window_end=SnapshotRule.PERIOD_CLOSE,
    window_start=None,
    cohort=None,
    comparison=None,
    support=1,
    cutoff=None,
    filters=(),
    status_filter=None,
    kind=HypothesisKind.ASSOCIATION,
) -> InvestigationPlan:
    population = dict(
        period=Q1,
        cohort=cohort or SnapshotSelection(rule=SnapshotRule.PERIOD_OPEN),
        window_end=SnapshotSelection(rule=window_end) if isinstance(window_end, SnapshotRule)
        else window_end,
        window_start=window_start,
        filters=list(filters),
    )
    if status_filter is not None:
        population["status_filter"] = status_filter
    return InvestigationPlan(
        question_restatement="test",
        hypotheses=[Hypothesis(statement="h", kind=kind)],
        stance=stance,
        knowledge_cutoff=cutoff,
        population=Population(**population),
        variables=list(variables),
        grouping=list(grouping),
        operation=operation,
        comparison=comparison,
        evidence=EvidenceRequirements(min_group_support=support),
    )


def gate(engine, p):
    return validate_investigation(
        p,
        dataset_id=engine.dataset_id,
        registry=engine.registry,
        bindings=engine.bindings,
        snapshots=engine.snapshots,
        calendar=engine.calendar,
    )


def run(engine, p):
    outcome = gate(engine, p)
    assert outcome.ok, [r.message for r in outcome.validation.rejections]
    with engine.store.connect() as conn:
        return run_investigation(
            conn, engine.scan, outcome, dataset_id=engine.dataset_id, calendar=engine.calendar
        )


def rows(result) -> list[dict]:
    return [dict(zip(result.column_names, r, strict=True)) for r in result.rows]


# ============================================================================
# Operations, with hand-computed goldens
# ============================================================================


def test_count_grouped_by_a_derived_feature(tiny):
    # Did forecast category change between 01-01 and 03-31?
    #   OPP-002 Best Case -> Commit; OPP-007 Commit -> Omitted: changed.
    #   OPP-001, 003, 004, 005: unchanged.
    result = run(
        tiny,
        plan(
            variables=[derived("fc_changed", F.CHANGED, C.FORECAST_CATEGORY)],
            grouping=[Grouping(variable="fc_changed")],
            operation=Operation(kind=Op.COUNT),
        ),
    )
    assert rows(result) == [
        {"fc_changed": "false", "units": 4},
        {"fc_changed": "true", "units": 2},
    ]


def test_sum_of_a_monetary_derived_value_is_exact_decimal(tiny):
    # Amount at the window end (03-31), by segment recorded at the cohort:
    #   Enterprise: OPP-001 100000 + OPP-004 250000 (raised at 03-31) = 350000
    #   Mid-Market: OPP-002  50000 + OPP-003  75000                   = 125000
    #   SMB:        OPP-005  40000 + OPP-007  60000                   = 100000
    result = run(
        tiny,
        plan(
            variables=[
                derived("amount_end", F.VALUE_AT_END, C.AMOUNT),
                concept("segment", C.CUSTOMER_SEGMENT),
            ],
            grouping=[Grouping(variable="segment")],
            operation=Operation(kind=Op.SUM, measure="amount_end"),
        ),
    )
    assert rows(result) == [
        {"segment": "Enterprise", "total": Decimal("350000.00"), "units": 2},
        {"segment": "Mid-Market", "total": Decimal("125000.00"), "units": 2},
        {"segment": "SMB", "total": Decimal("100000.00"), "units": 2},
    ]
    assert all(isinstance(r["total"], Decimal) for r in rows(result))
    assert "AVG(" not in result.compiled_sql


def test_distribution_of_push_counts_with_exact_shares(tiny):
    # Close-date pushes between adjacent snapshots, 01-01 through 06-30:
    #   OPP-001 03-15 -> 05-15 (at 03-31) -> 08-15 (at 06-30): 2 pushes.
    #   OPP-002 moved earlier (a pull-in, not a push): 0. Everyone else: 0.
    #   '0': 5 of 6 = 0.833333 (truncated); '2': 1 of 6 = 0.166666.
    result = run(
        tiny,
        plan(
            variables=[derived("pushes", F.PUSH_COUNT)],
            grouping=[Grouping(variable="pushes")],
            operation=Operation(kind=Op.DISTRIBUTION),
            window_end=SnapshotSelection(rule=SnapshotRule.AS_OF_EXACT,
                                         explicit_date=date(2025, 6, 30)),
        ),
    )
    assert rows(result) == [
        {"pushes": "0", "units": 5, "share": Decimal("0.833333")},
        {"pushes": "2", "units": 1, "share": Decimal("0.166666")},
    ]


def test_crosstab_of_two_groupings(tiny):
    #   Enterprise: 001 unchanged, 004 unchanged                -> (E, false) 2
    #   Mid-Market: 002 changed, 003 unchanged                  -> (M, false) 1, (M, true) 1
    #   SMB:        005 unchanged, 007 changed                  -> (S, false) 1, (S, true) 1
    result = run(
        tiny,
        plan(
            variables=[
                concept("segment", C.CUSTOMER_SEGMENT),
                derived("fc_changed", F.CHANGED, C.FORECAST_CATEGORY),
            ],
            grouping=[Grouping(variable="segment"), Grouping(variable="fc_changed")],
            operation=Operation(kind=Op.CROSSTAB),
        ),
    )
    assert [(r["segment"], r["fc_changed"], r["units"]) for r in rows(result)] == [
        ("Enterprise", "false", 2),
        ("Mid-Market", "false", 1),
        ("Mid-Market", "true", 1),
        ("SMB", "false", 1),
        ("SMB", "true", 1),
    ]


def test_rate_of_a_final_state_by_segment(tiny):
    # Traced through 06-30: 001 open, 002 won, 003 won, 004 open, 005 vanished,
    # 007 lost. Win rate within each cohort segment:
    #   Enterprise 0/2 = 0; Mid-Market 2/2 = 1; SMB 0/2 = 0.
    result = run(
        tiny,
        plan(
            variables=[concept("segment", C.CUSTOMER_SEGMENT), derived("fate", F.FINAL_STATE)],
            grouping=[Grouping(variable="segment")],
            operation=Operation(kind=Op.RATE_BY_GROUP, outcome="fate", outcome_value="won"),
            window_end=SnapshotSelection(rule=SnapshotRule.AS_OF_EXACT,
                                         explicit_date=date(2025, 6, 30)),
        ),
    )
    assert [(r["segment"], r["numerator"], r["denominator"], r["ratio"]) for r in rows(result)] == [
        ("Enterprise", 0, 2, Decimal("0.000000")),
        ("Mid-Market", 2, 2, Decimal("1.000000")),
        ("SMB", 0, 2, Decimal("0.000000")),
    ]


def test_difference_in_rates_against_a_reference_group(tiny):
    # Forecast-category volatility (changes, 01-01 to 03-31) against slippage
    # (close date in Q1 at 01-01, after Q1 at 03-31):
    #   volatility < 1: OPP-001 (slipped 03-15 -> 05-15), 003, 004, 005 -> 1/4 = 0.25
    #   volatility >= 1: OPP-002 (close in Q2 at the start, so cannot slip out
    #   of Q1), OPP-007 (02-28 throughout)                             -> 0/2 = 0
    #   difference vs '< 1': 0.25 - 0.25 = 0; 0 - 0.25 = -0.25
    result = run(
        tiny,
        plan(
            variables=[
                derived("volatility", F.CHANGE_COUNT, C.FORECAST_CATEGORY),
                derived("slipped", F.SLIPPED),
            ],
            grouping=[Grouping(variable="volatility",
                               binning=Binning(kind=BinningKind.EXPLICIT_EDGES, edges=[1]))],
            operation=Operation(kind=Op.DIFFERENCE_IN_RATES, outcome="slipped"),
            comparison=InvestigationComparison(reference="< 1"),
        ),
    )
    assert [
        (r["volatility"], r["numerator"], r["denominator"], r["ratio"],
         r["difference_vs_reference"])
        for r in rows(result)
    ] == [
        ("< 1", 1, 4, Decimal("0.250000"), Decimal("0.000000")),
        (">= 1", 0, 2, Decimal("0.000000"), Decimal("-0.250000")),
    ]


def test_rank_correlation(tiny):
    # Spearman between cohort amount and push count (through 06-30).
    #   amount ranks: 005=1 002=2 007=3 003=4 001=5 004=6
    #   push ranks (five ties at 0 average to 3): 001=6, others 3
    #   mean of each = 3.5; sum dx*dy = 4.5; sum dx^2 = 17.5; sum dy^2 = 7.5
    #   rho = 4.5 / sqrt(17.5 * 7.5) = 4.5 / 11.4564392 = 0.392792
    result = run(
        tiny,
        plan(
            variables=[concept("amount", C.AMOUNT), derived("pushes", F.PUSH_COUNT)],
            operation=Operation(kind=Op.RANK_CORRELATION, measure="amount", against="pushes"),
            window_end=SnapshotSelection(rule=SnapshotRule.AS_OF_EXACT,
                                         explicit_date=date(2025, 6, 30)),
        ),
    )
    assert rows(result) == [{"units": 6, "rank_correlation": Decimal("0.392792")}]


def test_trend_of_the_frozen_cohort_across_the_window(tiny):
    # The cohort's amount at each snapshot of the window, members only
    # (OPP-008 appears at 02-01 and is never counted):
    #   01-01: 100000+50000+75000+200000+40000+60000 = 525000, 6 present
    #   02-01: unchanged                            = 525000, 6 present
    #   03-31: OPP-004 raised to 250000             = 575000, 6 present
    result = run(
        tiny,
        plan(
            variables=[concept("amount", C.AMOUNT)],
            operation=Operation(kind=Op.TREND, measure="amount"),
            kind=HypothesisKind.TREND,
        ),
    )
    assert rows(result) == [
        {"as_of": date(2025, 1, 1), "units": 6, "total": Decimal("525000.00")},
        {"as_of": date(2025, 2, 1), "units": 6, "total": Decimal("525000.00")},
        {"as_of": date(2025, 3, 31), "units": 6, "total": Decimal("575000.00")},
    ]


def test_a_group_below_the_support_floor_reports_counts_and_no_rate(tiny):
    result = run(
        tiny,
        plan(
            variables=[concept("segment", C.CUSTOMER_SEGMENT), derived("slipped", F.SLIPPED)],
            grouping=[Grouping(variable="segment")],
            operation=Operation(kind=Op.RATE_BY_GROUP, outcome="slipped"),
            support=3,
        ),
    )
    for r in rows(result):
        assert r["denominator"] == 2
        assert r["ratio"] is None
        assert r["below_support"] is True


def test_a_filter_narrows_the_frozen_cohort(tiny):
    # Enterprise only: OPP-001, OPP-004, neither changed forecast category.
    result = run(
        tiny,
        plan(
            variables=[derived("fc_changed", F.CHANGED, C.FORECAST_CATEGORY)],
            grouping=[Grouping(variable="fc_changed")],
            operation=Operation(kind=Op.COUNT),
            filters=[Filter(column="segment", op=FilterOp.EQ, values=["Enterprise"])],
        ),
    )
    assert rows(result) == [{"fc_changed": "false", "units": 2}]


# ============================================================================
# Trust
# ============================================================================


def test_an_investigation_is_at_most_tier_b(tiny):
    result = run(
        tiny,
        plan(
            variables=[derived("fc_changed", F.CHANGED, C.FORECAST_CATEGORY)],
            grouping=[Grouping(variable="fc_changed")],
            operation=Operation(kind=Op.COUNT),
        ),
    )
    assert result.trust_tier is TrustTier.B
    assert TrustFactorKind.INVESTIGATION_PATH in {f.kind for f in result.trust.factors}
    assert result.compilation.path == "investigation"


def test_an_association_is_disclosed_as_not_causation(tiny):
    result = run(
        tiny,
        plan(
            variables=[derived("fc_changed", F.CHANGED, C.FORECAST_CATEGORY)],
            grouping=[Grouping(variable="fc_changed")],
            operation=Operation(kind=Op.COUNT),
        ),
    )
    assert any("not evidence that one causes the other" in a for a in result.assumptions)


# ============================================================================
# The semantic path wins
# ============================================================================


@pytest.mark.parametrize(
    ("p", "metric"),
    [
        (
            plan(variables=[concept("segment", C.CUSTOMER_SEGMENT)],
                 grouping=[Grouping(variable="segment")], operation=Operation(kind=Op.COUNT),
                 window_end=None),
            "deal_count",
        ),
        (
            plan(variables=[concept("amount", C.AMOUNT)],
                 operation=Operation(kind=Op.SUM, measure="amount"), window_end=None),
            "opening_pipeline",
        ),
        (
            plan(variables=[concept("amount", C.AMOUNT)],
                 operation=Operation(kind=Op.SUM, measure="amount"), window_end=None,
                 cohort=SnapshotSelection(rule=SnapshotRule.PERIOD_CLOSE)),
            "ending_pipeline",
        ),
        (
            plan(variables=[concept("status", C.OPPORTUNITY_STATUS),
                            concept("segment", C.CUSTOMER_SEGMENT)],
                 grouping=[Grouping(variable="segment")],
                 operation=Operation(kind=Op.RATE_BY_GROUP, outcome="status",
                                     outcome_value="won"),
                 window_end=None, status_filter=[]),
            "win_rate",
        ),
    ],
)
def test_a_plan_a_registry_metric_answers_is_rejected(tiny, p, metric):
    assert semantic_equivalent(p) == metric
    outcome = gate(tiny, p)
    assert outcome.validation.has(RejectionCode.SEMANTIC_PATH_AVAILABLE)
    rejection = next(
        r for r in outcome.validation.rejections
        if r.code is RejectionCode.SEMANTIC_PATH_AVAILABLE
    )
    assert rejection.metric == metric


def test_the_investigation_path_cannot_route_around_an_unavailable_metric(undeclared):
    """On the undeclared dataset opening_pipeline is unavailable (inferred
    bindings). An investigation summing the same amount must not get around it."""
    p = plan(variables=[concept("amount", C.AMOUNT)],
             operation=Operation(kind=Op.SUM, measure="amount"), window_end=None)
    outcome = gate(undeclared, p)
    assert outcome.validation.has(RejectionCode.SEMANTIC_PATH_AVAILABLE)


def test_a_derived_feature_is_never_a_registry_metric():
    p = plan(variables=[derived("pushes", F.PUSH_COUNT)],
             grouping=[Grouping(variable="pushes")], operation=Operation(kind=Op.COUNT))
    assert semantic_equivalent(p) is None


# ============================================================================
# Gate: shape, references, operations
# ============================================================================


def test_too_many_variables_is_rejected(tiny):
    variables = [derived(f"v{i}", F.CHANGED, C.STAGE) for i in range(7)]
    outcome = gate(tiny, plan(variables=variables, operation=Operation(kind=Op.COUNT)))
    assert outcome.validation.has(RejectionCode.TOO_MANY_VARIABLES)


def test_too_many_groupings_is_rejected(tiny):
    variables = [derived(f"v{i}", F.CHANGED, C.STAGE) for i in range(3)]
    outcome = gate(
        tiny,
        plan(variables=variables, grouping=[Grouping(variable=f"v{i}") for i in range(3)],
             operation=Operation(kind=Op.COUNT)),
    )
    assert outcome.validation.has(RejectionCode.TOO_MANY_GROUPINGS)


def test_an_undefined_variable_reference_is_rejected(tiny):
    outcome = gate(
        tiny,
        plan(variables=[derived("a", F.CHANGED, C.STAGE)],
             grouping=[Grouping(variable="nope")], operation=Operation(kind=Op.COUNT)),
    )
    assert outcome.validation.has(RejectionCode.UNKNOWN_VARIABLE)


@pytest.mark.parametrize(
    "p",
    [
        # A sum of a category.
        plan(variables=[derived("stage_end", F.VALUE_AT_END, C.STAGE)],
             operation=Operation(kind=Op.SUM, measure="stage_end")),
        # A distribution with no variable to distribute.
        plan(variables=[derived("a", F.CHANGED, C.STAGE)],
             operation=Operation(kind=Op.DISTRIBUTION)),
        # A crosstab of one grouping.
        plan(variables=[derived("a", F.CHANGED, C.STAGE)], grouping=[Grouping(variable="a")],
             operation=Operation(kind=Op.CROSSTAB)),
        # A rate over a categorical outcome with no value that counts as true.
        plan(variables=[derived("fate", F.FINAL_STATE)],
             operation=Operation(kind=Op.RATE_BY_GROUP, outcome="fate")),
        # A difference in rates with no reference group.
        plan(variables=[derived("s", F.SLIPPED), derived("c", F.CHANGED, C.STAGE)],
             grouping=[Grouping(variable="c")],
             operation=Operation(kind=Op.DIFFERENCE_IN_RATES, outcome="s")),
        # Binning a category at numeric edges.
        plan(variables=[derived("stage_end", F.VALUE_AT_END, C.STAGE)],
             grouping=[Grouping(variable="stage_end",
                                binning=Binning(kind=BinningKind.EXPLICIT_EDGES, edges=[1]))],
             operation=Operation(kind=Op.COUNT)),
    ],
)
def test_an_ill_formed_operation_is_rejected(tiny, p):
    assert gate(tiny, p).validation.has(RejectionCode.INVALID_OPERATION)


def test_a_window_that_does_not_contain_the_cohort_is_rejected(tiny):
    outcome = gate(
        tiny,
        plan(variables=[derived("a", F.CHANGED, C.STAGE)], grouping=[Grouping(variable="a")],
             operation=Operation(kind=Op.COUNT),
             cohort=SnapshotSelection(rule=SnapshotRule.PERIOD_CLOSE),
             window_end=SnapshotSelection(rule=SnapshotRule.AS_OF_EXACT,
                                          explicit_date=date(2025, 2, 1))),
    )
    assert outcome.validation.has(RejectionCode.SNAPSHOT_COVERAGE)


def test_an_unavailable_required_concept_is_rejected(tiny):
    p = plan(variables=[derived("a", F.CHANGED, C.STAGE)], grouping=[Grouping(variable="a")],
             operation=Operation(kind=Op.COUNT))
    p = p.model_copy(update={"evidence": EvidenceRequirements(
        min_group_support=1, required_concepts=[C.NARRATIVE_TEXT])})
    assert gate(tiny, p).validation.has(RejectionCode.CONCEPT_UNAVAILABLE)


def test_the_compiler_refuses_a_rejected_investigation(tiny):
    outcome = gate(tiny, plan(variables=[derived("a", F.CHANGED, C.STAGE)],
                              grouping=[Grouping(variable="nope")],
                              operation=Operation(kind=Op.COUNT)))
    with tiny.store.connect() as conn, pytest.raises(UnvalidatedPlan):
        run_investigation(conn, tiny.scan, outcome, dataset_id="tiny")


# ============================================================================
# Contracts: data, not a program
# ============================================================================


def test_a_variable_has_exactly_one_source():
    with pytest.raises(ValidationError):
        Variable(id="x", concept=C.AMOUNT, column="amount")
    with pytest.raises(ValidationError):
        Variable(id="x")


def test_a_plan_rejects_fields_it_does_not_define():
    """No smuggled SQL, expression, or code: unknown keys are errors."""
    good = plan(variables=[derived("a", F.CHANGED, C.STAGE)], operation=Operation(kind=Op.COUNT))
    data = good.model_dump(mode="json")
    for smuggled in ("sql", "expression", "python", "formula"):
        with pytest.raises(ValidationError):
            InvestigationPlan.model_validate({**data, smuggled: "SELECT 1"})
    with pytest.raises(ValidationError):
        Operation.model_validate({"kind": "sum", "measure": "a", "expression": "a * 1.1"})


def test_the_stance_has_no_default():
    with pytest.raises(ValidationError):
        InvestigationPlan(
            question_restatement="q",
            hypotheses=[Hypothesis(statement="h", kind=HypothesisKind.TREND)],
            population=Population(period=Q1),
            variables=[derived("a", F.CHANGED, C.STAGE)],
            operation=Operation(kind=Op.COUNT),
        )


def test_the_schema_is_not_recursive():
    """Structured outputs reject recursive schemas, and so does this design."""
    schema = InvestigationPlan.model_json_schema()
    for name, definition in schema.get("$defs", {}).items():
        assert f'"#/$defs/{name}"' not in str(definition).replace("'", '"')


def test_a_fixed_feature_cannot_be_pointed_at_another_concept():
    with pytest.raises(ValidationError):
        DerivedFeatureRef(feature=F.SLIPPED, concept=C.STAGE)
    with pytest.raises(ValidationError):
        DerivedFeatureRef(feature=F.CHANGE_COUNT)


def test_bin_edges_must_increase():
    with pytest.raises(ValidationError):
        Binning(kind=BinningKind.EXPLICIT_EDGES, edges=[2, 1])


# ============================================================================
# Temporal safety: adversarial leakage
# ============================================================================


def test_a_prospective_window_cannot_extend_past_the_cohort(tiny):
    outcome = gate(
        tiny,
        plan(variables=[derived("a", F.CHANGED, C.FORECAST_CATEGORY)],
             grouping=[Grouping(variable="a")], operation=Operation(kind=Op.COUNT),
             stance=PROSP),
    )
    assert outcome.validation.has(RejectionCode.STANCE_VIOLATION)
    assert outcome.validation.temporal_safety_rejections


def test_a_prospective_final_state_is_refused(tiny):
    outcome = gate(
        tiny,
        plan(variables=[derived("fate", F.FINAL_STATE)], grouping=[Grouping(variable="fate")],
             operation=Operation(kind=Op.COUNT), stance=PROSP, window_end=None),
    )
    assert outcome.validation.has(RejectionCode.STANCE_VIOLATION)


def test_a_knowledge_cutoff_bounds_even_a_retrospective_window(tiny):
    outcome = gate(
        tiny,
        plan(variables=[derived("a", F.CHANGED, C.FORECAST_CATEGORY)],
             grouping=[Grouping(variable="a")], operation=Operation(kind=Op.COUNT),
             cutoff=date(2025, 2, 15)),
    )
    assert outcome.validation.has(RejectionCode.KNOWLEDGE_CUTOFF_VIOLATION)


def test_a_prospective_cutoff_after_the_cohort_still_bounds_the_window(tiny):
    outcome = gate(
        tiny,
        plan(variables=[derived("a", F.CHANGED, C.FORECAST_CATEGORY)],
             grouping=[Grouping(variable="a")], operation=Operation(kind=Op.COUNT),
             stance=PROSP, cutoff=date(2025, 2, 15)),
    )
    assert outcome.validation.has(RejectionCode.KNOWLEDGE_CUTOFF_VIOLATION)


def _history_plan(stance=PROSP) -> InvestigationPlan:
    """History before the cohort: legal prospectively, reads nothing after it."""
    return plan(
        variables=[derived("fc_changes", F.CHANGE_COUNT, C.FORECAST_CATEGORY)],
        grouping=[Grouping(variable="fc_changes")],
        operation=Operation(kind=Op.COUNT),
        stance=stance,
        cohort=SnapshotSelection(rule=SnapshotRule.AS_OF_EXACT, explicit_date=date(2025, 2, 1)),
        window_start=SnapshotSelection(rule=SnapshotRule.AS_OF_EXACT,
                                       explicit_date=date(2025, 1, 1)),
        window_end=None,
    )


def test_a_prospective_history_window_is_allowed_and_bounded_in_the_sql(tiny):
    result = run(tiny, _history_plan())
    assert "as_of <= DATE '2025-02-01'" in result.compiled_sql
    # Open at 02-01: 001-005, 007, 008. Forecast changes 01-01 -> 02-01: none.
    assert rows(result) == [{"fc_changes": "0", "units": 7}]


def test_a_prospective_result_does_not_change_when_the_future_changes(tiny, tmp_path):
    """The decisive leakage test: rewrite every snapshot after the horizon and
    require the prospective answer to be byte-identical."""
    future = tmp_path / "future.csv"
    reader = list(csv.DictReader(TINY_CSV.open(encoding="utf-8")))
    for row in reader:
        if row["snapshot_date"] > "2025-02-01":
            row["forecast_cat"] = "Omitted"
            row["deal_amount"] = "1.00"
            row["sales_stage"] = "Closed Lost"
    with future.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(reader[0]))
        writer.writeheader()
        writer.writerows(reader)
    rewritten = build_engine(future, "future", tmp_path / "f")

    original = run(tiny, _history_plan())
    changed = run(rewritten, _history_plan())
    assert original.rows == changed.rows


def test_the_future_does_change_a_retrospective_trace(tiny, tmp_path):
    """The control for the test above: the same rewrite is visible in hindsight."""
    future = tmp_path / "future.csv"
    reader = list(csv.DictReader(TINY_CSV.open(encoding="utf-8")))
    for row in reader:
        if row["snapshot_date"] > "2025-02-01":
            row["sales_stage"] = "Closed Lost"
    with future.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(reader[0]))
        writer.writeheader()
        writer.writerows(reader)
    rewritten = build_engine(future, "future", tmp_path / "f")
    retro = plan(variables=[derived("fate", F.FINAL_STATE)], grouping=[Grouping(variable="fate")],
                 operation=Operation(kind=Op.COUNT))
    assert run(tiny, retro).rows != run(rewritten, retro).rows


def test_a_terminal_concept_is_refused_prospectively(production):
    p = plan(variables=[concept("fate", C.TERMINAL_OUTCOME),
                        derived("a", F.CHANGED, C.STAGE)],
             grouping=[Grouping(variable="fate")], operation=Operation(kind=Op.COUNT),
             stance=PROSP, window_end=None)
    outcome = gate(production, p)
    assert outcome.validation.has(RejectionCode.RETROSPECTIVE_CONCEPT_IN_PROSPECTIVE)


def test_a_future_contaminated_column_is_refused_prospectively(production):
    p = plan(variables=[Variable(id="fate", column="terminal_fate"),
                        derived("a", F.CHANGED, C.STAGE)],
             grouping=[Grouping(variable="fate")], operation=Operation(kind=Op.COUNT),
             stance=PROSP, window_end=None)
    assert gate(production, p).validation.has(RejectionCode.COLUMN_NOT_KNOWABLE_AT_SNAPSHOT)


def test_a_future_contaminated_column_is_allowed_and_disclosed_retrospectively(production):
    p = plan(variables=[Variable(id="fate", column="terminal_fate"),
                        derived("a", F.CHANGED, C.STAGE)],
             grouping=[Grouping(variable="fate")], operation=Operation(kind=Op.COUNT),
             status_filter=[])
    result = run(production, p)
    assert TrustFactorKind.RETROSPECTIVE_READ in {f.kind for f in result.trust.factors}
    assert result.compilation.stance == "retrospective"


def test_a_quarantined_column_is_refused_under_either_stance(production):
    quarantined = production.registry.quarantined()[0].name
    for stance in (PROSP, RETRO):
        p = plan(variables=[Variable(id="q", column=quarantined),
                            derived("a", F.CHANGED, C.STAGE)],
                 grouping=[Grouping(variable="q")], operation=Operation(kind=Op.COUNT),
                 stance=stance, window_end=None)
        codes = gate(production, p).validation.codes
        assert RejectionCode.COLUMN_QUARANTINED in codes or (
            RejectionCode.COLUMN_UNCLASSIFIED in codes
        )


def test_derived_features_never_read_a_precomputed_counter(production):
    p = plan(variables=[derived("pushes", F.PUSH_COUNT)], grouping=[Grouping(variable="pushes")],
             operation=Operation(kind=Op.COUNT), status_filter=[])
    result = run(production, p)
    assert "close_date_push_count" not in result.compiled_sql
    assert "push_count" not in result.compiled_sql.replace("f_pushes", "")


# ============================================================================
# Usage grants on the investigation path
# ============================================================================


def test_a_granted_column_is_usable_through_its_concept(custom_declared):
    # enterprise_amount = deal_amount / 2; amount at 03-31 for the Q1 cohort:
    #   (100000 + 50000 + 75000 + 250000 + 40000 + 60000) / 2 = 287500
    result = run(
        custom_declared,
        plan(variables=[derived("amount_end", F.VALUE_AT_END, C.AMOUNT)],
             operation=Operation(kind=Op.SUM, measure="amount_end")),
    )
    assert rows(result) == [{"total": Decimal("287500.00"), "units": 6}]
    assert 'CAST("enterprise_amount" AS DECIMAL(18,2))' in result.compiled_sql
    assert result.compilation.usage_grants


def test_a_granted_column_named_by_its_header_stays_blocked(custom_declared):
    p = plan(variables=[Variable(id="e", column="enterprise_amount"),
                        derived("a", F.CHANGED, C.STAGE)],
             grouping=[Grouping(variable="a")], operation=Operation(kind=Op.SUM, measure="e"))
    assert gate(custom_declared, p).validation.has(RejectionCode.COLUMN_UNCLASSIFIED)
