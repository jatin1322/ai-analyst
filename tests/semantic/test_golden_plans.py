"""Golden tests: manually authored plans with hand-computed expected values.

Every number below was derived by reading `tests/fixtures/tiny/snapshots.csv`
and `bridge_moves.csv` by hand, with the arithmetic shown. **No expected value
was produced by running the compiler.** That rule is what makes these tests an
instrument rather than a snapshot of current behaviour: a compiler bug that
changes a number breaks a test here, where blessing the output would have
hidden it.

The derivations refer to the two fixtures' documented hard cases. Status is
derived from stage keywords on these fixtures, so `Closed Won` is won,
`Closed Lost` is lost, and every other stage is open.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

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
from ai_analyst.contracts.rejection import RejectionCode
from ai_analyst.contracts.result import SnapshotRule, TrustTier
from ai_analyst.semantic.bridge import BRIDGE_COMPONENTS, COMPONENT_SIGNS

Q1 = Period(kind=PeriodKind.FISCAL_QUARTER, label="FY2025-Q1")
Q2 = Period(kind=PeriodKind.FISCAL_QUARTER, label="FY2025-Q2")


def spec(spec_id: str, **kwargs) -> AnalysisSpec:
    kwargs.setdefault("pattern", AnalysisPattern.POINT_IN_TIME)
    kwargs.setdefault("period", Q1)
    kwargs.setdefault("snapshot", SnapshotSelection(rule=SnapshotRule.PERIOD_OPEN))
    return AnalysisSpec(id=spec_id, **kwargs)


def value(result, column: str, row: int = 0):
    return result.cell(row, column)


# ---------------------------------------------------------------------------
# 1. Opening pipeline by quarter
# ---------------------------------------------------------------------------
# Q1 opens at the 2025-01-01 snapshot (exact, no drift). Open rows whose close
# date falls in Q1 (2025-01-01..2025-03-31):
#   OPP-001  close 2025-03-15  Negotiation  100000
#   OPP-005  close 2025-03-20  Discovery     40000
#   OPP-007  close 2025-02-28  Negotiation   60000
# OPP-002 (2025-05-20), OPP-003 (2025-06-10) and OPP-004 (2025-06-20) close in
# Q2 and are not Q1 pipeline.
#   100000 + 40000 + 60000 = 200000
#
# Q2 opens at 2025-04-01. Open rows closing in Q2:
#   OPP-001  close 2025-05-15  100000
#   OPP-003  close 2025-06-10   75000
#   OPP-004  close 2025-06-20  250000   (raised from 200000 at 2025-03-31)
#   OPP-008  close 2025-04-30   90000
# OPP-002 is Closed Won and OPP-007 Closed Lost, so neither is open pipeline.
#   100000 + 75000 + 250000 + 90000 = 515000


@pytest.mark.parametrize(
    ("period", "rule", "expected", "opportunities"),
    [
        (Q1, SnapshotRule.PERIOD_OPEN, Decimal("200000.00"), 3),
        (Q2, SnapshotRule.PERIOD_OPEN, Decimal("515000.00"), 4),
    ],
)
def test_opening_pipeline_by_quarter(tiny, period, rule, expected, opportunities):
    result = tiny.one(
        spec("opening", metrics=["opening_pipeline"], period=period,
             snapshot=SnapshotSelection(rule=rule))
    )
    assert value(result, "opening_pipeline") == expected
    assert result.resolved_snapshots[0].drift_days == 0


# ---------------------------------------------------------------------------
# 2. Ending pipeline by quarter
# ---------------------------------------------------------------------------
# Q1 closes at 2025-03-31. Open rows closing in Q1:
#   OPP-001 moved its close date to 2025-05-15, so it left Q1.
#   OPP-002 close 2025-03-25 but Closed Won, so not open.
#   OPP-005 close 2025-03-20 Discovery  40000   <- the only one
#   OPP-007 Closed Lost.
#   = 40000
#
# Q2 closes at 2025-06-30. Open rows closing in Q2:
#   OPP-004 close 2025-06-20 Negotiation 180000 (cut from 250000)   <- only one
#   OPP-001 moved to 2025-08-15 (Q3); OPP-003 and OPP-008 closed won.
#   = 180000


@pytest.mark.parametrize(
    ("period", "expected"),
    [(Q1, Decimal("40000.00")), (Q2, Decimal("180000.00"))],
)
def test_ending_pipeline_by_quarter(tiny, period, expected):
    result = tiny.one(
        spec("ending", metrics=["ending_pipeline"], period=period,
             snapshot=SnapshotSelection(rule=SnapshotRule.PERIOD_CLOSE))
    )
    assert value(result, "ending_pipeline") == expected


# ---------------------------------------------------------------------------
# 3, 4, 5, 6, 7. Created, slipped, pulled-in, won and lost pipeline
# ---------------------------------------------------------------------------
# All five are read off the bridge, so they are by construction the same
# numbers the full decomposition reports. The `bridge_moves` fixture is used
# because it is built so that every one of them is non-zero; on the tiny
# fixture created and pulled-in are both genuinely zero (see the bridge test).
#
# bridge_moves, Q1 2025, opening 2025-01-01, closing 2025-03-31:
#   B-004 first appears at 2025-03-31, created 2025-02-05 (in Q1), close
#         2025-03-25, Proposal  ->  created_in_period   40000
#   B-003 close 2025-05-10 at open (Q2, so not Q1 pipeline), 2025-03-20 at
#         close (Q1), still open  ->  pulled_in         30000
#   B-007 close 2025-03-07 at open, 2025-06-15 at close (Q2), still open
#                                 ->  slipped_out       70000
#   B-005 Closed Won at 2025-03-31 ->  closed_won       50000
#   B-006 Closed Lost at 2025-03-31 -> closed_lost      60000


@pytest.mark.parametrize(
    ("metric", "expected", "opportunities"),
    [
        ("created_pipeline", Decimal("40000.00"), 1),
        ("pulled_in_pipeline", Decimal("30000.00"), 1),
        ("slipped_pipeline", Decimal("70000.00"), 1),
        ("won_pipeline", Decimal("50000.00"), 1),
        ("lost_pipeline", Decimal("60000.00"), 1),
    ],
)
def test_bridge_term_metrics(moves, metric, expected, opportunities):
    result = moves.one(
        spec("terms", metrics=[metric], period=Q1,
             snapshot=SnapshotSelection(rule=SnapshotRule.PERIOD_CLOSE))
    )
    assert value(result, metric) == expected
    assert value(result, f"{metric}_opportunities") == opportunities


def test_a_bridge_term_metric_equals_the_full_bridges_own_term(moves):
    """The two paths must agree, or one view of the number is wrong."""
    term = moves.one(
        spec("term", metrics=["slipped_pipeline"], period=Q1,
             snapshot=SnapshotSelection(rule=SnapshotRule.PERIOD_CLOSE))
    )
    full = moves.one(
        spec("full", pattern=AnalysisPattern.BRIDGE, period=Q1,
             snapshot=SnapshotSelection(rule=SnapshotRule.PERIOD_CLOSE))
    )
    rows = {full.cell(i, "component"): full.cell(i, "amount") for i in range(full.row_count)}
    assert value(term, "slipped_pipeline") == rows["slipped_out"]


# ---------------------------------------------------------------------------
# 8. Win rate
# ---------------------------------------------------------------------------
# Q1, evaluated at the closing snapshot 2025-03-31, over rows whose close date
# falls in Q1:
#   OPP-002  close 2025-03-25  Closed Won   -> won
#   OPP-005  close 2025-03-20  Discovery    -> open, excluded from closed-only
#   OPP-007  close 2025-02-28  Closed Lost  -> lost
#   won = 1, won + lost = 2, ratio = 1/2 = 0.500000


def test_win_rate(tiny):
    result = tiny.one(
        spec("rate", pattern=AnalysisPattern.RATE, metrics=["win_rate"], period=Q1,
             snapshot=SnapshotSelection(rule=SnapshotRule.PERIOD_CLOSE))
    )
    assert value(result, "numerator") == 1
    assert value(result, "denominator") == 2
    assert value(result, "ratio") == Decimal("0.500000")


def test_win_rate_exposes_both_components_not_only_the_ratio(tiny):
    result = tiny.one(
        spec("rate", pattern=AnalysisPattern.RATE, metrics=["win_rate"], period=Q1,
             snapshot=SnapshotSelection(rule=SnapshotRule.PERIOD_CLOSE))
    )
    assert set(result.column_names) >= {"numerator", "denominator", "ratio"}


# ---------------------------------------------------------------------------
# 9. Average deal size
# ---------------------------------------------------------------------------
# Q1 opening pipeline is 200000 across 3 opportunities (derivation above).
#   200000 / 3 = 66666.666..., truncated at cent scale = 66666.66
# Truncation rather than rounding is the documented behaviour of `exact_divide`.


def test_average_deal_size(tiny):
    result = tiny.one(spec("avg", metrics=["average_deal_size"], period=Q1))
    assert value(result, "average_deal_size") == Decimal("66666.66")


def test_a_monetary_average_is_decimal_never_a_float(tiny):
    result = tiny.one(spec("avg", metrics=["average_deal_size"], period=Q1))
    assert isinstance(value(result, "average_deal_size"), Decimal)
    # The SQL must not contain a bare division, which DuckDB evaluates as DOUBLE.
    assert "AVG(" not in result.compiled_sql


# ---------------------------------------------------------------------------
# 10. Bridge reconciliation
# ---------------------------------------------------------------------------
# bridge_moves, Q1 2025. Opening pipeline at 2025-01-01, open with close in Q1:
#   B-001 10000, B-002 20000, B-005 50000, B-006 60000, B-007 70000, B-008 80000
#   (B-003 closes 2025-05-10, which is Q2, so it is not Q1 opening pipeline)
#   = 10000 + 20000 + 50000 + 60000 + 70000 + 80000 = 290000
# Ending pipeline at 2025-03-31, open with close in Q1:
#   B-001 15000, B-002 12000, B-003 30000, B-004 40000 = 97000
# Movement:
#   + created_in_period  40000  (B-004)
#   + pulled_in          30000  (B-003)
#   + amount_increased    5000  (B-001: 10000 -> 15000)
#   - amount_decreased    8000  (B-002: 20000 -> 12000)
#   - closed_won         50000  (B-005)
#   - closed_lost        60000  (B-006)
#   - slipped_out        70000  (B-007)
#   - other_removed      80000  (B-008, absent at close, no terminal state)
#   net = 40000 + 30000 + 5000 - 8000 - 50000 - 60000 - 70000 - 80000 = -193000
#   290000 - 193000 = 97000 = ending pipeline

EXPECTED_BRIDGE = {
    "opening_pipeline": (Decimal("290000.00"), 6),
    "created_in_period": (Decimal("40000.00"), 1),
    "pulled_in": (Decimal("30000.00"), 1),
    "amount_increased": (Decimal("5000.00"), 1),
    "amount_decreased": (Decimal("8000.00"), 1),
    "closed_won": (Decimal("50000.00"), 1),
    "closed_lost": (Decimal("60000.00"), 1),
    "slipped_out": (Decimal("70000.00"), 1),
    "other_removed": (Decimal("80000.00"), 1),
    "ending_pipeline": (Decimal("97000.00"), 4),
}


def test_bridge_reconciliation(moves):
    result = moves.one(
        spec("bridge", pattern=AnalysisPattern.BRIDGE, period=Q1,
             snapshot=SnapshotSelection(rule=SnapshotRule.PERIOD_CLOSE))
    )
    got = {
        result.cell(i, "component"): (
            result.cell(i, "amount"),
            result.cell(i, "opportunity_count"),
        )
        for i in range(result.row_count)
    }
    assert got == EXPECTED_BRIDGE


def test_the_bridge_identity_closes_exactly(moves):
    result = moves.one(
        spec("bridge", pattern=AnalysisPattern.BRIDGE, period=Q1,
             snapshot=SnapshotSelection(rule=SnapshotRule.PERIOD_CLOSE))
    )
    amounts = {
        result.cell(i, "component"): result.cell(i, "amount")
        for i in range(result.row_count)
    }
    movement = sum(
        (amounts[name] * sign for name, sign in COMPONENT_SIGNS.items()),
        start=Decimal("0"),
    )
    # 290000 + (-193000) = 97000, exactly, with no tolerance consumed.
    assert amounts["opening_pipeline"] + movement == amounts["ending_pipeline"]


def test_every_bridge_term_is_reported_even_when_zero(tiny):
    """A term with no rows reports zero rather than vanishing from the result."""
    result = tiny.one(
        spec("bridge", pattern=AnalysisPattern.BRIDGE, period=Q1,
             snapshot=SnapshotSelection(rule=SnapshotRule.PERIOD_CLOSE))
    )
    components = [result.cell(i, "component") for i in range(result.row_count)]
    assert components == list(BRIDGE_COMPONENTS)


# ---------------------------------------------------------------------------
# 11. A dimension breakdown
# ---------------------------------------------------------------------------
# Q1 opening pipeline at 2025-01-01, by the segment recorded at that snapshot:
#   OPP-001  Enterprise  100000
#   OPP-005  SMB          40000
#   OPP-007  SMB          60000
#   Enterprise = 100000; SMB = 40000 + 60000 = 100000
# Note OPP-003 is Mid-Market at this snapshot and Enterprise later, but it is
# not Q1 pipeline at all, so re-segmentation does not affect this breakdown.


def test_opening_pipeline_by_segment(tiny):
    result = tiny.one(
        spec("bysegment", metrics=["opening_pipeline"], dimensions=["segment"], period=Q1)
    )
    rows = {
        result.cell(i, "segment"): result.cell(i, "opening_pipeline")
        for i in range(result.row_count)
    }
    assert rows == {"Enterprise": Decimal("100000.00"), "SMB": Decimal("100000.00")}


def test_a_dimension_breakdown_sums_back_to_the_ungrouped_total(tiny):
    grouped = tiny.one(
        spec("g", metrics=["opening_pipeline"], dimensions=["segment"], period=Q1)
    )
    total = tiny.one(spec("t", metrics=["opening_pipeline"], period=Q1))
    summed = sum(
        (grouped.cell(i, "opening_pipeline") for i in range(grouped.row_count)),
        start=Decimal("0"),
    )
    assert summed == value(total, "opening_pipeline") == Decimal("200000.00")


# ---------------------------------------------------------------------------
# 12. A prospective query rejected because it references future information
# ---------------------------------------------------------------------------
# The tiny fixture has no future-contaminated column, so this uses the
# production-shaped export, where `terminal_fate` is classified
# future_contaminated and is not quarantined.


def test_a_prospective_query_cannot_read_a_future_contaminated_column(production):
    outcome = production.gate(
        spec("prospective", metrics=["deal_count"], dimensions=["terminal_fate"],
             stance=AnalysisStance.PROSPECTIVE,
             period=Period(kind=PeriodKind.RELATIVE, relative="current"),
             snapshot=SnapshotSelection(rule=SnapshotRule.LATEST))
    )
    assert outcome.validation.rejected
    assert outcome.validation.has(RejectionCode.COLUMN_NOT_KNOWABLE_AT_SNAPSHOT)
    assert outcome.validation.temporal_safety_rejections


def test_a_prospective_query_cannot_read_a_retrospective_concept(production):
    from ai_analyst.contracts.concepts import BusinessConcept
    from ai_analyst.contracts.rejection import PlanRejection
    from ai_analyst.semantic.resolver import ConceptResolver

    resolver = ConceptResolver(
        dataset_id=production.dataset_id,
        registry=production.registry,
        bindings=production.bindings,
        stance=AnalysisStance.PROSPECTIVE,
    )
    outcome = resolver.try_resolve(BusinessConcept.TERMINAL_OUTCOME)
    assert isinstance(outcome, PlanRejection)
    assert outcome.code is RejectionCode.RETROSPECTIVE_CONCEPT_IN_PROSPECTIVE


def test_a_contaminated_column_is_not_even_reachable_under_prospective(production):
    """Unreachable, not merely discouraged: it is absent from the allowlist."""
    from ai_analyst.semantic.resolver import ConceptResolver

    prospective = ConceptResolver(
        dataset_id=production.dataset_id, registry=production.registry,
        bindings=production.bindings, stance=AnalysisStance.PROSPECTIVE,
    )
    retrospective = ConceptResolver(
        dataset_id=production.dataset_id, registry=production.registry,
        bindings=production.bindings, stance=AnalysisStance.RETROSPECTIVE,
    )
    assert "terminal_fate" not in prospective.permitted_columns
    assert "terminal_fate" in retrospective.permitted_columns


# ---------------------------------------------------------------------------
# 13. A retrospective query allowed to use terminal outcome
# ---------------------------------------------------------------------------


def test_a_retrospective_query_may_read_the_terminal_outcome(production):
    outcome = production.gate(
        spec("retro", metrics=["deal_count"], dimensions=["terminal_fate"],
             stance=AnalysisStance.RETROSPECTIVE,
             period=Period(kind=PeriodKind.RELATIVE, relative="current"),
             snapshot=SnapshotSelection(rule=SnapshotRule.LATEST))
    )
    assert outcome.ok, [r.message for r in outcome.validation.rejections]


def test_a_retrospective_result_discloses_that_it_used_hindsight(production):
    results = production.run(
        spec("retro", metrics=["deal_count"], dimensions=["terminal_fate"],
             stance=AnalysisStance.RETROSPECTIVE,
             period=Period(kind=PeriodKind.RELATIVE, relative="current"),
             snapshot=SnapshotSelection(rule=SnapshotRule.LATEST))
    )
    result = results[0]
    assert result.compilation.stance == "retrospective"
    assert any("future_contaminated" in r for r in result.trust_reasons)
    assert result.trust_tier is TrustTier.B


# ---------------------------------------------------------------------------
# 14. A query rejected because a concept is unavailable
# ---------------------------------------------------------------------------
# The same data with no tenant declaration. Every concept resolves by header
# match alone, which is `inferred` and never `confirmed` (12.2), so every
# metric built on one is unavailable.


def test_a_metric_is_rejected_when_its_concept_is_not_confirmed(undeclared):
    outcome = undeclared.gate(spec("nope", metrics=["opening_pipeline"], period=Q1))
    assert outcome.validation.rejected
    assert outcome.validation.has(RejectionCode.CONCEPT_NOT_CONFIRMED)


def test_the_rejection_names_the_concept_and_says_what_would_fix_it(undeclared):
    outcome = undeclared.gate(spec("nope", metrics=["opening_pipeline"], period=Q1))
    rejection = next(
        r for r in outcome.validation.rejections
        if r.code is RejectionCode.CONCEPT_NOT_CONFIRMED
    )
    assert rejection.concept is not None
    assert rejection.metric is None or rejection.field == "metrics"
    assert "tenant profile" in rejection.remedy


def test_the_same_metric_is_available_once_the_tenant_declares_its_columns(tiny, undeclared):
    """Both halves of the switch, so neither side can rot unnoticed."""
    assert undeclared.gate(spec("a", metrics=["opening_pipeline"], period=Q1)).validation.rejected
    assert tiny.gate(spec("a", metrics=["opening_pipeline"], period=Q1)).ok


# ---------------------------------------------------------------------------
# 15. A discovered physical column reached only through a valid binding
# ---------------------------------------------------------------------------
# Two different cases, and the distinction matters.
#
# `region` is a *canonical* column of the tiny fixture that no concept binds:
# the ontology has no region concept. It is classified and knowable at its own
# snapshot, so a plan may group by it as a physical column.
#
# `enterprise_amount` is a genuinely *discovered* column: no registry places
# it, so it fails closed as unclassified and quarantined. A tenant declaration
# confirms a binding for it but says nothing about when its values are
# knowable, so the plan is still refused. The two gates are independent and
# both must pass.
#
# Q1 opening pipeline by region at 2025-01-01:
#   OPP-001 AMER 100000; OPP-005 AMER 40000; OPP-007 EMEA 60000
#   AMER = 140000, EMEA = 60000


def test_a_canonical_column_with_no_concept_can_still_be_a_dimension(tiny):
    result = tiny.one(
        spec("byregion", metrics=["opening_pipeline"], dimensions=["region"], period=Q1)
    )
    rows = {
        result.cell(i, "region"): result.cell(i, "opening_pipeline")
        for i in range(result.row_count)
    }
    assert rows == {"AMER": Decimal("140000.00"), "EMEA": Decimal("60000.00")}


def test_a_concept_named_dimension_resolves_through_the_binding_not_the_name(tiny):
    """`segment` is the concept's name; `segment` is also what it resolved to.

    The check that matters is that the *compilation record* shows the binding
    was consulted, so a tenant whose segment column is named otherwise would
    still work.
    """
    result = tiny.one(
        spec("byseg", metrics=["opening_pipeline"], dimensions=["segment"], period=Q1)
    )
    assert result.compilation.concept_columns["amount"] == "amount"
    assert "segment" in result.column_names


def test_a_measure_is_a_concept_never_a_physical_column(tiny):
    """ARCHITECTURE 13.7: `measure_column` is gone. A raw header is not a concept."""
    outcome = tiny.gate(
        spec("m", metrics=["opening_pipeline"], period=Q1, measure_concept="deal_amount")
    )
    assert outcome.validation.has(RejectionCode.UNKNOWN_MEASURE_CONCEPT)
    with pytest.raises(ValueError):
        spec("m", metrics=["opening_pipeline"], period=Q1, measure_column="amount")


def test_an_unknown_column_is_rejected_by_name(tiny):
    outcome = tiny.gate(
        spec("u", metrics=["opening_pipeline"], dimensions=["no_such_column"], period=Q1)
    )
    assert outcome.validation.has(RejectionCode.UNKNOWN_COLUMN)


# ---------------------------------------------------------------------------
# Filters, ordering, and the snapshot record
# ---------------------------------------------------------------------------
# Q1 opening pipeline filtered to segment = SMB:
#   OPP-005 40000 + OPP-007 60000 = 100000


def test_a_filter_narrows_the_population(tiny):
    result = tiny.one(
        spec("f", metrics=["opening_pipeline"], period=Q1,
             filters=[Filter(column="segment", op=FilterOp.EQ, values=["SMB"])])
    )
    assert value(result, "opening_pipeline") == Decimal("100000.00")


def test_every_result_reports_the_snapshot_it_actually_used(tiny):
    result = tiny.one(spec("s", metrics=["opening_pipeline"], period=Q1))
    assert [s.resolved_as_of for s in result.resolved_snapshots] == [date(2025, 1, 1)]
    assert result.dataset_id == "tiny"
    assert any("2025-01-01" in a for a in result.assumptions)


def test_a_discovered_column_is_refused_as_a_dimension(custom):
    """No registry places it, so nothing is established about it (5.15).

    The rejection is `column_unclassified` rather than `column_quarantined`,
    and deliberately the more specific of the two: being unclassified is *why*
    the column is quarantined, and it is the fact that tells a reader what to
    do about it.
    """
    assert "enterprise_amount" in custom.dataset.schema.discovered_columns
    assert custom.registry.get("enterprise_amount").is_quarantined
    outcome = custom.gate(
        spec("d", metrics=["opening_pipeline"], dimensions=["enterprise_amount"],
             period=Q1)
    )
    assert outcome.validation.rejected
    assert outcome.validation.has(RejectionCode.COLUMN_UNCLASSIFIED)


def test_a_discovered_column_is_outside_the_permitted_set_under_both_stances(custom):
    from ai_analyst.semantic.resolver import ConceptResolver

    for stance in (AnalysisStance.PROSPECTIVE, AnalysisStance.RETROSPECTIVE):
        resolver = ConceptResolver(
            dataset_id=custom.dataset_id, registry=custom.registry,
            bindings=custom.bindings, stance=stance,
        )
        assert "enterprise_amount" not in resolver.permitted_columns


def test_a_tenant_declaration_makes_a_discovered_column_usable_as_its_concept(
    custom_declared,
):
    """ARCHITECTURE 13.2: a declaration issues a purpose-scoped usage grant.

    `enterprise_amount` is deal_amount / 2 on every row. Q1 opening pipeline at
    2025-01-01 through it is therefore half the canonical figure:
      OPP-001 50000 + OPP-005 20000 + OPP-007 30000 = 100000
    (the canonical opening pipeline is 200000, golden test 1).
    """
    result = custom_declared.one(spec("m", metrics=["opening_pipeline"], period=Q1))
    assert value(result, "opening_pipeline") == Decimal("100000.00")
    # Read through the monetary boundary: a DOUBLE column, cast explicitly.
    assert 'CAST("enterprise_amount" AS DECIMAL(18,2))' in result.compiled_sql
    assert result.compilation.concept_columns["amount"] == "enterprise_amount"
    assert any("enterprise_amount" in g for g in result.compilation.usage_grants)


def test_the_grant_does_not_make_the_raw_column_generically_usable(custom_declared):
    outcome = custom_declared.gate(
        spec("d", metrics=["opening_pipeline"], dimensions=["enterprise_amount"], period=Q1)
    )
    assert outcome.validation.has(RejectionCode.COLUMN_UNCLASSIFIED)


def test_the_declared_binding_still_records_the_column_it_names(custom_declared):
    """The declaration is not discarded; only the read is refused."""
    from ai_analyst.contracts.concepts import BusinessConcept

    assert custom_declared.bindings.columns_for(BusinessConcept.AMOUNT) == (
        "enterprise_amount",
    )
