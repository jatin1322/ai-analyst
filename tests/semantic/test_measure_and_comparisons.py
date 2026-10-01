"""Measure concepts and typed compiled comparisons (ARCHITECTURE 13.7).

A plan names *what to measure* as an ontology concept, never as a physical
column, and the resolver alone turns the concept into a column. A comparison
is two typed operands, an operator and a result kind, compiled to one exact
expression; operands of different units are refused, never coerced.

Hand-computed values from `tests/fixtures/tiny/snapshots.csv`:

    Q1 opening pipeline, snapshot 2025-01-01, open, close date in Q1:
        OPP-001 100000 (Enterprise), OPP-005 40000 (SMB), OPP-007 60000 (SMB)
        = 200000
    Q2 opening pipeline, snapshot 2025-04-01, open, close date in Q2:
        OPP-001 100000 (Ent), OPP-003 75000 (Ent at 04-01), OPP-004 250000 (Ent),
        OPP-008 90000 (Mid-Market)
        = 515000
    Change 515000 - 200000 = 315000; relative 315000 / 200000 = 1.575000.

    deal_count at 2025-01-01: OPP-001..005, 007 = 6
    deal_count at 2025-02-01: the same six plus OPP-008 = 7
    Change 1; relative 1 / 6 = 0.166666 (truncated at six places, never rounded).
"""

from __future__ import annotations

from decimal import Decimal

import duckdb
import pytest
from pydantic import ValidationError

from ai_analyst.contracts.columns import (
    Availability,
    ColumnCategory,
    ColumnClassification,
    Disposition,
    MonetaryStatus,
)
from ai_analyst.contracts.comparison import ComparisonOperator, Operand
from ai_analyst.contracts.plan import (
    AnalysisPattern,
    AnalysisPlan,
    AnalysisSpec,
    AnalysisStance,
    Comparison,
    ComparisonKind,
    Period,
    PeriodKind,
    SnapshotSelection,
)
from ai_analyst.contracts.rejection import RejectionCode
from ai_analyst.contracts.result import SnapshotRule, TrustTier, ValueKind
from ai_analyst.contracts.tenant import ColumnDeclaration, TenantProfile
from ai_analyst.semantic import gate as gate_module
from ai_analyst.semantic.comparisons import IncompatibleComparison, compile_comparison
from ai_analyst.semantic.compiler import compile_spec
from ai_analyst.semantic.sql import CompilationError, exact_divide
from tests.semantic.conftest import CUSTOM_CSV, TINY_TENANT, build_engine

Q1 = Period(kind=PeriodKind.FISCAL_QUARTER, label="FY2025-Q1")
Q2 = Period(kind=PeriodKind.FISCAL_QUARTER, label="FY2025-Q2")
FEB = Period(kind=PeriodKind.MONTH, label="2025-02")
POP = Comparison(kind=ComparisonKind.PERIOD_OVER_PERIOD)


def spec(spec_id: str = "a", **kwargs) -> AnalysisSpec:
    kwargs.setdefault("pattern", AnalysisPattern.POINT_IN_TIME)
    kwargs.setdefault("period", Q1)
    kwargs.setdefault("metrics", ["opening_pipeline"])
    kwargs.setdefault("snapshot", SnapshotSelection(rule=SnapshotRule.PERIOD_OPEN))
    return AnalysisSpec(id=spec_id, **kwargs)


def row(result, index: int = 0) -> dict:
    return {c.name: result.cell(index, c.name) for c in result.columns}


def _generic_money_tenant() -> TenantProfile:
    """Tiny's declarations plus a confirmed *generic* classification of
    `enterprise_amount` as money: readable as a feature, never as a concept."""
    return TenantProfile(
        tenant_id="acme",
        source="owner@acme",
        concept_columns=TINY_TENANT.concept_columns,
        column_classifications=[
            ColumnDeclaration(
                column="enterprise_amount",
                classification=ColumnClassification(
                    name="enterprise_amount",
                    category=ColumnCategory.DEAL_FEATURE,
                    availability=Availability.AS_OF_FACT,
                    disposition=Disposition.DIRECT,
                    monetary=MonetaryStatus.MONETARY,
                ),
                source="owner@acme",
            )
        ],
        fiscal_year_start_month=1,
    )


# ============================================================================
# 1. Round-trip
# ============================================================================


def test_a_spec_with_a_measure_concept_and_comparison_round_trips():
    original = spec(period=Q2, measure_concept="amount", comparison=POP)
    plan = AnalysisPlan(question_restatement="q", specs=[original])
    again = AnalysisPlan.model_validate_json(plan.model_dump_json())
    assert again.specs[0] == original
    assert again.specs[0].measure_concept == "amount"
    assert again.specs[0].comparison.kind is ComparisonKind.PERIOD_OVER_PERIOD


def test_the_measure_concept_defaults_to_the_metrics_own():
    assert spec().measure_concept is None


def test_a_compiled_comparison_round_trips():
    left = Operand(label="l", sql="1", kind=ValueKind.MONEY)
    right = Operand(label="r", sql="2", kind=ValueKind.MONEY)
    compiled = compile_comparison("x", left, right, ComparisonOperator.DIFFERENCE)
    assert type(compiled).model_validate_json(compiled.model_dump_json()) == compiled


# ============================================================================
# 2. Concept resolution
# ============================================================================


def test_the_measure_concept_resolves_through_the_binding(tiny):
    result = tiny.one(spec(measure_concept="amount"))
    assert result.cell(0, "opening_pipeline") == Decimal("200000.00")
    # Tiny's deal_amount header is conformed to the canonical DECIMAL `amount`.
    assert 'SUM("amount")' in result.compiled_sql


def test_the_same_concept_resolves_to_another_tenants_column(custom_declared):
    # enterprise_amount is deal_amount / 2: 50000 + 20000 + 30000 = 100000.
    result = custom_declared.one(spec(measure_concept="amount"))
    assert result.cell(0, "opening_pipeline") == Decimal("100000.00")
    assert 'CAST("enterprise_amount" AS DECIMAL(18,2))' in result.compiled_sql
    assert "deal_amount" not in result.compiled_sql


def test_a_retrospective_measure_concept_is_refused_prospectively(tiny):
    outcome = tiny.gate(spec(measure_concept="terminal_amount"))
    assert outcome.validation.has(RejectionCode.RETROSPECTIVE_CONCEPT_IN_PROSPECTIVE)


def test_a_non_measure_concept_is_refused(tiny):
    outcome = tiny.gate(spec(measure_concept="stage"))
    assert outcome.validation.has(RejectionCode.MEASURE_CONCEPT_NOT_APPLICABLE)


def test_a_metric_that_aggregates_no_measure_takes_none(tiny):
    outcome = tiny.gate(spec(metrics=["deal_count"], measure_concept="amount"))
    assert outcome.validation.has(RejectionCode.MEASURE_CONCEPT_NOT_APPLICABLE)


# ============================================================================
# 3. Unknown concept rejection
# ============================================================================


@pytest.mark.parametrize("name", ["revenue", "deal_amount", "annual_recurring_revenue"])
def test_an_unknown_measure_concept_is_refused(tiny, name):
    outcome = tiny.gate(spec(measure_concept=name))
    (rejection,) = outcome.validation.rejections
    assert rejection.code is RejectionCode.UNKNOWN_MEASURE_CONCEPT
    assert rejection.field == "measure_concept"


@pytest.mark.parametrize(
    "name", ["Deal Amount", "amount; DROP TABLE x", '"deal_amount"', "", "a" * 65]
)
def test_a_measure_concept_is_a_bare_identifier(name):
    with pytest.raises(ValidationError):
        spec(measure_concept=name)


# ============================================================================
# 4. Numeric comparison
# ============================================================================


def test_a_count_comparison_is_exact_and_typed(tiny):
    result = tiny.one(spec(metrics=["deal_count"], period=FEB, comparison=POP))
    assert row(result) == {
        "deal_count": 7,
        "deal_count_baseline": 6,
        "deal_count_change": 1,
        "deal_count_pct_change": Decimal("0.166666"),
    }
    kinds = {c.name: c.kind for c in result.columns}
    assert kinds["deal_count_change"] is ValueKind.COUNT
    assert kinds["deal_count_pct_change"] is ValueKind.RATIO
    assert result.compilation.comparisons == (
        "deal_count_change = difference(deal_count 2025-02, deal_count 2025-01) as count",
        "deal_count_pct_change = relative_change(deal_count 2025-02, deal_count 2025-01) "
        "as ratio",
    )
    # deal_count reads no status, so nothing caps it.
    assert result.trust_tier is TrustTier.A


def test_both_snapshots_are_reported(tiny):
    result = tiny.one(spec(metrics=["deal_count"], period=FEB, comparison=POP))
    resolved = {s.resolved_as_of.isoformat() for s in result.resolved_snapshots}
    assert resolved == {"2025-02-01", "2025-01-01"}


# ============================================================================
# 5. Monetary comparison
# ============================================================================


def test_a_monetary_comparison_stays_decimal(tiny):
    result = tiny.one(spec(period=Q2, comparison=POP))
    assert row(result) == {
        "opening_pipeline": Decimal("515000.00"),
        "opening_pipeline_baseline": Decimal("200000.00"),
        "opening_pipeline_change": Decimal("315000.00"),
        "opening_pipeline_pct_change": Decimal("1.575000"),
    }
    assert all(isinstance(v, Decimal) for v in row(result).values())
    assert " / " not in result.compiled_sql
    kinds = {c.name: c.kind for c in result.columns}
    assert kinds["opening_pipeline_change"] is ValueKind.MONEY


def test_a_grouped_monetary_comparison_keeps_groups_on_either_side(tiny):
    # Q2 by segment at 04-01: Enterprise 100000 + 75000 + 250000 = 425000,
    # Mid-Market 90000. Q1 by segment at 01-01: Enterprise 100000, SMB 100000.
    result = tiny.one(spec(period=Q2, comparison=POP, dimensions=["segment"]))
    rows = {result.cell(i, "segment"): row(result, i) for i in range(result.row_count)}
    assert rows["Enterprise"]["opening_pipeline_change"] == Decimal("325000.00")
    assert rows["Enterprise"]["opening_pipeline_pct_change"] == Decimal("3.250000")
    # Absent in Q1: zero baseline, and a relative change that is undefined.
    assert rows["Mid-Market"]["opening_pipeline_baseline"] == Decimal("0.00")
    assert rows["Mid-Market"]["opening_pipeline_pct_change"] is None
    # Absent in Q2: the whole baseline was lost.
    assert rows["SMB"]["opening_pipeline_change"] == Decimal("-100000.00")
    assert rows["SMB"]["opening_pipeline_pct_change"] == Decimal("-1.000000")


def test_an_average_comparison_is_exact(tiny):
    # Q1 average: 200000 / 3 = 66666.66 (truncated). Q2: 515000 / 4 = 128750.
    # Change 62083.34; relative 62083.34 / 66666.66 = 0.931250...
    result = tiny.one(spec(metrics=["average_deal_size"], period=Q2, comparison=POP))
    assert row(result) == {
        "average_deal_size": Decimal("128750.00"),
        "average_deal_size_baseline": Decimal("66666.66"),
        "average_deal_size_change": Decimal("62083.34"),
        "average_deal_size_pct_change": Decimal("0.931250"),
    }


def test_exact_divide_never_rounds_a_decimal_denominator():
    # 100000 / 50000.50 = 1.99998000... A denominator cast to an integer would
    # round to 50001 and give 1.999960.
    sql = exact_divide("CAST(100000 AS DECIMAL(18,2))", "CAST(50000.50 AS DECIMAL(18,2))", 6, 2)
    value, dtype = duckdb.execute(f"SELECT {sql}, typeof({sql})").fetchone()
    assert value == Decimal("1.999980")
    assert dtype.startswith("DECIMAL")


def test_a_baseline_before_the_data_is_refused_not_zeroed(tiny):
    outcome = tiny.gate(spec(period=Q2, comparison=Comparison(kind=ComparisonKind.YEAR_OVER_YEAR)))
    (rejection,) = outcome.validation.rejections
    assert rejection.code is RejectionCode.SNAPSHOT_UNRESOLVABLE
    assert rejection.message.startswith("baseline FY2024-Q2")


def test_a_prospective_baseline_after_the_horizon_is_refused(tiny):
    outcome = tiny.gate(
        spec(period=Q1, stance=AnalysisStance.PROSPECTIVE,
             comparison=Comparison(kind=ComparisonKind.VS_PERIOD, baseline=Q2))
    )
    assert outcome.validation.has(RejectionCode.STANCE_VIOLATION)


# ============================================================================
# 6. Incompatible comparison rejection
# ============================================================================


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (ValueKind.MONEY, ValueKind.COUNT),
        (ValueKind.COUNT, ValueKind.RATIO),
        (ValueKind.MONEY, ValueKind.QUANTITY),
        (ValueKind.DATE, ValueKind.DATE),
        (ValueKind.TEXT, ValueKind.TEXT),
    ],
)
def test_operands_without_a_shared_comparable_unit_are_refused(left, right):
    with pytest.raises(IncompatibleComparison):
        compile_comparison(
            "x",
            Operand(label="l", sql="1", kind=left),
            Operand(label="r", sql="2", kind=right),
            ComparisonOperator.DIFFERENCE,
        )


def test_a_rate_cannot_be_compared_period_over_period(tiny):
    outcome = tiny.gate(
        spec(pattern=AnalysisPattern.RATE, metrics=["win_rate"], comparison=POP,
             snapshot=SnapshotSelection(rule=SnapshotRule.PERIOD_CLOSE))
    )
    assert outcome.validation.has(RejectionCode.INCOMPATIBLE_COMPARISON)


def test_a_bridge_term_cannot_be_compared_period_over_period(tiny):
    outcome = tiny.gate(spec(metrics=["created_pipeline"], period=Q2, comparison=POP,
                             snapshot=SnapshotSelection(rule=SnapshotRule.PERIOD_CLOSE)))
    assert outcome.validation.has(RejectionCode.INCOMPATIBLE_COMPARISON)


# ============================================================================
# 7. No raw-column measure injection
# ============================================================================


def test_a_physical_column_cannot_be_named_as_a_measure(tmp_path):
    """Even a confirmed, monetary, MEASURE-granted column is not a concept.

    The generic grant releases `enterprise_amount` for generic reads; it never
    makes the column a measure concept, so naming it as one is refused.
    """
    engine = build_engine(CUSTOM_CSV, "generic", tmp_path, tenant=_generic_money_tenant())
    grant = engine.bindings.generic_grant_for("enterprise_amount")
    assert grant is not None and grant.monetary
    outcome = engine.gate(spec(measure_concept="enterprise_amount"))
    assert outcome.validation.has(RejectionCode.UNKNOWN_MEASURE_CONCEPT)


@pytest.mark.parametrize("field", ["measure", "measure_column", "measure_sql"])
def test_there_is_no_field_for_a_raw_measure(field):
    with pytest.raises(ValidationError):
        spec(**{field: "deal_amount"})


# ============================================================================
# 8. Mutation: raw column names cannot bypass resolution
# ============================================================================


def test_mutation_without_the_gate_check_the_compiler_still_refuses(tiny, monkeypatch):
    monkeypatch.setattr(gate_module, "_check_measure_concept", lambda spec, resolver: [])
    outcome = tiny.gate(spec(measure_concept="deal_amount"))
    assert outcome.ok  # the gate layer is gone
    with pytest.raises(CompilationError, match="never a measure"):
        compile_spec(tiny.scan, outcome.specs["a"])


def test_mutation_the_gate_check_is_what_rejects_the_raw_name(tiny, monkeypatch):
    """The control: the rejection above comes from the measure check itself."""
    assert tiny.gate(spec(measure_concept="deal_amount")).validation.has(
        RejectionCode.UNKNOWN_MEASURE_CONCEPT
    )
    monkeypatch.setattr(gate_module, "_check_measure_concept", lambda spec, resolver: [])
    assert not tiny.gate(spec(measure_concept="deal_amount")).validation.has(
        RejectionCode.UNKNOWN_MEASURE_CONCEPT
    )
