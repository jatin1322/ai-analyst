"""AnalysisPlan contracts.

These assert the structural claim of ARCHITECTURE §8.1: the plan schema cannot
express free-form arithmetic, and incoherent plans fail before execution.
"""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from ai_analyst.contracts.plan import (
    AnalysisPattern,
    AnalysisPlan,
    AnalysisSpec,
    Attribution,
    Comparison,
    ComparisonKind,
    CreationBasis,
    Filter,
    FilterOp,
    Period,
    PeriodKind,
    RateKey,
    RelativePeriod,
    SlipBasis,
    SnapshotSelection,
    WinRateBasis,
)
from ai_analyst.contracts.result import SnapshotRule


def _quarter() -> Period:
    return Period(kind=PeriodKind.FISCAL_QUARTER, label="FY25-Q3")


def _spec(**overrides) -> AnalysisSpec:
    base = {
        "id": "q1",
        "pattern": AnalysisPattern.POINT_IN_TIME,
        "metrics": ["open_pipeline"],
        "period": _quarter(),
        "snapshot": SnapshotSelection(rule=SnapshotRule.PERIOD_OPEN),
    }
    return AnalysisSpec(**{**base, **overrides})


def test_spec_defaults_match_the_documented_ambiguity_resolutions():
    # ARCHITECTURE §5.3 argues for each of these defaults.
    spec = _spec()
    assert spec.creation_basis is CreationBasis.CREATED_DATE
    assert spec.slip_basis is SlipBasis.PERIOD_MOVE
    assert spec.win_rate_basis is WinRateBasis.CLOSED_ONLY
    assert spec.rate_key is RateKey.CLOSE_DATE
    assert spec.attribution is Attribution.PERIOD_OPEN
    # ARCHITECTURE 13.7: no measure enum; None means each metric's own concept.
    assert spec.measure_concept is None


def test_plan_has_no_field_that_accepts_arithmetic():
    # The structural anti-hallucination claim: nowhere to put an expression.
    forbidden = {"expression", "formula", "sql", "compute", "calculation"}
    assert forbidden.isdisjoint(AnalysisSpec.model_fields)
    assert forbidden.isdisjoint(AnalysisPlan.model_fields)


def test_spec_is_immutable():
    spec = _spec()
    with pytest.raises(ValidationError):
        spec.limit = 10


@pytest.mark.parametrize(
    ("op", "values"),
    [
        (FilterOp.EQ, []),
        (FilterOp.EQ, ["a", "b"]),
        (FilterOp.BETWEEN, [1]),
        (FilterOp.IS_NULL, ["a"]),
        (FilterOp.IN, []),
    ],
)
def test_filter_rejects_wrong_value_arity(op, values):
    with pytest.raises(ValidationError):
        Filter(column="segment", op=op, values=values)


def test_filter_accepts_correct_arity():
    assert Filter(column="segment", op=FilterOp.EQ, values=["Enterprise"]).values == [
        "Enterprise"
    ]
    assert len(Filter(column="amount", op=FilterOp.BETWEEN, values=[1, 2]).values) == 2
    assert Filter(column="arr", op=FilterOp.IS_NULL).values == []


def test_filter_values_stay_raw_scalars():
    # Coercion happens in the plan gate, where the schema is available. A date
    # supplied as a string must survive round-trip unchanged.
    f = Filter(column="close_date", op=FilterOp.EQ, values=["2025-03-31"])
    assert f.values == ["2025-03-31"]
    assert isinstance(f.values[0], str)


def test_custom_period_requires_both_bounds():
    with pytest.raises(ValidationError, match="requires both start and end"):
        Period(kind=PeriodKind.CUSTOM, start=date(2025, 1, 1))


def test_period_rejects_inverted_bounds():
    with pytest.raises(ValidationError, match="start is after"):
        Period(kind=PeriodKind.CUSTOM, start=date(2025, 6, 1), end=date(2025, 1, 1))


def test_relative_period_requires_a_relative_value():
    with pytest.raises(ValidationError, match="requires a relative value"):
        Period(kind=PeriodKind.RELATIVE)


def test_last_n_requires_n():
    with pytest.raises(ValidationError, match="last_n requires n"):
        Period(kind=PeriodKind.RELATIVE, relative=RelativePeriod.LAST_N)


def test_exact_snapshot_requires_a_date():
    with pytest.raises(ValidationError, match="requires explicit_date"):
        SnapshotSelection(rule=SnapshotRule.AS_OF_EXACT)
    assert SnapshotSelection(
        rule=SnapshotRule.AS_OF_EXACT, explicit_date=date(2025, 3, 31)
    ).explicit_date == date(2025, 3, 31)


def test_vs_period_comparison_requires_a_baseline():
    with pytest.raises(ValidationError, match="requires a baseline"):
        Comparison(kind=ComparisonKind.VS_PERIOD)


def test_non_bridge_pattern_requires_a_metric():
    with pytest.raises(ValidationError, match="requires at least one metric"):
        _spec(metrics=[])


def test_bridge_pattern_needs_no_explicit_metric():
    spec = _spec(pattern=AnalysisPattern.BRIDGE, metrics=[])
    assert spec.pattern is AnalysisPattern.BRIDGE


def test_plan_rejects_duplicate_spec_ids():
    with pytest.raises(ValidationError, match="duplicate spec ids"):
        AnalysisPlan(
            question_restatement="x",
            specs=[_spec(id="q1"), _spec(id="q1")],
        )


def test_plan_requires_at_least_one_spec():
    with pytest.raises(ValidationError):
        AnalysisPlan(question_restatement="x", specs=[])


def test_unresolved_ambiguities_trigger_clarification():
    plan = AnalysisPlan(
        question_restatement="How many deals slipped?",
        specs=[_spec()],
        unresolved_ambiguities=["no period was specified"],
    )
    assert plan.needs_clarification
    assert plan.spec("q1").id == "q1"
    with pytest.raises(KeyError):
        plan.spec("nope")


def test_spec_id_must_be_token_safe():
    # Spec ids appear inside answer reference tokens like {{q1.r0.col}}.
    with pytest.raises(ValidationError):
        _spec(id="q1.bad")
    with pytest.raises(ValidationError):
        _spec(id="1q")


def test_plan_round_trips_through_json():
    plan = AnalysisPlan(
        question_restatement="Opening pipeline by segment",
        specs=[_spec(dimensions=["segment"], limit=20)],
        assumptions=["fiscal year starts in January"],
    )
    assert AnalysisPlan.model_validate_json(plan.model_dump_json()) == plan
