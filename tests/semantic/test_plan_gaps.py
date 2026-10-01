"""Smaller gaps closed with the 13.7 / 13.12 milestone.

* `cohort_fate`: the registry metric that makes the COHORT_TRACE pattern
  reachable through a plan. The pattern already compiled and the gate already
  refused it prospectively, but no metric declared it, so every trace plan was
  rejected as a pattern mismatch.
* The creation basis: 'created in period' never silently falls back from a
  creation date to first appearance.
* `SetAnalysisOption`: a typed edit for the documented 5.3 options.

Cohort values hand-computed from `tests/fixtures/tiny/snapshots.csv`. The Q1
cohort is every opportunity open at 2025-01-01, with its amount at that date:

    OPP-001 100000  still open at 03-31 (Negotiation)
    OPP-002  50000  Closed Won at 03-31                 -> won
    OPP-003  75000  still open at 03-31 (Discovery)
    OPP-004 200000  still open at 03-31 (250000 then, but the cohort's own
                    amount is read when the cohort was fixed)
    OPP-005  40000  still open at 03-31 (Discovery)
    OPP-007  60000  Closed Lost at 03-31                -> lost

    won 1 / 50000, lost 1 / 60000, open 4 / 100000 + 75000 + 200000 + 40000 = 415000
    total 6 / 525000
"""

from __future__ import annotations

import csv
from decimal import Decimal

import pytest

from ai_analyst.contracts.concepts import BusinessConcept as Concept
from ai_analyst.contracts.plan import (
    AnalysisPattern,
    AnalysisPlan,
    AnalysisSpec,
    AnalysisStance,
    CreationBasis,
    Period,
    PeriodKind,
    SnapshotSelection,
    WinRateBasis,
)
from ai_analyst.contracts.rejection import RejectionCode
from ai_analyst.contracts.result import SnapshotRule, TrustTier
from ai_analyst.contracts.session import EditConflictCode, PlanEdit, SetAnalysisOption
from ai_analyst.contracts.tenant import TenantProfile
from ai_analyst.semantic.metrics import METRICS
from ai_analyst.session.edits import EditConflict, apply_edit
from tests.semantic.conftest import TINY_CSV, TINY_TENANT, build_engine

Q1 = Period(kind=PeriodKind.FISCAL_QUARTER, label="FY2025-Q1")
Q2 = Period(kind=PeriodKind.FISCAL_QUARTER, label="FY2025-Q2")


def trace(**kwargs) -> AnalysisSpec:
    kwargs.setdefault("stance", AnalysisStance.RETROSPECTIVE)
    return AnalysisSpec(
        id="c",
        pattern=AnalysisPattern.COHORT_TRACE,
        metrics=["cohort_fate"],
        period=Q1,
        snapshot=SnapshotSelection(rule=SnapshotRule.PERIOD_OPEN),
        **kwargs,
    )


# ============================================================================
# cohort_fate
# ============================================================================


def test_a_cohort_trace_is_reachable_through_a_plan(tiny):
    result = tiny.one(trace())
    fates = {
        result.cell(i, "terminal_state"): (
            result.cell(i, "opportunity_count"),
            result.cell(i, "cohort_amount"),
        )
        for i in range(result.row_count)
    }
    assert fates == {
        "won": (1, Decimal("50000.00")),
        "lost": (1, Decimal("60000.00")),
        "open": (4, Decimal("415000.00")),
    }


def test_the_fates_partition_the_cohort_exactly(tiny):
    result = tiny.one(trace())
    counts = [result.cell(i, "opportunity_count") for i in range(result.row_count)]
    amounts = [result.cell(i, "cohort_amount") for i in range(result.row_count)]
    assert sum(counts) == 6
    assert sum(amounts) == Decimal("525000.00")


def test_a_cohort_trace_is_refused_prospectively_twice_over(tiny):
    outcome = tiny.gate(trace(stance=AnalysisStance.PROSPECTIVE))
    assert outcome.validation.has(RejectionCode.METRIC_STANCE_INCOMPATIBLE)
    assert outcome.validation.has(RejectionCode.STANCE_VIOLATION)


def test_a_cohort_trace_reports_every_snapshot_it_reads(tiny):
    # The trace fixes the cohort at 01-01 and follows it to 03-31; reporting
    # only the cohort snapshot would hide the window the answer depends on.
    result = tiny.one(trace())
    reported = [(s.rule, s.resolved_as_of.isoformat()) for s in result.resolved_snapshots]
    assert reported == [
        (SnapshotRule.PERIOD_OPEN, "2025-01-01"),
        (SnapshotRule.PERIOD_CLOSE, "2025-03-31"),
    ]
    # Status on tiny comes from stage keywords, which caps the tier at B.
    assert result.trust_tier is TrustTier.B


def test_an_attributed_read_reports_its_snapshot(tiny):
    from ai_analyst.contracts.plan import Attribution

    result = tiny.one(AnalysisSpec(
        id="a", pattern=AnalysisPattern.POINT_IN_TIME, metrics=["opening_pipeline"],
        period=Q1, snapshot=SnapshotSelection(rule=SnapshotRule.PERIOD_OPEN),
        dimensions=["stage"], attribution=Attribution.AT_CLOSE,
        stance=AnalysisStance.RETROSPECTIVE,
    ))
    assert [s.resolved_as_of.isoformat() for s in result.resolved_snapshots] == [
        "2025-01-01", "2025-03-31",
    ]


def test_cohort_fate_serves_only_the_trace_pattern():
    definition = METRICS["cohort_fate"]
    assert definition.patterns == (AnalysisPattern.COHORT_TRACE,)
    assert definition.permitted_stances == (AnalysisStance.RETROSPECTIVE,)


# ============================================================================
# Creation basis: never a silent fallback
# ============================================================================


def _bridge(**kwargs) -> AnalysisSpec:
    return AnalysisSpec(
        id="b",
        pattern=AnalysisPattern.POINT_IN_TIME,
        metrics=["created_pipeline"],
        period=Q2,
        snapshot=SnapshotSelection(rule=SnapshotRule.PERIOD_CLOSE),
        **kwargs,
    )


@pytest.fixture
def no_created(tmp_path):
    """The tiny data with its creation-date column removed entirely."""
    reader = list(csv.DictReader(TINY_CSV.open(encoding="utf-8")))
    for row in reader:
        del row["created"]
    path = tmp_path / "no_created.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(reader[0]))
        writer.writeheader()
        writer.writerows(reader)
    columns = {
        k: v for k, v in TINY_TENANT.concept_columns.items() if k is not Concept.CREATED_DATE
    }
    tenant = TenantProfile(tenant_id="tiny", source="t", concept_columns=columns,
                           fiscal_year_start_month=1)
    return build_engine(path, "no_created", tmp_path, tenant=tenant)


def test_a_creation_date_basis_needs_the_created_date_concept(no_created):
    outcome = no_created.gate(_bridge())
    (rejection,) = [r for r in outcome.validation.rejections if r.field == "creation_basis"]
    assert rejection.code is RejectionCode.CONCEPT_UNAVAILABLE
    assert rejection.concept is Concept.CREATED_DATE
    assert "first_seen" in rejection.remedy


def test_first_appearance_is_used_only_when_selected(no_created):
    assert no_created.gate(_bridge(creation_basis=CreationBasis.FIRST_SEEN)).ok


def test_a_declared_creation_date_is_used_as_the_default(tiny):
    assert tiny.gate(_bridge()).ok


# ============================================================================
# SetAnalysisOption
# ============================================================================


def _plan() -> AnalysisPlan:
    return AnalysisPlan(
        question_restatement="q",
        specs=[AnalysisSpec(id="r", pattern=AnalysisPattern.RATE, metrics=["win_rate"],
                            period=Q1,
                            snapshot=SnapshotSelection(rule=SnapshotRule.PERIOD_CLOSE))],
    )


def test_an_analysis_option_is_set_by_a_typed_edit():
    plan = _plan()
    edited = apply_edit(
        plan,
        PlanEdit(base_plan_id=plan.plan_id,
                 operations=[SetAnalysisOption(option="win_rate_basis", value="all_cohort")]),
    )
    assert edited.plan.specs[0].win_rate_basis is WinRateBasis.ALL_COHORT
    assert [c.field for c in edited.report.changed] == ["win_rate_basis"]
    assert edited.plan.parent_plan_id == plan.plan_id


def test_an_invalid_option_value_is_an_edit_conflict():
    plan = _plan()
    with pytest.raises(EditConflict) as exc:
        apply_edit(plan, PlanEdit(base_plan_id=plan.plan_id,
                                  operations=[SetAnalysisOption(option="win_rate_basis",
                                                                value="most_of_them")]))
    assert exc.value.code is EditConflictCode.INVALID_RESULT


def test_the_same_option_twice_in_one_edit_is_contradictory():
    plan = _plan()
    with pytest.raises(EditConflict) as exc:
        apply_edit(plan, PlanEdit(base_plan_id=plan.plan_id, operations=[
            SetAnalysisOption(option="win_rate_basis", value="all_cohort"),
            SetAnalysisOption(option="win_rate_basis", value="closed_only"),
        ]))
    assert exc.value.code is EditConflictCode.CONTRADICTORY


def test_only_documented_options_can_be_set():
    with pytest.raises(ValueError):
        SetAnalysisOption(option="stance", value="retrospective")


def test_mutation_without_the_gate_check_the_bridge_still_refuses(no_created, monkeypatch):
    """Disable the gate's creation-basis check: the bridge compiler refuses rather
    than silently counting first appearance as creation."""
    import dataclasses

    from ai_analyst.semantic.compiler import compile_spec
    from ai_analyst.semantic.sql import CompilationError

    valid = no_created.gate(_bridge(creation_basis=CreationBasis.FIRST_SEEN)).specs["b"]
    forced = dataclasses.replace(
        valid, spec=valid.spec.model_copy(update={"creation_basis": CreationBasis.CREATED_DATE})
    )
    with pytest.raises(CompilationError, match="select 'first_seen' explicitly"):
        compile_spec(no_created.scan, forced)
