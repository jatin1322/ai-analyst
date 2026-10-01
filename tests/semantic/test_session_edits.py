"""Follow-ups are typed edits of the prior plan (ARCHITECTURE 13.13).

The three-turn scenario, on the tiny fixture's Q2 (the fixture has no Q3):

    "What was Q2 opening pipeline?"
    "Break that down by owner."
    "Only show deals above $100k."

Hand-computed at the Q2 opening snapshot, 2025-04-01, open with close in Q2:
    OPP-001 U-101 100000   OPP-003 U-103  75000
    OPP-004 U-101 250000   OPP-008 U-103  90000
    total 515000; by owner U-101 350000, U-103 165000;
    amount >= 100000: OPP-001 and OPP-004, both U-101 -> U-101 350000 only.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from ai_analyst.contracts.plan import (
    AnalysisPattern,
    AnalysisPlan,
    AnalysisSpec,
    AnalysisStance,
    Filter,
    FilterOp,
    FilterValueSource,
    Period,
    PeriodKind,
    SnapshotSelection,
)
from ai_analyst.contracts.result import SnapshotRule
from ai_analyst.contracts.session import (
    AddDimension,
    ChangePeriod,
    ChangeStance,
    EditConflictCode,
    PlanEdit,
    RemoveDimension,
    RemoveFilter,
    SetFilter,
)
from ai_analyst.session.edits import EditConflict, apply_edit
from ai_analyst.session.state import SessionState

Q2 = Period(kind=PeriodKind.FISCAL_QUARTER, label="FY2025-Q2")


def first_plan() -> AnalysisPlan:
    return AnalysisPlan(
        question_restatement="What was Q2 opening pipeline?",
        specs=[
            AnalysisSpec(
                id="opening",
                pattern=AnalysisPattern.POINT_IN_TIME,
                metrics=["opening_pipeline"],
                period=Q2,
                snapshot=SnapshotSelection(rule=SnapshotRule.PERIOD_OPEN),
            )
        ],
    )


def above_100k() -> Filter:
    return Filter(
        column="amount", op=FilterOp.GTE, values=[100000], value_source=FilterValueSource.USER
    )


def run_plan(engine, plan: AnalysisPlan):
    from ai_analyst.semantic.execute import run_plan as execute_plan

    outcome = engine.gate(*plan.specs)
    assert outcome.ok, [r.message for r in outcome.validation.rejections]
    with engine.store.connect() as conn:
        return execute_plan(
            conn, engine.scan, plan, outcome, dataset_id=engine.dataset_id,
            calendar=engine.calendar,
        )[0]


def as_rows(result) -> dict:
    if "owner_id" in result.column_names:
        return {
            result.cell(i, "owner_id"): result.cell(i, "opening_pipeline")
            for i in range(result.row_count)
        }
    return {"total": result.cell(0, "opening_pipeline")}


def test_the_three_turn_scenario_produces_the_hand_computed_answers(tiny):
    session = SessionState(session_id="s", dataset_id="tiny")
    p1 = first_plan()
    session.record(p1)
    assert as_rows(run_plan(tiny, p1)) == {"total": Decimal("515000.00")}

    p2, report2 = session.edit(
        PlanEdit(base_plan_id=p1.plan_id, operations=[AddDimension(dimension="owner")])
    )
    assert as_rows(run_plan(tiny, p2)) == {
        "U-101": Decimal("350000.00"),
        "U-103": Decimal("165000.00"),
    }

    p3, report3 = session.edit(
        PlanEdit(base_plan_id=p2.plan_id, operations=[SetFilter(filter=above_100k())])
    )
    assert as_rows(run_plan(tiny, p3)) == {"U-101": Decimal("350000.00")}

    assert session.lineage(p3.plan_id) == [p1.plan_id, p2.plan_id, p3.plan_id]


def test_the_period_and_snapshot_rule_are_carried_forward():
    p1 = first_plan()
    edited = apply_edit(
        p1, PlanEdit(base_plan_id=p1.plan_id, operations=[AddDimension(dimension="owner")])
    )
    spec = edited.plan.specs[0]
    assert spec.period == Q2
    assert spec.snapshot.rule is SnapshotRule.PERIOD_OPEN
    assert spec.metrics == ["opening_pipeline"]
    assert {"period", "snapshot", "metrics", "stance"} <= set(edited.report.carried)
    assert [c.field for c in edited.report.changed] == ["dimensions"]
    assert edited.report.meaning_changes == ()


def test_a_filter_is_added_and_marked_as_user_supplied():
    p1 = first_plan()
    edited = apply_edit(
        p1, PlanEdit(base_plan_id=p1.plan_id, operations=[SetFilter(filter=above_100k())])
    )
    (added,) = edited.plan.specs[0].filters
    assert added.column == "amount"
    assert added.values == [100000]
    assert added.value_source is FilterValueSource.USER


def test_setting_a_filter_on_the_same_column_replaces_it():
    p1 = first_plan()
    p2 = apply_edit(
        p1, PlanEdit(base_plan_id=p1.plan_id, operations=[SetFilter(filter=above_100k())])
    ).plan
    tighter = Filter(column="amount", op=FilterOp.GTE, values=[200000])
    p3 = apply_edit(
        p2, PlanEdit(base_plan_id=p2.plan_id, operations=[SetFilter(filter=tighter)])
    ).plan
    assert p3.specs[0].filters == [tighter]


def test_plan_lineage_is_retained():
    p1 = first_plan()
    p2 = apply_edit(
        p1, PlanEdit(base_plan_id=p1.plan_id, operations=[AddDimension(dimension="owner")])
    ).plan
    assert p2.parent_plan_id == p1.plan_id
    assert p2.plan_id != p1.plan_id
    # The base plan is untouched.
    assert p1.specs[0].dimensions == []


def test_a_meaning_change_is_surfaced_not_carried():
    p1 = first_plan()
    edited = apply_edit(
        p1,
        PlanEdit(
            base_plan_id=p1.plan_id,
            operations=[
                ChangePeriod(period=Period(kind=PeriodKind.FISCAL_QUARTER, label="FY2025-Q1")),
                ChangeStance(stance=AnalysisStance.RETROSPECTIVE),
            ],
        ),
    )
    assert set(edited.report.meaning_changes) == {"period", "stance"}
    assert "the question's meaning changed" in edited.report.render()


@pytest.mark.parametrize(
    ("operations", "code"),
    [
        ([AddDimension(dimension="owner"), RemoveDimension(dimension="owner")],
         EditConflictCode.CONTRADICTORY),
        ([SetFilter(filter=above_100k()), RemoveFilter(column="amount")],
         EditConflictCode.CONTRADICTORY),
        ([SetFilter(filter=above_100k()),
          SetFilter(filter=Filter(column="amount", op=FilterOp.LT, values=[5]))],
         EditConflictCode.CONTRADICTORY),
        ([ChangeStance(stance=AnalysisStance.RETROSPECTIVE),
          ChangeStance(stance=AnalysisStance.PROSPECTIVE)],
         EditConflictCode.CONTRADICTORY),
        ([RemoveDimension(dimension="owner")], EditConflictCode.NOT_PRESENT),
        ([RemoveFilter(column="amount")], EditConflictCode.NOT_PRESENT),
    ],
)
def test_conflicting_edits_are_rejected(operations, code):
    p1 = first_plan()
    with pytest.raises(EditConflict) as exc:
        apply_edit(p1, PlanEdit(base_plan_id=p1.plan_id, operations=operations))
    assert exc.value.code is code


def test_adding_an_existing_dimension_is_rejected():
    p1 = first_plan()
    p2 = apply_edit(
        p1, PlanEdit(base_plan_id=p1.plan_id, operations=[AddDimension(dimension="owner")])
    ).plan
    with pytest.raises(EditConflict) as exc:
        apply_edit(
            p2, PlanEdit(base_plan_id=p2.plan_id, operations=[AddDimension(dimension="owner")])
        )
    assert exc.value.code is EditConflictCode.DUPLICATE


def test_an_edit_against_the_wrong_plan_is_rejected():
    p1 = first_plan()
    with pytest.raises(EditConflict) as exc:
        apply_edit(p1, PlanEdit(base_plan_id="p_other", operations=[AddDimension(dimension="x")]))
    assert exc.value.code is EditConflictCode.BASE_MISMATCH


def test_an_edit_that_yields_an_invalid_spec_is_rejected():
    p1 = first_plan().model_copy(
        update={"specs": [first_plan().specs[0].model_copy(update={"features": ["owner"]})]}
    )
    with pytest.raises(EditConflict) as exc:
        apply_edit(
            p1, PlanEdit(base_plan_id=p1.plan_id, operations=[AddDimension(dimension="owner")])
        )
    assert exc.value.code is EditConflictCode.INVALID_RESULT


def test_a_carried_filter_is_revalidated_against_the_new_plan(tiny):
    """Carry-forward is not trust-forward: the edited plan goes through the gate."""
    p1 = first_plan()
    bad = Filter(column="no_such_column", op=FilterOp.EQ, values=["x"])
    p2 = apply_edit(p1, PlanEdit(base_plan_id=p1.plan_id, operations=[SetFilter(filter=bad)])).plan
    assert tiny.gate(*p2.specs).validation.rejected


def test_the_report_is_deterministic():
    p1 = first_plan()
    edit = PlanEdit(base_plan_id=p1.plan_id, operations=[AddDimension(dimension="owner")])
    a, b = apply_edit(p1, edit).report, apply_edit(p1, edit).report
    assert (a.carried, a.changed) == (b.carried, b.changed)


def test_a_knowledge_cutoff_can_be_set_with_a_stance_change():
    p1 = first_plan()
    edited = apply_edit(
        p1,
        PlanEdit(
            base_plan_id=p1.plan_id,
            operations=[ChangeStance(stance=AnalysisStance.PROSPECTIVE,
                                     knowledge_cutoff=date(2025, 4, 1))],
        ),
    )
    assert edited.plan.specs[0].knowledge_cutoff == date(2025, 4, 1)


def test_a_session_persists_plans_edits_and_reports_and_round_trips():
    session = SessionState(session_id="s", dataset_id="tiny")
    p1 = first_plan()
    session.record(p1)
    edit = PlanEdit(base_plan_id=p1.plan_id, operations=[AddDimension(dimension="owner")])
    p2, _ = session.edit(edit)

    restored = SessionState.load(session.dump())
    assert restored.edits == [edit]
    assert restored.reports[0].changed[0].field == "dimensions"
    assert restored.plans[p2.plan_id].parent_plan_id == p1.plan_id
    assert restored.lineage(p2.plan_id) == [p1.plan_id, p2.plan_id]
