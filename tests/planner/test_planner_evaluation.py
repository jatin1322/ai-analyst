"""The planner evaluation harness, verified without a model.

Three things are tested here:

1. **The goldens are right.** Every expected plan passes the gate and executes
   on its case's dataset, and the goldens that share a question with the
   acceptance suite normalise to the acceptance suite's own hand-written plans.
2. **The harness is right.** Replaying each case's reference trajectory
   through the real planning loop scores every case as passed.
3. **The taxonomy is right.** Scripted faulty planners, each wrong in one
   known way, are classified into that failure category.

None of this measures planner quality: that needs a real model and is the
opt-in `evals/planner/run_planner_eval.py`.
"""

from __future__ import annotations

from datetime import date

import pytest

from ai_analyst.agent.planner.evaluation import (
    CODE_CATEGORIES,
    EvaluationReport,
    FailureCategory,
    attempt_category,
    normalise_analysis,
)
from ai_analyst.agent.planner.fake import ScriptedModel, action
from ai_analyst.agent.tools import surface
from ai_analyst.contracts.investigation import InvestigationPlan
from ai_analyst.contracts.plan import (
    AnalysisPattern,
    AnalysisStance,
    Attribution,
    Filter,
    FilterOp,
    Period,
    PeriodKind,
)
from ai_analyst.contracts.rejection import RejectionCode
from ai_analyst.contracts.session import AddDimension, PlanEdit, SetFilter
from ai_analyst.contracts.tools import ClarificationReason, ClarificationRequest
from evals.planner.cases import (
    ACCEPTANCE_EQUIVALENTS,
    CASES,
    CASES_BY_ID,
    CLOSE,
    Q1_OPENING,
    Q2_OPENING,
    clarify,
    investigate,
    plan,
    spec,
    submit,
    submit_edit,
    unanswerable,
)
from evals.planner.runner import evaluate, oracle_model, run_case

F = FailureCategory


# ------------------------------------------------------------ goldens are right


def _golden_plans():
    for case in CASES:
        for index, expected in enumerate(case.expect.plans):
            yield pytest.param(case, expected, id=f"{case.id}[{index}]")


@pytest.mark.parametrize(("case", "expected"), list(_golden_plans()))
def test_every_golden_plan_passes_the_gate_and_executes(worlds, case, expected):
    ctx = worlds[case.world].tool_context(stance=case.stance, horizon=case.horizon)
    if isinstance(expected, InvestigationPlan):
        run = surface.run_investigation_plan(ctx, expected)
    else:
        run = surface.run_analysis_plan(ctx, expected)
    assert run.validation.plan_ok, [r.message for r in run.validation.rejections]
    assert run.executed


@pytest.mark.parametrize(("case_id", "acceptance_name"), sorted(ACCEPTANCE_EQUIVALENTS.items()))
def test_goldens_match_the_acceptance_suites_hand_written_plans(worlds, case_id, acceptance_name):
    from ai_analyst.agent.planner.evaluation import normalise_investigation
    from tests.acceptance.test_acceptance import CASES as ACCEPTANCE
    from tests.acceptance.test_acceptance import INVESTIGATION_CASE

    reference = {c.name: c.plan for c in (*ACCEPTANCE, INVESTIGATION_CASE)}[acceptance_name]
    case = CASES_BY_ID[case_id]
    ctx = worlds[case.world].tool_context(stance=case.stance)
    (golden,) = case.expect.plans
    if isinstance(golden, InvestigationPlan):
        assert normalise_investigation(golden, ctx) == normalise_investigation(reference, ctx)
    else:
        assert normalise_analysis(golden, ctx) == normalise_analysis(reference, ctx)


def test_the_golden_set_covers_every_required_category():
    categories = {c.category for c in CASES}
    assert categories >= {"metric", "investigation", "temporal", "ambiguity", "snapshot",
                          "refusal", "follow_up", "injection"}
    assert sum(c.category == "follow_up" for c in CASES) >= 3
    assert len(CASES) >= 30


# ------------------------------------------------------------ harness is right


def test_replaying_every_reference_trajectory_passes(worlds):
    report = evaluate(CASES, worlds, oracle_model)
    failed = [(r.case_id, [f.value for f in r.failures]) for r in report.results if not r.passed]
    assert failed == []
    metrics = report.metrics
    for rate in ("valid_plan_rate", "semantic_match_rate", "correct_refusal_rate",
                 "correct_clarification_rate", "tool_efficiency",
                 "deterministic_execution_success_rate"):
        assert metrics[rate] == 1.0, rate
    assert metrics["cases_with_future_leak_attempts"] == 0
    assert report.attempt_failure_counts == {}


def test_the_normaliser_compares_what_the_gate_resolved(tiny):
    ctx = tiny.tool_context()
    by_alias = plan("x", spec(["opening_pipeline"], dimensions=["owner"]))
    by_concept = plan("x", spec(["opening_pipeline"], dimensions=["owner_id"]))
    as_dates = plan("x", spec(["opening_pipeline"], period=Period(
        kind=PeriodKind.CUSTOM, start=date(2025, 1, 1), end=date(2025, 3, 31), label="Q1")))
    assert normalise_analysis(by_alias, ctx) == normalise_analysis(by_concept, ctx)
    assert normalise_analysis(as_dates, ctx) == normalise_analysis(Q1_OPENING, ctx)
    assert normalise_analysis(Q2_OPENING, ctx) != normalise_analysis(Q1_OPENING, ctx)


def test_every_rejection_code_has_a_failure_category():
    for code in RejectionCode:
        assert attempt_category(code.value) is not None, code
    assert attempt_category("malformed:no_tool_call") is F.MALFORMED_OUTPUT
    assert attempt_category("edit_conflict:duplicate") is F.BAD_EDIT
    assert set(CODE_CATEGORIES.values()) <= set(FailureCategory)


def test_the_report_holds_no_data_values(worlds):
    report = evaluate(CASES[:4], worlds, oracle_model)
    text = report.render()
    for value in ("515000", "515,000", "200000", "Commit", "OPP-0"):
        assert value not in text


# ------------------------------------------------------------ taxonomy is right


def faulty(case_id: str, actions, worlds):
    case = CASES_BY_ID[case_id]
    _, scored = run_case(case, worlds[case.world], ScriptedModel(actions))
    return scored


LEAK = plan("x", spec(["opening_pipeline"], dimensions=["stage"],
                      attribution=Attribution.AT_CLOSE))
Q2_CLOSE_OPENING = plan("x", spec(["opening_pipeline"], period=Q2_OPENING.specs[0].period,
                                  snapshot=CLOSE))


@pytest.mark.parametrize(
    ("case_id", "actions", "expected"),
    [
        ("metric_opening_q2",
         [submit(plan("x", spec(["ending_pipeline"], period=Q2_OPENING.specs[0].period)))],
         F.WRONG_METRIC),
        ("metric_opening_q2", [submit(Q2_CLOSE_OPENING)], F.WRONG_SNAPSHOT),
        ("metric_opening_q2", [submit(Q1_OPENING)], F.WRONG_TIME_WINDOW),
        ("metric_opening_q1",
         [submit(plan("x", spec(["opening_pipeline"], knowledge_cutoff=date(2025, 2, 1))))],
         F.WRONG_STANCE),
        ("snapshot_comparison", [submit(Q2_OPENING)], F.WRONG_COMPARISON),
        ("metric_ranked_owners_q2",
         [submit(plan("x", spec(["opening_pipeline"], period=Q2_OPENING.specs[0].period,
                                dimensions=["owner_id"])))],
         F.WRONG_ANALYSIS_TYPE),
        ("metric_opening_q1", [action(None), action(None)], F.MALFORMED_OUTPUT),
        ("metric_opening_q1", [action("list_available_metrics", {})] * 4 + [submit(Q1_OPENING)],
         F.EXCESSIVE_TOOL_USE),
        ("metric_opening_q1",
         [clarify(ClarificationRequest(reason=ClarificationReason.MISSING_PERIOD,
                                       question="Which quarter?"))],
         F.UNNECESSARY_CLARIFICATION),
        ("metric_opening_q1", [unanswerable("unsupported_analysis")], F.UNNECESSARY_REFUSAL),
        ("ambiguity_missing_period",
         [submit(plan("x", spec(["win_rate"], pattern=AnalysisPattern.RATE,
                                snapshot=CLOSE)))],
         F.MISSING_CLARIFICATION),
        ("refusal_forecast", [submit(Q2_OPENING)], F.MISSING_REFUSAL),
        ("temporal_future_attribution", [submit(LEAK)] * 3, F.FUTURE_LEAK_ATTEMPT),
        ("refusal_out_of_coverage", [unanswerable("unsupported_analysis")], F.WRONG_REASON),
        ("ambiguity_unknown_concept",
         [submit(plan("x", spec(["opening_pipeline"], dimensions=["industry"])))] * 3,
         F.INVENTED_BINDING),
        ("ambiguity_grant_scope",
         [submit(plan("x", spec(["opening_pipeline"], dimensions=["amount"])))] * 3,
         F.SCOPE_BYPASS),
        ("metric_opening_q1",
         [submit(plan("x", spec(["win_rate"])))] * 3,
         F.UNSUPPORTED_ANALYSIS),
        ("followup_break_down_by_owner",
         [submit_edit(PlanEdit(base_plan_id="p_wrong",
                               operations=[AddDimension(dimension="owner_id")]))] * 3,
         F.BAD_EDIT),
        ("followup_filter_over_100k",
         [submit_edit(PlanEdit(base_plan_id="p_followup_owner", operations=[SetFilter(
             filter=Filter(column="amount", op=FilterOp.GTE, values=[50000]))]))] * 3,
         F.INVENTED_LITERAL),
        ("followup_filter_over_100k",
         [submit_edit(PlanEdit(base_plan_id="p_followup_owner", operations=[SetFilter(
             filter=Filter(column="amount", op=FilterOp.GT, values=[100000]))]))],
         F.WRONG_FILTER),
    ],
    ids=lambda x: x.value if isinstance(x, FailureCategory) else None,
)
def test_a_faulty_planner_is_classified(worlds, case_id, actions, expected):
    scored = faulty(case_id, actions, worlds)
    assert not scored.passed
    assert expected in scored.failures or expected in scored.attempt_failures, (
        scored.failures, scored.attempt_failures,
    )


def test_an_investigation_bypass_is_an_attempt_failure_even_when_repaired(worlds):
    from ai_analyst.contracts.concepts import BusinessConcept as C
    from ai_analyst.contracts.investigation import Operation, StatisticalOperation, Variable
    from evals.planner.cases import _investigation

    bypass = _investigation(
        "opening pipeline, the long way round", [Variable(id="amt", concept=C.AMOUNT)], [],
        Operation(kind=StatisticalOperation.SUM, measure="amt"),
        stance=AnalysisStance.PROSPECTIVE, window_end=None,
    )
    scored = faulty("metric_opening_q1", [investigate(bypass), submit(Q1_OPENING)], worlds)
    assert scored.passed  # the final plan is right...
    assert F.INVESTIGATION_BYPASS in scored.attempt_failures  # ...but the attempt is recorded


def test_an_investigation_where_a_metric_was_expected_is_a_bypass(worlds):
    case = CASES_BY_ID["investigation_forecast_changes"]
    # A retrospective case whose answer is a registry metric, answered with a
    # (gate-valid) investigation instead.
    scored = faulty("temporal_retrospective_attribution", [investigate(case.expect.plans[0])],
                    worlds)
    assert F.INVESTIGATION_BYPASS in scored.failures


def test_a_loop_imposed_rejection_is_never_a_correct_refusal(worlds):
    scored = faulty("refusal_forecast", [action(None), action(None)], worlds)
    assert not scored.passed
    assert F.MALFORMED_OUTPUT in scored.failures


def test_the_report_counts_by_category_not_one_score(worlds):
    results = [
        faulty("metric_opening_q2", [submit(Q1_OPENING)], worlds),
        faulty("refusal_forecast", [submit(Q2_OPENING)], worlds),
        faulty("temporal_future_attribution", [submit(LEAK)] * 3, worlds),
    ]
    report = EvaluationReport(results)
    assert report.failure_counts == {
        "future_leak_attempt": 1, "missing_refusal": 1, "wrong_time_window": 1,
    }
    assert report.metrics["cases_with_future_leak_attempts"] == 1
    assert report.metrics["semantic_match_rate"] == 0.0
    assert "score" not in report.metrics
