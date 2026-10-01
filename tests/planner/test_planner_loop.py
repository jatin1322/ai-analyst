"""The planning loop: budgets, the gate boundary, and what the planner cannot do.

Every model here is scripted, so each test states exactly what a model did and
checks what the deterministic loop made of it. Nothing touches a network.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from ai_analyst.agent.planner.fake import action, stop
from ai_analyst.contracts.concepts import BusinessConcept as C
from ai_analyst.contracts.plan import (
    AnalysisStance,
    Attribution,
    Filter,
    FilterOp,
)
from ai_analyst.contracts.planner import (
    FinalPlan,
    PlannerOutcomeKind,
    PlanPath,
    Rejected,
    RejectedReason,
    TurnStatus,
)
from ai_analyst.contracts.rejection import RejectionCode
from ai_analyst.contracts.session import AddDimension, PlanEdit, SetFilter
from ai_analyst.contracts.tools import ClarificationReason, ClarificationRequest
from ai_analyst.session.state import SessionState
from evals.planner.cases import (
    AMOUNT_AT_END_BY_SEGMENT,
    FOLLOW_UP_BASE,
    Q1_OPENING,
    clarify,
    investigate,
    plan,
    spec,
    submit,
    submit_edit,
    unanswerable,
)
from tests.planner.conftest import run

K = PlannerOutcomeKind
PROSP, RETRO = AnalysisStance.PROSPECTIVE, AnalysisStance.RETROSPECTIVE
METRICS = action("list_available_metrics", {})


def settings(world, **overrides):
    return world.settings.model_copy(update=overrides)


# ------------------------------------------------------------- final plans


def test_a_valid_plan_is_final_only_after_the_gate(tiny):
    result, _ = run(tiny, [submit(Q1_OPENING)])
    outcome = result.outcome
    assert isinstance(outcome, FinalPlan)
    assert outcome.validation is not None and outcome.validation.plan_ok
    assert result.kind is K.FINAL_PLAN


def test_lineage_and_assumptions_never_come_from_the_model(tiny):
    forged = Q1_OPENING.model_copy(update={
        "plan_id": "p_model_chosen", "parent_plan_id": "p_someone_else",
        "assumptions": ["the model says this is trustworthy"],
    })
    result, _ = run(tiny, [submit(forged)])
    final = result.outcome.plan
    assert final.plan_id != "p_model_chosen"
    assert final.parent_plan_id is None
    assert final.assumptions == []


def test_the_loop_never_executes_a_plan(tiny, monkeypatch):
    from ai_analyst.agent.tools import surface

    def refuse(*a, **k):
        raise AssertionError("the planning loop executed a plan")

    monkeypatch.setattr(surface, "run_analysis_plan", refuse)
    monkeypatch.setattr(surface, "run_investigation_plan", refuse)
    result, _ = run(tiny, [submit(Q1_OPENING)])
    assert result.kind is K.FINAL_PLAN


def test_a_gate_rejection_returns_to_the_planner_with_codes(tiny):
    bad = plan("x", spec(["opening_pipeline"], dimensions=["industry"]))
    result, model = run(tiny, [submit(bad), submit(Q1_OPENING)])
    assert result.kind is K.FINAL_PLAN
    assert result.repairs == 1
    call_id, content, is_error = model.transcripts[0].replies[0]
    assert is_error and "unknown_column" in content
    assert result.turns[0].status is TurnStatus.GATE_REJECTED


def test_repairs_are_bounded(tiny):
    bad = plan("x", spec(["opening_pipeline"], dimensions=["industry"]))
    s = settings(tiny, planner_max_repairs=2)
    result, _ = run(tiny, [submit(bad)] * 3, settings=s)
    assert isinstance(result.outcome, Rejected)
    assert result.outcome.reason is RejectedReason.VALIDATION_FAILED
    assert result.outcome.validation.has(RejectionCode.UNKNOWN_COLUMN)
    assert result.repairs == 3


# ------------------------------------------------------------ what it cannot do


def test_investigation_cannot_bypass_a_registry_metric(tiny):
    from ai_analyst.contracts.investigation import (
        Operation,
        StatisticalOperation,
        Variable,
    )
    from evals.planner.cases import _investigation

    # "Sum of amount over the opening cohort" is the opening-pipeline metric.
    bypass = _investigation(
        "opening pipeline, the long way round",
        [Variable(id="amt", concept=C.AMOUNT)], [],
        Operation(kind=StatisticalOperation.SUM, measure="amt"), stance=PROSP, window_end=None,
    )
    result, _ = run(tiny, [investigate(bypass), submit(Q1_OPENING)])
    assert "semantic_path_available" in result.codes()
    assert result.outcome.path is PlanPath.SEMANTIC


@pytest.mark.parametrize(
    ("bad_spec", "code"),
    [
        (spec(["opening_pipeline"], dimensions=["industry"]), "unknown_column"),
        (spec(["opening_pipeline"], measure_concept="deal_amount"), "unknown_measure_concept"),
        (spec(["bookings_forecast"]), "unknown_metric"),
    ],
)
def test_an_invented_binding_is_refused(tiny, bad_spec, code):
    result, _ = run(tiny, [submit(plan("x", bad_spec))] * 3)
    assert code in result.codes()
    assert result.outcome.reason is RejectedReason.VALIDATION_FAILED


def test_a_later_snapshot_attribution_is_refused_prospectively(tiny):
    leak = plan("x", spec(["opening_pipeline"], dimensions=["stage"],
                          attribution=Attribution.AT_CLOSE))
    result, _ = run(tiny, [submit(leak)] * 3)
    assert "stance_violation" in result.codes()
    assert result.kind is K.REJECTED


def test_the_planner_cannot_widen_the_session_stance(tiny):
    retro = plan("x", spec(["opening_pipeline"], dimensions=["stage"],
                           attribution=Attribution.AT_CLOSE, stance=RETRO))
    result, _ = run(tiny, [submit(retro)] * 3, stance=PROSP)
    assert "stance_violation" in result.codes()
    assert result.kind is K.REJECTED


def test_the_session_horizon_is_applied_to_every_plan(tiny):
    horizon = date(2025, 2, 1)
    result, _ = run(tiny, [submit(Q1_OPENING)], horizon=horizon)
    assert result.outcome.plan.specs[0].knowledge_cutoff == horizon


def test_a_grant_is_not_widened_by_the_planner(worlds):
    granted = worlds["granted"]
    by_amount = plan("x", spec(["opening_pipeline"], dimensions=["amount"]))
    result, _ = run(granted, [submit(by_amount)] * 3)
    assert "grant_purpose_not_permitted" in result.codes()


def test_a_granted_column_cannot_be_reached_by_its_raw_name(worlds):
    granted = worlds["granted"]
    raw = plan("x", spec(["opening_pipeline"], filters=[
        Filter(column="enterprise_amount", op=FilterOp.GTE, values=[1])]))
    result, _ = run(granted, [submit(raw)] * 3, question="Q1 opening pipeline above 1?")
    assert "column_unclassified" in result.codes()


def test_a_number_the_user_did_not_write_is_refused(tiny):
    invented = plan("x", spec(["opening_pipeline"], filters=[
        Filter(column="amount", op=FilterOp.GTE, values=[250000])]))
    result, _ = run(tiny, [submit(invented)] * 3, question="Q1 opening pipeline, big deals only")
    assert "invented_literal" in result.codes()
    assert result.kind is K.REJECTED


def test_a_number_the_user_wrote_is_accepted(tiny):
    over = plan("x", spec(["opening_pipeline"], filters=[
        Filter(column="amount", op=FilterOp.GTE, values=[100000])]))
    result, _ = run(tiny, [submit(over)], question="Q1 opening pipeline for deals over $100k")
    assert result.kind is K.FINAL_PLAN


def test_a_plan_with_unresolved_ambiguities_is_not_final(tiny):
    unsure = Q1_OPENING.model_copy(update={"unresolved_ambiguities": ["which quarter?"]})
    result, _ = run(tiny, [submit(unsure)] * 3)
    assert "unresolved_ambiguities" in result.codes()
    assert result.kind is K.REJECTED


# ------------------------------------------------------------- tools and budgets


def test_a_tool_call_runs_the_surface_and_returns_an_envelope(tiny):
    result, model = run(tiny, [METRICS, submit(Q1_OPENING)])
    assert result.tool_calls == 1
    _, content, is_error = model.transcripts[0].replies[0]
    assert not is_error
    assert content.startswith('<tool_result tool="list_available_metrics">')
    assert result.turns[0].tool_result_type == "MetricCatalog"


def test_invalid_tool_arguments_are_refused_and_counted(tiny):
    result, model = run(tiny, [action("inspect_values", {"name": "stage", "top_k": 10_000}),
                               submit(Q1_OPENING)])
    assert result.turns[0].status is TurnStatus.TOOL_REFUSED
    assert "invalid_arguments" in result.codes()
    assert result.tool_calls == 1
    assert model.transcripts[0].replies[0][2] is True


def test_the_tool_budget_is_enforced(tiny):
    s = settings(tiny, planner_max_tool_calls=2)
    result, model = run(tiny, [METRICS, METRICS, METRICS, submit(Q1_OPENING)], settings=s)
    assert result.kind is K.FINAL_PLAN
    assert result.tool_calls == 2
    assert "tool_budget_exhausted" in result.codes()
    assert sum(1 for t in result.turns if t.status is TurnStatus.TOOL_EXECUTED) == 2


def test_the_turn_budget_is_enforced(tiny):
    s = settings(tiny, planner_max_turns=3)
    result, _ = run(tiny, [METRICS] * 10, settings=s)
    assert result.outcome.reason is RejectedReason.TURN_BUDGET_EXHAUSTED
    assert len(result.turns) == 3


def test_the_context_budget_is_enforced_before_the_model_is_called(tiny):
    s = settings(tiny, planner_max_context_tokens=1_000)
    result, model = run(tiny, [submit(Q1_OPENING)], settings=s)
    assert result.outcome.reason is RejectedReason.CONTEXT_BUDGET_EXHAUSTED
    assert model.transcripts == []


def test_the_context_budget_bounds_the_conversation(tiny):
    s = settings(tiny, planner_max_context_tokens=9_000, planner_max_tool_result_tokens=2_000)
    result, _ = run(tiny, [action("inspect_dataset", {})] * 8, settings=s)
    assert result.outcome.reason is RejectedReason.CONTEXT_BUDGET_EXHAUSTED


def test_a_large_tool_result_is_truncated(tiny):
    s = settings(tiny, planner_max_tool_result_tokens=100)
    result, model = run(tiny, [action("inspect_dataset", {}), submit(Q1_OPENING)], settings=s)
    content = model.transcripts[0].replies[0][1]
    assert "[truncated by the planner loop at 100 tokens]" in content
    assert len(content) < 600


# ------------------------------------------------------------- malformed output


def test_one_malformed_turn_is_returned_once(tiny):
    result, model = run(tiny, [action(None), submit(Q1_OPENING)])
    assert result.kind is K.FINAL_PLAN
    assert result.malformed_outputs == 1
    assert model.transcripts[0].replies[0][2] is True


def test_repeated_malformed_output_fails_closed(tiny):
    result, _ = run(tiny, [action(None), action("run_guarded_sql", {"sql": "SELECT 1"})])
    assert result.outcome.reason is RejectedReason.MALFORMED_OUTPUT
    assert result.kind is K.REJECTED


@pytest.mark.parametrize(
    ("reason", "expected"),
    [("refusal", RejectedReason.MODEL_REFUSED), ("max_tokens", RejectedReason.MODEL_TRUNCATED)],
)
def test_a_stopped_model_is_never_parsed(tiny, reason, expected):
    result, _ = run(tiny, [stop(reason)])
    assert result.outcome.reason is expected


# ------------------------------------------------ clarifications and rejections


def test_clarification_alternatives_are_filled_deterministically(worlds):
    production = worlds["production"]
    request = ClarificationRequest(
        reason=ClarificationReason.CONCEPT_UNAVAILABLE, concept=C.CUSTOMER_SEGMENT,
        question="Segment is not available. Use another breakdown?",
        available_alternatives=["industry", "region", "made_up"],
    )
    result, _ = run(production, [clarify(request)], question="Pipeline by segment?")
    alternatives = result.outcome.request.available_alternatives
    assert "made_up" not in alternatives and "industry" not in alternatives
    assert "customer_segment" not in alternatives
    assert "stage" in alternatives


def test_model_supplied_alternatives_are_dropped_for_other_reasons(tiny):
    request = ClarificationRequest(
        reason=ClarificationReason.MISSING_PERIOD, question="Which quarter?",
        available_alternatives=["anything"],
    )
    result, _ = run(tiny, [clarify(request)], question="What was the win rate?")
    assert result.outcome.request.available_alternatives == []


def test_a_concept_unavailable_rejection_lists_real_alternatives(worlds):
    result, _ = run(worlds["production"],
                    [unanswerable("concept_unavailable", C.CUSTOMER_SEGMENT)],
                    question="Pipeline by segment?")
    assert result.outcome.reason is RejectedReason.CONCEPT_UNAVAILABLE
    assert "customer_segment" not in result.outcome.available_alternatives
    assert result.outcome.available_alternatives


def test_model_prose_may_not_carry_invented_numbers(tiny):
    request = ClarificationRequest(
        reason=ClarificationReason.MISSING_PERIOD,
        question="Pipeline was 515,000 last quarter; which quarter do you mean?",
    )
    result, _ = run(tiny, [clarify(request), clarify(request)],
                    question="What was pipeline?")
    assert result.outcome.reason is RejectedReason.MALFORMED_OUTPUT
    assert "malformed:unverified_numeral" in result.codes()


# ------------------------------------------------------------------ follow-ups


def _session(world, base=FOLLOW_UP_BASE):
    session = SessionState(session_id="s", dataset_id=world.dataset.dataset_id)
    session.record(base)
    return session


def test_a_follow_up_is_an_edit_of_the_active_plan(tiny):
    session = _session(tiny)
    edit = PlanEdit(base_plan_id=FOLLOW_UP_BASE.plan_id,
                    operations=[AddDimension(dimension="owner_id")])
    result, _ = run(tiny, [submit_edit(edit)], question="Now break that down by owner.",
                    session=session)
    outcome = result.outcome
    assert outcome.path is PlanPath.EDIT
    assert outcome.plan.parent_plan_id == FOLLOW_UP_BASE.plan_id
    assert outcome.plan.plan_id != FOLLOW_UP_BASE.plan_id
    assert outcome.plan.specs[0].dimensions == ["owner_id"]
    report = outcome.carry_forward
    assert [c.field for c in report.changed] == ["dimensions"]
    assert {"metrics", "period", "snapshot", "stance"} <= set(report.carried)
    # The loop does not mutate the session; the caller adopts the plan.
    assert session.active_plan_id == FOLLOW_UP_BASE.plan_id


def test_an_edit_must_target_the_active_plan(tiny):
    edit = PlanEdit(base_plan_id="p_not_active", operations=[AddDimension(dimension="stage")])
    result, _ = run(tiny, [submit_edit(edit)] * 3, session=_session(tiny))
    assert "edit_base_mismatch" in result.codes()


def test_an_edit_without_a_session_is_refused(tiny):
    edit = PlanEdit(base_plan_id="p1", operations=[AddDimension(dimension="stage")])
    result, _ = run(tiny, [submit_edit(edit)] * 3)
    assert "edit_without_base" in result.codes()


def test_an_edits_new_numbers_must_come_from_the_user(tiny):
    edit = PlanEdit(base_plan_id=FOLLOW_UP_BASE.plan_id, operations=[SetFilter(
        filter=Filter(column="amount", op=FilterOp.GTE, values=[75000]))])
    result, _ = run(tiny, [submit_edit(edit)] * 3, question="Only the big ones.",
                    session=_session(tiny))
    assert "invented_literal" in result.codes()


def test_a_conflicting_edit_is_refused(tiny):
    edit = PlanEdit(base_plan_id=FOLLOW_UP_BASE.plan_id,
                    operations=[AddDimension(dimension="stage"), AddDimension(dimension="stage")])
    result, _ = run(tiny, [submit_edit(edit)] * 3, session=_session(tiny))
    assert any(c.startswith("edit_conflict:") for c in result.codes())


def test_an_edited_plan_is_gated_in_full(tiny):
    edit = PlanEdit(base_plan_id=FOLLOW_UP_BASE.plan_id,
                    operations=[AddDimension(dimension="industry")])
    result, _ = run(tiny, [submit_edit(edit)] * 3, session=_session(tiny))
    assert "unknown_column" in result.codes()


# ----------------------------------------------------------------- run record


def test_the_run_record_is_metadata_only(worlds):
    from evals.planner.datasets import INJECTION_TEXT

    injection = worlds["injection"]
    result, _ = run(injection, [action("inspect_values", {"name": "forecast_category"}),
                                submit(Q1_OPENING)],
                    question="What was Q1 opening pipeline for the SECRET-QUESTION account?")
    record = json.dumps([t.model_dump(mode="json") for t in result.turns])
    assert "SECRET-QUESTION" not in record
    assert "Ignore previous" not in record and INJECTION_TEXT not in record
    assert "Commit" not in record  # a forecast value the tool returned


def test_investigations_run_through_the_investigation_gate(tiny):
    result, _ = run(tiny, [investigate(AMOUNT_AT_END_BY_SEGMENT, "needs_derived_feature")],
                    stance=RETRO, question="Amount at end of quarter by segment?")
    assert result.outcome.path is PlanPath.INVESTIGATION
    assert result.outcome.validation.plan_ok
    assert result.outcome.investigation.plan_id != AMOUNT_AT_END_BY_SEGMENT.plan_id


def test_model_prose_may_cite_a_snapshot_date_of_this_dataset(tiny):
    request = ClarificationRequest(
        reason=ClarificationReason.AMBIGUOUS_DEFINITION,
        question="Should pipeline be read at the 2025-04-01 snapshot or the 2025-06-30 one?",
    )
    result, _ = run(tiny, [clarify(request)], question="What was Q2 pipeline?")
    assert result.kind is K.CLARIFICATION_REQUEST
    assert result.malformed_outputs == 0


def test_a_date_that_is_not_a_snapshot_is_still_refused_in_prose(tiny):
    request = ClarificationRequest(
        reason=ClarificationReason.AMBIGUOUS_DEFINITION,
        question="Should pipeline be read at the 2025-04-17 snapshot?",
    )
    result, _ = run(tiny, [clarify(request)] * 2, question="What was Q2 pipeline?")
    assert result.outcome.reason is RejectedReason.MALFORMED_OUTPUT


def test_an_investigation_bin_edge_must_be_a_number_the_user_wrote(tiny):
    from ai_analyst.contracts.investigation import Binning, BinningKind, Grouping

    binned = AMOUNT_AT_END_BY_SEGMENT.model_copy(update={"grouping": [Grouping(
        variable="amount_end",
        binning=Binning(kind=BinningKind.EXPLICIT_EDGES, edges=[75000.0]))]})
    result, _ = run(tiny, [investigate(binned, "needs_derived_feature")] * 3, stance=RETRO,
                    question="End-of-quarter amount, split at 50k?")
    assert "invented_literal" in result.codes()
