"""Planner output parsing: strict, typed, fail-closed.

Every model turn becomes exactly one `PlannerOutcome` or a `MalformedOutput`.
Nothing is repaired: a turn without a tool call, with several, with an unknown
tool, or with terminal arguments that do not validate is malformed.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ai_analyst.agent.context import PlannerContext
from ai_analyst.agent.planner.fake import ScriptedModel, action
from ai_analyst.agent.planner.planner import LLMPlanner, MalformedOutput
from ai_analyst.agent.planner.tools import TOOLS, tool_specs
from ai_analyst.contracts.concepts import BusinessConcept as C
from ai_analyst.contracts.planner import (
    ClarificationOutcome,
    FinalPlan,
    PlannerOutcomeKind,
    PlanningResult,
    PlanPath,
    Rejected,
    RejectedReason,
    ToolRequest,
)
from ai_analyst.contracts.session import AddDimension, PlanEdit
from evals.planner.cases import CASES_BY_ID, Q1_OPENING, submit

CONTEXT = PlannerContext(card="card", catalog="metrics", temporal="stance", session="",
                         budget=2000, index_degraded=False)


def parse(act):
    return LLMPlanner(ScriptedModel([act])).plan("q", CONTEXT)


# ------------------------------------------------------------- each outcome


def test_an_inspection_call_is_a_tool_request():
    outcome = parse(action("inspect_concept", {"concept": "amount"}))
    assert isinstance(outcome, ToolRequest)
    assert outcome.kind is PlannerOutcomeKind.TOOL_REQUEST
    assert outcome.tool == "inspect_concept"
    assert outcome.arguments == {"concept": "amount"}


def test_a_submitted_plan_is_a_final_plan_candidate():
    outcome = parse(submit(Q1_OPENING))
    assert isinstance(outcome, FinalPlan)
    assert outcome.path is PlanPath.SEMANTIC
    assert outcome.plan.specs[0].metrics == ["opening_pipeline"]
    # The planner does not attach a validation; only the loop's gate can.
    assert outcome.validation is None


def test_a_submitted_edit_is_an_edit_outcome():
    edit = PlanEdit(base_plan_id="p1", operations=[AddDimension(dimension="owner_id")])
    outcome = parse(action("run_analysis_plan", {"edit": edit.model_dump(mode="json")}))
    assert isinstance(outcome, FinalPlan) and outcome.path is PlanPath.EDIT
    assert outcome.edit == edit


def test_an_investigation_carries_its_reason():
    case = CASES_BY_ID["investigation_forecast_changes"]
    outcome = parse(case.oracle[-1])
    assert isinstance(outcome, FinalPlan) and outcome.path is PlanPath.INVESTIGATION
    assert outcome.why_not_semantic.value == "needs_derived_feature"


def test_a_clarification_is_typed():
    outcome = parse(CASES_BY_ID["ambiguity_missing_period"].oracle[-1])
    assert isinstance(outcome, ClarificationOutcome)
    assert outcome.request.reason.value == "missing_period"


def test_a_declared_unanswerable_question_is_rejected():
    outcome = parse(action("declare_unanswerable",
                           {"reason": "concept_unavailable", "concept": "customer_segment"}))
    assert isinstance(outcome, Rejected)
    assert outcome.reason is RejectedReason.CONCEPT_UNAVAILABLE
    assert outcome.concept is C.CUSTOMER_SEGMENT
    assert outcome.reason.chosen_by_model


# ------------------------------------------------------------ malformed output


@pytest.mark.parametrize(
    ("act", "code"),
    [
        (action(None), "no_tool_call"),
        (action("run_guarded_sql", {"sql": "SELECT 1"}), "unknown_tool"),
        (action("execute_python", {"code": "1+1"}), "unknown_tool"),
        (action("run_analysis_plan", ["not", "an", "object"]), "arguments_not_an_object"),
        (action("run_analysis_plan", {}), "invalid_arguments"),
        (action("run_analysis_plan", {"plan": {"specs": []}}), "invalid_arguments"),
        (action("run_analysis_plan", {"plan": Q1_OPENING.model_dump(mode="json"),
                                      "edit": {"base_plan_id": "x", "operations": []}}),
         "invalid_arguments"),
        (action("declare_unanswerable", {"reason": "concept_unavailable"}),
         "invalid_arguments"),
        (action("declare_unanswerable", {"reason": "because_i_said_so"}), "invalid_arguments"),
        (action("request_clarification", {"request": {"reason": "missing_period"}}),
         "invalid_arguments"),
    ],
)
def test_malformed_output_fails_closed(act, code):
    with pytest.raises(MalformedOutput) as exc:
        parse(act)
    assert exc.value.code == code


def test_several_tool_calls_in_one_turn_are_malformed():
    act = action("inspect_dataset", {}, extra_tool_calls=1)
    with pytest.raises(MalformedOutput) as exc:
        parse(act)
    assert exc.value.code == "several_tool_calls"


def test_a_plan_with_an_expression_field_is_malformed():
    plan = Q1_OPENING.model_dump(mode="json")
    plan["specs"][0]["measure_sql"] = "SUM(amount) * 1.1"
    with pytest.raises(MalformedOutput):
        parse(action("run_analysis_plan", {"plan": plan}))


def test_a_trust_tier_cannot_be_supplied_anywhere():
    plan = Q1_OPENING.model_dump(mode="json")
    plan["trust_tier"] = "A"
    with pytest.raises(MalformedOutput):
        parse(action("run_analysis_plan", {"plan": plan}))


def test_the_error_names_locations_never_input_values():
    plan = Q1_OPENING.model_dump(mode="json")
    plan["specs"][0]["metrics"] = "SECRET-VALUE-123"
    with pytest.raises(MalformedOutput) as exc:
        parse(action("run_analysis_plan", {"plan": plan}))
    assert "SECRET-VALUE-123" not in str(exc.value)
    assert "metrics" in str(exc.value)


# ----------------------------------------------------------------- contracts


def test_a_final_plan_carries_exactly_one_plan_for_its_path():
    with pytest.raises(ValidationError):
        FinalPlan(path=PlanPath.SEMANTIC)
    with pytest.raises(ValidationError):
        FinalPlan(path=PlanPath.INVESTIGATION,
                  investigation=CASES_BY_ID["investigation_forecast_changes"].expect.plans[0])
    with pytest.raises(ValidationError):
        FinalPlan(path=PlanPath.EDIT)


def test_a_planning_result_is_terminal_only():
    with pytest.raises(ValidationError):
        PlanningResult(outcome=ToolRequest(tool="inspect_dataset"))


def test_the_tool_list_is_the_existing_surface_plus_one_abstention():
    assert set(TOOLS) == {
        "inspect_dataset", "inspect_concept", "inspect_column", "inspect_values",
        "inspect_relationship", "inspect_sample_rows", "list_available_metrics",
        "probe_materiality", "run_analysis_plan", "run_investigation",
        "request_clarification", "declare_unanswerable",
    }
    assert "run_guarded_sql" not in TOOLS


def test_every_tool_schema_is_a_json_object_schema():
    import json

    for spec in tool_specs():
        schema = json.loads(json.dumps(spec.input_schema))
        assert schema["type"] == "object", spec.name
        assert spec.description
