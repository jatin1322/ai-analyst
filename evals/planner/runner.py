"""Run golden cases through the real planning loop and score them.

The same function serves the network-free replay (a scripted model playing
each case's reference trajectory) and the opt-in real-model evaluation: only
the model differs. Everything downstream of the model is the production code
path: the planner, the loop, the surface tools, the gate, and execution.
"""

from __future__ import annotations

from collections.abc import Callable

from ai_analyst.agent.planner import plan_question
from ai_analyst.agent.planner.evaluation import CaseResult, EvaluationReport, evaluate_case
from ai_analyst.agent.planner.fake import ScriptedModel
from ai_analyst.agent.planner.model import PlannerModel
from ai_analyst.contracts.planner import PlanningResult
from ai_analyst.session.state import SessionState
from evals.planner.cases import PlannerCase
from evals.planner.datasets import World


def session_for(case: PlannerCase, world: World) -> SessionState | None:
    if case.session_plan is None:
        return None
    session = SessionState(
        session_id=f"eval-{case.id}", dataset_id=world.dataset.dataset_id, stance=case.stance
    )
    session.record(case.session_plan)
    return session


def run_case(
    case: PlannerCase, world: World, model: PlannerModel, *, execute: bool = True
) -> tuple[PlanningResult, CaseResult]:
    ctx = world.tool_context(stance=case.stance, horizon=case.horizon)
    session = session_for(case, world)
    context = world.planner_context(ctx, session)
    result = plan_question(case.question, model, ctx, context, session=session)
    scored = evaluate_case(
        case.id, case.category, case.expect, result, ctx,
        tool_budget=case.tool_budget, execute=execute,
    )
    return result, scored


def oracle_model(case: PlannerCase) -> ScriptedModel:
    return ScriptedModel(case.oracle)


def evaluate(
    cases: tuple[PlannerCase, ...],
    worlds: dict[str, World],
    model_for: Callable[[PlannerCase], PlannerModel],
) -> EvaluationReport:
    return EvaluationReport(
        [run_case(case, worlds[case.world], model_for(case))[1] for case in cases]
    )
