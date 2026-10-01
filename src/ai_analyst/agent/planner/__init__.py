"""The LLM planner (ARCHITECTURE 13.6, 13.23).

The planner proposes a typed plan; the deterministic system decides whether it
is valid and computes the answer. Importing this package never imports a
provider SDK: the Anthropic adapter loads the SDK only when it is used.
"""

from __future__ import annotations

from ai_analyst.agent.context import PlannerContext
from ai_analyst.agent.planner.loop import PlanningLoop
from ai_analyst.agent.planner.model import PlannerModel
from ai_analyst.agent.planner.planner import LLMPlanner, Planner
from ai_analyst.agent.tools.surface import ToolContext
from ai_analyst.contracts.planner import PlanningResult
from ai_analyst.session.state import SessionState

__all__ = ["LLMPlanner", "Planner", "PlanningLoop", "plan_question"]


def plan_question(
    question: str,
    model: PlannerModel,
    ctx: ToolContext,
    context: PlannerContext,
    *,
    session: SessionState | None = None,
) -> PlanningResult:
    """Plan one question with one model under one tool scope. Executes nothing."""
    loop = PlanningLoop(ctx=ctx, settings=ctx.settings, session=session)
    return loop.run(LLMPlanner(model), question, context)
