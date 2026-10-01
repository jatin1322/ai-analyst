"""Planner test fixtures. Network-free: every model here is scripted."""

from __future__ import annotations

import pytest

from ai_analyst.agent.planner import LLMPlanner, PlanningLoop
from ai_analyst.agent.planner.fake import ScriptedModel
from evals.planner.datasets import build_worlds


@pytest.fixture(scope="session")
def worlds(tmp_path_factory):
    """Every evaluation world, ingested once for the session."""
    return build_worlds(tmp_path_factory.mktemp("planner_worlds"))


@pytest.fixture
def tiny(worlds):
    return worlds["tiny"]


def run(world, actions, question="What was Q1 opening pipeline?", *, stance=None,
        horizon=None, session=None, settings=None):
    """Plan one question with a scripted model. Returns (result, model)."""
    from ai_analyst.contracts.plan import AnalysisStance

    ctx = world.tool_context(stance=stance or AnalysisStance.PROSPECTIVE, horizon=horizon)
    if settings is not None:
        ctx.settings = settings
    model = ScriptedModel(actions)
    context = world.planner_context(ctx, session)
    loop = PlanningLoop(ctx=ctx, settings=ctx.settings, session=session)
    return loop.run(LLMPlanner(model), question, context), model
