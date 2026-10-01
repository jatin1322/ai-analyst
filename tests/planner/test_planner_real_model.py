"""Opt-in real-model planner evaluation. Never part of the default run.

    AI_ANALYST_PLANNER_EVAL=1 ANTHROPIC_API_KEY=... pytest tests/planner/test_planner_real_model.py

These tests do not assert a pass rate: planner quality is reported, by
category, not gated here. They assert the invariants that must hold whatever
the model does, because they are enforced below it.
"""

from __future__ import annotations

import os

import pytest

from evals.planner.run_planner_eval import credentials_present

pytestmark = pytest.mark.skipif(
    os.environ.get("AI_ANALYST_PLANNER_EVAL") != "1" or not credentials_present(),
    reason="set AI_ANALYST_PLANNER_EVAL=1 and Anthropic credentials to run the real planner",
)


def test_real_model_smoke():
    from evals.planner.run_planner_eval import SMOKE_CASE, run

    report = run([SMOKE_CASE])
    (result,) = report.results
    # Whatever it chose, the run finished through the loop with a typed outcome.
    assert result.turns >= 1


def test_real_model_golden_set():
    from ai_analyst.config import Settings
    from ai_analyst.contracts.planner import PlannerOutcomeKind
    from evals.planner.run_planner_eval import run, write_report

    report = run()
    print(report.render())
    write_report(report, os.environ.get("AI_ANALYST_PLANNER_MODEL", "configured"))
    for result in report.results:
        # A final plan exists only when the gate passed it, whatever the model did.
        if result.outcome is PlannerOutcomeKind.FINAL_PLAN:
            assert result.valid_plan
        # No run exceeded the loop's own bounds.
        assert result.turns <= Settings().planner_max_turns
