"""Print an `AnalysisTrace` as Markdown for one planner golden case.

Builds a synthetic world (default: the case's own world, usually `tiny`) from
`evals.planner.datasets`, runs a named case's expected plan from
`evals.planner.cases.CASES_BY_ID` through `ai_analyst.agent.trace.trace_plan`,
and prints the resulting trace as Markdown. No model and no network: the plan
run is the case's own hand-written golden plan, not a planner's output, so
this is a debugging aid for the deterministic explanation path (WP8), not an
evaluation of the planner.

    python -m scripts.explain metric_opening_q2
    python -m scripts.explain metric_opening_q2 --world tiny
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from ai_analyst.agent.trace import render_markdown, trace_plan
from evals.planner.cases import CASES_BY_ID
from evals.planner.datasets import build_worlds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("case_id", help="a golden case id from evals.planner.cases.CASES_BY_ID")
    parser.add_argument("--world", default=None, help="override the case's own world name")
    args = parser.parse_args(argv)

    case = CASES_BY_ID.get(args.case_id)
    if case is None:
        print(f"no such case: {args.case_id!r}; known: {sorted(CASES_BY_ID)}", file=sys.stderr)
        return 2
    if not case.expect.plans:
        print(f"case {args.case_id!r} has no expected plan to trace", file=sys.stderr)
        return 2

    world_name = args.world or case.world
    with tempfile.TemporaryDirectory() as tmp:
        worlds = build_worlds(Path(tmp), names={world_name})
        world = worlds[world_name]
        ctx = world.tool_context(stance=case.stance, horizon=case.horizon)
        plan = case.expect.plans[0]
        trace = trace_plan(ctx, plan, case.question)

    print(render_markdown(trace))
    return 0


if __name__ == "__main__":
    sys.exit(main())
