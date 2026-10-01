"""Opt-in: evaluate the real planner model against the golden set.

    python -m evals.planner.run_planner_eval              # every case
    python -m evals.planner.run_planner_eval --smoke      # one case: checks the request shape
    python -m evals.planner.run_planner_eval --case metric_opening_q1 --case refusal_forecast

Requires the `llm` extra (`pip install -e '.[llm]'`) and Anthropic credentials
in the SDK's standard environment (for example `ANTHROPIC_API_KEY`). The key is
never read, printed or stored here; the SDK resolves it.

Runs on the synthetic evaluation worlds only, never on a real export. The
report printed and written (under `evals/reports/`, which git ignores) holds
case ids, outcome kinds, failure categories, tool counts and token usage. No
prompt, tool result, plan value or dataset value is written.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from ai_analyst.agent.planner.anthropic_adapter import AnthropicPlannerModel
from ai_analyst.agent.planner.evaluation import EvaluationReport
from ai_analyst.config import Settings
from evals.planner.cases import CASES, CASES_BY_ID
from evals.planner.datasets import build_worlds
from evals.planner.runner import run_case

REPORTS = Path(__file__).resolve().parents[1] / "reports"
SMOKE_CASE = "metric_opening_q1"


def credentials_present() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def run(case_ids: list[str] | None = None, *, model_name: str | None = None) -> EvaluationReport:
    settings = Settings()
    if model_name:
        settings = settings.model_copy(update={"planner_model": model_name})
    cases = tuple(CASES_BY_ID[c] for c in case_ids) if case_ids else CASES
    with tempfile.TemporaryDirectory() as root:
        worlds = build_worlds(Path(root), {c.world for c in cases})
        results = []
        for case in cases:
            model = AnthropicPlannerModel.from_settings(settings)
            _, scored = run_case(case, worlds[case.world], model)
            results.append(scored)
        return EvaluationReport(results)


def write_report(report: EvaluationReport, model_name: str) -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = REPORTS / f"planner-{stamp}.json"
    payload = {
        "model": model_name,
        "metrics": report.metrics,
        "failure_counts": report.failure_counts,
        "attempt_failure_counts": report.attempt_failure_counts,
        "cases": [
            {
                "id": r.case_id, "category": r.category, "expected": r.expected.value,
                "outcome": r.outcome.value, "path": r.path.value if r.path else None,
                "reason": r.reason, "passed": r.passed,
                "failures": [f.value for f in r.failures],
                "attempt_failures": [f.value for f in r.attempt_failures],
                "mismatched_fields": r.mismatched_fields, "tool_calls": r.tool_calls,
                "turns": r.turns, "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
            }
            for r in report.results
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--case", action="append", help="a case id; repeatable")
    parser.add_argument("--smoke", action="store_true", help=f"run only {SMOKE_CASE}")
    parser.add_argument("--model", help="override the configured planner model")
    args = parser.parse_args(argv)
    if not credentials_present():
        print("planner evaluation not run: no Anthropic credentials in the environment",
              file=sys.stderr)
        return 2
    cases = [SMOKE_CASE] if args.smoke else args.case
    model_name = args.model or Settings().planner_model
    report = run(cases, model_name=model_name)
    print(report.render())
    print(f"\nreport written to {write_report(report, model_name)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
