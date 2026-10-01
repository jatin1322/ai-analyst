"""AnalysisTrace: one deterministic object that explains an answer end to end.

Every test here reads a trace built by `build_trace`/`trace_plan` and checks
that it names the right things: a binding's evidence, a resolved snapshot, an
attribution's relation, a repaired rejection code, the trust tier and its
factors. None of it touches a network; the planner-driven test uses
`ScriptedModel`, as every other planner test does.
"""

from __future__ import annotations

import json
from datetime import date

from ai_analyst.agent.planner import LLMPlanner, PlanningLoop
from ai_analyst.agent.planner.fake import ScriptedModel, action
from ai_analyst.agent.tools.surface import validate_analysis_plan
from ai_analyst.agent.trace import build_trace, render_markdown, trace_plan
from ai_analyst.contracts.answer import AnswerDraft, NumeralSource
from ai_analyst.contracts.binding import EvidenceKind
from ai_analyst.contracts.plan import (
    AnalysisPattern,
    AnalysisPlan,
    AnalysisSpec,
    AnalysisStance,
    Period,
    PeriodKind,
    SnapshotSelection,
)
from ai_analyst.contracts.planner import PlanPath
from ai_analyst.contracts.result import SnapshotRule, TemporalRelation
from ai_analyst.semantic.execute import run_plan
from evals.planner.cases import CASES_BY_ID, Q1_OPENING, Q2_OPENING, submit
from evals.planner.datasets import INJECTION_TEXT

PROSPECTIVE = AnalysisStance.PROSPECTIVE


def _run_plan(ctx, plan):
    """Validate and execute a plan the same way `trace_plan` does, but keep
    the checked plan and validation around for tests that also drive the
    planning loop."""
    checked = validate_analysis_plan(ctx, plan)
    assert checked.ok, checked.validation.rejections
    with ctx.store.connect() as conn:
        results = run_plan(
            conn, ctx.scan, checked.plan, checked.outcome,
            dataset_id=ctx.dataset_id, calendar=ctx.calendar, settings=ctx.settings,
        )
    return checked, results


# --------------------------------------------------------------- trace_plan


def test_trace_metric_opening_q2_names_binding_snapshot_sql_trust(tiny):
    case = CASES_BY_ID["metric_opening_q2"]
    ctx = tiny.tool_context(stance=PROSPECTIVE)
    trace = trace_plan(ctx, case.expect.plans[0], case.question)

    binding = next(b for b in trace.bindings if b.concept == "amount")
    assert binding.columns == ("amount",)
    assert any(e.kind is EvidenceKind.TENANT_CONFIG for e in binding.evidence)

    assert len(trace.time.resolved_snapshots) == 1
    snapshot = trace.time.resolved_snapshots[0]
    assert snapshot.rule is SnapshotRule.PERIOD_OPEN
    assert snapshot.resolved_as_of == date(2025, 4, 1)

    assert len(trace.computation.queries) == 1
    query = trace.computation.queries[0]
    assert query.compiled_sql is not None
    assert "opening_pipeline" in query.compiled_sql
    assert query.row_count == 1

    assert trace.trust.tier is not None
    assert trace.trust.factors  # at least the semantic-path factor
    assert trace.plan_path is PlanPath.SEMANTIC
    assert trace.planner is None
    assert trace.provenance.report is None
    assert "no answer draft" in trace.provenance.note


def test_trace_grant_world_lists_usage_grant(worlds):
    granted = worlds["granted"]
    ctx = granted.tool_context(stance=PROSPECTIVE)
    trace = trace_plan(ctx, Q1_OPENING, "What was Q1 FY2025 opening pipeline?")

    binding = next(b for b in trace.bindings if b.concept == "amount")
    assert binding.columns == ("enterprise_amount",)

    assert len(trace.grants) == 1
    grant = trace.grants[0]
    assert grant.kind.value == "concept"
    assert "measure" in grant.purposes
    assert grant.source == "evaluation fixture"

    reasons = [f.reason for f in trace.trust.factors]
    assert any("enterprise_amount" in r for r in reasons)


def test_trace_retrospective_attribution_shows_later_relation(worlds):
    case = CASES_BY_ID["temporal_retrospective_attribution"]
    world = worlds[case.world]
    ctx = world.tool_context(stance=case.stance, horizon=case.horizon)
    trace = trace_plan(ctx, case.expect.plans[0], case.question)

    assert len(trace.time.attribution_reads) == 1
    read = trace.time.attribution_reads[0]
    assert read.relation is TemporalRelation.LATER
    assert read.relation.value == "later"
    assert read.field == "stage"
    assert trace.time.stance is AnalysisStance.RETROSPECTIVE


# ------------------------------------------------------------ from a planner


def test_trace_from_planning_result_records_repairs_and_hides_tool_content(worlds):
    world = worlds["injection"]
    ctx = world.tool_context(stance=PROSPECTIVE)
    context = world.planner_context(ctx, None)

    bad_plan = AnalysisPlan(
        question_restatement="bad",
        specs=[
            AnalysisSpec(
                id="s",
                pattern=AnalysisPattern.POINT_IN_TIME,
                metrics=["opening_pipeline"],
                dimensions=["not_a_real_dimension"],
                period=Period(kind=PeriodKind.FISCAL_QUARTER, label="FY2025-Q2"),
                snapshot=SnapshotSelection(rule=SnapshotRule.PERIOD_OPEN),
            )
        ],
    )
    actions = [
        action("inspect_values", {"name": "forecast_category"}),
        submit(bad_plan),
        submit(Q2_OPENING),
    ]
    model = ScriptedModel(actions)
    loop = PlanningLoop(ctx=ctx, settings=ctx.settings, session=None)
    planning = loop.run(LLMPlanner(model), "What was Q2 opening pipeline?", context)

    assert planning.kind.value == "final_plan"
    assert planning.repairs == 1
    assert "unknown_column" in planning.codes()
    assert planning.actions() == [
        "inspect_values", "run_analysis_plan", "run_analysis_plan",
    ]

    with ctx.store.connect() as conn:
        checked = validate_analysis_plan(ctx, planning.outcome.plan)
        results = run_plan(
            conn, ctx.scan, checked.plan, checked.outcome,
            dataset_id=ctx.dataset_id, calendar=ctx.calendar, settings=ctx.settings,
        )

    trace = build_trace("What was Q2 opening pipeline?", planning, ctx, checked.plan, results)

    assert trace.planner is not None
    assert trace.planner.repairs == 1
    assert "unknown_column" in trace.planner.rejection_codes
    assert trace.planner.tool_calls == ("inspect_values", "run_analysis_plan", "run_analysis_plan")

    # The tool result content (a forecast category value, including the
    # injected text) must never reach the trace: only the tool's name and
    # outcome metadata do.
    serialized = trace.model_dump_json()
    assert INJECTION_TEXT not in serialized
    assert "forecast_cat" not in json.dumps(json.loads(serialized).get("planner", {}))


# ------------------------------------------------------------------ rendering


def test_render_markdown_all_headers_present(tiny):
    case = CASES_BY_ID["metric_opening_q2"]
    ctx = tiny.tool_context(stance=PROSPECTIVE)
    trace = trace_plan(ctx, case.expect.plans[0], case.question)
    text = render_markdown(trace)

    for header in (
        "## What was asked",
        "## How it was planned",
        "## What was checked",
        "## Which data it used",
        "## When (snapshots and horizon)",
        "## How it was computed",
        "## How far to trust it",
        "## Where every number came from",
    ):
        assert header in text
    assert "```sql" in text


def test_provenance_marks_invented_number_unverified(tiny):
    ctx = tiny.tool_context(stance=PROSPECTIVE)
    checked, results = _run_plan(ctx, Q2_OPENING)
    question = "What was Q2 opening pipeline?"
    draft = AnswerDraft(
        headline="Opening pipeline was {{q1.r0.opening_pipeline}}; certainly not 515001.",
    )
    trace = build_trace(
        question, None, ctx, checked.plan, results, draft=draft, validation=checked.validation,
    )

    assert trace.provenance.report is not None
    unverified = trace.provenance.report.unverified
    assert any(f.text == "515001" for f in unverified)
    assert all(f.source is NumeralSource.UNVERIFIED for f in unverified)

    text = render_markdown(trace)
    assert "515001" in text
    assert "unverified" in text


# -------------------------------------------------------------- determinism


def test_trace_is_deterministic_apart_from_query_ids(tiny):
    ctx = tiny.tool_context(stance=PROSPECTIVE)
    question = "What was Q2 opening pipeline?"

    first = trace_plan(ctx, Q2_OPENING, question)
    second = trace_plan(ctx, Q2_OPENING, question)

    def normalized(trace):
        data = json.loads(trace.model_dump_json())
        for query in data["computation"]["queries"]:
            query["query_id"] = "<query_id>"
        return data

    assert normalized(first) == normalized(second)
    # Sanity: the ids really do differ between two independent executions,
    # so the normalisation above is doing real work rather than nothing.
    assert first.computation.queries[0].query_id != second.computation.queries[0].query_id


# ----------------------------------------------------------------- explain


def test_explain_script_runs_for_one_case(capsys):
    from scripts.explain import main

    exit_code = main(["metric_opening_q2"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "## What was asked" in out
