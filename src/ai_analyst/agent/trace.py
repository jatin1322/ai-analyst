"""Build an `AnalysisTrace` from what the deterministic core already produced
(WP8, ARCHITECTURE 13.10, 13.14-13.15, 13.21-13.23).

`build_trace` performs no computation of its own: every field is read off the
gate's `PlanValidation`, each `ResultSet` (its `compilation`, `trust`,
`resolved_snapshots`, `compiled_sql`), and the dataset's `ConceptBindings`.
`trace_plan` is the entry point for a trace with no planner involved -- it
validates and executes a plan through the same functions the run tools and the
planner loop both use, so a traced plan is never validated or executed any
differently than a plan run through the ordinary paths.
"""

from __future__ import annotations

from ai_analyst.agent.tools.surface import (
    ToolContext,
    validate_analysis_plan,
    validate_investigation_plan,
)
from ai_analyst.contracts.answer import AnswerDraft
from ai_analyst.contracts.binding import ConceptBinding, UsageGrant
from ai_analyst.contracts.concepts import BusinessConcept
from ai_analyst.contracts.investigation import InvestigationPlan
from ai_analyst.contracts.plan import AnalysisPlan
from ai_analyst.contracts.planner import FinalPlan, PlanningResult, PlanPath
from ai_analyst.contracts.rejection import PlanValidation
from ai_analyst.contracts.result import ResultSet, TrustAssessment
from ai_analyst.contracts.trace import (
    AnalysisTrace,
    BindingTrace,
    ComputationTrace,
    ComputedQuery,
    EvidenceUsed,
    GrantTrace,
    PlannerTrace,
    ProvenanceSection,
    TimeTrace,
)
from ai_analyst.semantic.execute import run_plan
from ai_analyst.semantic.investigation import run_investigation
from ai_analyst.validation.provenance import scan
from ai_analyst.validation.rendering import ResultRegistry, render


def _binding_trace(binding: ConceptBinding) -> BindingTrace:
    return BindingTrace(
        concept=binding.concept.value,
        columns=binding.columns,
        status=binding.status,
        evidence=tuple(EvidenceUsed(kind=e.kind, source=e.source) for e in binding.evidence),
        caveats=binding.caveats,
    )


def _grant_trace(grant: UsageGrant) -> GrantTrace:
    return GrantTrace(
        grant_id=grant.grant_id,
        kind=grant.kind,
        purposes=tuple(sorted(p.value for p in grant.purposes)),
        source=grant.source.source,
    )


def _plan_stance_cutoff(plan: AnalysisPlan | InvestigationPlan):
    """Where a plan's stance and cutoff live. One field on an investigation,
    the first spec's on an analysis plan (every spec was tightened to the same
    session horizon by `validate_analysis_plan` before this plan ever runs)."""
    if isinstance(plan, InvestigationPlan):
        return plan.stance, plan.knowledge_cutoff
    spec = plan.specs[0]
    return spec.stance, spec.knowledge_cutoff


def _planner_trace(planning: PlanningResult, outcome: FinalPlan) -> PlannerTrace:
    return PlannerTrace(
        outcome_kind=planning.kind,
        path=outcome.path,
        turns=len(planning.turns),
        tool_calls=tuple(planning.actions()),
        repairs=planning.repairs,
        malformed_outputs=planning.malformed_outputs,
        rejection_codes=tuple(sorted(planning.codes())),
        input_tokens=sum(t.input_tokens or 0 for t in planning.turns),
        output_tokens=sum(t.output_tokens or 0 for t in planning.turns),
        context_tokens=planning.context_tokens,
    )


def build_trace(
    question: str,
    planning: PlanningResult | None,
    ctx: ToolContext,
    plan: AnalysisPlan | InvestigationPlan,
    results: list[ResultSet],
    *,
    draft: AnswerDraft | None = None,
    validation: PlanValidation | None = None,
) -> AnalysisTrace:
    """Assemble a trace. Deterministic: reads existing objects, computes nothing.

    `planning` is the run record when a planner produced this plan; its
    `FinalPlan` outcome supplies the plan's path, carry-forward report and gate
    validation. Pass `planning=None` with `validation` supplied explicitly to
    trace a plan that was validated and run with no planner involved, which is
    what `trace_plan` does below.
    """
    planner_trace = None
    carry_forward = None
    plan_path: PlanPath

    if planning is not None:
        outcome = planning.outcome
        if not isinstance(outcome, FinalPlan):
            raise ValueError(
                "build_trace needs a FinalPlan planning outcome to trace an executed plan; "
                f"got {outcome.kind.value}"
            )
        plan_path = outcome.path
        carry_forward = outcome.carry_forward
        if validation is None:
            validation = outcome.validation
        planner_trace = _planner_trace(planning, outcome)
    else:
        plan_path = (
            PlanPath.INVESTIGATION if isinstance(plan, InvestigationPlan) else PlanPath.SEMANTIC
        )

    if validation is None:
        raise ValueError(
            "build_trace needs a gate PlanValidation: from the planner's FinalPlan outcome, "
            "or supplied explicitly"
        )

    is_investigation = isinstance(plan, InvestigationPlan)

    concept_columns: dict[str, str] = {}
    grant_disclosures: set[str] = set()
    resolved_snapshots = []
    attributions = []
    queries: list[ComputedQuery] = []
    trust = TrustAssessment()

    for result in results:
        if result.compilation is not None:
            concept_columns.update(result.compilation.concept_columns)
            grant_disclosures.update(result.compilation.usage_grants)
            attributions.extend(result.compilation.attributions)
        resolved_snapshots.extend(result.resolved_snapshots)
        queries.append(
            ComputedQuery(
                query_id=result.query_id,
                compiled_sql=result.compiled_sql,
                columns=tuple(result.columns),
                row_count=result.row_count,
            )
        )
        trust = trust.combine(result.trust)

    bindings = tuple(
        _binding_trace(ctx.bindings.get(BusinessConcept(name))) for name in sorted(concept_columns)
    )
    grants = tuple(
        _grant_trace(g) for g in ctx.bindings.grants if g.disclosure in grant_disclosures
    )

    stance, cutoff = _plan_stance_cutoff(plan)

    if draft is not None:
        registry = ResultRegistry()
        for result in results:
            registry.register(result)
        rendered = render(draft, registry, trust)
        provenance = ProvenanceSection(report=scan(rendered, question=question))
    else:
        provenance = ProvenanceSection(note="no answer draft was supplied; no prose was rendered")

    return AnalysisTrace(
        question=question,
        planner=planner_trace,
        plan_path=plan_path,
        plan=None if is_investigation else plan,
        investigation=plan if is_investigation else None,
        carry_forward=carry_forward,
        gate=validation,
        bindings=bindings,
        grants=grants,
        time=TimeTrace(
            resolved_snapshots=tuple(resolved_snapshots),
            attribution_reads=tuple(attributions),
            stance=stance,
            knowledge_cutoff=cutoff,
        ),
        computation=ComputationTrace(queries=tuple(queries)),
        trust=trust,
        provenance=provenance,
    )


def trace_plan(
    ctx: ToolContext, plan: AnalysisPlan | InvestigationPlan, question: str
) -> AnalysisTrace:
    """Validate, execute and trace a plan with no planner involved.

    Goes through `validate_analysis_plan`/`validate_investigation_plan` and
    `semantic.execute.run_plan`/`semantic.investigation.run_investigation`, the
    same functions the run tools and the planner loop execute through, so a
    plan traced this way is validated and run exactly as it would be anywhere
    else in the system. Raises `ValueError` if the gate rejects the plan: there
    is nothing to trace that was not executed.
    """
    if isinstance(plan, InvestigationPlan):
        checked = validate_investigation_plan(ctx, plan)
        if not checked.ok:
            raise ValueError(f"plan rejected by the gate: {checked.validation.rejections}")
        with ctx.store.connect() as conn:
            result = run_investigation(
                conn,
                ctx.scan,
                checked.outcome,
                dataset_id=ctx.dataset_id,
                calendar=ctx.calendar,
                settings=ctx.settings,
            )
        results = [result]
    else:
        checked = validate_analysis_plan(ctx, plan)
        if not checked.ok:
            raise ValueError(f"plan rejected by the gate: {checked.validation.rejections}")
        with ctx.store.connect() as conn:
            results = run_plan(
                conn,
                ctx.scan,
                checked.plan,
                checked.outcome,
                dataset_id=ctx.dataset_id,
                calendar=ctx.calendar,
                settings=ctx.settings,
            )
    return build_trace(question, None, ctx, checked.plan, results, validation=checked.validation)


# ------------------------------------------------------------------ rendering


def render_markdown(trace: AnalysisTrace) -> str:
    """A readable, sectioned explanation of one trace.

    Every number here comes from a result's own metadata (row counts, drift
    days, a trust tier) or from the provenance scan; nothing is computed for
    the rendering, and the compiled SQL is the only thing shown verbatim.
    """
    lines: list[str] = ["# Analysis trace", ""]

    lines += ["## What was asked", "", trace.question, ""]

    lines += ["## How it was planned", ""]
    if trace.planner is None:
        lines.append("No planner was involved; the plan was supplied directly.")
    else:
        p = trace.planner
        path = f" ({p.path.value})" if p.path else ""
        lines.append(f"- Outcome: {p.outcome_kind.value}{path}")
        lines.append(
            f"- Turns: {p.turns}, tool calls: {len(p.tool_calls)}, "
            f"repairs: {p.repairs}, malformed outputs: {p.malformed_outputs}"
        )
        if p.tool_calls:
            lines.append(f"- Tools called, in order: {', '.join(p.tool_calls)}")
        if p.rejection_codes:
            lines.append(f"- Rejection codes seen during planning: {', '.join(p.rejection_codes)}")
        lines.append(
            f"- Tokens: {p.input_tokens} in / {p.output_tokens} out / {p.context_tokens} context"
        )
    lines.append("")

    lines += ["## What was checked", ""]
    lines.append(f"- Gate verdict: {'passed' if trace.gate.plan_ok else 'rejected'}")
    for r in trace.gate.rejections:
        remedy = f" — remedy: {r.remedy}" if r.remedy else ""
        lines.append(f"- Rejected (`{r.code.value}`) on `{r.field}`: {r.message}{remedy}")
    if trace.gate.assumptions:
        lines.append("- Assumptions:")
        lines += [f"  - {a}" for a in trace.gate.assumptions]
    if trace.gate.warnings:
        lines.append("- Warnings:")
        lines += [f"  - {w}" for w in trace.gate.warnings]
    lines.append("")

    lines += ["## Which data it used", ""]
    if trace.bindings:
        for b in trace.bindings:
            cols = ", ".join(f"`{c}`" for c in b.columns) or "no column"
            evidence = ", ".join(
                f"{e.kind.value}" + (f" ({e.source})" if e.source else "") for e in b.evidence
            ) or "none"
            lines.append(
                f"- **{b.concept}** -> {cols} (status: {b.status.value}; evidence: {evidence})"
            )
            if b.caveats:
                lines.append(f"  - caveats: {', '.join(b.caveats)}")
    else:
        lines.append("No concept bindings were used by this computation.")
    if trace.grants:
        lines.append("")
        lines.append("Usage grants relied on:")
        for g in trace.grants:
            purposes = ", ".join(g.purposes)
            lines.append(f"- `{g.grant_id}` ({g.kind.value}) for {purposes}: {g.source}")
    lines.append("")

    lines += ["## When (snapshots and horizon)", ""]
    cutoff = ""
    if trace.time.knowledge_cutoff:
        cutoff = f", knowledge cutoff {trace.time.knowledge_cutoff.isoformat()}"
    lines.append(f"- Stance: {trace.time.stance.value}{cutoff}")
    for s in trace.time.resolved_snapshots:
        requested = ""
        if s.requested_boundary:
            requested = f" (requested boundary {s.requested_boundary.isoformat()})"
        lines.append(
            f"- {s.rule.value}: resolved to {s.resolved_as_of.isoformat()}{requested}, "
            f"drift {s.drift_days}d"
        )
    for a in trace.time.attribution_reads:
        lines.append(
            f"- attribution `{a.field}` read from `{a.column}` as of {a.read_as_of.isoformat()} "
            f"({a.relation.value})"
        )
    lines.append("")

    lines += ["## How it was computed", ""]
    for q in trace.computation.queries:
        columns = ", ".join(c.name for c in q.columns)
        lines.append(f"- Query `{q.query_id}`: {q.row_count} row(s); columns: {columns}")
        if q.compiled_sql:
            lines.append(f"```sql\n{q.compiled_sql}\n```")
    lines.append("")

    lines += ["## How far to trust it", ""]
    lines.append(f"- Tier: {trace.trust.tier.value}")
    for f in trace.trust.factors:
        lines.append(f"- {f.kind.value} ({f.subject}): {f.reason}")
    lines.append("")

    lines += ["## Where every number came from", ""]
    if trace.provenance.report is not None:
        report = trace.provenance.report
        lines.append(f"- Coverage: {report.coverage:.0%}")
        for finding in report.findings:
            reason = f": {finding.reason}" if finding.reason else ""
            lines.append(f"- `{finding.text}` — {finding.source.value}{reason}")
    else:
        lines.append(trace.provenance.note)

    return "\n".join(lines)
