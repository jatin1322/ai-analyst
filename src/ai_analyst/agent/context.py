"""The planner's default context (ARCHITECTURE 13.4).

A deterministic projection, built per turn and budgeted in tokens. Tier 0 is the
analyst card of 12.5 plus three compact blocks: what can be planned (the metric
catalog under the session stance), what the temporal rules are right now, and
what the session is working on. Nothing here reads a row: tier 1 and tier 2
arrive only through the stance-bounded tools.

The budget is enforced, not hoped for. When the projection is over budget the
card drops its column index first, which is the one block the listing tool can
always recover on request.
"""

from __future__ import annotations

from dataclasses import dataclass

from ai_analyst.agent.tools.surface import ToolContext, list_available_metrics
from ai_analyst.contracts.concepts import RETROSPECTIVE_CONCEPTS
from ai_analyst.contracts.context import estimate_tokens
from ai_analyst.contracts.plan import AnalysisPlan
from ai_analyst.contracts.tenant import TenantProfile
from ai_analyst.data.understanding import build_context
from ai_analyst.session.state import SessionState


@dataclass(frozen=True)
class PlannerContext:
    card: str
    catalog: str
    temporal: str
    session: str
    budget: int
    index_degraded: bool

    def render(self) -> str:
        blocks = [self.card, self.catalog, self.temporal]
        if self.session:
            blocks.append(self.session)
        return "\n\n".join(blocks)

    @property
    def estimated_tokens(self) -> int:
        return estimate_tokens(self.render())

    @property
    def within_budget(self) -> bool:
        return self.estimated_tokens <= self.budget


def catalog_block(ctx: ToolContext) -> str:
    """Metric availability, names only: definitions are one tool call away."""
    catalog = list_available_metrics(ctx)
    names = ", ".join(m.name for m in catalog.available) or "none"
    lines = [f"metrics ({ctx.stance.value}): {names}"]
    if catalog.unavailable:
        missing = "; ".join(
            f"{m.name} ({', '.join(m.missing_concepts) or m.reason})" for m in catalog.unavailable
        )
        lines.append(f"unavailable: {missing}")
    return "\n".join(lines)


def temporal_block(ctx: ToolContext) -> str:
    """The rules that decide what may be read this turn."""
    horizon = ctx.horizon.isoformat() if ctx.horizon else "none (latest snapshot)"
    calendar = (
        f"month {ctx.calendar.start_month}"
        + ("" if ctx.calendar.is_resolved else " UNRESOLVED (configured default)")
    )
    hidden = ", ".join(sorted(c.value for c in RETROSPECTIVE_CONCEPTS))
    permitted = len(ctx.resolver().permitted_columns)
    return (
        f"stance: {ctx.stance.value}   horizon: {horizon}   fiscal year start: {calendar}\n"
        f"retrospective only: {hidden}\n"
        f"readable columns under this stance: {permitted} of {len(ctx.registry.columns)}"
    )


def session_block(session: SessionState | None) -> str:
    """The active plan in compact form, so a follow-up edits it rather than re-plans."""
    if session is None or not isinstance(session.active, AnalysisPlan):
        return ""
    plan = session.active
    lineage = f" (from {plan.parent_plan_id})" if plan.parent_plan_id else ""
    lines = [f"active plan {plan.plan_id}{lineage}"]
    for spec in plan.specs:
        period = spec.period.label or spec.period.kind.value
        parts = [
            f"{spec.id}: {spec.pattern.value} {','.join(spec.metrics) or '-'}",
            f"period {period}",
            f"snapshot {spec.snapshot.rule.value}",
            f"stance {spec.stance.value}",
        ]
        if spec.dimensions:
            parts.append(f"by {','.join(spec.dimensions)}")
        if spec.filters:
            parts.append(
                "where " + " and ".join(
                    f"{f.column} {f.op.value} {','.join(map(str, f.values))}" for f in spec.filters
                )
            )
        lines.append("  " + " | ".join(parts))
    return "\n".join(lines)


def build_planner_context(
    dataset,
    understanding,
    ctx: ToolContext,
    *,
    tenant: TenantProfile | None = None,
    session: SessionState | None = None,
    budget: int | None = None,
) -> PlannerContext:
    """Tier 0 for one turn, within budget."""
    budget = budget or ctx.settings.context_card_token_budget
    card = understanding.context
    blocks = (catalog_block(ctx), temporal_block(ctx), session_block(session))

    def assemble(analyst) -> PlannerContext:
        return PlannerContext(
            card=analyst.render(),
            catalog=blocks[0],
            temporal=blocks[1],
            session=blocks[2],
            budget=budget,
            index_degraded=analyst.index_degraded,
        )

    context = assemble(card)
    if context.within_budget or card.index_degraded:
        return context
    degraded = build_context(
        dataset.schema,
        dataset.registry,
        dataset.profile,
        understanding.bindings,
        understanding.agreement,
        ctx.settings.model_copy(update={"context_card_max_indexed_columns": 0}),
        tenant,
    )
    return assemble(degraded)
