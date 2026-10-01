"""The tools the planner may call: the existing surface, nothing new (13.5).

Inspection tools run the functions in `agent/tools/surface.py` under the
session's `ToolContext`, so every stance, horizon, grant and row limit that
surface enforces applies unchanged. The planner loop adds only argument
validation (a strict Pydantic model per tool) and a bounded, escaped envelope
for the result.

The terminal tools do not execute anything inside the planning loop. Calling
`run_analysis_plan` or `run_investigation` *submits* a plan: the loop passes it
through the same scope check and gate the run tools use, and a plan that
passes is handed to the deterministic pipeline by the caller (13.3). The
planner never sees a result and never adjusts a plan to one.

`run_guarded_sql` does not exist: guarded SQL is not built.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai_analyst.agent.planner.model import ToolSpec
from ai_analyst.agent.tools import surface
from ai_analyst.contracts.concepts import BusinessConcept
from ai_analyst.contracts.context import CHARS_PER_TOKEN, estimate_tokens
from ai_analyst.contracts.investigation import InvestigationPlan
from ai_analyst.contracts.plan import AnalysisPlan
from ai_analyst.contracts.planner import NotSemanticReason
from ai_analyst.contracts.session import PlanEdit
from ai_analyst.contracts.tools import ClarificationRequest
from ai_analyst.semantic.materiality import MaterialityProbe


class _Args(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# ------------------------------------------------------------------ inspection


class InspectDatasetArgs(_Args):
    class_filter: str | None = Field(default=None, max_length=40)
    name_pattern: str | None = Field(default=None, max_length=64)


class InspectConceptArgs(_Args):
    concept: BusinessConcept


class InspectColumnArgs(_Args):
    name: str = Field(min_length=1, max_length=128)


class InspectValuesArgs(_Args):
    name: str = Field(min_length=1, max_length=128)
    top_k: int = Field(default=20, ge=1, le=surface.MAX_TOP_K)


class InspectRelationshipArgs(_Args):
    a: str = Field(min_length=1, max_length=128)
    b: str = Field(min_length=1, max_length=128)


class InspectSampleRowsArgs(_Args):
    n: int = Field(default=5, ge=1, le=20)
    columns: list[str] | None = Field(default=None, max_length=20)


class ListAvailableMetricsArgs(_Args):
    pass


class ProbeMaterialityArgs(_Args):
    probe: MaterialityProbe
    base_plan: AnalysisPlan


# -------------------------------------------------------------------- terminal


class RunAnalysisPlanArgs(_Args):
    """A fresh semantic plan, or an edit of the session's active plan. Not both."""

    plan: AnalysisPlan | None = None
    edit: PlanEdit | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> RunAnalysisPlanArgs:
        if (self.plan is None) == (self.edit is None):
            raise ValueError("submit exactly one of 'plan' or 'edit'")
        return self


class RunInvestigationArgs(_Args):
    plan: InvestigationPlan
    why_not_semantic: NotSemanticReason


class RequestClarificationArgs(_Args):
    request: ClarificationRequest


class UnanswerableReason(StrEnum):
    CONCEPT_UNAVAILABLE = "concept_unavailable"
    TEMPORAL_VIOLATION = "temporal_violation"
    UNSUPPORTED_ANALYSIS = "unsupported_analysis"
    OUT_OF_COVERAGE = "out_of_coverage"


class DeclareUnanswerableArgs(_Args):
    reason: UnanswerableReason
    concept: BusinessConcept | None = None
    detail: str = Field(default="", max_length=500)

    @model_validator(mode="after")
    def _concept_named(self) -> DeclareUnanswerableArgs:
        if self.reason is UnanswerableReason.CONCEPT_UNAVAILABLE and self.concept is None:
            raise ValueError("an unavailable-concept rejection must name the concept")
        return self


# -------------------------------------------------------------------- registry


@dataclass(frozen=True)
class PlannerTool:
    name: str
    description: str
    args: type[_Args]
    terminal: bool


INSPECTION_TOOLS: tuple[PlannerTool, ...] = (
    PlannerTool(
        "inspect_dataset",
        "The dataset's shape, its concepts and their bindings, and a bounded column "
        "listing. Optional class_filter (an information class) and name_pattern "
        "(a glob over column names).",
        InspectDatasetArgs,
        False,
    ),
    PlannerTool(
        "inspect_concept",
        "One business concept: definition, binding status, bound columns, evidence, "
        "caveats, rival candidates when ambiguous, and whether it is usable now.",
        InspectConceptArgs,
        False,
    ),
    PlannerTool(
        "inspect_column",
        "One physical column: classification, timing, usability under this stance, "
        "grants, and a structural summary when the stance permits.",
        InspectColumnArgs,
        False,
    ),
    PlannerTool(
        "inspect_values",
        "The most frequent values of one permitted, non-text column, bounded by the "
        "knowledge horizon. Use it to check a filter value exists; never to compute "
        "an answer.",
        InspectValuesArgs,
        False,
    ),
    PlannerTool(
        "inspect_relationship",
        "A bounded contingency table of two permitted columns. Descriptive evidence "
        "only.",
        InspectRelationshipArgs,
        False,
    ),
    PlannerTool(
        "inspect_sample_rows",
        "Up to 20 sample rows over permitted, non-text columns, within the horizon.",
        InspectSampleRowsArgs,
        False,
    ),
    PlannerTool(
        "list_available_metrics",
        "Registry metrics available under this stance, with definitions and default "
        "snapshot rules, and every unavailable metric with its missing concepts.",
        ListAvailableMetricsArgs,
        False,
    ),
    PlannerTool(
        "probe_materiality",
        "For an ambiguity with two to four readings, each a typed edit of a base "
        "plan: whether the choice changes the answer (material, not_material, "
        "inconclusive). It never chooses a reading.",
        ProbeMaterialityArgs,
        False,
    ),
)

TERMINAL_TOOLS: tuple[PlannerTool, ...] = (
    PlannerTool(
        "run_analysis_plan",
        "Terminal. Submit a semantic AnalysisPlan built on registry metrics, or, for a "
        "follow-up, a PlanEdit of the active plan. Deterministic validation decides "
        "whether it is valid; a rejection comes back with codes and remedies.",
        RunAnalysisPlanArgs,
        True,
    ),
    PlannerTool(
        "run_investigation",
        "Terminal. Submit an InvestigationPlan, only when no registry metric can answer "
        "the question, with the reason the semantic path does not apply.",
        RunInvestigationArgs,
        True,
    ),
    PlannerTool(
        "request_clarification",
        "Terminal. Ask the user a typed question when an ambiguity would materially "
        "change the answer or a required concept is ambiguous or missing.",
        RequestClarificationArgs,
        True,
    ),
    PlannerTool(
        "declare_unanswerable",
        "Terminal. The question cannot be answered safely from this dataset: a "
        "required concept is unavailable, it needs information unavailable at the "
        "knowledge horizon, the analysis is unsupported, or the period is not covered.",
        DeclareUnanswerableArgs,
        True,
    ),
)

TOOLS: dict[str, PlannerTool] = {t.name: t for t in (*INSPECTION_TOOLS, *TERMINAL_TOOLS)}


def tool_specs() -> list[ToolSpec]:
    """Every tool, in a fixed order so a cached prompt prefix stays stable."""
    return [
        ToolSpec(name=t.name, description=t.description, input_schema=t.args.model_json_schema())
        for t in TOOLS.values()
    ]


# ------------------------------------------------------------------ execution


def run_inspection(ctx: surface.ToolContext, name: str, args: _Args) -> BaseModel:
    """Run one inspection tool under the session scope. The surface enforces everything."""
    match args:
        case InspectDatasetArgs(class_filter=cls, name_pattern=pattern):
            if pattern is not None:
                pattern = surface.safe_name_pattern(pattern)
            return surface.inspect_dataset(ctx, class_filter=cls, name_pattern=pattern)
        case InspectConceptArgs(concept=concept):
            return surface.inspect_concept(ctx, concept)
        case InspectColumnArgs(name=column):
            return surface.inspect_column(ctx, column)
        case InspectValuesArgs(name=column, top_k=k):
            return surface.inspect_values(ctx, column, top_k=k)
        case InspectRelationshipArgs(a=a, b=b):
            return surface.inspect_relationship(ctx, a, b)
        case InspectSampleRowsArgs(n=n, columns=columns):
            return surface.inspect_sample_rows(ctx, n=n, columns=columns)
        case ListAvailableMetricsArgs():
            return surface.list_available_metrics(ctx)
        case ProbeMaterialityArgs(probe=probe, base_plan=base):
            return surface.probe_materiality(ctx, probe, base)
    raise ValueError(f"{name!r} is not an inspection tool")  # pragma: no cover


def envelope(tool: str, payload: dict[str, Any] | BaseModel, max_tokens: int) -> str:
    """A tool result as data: JSON, angle brackets escaped, bounded in size.

    Tenant strings (column names, category values, sampled cells) arrive here and
    can contain anything, including text shaped like instructions or like the
    envelope's own closing tag. Escaping `<` and `>` as JSON unicode escapes
    keeps the JSON equivalent and makes the tag impossible to close from inside.
    """
    data = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
    body = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    body = body.replace("<", "\\u003c").replace(">", "\\u003e")
    if estimate_tokens(body) > max_tokens:
        keep = int(max_tokens * CHARS_PER_TOKEN)
        body = body[:keep] + f"...[truncated by the planner loop at {max_tokens} tokens]"
    return f'<tool_result tool="{tool}">{body}</tool_result>'
