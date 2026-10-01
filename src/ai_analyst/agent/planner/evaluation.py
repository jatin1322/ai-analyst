"""Planner evaluation: structural comparison and a failure taxonomy (ARCHITECTURE 13.18).

Plans are compared semantically, never as serialized text. The normaliser runs
a plan through the same scope and gate the planner's submissions go through and
compares what the gate *resolved*: the period's actual start and end, the
snapshot's actual date, the physical column each dimension and filter resolved
to. That makes `owner` and `owner_id`, or a quarter label and the same quarter
as custom dates, compare equal without loosening anything else. The snapshot
rule, metrics, stance, analysis options and measure concept are compared as
written. Plan ids, restatements, assumptions and spec ids are ignored.

Where two plans are both legitimate answers to a question, the case lists
both; the normaliser is never loosened globally.

Failures are recorded at two levels:

* **attempt** failures, from every rejected submission or refused tool call in
  the run, including ones the planner later repaired. A future-leak attempt that
  the gate caught is still a future-leak attempt.
* **final** failures, from comparing the terminal outcome with the case's
  expectation.

No score here is produced by a model, and there is no single opaque number.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ai_analyst.agent.tools import surface
from ai_analyst.contracts.concepts import BusinessConcept
from ai_analyst.contracts.investigation import InvestigationPlan
from ai_analyst.contracts.plan import AnalysisPlan
from ai_analyst.contracts.planner import (
    FinalPlan,
    PlannerOutcomeKind,
    PlanningResult,
    PlanPath,
    Rejected,
    RejectedReason,
)
from ai_analyst.contracts.rejection import RejectionCode
from ai_analyst.contracts.tools import ClarificationReason


class FailureCategory(StrEnum):
    WRONG_METRIC = "wrong_metric"
    WRONG_CONCEPT = "wrong_concept"
    WRONG_SNAPSHOT = "wrong_snapshot"
    WRONG_FILTER = "wrong_filter"
    WRONG_TIME_WINDOW = "wrong_time_window"
    WRONG_STANCE = "wrong_stance"
    WRONG_COMPARISON = "wrong_comparison"
    WRONG_ANALYSIS_TYPE = "wrong_analysis_type"
    WRONG_INVESTIGATION = "wrong_investigation"
    WRONG_REASON = "wrong_reason"
    FUTURE_LEAK_ATTEMPT = "future_leak_attempt"
    INVESTIGATION_BYPASS = "investigation_bypass"
    INVENTED_BINDING = "invented_binding"
    INVENTED_LITERAL = "invented_literal"
    SCOPE_BYPASS = "scope_bypass"
    BAD_EDIT = "bad_edit"
    UNNECESSARY_CLARIFICATION = "unnecessary_clarification"
    MISSING_CLARIFICATION = "missing_clarification"
    UNNECESSARY_REFUSAL = "unnecessary_refusal"
    MISSING_REFUSAL = "missing_refusal"
    MALFORMED_OUTPUT = "malformed_output"
    UNSUPPORTED_ANALYSIS = "unsupported_analysis"
    EXCESSIVE_TOOL_USE = "excessive_tool_use"
    EXECUTION_FAILED = "execution_failed"
    RUN_FAILED = "run_failed"


F = FailureCategory
R = RejectionCode

# Gate and loop codes, as failure categories. Every rejection code is mapped.
CODE_CATEGORIES: dict[str, FailureCategory] = {
    R.STANCE_VIOLATION.value: F.FUTURE_LEAK_ATTEMPT,
    R.KNOWLEDGE_CUTOFF_VIOLATION.value: F.FUTURE_LEAK_ATTEMPT,
    R.RETROSPECTIVE_CONCEPT_IN_PROSPECTIVE.value: F.FUTURE_LEAK_ATTEMPT,
    R.COLUMN_NOT_KNOWABLE_AT_SNAPSHOT.value: F.FUTURE_LEAK_ATTEMPT,
    R.METRIC_STANCE_INCOMPATIBLE.value: F.FUTURE_LEAK_ATTEMPT,
    R.SEMANTIC_PATH_AVAILABLE.value: F.INVESTIGATION_BYPASS,
    R.UNKNOWN_METRIC.value: F.INVENTED_BINDING,
    R.UNKNOWN_COLUMN.value: F.INVENTED_BINDING,
    R.COLUMN_UNCLASSIFIED.value: F.INVENTED_BINDING,
    R.COLUMN_QUARANTINED.value: F.INVENTED_BINDING,
    R.UNKNOWN_MEASURE_CONCEPT.value: F.INVENTED_BINDING,
    R.CONCEPT_NOT_CONFIRMED.value: F.INVENTED_BINDING,
    R.CONCEPT_AMBIGUOUS.value: F.INVENTED_BINDING,
    R.CONCEPT_UNAVAILABLE.value: F.INVENTED_BINDING,
    R.CONCEPT_WITHHELD.value: F.INVENTED_BINDING,
    R.METRIC_UNAVAILABLE.value: F.INVENTED_BINDING,
    R.GRANT_PURPOSE_NOT_PERMITTED.value: F.SCOPE_BYPASS,
    R.DECLARATION_CONFLICT.value: F.SCOPE_BYPASS,
    R.METRIC_PATTERN_MISMATCH.value: F.UNSUPPORTED_ANALYSIS,
    R.INVALID_OUTPUT_SHAPE.value: F.UNSUPPORTED_ANALYSIS,
    R.INEXPRESSIBLE.value: F.UNSUPPORTED_ANALYSIS,
    R.INVALID_OPERATION.value: F.UNSUPPORTED_ANALYSIS,
    R.TOO_MANY_VARIABLES.value: F.UNSUPPORTED_ANALYSIS,
    R.TOO_MANY_GROUPINGS.value: F.UNSUPPORTED_ANALYSIS,
    R.UNKNOWN_VARIABLE.value: F.UNSUPPORTED_ANALYSIS,
    R.MEASURE_CONCEPT_NOT_APPLICABLE.value: F.UNSUPPORTED_ANALYSIS,
    R.INCOMPATIBLE_COMPARISON.value: F.WRONG_COMPARISON,
    R.INVALID_FILTER_VALUE.value: F.WRONG_FILTER,
    R.PERIOD_UNRESOLVABLE.value: F.WRONG_TIME_WINDOW,
    R.SNAPSHOT_UNRESOLVABLE.value: F.WRONG_SNAPSHOT,
    R.SNAPSHOT_COVERAGE.value: F.WRONG_SNAPSHOT,
    # Loop codes.
    "invented_literal": F.INVENTED_LITERAL,
    "unresolved_ambiguities": F.MISSING_CLARIFICATION,
    "edit_without_base": F.BAD_EDIT,
    "edit_base_mismatch": F.BAD_EDIT,
    "tool_budget_exhausted": F.EXCESSIVE_TOOL_USE,
}


def attempt_category(code: str) -> FailureCategory | None:
    if code.startswith("malformed:"):
        return F.MALFORMED_OUTPUT
    if code.startswith("edit_conflict:"):
        return F.BAD_EDIT
    return CODE_CATEGORIES.get(code)


# Which plan field a difference belongs to.
FIELD_CATEGORIES: dict[str, FailureCategory] = {
    "pattern": F.WRONG_ANALYSIS_TYPE,
    "metrics": F.WRONG_METRIC,
    "measure_concept": F.WRONG_CONCEPT,
    "options": F.WRONG_METRIC,
    "dimensions": F.WRONG_CONCEPT,
    "features": F.WRONG_CONCEPT,
    "period": F.WRONG_TIME_WINDOW,
    "snapshot": F.WRONG_SNAPSHOT,
    "attribution": F.WRONG_SNAPSHOT,
    "stance": F.WRONG_STANCE,
    "knowledge_cutoff": F.WRONG_STANCE,
    "filters": F.WRONG_FILTER,
    "comparison": F.WRONG_COMPARISON,
    "shape": F.WRONG_ANALYSIS_TYPE,
    "population_period": F.WRONG_TIME_WINDOW,
    "population_snapshots": F.WRONG_SNAPSHOT,
    "population_filters": F.WRONG_FILTER,
    "variables": F.WRONG_INVESTIGATION,
    "grouping": F.WRONG_INVESTIGATION,
    "operation": F.WRONG_INVESTIGATION,
    "specs": F.WRONG_ANALYSIS_TYPE,
}


# ------------------------------------------------------------------ normalise


def _value(v: Any) -> str:
    return str(v).strip().lower()


def normalise_analysis(plan: AnalysisPlan, ctx: surface.ToolContext) -> list[dict[str, Any]]:
    """Each spec reduced to what it means, resolved by the gate where possible."""
    from ai_analyst.semantic.compiler import _dimension_columns, _resolved_filters

    checked = surface.validate_analysis_plan(ctx, plan)
    specs = []
    for spec in checked.plan.specs:
        validated = checked.outcome.specs.get(spec.id) if checked.ok else None
        if validated is not None:
            period = (validated.period_start, validated.period_end)
            # The rule and any explicit date. The resolved date follows from the
            # period, so comparing it would count a wrong period twice.
            snapshot = (spec.snapshot.rule.value, spec.snapshot.explicit_date)
            dimensions = sorted(_dimension_columns(validated))
            filters = sorted(
                (f.column, f.op.value, tuple(sorted(_value(v) for v in f.values)))
                for f in _resolved_filters(validated)
            )
            comparison = (
                spec.comparison.kind.value,
                validated.baseline.period_start if validated.baseline else None,
            )
        else:
            period = (spec.period.kind.value, spec.period.label, spec.period.start,
                      spec.period.end, spec.period.relative)
            snapshot = (spec.snapshot.rule.value, spec.snapshot.explicit_date)
            dimensions = sorted(spec.dimensions)
            filters = sorted(
                (f.column, f.op.value, tuple(sorted(_value(v) for v in f.values)))
                for f in spec.filters
            )
            comparison = (spec.comparison.kind.value, spec.comparison.baseline)
        specs.append(
            {
                "pattern": spec.pattern.value,
                "metrics": sorted(spec.metrics),
                "measure_concept": spec.measure_concept,
                "options": (
                    spec.creation_basis.value, spec.slip_basis.value,
                    spec.win_rate_basis.value, spec.rate_key.value,
                ),
                "attribution": spec.attribution.value if spec.dimensions or spec.features
                else None,
                "dimensions": dimensions,
                "features": sorted(spec.features),
                "period": period,
                "snapshot": snapshot,
                "stance": spec.stance.value,
                "knowledge_cutoff": spec.knowledge_cutoff,
                "filters": filters,
                "comparison": comparison,
                "shape": (spec.limit, tuple((o.column, o.direction.value) for o in spec.order_by)),
            }
        )
    return sorted(specs, key=lambda s: (s["pattern"], s["metrics"]))


def _source(variable) -> tuple:
    if variable.concept is not None:
        return ("concept", variable.concept.value)
    if variable.column is not None:
        return ("column", variable.column)
    ref = variable.derived
    return ("derived", ref.feature.value, ref.concept.value if ref.concept else None)


def normalise_investigation(plan: InvestigationPlan, ctx: surface.ToolContext) -> dict:
    """Variables by what they read, not by the ids the planner gave them."""
    checked = surface.validate_investigation_plan(ctx, plan)
    plan = checked.plan
    sources = {v.id: _source(v) for v in plan.variables}
    pop = plan.population
    validated = checked.outcome.validated if checked.ok else None
    period = (
        (validated.period.start, validated.period.end)
        if validated is not None
        else (pop.period.kind.value, pop.period.label, pop.period.start, pop.period.end)
    )
    op = plan.operation
    return {
        "stance": plan.stance.value,
        "knowledge_cutoff": plan.knowledge_cutoff,
        "population_period": period,
        "population_snapshots": (
            pop.cohort.rule.value,
            pop.window_start.rule.value if pop.window_start else None,
            pop.window_end.rule.value if pop.window_end else None,
        ),
        "population_filters": sorted(
            (f.column, f.op.value, tuple(sorted(_value(v) for v in f.values)))
            for f in pop.filters
        ),
        "variables": sorted(sources.values(), key=str),
        "grouping": sorted((sources.get(g.variable, g.variable) for g in plan.grouping), key=str),
        "operation": (
            op.kind.value,
            sources.get(op.measure) if op.measure else None,
            sources.get(op.against) if op.against else None,
            sources.get(op.outcome) if op.outcome else None,
        ),
    }


def diff_plans(expected: dict | list, actual: dict | list) -> list[str]:
    """The fields that differ, by name."""
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return ["specs"]
        return sorted({f for e, a in zip(expected, actual, strict=True) for f in diff_plans(e, a)})
    return sorted(k for k in expected if expected[k] != actual.get(k))


# ---------------------------------------------------------------------- cases


@dataclass(frozen=True)
class Expectation:
    """What a correct planner does with a case.

    `kinds` lists every acceptable terminal outcome kind. For a final plan,
    `plans` lists every acceptable plan (for an edit: the plan the edit must
    produce). A clarification or rejection may name the acceptable reasons and
    the concept at issue.
    """

    kinds: frozenset[PlannerOutcomeKind]
    path: PlanPath | None = None
    plans: tuple[AnalysisPlan | InvestigationPlan, ...] = ()
    clarification_reasons: frozenset[ClarificationReason] = frozenset()
    rejection_reasons: frozenset[RejectedReason] = frozenset()
    concept: BusinessConcept | None = None

    @property
    def primary(self) -> PlannerOutcomeKind:
        # A case that accepts a refusal or a clarification is a refusal case.
        for kind in (
            PlannerOutcomeKind.FINAL_PLAN,
            PlannerOutcomeKind.REJECTED,
            PlannerOutcomeKind.CLARIFICATION_REQUEST,
        ):
            if kind in self.kinds:
                return kind
        raise ValueError("an expectation names at least one outcome kind")


@dataclass
class CaseResult:
    case_id: str
    category: str
    expected: PlannerOutcomeKind
    outcome: PlannerOutcomeKind
    path: PlanPath | None
    reason: str | None
    passed: bool
    failures: list[FailureCategory]
    attempt_failures: list[FailureCategory]
    tool_calls: int
    turns: int
    tool_budget: int
    valid_plan: bool = False
    semantic_match: bool = False
    executed: bool | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    mismatched_fields: list[str] = field(default_factory=list)


def _normalise(plan, ctx):
    if isinstance(plan, InvestigationPlan):
        return normalise_investigation(plan, ctx)
    return normalise_analysis(plan, ctx)


def _execute(final: FinalPlan, ctx: surface.ToolContext) -> bool:
    plan = final.executable
    if isinstance(plan, InvestigationPlan):
        run = surface.run_investigation_plan(ctx, plan)
    else:
        run = surface.run_analysis_plan(ctx, plan)
    return run.executed


def evaluate_case(
    case_id: str,
    category: str,
    expectation: Expectation,
    result: PlanningResult,
    ctx: surface.ToolContext,
    *,
    tool_budget: int,
    execute: bool = True,
) -> CaseResult:
    """Compare one planning run with its expectation. Deterministic."""
    outcome = result.outcome
    attempts = sorted(
        {c for code in result.codes() if (c := attempt_category(code)) is not None}
    )
    failures: set[FailureCategory] = set()
    mismatched: list[str] = []
    valid_plan = semantic_match = False
    executed: bool | None = None

    if outcome.kind in expectation.kinds:
        if isinstance(outcome, FinalPlan):
            valid_plan = bool(outcome.validation and outcome.validation.plan_ok)
            expected_path = expectation.path
            if expected_path is PlanPath.INVESTIGATION and outcome.path is not expected_path:
                failures.add(F.WRONG_ANALYSIS_TYPE)
            elif expected_path in (PlanPath.SEMANTIC, PlanPath.EDIT) and (
                outcome.path is PlanPath.INVESTIGATION
            ):
                failures.add(F.INVESTIGATION_BYPASS)
            elif expected_path is PlanPath.EDIT and outcome.path is not PlanPath.EDIT:
                failures.add(F.BAD_EDIT)
            else:
                actual = _normalise(outcome.executable, ctx)
                best: list[str] | None = None
                for plan in expectation.plans:
                    diff = diff_plans(_normalise(plan, ctx), actual)
                    if best is None or len(diff) < len(best):
                        best = diff
                mismatched = best or []
                semantic_match = best == []
                failures.update(FIELD_CATEGORIES.get(f, F.WRONG_ANALYSIS_TYPE) for f in mismatched)
            if execute and valid_plan:
                executed = _execute(outcome, ctx)
                if not executed:
                    failures.add(F.EXECUTION_FAILED)
        elif outcome.kind is PlannerOutcomeKind.CLARIFICATION_REQUEST:
            reason = outcome.request.reason
            allowed = expectation.clarification_reasons
            if allowed and reason not in allowed:
                failures.add(F.WRONG_REASON)
            if expectation.concept and outcome.request.concept not in (None, expectation.concept):
                failures.add(F.WRONG_CONCEPT)
        elif isinstance(outcome, Rejected):
            if not outcome.reason.chosen_by_model:
                failures.add(_run_failure(outcome, attempts))
            elif expectation.rejection_reasons and (
                outcome.reason not in expectation.rejection_reasons
            ):
                failures.add(F.WRONG_REASON)
            if expectation.concept and outcome.concept not in (None, expectation.concept):
                failures.add(F.WRONG_CONCEPT)
    else:
        failures.add(_kind_mismatch(expectation, outcome, attempts, category))

    if result.tool_calls > tool_budget:
        failures.add(F.EXCESSIVE_TOOL_USE)
    turns = result.turns
    return CaseResult(
        case_id=case_id,
        category=category,
        expected=expectation.primary,
        outcome=outcome.kind,
        path=getattr(outcome, "path", None),
        reason=outcome.reason.value if isinstance(outcome, Rejected) else (
            outcome.request.reason.value if hasattr(outcome, "request") else None
        ),
        passed=not failures,
        failures=sorted(failures),
        attempt_failures=attempts,
        tool_calls=result.tool_calls,
        turns=len(turns),
        tool_budget=tool_budget,
        valid_plan=valid_plan,
        semantic_match=semantic_match,
        executed=executed,
        input_tokens=sum(t.input_tokens or 0 for t in turns),
        output_tokens=sum(t.output_tokens or 0 for t in turns),
        mismatched_fields=mismatched,
    )


def _run_failure(outcome: Rejected, attempts: list[FailureCategory]) -> FailureCategory:
    """A rejection the loop imposed: name what drove it."""
    match outcome.reason:
        case RejectedReason.MALFORMED_OUTPUT:
            return F.MALFORMED_OUTPUT
        case RejectedReason.TURN_BUDGET_EXHAUSTED | RejectedReason.CONTEXT_BUDGET_EXHAUSTED:
            return F.EXCESSIVE_TOOL_USE
        case RejectedReason.VALIDATION_FAILED if attempts:
            return attempts[0]
    return F.RUN_FAILED


def _kind_mismatch(expectation, outcome, attempts, category) -> FailureCategory:
    expected = expectation.primary
    got = outcome.kind
    if isinstance(outcome, Rejected) and not outcome.reason.chosen_by_model:
        return _run_failure(outcome, attempts)
    if expected is PlannerOutcomeKind.FINAL_PLAN:
        return (
            F.UNNECESSARY_CLARIFICATION
            if got is PlannerOutcomeKind.CLARIFICATION_REQUEST
            else F.UNNECESSARY_REFUSAL
        )
    if expected is PlannerOutcomeKind.CLARIFICATION_REQUEST:
        return F.MISSING_CLARIFICATION if got is PlannerOutcomeKind.FINAL_PLAN else F.WRONG_REASON
    # A refusal was expected.
    if got is PlannerOutcomeKind.FINAL_PLAN:
        return F.FUTURE_LEAK_ATTEMPT if category == "temporal" else F.MISSING_REFUSAL
    return F.UNNECESSARY_CLARIFICATION


# --------------------------------------------------------------------- report


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else round(numerator / denominator, 4)


@dataclass
class EvaluationReport:
    results: list[CaseResult]

    def _expecting(self, kind: PlannerOutcomeKind) -> list[CaseResult]:
        return [r for r in self.results if r.expected is kind]

    @property
    def metrics(self) -> dict[str, float | int | None]:
        plans = self._expecting(PlannerOutcomeKind.FINAL_PLAN)
        refusals = self._expecting(PlannerOutcomeKind.REJECTED)
        clarifications = self._expecting(PlannerOutcomeKind.CLARIFICATION_REQUEST)
        produced = [r for r in self.results if r.executed is not None]
        within_budget = [r for r in self.results if r.tool_calls <= r.tool_budget]
        leaks = sum(F.FUTURE_LEAK_ATTEMPT in r.attempt_failures for r in self.results)
        return {
            "cases": len(self.results),
            "passed": sum(r.passed for r in self.results),
            "valid_plan_rate": _rate(sum(r.valid_plan for r in plans), len(plans)),
            "semantic_match_rate": _rate(sum(r.semantic_match for r in plans), len(plans)),
            "correct_refusal_rate": _rate(sum(r.passed for r in refusals), len(refusals)),
            "correct_clarification_rate": _rate(
                sum(r.passed for r in clarifications), len(clarifications)
            ),
            "tool_efficiency": _rate(len(within_budget), len(self.results)),
            "mean_tool_calls": _rate(sum(r.tool_calls for r in self.results), len(self.results)),
            "deterministic_execution_success_rate": _rate(
                sum(bool(r.executed) for r in produced), len(produced)
            ),
            "cases_with_future_leak_attempts": leaks,
        }

    @property
    def failure_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(f.value for r in self.results for f in r.failures).items()))

    @property
    def attempt_failure_counts(self) -> dict[str, int]:
        return dict(
            sorted(Counter(f.value for r in self.results for f in r.attempt_failures).items())
        )

    def render(self) -> str:
        """A plain-text report: case ids, kinds, categories, counts. No data values."""
        lines = ["planner evaluation", ""]
        for name, value in self.metrics.items():
            lines.append(f"  {name:<40} {value}")
        lines += ["", "final failures:"]
        lines += [f"  {k:<30} {v}" for k, v in self.failure_counts.items()] or ["  none"]
        lines += ["", "attempt failures (including repaired):"]
        lines += [f"  {k:<30} {v}" for k, v in self.attempt_failure_counts.items()] or ["  none"]
        lines += ["", "cases:"]
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            detail = ",".join(f.value for f in r.failures)
            lines.append(
                f"  {status} {r.case_id:<34} {r.category:<14} expected={r.expected.value:<22} "
                f"got={r.outcome.value:<22} tools={r.tool_calls} {detail}"
            )
        return "\n".join(lines)

