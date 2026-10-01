"""The planning loop: a small deterministic state machine (ARCHITECTURE 13.3).

    QUESTION -> PLANNER -> parse
        TOOL_REQUEST  -> budget check -> validate arguments -> run the surface
                         tool under the session scope -> envelope -> PLANNER
        FINAL_PLAN    -> boundary checks -> scope + gate
                         passed: terminal; rejected: back to the planner, bounded
        CLARIFICATION -> prose check, deterministic alternatives: terminal
        REJECTED      -> prose check, deterministic alternatives: terminal

Every budget comes from `Settings`: model turns, inspection calls, gate
repairs, malformed-output retries, conversation tokens, and the size of one
tool result. Exhausting any of them ends the run with a `Rejected` outcome that
says which; nothing is ever executed on a best-effort basis.

The loop is the only place tools run, and it runs them through the existing
surface, so the stance, horizon, grants and row limits apply exactly as they
do everywhere else. A final plan is validated by `validate_analysis_plan` or
`validate_investigation_plan`, the same functions the run tools execute
through, so the planner cannot validate a plan differently from how it will be
run. The loop never executes a plan: a passed plan is returned to the caller.

Boundary checks, before the gate, close what the gate cannot see:

* plan ids and lineage are overwritten, never taken from the model;
* model-written plan assumptions are dropped: text reaches a user only
  through the deterministic renderer;
* a plan with unresolved ambiguities is not final: it must ask instead;
* a numeric filter value or limit must be a number the user wrote (13.6);
* an edit must target the session's active plan, and is applied by the
  deterministic `apply_edit`, never by the model;
* model prose that reaches a user (a clarification, a rejection) may carry no
  numeral the question did not.

The run record is metadata only.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

from pydantic import BaseModel, ValidationError

from ai_analyst.agent.context import PlannerContext
from ai_analyst.agent.planner.model import ModelStopped, ModelUnavailable
from ai_analyst.agent.planner.planner import (
    MalformedOutput,
    Observation,
    Planner,
    validation_summary,
)
from ai_analyst.agent.planner.prompt import SYSTEM_PROMPT, first_message
from ai_analyst.agent.planner.tools import TOOLS, envelope, run_inspection, tool_specs
from ai_analyst.agent.tools import surface
from ai_analyst.config import Settings
from ai_analyst.contracts.answer import RenderedAnswer, Segment, SegmentSource
from ai_analyst.contracts.context import estimate_tokens
from ai_analyst.contracts.investigation import InvestigationPlan
from ai_analyst.contracts.plan import AnalysisPlan, Filter, new_plan_id
from ai_analyst.contracts.planner import (
    ClarificationOutcome,
    FinalPlan,
    PlanningResult,
    PlanPath,
    Rejected,
    RejectedReason,
    ToolRequest,
    TurnRecord,
    TurnStatus,
)
from ai_analyst.contracts.result import TrustTier
from ai_analyst.contracts.session import SetFilter, SetLimit
from ai_analyst.contracts.tools import ClarificationReason
from ai_analyst.session.edits import EditConflict, apply_edit
from ai_analyst.session.state import SessionState
from ai_analyst.validation.provenance import scan, user_values

logger = logging.getLogger("ai_analyst.planner")


class _BoundaryRejection(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def error_envelope(code: str, message: str) -> str:
    """An error returned to the planner, escaped like any tool result."""
    body = json.dumps({"error": code, "message": message}, ensure_ascii=False)
    body = body.replace("<", "\\u003c").replace(">", "\\u003e")
    return f"<tool_error>{body}</tool_error>"


def _numeric(value) -> Decimal | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float | Decimal):
        return Decimal(str(value))
    if isinstance(value, str):
        try:
            return Decimal(value.replace(",", ""))
        except InvalidOperation:
            return None
    return None


def _check_literals(question: str, filters: list[Filter], limit: int | None) -> None:
    """Every number in a filter or limit must be one the user wrote."""
    allowed = user_values(question)
    for item in filters:
        for value in item.values:
            number = _numeric(value)
            if number is not None and number not in allowed and abs(number) not in allowed:
                raise _BoundaryRejection(
                    "invented_literal",
                    f"the filter on {item.column!r} uses a number the question does not "
                    "contain; only numbers the user wrote may appear in a plan",
                )
    if limit is not None and Decimal(limit) not in allowed:
        raise _BoundaryRejection(
            "invented_literal",
            "the limit is not a number the question contains; omit it or ask",
        )


def _check_investigation_literals(question: str, plan: InvestigationPlan) -> None:
    """Bin edges and a numeric reference group are thresholds, so the user's own.

    The minimum group support is a reporting floor, a structural parameter of
    the evidence requirements, and is not checked here.
    """
    allowed = user_values(question)
    numbers = [Decimal(str(edge)) for g in plan.grouping for edge in g.binning.edges]
    if plan.comparison is not None:
        reference = _numeric(plan.comparison.reference.strip().lstrip("<>=! ").strip())
        if reference is not None:
            numbers.append(reference)
    for number in numbers:
        if number not in allowed and abs(number) not in allowed:
            raise _BoundaryRejection(
                "invented_literal",
                "a bin edge or reference value is not a number the question contains",
            )


def _check_prose(question: str, known_dates: frozenset[str], *texts: str) -> None:
    """Model text that reaches a user may carry no numeral the question did not.

    A snapshot date of this dataset is structural, as it is for the responder.
    """
    answer = RenderedAnswer(
        segments=[Segment(text=t, source=SegmentSource.MODEL) for t in texts if t],
        trust_tier=TrustTier.C,
    )
    report = scan(answer, question=question, known_dates=known_dates)
    if not report.ok:
        raise MalformedOutput(
            "unverified_numeral",
            "text for the user may not contain numbers the question did not; "
            "numbers come from the engine",
        )


@dataclass
class _Run:
    turns: list[TurnRecord] = field(default_factory=list)
    tool_calls: int = 0
    repairs: int = 0
    malformed: int = 0
    tokens: int = 0


@dataclass
class PlanningLoop:
    """Drives one planner through one question, under one tool scope."""

    ctx: surface.ToolContext
    settings: Settings
    session: SessionState | None = None

    # ---------------------------------------------------------------- entry

    def run(self, planner: Planner, question: str, context: PlannerContext) -> PlanningResult:
        s = self.settings
        run = _Run()
        opening = first_message(question, context)
        spec_tokens = estimate_tokens(json.dumps([t.input_schema for t in tool_specs()]))
        run.tokens = estimate_tokens(SYSTEM_PROMPT) + estimate_tokens(opening) + spec_tokens
        if run.tokens > s.planner_max_context_tokens:
            return self._finish(
                run,
                Rejected(
                    reason=RejectedReason.CONTEXT_BUDGET_EXHAUSTED,
                    detail="the planning context exceeds its token budget before any turn",
                ),
            )

        def first():
            return planner.plan(question, context)

        step = first
        turn = 0
        while True:
            turn += 1
            if turn > s.planner_max_turns:
                return self._finish(
                    run,
                    Rejected(
                        reason=RejectedReason.TURN_BUDGET_EXHAUSTED,
                        detail=f"no terminal decision within {s.planner_max_turns} model turns",
                    ),
                )
            try:
                outcome = step()
            except MalformedOutput as exc:
                run.malformed += 1
                self._record(
                    run, planner, turn, None, TurnStatus.MALFORMED, (f"malformed:{exc.code}",)
                )
                if run.malformed > s.planner_max_output_retries:
                    return self._finish(
                        run, Rejected(reason=RejectedReason.MALFORMED_OUTPUT, detail=exc.code)
                    )
                observation = Observation(exc.call_id, error_envelope(exc.code, str(exc)), True)
            except ModelStopped as exc:
                self._record(
                    run, planner, turn, None, TurnStatus.STOPPED, (f"stop:{exc.stop_reason}",)
                )
                reason = (
                    RejectedReason.MODEL_REFUSED
                    if exc.stop_reason == "refusal"
                    else RejectedReason.MODEL_TRUNCATED
                )
                return self._finish(run, Rejected(reason=reason, detail=exc.stop_reason))
            except ModelUnavailable:
                self._record(run, planner, turn, None, TurnStatus.STOPPED, ("model_unavailable",))
                return self._finish(run, Rejected(reason=RejectedReason.MODEL_UNAVAILABLE))
            else:
                terminal, observation = self._handle(run, planner, turn, question, outcome)
                if terminal is not None:
                    return self._finish(run, terminal)

            run.tokens += estimate_tokens(observation.content) + self._action_tokens(planner)
            if run.tokens > s.planner_max_context_tokens:
                return self._finish(
                    run,
                    Rejected(
                        reason=RejectedReason.CONTEXT_BUDGET_EXHAUSTED,
                        detail="the planning conversation exceeded its token budget",
                    ),
                )

            def resume(obs=observation):
                return planner.observe(obs)

            step = resume

    # ------------------------------------------------------------- handlers

    def _handle(self, run: _Run, planner: Planner, turn: int, question: str, outcome):
        """One parsed outcome: a terminal result, or the observation to return."""
        call_id = planner.last_action.call_id if planner.last_action else None
        if isinstance(outcome, ToolRequest):
            return None, self._tool(run, planner, turn, outcome)
        if isinstance(outcome, FinalPlan):
            return self._final(run, planner, turn, question, outcome, call_id)
        try:
            if isinstance(outcome, ClarificationOutcome):
                request = outcome.request
                _check_prose(
                    question, self._known_dates(), request.question,
                    *(o.label for o in request.options),
                )
                request = request.model_copy(update={"available_alternatives": []})
                if request.reason is ClarificationReason.CONCEPT_UNAVAILABLE:
                    request = surface.request_clarification(self.ctx, request)
                final = ClarificationOutcome(request=request)
            else:
                _check_prose(question, self._known_dates(), outcome.detail)
                alternatives = []
                if outcome.reason is RejectedReason.CONCEPT_UNAVAILABLE:
                    alternatives = [
                        a
                        for a in surface.available_alternatives(self.ctx)
                        if outcome.concept is None or a != outcome.concept.value
                    ]
                final = outcome.model_copy(update={"available_alternatives": alternatives})
        except MalformedOutput as exc:
            run.malformed += 1
            self._record(
                run, planner, turn, outcome.kind, TurnStatus.MALFORMED, (f"malformed:{exc.code}",)
            )
            if run.malformed > self.settings.planner_max_output_retries:
                return Rejected(reason=RejectedReason.MALFORMED_OUTPUT, detail=exc.code), None
            return None, Observation(call_id, error_envelope(exc.code, str(exc)), True)
        codes = (f"rejected:{final.reason.value}",) if isinstance(final, Rejected) else ()
        self._record(run, planner, turn, final.kind, TurnStatus.ACCEPTED, codes)
        return final, None

    def _tool(self, run: _Run, planner: Planner, turn: int, request: ToolRequest) -> Observation:
        s = self.settings
        tool = TOOLS[request.tool]
        if run.tool_calls >= s.planner_max_tool_calls:
            self._record(run, planner, turn, request.kind, TurnStatus.TOOL_REFUSED,
                         ("tool_budget_exhausted",))
            return Observation(
                request.call_id,
                error_envelope(
                    "tool_budget_exhausted",
                    f"the inspection budget of {s.planner_max_tool_calls} calls is spent; "
                    "decide with what you have: submit, ask, or declare unanswerable",
                ),
                True,
            )
        run.tool_calls += 1
        try:
            args = tool.args.model_validate(request.arguments)
        except ValidationError as exc:
            self._record(run, planner, turn, request.kind, TurnStatus.TOOL_REFUSED,
                         ("invalid_arguments",))
            return Observation(
                request.call_id,
                error_envelope("invalid_arguments", validation_summary(exc)),
                True,
            )
        try:
            result = run_inspection(self.ctx, tool.name, args)
        except (KeyError, ValueError) as exc:
            self._record(run, planner, turn, request.kind, TurnStatus.TOOL_REFUSED, ("tool_error",))
            return Observation(request.call_id, error_envelope("tool_error", str(exc)), True)
        codes = _result_codes(result)
        self._record(run, planner, turn, request.kind, TurnStatus.TOOL_EXECUTED, codes,
                     tool_result_type=type(result).__name__)
        return Observation(
            request.call_id,
            envelope(tool.name, result, s.planner_max_tool_result_tokens),
        )

    def _final(self, run, planner, turn, question, outcome: FinalPlan, call_id):
        try:
            final = self._checked(question, outcome)
        except _BoundaryRejection as exc:
            return self._repair(run, planner, turn, outcome, call_id, (exc.code,), {
                "rejected_before_validation": exc.code, "message": str(exc),
            })
        if final.validation is not None and final.validation.plan_ok:
            self._record(run, planner, turn, final.kind, TurnStatus.ACCEPTED)
            return final, None
        validation = final.validation
        codes = tuple(sorted({r.code.value for r in validation.rejections}))
        return self._repair(
            run, planner, turn, outcome, call_id, codes,
            validation.model_dump(mode="json"), validation=validation,
        )

    def _repair(self, run, planner, turn, outcome, call_id, codes, payload, validation=None):
        run.repairs += 1
        status = TurnStatus.BOUNDARY_REJECTED if validation is None else TurnStatus.GATE_REJECTED
        self._record(run, planner, turn, outcome.kind, status, codes)
        if run.repairs > self.settings.planner_max_repairs:
            return Rejected(
                reason=RejectedReason.VALIDATION_FAILED,
                detail=", ".join(codes),
                validation=validation,
            ), None
        return None, Observation(
            call_id,
            envelope("plan_validation", payload, self.settings.planner_max_tool_result_tokens),
            True,
        )

    # --------------------------------------------------- boundary + the gate

    def _checked(self, question: str, outcome: FinalPlan) -> FinalPlan:
        if outcome.path is PlanPath.INVESTIGATION:
            plan: InvestigationPlan = outcome.investigation.model_copy(
                update={"plan_id": new_plan_id()}
            )
            _check_literals(question, list(plan.population.filters), plan.limit)
            _check_investigation_literals(question, plan)
            checked = surface.validate_investigation_plan(self.ctx, plan)
            return outcome.model_copy(
                update={"investigation": checked.plan, "validation": checked.validation}
            )

        carry_forward = None
        edit = outcome.edit
        if outcome.path is PlanPath.EDIT:
            base = self.session.active if self.session else None
            if not isinstance(base, AnalysisPlan):
                raise _BoundaryRejection(
                    "edit_without_base", "there is no active analysis plan to edit"
                )
            if edit.base_plan_id != base.plan_id:
                raise _BoundaryRejection(
                    "edit_base_mismatch",
                    f"an edit must target the active plan {base.plan_id}",
                )
            added = [o.filter for o in edit.operations if isinstance(o, SetFilter)]
            limits = [o.limit for o in edit.operations if isinstance(o, SetLimit) and o.limit]
            _check_literals(question, added, limits[0] if limits else None)
            try:
                edited = apply_edit(base, edit)
            except EditConflict as exc:
                raise _BoundaryRejection(f"edit_conflict:{exc.code.value}", str(exc)) from exc
            plan, carry_forward = edited.plan, edited.report
        else:
            plan = outcome.plan.model_copy(
                update={"plan_id": new_plan_id(), "parent_plan_id": None, "assumptions": []}
            )
            for spec in plan.specs:
                _check_literals(question, list(spec.filters), spec.limit)
        if plan.unresolved_ambiguities:
            raise _BoundaryRejection(
                "unresolved_ambiguities",
                "a plan with unresolved ambiguities is not final; ask with "
                "request_clarification instead",
            )
        checked = surface.validate_analysis_plan(self.ctx, plan)
        return outcome.model_copy(
            update={
                "plan": checked.plan,
                "edit": edit,
                "carry_forward": carry_forward,
                "validation": checked.validation,
            }
        )

    def _known_dates(self) -> frozenset[str]:
        return frozenset(d.isoformat() for d in self.ctx.snapshots.dates)

    # --------------------------------------------------------------- record

    @staticmethod
    def _action_tokens(planner: Planner) -> int:
        action = planner.last_action
        if action is None or action.arguments is None:
            return 0
        return estimate_tokens(json.dumps(action.arguments, default=str))

    @staticmethod
    def _record(run, planner, turn, kind, status, codes=(), tool_result_type=None) -> None:
        action = planner.last_action
        usage = action.usage if action else None
        record = TurnRecord(
            turn=turn,
            action=action.tool_name if action else None,
            outcome_kind=kind,
            status=status,
            codes=tuple(codes),
            tool_result_type=tool_result_type,
            input_tokens=usage.input_tokens if usage else None,
            output_tokens=usage.output_tokens if usage else None,
            latency_ms=action.latency_ms if action else None,
        )
        run.turns.append(record)
        logger.info("planner_turn %s", record.model_dump_json(exclude_none=True))

    def _finish(self, run: _Run, outcome) -> PlanningResult:
        result = PlanningResult(
            outcome=outcome,
            turns=tuple(run.turns),
            tool_calls=run.tool_calls,
            repairs=run.repairs,
            malformed_outputs=run.malformed,
            context_tokens=run.tokens,
        )
        logger.info(
            "planner_result %s",
            json.dumps(
                {
                    "kind": result.kind.value,
                    "turns": len(run.turns),
                    "tool_calls": run.tool_calls,
                    "repairs": run.repairs,
                    "malformed": run.malformed,
                    "reason": getattr(outcome, "reason", None) and outcome.reason.value,
                }
            ),
        )
        return result


def _result_codes(result: BaseModel) -> tuple[str, ...]:
    """Refusals a tool result carries, as codes: what the scope withheld."""
    codes = []
    rejection = getattr(result, "rejection", None)
    if rejection is not None:
        codes.append(f"withheld:{rejection.value}")
    if getattr(result, "values_withheld_reason", ""):
        codes.append("withheld:values")
    excluded = getattr(result, "excluded", None)
    if excluded:
        codes.append("withheld:columns")
    verdict = getattr(result, "verdict", None)
    if verdict is not None and hasattr(verdict, "value"):
        codes.append(f"materiality:{verdict.value}")
    return tuple(codes)
