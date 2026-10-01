"""Applying plan edits deterministically (ARCHITECTURE 13.13).

`apply_edit(plan, edit)` is a pure function: the same plan and the same edit
always give the same new plan, apart from its fresh id. It never re-reads the
transcript and never re-interprets the question; it mutates a typed object and
reports the diff. The result must still pass the plan gate in full, because a
filter carried forward is re-validated against the new plan rather than trusted
because it passed last turn.
"""

from __future__ import annotations

from ai_analyst.contracts.plan import AnalysisPlan, AnalysisSpec, new_plan_id
from ai_analyst.contracts.session import (
    AddDimension,
    CarryForwardReport,
    ChangeMetrics,
    ChangePeriod,
    ChangeSnapshotRule,
    ChangeStance,
    EditConflictCode,
    EditedPlan,
    FieldChange,
    PlanEdit,
    RemoveDimension,
    RemoveFilter,
    SetAnalysisOption,
    SetFilter,
    SetLimit,
    SetOrder,
)

# Spec fields compared by the carry-forward report, in reading order.
REPORTED_FIELDS: tuple[str, ...] = (
    "pattern",
    "metrics",
    "period",
    "snapshot",
    "stance",
    "knowledge_cutoff",
    "dimensions",
    "features",
    "filters",
    "attribution",
    "creation_basis",
    "slip_basis",
    "win_rate_basis",
    "order_by",
    "limit",
)
MEANING_FIELDS = frozenset({"metrics", "period", "stance", "pattern"})

# Each single-valued field may be set at most once per edit.
_SINGLE_VALUED = (ChangePeriod, ChangeMetrics, ChangeSnapshotRule, ChangeStance, SetLimit, SetOrder)


class EditConflict(ValueError):
    """An edit that cannot be applied, with a code a caller can branch on."""

    def __init__(self, code: EditConflictCode, message: str) -> None:
        super().__init__(message)
        self.code = code


def _target(plan: AnalysisPlan, edit: PlanEdit) -> AnalysisSpec:
    if edit.spec_id is None:
        if len(plan.specs) != 1:
            raise EditConflict(
                EditConflictCode.AMBIGUOUS_SPEC,
                f"plan {plan.plan_id} has {len(plan.specs)} specs; the edit must name one",
            )
        return plan.specs[0]
    try:
        return plan.spec(edit.spec_id)
    except KeyError as exc:
        raise EditConflict(EditConflictCode.UNKNOWN_SPEC, str(exc)) from exc


def _check_consistency(edit: PlanEdit) -> None:
    """Reject an edit that contradicts itself before touching the plan."""
    for kind in _SINGLE_VALUED:
        if sum(isinstance(o, kind) for o in edit.operations) > 1:
            raise EditConflict(
                EditConflictCode.CONTRADICTORY,
                f"{kind.__name__} appears more than once in one edit",
            )
    added = [o.dimension for o in edit.operations if isinstance(o, AddDimension)]
    removed = [o.dimension for o in edit.operations if isinstance(o, RemoveDimension)]
    if set(added) & set(removed):
        raise EditConflict(
            EditConflictCode.CONTRADICTORY,
            f"dimensions both added and removed: {sorted(set(added) & set(removed))}",
        )
    set_columns = [o.filter.column for o in edit.operations if isinstance(o, SetFilter)]
    removed_columns = [o.column for o in edit.operations if isinstance(o, RemoveFilter)]
    options = [o.option for o in edit.operations if isinstance(o, SetAnalysisOption)]
    if len(options) != len(set(options)):
        raise EditConflict(
            EditConflictCode.CONTRADICTORY, "an analysis option is set more than once"
        )
    duplicates = {c for c in set_columns if set_columns.count(c) > 1}
    both = set(set_columns) & set(removed_columns)
    if duplicates or both:
        raise EditConflict(
            EditConflictCode.CONTRADICTORY,
            f"filters set more than once, or set and removed: {sorted(duplicates | both)}",
        )


def _apply(spec: AnalysisSpec, edit: PlanEdit) -> dict:
    data = spec.model_dump()
    dimensions = list(spec.dimensions)
    filters = list(spec.filters)
    for operation in edit.operations:
        match operation:
            case AddDimension(dimension=d):
                if d in dimensions:
                    raise EditConflict(EditConflictCode.DUPLICATE, f"{d!r} is already a dimension")
                dimensions.append(d)
            case RemoveDimension(dimension=d):
                if d not in dimensions:
                    raise EditConflict(EditConflictCode.NOT_PRESENT, f"{d!r} is not a dimension")
                dimensions.remove(d)
            case SetFilter(filter=f):
                filters = [x for x in filters if x.column != f.column] + [f]
            case RemoveFilter(column=c):
                if not any(x.column == c for x in filters):
                    raise EditConflict(
                        EditConflictCode.NOT_PRESENT, f"no filter on {c!r} to remove"
                    )
                filters = [x for x in filters if x.column != c]
            case ChangePeriod(period=p):
                data["period"] = p
            case ChangeMetrics(metrics=m):
                data["metrics"] = list(m)
            case ChangeSnapshotRule(snapshot=s):
                data["snapshot"] = s
            case ChangeStance(stance=st, knowledge_cutoff=cutoff):
                data["stance"] = st
                data["knowledge_cutoff"] = cutoff
            case SetLimit(limit=n):
                data["limit"] = n
            case SetOrder(order_by=o):
                data["order_by"] = list(o)
            case SetAnalysisOption(option=name, value=v):
                data[name] = v
    data["dimensions"] = dimensions
    data["filters"] = filters
    return data


def _render(value) -> str:
    if hasattr(value, "model_dump"):
        return str(value.model_dump(mode="json", exclude_defaults=True))
    if isinstance(value, list):
        return "[" + ", ".join(_render(v) for v in value) + "]"
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


def diff_specs(before: AnalysisSpec, after: AnalysisSpec) -> tuple[list[str], list[FieldChange]]:
    """Which spec fields were carried and which changed, deterministically."""
    carried: list[str] = []
    changed: list[FieldChange] = []
    for name in REPORTED_FIELDS:
        a, b = getattr(before, name), getattr(after, name)
        if a == b:
            carried.append(name)
        else:
            changed.append(FieldChange(field=name, before=_render(a), after=_render(b)))
    return carried, changed


def apply_edit(plan: AnalysisPlan, edit: PlanEdit) -> EditedPlan:
    """Apply a typed edit to a plan. The new plan names the old one as its parent."""
    if edit.base_plan_id != plan.plan_id:
        raise EditConflict(
            EditConflictCode.BASE_MISMATCH,
            f"the edit targets plan {edit.base_plan_id}, not {plan.plan_id}",
        )
    spec = _target(plan, edit)
    _check_consistency(edit)
    try:
        new_spec = AnalysisSpec.model_validate(_apply(spec, edit))
    except ValueError as exc:
        if isinstance(exc, EditConflict):
            raise
        raise EditConflict(EditConflictCode.INVALID_RESULT, str(exc)) from exc

    new_plan = plan.model_copy(
        update={
            "plan_id": new_plan_id(),
            "parent_plan_id": plan.plan_id,
            "specs": [new_spec if s.id == spec.id else s for s in plan.specs],
        }
    )
    carried, changed = diff_specs(spec, new_spec)
    report = CarryForwardReport(
        base_plan_id=plan.plan_id,
        plan_id=new_plan.plan_id,
        spec_id=spec.id,
        carried=tuple(carried),
        changed=tuple(changed),
        meaning_changes=tuple(c.field for c in changed if c.field in MEANING_FIELDS),
    )
    return EditedPlan(plan=new_plan, report=report)
