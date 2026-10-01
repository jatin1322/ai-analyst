"""Session contracts: typed plan edits and the carry-forward report (ARCHITECTURE 13.13).

A session holds plans, not a transcript. A follow-up such as "break that down
by owner" is a `PlanEdit` against the previous plan, applied deterministically,
and the new plan names its parent. What was carried forward is computed by
diffing the two plans, never narrated by a model, so the answer can show it.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from ai_analyst.contracts.plan import (
    AnalysisPlan,
    AnalysisStance,
    Filter,
    OrderSpec,
    Period,
    SnapshotSelection,
)


class _Op(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class AddDimension(_Op):
    op: Literal["add_dimension"] = "add_dimension"
    dimension: str


class RemoveDimension(_Op):
    op: Literal["remove_dimension"] = "remove_dimension"
    dimension: str


class SetFilter(_Op):
    """Add a filter, or replace the existing filter on the same column."""

    op: Literal["set_filter"] = "set_filter"
    filter: Filter


class RemoveFilter(_Op):
    op: Literal["remove_filter"] = "remove_filter"
    column: str


class ChangePeriod(_Op):
    op: Literal["change_period"] = "change_period"
    period: Period


class ChangeMetrics(_Op):
    op: Literal["change_metrics"] = "change_metrics"
    metrics: list[str] = Field(min_length=1)


class ChangeSnapshotRule(_Op):
    op: Literal["change_snapshot_rule"] = "change_snapshot_rule"
    snapshot: SnapshotSelection


class ChangeStance(_Op):
    op: Literal["change_stance"] = "change_stance"
    stance: AnalysisStance
    knowledge_cutoff: date | None = None


class SetLimit(_Op):
    op: Literal["set_limit"] = "set_limit"
    limit: int | None = Field(default=None, ge=1)


class SetAnalysisOption(_Op):
    """Set one of the documented 5.3 ambiguity resolutions on a spec.

    The option names are fixed; the value is validated against the spec's own
    enum when the edit is applied, so an invalid value is an edit conflict.
    """

    op: Literal["set_analysis_option"] = "set_analysis_option"
    option: Literal["creation_basis", "slip_basis", "win_rate_basis", "rate_key", "attribution"]
    value: str


class SetOrder(_Op):
    op: Literal["set_order"] = "set_order"
    order_by: list[OrderSpec]


type EditOperation = Annotated[
    AddDimension
    | RemoveDimension
    | SetFilter
    | RemoveFilter
    | ChangePeriod
    | ChangeMetrics
    | ChangeSnapshotRule
    | ChangeStance
    | SetLimit
    | SetOrder
    | SetAnalysisOption,
    Field(discriminator="op"),
]


class PlanEdit(BaseModel):
    """A follow-up, as typed operations on a named prior plan."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    base_plan_id: str
    # Which spec to edit. May be omitted when the base plan has exactly one.
    spec_id: str | None = None
    operations: list[EditOperation] = Field(min_length=1)


class EditConflictCode(StrEnum):
    BASE_MISMATCH = "base_mismatch"
    AMBIGUOUS_SPEC = "ambiguous_spec"
    UNKNOWN_SPEC = "unknown_spec"
    DUPLICATE = "duplicate"
    NOT_PRESENT = "not_present"
    CONTRADICTORY = "contradictory"
    INVALID_RESULT = "invalid_result"


class FieldChange(BaseModel):
    model_config = ConfigDict(frozen=True)

    field: str
    before: str
    after: str


class CarryForwardReport(BaseModel):
    """What a follow-up kept, and what it changed. Computed by diffing plans."""

    model_config = ConfigDict(frozen=True)

    base_plan_id: str
    plan_id: str
    spec_id: str
    carried: tuple[str, ...]
    changed: tuple[FieldChange, ...]
    # Changes that alter what the question means, not just how it is cut:
    # stance, metric, period. Surfaced as changes, never as carry-forward.
    meaning_changes: tuple[str, ...] = ()

    def render(self) -> str:
        lines = [f"edit on {self.base_plan_id} -> {self.plan_id}"]
        if self.carried:
            lines.append(f"  carried: {', '.join(self.carried)}")
        for change in self.changed:
            lines.append(f"  changed {change.field}: {change.before} -> {change.after}")
        if self.meaning_changes:
            lines.append(
                "  the question's meaning changed: " + ", ".join(self.meaning_changes)
            )
        return "\n".join(lines)


class EditedPlan(BaseModel):
    """The result of applying an edit: the new plan and its report."""

    plan: AnalysisPlan
    report: CarryForwardReport
