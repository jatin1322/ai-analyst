"""Typed compiled comparisons (ARCHITECTURE 13.7).

A comparison is two operands, an operator, and the kind of value it produces,
compiled to one deterministic expression. The responder may describe a change
only when the engine produced it, and it cites it by token: this is the object
that produces it.

Operands carry their `ValueKind`, and the compiler refuses to compare two
kinds that do not share a unit. Money minus a count, or a ratio of dates, is an
error, never a silent coercion.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from ai_analyst.contracts.result import ValueKind


class ComparisonOperator(StrEnum):
    # left - right, in the operands' own unit.
    DIFFERENCE = "difference"
    # (left - right) / right, as a ratio, through exact division.
    RELATIVE_CHANGE = "relative_change"


class Operand(BaseModel):
    """One side of a comparison: an engine-computed value and what it means."""

    model_config = ConfigDict(frozen=True)

    label: str
    sql: str
    kind: ValueKind
    period_label: str | None = None
    as_of: date | None = None


class CompiledComparison(BaseModel):
    """A comparison reduced to one expression, with its semantics kept."""

    model_config = ConfigDict(frozen=True)

    name: str
    left: Operand
    right: Operand
    operator: ComparisonOperator
    result_kind: ValueKind
    expression_sql: str

    @property
    def summary(self) -> str:
        return (
            f"{self.name} = {self.operator.value}({self.left.label}, {self.right.label}) "
            f"as {self.result_kind.value}"
        )
