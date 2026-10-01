"""Compiling comparisons between engine-computed values (ARCHITECTURE 13.7).

Two rules:

* **Same unit or nothing.** Operands must share a `ValueKind`. Money compares
  with money, a count with a count, a ratio with a ratio. Anything else raises:
  there is no implicit conversion to make two kinds comparable.
* **Exact arithmetic.** A difference of two DECIMALs stays DECIMAL. A relative
  change goes through `sql.exact_divide`, never `/`, so a monetary comparison
  never passes through binary floating point.
"""

from __future__ import annotations

from ai_analyst.contracts.comparison import (
    ComparisonOperator,
    CompiledComparison,
    Operand,
)
from ai_analyst.contracts.result import ValueKind
from ai_analyst.semantic.sql import CompilationError, exact_divide

# Kinds that support arithmetic comparison. Dates, text and booleans do not.
COMPARABLE_KINDS = frozenset(
    {ValueKind.MONEY, ValueKind.COUNT, ValueKind.QUANTITY, ValueKind.RATIO}
)
RELATIVE_SCALE = 6
# Decimal places each kind carries, so exact division lifts both operands and
# never rounds a denominator to an integer.
OPERAND_SCALE = {
    ValueKind.MONEY: 2,
    ValueKind.COUNT: 0,
    ValueKind.QUANTITY: 6,
    ValueKind.RATIO: 6,
}


class IncompatibleComparison(CompilationError):
    """Two operands that do not share a comparable unit."""


def compile_comparison(
    name: str, left: Operand, right: Operand, operator: ComparisonOperator
) -> CompiledComparison:
    """One typed comparison, or an error. Never a coercion."""
    if left.kind is not right.kind:
        raise IncompatibleComparison(
            f"{name}: cannot compare {left.label} ({left.kind.value}) with "
            f"{right.label} ({right.kind.value}); the units differ"
        )
    if left.kind not in COMPARABLE_KINDS:
        raise IncompatibleComparison(
            f"{name}: {left.kind.value} values do not support arithmetic comparison"
        )
    difference = f"(({left.sql}) - ({right.sql}))"
    match operator:
        case ComparisonOperator.DIFFERENCE:
            expression, kind = difference, left.kind
        case ComparisonOperator.RELATIVE_CHANGE:
            expression = exact_divide(
                difference, f"({right.sql})", RELATIVE_SCALE, OPERAND_SCALE[left.kind]
            )
            kind = ValueKind.RATIO
        case _:  # pragma: no cover - exhaustive over ComparisonOperator
            raise CompilationError(f"unsupported comparison operator {operator}")
    return CompiledComparison(
        name=name,
        left=left,
        right=right,
        operator=operator,
        result_kind=kind,
        expression_sql=expression,
    )
