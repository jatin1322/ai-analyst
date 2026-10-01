"""The rate primitive (ARCHITECTURE 5.4).

`rate(numerator, denominator, by)` returns a ratio **with both components
exposed**. That is the whole design. "A 42% win rate" is unreadable without
knowing whether it came from 42 of 100 deals or 3 of 7, and an answer that
carries only the ratio has thrown away the part a reader needs to judge it.

Two rules are enforced rather than trusted:

* **A zero denominator yields NULL, never a number.** Division by zero is the
  classic way a rate becomes infinity or a silent crash; here it becomes an
  absent value the caller has to handle.
* **A subset rate cannot exceed 1.** When the numerator population is a subset
  of the denominator population — won out of closed, say — a ratio above 1 is
  arithmetically impossible and means the two predicates overlap wrongly.
  `check_rate` fails the result rather than reporting it.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from pydantic import BaseModel, ConfigDict

from ai_analyst.semantic.sql import build_with, exact_divide

RATE_COLUMNS: tuple[str, ...] = ("numerator", "denominator", "ratio")

# Ratios are reported to six decimal places. Enough to distinguish rates that
# matter, fixed so the same data gives the same digits on every platform, and
# exact because the division never passes through a float.
RATIO_SCALE = 6


@dataclass(frozen=True)
class RateSpec:
    """One ratio: two predicates over the same rows, plus how to group them."""

    numerator_predicate: str
    denominator_predicate: str
    # The value being ratioed. None counts rows instead of summing a measure.
    measure_sql: str | None = None
    by: tuple[str, ...] = ()
    # True when the numerator population is contained in the denominator's, so
    # the ratio is bounded by 1 and may be checked against that.
    numerator_is_subset: bool = True
    numerator_label: str = "numerator"
    denominator_label: str = "denominator"


class RateViolation(AssertionError):
    """A rate came back arithmetically impossible."""


class RateRow(BaseModel):
    """One group's ratio with both components kept."""

    model_config = ConfigDict(frozen=True)

    group: tuple[str, ...] = ()
    numerator: Decimal = Decimal("0")
    denominator: Decimal = Decimal("0")
    ratio: Decimal | None = None

    @property
    def is_defined(self) -> bool:
        return self.ratio is not None


def rate_sql(base: str, spec: RateSpec) -> str:
    """Compile a rate over a base row set.

    `base` is a complete SELECT producing the rows to ratio. The two predicates
    are evaluated against those same rows, so the numerator and denominator can
    never be drawn from differently filtered populations, which is the other
    way a rate quietly lies.

    The division is exact throughout. `exact_divide` keeps it out of binary
    floating point, and a zero denominator produces NULL rather than an error
    or an infinity.
    """
    value = spec.measure_sql or "1"
    operand_scale = 2 if spec.measure_sql else 0
    numerator = f"COALESCE(SUM(CASE WHEN {spec.numerator_predicate} THEN {value} END), 0)"
    denominator = f"COALESCE(SUM(CASE WHEN {spec.denominator_predicate} THEN {value} END), 0)"

    group_select = "".join(f'    "{g}" AS "{g}",\n' for g in spec.by)
    group_by = ", ".join(f'"{g}"' for g in spec.by)

    lines = [
        "SELECT",
        group_select + f"    {numerator} AS numerator,",
        f"    {denominator} AS denominator,",
        # A summed measure is money here (coverage), carried at 2 places; a
        # count carries none. Both are divided exactly, never through `/`.
        f"    {exact_divide(numerator, denominator, RATIO_SCALE, operand_scale)} AS ratio",
        "FROM rate_base",
    ]
    if spec.by:
        lines.append(f"GROUP BY {group_by}")
        lines.append(f"ORDER BY {group_by}")
    return build_with([("rate_base", base)], "\n".join(lines))


def check_rate(rows: list[RateRow], spec: RateSpec) -> None:
    """Assert what must hold of every rate, or fail the result."""
    for row in rows:
        if row.denominator < 0:
            raise RateViolation(
                f"denominator is negative ({row.denominator}) for group {row.group}"
            )
        if row.denominator == 0 and row.ratio is not None:
            raise RateViolation(
                f"group {row.group} has a zero denominator but reports a ratio of "
                f"{row.ratio}; a zero denominator must yield no value"
            )
        if spec.numerator_is_subset and row.numerator > row.denominator:
            raise RateViolation(
                f"group {row.group} has numerator {row.numerator} above denominator "
                f"{row.denominator}; the {spec.numerator_label} population is supposed "
                f"to be contained in the {spec.denominator_label} population, so the "
                "two predicates overlap wrongly"
            )
