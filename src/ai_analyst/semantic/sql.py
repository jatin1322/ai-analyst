"""SQL building helpers shared by every semantic primitive.

Nothing here reaches an LLM and nothing here holds business semantics beyond
what ARCHITECTURE 5.3 already fixes. Literals are always rendered through
`literal()` so a filter value can never become an injection.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from ai_analyst.contracts.plan import Attribution, Filter, FilterOp
from ai_analyst.contracts.result import SnapshotRule
from ai_analyst.contracts.schema import CanonicalColumn, DatasetSchema
from ai_analyst.data.conform import quote_ident
from ai_analyst.semantic.calendar import ResolvedPeriod


class CompilationError(ValueError):
    """Raised when a plan cannot be compiled deterministically."""


def literal(value: object) -> str:
    """Render a Python value as a DuckDB literal."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, date):
        return f"DATE '{value.isoformat()}'"
    if isinstance(value, (int, Decimal)):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"


def filter_sql(item: Filter, alias: str | None = None) -> str:
    """Render one plan filter as a SQL predicate."""
    column = quote_ident(item.column)
    if alias:
        column = f"{quote_ident(alias)}.{column}"

    match item.op:
        case FilterOp.IS_NULL:
            return f"{column} IS NULL"
        case FilterOp.IS_NOT_NULL:
            return f"{column} IS NOT NULL"
        case FilterOp.EQ:
            return f"{column} = {literal(item.values[0])}"
        case FilterOp.NE:
            return f"{column} IS DISTINCT FROM {literal(item.values[0])}"
        case FilterOp.GT:
            return f"{column} > {literal(item.values[0])}"
        case FilterOp.GTE:
            return f"{column} >= {literal(item.values[0])}"
        case FilterOp.LT:
            return f"{column} < {literal(item.values[0])}"
        case FilterOp.LTE:
            return f"{column} <= {literal(item.values[0])}"
        case FilterOp.BETWEEN:
            low, high = item.values
            return f"{column} BETWEEN {literal(low)} AND {literal(high)}"
        case FilterOp.IN:
            rendered = ", ".join(literal(v) for v in item.values)
            return f"{column} IN ({rendered})"
        case FilterOp.NOT_IN:
            rendered = ", ".join(literal(v) for v in item.values)
            return f"({column} IS NULL OR {column} NOT IN ({rendered}))"
        case _:  # pragma: no cover - exhaustive over FilterOp
            raise CompilationError(f"unsupported filter op {item.op}")


def filters_sql(items: list[Filter], alias: str | None = None) -> str:
    """Conjoin plan filters. Returns TRUE when there are none."""
    if not items:
        return "TRUE"
    return " AND ".join(f"({filter_sql(f, alias)})" for f in items)


def close_date_in_period(period: ResolvedPeriod, alias: str | None = None) -> str:
    column = quote_ident(CanonicalColumn.CLOSE_DATE.value)
    if alias:
        column = f"{quote_ident(alias)}.{column}"
    return f"{column} BETWEEN {literal(period.start)} AND {literal(period.end)}"


def created_in_period(
    period: ResolvedPeriod, schema: DatasetSchema, alias: str | None = None
) -> str:
    """Creation predicate for `creation_basis = created_date`."""
    if not schema.has(CanonicalColumn.CREATED_DATE):
        raise CompilationError(
            "creation_basis 'created_date' needs a created_date column, "
            "which this dataset lacks; use 'first_seen' instead"
        )
    column = quote_ident(CanonicalColumn.CREATED_DATE.value)
    if alias:
        column = f"{quote_ident(alias)}.{column}"
    return f"{column} BETWEEN {literal(period.start)} AND {literal(period.end)}"


def snapshot_rows_cte(scan: str, as_of: date, extra_predicate: str = "TRUE") -> str:
    """One snapshot's rows, as a CTE body."""
    return (
        f"SELECT * FROM {scan} "
        f"WHERE as_of = {literal(as_of)} AND ({extra_predicate})"
    )


def attribution_rule(attribution: Attribution) -> SnapshotRule:
    """Which snapshot supplies dimension values (ARCHITECTURE 5.3 ambiguity 5)."""
    match attribution:
        case Attribution.PERIOD_OPEN:
            return SnapshotRule.PERIOD_OPEN
        case Attribution.LATEST:
            return SnapshotRule.LATEST
        case Attribution.AT_CLOSE:
            return SnapshotRule.PERIOD_CLOSE
        case _:  # pragma: no cover - exhaustive over Attribution
            raise CompilationError(f"unsupported attribution {attribution}")


def indent(sql: str, spaces: int = 4) -> str:
    pad = " " * spaces
    return "\n".join(pad + line if line.strip() else line for line in sql.splitlines())


def build_with(ctes: list[tuple[str, str]], body: str) -> str:
    """Assemble a readable WITH clause. Order is preserved for determinism."""
    if not ctes:
        return body
    rendered = ",\n".join(f"{name} AS (\n{indent(sql)}\n)" for name, sql in ctes)
    return f"WITH {rendered}\n{body}"


def exact_divide(
    numerator: str, denominator: str, scale: int = 2, operand_scale: int = 0
) -> str:
    """Divide without ever touching binary floating point.

    DuckDB's `/` returns DOUBLE for every operand type, DECIMAL included, so
    `SUM(amount) / COUNT(*)` silently converts an exact monetary total into a
    float. That is the conversion ARCHITECTURE 12.16 forbids, and it is
    invisible: the number looks right until it is summed or compared.

    The route here is exact end to end. Scale the numerator by a power of ten
    and cast to HUGEINT, use integer division (`//`, exact), then bring the
    result back down by multiplying by a DECIMAL power of ten, which stays
    DECIMAL. The quotient truncates at `scale` digits rather than rounding,
    which is deterministic and documented rather than platform-dependent.

    `operand_scale` is the number of decimal places the operands carry. Both
    sides are lifted by it before the integer cast, so a denominator of
    `50000.50` is divided as `5000050`, not rounded to `50001`. Money is 2.

    A zero denominator yields NULL through NULLIF, never an error.
    """
    if scale < 0 or operand_scale < 0:
        raise CompilationError(f"scales must not be negative, got {scale}, {operand_scale}")
    factor = 10**scale
    lift = 10**operand_scale
    scaled = f"CAST(({numerator}) * {factor * lift} AS HUGEINT)"
    divisor = (
        f"NULLIF(CAST(({denominator}) * {lift} AS HUGEINT), 0)"
        if operand_scale
        else f"NULLIF(CAST(({denominator}) AS HUGEINT), 0)"
    )
    unit = literal(Decimal(1) / Decimal(factor))
    return (
        f"(CAST({scaled} // {divisor} AS DECIMAL(38,0))"
        f" * CAST({unit} AS DECIMAL(38,{scale})))"
    )
