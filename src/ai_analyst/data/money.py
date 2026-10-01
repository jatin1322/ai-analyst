"""Monetary handling at the boundary (ARCHITECTURE 12.16).

Two different things happen to a number that might be money, and they are kept
apart on purpose:

* **Structural numeric profiling** describes a column: nulls, distinct values,
  exact extremes. It makes no claim about what the values mean and, for money
  or anything not established as non-money, reports no float mean, median or
  deviation. That lives in `column_profiler`.
* **A monetary analytical measure** is a number an answer is built from. The
  source representation is preserved at rest, and a measure is converted to
  `DECIMAL` explicitly, in visible SQL, *before* anything is aggregated. That is
  this module.

Whether a column is money is decided by classification or by a tenant's concept
declaration, never by numeric shape: two decimal places make a ratio as easily as
an amount, so nothing here infers money from values.
"""

from __future__ import annotations

import duckdb
from pydantic import BaseModel, ConfigDict

from ai_analyst.contracts.concepts import MONETARY_CONCEPTS
from ai_analyst.contracts.schema import DatasetSchema, DataType
from ai_analyst.contracts.tenant import TenantProfile
from ai_analyst.data.conform import quote_ident

MONEY_SCALE = 2
MONEY_PRECISION = 18

# Storage types a monetary measure may be converted from.
_CONVERTIBLE = frozenset({DataType.DECIMAL, DataType.DOUBLE, DataType.BIGINT, DataType.INTEGER})


def money_type(scale: int = MONEY_SCALE) -> str:
    return f"DECIMAL({MONEY_PRECISION},{scale})"


def monetary_measure_sql(
    column: str, dtype: DataType, scale: int = MONEY_SCALE, alias: str | None = None
) -> str:
    """The SQL that resolves a column to a monetary measure.

    A `DECIMAL` column is already exact and passes through untouched. A `DOUBLE`
    or integer column is cast to fixed-scale `DECIMAL` here, so the conversion is
    a declared boundary that appears in the compiled query and binary
    floating-point arithmetic never defines a monetary result. Any other storage
    type cannot hold money and is refused rather than coerced.
    """
    if dtype not in _CONVERTIBLE:
        raise ValueError(
            f"{column!r} is stored as {dtype.value}, which cannot be resolved as a "
            "monetary measure"
        )
    ident = quote_ident(column)
    if alias:
        ident = f"{quote_ident(alias)}.{ident}"
    if dtype is DataType.DECIMAL:
        return ident
    return f"CAST({ident} AS {money_type(scale)})"


class MonetaryConversion(BaseModel):
    """What converting one column to a monetary measure would change."""

    model_config = ConfigDict(frozen=True)

    column: str
    scale: int
    checked_rows: int
    # Rows whose source value has more precision than the scale keeps, so the
    # conversion rounds them. Reported, not hidden.
    altered_rows: int
    # Rows too large for the target precision, which the conversion cannot hold.
    unrepresentable_rows: int

    @property
    def is_lossless(self) -> bool:
        return self.altered_rows == 0 and self.unrepresentable_rows == 0


def measure_monetary_conversion(
    conn: duckdb.DuckDBPyConnection,
    scan: str,
    column: str,
    dtype: DataType,
    scale: int = MONEY_SCALE,
) -> MonetaryConversion:
    """Count the rows the DECIMAL conversion would round or cannot hold."""
    ident = quote_ident(column)
    converted = f"TRY_CAST({ident} AS {money_type(scale)})"
    if dtype is DataType.DECIMAL:
        return MonetaryConversion(
            column=column,
            scale=scale,
            checked_rows=int(
                conn.execute(f"SELECT COUNT({ident}) FROM {scan}").fetchone()[0]
            ),
            altered_rows=0,
            unrepresentable_rows=0,
        )
    checked, altered, unrepresentable = conn.execute(
        f"SELECT COUNT({ident}), "
        f"COUNT(*) FILTER (WHERE {converted} IS NOT NULL "
        f"AND {ident} <> CAST({converted} AS DOUBLE)), "
        f"COUNT(*) FILTER (WHERE {ident} IS NOT NULL AND {converted} IS NULL) "
        f"FROM {scan}"
    ).fetchone()
    return MonetaryConversion(
        column=column,
        scale=scale,
        checked_rows=int(checked),
        altered_rows=int(altered),
        unrepresentable_rows=int(unrepresentable),
    )


def declared_monetary_columns(
    tenant: TenantProfile | None, schema: DatasetSchema
) -> frozenset[str]:
    """Conformed names of the columns a tenant declared as money.

    A tenant declares by its own header, which conformance may have renamed, so
    each source header resolves to the column it became. This is how a tenant's
    declaration reaches the profiler, which runs before any concept binding
    exists. A declaration can only add money, never remove it.
    """
    if tenant is None:
        return frozenset()
    renamed = {m.source_column: m.canonical_column.value for m in schema.mapping.mappings}
    declared: set[str] = set()
    for concept, columns in tenant.concept_columns.items():
        if concept in MONETARY_CONCEPTS:
            declared.update(renamed.get(c, c) for c in columns)
    # A tenant's column classification can declare money too, and only add it.
    family_declared, _ = tenant.expanded_declarations(
        [*(c.name for c in schema.columns), *schema.discovered_columns]
    )
    for declaration in [*tenant.column_classifications, *family_declared]:
        if declaration.is_declared and declaration.classification.is_monetary:
            declared.add(renamed.get(declaration.column, declaration.column))
    return frozenset(declared)
