"""Measuring and validating declared date columns (ARCHITECTURE 12.19).

A column declared as Excel serial dates is converted by `conform.date_sql`. This
module answers the two questions that conversion cannot answer for itself:

* **Did every value convert?** A value that does not is an error, not a null.
  A declared date column that quietly produced nulls would read as missing data
  rather than as a wrong declaration, which is the failure this exists to stop.
* **What was converted?** Counts of serial, ISO and null rows, how many serials
  carried a time-of-day that DATE cannot store, and the date range produced.
  That is the provenance `DatasetSchema.date_conversions` records.

It runs against the raw source, before anything is written.
"""

from __future__ import annotations

import duckdb

from ai_analyst.contracts.errors import IngestionError, InvalidDateValues
from ai_analyst.contracts.schema import DateConversion, DateEncoding
from ai_analyst.data.conform import (
    SERIAL_PATTERN,
    _cleaned,
    date_sql,
    quote_literal,
)

MAX_INVALID_SAMPLES = 5


def _measure_sql(source: str, encoding: DateEncoding) -> str:
    raw = _cleaned(source)
    converted = date_sql(source, encoding)
    serial = f"regexp_matches({raw}, {quote_literal(SERIAL_PATTERN)})"
    number = f"TRY_CAST({raw} AS DOUBLE)"
    return (
        "SELECT "
        f"COUNT(*) FILTER (WHERE {raw} IS NULL), "
        f"COUNT(*) FILTER (WHERE {raw} IS NOT NULL AND {serial} AND {converted} IS NOT NULL), "
        f"COUNT(*) FILTER (WHERE {raw} IS NOT NULL AND NOT {serial} AND {converted} IS NOT NULL), "
        f"COUNT(*) FILTER (WHERE {raw} IS NOT NULL AND {converted} IS NULL), "
        f"COUNT(*) FILTER (WHERE {raw} IS NOT NULL AND {serial} AND {converted} IS NOT NULL "
        f"AND {number} <> FLOOR({number})), "
        f"MIN({converted}), MAX({converted})"
    )


def _invalid_samples(
    conn: duckdb.DuckDBPyConnection, read_expr: str, source: str, encoding: DateEncoding
) -> list[str]:
    raw = _cleaned(source)
    rows = conn.execute(
        f"SELECT DISTINCT {raw} FROM {read_expr} "
        f"WHERE {raw} IS NOT NULL AND {date_sql(source, encoding)} IS NULL "
        f"ORDER BY 1 LIMIT {MAX_INVALID_SAMPLES}"
    ).fetchall()
    return [str(r[0]) for r in rows]


def measure_date_conversions(
    conn: duckdb.DuckDBPyConnection,
    read_expr: str,
    encoded: dict[str, tuple[str, DateEncoding]],
) -> list[DateConversion]:
    """Convert-check every declared date column, raising on the first invalid one.

    `encoded` maps a source column to (conformed column name, encoding).
    """
    conversions: list[DateConversion] = []
    for source, (target, encoding) in encoded.items():
        nulls, serial, iso, invalid, fractional, low, high = conn.execute(
            f"{_measure_sql(source, encoding)} FROM {read_expr}"
        ).fetchone()

        if int(invalid):
            samples = _invalid_samples(conn, read_expr, source, encoding)
            raise IngestionError(
                InvalidDateValues(
                    message=(
                        f"column {source!r} is declared as {encoding.value} dates but "
                        f"{int(invalid)} value(s) are not valid dates, for example "
                        f"{samples}. Not converted and not guessed: fix the declaration "
                        "or the data."
                    ),
                    column=source,
                    encoding=encoding.value,
                    invalid_rows=int(invalid),
                    samples=samples,
                )
            )

        conversions.append(
            DateConversion(
                source_column=source,
                column=target,
                encoding=encoding,
                serial_rows=int(serial),
                iso_rows=int(iso),
                null_rows=int(nulls),
                fractional_rows=int(fractional),
                min_date=low,
                max_date=high,
                note=(
                    "Decoded from Excel serial numbers. Time of day is discarded because "
                    "the conformed type is DATE."
                    if int(serial)
                    else "Declared as Excel serial dates; every value was already ISO."
                ),
            )
        )
    return conversions
