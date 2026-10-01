"""Type-aware profiling of a single column (ARCHITECTURE 5.15).

Every statistic is computed in DuckDB SQL; no column is materialized into
Python. The profile records what values occur. It never decides what a column
means: the classification, if one is supplied, is only read, to learn which
sentinel values are declared and whether a column is declared to be text.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import duckdb

from ai_analyst.config import Settings
from ai_analyst.contracts.columns import ColumnCategory, ColumnClassification, MonetaryStatus
from ai_analyst.contracts.profile import ColumnProfile, ProfileKind, TextColumnProfile, TopValue
from ai_analyst.contracts.schema import DataType
from ai_analyst.data.conform import quote_ident

NUMERIC_TYPES = frozenset({DataType.BIGINT, DataType.INTEGER, DataType.DOUBLE, DataType.DECIMAL})
# Money is DECIMAL. Averaging it would route an amount through a float.
SUMMARY_STAT_TYPES = frozenset({DataType.BIGINT, DataType.INTEGER, DataType.DOUBLE})

TEXT_NOTE = (
    "Catalogued only. Never a dimension or measure; only null and non-null filters. "
    "Text analysis is a later milestone."
)


def stringify(value: object) -> str | None:
    """Exact string form. Never routes a Decimal through a float."""
    if value is None:
        return None
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _number_literal(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else repr(float(value))


def _top_values(
    conn: duckdb.DuckDBPyConnection, scan: str, ident: str, limit: int
) -> list[TopValue]:
    rows = conn.execute(
        f"SELECT {ident} AS v, COUNT(*) AS n FROM {scan} "
        f"GROUP BY {ident} ORDER BY n DESC, v LIMIT {int(limit)}"
    ).fetchall()
    return [TopValue(value=stringify(r[0]), count=int(r[1])) for r in rows]


def _profile_numeric(
    conn: duckdb.DuckDBPyConnection,
    scan: str,
    name: str,
    dtype: DataType,
    row_count: int,
    sentinels: tuple[float, ...],
    monetary: MonetaryStatus = MonetaryStatus.NON_MONETARY,
) -> ColumnProfile:
    x = quote_ident(name)
    if sentinels:
        listed = ", ".join(_number_literal(v) for v in sentinels)
        observed = f"CASE WHEN {x} IN ({listed}) THEN NULL ELSE {x} END"
        sentinel_count = f"COUNT(*) FILTER (WHERE {x} IN ({listed}))"
    else:
        observed = x
        sentinel_count = "0"

    # Float summary statistics are withheld for money and for anything whose
    # monetary status is unknown (ARCHITECTURE 12.16). A DECIMAL column is
    # excluded by type; a money column stored as DOUBLE is excluded here, which
    # is the case `SUMMARY_STAT_TYPES` alone used to miss.
    summary = dtype in SUMMARY_STAT_TYPES and monetary.allows_float_summary
    stats = (
        f", AVG({observed}), MEDIAN({observed}), STDDEV_SAMP({observed})" if summary else ""
    )
    row = conn.execute(
        f"""
        SELECT COUNT(*) FILTER (WHERE {x} IS NULL),
               {sentinel_count},
               COUNT(DISTINCT {observed}),
               MIN({observed}),
               MAX({observed}),
               COUNT({observed}){stats}
        FROM {scan}
        """
    ).fetchone()
    nulls, sentinels_seen, distinct, low, high, observed_count = row[:6]
    mean = median = stddev = None
    if summary:
        mean, median, stddev = (None if v is None else float(v) for v in row[6:9])

    return ColumnProfile(
        name=name,
        dtype=dtype,
        kind=ProfileKind.EMPTY if int(nulls) == row_count else ProfileKind.NUMERIC,
        row_count=row_count,
        null_count=int(nulls),
        distinct_count=int(distinct),
        min_value=stringify(low),
        max_value=stringify(high),
        mean=mean,
        median=median,
        stddev=stddev,
        sentinel_values=sentinels,
        sentinel_count=int(sentinels_seen),
        observed_count=int(observed_count),
        monetary=monetary,
    )


def _profile_date(
    conn: duckdb.DuckDBPyConnection, scan: str, name: str, dtype: DataType, row_count: int
) -> ColumnProfile:
    x = quote_ident(name)
    nulls, distinct, low, high = conn.execute(
        f"SELECT COUNT(*) FILTER (WHERE {x} IS NULL), COUNT(DISTINCT {x}), MIN({x}), MAX({x}) "
        f"FROM {scan}"
    ).fetchone()
    return ColumnProfile(
        name=name,
        dtype=dtype,
        kind=ProfileKind.EMPTY if int(nulls) == row_count else ProfileKind.DATE,
        row_count=row_count,
        null_count=int(nulls),
        distinct_count=int(distinct),
        min_value=stringify(low),
        max_value=stringify(high),
    )


def _profile_boolean(
    conn: duckdb.DuckDBPyConnection,
    scan: str,
    name: str,
    dtype: DataType,
    row_count: int,
    top_k: int,
) -> ColumnProfile:
    x = quote_ident(name)
    nulls, distinct = conn.execute(
        f"SELECT COUNT(*) FILTER (WHERE {x} IS NULL), COUNT(DISTINCT {x}) FROM {scan}"
    ).fetchone()
    return ColumnProfile(
        name=name,
        dtype=dtype,
        kind=ProfileKind.EMPTY if int(nulls) == row_count else ProfileKind.BOOLEAN,
        row_count=row_count,
        null_count=int(nulls),
        distinct_count=int(distinct),
        top_values=_top_values(conn, scan, x, top_k),
    )


def _profile_string(
    conn: duckdb.DuckDBPyConnection,
    scan: str,
    name: str,
    dtype: DataType,
    row_count: int,
    classification: ColumnClassification | None,
    settings: Settings,
    monetary_override: MonetaryStatus | None = None,
) -> tuple[ColumnProfile, TextColumnProfile | None]:
    x = quote_ident(name)
    nulls, distinct, low, high, mean_len, max_len = conn.execute(
        f"""
        SELECT COUNT(*) FILTER (WHERE {x} IS NULL),
               COUNT(DISTINCT {x}),
               MIN({x}), MAX({x}),
               COALESCE(AVG(LENGTH({x})), 0),
               COALESCE(MAX(LENGTH({x})), 0)
        FROM {scan}
        """
    ).fetchone()
    nulls, distinct, mean_len, max_len = int(nulls), int(distinct), float(mean_len), int(max_len)
    populated = row_count - nulls

    declared_text = classification is not None and classification.category is ColumnCategory.TEXT
    looks_like_text = (
        populated > 0
        and mean_len >= settings.text_min_mean_length
        and distinct / populated >= settings.text_min_distinct_ratio
    )

    if declared_text or looks_like_text:
        profile = ColumnProfile(
            name=name,
            dtype=dtype,
            kind=ProfileKind.TEXT,
            row_count=row_count,
            null_count=nulls,
            distinct_count=distinct,
            mean_length=mean_len,
            max_length=max_len,
        )
        catalogue = TextColumnProfile(
            name=name,
            row_count=row_count,
            null_count=nulls,
            distinct_count=distinct,
            mean_length=mean_len,
            max_length=max_len,
            basis="classification" if declared_text else "detected",
            note=TEXT_NOTE,
        )
        return profile, catalogue

    listable = distinct <= settings.profile_max_top_value_cardinality
    profile = ColumnProfile(
        name=name,
        dtype=dtype,
        kind=ProfileKind.EMPTY if nulls == row_count else ProfileKind.CATEGORICAL,
        row_count=row_count,
        null_count=nulls,
        distinct_count=distinct,
        min_value=stringify(low),
        max_value=stringify(high),
        top_values=_top_values(conn, scan, x, settings.top_k_values) if listable else [],
        high_cardinality=not listable,
    )
    return profile, None


def profile_column(
    conn: duckdb.DuckDBPyConnection,
    scan: str,
    name: str,
    dtype: DataType,
    row_count: int,
    classification: ColumnClassification | None,
    settings: Settings,
    monetary_override: MonetaryStatus | None = None,
) -> tuple[ColumnProfile, TextColumnProfile | None]:
    """Profile one column according to its storage type.

    Returns the profile and, when the column is text, its catalogue entry.
    Sentinels are taken from the classification and only apply to numeric
    columns that declare them; nothing is ever treated as a sentinel by value
    alone.
    """
    if dtype in NUMERIC_TYPES:
        sentinels = classification.sentinels if classification else ()
        # An unclassified column is UNKNOWN, never NON_MONETARY: nothing about
        # it has been established, so it gets no float summary either.
        monetary = (
            classification.monetary_status if classification else MonetaryStatus.UNKNOWN
        )
        # A tenant's declaration that a column is money outranks a classification
        # that said otherwise or nothing at all. It can only add money.
        if monetary_override is MonetaryStatus.MONETARY:
            monetary = MonetaryStatus.MONETARY
        return (
            _profile_numeric(conn, scan, name, dtype, row_count, sentinels, monetary),
            None,
        )
    if dtype is DataType.DATE:
        return _profile_date(conn, scan, name, dtype, row_count), None
    if dtype is DataType.BOOLEAN:
        return _profile_boolean(conn, scan, name, dtype, row_count, settings.top_k_values), None
    return _profile_string(conn, scan, name, dtype, row_count, classification, settings)
