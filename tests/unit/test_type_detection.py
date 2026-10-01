"""Type detection for discovered columns.

Each case is a trap: DuckDB rounds instead of failing on a decimal-to-integer
cast, and casts '1' to TRUE, so a naive detector silently corrupts columns.
"""

from __future__ import annotations

import duckdb
import pytest

from ai_analyst.contracts.schema import DataType
from ai_analyst.data.conform import (
    build_type_detection_select,
    decide_type,
    detect_types,
)


def _detect(columns: dict[str, list[str | None]]) -> dict[str, DataType]:
    conn = duckdb.connect()
    names = list(columns)
    length = len(next(iter(columns.values())))
    def cell(name: str, i: int) -> str:
        value = columns[name][i]
        literal = "NULL" if value is None else repr(value)
        return f'CAST({literal} AS VARCHAR) AS "{name}"'

    selects = " UNION ALL ".join(
        "SELECT " + ", ".join(cell(n, i) for n in names) for i in range(length)
    )
    conn.execute(f"CREATE TABLE t AS {selects}")
    counts = conn.execute(build_type_detection_select(names, "t")).fetchone()
    return detect_types(counts, names)


def test_integers_are_bigint():
    assert _detect({"c": ["7", "8", "-999999", None]})["c"] is DataType.BIGINT


def test_a_decimal_is_not_mistaken_for_an_integer():
    # TRY_CAST('1.5' AS BIGINT) is 2. A cast-based detector would round it.
    assert _detect({"c": ["7", "1.5"]})["c"] is DataType.DOUBLE


def test_scientific_notation_is_numeric():
    assert _detect({"c": ["1e-05", "2.5"]})["c"] is DataType.DOUBLE


def test_zero_one_columns_are_numeric_not_boolean():
    # DuckDB casts '1' to TRUE, so trying boolean first would corrupt these.
    assert _detect({"c": ["0", "1", "1"]})["c"] is DataType.BIGINT


def test_only_the_literal_true_and_false_tokens_are_boolean():
    assert _detect({"c": ["true", "FALSE", None]})["c"] is DataType.BOOLEAN
    assert _detect({"c": ["yes", "no"]})["c"] is DataType.VARCHAR


def test_dates_are_detected():
    assert _detect({"c": ["2025-01-31", "2025-02-01", None]})["c"] is DataType.DATE


def test_leading_zeros_keep_a_code_as_text():
    # '0012' cast to a number silently becomes 12.
    assert _detect({"c": ["0012", "0034"]})["c"] is DataType.VARCHAR


def test_one_stray_value_keeps_the_whole_column_text():
    assert _detect({"c": ["1", "2", "n/a"]})["c"] is DataType.VARCHAR


def test_an_all_empty_column_is_text():
    assert _detect({"c": [None, None]})["c"] is DataType.VARCHAR


def test_quarter_labels_are_text_not_dates():
    assert _detect({"c": ["FY2025-Q1", "FY2025-Q2"]})["c"] is DataType.VARCHAR


def test_integers_too_long_for_bigint_fall_back_to_double():
    assert _detect({"c": ["99999999999999999999"]})["c"] is DataType.DOUBLE


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        ((0, 0, 0, 0, 0), DataType.VARCHAR),
        ((3, 3, 3, 0, 0), DataType.BIGINT),
        ((3, 1, 3, 0, 0), DataType.DOUBLE),
        ((3, 0, 0, 3, 0), DataType.BOOLEAN),
        ((3, 0, 0, 0, 3), DataType.DATE),
        ((3, 2, 2, 0, 0), DataType.VARCHAR),
    ],
)
def test_decide_type(counts, expected):
    assert decide_type(*counts) is expected


def test_many_columns_are_typed_from_a_single_scan():
    columns = {f"c{i}": [str(i), str(i + 1)] for i in range(40)}
    assert set(_detect(columns).values()) == {DataType.BIGINT}
