"""Result contracts.

ResultSet exists to serve the provenance scanner, which addresses values as
(query_id, row, column). These tests pin that access pattern.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from ai_analyst.contracts.result import (
    ResolvedSnapshot,
    ResultColumn,
    ResultSet,
    SnapshotRule,
    TrustTier,
    new_query_id,
)
from ai_analyst.contracts.schema import DataType


def _result() -> ResultSet:
    return ResultSet(
        query_id="q1",
        columns=[
            ResultColumn(name="segment", dtype=DataType.VARCHAR),
            ResultColumn(name="open_pipeline", dtype=DataType.DECIMAL),
        ],
        rows=[
            ["Enterprise", Decimal("14207500.00")],
            ["Mid-Market", Decimal("3100000.00")],
        ],
    )


def test_cell_lookup_is_by_row_index_and_column_name():
    result = _result()
    assert result.cell(0, "open_pipeline") == Decimal("14207500.00")
    assert result.cell(1, "segment") == "Mid-Market"


def test_cell_lookup_errors_name_the_available_columns():
    result = _result()
    with pytest.raises(KeyError, match="open_pipeline"):
        result.cell(0, "no_such_column")
    with pytest.raises(IndexError, match="out of range"):
        result.cell(5, "segment")


def test_decimal_values_survive_without_becoming_floats():
    result = _result()
    value = result.cell(0, "open_pipeline")
    assert isinstance(value, Decimal)
    assert str(value) == "14207500.00"


def test_ragged_rows_are_rejected():
    with pytest.raises(ValidationError, match="but there are 2 columns"):
        ResultSet(
            query_id="q1",
            columns=[
                ResultColumn(name="a", dtype=DataType.VARCHAR),
                ResultColumn(name="b", dtype=DataType.VARCHAR),
            ],
            rows=[["only-one"]],
        )


def test_row_count_and_emptiness():
    assert _result().row_count == 2
    assert not _result().is_empty
    empty = ResultSet(
        query_id="q1", columns=[ResultColumn(name="a", dtype=DataType.VARCHAR)]
    )
    assert empty.is_empty
    assert empty.row_count == 0


def test_default_trust_tier_is_a():
    assert _result().trust_tier is TrustTier.A


def test_result_carries_resolved_snapshot_dates_not_just_the_rule():
    # ARCHITECTURE §5.1: answers must report the snapshot actually used.
    result = ResultSet(
        query_id="q1",
        columns=[ResultColumn(name="open_pipeline", dtype=DataType.DECIMAL)],
        rows=[[Decimal("1.00")]],
        resolved_snapshots=[
            ResolvedSnapshot(
                rule=SnapshotRule.PERIOD_OPEN,
                requested_boundary=date(2025, 7, 1),
                resolved_as_of=date(2025, 6, 29),
                drift_days=2,
                within_tolerance=True,
            )
        ],
    )
    resolved = result.resolved_snapshots[0]
    assert resolved.resolved_as_of == date(2025, 6, 29)
    assert resolved.drift_days == 2


@pytest.mark.parametrize("bad", ["1q", "q-1", "q.1", "", "a" * 65])
def test_query_id_pattern_is_enforced(bad):
    with pytest.raises(ValidationError):
        ResultSet(query_id=bad, columns=[])


def test_generated_query_ids_are_valid_reference_tokens():
    for _ in range(20):
        ResultSet(query_id=new_query_id(), columns=[])
