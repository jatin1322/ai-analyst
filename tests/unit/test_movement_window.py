"""The close-date movement test, scoped to the export window (ARCHITECTURE 12.19).

`close_date_push_count` is cumulative since the opportunity was created, so its
absolute value describes the opportunity's whole life, not the slice of it the
export contains. These tests pin the distinction, because getting it wrong in
either direction is costly:

* scoping on "count above zero" fails an opportunity whose pushes all happened
  before the first snapshot, reporting a disproof the data does not support;
* dropping the test entirely would lose the one check that distinguishes a
  close date reconstructed from the snapshot's own horizon from one
  reconstructed against the terminal date.

The real production sibling export has no `close_date_push_count` column at
all, so it skips this test and cannot exercise the rescoping. That is why these
run against purpose-built rows.
"""

from __future__ import annotations

import duckdb
import pytest

from ai_analyst.data.reconstruct import run_close_date_movement_test

COLUMNS = {"opp_id", "close_date", "close_date_push_count"}


def _table(rows: list[tuple[str, str, str, int]]) -> duckdb.DuckDBPyConnection:
    """(opp_id, as_of, close_date, push_count) rows as a scannable table."""
    conn = duckdb.connect()
    conn.execute(
        "CREATE TABLE t (opp_id VARCHAR, as_of DATE, close_date DATE, "
        "close_date_push_count BIGINT)"
    )
    for opp, as_of, close, count in rows:
        conn.execute(
            "INSERT INTO t VALUES (?, CAST(? AS DATE), CAST(? AS DATE), ?)",
            [opp, as_of, close, count],
        )
    return conn


def test_a_push_inside_the_window_that_moves_the_date_passes():
    conn = _table(
        [
            ("O-1", "2025-01-01", "2025-03-15", 1),
            ("O-1", "2025-02-01", "2025-05-15", 2),  # counter rose, date moved
        ]
    )
    result = run_close_date_movement_test(conn, "t", COLUMNS)
    assert result.passed
    assert result.checked_rows == 1
    assert result.disagreeing_rows == 0


def test_a_push_inside_the_window_that_leaves_the_date_stuck_fails():
    conn = _table(
        [
            ("O-1", "2025-01-01", "2025-03-15", 1),
            ("O-1", "2025-02-01", "2025-03-15", 2),  # counter rose, date did not
        ]
    )
    result = run_close_date_movement_test(conn, "t", COLUMNS)
    assert result.failed
    assert (result.checked_rows, result.disagreeing_rows) == (1, 1)


def test_a_failure_names_the_snapshot_pair_and_the_counter_values():
    conn = _table(
        [
            ("O-1", "2025-01-01", "2025-03-15", 4),
            ("O-1", "2025-02-01", "2025-03-15", 7),
        ]
    )
    result = run_close_date_movement_test(conn, "t", COLUMNS)
    values = result.samples[0].values
    assert values["opp_id"] == "O-1"
    assert values["from_as_of"] == "2025-01-01"
    assert values["to_as_of"] == "2025-02-01"
    assert values["push_count"] == "4 -> 7"


def test_a_nonzero_counter_that_never_rises_is_skipped_not_failed():
    """The false negative this rescoping exists to remove.

    Five pushes, all of them before the export window. The close date is
    correctly stable inside the window, and the old whole-lifetime scope would
    have called that a disagreement and withheld the concept.
    """
    conn = _table(
        [
            ("O-1", "2025-01-01", "2025-03-15", 5),
            ("O-1", "2025-02-01", "2025-03-15", 5),
            ("O-1", "2025-03-01", "2025-03-15", 5),
        ]
    )
    result = run_close_date_movement_test(conn, "t", COLUMNS)
    assert result.skipped
    assert not result.failed
    assert "cumulative since creation" in result.skip_reason


def test_a_single_snapshot_export_is_skipped():
    conn = _table([("O-1", "2025-01-01", "2025-03-15", 3)])
    result = run_close_date_movement_test(conn, "t", COLUMNS)
    assert result.skipped and not result.failed


def test_a_skipped_movement_test_is_never_a_pass():
    """Skipped is inconclusive, and inconclusive withholds the concept."""
    conn = _table([("O-1", "2025-01-01", "2025-03-15", 5)])
    result = run_close_date_movement_test(conn, "t", COLUMNS)
    assert not result.passed
    assert result.inconclusive


def test_only_the_pairs_where_the_counter_rose_are_checked():
    """A stable date across a pair with no push must not count against it."""
    conn = _table(
        [
            ("O-1", "2025-01-01", "2025-03-15", 1),
            ("O-1", "2025-02-01", "2025-03-15", 1),  # no push: out of scope
            ("O-1", "2025-03-01", "2025-05-15", 2),  # push, and the date moved
        ]
    )
    result = run_close_date_movement_test(conn, "t", COLUMNS)
    assert (result.checked_rows, result.disagreeing_rows) == (1, 0)
    assert result.passed


def test_pairs_are_scoped_per_opportunity_not_across_them():
    conn = _table(
        [
            ("O-1", "2025-01-01", "2025-03-15", 1),
            ("O-1", "2025-02-01", "2025-05-15", 2),
            ("O-2", "2025-01-01", "2025-04-10", 9),
            ("O-2", "2025-02-01", "2025-04-10", 9),
        ]
    )
    # Only O-1 has a within-window push; O-2's flat counter contributes nothing.
    result = run_close_date_movement_test(conn, "t", COLUMNS)
    assert (result.checked_rows, result.disagreeing_rows) == (1, 0)


def test_a_counter_that_decreases_is_not_treated_as_a_push():
    conn = _table(
        [
            ("O-1", "2025-01-01", "2025-03-15", 4),
            ("O-1", "2025-02-01", "2025-03-15", 2),
        ]
    )
    result = run_close_date_movement_test(conn, "t", COLUMNS)
    assert result.skipped


@pytest.mark.parametrize("missing", sorted(COLUMNS))
def test_an_absent_column_skips_rather_than_fails(missing):
    conn = _table([("O-1", "2025-01-01", "2025-03-15", 1)])
    result = run_close_date_movement_test(conn, "t", COLUMNS - {missing})
    assert result.skipped
    assert missing in result.skip_reason


def test_the_comparison_is_normalized_to_calendar_dates():
    """The DATE normalization is load-bearing whenever the close date is a timestamp.

    Conformance writes a DATE, so on a conformed dataset the cast is a no-op.
    It is not a no-op here, and that is the point: given a timestamp-typed close
    date, two rows on the same calendar day but at 00:00 and 11:00 compare
    unequal without it. The test would then report agreement for a push whose
    close date did not actually move, which is the exact false pass this check
    exists to prevent. Removing the cast makes this test fail.
    """
    conn = duckdb.connect()
    conn.execute(
        "CREATE TABLE t (opp_id VARCHAR, as_of DATE, close_date TIMESTAMP, "
        "close_date_push_count BIGINT)"
    )
    conn.execute(
        "INSERT INTO t VALUES "
        "('O-1', DATE '2025-01-01', TIMESTAMP '2025-03-15 00:00:00', 1), "
        "('O-1', DATE '2025-02-01', TIMESTAMP '2025-03-15 11:00:00', 2)"
    )
    result = run_close_date_movement_test(conn, "t", COLUMNS)
    # Same calendar day at both snapshots, so the push did not move the date.
    # Without normalization the differing time of day would read as a move and
    # the test would pass for the wrong reason.
    assert result.failed
    assert result.disagreeing_rows == 1
