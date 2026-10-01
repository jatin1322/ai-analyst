"""The tiny fixture is the verification instrument for every later milestone.

These tests assert that each hard case from ARCHITECTURE §9.1 is actually
present in the data, so a later refactor cannot quietly remove one.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from ai_analyst.data.ingest import IngestionResult
from ai_analyst.data.store import DuckDBStore
from tests.conftest import TINY_DATASET_ID


def _rows(store: DuckDBStore, sql: str) -> list[tuple]:
    with store.connect() as conn:
        return conn.execute(sql.format(scan=store.snapshots_scan(TINY_DATASET_ID))).fetchall()


def test_fixture_shape(ingested: IngestionResult, store: DuckDBStore):
    rows = _rows(
        store,
        "SELECT COUNT(*), COUNT(DISTINCT opp_id), COUNT(DISTINCT as_of) FROM {scan}",
    )
    assert rows[0] == (40, 8, 6)


def test_opp_001_slips_across_three_quarters(
    ingested: IngestionResult, store: DuckDBStore
):
    rows = _rows(
        store,
        "SELECT as_of, close_date FROM {scan} WHERE opp_id = 'OPP-001' ORDER BY as_of",
    )
    close_dates = [r[1] for r in rows]
    assert close_dates == [
        date(2025, 3, 15),  # Q1
        date(2025, 3, 15),
        date(2025, 5, 15),  # slipped to Q2
        date(2025, 5, 15),
        date(2025, 5, 15),
        date(2025, 8, 15),  # slipped to Q3
    ]


def test_opp_002_is_pulled_in_from_q2_to_q1_and_wins(
    ingested: IngestionResult, store: DuckDBStore
):
    rows = _rows(
        store,
        "SELECT as_of, close_date, is_won FROM {scan} "
        "WHERE opp_id = 'OPP-002' ORDER BY as_of",
    )
    assert rows[0][1] == date(2025, 5, 20)  # Q2 at first
    assert rows[2][1] == date(2025, 3, 25)  # pulled into Q1
    assert rows[2][2] is True


def test_opp_003_changes_segment_mid_life(ingested: IngestionResult, store: DuckDBStore):
    rows = _rows(
        store,
        "SELECT as_of, segment FROM {scan} WHERE opp_id = 'OPP-003' ORDER BY as_of",
    )
    segments = [r[1] for r in rows]
    assert segments == [
        "Mid-Market",
        "Mid-Market",
        "Enterprise",
        "Enterprise",
        "Enterprise",
        "Enterprise",
    ]


def test_opp_004_amount_moves_up_then_down(ingested: IngestionResult, store: DuckDBStore):
    rows = _rows(
        store,
        "SELECT as_of, amount FROM {scan} WHERE opp_id = 'OPP-004' ORDER BY as_of",
    )
    amounts = [r[1] for r in rows]
    assert amounts == [
        Decimal("200000.00"),
        Decimal("200000.00"),
        Decimal("250000.00"),
        Decimal("250000.00"),
        Decimal("250000.00"),
        Decimal("180000.00"),
    ]


def test_opp_005_vanishes_without_closing(ingested: IngestionResult, store: DuckDBStore):
    rows = _rows(
        store,
        "SELECT MAX(as_of), BOOL_OR(is_closed) FROM {scan} WHERE opp_id = 'OPP-005'",
    )
    last_as_of, ever_closed = rows[0]
    assert last_as_of == date(2025, 3, 31)
    assert ever_closed is False


def test_opp_006_is_created_mid_quarter_in_q2(
    ingested: IngestionResult, store: DuckDBStore
):
    rows = _rows(
        store,
        "SELECT MIN(as_of), MIN(created_date), COUNT(*) FROM {scan} "
        "WHERE opp_id = 'OPP-006'",
    )
    first_as_of, created, count = rows[0]
    assert created == date(2025, 4, 20)
    assert first_as_of == date(2025, 5, 1)
    assert count == 2


def test_opp_008_is_created_mid_quarter_in_q1(
    ingested: IngestionResult, store: DuckDBStore
):
    rows = _rows(
        store,
        "SELECT MIN(as_of), MIN(created_date), COUNT(*) FROM {scan} "
        "WHERE opp_id = 'OPP-008'",
    )
    first_as_of, created, count = rows[0]
    assert created == date(2025, 1, 15)
    assert first_as_of == date(2025, 2, 1)
    assert count == 5


def test_terminal_outcomes_are_present(ingested: IngestionResult, store: DuckDBStore):
    rows = _rows(
        store,
        "SELECT opp_id FROM {scan} WHERE as_of = DATE '2025-06-30' AND is_won "
        "ORDER BY opp_id",
    )
    assert [r[0] for r in rows] == ["OPP-002", "OPP-003", "OPP-008"]

    lost = _rows(
        store,
        "SELECT opp_id FROM {scan} WHERE as_of = DATE '2025-06-30' "
        "AND is_closed AND NOT is_won ORDER BY opp_id",
    )
    assert [r[0] for r in lost] == ["OPP-007"]


def test_snapshots_span_two_quarters_while_close_dates_reach_a_third(
    ingested: IngestionResult, store: DuckDBStore
):
    rows = _rows(store, "SELECT MIN(as_of), MAX(as_of), MAX(close_date) FROM {scan}")
    min_as_of, max_as_of, max_close = rows[0]
    assert (min_as_of, max_as_of) == (date(2025, 1, 1), date(2025, 6, 30))
    assert max_close == date(2025, 9, 15)
