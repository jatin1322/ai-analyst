"""Declared stage-to-status map (WP5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_analyst.config import Settings
from ai_analyst.contracts.errors import IngestionError
from ai_analyst.contracts.schema import CanonicalColumn
from ai_analyst.contracts.status import OpportunityStatus, StatusStrategy
from ai_analyst.data.ingest import IngestionResult, ingest
from ai_analyst.data.store import DuckDBStore

HEADER = "snapshot_date,opportunity_id,expected_close_date,sales_stage,deal_amount\n"
ROWS = [
    "2025-01-01,OPP-001,2025-03-15,4 - Negotiation,100.00",
    "2025-01-01,OPP-002,2025-02-20,6 - Order Placed,50.00",
    "2025-01-01,OPP-003,2025-02-28,Closed Lost,60.00",
    "2025-01-01,OPP-004,2025-03-10,SFDCDELETED,40.00",
    "2025-01-01,OPP-005,2025-03-10,not available,30.00",
    "2025-01-01,OPP-006,2025-03-10,4 - negotiation,20.00",
]
OVERRIDES = {
    "snapshot_date": CanonicalColumn.AS_OF,
    "opportunity_id": CanonicalColumn.OPP_ID,
    "expected_close_date": CanonicalColumn.CLOSE_DATE,
    "sales_stage": CanonicalColumn.STAGE,
    "deal_amount": CanonicalColumn.AMOUNT,
}
DECLARED = {
    "4 - Negotiation": OpportunityStatus.OPEN,
    "6 - Order Placed": OpportunityStatus.WON,
    "Closed Lost": OpportunityStatus.LOST,
}


def _run(tmp_path: Path, stage_map=None) -> tuple[IngestionResult, dict[str, str]]:
    csv = tmp_path / "s.csv"
    csv.write_text(HEADER + "\n".join(ROWS) + "\n", encoding="utf-8")
    settings = Settings(data_root=tmp_path / "data")
    store = DuckDBStore(settings)
    result = ingest(
        csv,
        "d",
        mapping_overrides=OVERRIDES,
        stage_status_map=stage_map,
        settings=settings,
        store=store,
    )
    with store.connect() as conn:
        rows = dict(
            conn.execute(
                f"SELECT opp_id, status FROM {store.snapshots_scan('d')} ORDER BY opp_id"
            ).fetchall()
        )
    return result, rows


def test_declared_map_is_authoritative_and_junk_is_excluded_never_open(tmp_path):
    result, rows = _run(tmp_path, DECLARED)
    resolution = result.schema.status_resolution
    assert resolution.strategy is StatusStrategy.DECLARED_STAGE_MAP
    assert resolution.is_authoritative
    assert rows["OPP-002"] == "won"  # keyword fallback would have read this as open
    assert rows["OPP-003"] == "lost"
    assert rows["OPP-001"] == "open"
    # Junk and undeclared values are excluded; no case-folded near match either.
    assert rows["OPP-004"] == rows["OPP-005"] == rows["OPP-006"] == "excluded"
    assert sum(1 for s in rows.values() if s == "open") == 1


def test_unmapped_rows_and_values_are_recorded_as_a_caveat(tmp_path):
    result, _ = _run(tmp_path, DECLARED)
    resolution = result.schema.status_resolution
    assert resolution.unmapped_rows == 3
    assert resolution.unmapped_values == ["4 - negotiation", "SFDCDELETED", "not available"]
    assert resolution.requires_confirmation


def test_no_map_keeps_the_non_authoritative_keyword_fallback(tmp_path):
    result, rows = _run(tmp_path, None)
    resolution = result.schema.status_resolution
    assert resolution.strategy is StatusStrategy.STAGE_KEYWORD
    assert not resolution.is_authoritative
    assert rows["OPP-002"] == "open"  # the known keyword blind spot, unchanged
    assert rows["OPP-004"] == rows["OPP-005"] == "excluded"


def test_map_may_not_map_to_unknown(tmp_path):
    with pytest.raises(IngestionError):
        _run(tmp_path, {"4 - Negotiation": OpportunityStatus.UNKNOWN})
