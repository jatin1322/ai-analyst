"""Status resolution through the ingestion pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_analyst.config import Settings
from ai_analyst.contracts.schema import CanonicalColumn, DerivationRule
from ai_analyst.contracts.status import StatusMapping, StatusStrategy
from ai_analyst.data.ingest import IngestionResult, ingest
from ai_analyst.data.store import DuckDBStore

HEADER = (
    "snapshot_date,opportunity_id,expected_close_date,sales_stage,deal_amount,opp_status\n"
)
ROWS = [
    "2025-01-01,OPP-001,2025-03-15,4 - Negotiation,100000.00,O",
    "2025-01-01,OPP-002,2025-02-20,6 - Order Placed,50000.00,W",
    "2025-01-01,OPP-003,2025-02-28,Closed Lost,60000.00,L",
    "2025-01-01,OPP-004,2025-03-10,SFDCDELETED,40000.00,DELETED",
]


@pytest.fixture
def status_csv(tmp_path: Path) -> Path:
    path = tmp_path / "with_status.csv"
    path.write_text(HEADER + "\n".join(ROWS) + "\n", encoding="utf-8")
    return path


def test_fixture_without_status_falls_back_and_says_so(ingested: IngestionResult):
    resolution = ingested.schema.status_resolution
    assert resolution is not None
    assert resolution.strategy is StatusStrategy.STAGE_KEYWORD
    assert not resolution.is_authoritative
    assert resolution.requires_confirmation
    assert "NOT AUTHORITATIVE" in resolution.note
    assert ingested.schema.requires_confirmation


def test_status_column_is_always_materialised(ingested: IngestionResult):
    assert ingested.schema.has(CanonicalColumn.STATUS)


def test_authoritative_status_column_is_used_when_configured(
    status_csv: Path, tmp_path: Path
):
    settings = Settings(data_root=tmp_path / "data", status_column="opp_status")
    store = DuckDBStore(settings)
    mapping = StatusMapping(
        won_values=("W",), lost_values=("L",), open_values=("O",),
        excluded_values=("DELETED",),
    )
    result = ingest(
        status_csv,
        "with_status",
        mapping_overrides={
            "snapshot_date": CanonicalColumn.AS_OF,
            "opportunity_id": CanonicalColumn.OPP_ID,
            "expected_close_date": CanonicalColumn.CLOSE_DATE,
            "sales_stage": CanonicalColumn.STAGE,
            "deal_amount": CanonicalColumn.AMOUNT,
            "opp_status": CanonicalColumn.STATUS,
        },
        status_mapping=mapping,
        settings=settings,
        store=store,
    )

    resolution = result.schema.status_resolution
    assert resolution.strategy is StatusStrategy.AUTHORITATIVE_COLUMN
    assert result.schema.status_is_authoritative

    with store.connect() as conn:
        rows = dict(
            conn.execute(
                f"SELECT opp_id, status FROM {store.snapshots_scan('with_status')} "
                "ORDER BY opp_id"
            ).fetchall()
        )
    # The authoritative column wins over the stage label in both directions.
    assert rows["OPP-002"] == "won"    # stage 6 - Order Placed reads open
    assert rows["OPP-004"] == "excluded"
    assert rows["OPP-001"] == "open"
    assert rows["OPP-003"] == "lost"


def test_stage_inference_would_have_got_order_placed_wrong(
    status_csv: Path, tmp_path: Path
):
    # Same file, no status column configured: the win becomes open pipeline.
    settings = Settings(data_root=tmp_path / "data2")
    store = DuckDBStore(settings)
    ingest(
        status_csv,
        "no_status",
        mapping_overrides={
            "snapshot_date": CanonicalColumn.AS_OF,
            "opportunity_id": CanonicalColumn.OPP_ID,
            "expected_close_date": CanonicalColumn.CLOSE_DATE,
            "sales_stage": CanonicalColumn.STAGE,
            "deal_amount": CanonicalColumn.AMOUNT,
        },
        settings=settings,
        store=store,
    )
    with store.connect() as conn:
        rows = dict(
            conn.execute(
                f"SELECT opp_id, status FROM {store.snapshots_scan('no_status')}"
            ).fetchall()
        )
    assert rows["OPP-002"] == "open"       # wrong, and this is the point
    assert rows["OPP-004"] == "excluded"   # deleted records still excluded


def test_flags_are_derived_from_status_and_cannot_disagree(
    ingested: IngestionResult, store: DuckDBStore
):
    with store.connect() as conn:
        disagreements = conn.execute(
            f"""
            SELECT COUNT(*) FROM {store.snapshots_scan('tiny')}
            WHERE is_closed <> (status IN ('won', 'lost'))
               OR is_won <> (status = 'won')
            """
        ).fetchone()[0]
    assert disagreements == 0


def test_status_derivation_records_its_provenance(ingested: IngestionResult):
    derived = {d.column: d for d in ingested.schema.derived_columns}
    assert derived[CanonicalColumn.STATUS].rule is DerivationRule.STAGE_KEYWORD
    assert derived[CanonicalColumn.IS_WON].requires_confirmation


def test_tiny_fixture_status_distribution_is_unchanged_by_the_refactor(
    ingested: IngestionResult, store: DuckDBStore
):
    with store.connect() as conn:
        rows = dict(
            conn.execute(
                f"SELECT status, COUNT(*) FROM {store.snapshots_scan('tiny')} "
                "GROUP BY status ORDER BY status"
            ).fetchall()
        )
    assert rows == {"lost": 4, "open": 30, "won": 6}
    assert sum(rows.values()) == 40
