"""Text column detection and cataloguing (ARCHITECTURE 5.11).

Catalogue only. No embeddings, no retrieval, no content summarization.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_analyst.config import Settings
from ai_analyst.contracts.profile import DatasetProfile
from ai_analyst.data.ingest import ingest
from ai_analyst.data.profiler import profile_dataset
from ai_analyst.data.store import DuckDBStore

NARRATIVE = (
    "Customer confirmed budget and named an executive sponsor this week. "
    "Security review is the remaining gate before signature, and legal has the "
    "paperwork in flight with a target of month end."
)


@pytest.fixture
def narrative_csv(tmp_path: Path) -> Path:
    path = tmp_path / "narrative.csv"
    header = (
        "snapshot_date,opportunity_id,expected_close_date,sales_stage,"
        "deal_amount,customer_segment,ManagerNotes\n"
    )
    rows = [
        f"2025-01-01,OPP-00{i},2025-03-15,4 - Negotiation,10000.00,Enterprise,"
        f"\"{NARRATIVE} Iteration {i}.\""
        for i in range(1, 6)
    ]
    path.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_short_categorical_columns_are_not_catalogued_as_text(
    tiny_profile: DatasetProfile,
):
    # Segment and stage are dimensions, not narrative.
    catalogued = {t.name for t in tiny_profile.text_columns}
    assert "segment" not in catalogued
    assert "stage" not in catalogued
    assert tiny_profile.text_columns == []


def test_narrative_column_is_detected_and_catalogued(narrative_csv: Path, tmp_path: Path):
    settings = Settings(data_root=tmp_path / "data")
    store = DuckDBStore(settings)
    result = ingest(narrative_csv, "narrative", settings=settings, store=store)
    profile = profile_dataset(
        "narrative", result.schema, settings=settings, store=store
    )

    catalogued = {t.name for t in profile.text_columns}
    # ManagerNotes is not a canonical column; it survives as a discovered one.
    assert "ManagerNotes" in catalogued
    assert "ManagerNotes" in result.schema.discovered_columns

    entry = next(t for t in profile.text_columns if t.name == "ManagerNotes")
    assert entry.mean_length >= settings.text_min_mean_length
    assert entry.distinct_ratio >= settings.text_min_distinct_ratio
    assert entry.max_length > 0
    assert "Catalogued only" in entry.note


def test_catalogue_records_shape_not_content(narrative_csv: Path, tmp_path: Path):
    settings = Settings(data_root=tmp_path / "data")
    store = DuckDBStore(settings)
    result = ingest(narrative_csv, "narrative", settings=settings, store=store)
    profile = profile_dataset("narrative", result.schema, settings=settings, store=store)

    # No field on the catalogue entry may carry narrative content.
    for entry in profile.text_columns:
        serialized = entry.model_dump_json()
        assert "executive sponsor" not in serialized
        assert "Security review" not in serialized


def test_detection_thresholds_are_configurable(narrative_csv: Path, tmp_path: Path):
    strict = Settings(data_root=tmp_path / "data", text_min_mean_length=10_000)
    store = DuckDBStore(strict)
    result = ingest(narrative_csv, "narrative", settings=strict, store=store)
    profile = profile_dataset("narrative", result.schema, settings=strict, store=store)
    assert profile.text_columns == []


def test_text_profile_reports_distinct_ratio(tiny_profile: DatasetProfile):
    from ai_analyst.contracts.profile import TextColumnProfile

    entry = TextColumnProfile(
        name="notes", row_count=10, null_count=2, distinct_count=8,
        mean_length=120.0, max_length=400,
    )
    assert entry.distinct_ratio == 1.0


def test_discovered_columns_are_preserved_not_dropped(
    narrative_csv: Path, tmp_path: Path
):
    # ARCHITECTURE 5.10: a non-canonical column must survive conform so it can
    # be classified. Dropping it at ingest makes classification impossible.
    settings = Settings(data_root=tmp_path / "data")
    store = DuckDBStore(settings)
    result = ingest(narrative_csv, "narrative", settings=settings, store=store)
    assert result.schema.discovered_columns == ["ManagerNotes"]

    with store.connect() as conn:
        columns = conn.execute(
            f"SELECT * FROM {store.snapshots_scan('narrative')} LIMIT 0"
        ).description
    assert "ManagerNotes" in {c[0] for c in columns}


def test_tiny_fixture_has_no_discovered_columns(ingested):
    # Every header in the tiny fixture maps onto the canonical schema.
    assert ingested.schema.discovered_columns == []
