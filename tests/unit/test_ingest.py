"""Ingestion pipeline, including the hard grain assertion."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from ai_analyst.config import Settings
from ai_analyst.contracts.errors import (
    GrainViolation,
    IngestionError,
    MappingError,
    MissingRequiredColumns,
    NullGrainKey,
    UnreadableSource,
)
from ai_analyst.contracts.schema import CanonicalColumn, DerivationRule
from ai_analyst.data.ingest import IngestionResult, ingest, load_schema, read_source_columns
from ai_analyst.data.store import DuckDBStore
from tests.conftest import (
    DUPLICATE_KEY_CSV,
    MISSING_REQUIRED_CSV,
    NULL_KEY_CSV,
    TINY_CSV,
    TINY_DATASET_ID,
)


def test_reads_source_headers_without_ingesting(store: DuckDBStore):
    headers = read_source_columns(TINY_CSV, store)
    assert headers[0] == "snapshot_date"
    assert "annual_recurring_revenue" in headers
    assert len(headers) == 12


def test_missing_file_raises_typed_error(store: DuckDBStore, tmp_path):
    with pytest.raises(IngestionError) as excinfo:
        read_source_columns(tmp_path / "nope.csv", store)
    assert isinstance(excinfo.value.detail, UnreadableSource)
    assert excinfo.value.detail.reason == "not_found"


def test_unsupported_extension_is_rejected(store: DuckDBStore, tmp_path):
    bad = tmp_path / "data.xlsx"
    bad.write_text("x", encoding="utf-8")
    with pytest.raises(IngestionError, match="unsupported source extension"):
        read_source_columns(bad, store)


def test_ingests_the_tiny_fixture(ingested: IngestionResult):
    # Hand-counted from tests/fixtures/tiny/README.md.
    assert ingested.row_count == 40
    assert ingested.snapshot_count == 6
    assert ingested.dataset_id == TINY_DATASET_ID


def test_writes_one_parquet_partition_per_snapshot(
    ingested: IngestionResult, settings: Settings
):
    partitions = sorted(p.name for p in ingested.canonical_path.glob("as_of=*"))
    assert partitions == [
        "as_of=2025-01-01",
        "as_of=2025-02-01",
        "as_of=2025-03-31",
        "as_of=2025-04-01",
        "as_of=2025-05-01",
        "as_of=2025-06-30",
    ]
    assert list(ingested.canonical_path.rglob("part-*.parquet"))


def test_canonical_data_reads_back_with_correct_types(
    ingested: IngestionResult, store: DuckDBStore
):
    with store.connect() as conn:
        row = conn.execute(
            f"""
            SELECT as_of, opp_id, close_date, amount, is_closed, is_won
            FROM {store.snapshots_scan(TINY_DATASET_ID)}
            WHERE opp_id = 'OPP-002' AND as_of = DATE '2025-03-31'
            """
        ).fetchone()
    assert row[0] == date(2025, 3, 31)
    assert row[1] == "OPP-002"
    assert row[2] == date(2025, 3, 25)
    assert row[3] == Decimal("50000.00")
    assert row[4] is True
    assert row[5] is True


def test_money_is_decimal_not_float(ingested: IngestionResult, store: DuckDBStore):
    with store.connect() as conn:
        total = conn.execute(
            f"SELECT SUM(amount) FROM {store.snapshots_scan(TINY_DATASET_ID)}"
        ).fetchone()[0]
    assert isinstance(total, Decimal)


def test_blank_source_cells_become_nulls(ingested: IngestionResult, store: DuckDBStore):
    with store.connect() as conn:
        arr_nulls = conn.execute(
            f"SELECT COUNT(*) FROM {store.snapshots_scan(TINY_DATASET_ID)} "
            "WHERE arr IS NULL"
        ).fetchone()[0]
        region_nulls = conn.execute(
            f"SELECT COUNT(*) FROM {store.snapshots_scan(TINY_DATASET_ID)} "
            "WHERE region IS NULL"
        ).fetchone()[0]
    # OPP-005 has 3 rows with a blank arr; OPP-006 has 2 rows with a blank region.
    assert arr_nulls == 3
    assert region_nulls == 2


def test_flags_are_derived_and_provenance_is_recorded(ingested: IngestionResult):
    schema = ingested.schema
    derived = {d.column: d for d in schema.derived_columns}
    assert derived[CanonicalColumn.IS_CLOSED].rule is DerivationRule.STAGE_KEYWORD
    assert derived[CanonicalColumn.IS_WON].rule is DerivationRule.STAGE_KEYWORD
    assert derived[CanonicalColumn.IS_WON].requires_confirmation
    assert schema.requires_confirmation


def test_derived_flags_agree_with_the_python_classifier(
    ingested: IngestionResult, store: DuckDBStore
):
    # conform.classify_stage and the SQL CASE expression must not drift apart.
    from ai_analyst.data.conform import classify_stage

    with store.connect() as conn:
        rows = conn.execute(
            f"SELECT DISTINCT stage, is_closed, is_won "
            f"FROM {store.snapshots_scan(TINY_DATASET_ID)} ORDER BY stage"
        ).fetchall()
    assert rows, "expected distinct stages"
    for stage, is_closed, is_won in rows:
        assert (is_closed, is_won) == classify_stage(stage), stage


def test_absent_optional_columns_are_not_invented(ingested: IngestionResult):
    # The fixture has no industry, account_id, or probability column.
    assert not ingested.schema.has(CanonicalColumn.INDUSTRY)
    assert not ingested.schema.has(CanonicalColumn.PROBABILITY)
    assert ingested.schema.has(CanonicalColumn.SEGMENT)


def test_schema_is_persisted_and_reloadable(
    ingested: IngestionResult, settings: Settings
):
    assert settings.schema_path(TINY_DATASET_ID).exists()
    assert load_schema(TINY_DATASET_ID, settings) == ingested.schema


def test_raw_file_is_preserved(ingested: IngestionResult, settings: Settings):
    assert (settings.raw_dir(TINY_DATASET_ID) / "snapshots.csv").exists()


def test_duplicate_grain_keys_hard_fail_and_report_the_offenders(
    settings: Settings, store: DuckDBStore
):
    with pytest.raises(IngestionError) as excinfo:
        ingest(DUPLICATE_KEY_CSV, "dup", settings=settings, store=store)

    detail = excinfo.value.detail
    assert isinstance(detail, GrainViolation)
    # duplicate_key.csv duplicates (2025-01-01, OPP-001) and (2025-02-01, OPP-002).
    assert detail.duplicate_key_count == 2
    assert detail.offending_row_count == 4
    offenders = {(s.as_of, s.opp_id) for s in detail.samples}
    assert offenders == {
        (date(2025, 1, 1), "OPP-001"),
        (date(2025, 2, 1), "OPP-002"),
    }


def test_grain_violation_writes_no_canonical_data(settings: Settings, store: DuckDBStore):
    with pytest.raises(IngestionError):
        ingest(DUPLICATE_KEY_CSV, "dup", settings=settings, store=store)
    assert not store.canonical_exists("dup")


def test_a_file_without_a_close_date_or_stage_still_ingests(
    settings: Settings, store: DuckDBStore
):
    # ARCHITECTURE 12.15: the ingestion minimum is the grain. This file has no
    # close date and no stage, and that now makes it a limited dataset rather
    # than an unusable one.
    result = ingest(MISSING_REQUIRED_CSV, "sparse", settings=settings, store=store)
    present = {c.name for c in result.schema.columns}
    assert CanonicalColumn.AS_OF in present and CanonicalColumn.OPP_ID in present
    assert CanonicalColumn.CLOSE_DATE not in present
    assert CanonicalColumn.STAGE not in present
    assert result.row_count > 0


def test_a_file_missing_a_grain_column_still_hard_fails(
    tmp_path, settings: Settings, store: DuckDBStore
):
    # The grain is the one thing that cannot be missing: without it every
    # snapshot metric is silently wrong.
    csv = tmp_path / "no_opp_id.csv"
    csv.write_text("snapshot_date,deal_amount\n2025-01-01,100.00\n", encoding="utf-8")
    with pytest.raises(MappingError) as excinfo:
        ingest(csv, "nograin", settings=settings, store=store)
    detail = excinfo.value.detail
    assert isinstance(detail, MissingRequiredColumns)
    assert detail.missing == [CanonicalColumn.OPP_ID]


def test_unparseable_grain_key_hard_fails(settings: Settings, store: DuckDBStore):
    with pytest.raises(IngestionError) as excinfo:
        ingest(NULL_KEY_CSV, "nullkey", settings=settings, store=store)
    detail = excinfo.value.detail
    assert isinstance(detail, NullGrainKey)
    assert detail.null_as_of_rows == 1


def test_mapping_overrides_are_honoured(settings: Settings, store: DuckDBStore):
    result = ingest(
        TINY_CSV,
        "overridden",
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
    assert result.row_count == 40
    # Nothing was mapped to segment, so it must be absent rather than guessed.
    assert not result.schema.has(CanonicalColumn.SEGMENT)


def test_reingestion_replaces_rather_than_appends(
    ingested: IngestionResult, settings: Settings, store: DuckDBStore
):
    again = ingest(TINY_CSV, TINY_DATASET_ID, settings=settings, store=store)
    assert again.row_count == 40
    with store.connect() as conn:
        count = conn.execute(
            f"SELECT COUNT(*) FROM {store.snapshots_scan(TINY_DATASET_ID)}"
        ).fetchone()[0]
    assert count == 40


def test_genuine_cast_failures_are_measured_against_the_raw_text(
    settings: Settings, store: DuckDBStore
):
    # bad_amount.csv holds one unparseable amount, one unparseable close_date,
    # and one legitimately blank amount. The blank is not a failure.
    from tests.conftest import BAD_VALUES_CSV

    result = ingest(BAD_VALUES_CSV, "bad_values", settings=settings, store=store)
    assert result.schema.cast_failures == {"close_date": 1, "amount": 1}


def test_blank_cells_are_not_counted_as_cast_failures(ingested: IngestionResult):
    # The tiny fixture's three blank arr cells are nulls, not failures.
    assert ingested.schema.cast_failures == {}


def test_ingests_a_parquet_source(settings: Settings, store: DuckDBStore, tmp_path):
    # ARCHITECTURE 7.2 accepts Parquet as well as CSV.
    parquet = tmp_path / "snapshots.parquet"
    with store.connect() as conn:
        conn.execute(
            f"COPY (SELECT * FROM read_csv('{TINY_CSV}', header = true, "
            f"all_varchar = true)) TO '{parquet}' (FORMAT PARQUET)"
        )

    result = ingest(parquet, "from_parquet", settings=settings, store=store)
    assert result.row_count == 40
    assert result.snapshot_count == 6

    with store.connect() as conn:
        amount, as_of = conn.execute(
            f"SELECT amount, as_of FROM {store.snapshots_scan('from_parquet')} "
            "WHERE opp_id = 'OPP-004' AND as_of = DATE '2025-03-31'"
        ).fetchone()
    assert amount == Decimal("250000.00")
    assert as_of == date(2025, 3, 31)


def test_ingests_a_typed_parquet_source(settings: Settings, store: DuckDBStore, tmp_path):
    # Typed Parquet columns take a different path through conform than text
    # does, because casting starts from a real DATE or DECIMAL rather than text.
    parquet = tmp_path / "typed.parquet"
    with store.connect() as conn:
        conn.execute(
            f"""
            COPY (
                SELECT CAST(snapshot_date AS DATE) AS snapshot_date,
                       opportunity_id,
                       CAST(expected_close_date AS DATE) AS expected_close_date,
                       sales_stage,
                       CAST(deal_amount AS DECIMAL(18,2)) AS deal_amount
                FROM read_csv('{TINY_CSV}', header = true, all_varchar = true)
            ) TO '{parquet}' (FORMAT PARQUET)
            """
        )

    result = ingest(parquet, "typed_parquet", settings=settings, store=store)
    assert result.row_count == 40
    assert result.schema.cast_failures == {}

    with store.connect() as conn:
        total = conn.execute(
            f"SELECT SUM(amount) FROM {store.snapshots_scan('typed_parquet')}"
        ).fetchone()[0]
    assert isinstance(total, Decimal)
