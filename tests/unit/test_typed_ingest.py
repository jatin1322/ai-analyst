"""Typed fast-path ingestion (WP3).

Typed sources (Parquet) keep their types and are cast only where the canonical
type differs; CSV keeps its all-text path. Either way a failed cast stays a
countable event.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from ai_analyst.config import Settings
from ai_analyst.contracts.schema import DataType
from ai_analyst.contracts.source import SourceFormat, TableSource
from ai_analyst.data.ingest import IngestionResult, ingest
from ai_analyst.data.store import DuckDBStore
from ai_analyst.synthetic.feature_store import GeneratorConfig, generate, write_csv, write_parquet


def _ingest(source, tmp_path: Path, dataset_id: str) -> IngestionResult:
    settings = Settings(data_root=tmp_path / "data")
    return ingest(source, dataset_id, settings=settings, store=DuckDBStore(settings))


def _conformed(result: IngestionResult) -> duckdb.DuckDBPyRelation:
    return duckdb.sql(
        f"SELECT * FROM read_parquet('{result.canonical_path.as_posix()}/**/*.parquet')"
    )


def _parquet(tmp_path: Path, select_sql: str, name: str = "typed.parquet") -> Path:
    path = tmp_path / name
    duckdb.execute(f"COPY ({select_sql}) TO '{path.as_posix()}' (FORMAT PARQUET)")
    return path


def test_typed_parquet_matches_csv_conformed_values(tmp_path):
    dataset = generate(
        GeneratorConfig(
            seed=3, n_opportunities=25, n_quarters=1, dst_duplicates_per_date=0, conflict_pairs=0
        )
    )
    csv_path = write_csv(dataset, tmp_path / "s.csv")
    parquet_path = write_parquet(dataset, csv_path, tmp_path / "s.parquet", "base")

    via_csv = _ingest(csv_path, tmp_path, "csv")
    via_parquet = _ingest(parquet_path, tmp_path, "pq")

    assert via_csv.row_count == via_parquet.row_count
    assert via_csv.schema.cast_failures == via_parquet.schema.cast_failures == {}

    def rows(result: IngestionResult):
        rel = _conformed(result)
        cols = ", ".join(f'CAST("{c}" AS VARCHAR)' for c in rel.columns)
        return sorted(
            duckdb.sql(f"SELECT {cols} FROM rel").fetchall(),
            key=lambda r: (r[rel.columns.index("as_of")], r[rel.columns.index("opp_id")]),
        )

    assert _conformed(via_csv).columns == _conformed(via_parquet).columns
    assert rows(via_csv) == rows(via_parquet)


def test_typed_cast_failures_are_counted_exactly(tmp_path):
    path = _parquet(
        tmp_path,
        """
        SELECT * FROM (VALUES
            (DATE '2025-01-01', 'A', '2025-03-01',   CAST(10.50 AS DECIMAL(38,2)), 1),
            (DATE '2025-01-01', 'B', 'not-a-date',   CAST(1e30 AS DECIMAL(38,2)), 2),
            (DATE '2025-01-01', 'C', NULL,           CAST(7.25 AS DECIMAL(38,2)), 3)
        ) AS t(as_of, opp_id, close_date, amount, seats)
        """,
    )
    result = _ingest(TableSource(format=SourceFormat.PARQUET, uri=str(path)), tmp_path, "d")
    # One VARCHAR date that is not a date (text path), one DECIMAL(38,2) that
    # overflows DECIMAL(18,2) (direct path). A genuine NULL is not a failure.
    assert result.schema.cast_failures == {"close_date": 1, "amount": 1}
    # An integer discovered column keeps its type instead of being re-detected.
    assert result.schema.discovered_types["seats"] is DataType.BIGINT


def test_projection_limits_columns_and_keeps_grain(tmp_path):
    path = _parquet(
        tmp_path,
        """
        SELECT DATE '2025-01-01' AS as_of, 'A' AS opp_id, 5.00 AS amount,
               'x' AS keep_me, 'y' AS drop_me
        """,
    )
    source = TableSource(format=SourceFormat.PARQUET, uri=str(path), columns=("amount", "keep_me"))
    result = _ingest(source, tmp_path, "p")
    columns = set(_conformed(result).columns)
    assert "drop_me" not in columns
    assert {"as_of", "opp_id", "amount", "keep_me"} <= columns
    assert result.schema.discovered_columns == ["keep_me"]


def test_csv_path_is_unchanged_all_text_cast(tmp_path):
    csv = tmp_path / "c.csv"
    csv.write_text("as_of,opp_id,amount,seats\n2025-01-01,A,1e30,3\n2025-01-01,B,2.50,x\n")
    result = _ingest(csv, tmp_path, "c")
    assert result.schema.cast_failures == {"amount": 1}
    assert result.schema.discovered_types["seats"] is DataType.VARCHAR


def test_typed_timestamp_as_of_floors_to_date(tmp_path):
    path = _parquet(
        tmp_path,
        "SELECT TIMESTAMP '2025-03-09 23:30:00' AS as_of, 'A' AS opp_id",
    )
    result = _ingest(TableSource(format=SourceFormat.PARQUET, uri=str(path)), tmp_path, "t")
    assert str(_conformed(result).project("as_of").fetchone()[0]) == "2025-03-09"
