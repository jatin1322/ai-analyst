"""Lake sources: partitioned Parquet, Delta, and S3 (WP1).

Every source is read-only, so these tests build fixtures with DuckDB's own
`COPY ... TO ... (FORMAT parquet)` rather than any real export, and a
hand-written minimal Delta log rather than a real Delta writer.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import duckdb
import pytest

from ai_analyst.config import Settings
from ai_analyst.contracts.source import SourceFormat, TableSource
from ai_analyst.data.dataset import register_dataset
from ai_analyst.data.probe import ProbeAccessError
from ai_analyst.data.sources import connect_for, scan_sql
from tests.conftest import TINY_CSV


def _delta_available() -> bool:
    """Whether the DuckDB delta extension can be installed/loaded here.

    Tried first, as instructed: only skipped, with a reason, if it genuinely
    cannot be set up in this environment (e.g. no network and nothing cached).
    """
    try:
        conn = duckdb.connect()
        conn.execute("INSTALL delta")
        conn.execute("LOAD delta")
        conn.close()
        return True
    except duckdb.Error:
        return False


DELTA_AVAILABLE = _delta_available()
requires_delta = pytest.mark.skipif(
    not DELTA_AVAILABLE, reason="DuckDB delta extension not installable offline in this environment"
)


# ---------------------------------------------------------------------------
# partitioned parquet directory
# ---------------------------------------------------------------------------


def _write_partitioned(tmp_path: Path) -> Path:
    """Two hive partitions of a small table, written by DuckDB itself."""
    target = tmp_path / "partitioned"
    conn = duckdb.connect()
    conn.execute(
        f"""
        COPY (
            SELECT * FROM (VALUES
                ('2025Q1', 1, 'a'),
                ('2025Q1', 2, 'b'),
                ('2025Q2', 3, 'c')
            ) AS t(as_of_qtr, id, label)
        ) TO '{target}' (FORMAT parquet, PARTITION_BY (as_of_qtr))
        """
    )
    conn.close()
    return target


def test_partitioned_directory_reads_all_rows_with_partition_column(tmp_path):
    directory = _write_partitioned(tmp_path)
    # DuckDB writes one level of hive partitions: as_of_qtr=2025Q1/, .../2025Q2/.
    assert sorted(p.name for p in directory.iterdir()) == ["as_of_qtr=2025Q1", "as_of_qtr=2025Q2"]

    source = TableSource(format=SourceFormat.PARQUET_DIR, uri=str(directory))
    conn = connect_for(source)
    try:
        rows = conn.execute(
            f"SELECT as_of_qtr, id, label FROM {scan_sql(source)} ORDER BY id"
        ).fetchall()
    finally:
        conn.close()
    assert rows == [("2025Q1", 1, "a"), ("2025Q1", 2, "b"), ("2025Q2", 3, "c")]


def test_projection_returns_only_requested_columns(tmp_path):
    directory = _write_partitioned(tmp_path)
    source = TableSource(
        format=SourceFormat.PARQUET_DIR, uri=str(directory), columns=("id", "label")
    )
    conn = connect_for(source)
    try:
        rel = conn.sql(f"SELECT * FROM {scan_sql(source)} LIMIT 0")
        columns = list(rel.columns)
    finally:
        conn.close()
    assert columns == ["id", "label"]


def test_ingests_a_partitioned_parquet_source_end_to_end(settings: Settings, tmp_path):
    # The tiny fixture, physically relocated into a hive-partitioned tree, must
    # ingest to the same row count and profile as the CSV path.
    import csv as csv_module

    rows = list(csv_module.DictReader(TINY_CSV.open(encoding="utf-8")))
    fieldnames = list(rows[0])
    directory = tmp_path / "tiny_partitioned"
    conn = duckdb.connect()
    conn.execute(f"CREATE TABLE t ({', '.join(f'{c} VARCHAR' for c in fieldnames)})")
    placeholders = ", ".join("?" for _ in fieldnames)
    for row in rows:
        conn.execute(f"INSERT INTO t VALUES ({placeholders})", list(row.values()))
    conn.execute(
        f"COPY t TO '{directory}' (FORMAT parquet, PARTITION_BY (snapshot_date))"
    )
    conn.close()

    csv_result = register_dataset(TINY_CSV, "tiny_csv_baseline", settings=settings)

    dir_settings = settings.model_copy(update={"data_root": settings.data_root.parent / "data_dir"})
    source = TableSource(format=SourceFormat.PARQUET_DIR, uri=str(directory))
    dir_result = register_dataset(source, "tiny_partitioned", settings=dir_settings)

    assert dir_result.schema.dataset_id == "tiny_partitioned"
    assert dir_result.profile is not None
    assert csv_result.profile is not None
    csv_by_name = {c.name: c for c in csv_result.profile.columns}
    dir_by_name = {c.name: c for c in dir_result.profile.columns}
    assert set(dir_by_name) == set(csv_by_name)
    for name, dir_col in dir_by_name.items():
        assert dir_col.row_count == csv_by_name[name].row_count, name
        assert dir_col.null_count == csv_by_name[name].null_count, name


# ---------------------------------------------------------------------------
# delta
# ---------------------------------------------------------------------------


def _write_delta(tmp_path: Path) -> Path:
    """A minimal, hand-written Delta table: two commits, one file removed.

    Commit 0 adds file A. Commit 1 removes file A and adds file B. A reader
    that ignores the log and globs the directory would see both files; a
    correct Delta reader sees only file B's rows.
    """
    directory = tmp_path / "delta_table"
    directory.mkdir()
    conn = duckdb.connect()
    file_a, file_b = directory / "fileA.parquet", directory / "fileB.parquet"
    conn.execute(f"COPY (SELECT 1 AS id, 'a' AS label) TO '{file_a}' (FORMAT parquet)")
    conn.execute(f"COPY (SELECT 2 AS id, 'b' AS label) TO '{file_b}' (FORMAT parquet)")
    conn.close()

    log_dir = directory / "_delta_log"
    log_dir.mkdir()
    schema_string = json.dumps(
        {
            "type": "struct",
            "fields": [
                {"name": "id", "type": "integer", "nullable": True, "metadata": {}},
                {"name": "label", "type": "string", "nullable": True, "metadata": {}},
            ],
        }
    )
    now_ms = int(time.time() * 1000)

    def _add(name: str) -> dict:
        return {
            "add": {
                "path": name,
                "size": (directory / name).stat().st_size,
                "partitionValues": {},
                "modificationTime": now_ms,
                "dataChange": True,
            }
        }

    commit_0 = [
        {"protocol": {"minReaderVersion": 1, "minWriterVersion": 2}},
        {
            "metaData": {
                "id": "wp1-test-delta-table",
                "format": {"provider": "parquet", "options": {}},
                "schemaString": schema_string,
                "partitionColumns": [],
                "configuration": {},
                "createdTime": now_ms,
            }
        },
        _add("fileA.parquet"),
    ]
    (log_dir / "00000000000000000000.json").write_text(
        "\n".join(json.dumps(action) for action in commit_0) + "\n", encoding="utf-8"
    )

    commit_1 = [
        {"remove": {"path": "fileA.parquet", "deletionTimestamp": now_ms, "dataChange": True}},
        _add("fileB.parquet"),
    ]
    (log_dir / "00000000000000000001.json").write_text(
        "\n".join(json.dumps(action) for action in commit_1) + "\n", encoding="utf-8"
    )
    return directory


@requires_delta
def test_delta_table_reads_only_the_current_files(tmp_path):
    directory = _write_delta(tmp_path)
    source = TableSource(format=SourceFormat.DELTA, uri=str(directory))
    conn = connect_for(source)
    try:
        rows = conn.execute(f"SELECT id, label FROM {scan_sql(source)} ORDER BY id").fetchall()
    finally:
        conn.close()
    # Only file B's row: file A was removed by the second commit and must
    # never be read, even though its parquet file still sits on disk.
    assert rows == [(2, "b")]


@requires_delta
def test_delta_format_is_inferred_from_delta_log(tmp_path):
    directory = _write_delta(tmp_path)
    assert TableSource.infer(str(directory)).format is SourceFormat.DELTA


# ---------------------------------------------------------------------------
# format inference
# ---------------------------------------------------------------------------


def test_csv_and_parquet_suffixes_are_inferred(tmp_path):
    csv_path = tmp_path / "export.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")
    assert TableSource.infer(str(csv_path)).format is SourceFormat.CSV

    tsv_path = tmp_path / "export.tsv"
    tsv_path.write_text("a\tb\n1\t2\n", encoding="utf-8")
    assert TableSource.infer(str(tsv_path)).format is SourceFormat.CSV

    parquet_path = tmp_path / "export.parquet"
    duckdb.connect().execute(f"COPY (SELECT 1 AS a) TO '{parquet_path}' (FORMAT parquet)")
    assert TableSource.infer(str(parquet_path)).format is SourceFormat.PARQUET


def test_a_bare_local_directory_is_inferred_as_parquet_dir(tmp_path):
    directory = _write_partitioned(tmp_path)
    assert TableSource.infer(str(directory)).format is SourceFormat.PARQUET_DIR


def test_s3_without_a_recognisable_suffix_requires_a_declared_format():
    with pytest.raises(ValueError, match="must declare its format explicitly"):
        TableSource.infer("s3://bucket/prefix/")
    with pytest.raises(ValueError, match="must declare its format explicitly"):
        TableSource.infer("s3://bucket/prefix")


def test_s3_with_a_csv_or_parquet_suffix_is_still_inferred():
    assert TableSource.infer("s3://bucket/export.csv").format is SourceFormat.CSV
    assert TableSource.infer("s3://bucket/export.parquet").format is SourceFormat.PARQUET


def test_a_path_that_is_neither_a_recognised_file_nor_a_directory_cannot_be_inferred(tmp_path):
    with pytest.raises(ValueError, match="cannot infer a format"):
        TableSource.infer(str(tmp_path / "does_not_exist"))


# ---------------------------------------------------------------------------
# s3 credentials
# ---------------------------------------------------------------------------


def test_s3_source_without_credentials_raises_probe_access_error(monkeypatch):
    for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_PROFILE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", "/nonexistent/aws-config-for-test")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", "/nonexistent/aws-credentials-for-test")

    source = TableSource(
        format=SourceFormat.PARQUET, uri="s3://ai-analyst-wp1-nonexistent-bucket/export.parquet"
    )
    try:
        conn = connect_for(source)
    except ProbeAccessError:
        return
    # The standard chain resolved something in this environment (e.g. an
    # instance role) despite the cleared env and file paths. Nothing was
    # written by connect_for either way, so confirm that and skip rather than
    # assert a false negative.
    conn.close()
    pytest.skip("a credential chain resolved in this environment; cannot exercise the failure path")
