"""DuckDB connection management."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from ai_analyst.config import Settings
from ai_analyst.data.store import DuckDBStore


def test_in_memory_connection_works(store: DuckDBStore):
    with store.connect() as conn:
        assert conn.execute("SELECT 42").fetchone()[0] == 42


def test_connection_is_closed_on_exit(store: DuckDBStore):
    with store.connect() as conn:
        pass
    with pytest.raises(duckdb.Error):
        conn.execute("SELECT 1")


def test_settings_are_applied(settings: Settings):
    settings = settings.model_copy(update={"duckdb_threads": 2})
    with DuckDBStore(settings).connect() as conn:
        assert int(conn.execute("SELECT current_setting('threads')").fetchone()[0]) == 2


def test_read_only_connection_rejects_writes(tmp_path: Path, settings: Settings):
    db = tmp_path / "x.duckdb"
    store = DuckDBStore(settings)
    with store.connect(db) as conn:
        conn.execute("CREATE TABLE t AS SELECT 1 AS a")
    with store.connect(db, read_only=True) as conn:
        assert conn.execute("SELECT a FROM t").fetchone()[0] == 1
        with pytest.raises(duckdb.Error):
            conn.execute("CREATE TABLE u AS SELECT 2 AS b")


def test_snapshots_scan_declares_the_partition_key_type(store: DuckDBStore):
    sql = store.snapshots_scan("tiny")
    assert "hive_partitioning = true" in sql
    assert "'as_of': DATE" in sql


def test_canonical_exists_is_false_before_ingestion(store: DuckDBStore):
    assert not store.canonical_exists("never-ingested")
