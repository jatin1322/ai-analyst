"""DuckDB connection management.

Read-only connections are the architectural commitment behind the SQL escape
hatch (ARCHITECTURE §7.4). The escape hatch itself is a later milestone, but
the connection mode is built in now so it does not have to be retrofitted.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import duckdb

from ai_analyst.config import Settings, get_settings


class DuckDBStore:
    """Owns connection lifecycle and the canonical-snapshot scan expression."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def _configure(self, conn: duckdb.DuckDBPyConnection) -> None:
        conn.execute(f"SET memory_limit = '{self.settings.duckdb_memory_limit}'")
        conn.execute(f"SET threads = {self.settings.duckdb_threads}")

    @contextmanager
    def connect(
        self, database: str | Path = ":memory:", *, read_only: bool = False
    ) -> Iterator[duckdb.DuckDBPyConnection]:
        """Open a connection, configure it, and always close it.

        `read_only` is ignored for in-memory databases, which DuckDB does not
        allow to be opened read-only.
        """
        db = str(database)
        kwargs = {}
        if db != ":memory:":
            kwargs["read_only"] = read_only
        conn = duckdb.connect(db, **kwargs)
        try:
            # Read-only connections accept SET, and the execution caps in
            # ARCHITECTURE 7.4 apply to them too.
            self._configure(conn)
            yield conn
        finally:
            conn.close()

    def snapshots_scan(self, dataset_id: str) -> str:
        """SQL expression that reads a dataset's canonical partitioned Parquet.

        `as_of` is a Hive partition key, so its type must be declared or DuckDB
        returns it as VARCHAR.
        """
        root = self.settings.canonical_dir(dataset_id).resolve()
        pattern = str(root / "**" / "*.parquet").replace("'", "''")
        return (
            f"read_parquet('{pattern}', hive_partitioning = true, "
            f"hive_types = {{'as_of': DATE}})"
        )

    def canonical_exists(self, dataset_id: str) -> bool:
        root = self.settings.canonical_dir(dataset_id)
        return root.exists() and any(root.rglob("*.parquet"))
