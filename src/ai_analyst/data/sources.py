"""Reading a `TableSource`: the DuckDB scan expression, and connection setup.

Every source is read-only, always. Nothing here ever issues `COPY`, `INSERT`,
`CREATE TABLE ... AS`, or any other writing statement against a source; the
only writes in this codebase are to the canonical Parquet layout `ingest.py`
owns, never back to a source.

Credentials are never accepted as an argument and never read from anywhere but
DuckDB's standard AWS credential chain (environment, `~/.aws`, instance role),
exactly like `probe.py`'s original `_connect` did; this module now owns that
pattern so ingestion and the probe share one implementation.
"""

from __future__ import annotations

import os

import duckdb

from ai_analyst.contracts.source import SourceFormat, TableSource
from ai_analyst.data.conform import quote_ident, quote_literal


class ProbeAccessError(RuntimeError):
    """A source could not be reached. Nothing about its contents is claimed."""


def prepare_source(conn: duckdb.DuckDBPyConnection, source: TableSource) -> None:
    """Make an open connection able to read `source`.

    Loads `httpfs` and a credential-chain S3 secret for `s3://` sources, and
    the `delta` extension for Delta sources. Raises `ProbeAccessError` if
    either cannot be set up; the caller then knows nothing was read rather
    than getting a confusing failure later. The caller owns the connection.
    """
    if source.uri.startswith("s3://"):
        try:
            conn.execute("LOAD httpfs")
            # Credentials come from the standard AWS chain, never from an argument.
            # The Delta reader does not resolve a region from a named profile, so
            # the standard region variables are passed through when set.
            region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
            option = f", REGION {quote_literal(region)}" if region else ""
            conn.execute(f"CREATE SECRET (TYPE s3, PROVIDER credential_chain{option})")
        except duckdb.Error as exc:
            raise ProbeAccessError(
                "no usable AWS credentials in the standard chain (environment, "
                "~/.aws, instance role); the source was not read. "
                f"DuckDB said: {str(exc).splitlines()[0]}"
            ) from exc
    if source.format is SourceFormat.DELTA:
        try:
            conn.execute("INSTALL delta")
            conn.execute("LOAD delta")
        except duckdb.Error as exc:
            raise ProbeAccessError(
                f"the DuckDB delta extension could not be installed/loaded: {exc}"
            ) from exc


def connect_for(source: TableSource) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection able to read `source` (see `prepare_source`)."""
    conn = duckdb.connect()
    try:
        prepare_source(conn, source)
    except ProbeAccessError:
        conn.close()
        raise
    return conn


def _table_expression(source: TableSource) -> str:
    literal = quote_literal(source.uri)
    if source.format is SourceFormat.CSV:
        # Read as text and cast explicitly downstream (ARCHITECTURE §8), so a
        # cast failure is a countable event rather than a silent type surprise.
        return f"read_csv({literal}, header = true, all_varchar = true)"
    if source.format is SourceFormat.PARQUET:
        return f"read_parquet({literal})"
    if source.format is SourceFormat.PARQUET_DIR:
        # Recursive: a directory of hive-partitioned files, arbitrarily nested
        # (e.g. as_of_qtr=2025Q1/part-0.parquet), not just one level deep.
        pattern = source.uri.rstrip("/") + "/**/*.parquet"
        return f"read_parquet({quote_literal(pattern)}, hive_partitioning = true)"
    if source.format is SourceFormat.DELTA:
        return f"delta_scan({literal})"
    raise AssertionError(f"unhandled source format: {source.format!r}")  # pragma: no cover


def scan_sql(source: TableSource) -> str:
    """The DuckDB table expression that reads `source`, with any projection applied."""
    table = _table_expression(source)
    if not source.columns:
        return table
    projection = ", ".join(quote_ident(c) for c in source.columns)
    return f"(SELECT {projection} FROM {table})"


__all__ = ["ProbeAccessError", "connect_for", "prepare_source", "scan_sql"]
