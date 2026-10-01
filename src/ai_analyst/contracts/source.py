"""Table sources: where a snapshot table physically lives (WP1: lake sources).

A `TableSource` names a format and a location, nothing more. Reading it is
`data.sources`'s job; this module only describes what to read, and how that
description is inferred -- deterministically, never by sniffing file content.

Format inference for a local path is decided by shape alone: a `_delta_log/`
subdirectory means Delta, any other directory means a partitioned Parquet
tree, and a `.csv`/`.tsv`/`.parquet` suffix means exactly what it says. An
`s3://` URI is never inferred beyond its suffix, because there is no cheap,
read-only way to tell a partitioned Parquet prefix from a Delta table without
first reading it: the caller must declare the format explicitly.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class SourceFormat(StrEnum):
    CSV = "csv"
    PARQUET = "parquet"
    PARQUET_DIR = "parquet_dir"
    DELTA = "delta"


_FILE_SUFFIX_FORMATS: dict[str, SourceFormat] = {
    ".csv": SourceFormat.CSV,
    ".tsv": SourceFormat.CSV,
    ".parquet": SourceFormat.PARQUET,
}


class TableSource(BaseModel):
    """One readable table: its format, its location, and an optional projection."""

    model_config = ConfigDict(frozen=True)

    format: SourceFormat
    uri: str
    # Read only these columns when set. None reads every column.
    columns: tuple[str, ...] | None = None

    @classmethod
    def infer(cls, uri: str, *, columns: tuple[str, ...] | None = None) -> TableSource:
        """Infer the format from `uri`'s shape alone. Raises when it cannot.

        Never guesses from file content. For `s3://` URIs, only a `.csv` or
        `.parquet` suffix is recognised; anything else (a bare prefix, a
        partitioned tree, a Delta table) must be declared with `TableSource(...)`
        directly, because telling those apart without reading the object store
        would mean fetching data to decide how to read data.
        """
        return cls(format=_infer_format(uri), uri=uri, columns=columns)


def _infer_format(uri: str) -> SourceFormat:
    lower = uri.lower()
    for suffix, fmt in _FILE_SUFFIX_FORMATS.items():
        if lower.endswith(suffix):
            return fmt

    if uri.startswith("s3://"):
        raise ValueError(
            f"cannot infer a format for {uri!r}: an s3:// source must declare its "
            "format explicitly (TableSource(format=..., uri=...)) unless the URI "
            "ends in .csv or .parquet"
        )

    path = Path(uri)
    if path.is_dir():
        if (path / "_delta_log").is_dir():
            return SourceFormat.DELTA
        return SourceFormat.PARQUET_DIR

    raise ValueError(
        f"cannot infer a format for {uri!r}: expected a .csv, .tsv, or .parquet "
        "file, or a directory (a Delta table or a partitioned Parquet tree)"
    )


__all__ = ["SourceFormat", "TableSource"]
