"""Diagnose a real export's close-date behaviour (ARCHITECTURE 12.19).

Prints aggregate diagnostics only: counts, rates, quantiles, and category labels
for low-cardinality status candidates. No row is printed, no identifier value is
printed, and nothing is written or persisted.

    python scripts/probe_production.py /path/to/export.parquet
    python scripts/probe_production.py /path/to/partitioned_dir/
    python scripts/probe_production.py /path/to/delta_table/
    python scripts/probe_production.py 's3://bucket/prefix/snapshot.parquet'
    python scripts/probe_production.py 's3://bucket/prefix/' --format parquet_dir
    python scripts/probe_production.py export.parquet --as-of-encoding iso
    python scripts/probe_production.py export.parquet --json

Credentials are never accepted as arguments. For S3, DuckDB resolves them through
the standard AWS credential chain, so configure the environment first (for
example `aws configure` or `AWS_PROFILE`).

Format is inferred from the URI's shape (a `.csv`/`.parquet` suffix, or, for a
local directory, whether it holds a `_delta_log/`) and never guessed for an
`s3://` location that isn't obviously `.csv`/`.parquet`; pass `--format` to
declare it explicitly in that case.

`--as-of-encoding` is explicit. The default, `excel_serial`, is what the export
registry declares for `as_of`; nothing about the encoding is detected.
"""

from __future__ import annotations

import argparse
import sys

from ai_analyst.contracts.schema import DateEncoding
from ai_analyst.contracts.source import SourceFormat, TableSource
from ai_analyst.data.probe import ProbeAccessError, probe_export


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "uri",
        help="a local or s3:// csv/parquet file, a partitioned parquet directory, or a delta table",
    )
    parser.add_argument(
        "--format",
        choices=[f.value for f in SourceFormat],
        default=None,
        help="declare the source format when it cannot be inferred (e.g. an s3:// "
        "prefix with no .csv/.parquet suffix)",
    )
    parser.add_argument(
        "--as-of-encoding",
        choices=[e.value for e in DateEncoding],
        default=DateEncoding.EXCEL_SERIAL.value,
        help="how as_of is stored (declared, never detected)",
    )
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = parser.parse_args(argv)

    source: str | TableSource = args.uri
    if args.format is not None:
        source = TableSource(format=SourceFormat(args.format), uri=args.uri)

    try:
        report = probe_export(source, as_of_encoding=DateEncoding(args.as_of_encoding))
    except ProbeAccessError as exc:
        print(f"probe not run: {exc}", file=sys.stderr)
        return 2
    print(report.model_dump_json(indent=2) if args.json else report.render())
    return 0


if __name__ == "__main__":
    sys.exit(main())
