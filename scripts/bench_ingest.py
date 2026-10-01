"""Opt-in ingestion benchmark (not a test).

Generates a synthetic feature-store panel of the requested size (entirely
synthetic, no real data), writes it as hive-partitioned Parquet into a temp
directory, ingests it through the typed fast path, and prints rows/sec.

    python scripts/bench_ingest.py --opportunities 200 --quarters 1

Size scales as roughly ``opportunities x 91 x quarters`` rows; generation is
pure Python and is the slow part for large sizes, and it is not timed.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ai_analyst.config import Settings  # noqa: E402
from ai_analyst.contracts.snapshot_policy import IntradayPolicy, SnapshotPolicy  # noqa: E402
from ai_analyst.contracts.source import SourceFormat, TableSource  # noqa: E402
from ai_analyst.data.ingest import ingest  # noqa: E402
from ai_analyst.data.store import DuckDBStore  # noqa: E402
from ai_analyst.synthetic.feature_store import (  # noqa: E402
    GeneratorConfig,
    generate,
    write_csv,
    write_hive_parquet,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opportunities", type=int, default=200)
    parser.add_argument("--quarters", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="bench_ingest_") as tmp:
        root = Path(tmp)
        t0 = time.perf_counter()
        dataset = generate(
            GeneratorConfig(
                seed=args.seed, n_opportunities=args.opportunities, n_quarters=args.quarters
            )
        )
        csv_path = write_csv(dataset, root / "panel.csv")
        parquet_dir = write_hive_parquet(dataset, csv_path, root / "panel", "base")
        print(f"generated {len(dataset.rows):,} rows in {time.perf_counter() - t0:.1f}s")

        settings = Settings(data_root=root / "data")
        source = TableSource(format=SourceFormat.PARQUET_DIR, uri=str(parquet_dir))
        # The synthetic panel plants same-day double captures, so declare how
        # to resolve them (the default is a hard failure).
        policy = SnapshotPolicy(intraday_policy=IntradayPolicy.LATEST_CAPTURE)
        t1 = time.perf_counter()
        result = ingest(
            source,
            "bench",
            snapshot_policy=policy,
            settings=settings,
            store=DuckDBStore(settings),
        )
        elapsed = time.perf_counter() - t1

    resolution = result.schema.capture_resolution
    print(f"ingested  {result.row_count:,} rows in {elapsed:.2f}s")
    print(f"throughput {result.row_count / elapsed:,.0f} rows/sec")
    if resolution is not None:
        print(
            f"captures: {resolution.duplicate_groups} groups resolved, "
            f"{resolution.rows_dropped} rows dropped, "
            f"{resolution.conflicting_groups} conflicting"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
