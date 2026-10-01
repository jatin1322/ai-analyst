"""CLI for the synthetic feature-store generator (WP7).

Writes a generic, entirely-synthetic opportunity-panel feature store plus a
``ground_truth.json`` describing exactly what was planted (duplicate/conflict
counts, planted effects and their parameters, row/opportunity/snapshot
counts). No real company data is read or produced by this script.

Examples
--------
Small deterministic dataset for local dev/tests::

    python scripts/generate_synthetic.py --out data/synthetic/demo

The later ingestion-benchmark size (not exercised by the test suite;
produces millions of rows)::

    python scripts/generate_synthetic.py --opportunities 20000 --quarters 8 \\
        --format hive_parquet --out data/synthetic/benchmark
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ai_analyst.synthetic.feature_store import (  # noqa: E402
    GeneratorConfig,
    generate,
    write_csv,
    write_hive_parquet,
    write_parquet,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--opportunities", type=int, default=100)
    parser.add_argument("--quarters", type=int, default=2)
    parser.add_argument(
        "--start-date", type=date.fromisoformat, default=date(2025, 1, 1)
    )
    parser.add_argument(
        "--variant", choices=("base", "alpha", "beta"), default="base"
    )
    parser.add_argument(
        "--format", choices=("csv", "parquet", "hive_parquet"), default="csv"
    )
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    # dst_dates has no CLI flag; derive it from --start-date rather than
    # relying on GeneratorConfig's hardcoded default, so a start date other
    # than the default's own window doesn't fail validation (dst_dates must
    # fall inside the panel).
    dst_date = args.start_date + timedelta(days=60)
    config = GeneratorConfig(
        seed=args.seed,
        n_opportunities=args.opportunities,
        start_date=args.start_date,
        n_quarters=args.quarters,
        dst_dates=[dst_date],
        variant=args.variant,
        output_format=args.format,
    )
    dataset = generate(config)

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "snapshots.csv"
    write_csv(dataset, csv_path, variant=config.variant)

    if config.output_format == "parquet":
        write_parquet(dataset, csv_path, out_dir / "snapshots.parquet", config.variant)
        csv_path.unlink()  # the CSV was only a staging file for the parquet writer
    elif config.output_format == "hive_parquet":
        write_hive_parquet(dataset, csv_path, out_dir / "snapshots_by_qtr", config.variant)
        csv_path.unlink()

    (out_dir / "ground_truth.json").write_text(
        dataset.ground_truth.model_dump_json(indent=2), encoding="utf-8"
    )
    print(f"wrote {dataset.ground_truth.n_rows} rows to {out_dir}")


if __name__ == "__main__":
    main()
