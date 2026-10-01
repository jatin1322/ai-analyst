"""Onboard a new dataset: profile, propose a draft declaration, approve it (WP6).

Nothing is confirmed by this tool. `propose` writes a draft in which every
item is `inferred` or `needs_review`; a human edits it, setting items to
`confirmed`; `approve` refuses the draft, naming every blocking item, until
the load-bearing ones are confirmed, and only then registers the dataset.

    python -m scripts.onboard propose data/synthetic/demo/snapshots.csv \\
        --out local_data/demo.draft.json
    # edit the draft: confirm grain, amount, stage, every stage value, ...
    python -m scripts.onboard approve local_data/demo.draft.json --dataset-id demo

Keep drafts of real data out of git: they list the stage column's values.
Write them under the gitignored `local_data/`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ai_analyst.contracts.source import SourceFormat, TableSource  # noqa: E402
from ai_analyst.data.onboarding import (  # noqa: E402
    OnboardingDraft,
    OnboardingRefused,
    approve,
    propose,
    summarize,
)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("propose", help="profile a source and write a draft declaration")
    p.add_argument("uri", help="a file, a partitioned directory, or a Delta table (local or s3://)")
    p.add_argument(
        "--format",
        choices=[f.value for f in SourceFormat],
        help="declare the source format; required for s3:// directories and Delta tables",
    )
    p.add_argument("--out", type=Path, required=True, help="where to write the draft JSON")
    p.add_argument("--tenant-id")
    p.add_argument("--stage-column", help="list this column's values for the stage map")

    a = sub.add_parser("approve", help="register a dataset from a human-confirmed draft")
    a.add_argument("draft", type=Path)
    a.add_argument("--dataset-id", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "propose":
        source = (
            TableSource(format=SourceFormat(args.format), uri=args.uri)
            if args.format
            else TableSource.infer(args.uri)
        )
        draft = propose(source, tenant_id=args.tenant_id, stage_column=args.stage_column)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        draft.save(args.out)
        print(summarize(draft))
        print(f"\ndraft written to {args.out}")
        return 0

    draft = OnboardingDraft.load(args.draft)
    try:
        dataset, registration = approve(draft, args.dataset_id)
    except OnboardingRefused as refused:
        print("refused: the draft is not approvable yet", file=sys.stderr)
        for problem in refused.problems:
            print(f"  - {problem}", file=sys.stderr)
        return 2
    resolution = dataset.schema.status_resolution
    print(
        f"registered {dataset.dataset_id}: status from "
        f"{resolution.strategy.value if resolution else 'nothing (unresolved)'}; "
        f"{len(registration.tenant.concept_columns)} declared concepts"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
