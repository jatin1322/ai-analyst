"""Declared capture de-duplication for intraday snapshots (WP2).

The default stays a hard failure. `latest_capture` is a tenant declaration, and
what it did is reported: groups resolved, rows dropped, and conflicting groups.
Expected counts come from the synthetic generator's planted ground truth, not
from re-deriving them with the code under test.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from ai_analyst.config import Settings
from ai_analyst.contracts.errors import ErrorCode, IngestionError
from ai_analyst.contracts.snapshot_policy import (
    IntradayPolicy,
    SnapshotGranularity,
    SnapshotPolicy,
)
from ai_analyst.contracts.tenant import TenantProfile
from ai_analyst.data.ingest import ingest
from ai_analyst.data.store import DuckDBStore
from ai_analyst.synthetic.feature_store import GeneratorConfig, generate, write_csv
from tests.conftest import DUPLICATE_KEY_CSV

LATEST = SnapshotPolicy(intraday_policy=IntradayPolicy.LATEST_CAPTURE)


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    dataset = generate(GeneratorConfig(seed=1, n_opportunities=30, n_quarters=1))
    path = tmp_path_factory.mktemp("capture") / "snapshots.csv"
    write_csv(dataset, path)
    return dataset.ground_truth, path


def _ingest(path: Path, tmp_path: Path, policy: SnapshotPolicy | None):
    settings = Settings(data_root=tmp_path / "data")
    return ingest(
        path, "ds", snapshot_policy=policy, settings=settings, store=DuckDBStore(settings)
    )


def test_default_policy_still_hard_fails_on_duplicate_captures(synthetic, tmp_path):
    truth, path = synthetic
    with pytest.raises(IngestionError) as info:
        _ingest(path, tmp_path, None)
    assert info.value.detail.code is ErrorCode.GRAIN_VIOLATION
    assert info.value.detail.duplicate_key_count == truth.duplicate_groups


def test_latest_capture_resolves_and_matches_ground_truth(synthetic, tmp_path):
    truth, path = synthetic
    result = _ingest(path, tmp_path, LATEST)
    resolution = result.schema.capture_resolution
    assert resolution is not None
    assert resolution.policy is IntradayPolicy.LATEST_CAPTURE
    assert resolution.duplicate_groups == truth.duplicate_groups
    # Each planted group is exactly one pair, so one row is dropped per group.
    assert resolution.rows_dropped == truth.duplicate_groups
    assert result.row_count == truth.n_rows - truth.duplicate_groups
    assert resolution.conflicting_groups == truth.conflicting_groups
    assert resolution.has_conflicts


def test_kept_row_is_the_latest_capture(synthetic, tmp_path):
    _, path = synthetic
    result = _ingest(path, tmp_path, LATEST)
    raw = duckdb.connect()
    # The later of each pair is the one captured at the later time of day.
    expected = raw.execute(
        f"""
        SELECT COUNT(*) FROM (
            SELECT opp_id, CAST(as_of AS DATE) d, MAX(as_of) latest
            FROM read_csv('{path.as_posix()}', header = true) GROUP BY 1, 2
        )
        """
    ).fetchone()[0]
    assert expected == result.row_count


def test_conflicting_groups_are_counted_exactly(tmp_path):
    csv = tmp_path / "hand.csv"
    csv.write_text(
        "as_of,opp_id,amount,stage\n"
        # identical duplicate pair (not a conflict)
        "2025-03-09 03:00:00,A,100.00,Proposal\n"
        "2025-03-09 04:00:00,A,100.00,Proposal\n"
        # differing pair (conflict): the later capture wins
        "2025-03-09 03:00:00,B,50.00,Proposal\n"
        "2025-03-09 04:00:00,B,60.00,Negotiation\n"
        "2025-03-10 03:00:00,B,60.00,Negotiation\n",
        encoding="utf-8",
    )
    result = _ingest(csv, tmp_path, LATEST)
    resolution = result.schema.capture_resolution
    assert (resolution.duplicate_groups, resolution.rows_dropped) == (2, 2)
    assert resolution.conflicting_groups == 1
    assert result.row_count == 3
    kept = duckdb.connect().execute(
        f"SELECT amount FROM read_parquet('{result.canonical_path.as_posix()}/**/*.parquet', "
        "hive_partitioning = true) WHERE opp_id = 'B' AND as_of = DATE '2025-03-09'"
    ).fetchall()
    assert [float(a) for (a,) in kept] == [60.0]


def test_date_only_source_cannot_resolve_and_fails_clearly(tmp_path):
    with pytest.raises(IngestionError) as info:
        _ingest(DUPLICATE_KEY_CSV, tmp_path, LATEST)
    detail = info.value.detail
    assert detail.code is ErrorCode.CAPTURE_RESOLUTION_FAILED
    assert detail.reason == "capture_timestamp_unresolvable"
    assert detail.unresolvable_groups == 2


def test_capture_granularity_is_declared_but_refused(synthetic, tmp_path):
    _, path = synthetic
    policy = SnapshotPolicy(snapshot_granularity=SnapshotGranularity.CAPTURE)
    with pytest.raises(IngestionError) as info:
        _ingest(path, tmp_path, policy)
    assert info.value.detail.reason == "granularity_capture_unsupported"


def test_policy_is_not_part_of_the_grant_fingerprint():
    base = TenantProfile(tenant_id="t")
    declared = TenantProfile(tenant_id="t", snapshot_policy=LATEST)
    assert base.declarations_fingerprint == declared.declarations_fingerprint
