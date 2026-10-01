"""Declared family invariants as reconciliation evidence (WP5)."""

from __future__ import annotations

from pathlib import Path

import duckdb

from ai_analyst.config import Settings
from ai_analyst.contracts.agreement import AgreementRole
from ai_analyst.contracts.tenant import FamilyInvariant, FamilyInvariantKind
from ai_analyst.data.ingest import ingest
from ai_analyst.data.store import DuckDBStore
from ai_analyst.synthetic.feature_store import (
    WINDOW_LABELS,
    GeneratorConfig,
    generate,
    write_csv,
)

WINDOWS = tuple(WINDOW_LABELS)
INVARIANTS = [
    FamilyInvariant(
        id="emails_monotone",
        kind=FamilyInvariantKind.WINDOW_MONOTONE,
        template="email_count_{w}",
        windows=WINDOWS,
        source="test",
    ),
    FamilyInvariant(
        id="email_parts",
        kind=FamilyInvariantKind.PARTS_SUM,
        parts=("inbound_count_{w}", "outbound_count_{w}"),
        whole="email_count_{w}",
        windows=WINDOWS,
        source="test",
    ),
    FamilyInvariant(
        id="win_mask",
        kind=FamilyInvariantKind.MASK_IMPLIES_NULL,
        label="win_label",
        mask="win_label_mask",
        source="test",
    ),
]


def _csv(tmp_path: Path) -> Path:
    dataset = generate(
        GeneratorConfig(
            seed=5, n_opportunities=30, n_quarters=1, dst_duplicates_per_date=0, conflict_pairs=0
        )
    )
    return write_csv(dataset, tmp_path / "s.csv")


def _ingest(csv: Path, tmp_path: Path, name: str):
    settings = Settings(data_root=tmp_path / name)
    result = ingest(
        csv, name, family_invariants=INVARIANTS, settings=settings, store=DuckDBStore(settings)
    )
    return result, settings


def test_invariants_hold_on_clean_synthetic_data(tmp_path):
    result, settings = _ingest(_csv(tmp_path), tmp_path, "clean")
    by_id = result.agreement.by_id()
    assert len(by_id) == 3
    for inv in INVARIANTS:
        r = by_id[f"family_invariant:{inv.id}"]
        assert r.role is AgreementRole.RECONCILIATION
        assert r.checked_rows > 0 and r.disagreeing_rows == 0, r.summary
    # Persisted like other agreement results.
    assert settings.agreement_path("clean").exists()


def test_corrupted_copy_reports_exactly_the_edited_rows(tmp_path):
    clean = _csv(tmp_path)
    bad = tmp_path / "bad.csv"
    con = duckdb.connect()
    con.execute(
        f"CREATE TABLE t AS SELECT * FROM "
        f"read_csv('{clean.as_posix()}', header=true, all_varchar=true)"
    )
    keys = con.execute(
        "SELECT opp_id, as_of FROM t WHERE email_count_lifetime IS NOT NULL ORDER BY 1, 2 LIMIT 3"
    ).fetchall()
    # Break monotonicity on 3 rows: 7d larger than lifetime.
    for opp, as_of in keys:
        con.execute(
            "UPDATE t SET email_count_7d = "
            "CAST(CAST(email_count_lifetime AS BIGINT) + 5 AS VARCHAR) "
            "WHERE opp_id = ? AND as_of = ?",
            [opp, as_of],
        )
    con.execute(f"COPY t TO '{bad.as_posix()}' (HEADER, DELIMITER ',')")
    result, _ = _ingest(bad, tmp_path, "bad")
    by_id = result.agreement.by_id()
    assert by_id["family_invariant:emails_monotone"].disagreeing_rows == 3
    # email_count_7d also feeds the parts identity for exactly those rows.
    assert by_id["family_invariant:email_parts"].disagreeing_rows == 3
    assert by_id["family_invariant:win_mask"].disagreeing_rows == 0
