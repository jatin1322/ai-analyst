"""Close-date reconstruction and its agreement tests (ARCHITECTURE 12.15, 12.3).

The production export carries no per-snapshot close date, only `days_to_close`.
Rebuilding it is the difference between a dataset that can answer pipeline and
slip questions and one that cannot. Getting it *wrong* is worse than not doing
it, so every test here is about the checking, not the arithmetic.
"""

from __future__ import annotations

import csv

import pytest

from ai_analyst.config import Settings
from ai_analyst.contracts.agreement import AgreementKind, AgreementResult, AgreementTest
from ai_analyst.contracts.binding import BindingStatus
from ai_analyst.contracts.concepts import BusinessConcept
from ai_analyst.contracts.opportunity_snapshot_v1 import OPPORTUNITY_SNAPSHOT_V1
from ai_analyst.contracts.schema import CanonicalColumn, DerivationRule
from ai_analyst.data.agreement import run_agreement_test
from ai_analyst.data.binding import build_bindings
from ai_analyst.data.ingest import ingest
from ai_analyst.data.profiler import profile_dataset
from ai_analyst.data.reconstruct import (
    close_date_agreement_tests,
    fiscal_quarter_label_sql,
    plan_reconstructions,
)
from ai_analyst.data.store import DuckDBStore
from ai_analyst.data.understanding import evaluate_agreement
from tests.fixtures.production_shape import write_production_csv


def _build(tmp_path, settings, *, corrupt: str | None = None, registry=OPPORTUNITY_SNAPSHOT_V1):
    """Ingest the production-shaped fixture with no close_date column."""
    source = tmp_path / "export.csv"
    write_production_csv(source, include_close_date=False)
    if corrupt:
        rows = list(csv.DictReader(source.open(encoding="utf-8")))
        for row in rows[:4]:
            row[corrupt] = "FY1999-Q1" if corrupt.endswith("qtr") else "-12345"
        with source.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    result = ingest(source, "recon", settings=settings, column_registry=registry)
    profile = profile_dataset("recon", result.schema, settings=settings, registry=result.registry)
    store = DuckDBStore(settings)
    with store.connect() as conn:
        report = evaluate_agreement(
            conn,
            store.snapshots_scan("recon"),
            result.schema,
            dataset_id="recon",
            settings=settings,
        )
    return result, profile, report


def test_the_export_ingests_and_the_close_date_is_rebuilt(tmp_path, settings: Settings):
    result, _, _ = _build(tmp_path, settings)
    assert CanonicalColumn.CLOSE_DATE in {c.name for c in result.schema.columns}


def test_the_rebuild_is_recorded_as_a_derivation_not_disguised_as_source(
    tmp_path, settings: Settings
):
    # "Do NOT silently pretend the source contained close_date."
    result, _, _ = _build(tmp_path, settings)
    assert CanonicalColumn.CLOSE_DATE not in result.schema.mapping.by_canonical()
    derived = {d.column: d for d in result.schema.derived_columns}
    close = derived[CanonicalColumn.CLOSE_DATE]
    assert close.rule is DerivationRule.RECONSTRUCTED
    assert close.is_reconstructed
    assert close.expression == "as_of + days_to_close"
    assert close.sources == ("as_of", "days_to_close")
    assert close.requires_confirmation
    assert "carried no close_date column" in close.note
    assert close.agreement_test_ids


def test_the_rebuilt_date_matches_the_real_one_row_for_row(tmp_path, settings: Settings):
    # The fixture can emit the same data with and without the close_date
    # column, which makes an exact ground-truth comparison possible.
    truth = tmp_path / "truth.csv"
    write_production_csv(truth, include_close_date=True)
    ingest(truth, "truth", settings=settings, column_registry=OPPORTUNITY_SNAPSHOT_V1)
    _build(tmp_path, settings)

    store = DuckDBStore(settings)
    with store.connect() as conn:
        a, b = store.snapshots_scan("truth"), store.snapshots_scan("recon")
        disagreeing = conn.execute(
            f"SELECT COUNT(*) FROM {a} t JOIN {b} r USING (as_of, opp_id) "
            "WHERE t.close_date IS DISTINCT FROM r.close_date"
        ).fetchone()[0]
        total = conn.execute(f"SELECT COUNT(*) FROM {b}").fetchone()[0]
    assert total > 0
    assert disagreeing == 0


def test_every_agreement_test_passes_on_a_consistent_export(tmp_path, settings: Settings):
    _, _, report = _build(tmp_path, settings)
    assert report.results, "expected agreement tests to run"
    assert not report.failures
    for result in report.results:
        assert result.passed, f"{result.test_id}: {result.summary}"
        assert result.checked_rows > 0


def test_an_agreement_result_reports_counts_and_samples(tmp_path, settings: Settings):
    _, _, report = _build(tmp_path, settings, corrupt="close_date_qtr")
    failed = report.by_id()["close_date_matches_close_date_qtr"]
    assert failed.failed and not failed.passed
    assert failed.disagreeing_rows == 4
    assert failed.checked_rows > failed.disagreeing_rows
    assert 0.0 < failed.agreement_rate < 1.0
    assert failed.samples, "a failure must show what disagreed"
    assert "close_date_qtr" in failed.samples[0].values
    assert failed.assertion


def test_a_failing_reconciliation_test_warns_but_does_not_withhold(
    tmp_path, settings: Settings
):
    # ARCHITECTURE 13.1: close_date_qtr's convention is undeclared, so a
    # disagreement with it is a disclosed warning, not a contradiction.
    result, profile, report = _build(tmp_path, settings, corrupt="close_date_qtr")
    bindings = build_bindings(result.schema, result.registry, profile, agreement=report)
    close = bindings.get(BusinessConcept.EXPECTED_CLOSE_DATE)
    assert close.status is BindingStatus.CONFIRMED
    assert "reconciliation_warning:close_date_matches_close_date_qtr" in close.caveats
    assert "valid_with_warnings" in close.note


def test_a_passing_reconstruction_confirms_only_because_the_registry_declared_it(
    tmp_path, settings: Settings
):
    result, profile, report = _build(tmp_path, settings)
    bindings = build_bindings(result.schema, result.registry, profile, agreement=report)
    close = bindings.get(BusinessConcept.EXPECTED_CLOSE_DATE)
    assert close.status is BindingStatus.CONFIRMED
    # The confirmation comes from the declaration, never from the passing tests.
    assert all(e.kind.value == "export_registry" for e in close.confirming_evidence)


def test_no_reconstruction_happens_without_a_declared_derivation(
    tmp_path, settings: Settings
):
    # Without a registry nothing documents the derivation, so nothing is built.
    result, _, _ = _build(tmp_path, settings, registry=None)
    assert CanonicalColumn.CLOSE_DATE not in {c.name for c in result.schema.columns}
    assert not [d for d in result.schema.derived_columns if d.is_reconstructed]


def test_reconstruction_never_overrides_a_column_the_export_carried(
    tmp_path, settings: Settings
):
    source = tmp_path / "with_close.csv"
    write_production_csv(source, include_close_date=True)
    result = ingest(
        source, "withcd", settings=settings, column_registry=OPPORTUNITY_SNAPSHOT_V1
    )
    assert CanonicalColumn.CLOSE_DATE in result.schema.mapping.by_canonical()
    assert not [d for d in result.schema.derived_columns if d.is_reconstructed]

    plans = plan_reconstructions(
        OPPORTUNITY_SNAPSHOT_V1, result.schema.mapping, list(result.schema.discovered_columns)
    )
    assert CanonicalColumn.CLOSE_DATE not in plans


def test_the_movement_test_is_the_one_that_catches_a_terminal_derived_horizon(
    tmp_path, settings: Settings
):
    # If days_to_close were measured to terminal_date, the rebuilt date would
    # never move and every slip metric would silently return zero.
    _, _, report = _build(tmp_path, settings)
    movement = report.by_id()["close_date_moves_for_pushed_deals"]
    assert movement.passed
    assert movement.checked_rows > 0
    assert "silently return zero" in movement.note


def test_a_test_whose_columns_are_absent_is_skipped_never_passed(settings: Settings):
    import duckdb

    conn = duckdb.connect()
    conn.execute("CREATE TABLE t AS SELECT 1 AS a")
    test = close_date_agreement_tests()[0]
    result = run_agreement_test(conn, "t", test, available_columns={"a"})
    assert result.skipped
    assert not result.passed
    assert not result.failed
    assert result.inconclusive
    assert "columns absent" in result.skip_reason


def test_a_null_predicate_is_out_of_scope_rather_than_agreement(settings: Settings):
    # A column of nulls must not read as universal agreement, which would be
    # the most misleading pass available.
    import duckdb

    conn = duckdb.connect()
    conn.execute("CREATE TABLE t AS SELECT * FROM (VALUES (1, NULL), (2, NULL)) v(a, b)")
    test = AgreementTest(
        id="null_scope",
        kind=AgreementKind.COLUMNS_SATISFY_RELATIONSHIP,
        assertion="a equals b",
        predicate_sql="a = b",
    )
    result = run_agreement_test(conn, "t", test, available_columns={"a", "b"})
    assert result.checked_rows == 0
    assert not result.passed
    assert result.inconclusive


def test_a_skipped_test_must_say_why():
    with pytest.raises(ValueError, match="a skipped test must say why"):
        AgreementResult(
            test_id="t",
            assertion="a",
            kind=AgreementKind.COLUMNS_SATISFY_RELATIONSHIP,
            skipped=True,
        )


@pytest.mark.parametrize(
    ("month", "start", "expected"),
    [
        (1, 1, "FY2025-Q1"),
        (6, 1, "FY2025-Q2"),
        (7, 1, "FY2025-Q3"),
        (12, 1, "FY2025-Q4"),
        (1, 2, "FY2024-Q4"),
        (4, 2, "FY2025-Q1"),
    ],
)
def test_the_quarter_label_matches_the_fiscal_calendar(month, start, expected):
    # DuckDB's `/` is true division and CAST rounds to nearest, so a naive
    # expression puts June in Q3. FLOOR is not optional here.
    from datetime import date

    import duckdb

    from ai_analyst.semantic.calendar import FiscalCalendar

    conn = duckdb.connect()
    day = date(2025, month, 15)
    conn.execute(f"CREATE TABLE t AS SELECT DATE '{day.isoformat()}' AS d")
    sql = fiscal_quarter_label_sql("d", start)
    assert conn.execute(f"SELECT {sql} FROM t").fetchone()[0] == expected
    assert FiscalCalendar(start).quarter_of(day).label == expected


def test_the_recorded_provenance_names_every_test_that_runs(tmp_path, settings: Settings):
    # Under-reporting here would omit the discriminating movement test, which
    # is the one that separates a snapshot horizon from a terminal-derived one.
    result, _, report = _build(tmp_path, settings)
    close = {d.column: d for d in result.schema.derived_columns}[CanonicalColumn.CLOSE_DATE]
    assert set(close.agreement_test_ids) == {r.test_id for r in report.results}
    assert "close_date_moves_for_pushed_deals" in close.agreement_test_ids


def test_the_open_questions_about_the_reconstruction_are_recorded_as_data(
    tmp_path, settings: Settings
):
    # ARCHITECTURE 12.19: the gaps left by the real-data inspection are
    # UnresolvedItem records, not prose, so a reader of the card sees them
    # beside the binding they qualify rather than seeing only "confirmed".
    result, _, _ = _build(tmp_path, settings)
    open_ids = {u.id for u in result.registry.open_unresolved()}
    assert {
        "days_to_close_edge_cases",
        "eoq_close_diff_convention",
        "fiscal_year_start_month",
    } <= open_ids

    # And they attach to the reconstructed column itself.
    attached = {u.id for u in result.registry.unresolved_for("close_date")}
    assert "days_to_close_edge_cases" in attached
    assert "eoq_close_diff_convention" in attached


def test_the_confirmed_close_date_binding_still_carries_its_caveats(
    tmp_path, settings: Settings
):
    result, profile, report = _build(tmp_path, settings)
    bindings = build_bindings(result.schema, result.registry, profile, agreement=report)
    close = bindings.get(BusinessConcept.EXPECTED_CLOSE_DATE)
    assert close.status is BindingStatus.CONFIRMED
    # Confirmed, but not unqualified: the open questions travel with it.
    assert "days_to_close_edge_cases" in close.caveats
    assert "eoq_close_diff_convention" in close.caveats
