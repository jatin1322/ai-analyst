"""The production probe (ARCHITECTURE 12.19).

The probe exists to read a real export safely, so the tests that matter most are
about what it must never do: print a row, print an identifier, persist anything,
or promote a candidate. Expected numbers come from the fixture's own rows, never
from the probe.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import re
import statistics
from datetime import date
from pathlib import Path

import duckdb
import pytest

from ai_analyst.contracts.schema import DateEncoding
from ai_analyst.data.probe import ProbeReport, probe_export
from tests.fixtures.production_shape import (
    excel_serial,
    production_rows,
    write_serial_production_csv,
)

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "probe_production.py"


def _parquet(tmp_path: Path, *, drop=(), edit=None, name="export.parquet") -> Path:
    """The serial-dated production fixture, written as a typed parquet file."""
    source = tmp_path / "source.csv"
    write_serial_production_csv(source)
    rows = list(csv.DictReader(source.open(encoding="utf-8")))
    keep = [c for c in rows[0] if c not in drop]
    for index, row in enumerate(rows):
        if edit:
            edit(index, row)
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keep, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    target = tmp_path / name
    conn = duckdb.connect()
    conn.execute(
        f"COPY (SELECT * FROM read_csv('{source}', header = true)) TO '{target}' (FORMAT PARQUET)"
    )
    return target


@pytest.fixture
def report(tmp_path) -> ProbeReport:
    return probe_export(str(_parquet(tmp_path)))


def _identifiers() -> set[str]:
    rows = production_rows(include_close_date=False)
    found: set[str] = set()
    for column in ("opp_id", "account_id", "OwnerID", "RenewalManager"):
        found.update(r[column] for r in rows if r.get(column))
    return found


# -- shape -----------------------------------------------------------------------


def test_the_report_has_every_section_the_milestone_lists(report: ProbeReport):
    assert set(ProbeReport.model_fields) >= {
        "source",
        "assumptions",
        "extent",
        "identifying",
        "days_to_close",
        "status_candidates",
        "days_by_candidate",
        "reconstruction",
        "agreement",
        "eoq_close_diff",
        "cd_in_qtr_boundaries",
        "flags",
        "quarter_labels",
        "not_evaluable",
    }
    rendered = report.render()
    for heading in (
        "== extent ==",
        "== days_to_close ==",
        "== status candidates (candidates only) ==",
        "== reconstruction: as_of + days_to_close ==",
        "== agreement tests (validity failures withhold; reconciliation failures warn) ==",
        "== eoq_close_diff ==",
        "== stamped quarter labels ==",
        "== assumptions ==",
    ):
        assert heading in rendered


def test_extent_matches_the_fixtures_own_rows(report: ProbeReport):
    rows = production_rows(include_close_date=False)
    extent = report.extent
    assert extent.rows == len(rows)
    assert extent.distinct_opportunities == len({r["opp_id"] for r in rows})
    assert extent.snapshots == len({r["as_of"] for r in rows})
    assert extent.first_snapshot == min(r["as_of"] for r in rows)
    assert extent.last_snapshot == max(r["as_of"] for r in rows)
    assert extent.duplicate_grain_keys == 0


def test_as_of_representation_is_reported_from_the_declared_encoding(report: ProbeReport):
    extent = report.extent
    assert extent.as_of_serial_rows == extent.rows
    assert extent.as_of_iso_rows == 0
    assert extent.as_of_invalid_rows == 0
    assert extent.as_of_fractional_rows == extent.rows  # every as_of carries .4583333
    assert report.assumptions.as_of_encoding == "excel_serial"
    assert report.assumptions.excel_epoch == "1899-12-30"


def test_days_to_close_summary_matches_the_fixtures_rows(report: ProbeReport):
    days = [int(r["days_to_close"]) for r in production_rows(include_close_date=False)]
    summary = report.days_to_close
    assert summary.present
    assert summary.non_null == len(days)
    assert summary.null == 0
    assert summary.fractional == 0
    assert summary.negative == sum(d < 0 for d in days)
    assert summary.zero == sum(d == 0 for d in days)
    assert summary.positive == sum(d > 0 for d in days)
    assert summary.quantiles["0"] == min(days)
    assert summary.quantiles["100"] == max(days)
    assert summary.quantiles["50"] == statistics.median(days)


def test_reconstruction_and_agreement_are_reported(report: ProbeReport):
    rows = production_rows(include_close_date=False)
    assert report.reconstruction.evaluable
    assert report.reconstruction.non_null == len(rows)
    assert report.reconstruction.null == 0
    assert report.reconstruction.opportunities_with_moving_date > 0
    assert report.reconstruction.pushed_opportunities is not None

    lines = {line.test_id: line for line in report.agreement}
    assert set(lines) == {
        "close_date_is_structurally_valid",
        "close_date_matches_close_date_qtr",
        "close_date_matches_cd_in_qtr",
        "close_date_matches_eoq_close_diff",
        "close_date_moves_for_pushed_deals",
    }
    assert all(line.status == "passed" for line in lines.values())
    assert {line.role for line in lines.values()} == {"validity", "reconciliation"}
    assert all(line.checked_rows > 0 for line in lines.values())


def test_identifying_fields_are_counts_only(report: ProbeReport):
    rows = production_rows(include_close_date=False)
    by_name = {f.column: f for f in report.identifying}
    assert by_name["opp_id"].distinct == len({r["opp_id"] for r in rows})
    assert by_name["account_id"].nulls == 0
    assert set(by_name) >= {"opp_id", "account_id", "OwnerID"}


# -- what it must never do ---------------------------------------------------------


def test_the_rendered_report_contains_no_identifier_from_the_data(report: ProbeReport):
    text = report.render()
    blob = report.model_dump_json()
    assert _identifiers(), "the fixture must actually have identifiers to leak"
    for identifier in _identifiers():
        assert identifier not in text, f"identifier {identifier!r} was printed"
        assert identifier not in blob, f"identifier {identifier!r} is in the JSON"


def test_no_narrative_text_is_ever_read_into_the_report(report: ProbeReport):
    rows = production_rows(include_close_date=False)
    notes = {r["ManagerNotes"] for r in rows if r.get("ManagerNotes")}
    assert notes
    text = report.render() + report.model_dump_json()
    for note in notes:
        assert note[:25] not in text


def test_a_failing_test_prints_counts_and_never_the_offending_rows(tmp_path):
    def corrupt(index: int, row: dict[str, str]) -> None:
        if index < 4:
            row["close_date_qtr"] = "FY1999-Q1"

    failing = probe_export(str(_parquet(tmp_path, edit=corrupt)))
    line = {ln.test_id: ln for ln in failing.agreement}["close_date_matches_close_date_qtr"]
    assert line.status == "FAILED"
    assert line.disagreeing_rows == 4
    blob = failing.model_dump_json() + failing.render()
    for identifier in _identifiers():
        assert identifier not in blob
    # No sample is attached to a failure: a sample is a row.
    assert "samples" not in blob


def test_probing_writes_nothing(tmp_path):
    export = _parquet(tmp_path)
    before = sorted(p.name for p in tmp_path.rglob("*"))
    probe_export(str(export))
    assert sorted(p.name for p in tmp_path.rglob("*")) == before


def test_the_report_names_the_file_and_never_the_path(tmp_path):
    export = _parquet(tmp_path, name="tenant_export.parquet")
    result = probe_export(str(export))
    assert result.source == "tenant_export.parquet"
    assert str(tmp_path) not in result.render()


# -- status candidates ----------------------------------------------------------------


def test_status_fields_are_reported_as_candidates_only(report: ProbeReport):
    names = {c.column for c in report.status_candidates}
    assert {"Stage", "terminal_fate"} <= names
    for candidate in report.status_candidates:
        assert "candidate only" in candidate.note
        assert "not mapped" in candidate.note
    assert "candidates only" in report.render()


def test_a_candidate_is_never_promoted_to_the_status_configuration(tmp_path, settings):
    # The probe reports and stops. Nothing it lists reaches configuration or the
    # schema, so status stays exactly as configured (unset), and no name match
    # becomes a mapping.
    probe_export(str(_parquet(tmp_path)))
    assert settings.status_column is None
    assert settings.status_mapping() is None


def test_days_to_close_is_summarised_per_candidate_value(report: ProbeReport):
    rows = production_rows(include_close_date=False)
    groups = {g.value: g for g in report.days_by_candidate["Stage"]}
    for stage in {r["Stage"] for r in rows}:
        expected = [int(r["days_to_close"]) for r in rows if r["Stage"] == stage]
        group = groups[stage]
        assert group.rows == len(expected)
        assert group.minimum == min(expected)
        assert group.maximum == max(expected)
        assert group.negative == sum(d < 0 for d in expected)


# -- quarter labels and fiscal assumptions ---------------------------------------------


def test_the_stamped_label_shape_is_reported_and_matches_the_assumed_format(
    report: ProbeReport,
):
    close = {q.column: q for q in report.quarter_labels}["close_date_qtr"]
    assert close.matching_assumed_format == close.non_null
    assert set(close.shapes) == {"FYdddd-Qd"}  # digits masked: the shape, not the value


def test_a_different_label_format_is_visible_not_assumed_away(tmp_path):
    def other_format(index: int, row: dict[str, str]) -> None:
        row["close_date_qtr"] = "2026Q3"

    result = probe_export(str(_parquet(tmp_path, edit=other_format)))
    close = {q.column: q for q in result.quarter_labels}["close_date_qtr"]
    assert close.matching_assumed_format == 0
    assert set(close.shapes) == {"ddddQd"}
    # And the agreement test fails on it, which is the honest outcome.
    line = {ln.test_id: ln for ln in result.agreement}["close_date_matches_close_date_qtr"]
    assert line.status == "FAILED"


def test_the_fiscal_assumptions_are_printed_and_marked_unverified(report: ProbeReport):
    assumptions = report.assumptions
    assert assumptions.fiscal_year_start_month == 1
    assert not assumptions.fiscal_year_start_month_verified
    assert assumptions.quarter_label_format == "FY<year>-Q<n>"
    rendered = report.render()
    assert "NOT verified" in rendered
    assert "month 1" in rendered


# -- what cannot be evaluated is said, not skipped ----------------------------------------


def test_missing_columns_are_reported_as_not_evaluable(tmp_path):
    result = probe_export(
        str(
            _parquet(
                tmp_path, drop=("close_date_qtr", "close_date_push_count", "eoq_close_diff")
            )
        )
    )
    said = " ".join(result.not_evaluable)
    assert "close_date_push_count" in said
    assert "close_date_qtr" in said
    assert "eoq_close_diff" in said
    assert not result.eoq_close_diff.present
    assert result.reconstruction.pushed_opportunities is None
    statuses = {ln.test_id: ln.status for ln in result.agreement}
    assert statuses["close_date_matches_close_date_qtr"] == "skipped"
    assert statuses["close_date_moves_for_pushed_deals"] == "skipped"
    assert "== not evaluable here ==" in result.render()


def test_an_export_without_days_to_close_cannot_be_reconstructed(tmp_path):
    result = probe_export(str(_parquet(tmp_path, drop=("days_to_close",))))
    assert not result.days_to_close.present
    assert not result.reconstruction.evaluable
    assert result.agreement == []
    assert any("days_to_close" in item for item in result.not_evaluable)


def test_an_export_without_the_grain_is_refused(tmp_path):
    path = tmp_path / "nograin.parquet"
    duckdb.connect().execute(f"COPY (SELECT 1 AS x) TO '{path}' (FORMAT PARQUET)")
    with pytest.raises(ValueError, match="cannot be probed"):
        probe_export(str(path))


# -- eoq_close_diff and the boundary diagnostics ------------------------------------------


def _crafted(tmp_path) -> Path:
    """Five rows, hand-built so every diagnostic has a known answer.

    as_of is 2025-03-15, so the quarter ends 2025-03-31 and days_to_eoq is 16.

    row  close        days  flag  what it pins
    A    2025-03-31     16    0   last day of the quarter; flag says out, calendar in
    B    2025-01-01    -73    0   first day of the quarter; flag says out, calendar in
    C    2025-03-20      5  100   interior, agrees
    D    2025-04-15     31    0   next quarter, agrees
    E    2025-02-10    -33    0   interior; flag says out, calendar in
    """
    as_of = excel_serial(date(2025, 3, 15))
    rows = [
        # opp, days_to_close, flag, eoq_close_diff = (16 - days) + jitter
        ("A", 16, 0, 0),
        ("B", -73, 0, 90),  # 89 + 1
        ("C", 5, 100, 10),  # 11 - 1
        ("D", 31, 0, -15),
        ("E", -33, 0, 49),
    ]
    conn = duckdb.connect()
    conn.execute(
        "CREATE TABLE t (opp_id VARCHAR, as_of VARCHAR, days_to_close BIGINT, "
        "days_to_eoq BIGINT, eoq_close_diff BIGINT, CD_in_qtr BIGINT, CD_in_past BIGINT)"
    )
    for opp, days, flag, eoq in rows:
        conn.execute(
            "INSERT INTO t VALUES (?, ?, ?, 16, ?, ?, ?)",
            [opp, as_of, days, eoq, flag, 100 if days < 0 else 0],
        )
    target = tmp_path / "crafted.parquet"
    conn.execute(f"COPY t TO '{target}' (FORMAT PARQUET)")
    return target


def test_disagreements_are_placed_relative_to_the_quarters_edges(tmp_path):
    result = probe_export(str(_crafted(tmp_path)))
    boundary = result.cd_in_qtr_boundaries
    assert boundary.evaluable
    assert boundary.disagreeing == 3  # A, B, E
    assert boundary.flag_out_calendar_in == 3
    assert boundary.flag_in_calendar_out == 0
    assert boundary.on_quarter_first_day == 1  # B
    assert boundary.on_quarter_last_day == 1  # A
    assert boundary.interior == 1  # E: a genuine disagreement, not a boundary effect
    assert "on a quarter's first day 1, last day 1, interior 1" in result.render()


def test_eoq_offsets_are_a_histogram_not_a_pass_or_fail(tmp_path):
    result = probe_export(str(_crafted(tmp_path)))
    eoq = result.eoq_close_diff
    assert eoq.closest_convention == "days_to_eoq - days_to_close"
    assert eoq.conventions["days_to_eoq - days_to_close"] == (3, 5)  # A, D, E exact
    assert eoq.delta_histogram == {"+0": 3, "+1": 1, "-1": 1}
    assert "rows by delta" in result.render()


def test_the_agreement_tests_stay_exact_and_fail_on_that_data(tmp_path):
    # No tolerance is added to make this pass: the probe reports the near-miss
    # and the test keeps failing. Allowing a one-day tolerance is a decision for
    # the project owner (ARCHITECTURE 12.19), not something the probe smuggles in.
    result = probe_export(str(_crafted(tmp_path)))
    lines = {ln.test_id: ln for ln in result.agreement}
    assert lines["close_date_matches_cd_in_qtr"].status == "FAILED"
    assert lines["close_date_matches_cd_in_qtr"].disagreeing_rows == 3
    assert lines["close_date_matches_eoq_close_diff"].status == "FAILED"
    assert lines["close_date_matches_eoq_close_diff"].disagreeing_rows == 2  # B and C


def test_zero_one_style_flags_report_their_value_set_before_a_convention(tmp_path):
    result = probe_export(str(_crafted(tmp_path)))
    flags = {f.column: f for f in result.flags}
    assert flags["CD_in_qtr"].values == {"0": 4, "100": 1}  # 0/100, not 0/1
    assert flags["CD_in_past"].distinct == 2


# -- the script ----------------------------------------------------------------------------


def _script():
    spec = importlib.util.spec_from_file_location("probe_production", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_script_prints_the_report(tmp_path, capsys):
    export = _parquet(tmp_path)
    assert _script().main([str(export)]) == 0
    printed = capsys.readouterr().out
    assert "aggregates only; no row is printed" in printed
    for identifier in _identifiers():
        assert identifier not in printed


def test_the_script_can_emit_json(tmp_path, capsys):
    export = _parquet(tmp_path)
    assert _script().main([str(export), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["extent"]["rows"] == 56
    assert payload["assumptions"]["as_of_encoding"] == "excel_serial"


def test_the_script_takes_no_credentials(capsys):
    # Credentials resolve through the standard AWS chain. An argument for them
    # would put a secret in shell history and process listings.
    with pytest.raises(SystemExit):
        _script().main(["--help"])
    options = set(re.findall(r"--[a-z][a-z-]*", capsys.readouterr().out))
    assert options >= {"--json", "--as-of-encoding"}
    assert not any(word in opt for opt in options for word in ("key", "secret", "token"))


def test_the_script_can_read_an_iso_export(tmp_path, capsys):
    source = tmp_path / "iso.csv"
    source.write_text(
        "as_of,opp_id,days_to_close\n2025-03-15,O-1,10\n2025-03-16,O-1,9\n", encoding="utf-8"
    )
    assert _script().main([str(source), "--as-of-encoding", "iso"]) == 0
    printed = capsys.readouterr().out
    assert "as_of encoding        : iso" in printed
    assert "O-1" not in printed


def test_the_iso_encoding_is_explicit_not_guessed(tmp_path):
    source = tmp_path / "iso.csv"
    source.write_text(
        "as_of,opp_id,days_to_close\n2025-03-15,O-1,10\n2025-03-16,O-1,9\n", encoding="utf-8"
    )
    # Read as Excel serials the ISO strings are converted by the fallback branch,
    # and read as ISO they convert directly: same dates, encoding stated either way.
    a = probe_export(str(source), as_of_encoding=DateEncoding.ISO)
    b = probe_export(str(source), as_of_encoding=DateEncoding.EXCEL_SERIAL)
    assert a.extent.first_snapshot == b.extent.first_snapshot == "2025-03-15"
    assert a.assumptions.as_of_encoding == "iso"
