"""Excel serial date support (ARCHITECTURE 12.19).

Both real exports store their date fields as Excel serial numbers. The rule that
governs everything here is that a numeric column is never a date because its
values look like one: serial conversion happens only where a declaration says
so, and an invalid value is an error rather than a null.

Expected dates are computed by hand or from first principles, never by calling
the code under test.
"""

from __future__ import annotations

from datetime import date

import duckdb
import pytest

from ai_analyst.config import Settings
from ai_analyst.contracts.errors import (
    DateEncodingInvalid,
    IngestionError,
    InvalidDateValues,
    NullGrainKey,
)
from ai_analyst.contracts.opportunity_snapshot_v1 import OPPORTUNITY_SNAPSHOT_V1
from ai_analyst.contracts.profile import ProfileKind
from ai_analyst.contracts.schema import (
    EXCEL_EPOCH,
    EXCEL_MAX_SERIAL,
    EXCEL_MIN_SERIAL,
    CanonicalColumn,
    DataType,
    DateEncoding,
)
from ai_analyst.data.conform import date_sql
from ai_analyst.data.dataset import register_dataset
from ai_analyst.data.ingest import ingest
from ai_analyst.data.store import DuckDBStore
from tests.fixtures.production_shape import (
    excel_serial,
    write_production_csv,
    write_serial_production_csv,
)

SERIAL = DateEncoding.EXCEL_SERIAL


def _convert(value: str | None, encoding: DateEncoding = SERIAL) -> date | None:
    conn = duckdb.connect()
    conn.execute("CREATE TABLE t (x VARCHAR)")
    conn.execute("INSERT INTO t VALUES (?)", [value])
    return conn.execute(f"SELECT {date_sql('x', encoding)} FROM t").fetchone()[0]


def _write(tmp_path, body: str):
    path = tmp_path / "source.csv"
    path.write_text(body, encoding="utf-8")
    return path


# -- the epoch is explicit and documented -----------------------------------


def test_the_epoch_is_the_documented_1900_system_base():
    # 1899-12-30, not 1900-01-01, because Excel counts a nonexistent 1900-02-29
    # (the Lotus 1-2-3 leap-year bug) as serial 60. From serial 61 on, the base
    # plus the serial is the true date.
    assert EXCEL_EPOCH == "1899-12-30"
    assert EXCEL_MIN_SERIAL == 61
    assert date(1899, 12, 30).toordinal() + EXCEL_MAX_SERIAL == date(9999, 12, 31).toordinal()


# -- known serial -> known date ---------------------------------------------


@pytest.mark.parametrize(
    ("serial", "expected"),
    [
        ("45973", date(2025, 11, 12)),  # 2025-01-01 is 45658; +304 to Nov 1; +11
        ("45658", date(2025, 1, 1)),
        ("36526", date(2000, 1, 1)),
        ("61", date(1900, 3, 1)),  # the first serial that is unambiguous
        (str(EXCEL_MAX_SERIAL), date(9999, 12, 31)),
        ("45973.0", date(2025, 11, 12)),  # a DOUBLE column prints a whole day as 45973.0
        ("4.5973e4", date(2025, 11, 12)),  # and may print in scientific notation
    ],
)
def test_a_known_serial_converts_to_the_known_date(serial: str, expected: date):
    assert _convert(serial) == expected


def test_the_fixture_helper_agrees_with_hand_arithmetic():
    assert excel_serial(date(2025, 11, 12)) == "45973"


# -- fractional serials ------------------------------------------------------


@pytest.mark.parametrize("fraction", [".0", ".4583333", ".5", ".6", ".999"])
def test_a_time_of_day_fraction_is_discarded_never_rounded_up(fraction: str):
    # `.6` is the case that matters: DuckDB rounds on cast, so a CAST would move
    # this to the next day. FLOOR keeps the whole-day part, and the time of day is
    # dropped because the conformed type is DATE.
    assert _convert(f"45973{fraction}") == date(2025, 11, 12)


# -- invalid serials ---------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "0",  # before the valid range
        "60",  # Excel's phantom 1900-02-29, ambiguous by construction
        "-5",
        "2958466",  # one day past 9999-12-31
        "20250101",  # an 8-digit integer is not a date and is not guessed at
        "abc",
        "45973.5.5",
    ],
)
def test_an_invalid_serial_converts_to_nothing(value: str):
    assert _convert(value) is None


def test_nothing_is_converted_under_the_iso_encoding():
    # Serials are only ever recognised where a column is declared as serials.
    assert _convert("45973", DateEncoding.ISO) is None
    assert _convert("2025-11-12", DateEncoding.ISO) == date(2025, 11, 12)


def test_an_invalid_declared_value_stops_ingestion_and_says_which(tmp_path, settings: Settings):
    source = _write(
        tmp_path,
        "as_of,opp_id,terminal_date\n"
        "45973,O-1,45980\n"
        "45974,O-2,0\n"
        "45975,O-3,20250101\n"
        "45976,O-4,not-a-date\n",
    )
    with pytest.raises(IngestionError) as excinfo:
        ingest(source, "bad", settings=settings, column_registry=OPPORTUNITY_SNAPSHOT_V1)
    detail = excinfo.value.detail
    assert isinstance(detail, InvalidDateValues)
    assert detail.column == "terminal_date"
    assert detail.encoding == "excel_serial"
    assert detail.invalid_rows == 3
    assert set(detail.samples) == {"0", "20250101", "not-a-date"}
    assert "not guessed" in detail.message
    # It failed before writing anything.
    assert not DuckDBStore(settings).canonical_exists("bad")


# -- nulls -------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, "", "   "])
def test_a_null_or_blank_is_null_and_not_an_error(value):
    assert _convert(value) is None


def test_nulls_in_a_declared_column_are_counted_not_rejected(tmp_path, settings: Settings):
    source = _write(
        tmp_path,
        "as_of,opp_id,terminal_date\n45973,O-1,45980\n45974,O-2,\n45975,O-3,   \n",
    )
    result = ingest(source, "nulls", settings=settings, column_registry=OPPORTUNITY_SNAPSHOT_V1)
    conversion = {c.column: c for c in result.schema.date_conversions}["terminal_date"]
    assert conversion.null_rows == 2
    assert conversion.serial_rows == 1


# -- ISO and mixed representations -------------------------------------------


def test_a_normal_date_string_is_preserved_under_a_serial_declaration():
    assert _convert("2025-11-12") == date(2025, 11, 12)


def test_a_column_may_mix_serials_and_iso_dates(tmp_path, settings: Settings):
    source = _write(
        tmp_path,
        "as_of,opp_id\n45973,O-1\n2025-11-13,O-2\n45975.75,O-3\n",
    )
    result = ingest(source, "mixed", settings=settings, column_registry=OPPORTUNITY_SNAPSHOT_V1)
    conversion = {c.column: c for c in result.schema.date_conversions}["as_of"]
    assert (conversion.serial_rows, conversion.iso_rows) == (2, 1)
    assert conversion.fractional_rows == 1
    assert (conversion.min_date, conversion.max_date) == (date(2025, 11, 12), date(2025, 11, 14))


# -- no heuristic date guessing ------------------------------------------------


def test_an_undeclared_numeric_column_is_never_read_as_dates(tmp_path, settings: Settings):
    # 45973 is an integer until somebody with authority says the column is an
    # Excel serial. Type detection must not find it.
    source = _write(
        tmp_path,
        "as_of,opp_id,visit_count\n2025-11-12,O-1,45973\n2025-11-13,O-1,45974\n",
    )
    result = ingest(source, "count", settings=settings)
    assert result.schema.discovered_types["visit_count"] is DataType.BIGINT
    assert not result.schema.date_conversions


def test_an_undeclared_serial_as_of_is_not_guessed_and_hard_fails(tmp_path, settings: Settings):
    # Without a declaration nothing converts the serial, so the grain key is
    # null and ingestion refuses the file. It does not decide the numbers are dates.
    source = _write(tmp_path, "as_of,opp_id\n45973,O-1\n45974,O-2\n")
    with pytest.raises(IngestionError) as excinfo:
        ingest(source, "undeclared", settings=settings)
    assert isinstance(excinfo.value.detail, NullGrainKey)


def test_a_caller_can_declare_a_serial_column_without_a_registry(tmp_path, settings: Settings):
    source = _write(tmp_path, "as_of,opp_id\n45973,O-1\n45974,O-2\n")
    result = ingest(
        source, "explicit", settings=settings, date_encodings={"as_of": SERIAL}
    )
    assert result.row_count == 2
    assert result.snapshot_count == 2


def test_an_explicit_iso_declaration_cancels_the_registrys(tmp_path, settings: Settings):
    source = _write(tmp_path, "as_of,opp_id\n2025-11-12,O-1\n2025-11-13,O-2\n")
    result = ingest(
        source,
        "cancel",
        settings=settings,
        column_registry=OPPORTUNITY_SNAPSHOT_V1,
        date_encodings={"as_of": DateEncoding.ISO},
    )
    assert not result.schema.date_conversions


def test_declaring_a_column_the_source_lacks_is_an_error(tmp_path, settings: Settings):
    source = _write(tmp_path, "as_of,opp_id\n2025-11-12,O-1\n")
    with pytest.raises(IngestionError) as excinfo:
        ingest(source, "missing", settings=settings, date_encodings={"nope": SERIAL})
    detail = excinfo.value.detail
    assert isinstance(detail, DateEncodingInvalid)
    assert detail.reason == "column_not_in_source"


def test_declaring_dates_on_a_column_bound_to_a_non_date_is_an_error(
    tmp_path, settings: Settings
):
    source = _write(tmp_path, "as_of,opp_id,Stage\n2025-11-12,O-1,Won\n")
    with pytest.raises(IngestionError) as excinfo:
        ingest(source, "misbound", settings=settings, date_encodings={"Stage": SERIAL})
    detail = excinfo.value.detail
    assert isinstance(detail, DateEncodingInvalid)
    assert detail.reason == "bound_to_non_date_column"


# -- end to end ----------------------------------------------------------------


def test_a_serial_export_ingests_identically_to_the_iso_export(tmp_path, settings: Settings):
    iso_csv, serial_csv = tmp_path / "iso.csv", tmp_path / "serial.csv"
    write_production_csv(iso_csv, include_close_date=False)
    write_serial_production_csv(serial_csv)
    ingest(iso_csv, "iso", settings=settings, column_registry=OPPORTUNITY_SNAPSHOT_V1)
    ingest(serial_csv, "serial", settings=settings, column_registry=OPPORTUNITY_SNAPSHOT_V1)

    store = DuckDBStore(settings)
    with store.connect() as conn:
        a, b = store.snapshots_scan("iso"), store.snapshots_scan("serial")
        total = conn.execute(f"SELECT COUNT(*) FROM {b}").fetchone()[0]
        differing = conn.execute(
            f"SELECT COUNT(*) FROM {a} x JOIN {b} y USING (as_of, opp_id) "
            "WHERE x.close_date IS DISTINCT FROM y.close_date "
            "OR x.terminal_date IS DISTINCT FROM y.terminal_date "
            "OR x.deal_first_seen_close_date IS DISTINCT FROM y.deal_first_seen_close_date"
        ).fetchone()[0]
    assert total > 0
    assert differing == 0


def test_the_close_date_is_rebuilt_through_a_serial_as_of(tmp_path, settings: Settings):
    # If the reconstruction used a plain cast, every rebuilt date would be NULL
    # on exactly the exports that need rebuilding.
    source = tmp_path / "serial.csv"
    write_serial_production_csv(source)
    result = ingest(source, "recon", settings=settings, column_registry=OPPORTUNITY_SNAPSHOT_V1)
    store = DuckDBStore(settings)
    with store.connect() as conn:
        nulls = conn.execute(
            f"SELECT COUNT(*) FROM {store.snapshots_scan('recon')} WHERE close_date IS NULL"
        ).fetchone()[0]
    assert nulls == 0
    assert result.agreement is not None
    assert all(r.passed for r in result.agreement.results)


def test_a_serial_export_reports_no_false_cast_failures(tmp_path, settings: Settings):
    source = tmp_path / "serial.csv"
    write_serial_production_csv(source)
    result = ingest(source, "clean", settings=settings, column_registry=OPPORTUNITY_SNAPSHOT_V1)
    # Every serial would count as a failed cast if the declared columns were
    # measured as ordinary dates. A clean dataset must not report problems.
    assert result.schema.cast_failures == {}


def test_serial_provenance_is_recorded_on_the_schema(tmp_path, settings: Settings):
    source = tmp_path / "serial.csv"
    write_serial_production_csv(source)
    result = ingest(source, "prov", settings=settings, column_registry=OPPORTUNITY_SNAPSHOT_V1)
    conversions = {c.column: c for c in result.schema.date_conversions}

    as_of = conversions["as_of"]
    assert as_of.source_column == "as_of"
    assert as_of.encoding is SERIAL
    assert as_of.epoch == EXCEL_EPOCH
    assert as_of.serial_rows == result.row_count
    assert as_of.iso_rows == 0
    assert as_of.fractional_rows == result.row_count  # every as_of carries .4583333
    assert as_of.converted_from_serial
    assert "Time of day is discarded" in as_of.note
    assert as_of.min_date == date(2025, 1, 1)

    # A sparse column: nulls are counted apart from conversions.
    terminal = conversions["terminal_date"]
    assert terminal.serial_rows + terminal.null_rows == result.row_count
    assert terminal.fractional_rows == 0


def test_a_declared_discovered_date_column_is_typed_and_profiled_as_a_date(
    tmp_path, settings: Settings
):
    source = tmp_path / "serial.csv"
    write_serial_production_csv(source)
    dataset = register_dataset(
        source, "typed", column_registry=OPPORTUNITY_SNAPSHOT_V1, settings=settings
    )
    assert dataset.schema.discovered_types["terminal_date"] is DataType.DATE
    column = {c.name: c for c in dataset.profile.columns}["terminal_date"]
    assert column.kind is ProfileKind.DATE
    assert column.min_value and column.min_value.startswith("2025-")


def test_a_registry_declaration_for_an_absent_column_is_simply_skipped(
    tmp_path, settings: Settings
):
    # The registry declares five serial columns. A tenant that exports only some
    # of them must not be blocked by the ones it lacks.
    source = _write(tmp_path, "as_of,opp_id\n45973,O-1\n")
    result = ingest(source, "few", settings=settings, column_registry=OPPORTUNITY_SNAPSHOT_V1)
    assert [c.column for c in result.schema.date_conversions] == ["as_of"]


def test_the_canonical_as_of_stays_a_date_type(tmp_path, settings: Settings):
    source = tmp_path / "serial.csv"
    write_serial_production_csv(source)
    result = ingest(source, "type", settings=settings, column_registry=OPPORTUNITY_SNAPSHOT_V1)
    spec = {c.name: c for c in result.schema.columns}[CanonicalColumn.AS_OF]
    assert spec.dtype is DataType.DATE
