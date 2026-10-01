"""Diagnosing a real export before trusting it (ARCHITECTURE 12.19).

The probe answers the questions that decide whether a production export can
support close-date analysis: what `days_to_close` actually holds, whether the
reconstructed dates move, which stamped columns corroborate them, and what fiscal
calendar the quarter checks assumed.

**Aggregates only.** Every value this module emits is one of:

* a count, a rate, or a quantile;
* a category label for a low-cardinality *status candidate*, a 0/100 style flag,
  or a quarter-label shape, none of which identify anyone;
* a column name.

Identifiers (opportunity, account, owner) are reported as cardinality and null
count and never as values. Agreement samples are switched off, because a sample
is a row. A test asserts the rendered report contains no identifier from the data.

The probe reads; it never writes, never persists, and never promotes anything. A
column it lists as a status candidate is a candidate and nothing more: no name
match becomes a mapping (ARCHITECTURE 12.2).
"""

from __future__ import annotations

import re
from pathlib import Path

import duckdb
from pydantic import BaseModel, ConfigDict, Field

from ai_analyst.config import Settings, get_settings
from ai_analyst.contracts.schema import EXCEL_EPOCH, CanonicalColumn, DateEncoding
from ai_analyst.contracts.source import TableSource
from ai_analyst.data.conform import date_sql, quote_ident, quote_literal
from ai_analyst.data.reconstruct import (
    WHOLE_DAYS_PATTERN,
    _close_date_expression,
    evaluate_reconstruction_agreement,
    fiscal_quarter_label_sql,
)
from ai_analyst.data.sources import ProbeAccessError, connect_for, scan_sql

# The label format the quarter agreement tests compare against. It mirrors
# `FiscalCalendar.label`. Whether a real export stamps this shape is exactly what
# the probe reports rather than assumes.
ASSUMED_QUARTER_LABEL = "FY<year>-Q<n>"
ASSUMED_QUARTER_PATTERN = r"^FY[0-9]{4}-Q[1-4]$"

# Columns whose values identify a person or account. Counts only, ever.
IDENTIFYING_NAMES = frozenset(
    {"opp_id", "account_id", "ownerid", "owner_id", "renewalmanager", "opp_name", "account_name"}
)
STATUS_HINT = re.compile(r"(status|fate|outcome|stage|closed|won|lost|deleted|state)", re.I)
# Recency and timestamp derivatives are numeric measures, not status fields.
STATUS_EXCLUDE = re.compile(r"(_updated_days|_last_ts|_rolling|_count$|_amount)", re.I)
# A status candidate is listed by value only when it has this few of them.
MAX_LISTED_VALUES = 20
MAX_FLAG_VALUES = 8

QUANTILES = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
EOQ_CONVENTIONS: tuple[str, ...] = (
    "days_to_eoq - days_to_close",
    "days_to_close - days_to_eoq",
    "days_to_eoq - days_to_close + 1",
)


class ProbeAssumptions(BaseModel):
    """What the probe had to take as given, printed so nothing is implicit."""

    model_config = ConfigDict(frozen=True)

    as_of_encoding: str
    excel_epoch: str
    fiscal_year_start_month: int
    quarter_label_format: str
    fiscal_year_start_month_verified: bool = False
    note: str = (
        "The fiscal year start month is the configured default, not something the "
        "export establishes. The quarter checks are only as good as this assumption."
    )


class Extent(BaseModel):
    model_config = ConfigDict(frozen=True)

    rows: int
    distinct_opportunities: int
    snapshots: int
    first_snapshot: str | None = None
    last_snapshot: str | None = None
    as_of_serial_rows: int = 0
    as_of_iso_rows: int = 0
    as_of_invalid_rows: int = 0
    as_of_fractional_rows: int = 0
    duplicate_grain_keys: int = 0


class IdentifyingField(BaseModel):
    model_config = ConfigDict(frozen=True)

    column: str
    distinct: int
    nulls: int


class DaysToClose(BaseModel):
    model_config = ConfigDict(frozen=True)

    present: bool
    non_null: int = 0
    null: int = 0
    fractional: int = 0
    negative: int = 0
    zero: int = 0
    positive: int = 0
    quantiles: dict[str, float] = Field(default_factory=dict)


class StatusCandidate(BaseModel):
    """A column that might be a status field. A candidate, never a mapping."""

    model_config = ConfigDict(frozen=True)

    column: str
    distinct: int
    nulls: int
    # Category labels with counts, only when the column has few enough values.
    values: dict[str, int] | None = None
    note: str = "candidate only: not promoted, not mapped, chosen by nobody"


class DaysByGroup(BaseModel):
    model_config = ConfigDict(frozen=True)

    value: str
    rows: int
    null: int
    negative: int
    minimum: float | None = None
    median: float | None = None
    maximum: float | None = None


class Reconstruction(BaseModel):
    model_config = ConfigDict(frozen=True)

    evaluable: bool
    non_null: int = 0
    null: int = 0
    earliest: str | None = None
    latest: str | None = None
    opportunities_with_moving_date: int = 0
    opportunities_multi_snapshot: int = 0
    # Opportunities the export marks as pushed (cumulative counter above zero).
    pushed_opportunities: int | None = None
    pushed_with_moving_date: int | None = None
    # Consecutive snapshot pairs where the push counter rose, and what the
    # reconstructed date did across them. The counter is cumulative, so a pushed
    # opportunity that moved before the window can be legitimately still.
    counter_increases: int | None = None
    increases_with_later_date: int | None = None
    increases_with_earlier_date: int | None = None
    increases_with_unchanged_date: int | None = None


class AgreementLine(BaseModel):
    model_config = ConfigDict(frozen=True)

    test_id: str
    # validity or reconciliation (ARCHITECTURE 13.1): what a failure means.
    role: str
    checked_rows: int
    disagreeing_rows: int
    status: str
    detail: str = ""


class EoqDiagnostics(BaseModel):
    model_config = ConfigDict(frozen=True)

    present: bool
    non_null: int = 0
    negative: int = 0
    zero: int = 0
    positive: int = 0
    minimum: float | None = None
    maximum: float | None = None
    # Rows matching each candidate convention, out of rows where both sides exist.
    conventions: dict[str, tuple[int, int]] = Field(default_factory=dict)
    # For the closest convention: how far off each row is, as `delta -> rows`.
    # A near-match that is off by exactly one on many rows is a rounding effect,
    # which is a different problem from a wrong convention.
    closest_convention: str | None = None
    delta_histogram: dict[str, int] = Field(default_factory=dict)


class FlagDiagnostics(BaseModel):
    model_config = ConfigDict(frozen=True)

    column: str
    distinct: int
    values: dict[str, int] | None = None


class BoundaryBreakdown(BaseModel):
    """Where the CD_in_qtr disagreements sit relative to the quarter's edges."""

    model_config = ConfigDict(frozen=True)

    evaluable: bool
    disagreeing: int = 0
    flag_out_calendar_in: int = 0
    flag_in_calendar_out: int = 0
    on_quarter_first_day: int = 0
    on_quarter_last_day: int = 0
    interior: int = 0


class QuarterLabels(BaseModel):
    model_config = ConfigDict(frozen=True)

    column: str
    distinct: int
    # Labels with every digit masked to `d`, with counts: the *shape*, not the value.
    shapes: dict[str, int]
    matching_assumed_format: int
    non_null: int


class ProbeReport(BaseModel):
    """Everything the probe found. Aggregates only."""

    model_config = ConfigDict(frozen=True)

    source: str
    assumptions: ProbeAssumptions
    extent: Extent
    identifying: list[IdentifyingField]
    days_to_close: DaysToClose
    status_candidates: list[StatusCandidate]
    days_by_candidate: dict[str, list[DaysByGroup]]
    reconstruction: Reconstruction
    agreement: list[AgreementLine]
    eoq_close_diff: EoqDiagnostics
    cd_in_qtr_boundaries: BoundaryBreakdown
    flags: list[FlagDiagnostics]
    quarter_labels: list[QuarterLabels]
    not_evaluable: list[str]

    def render(self) -> str:
        out: list[str] = [f"PROBE {self.source}", "aggregates only; no row is printed", ""]

        a = self.assumptions
        out += [
            "== assumptions ==",
            f"  as_of encoding        : {a.as_of_encoding} (epoch {a.excel_epoch})",
            f"  fiscal year starts    : month {a.fiscal_year_start_month} (NOT verified)",
            f"  quarter label format  : {a.quarter_label_format}",
        ]

        e = self.extent
        out += [
            "",
            "== extent ==",
            f"  rows {e.rows}, opportunities {e.distinct_opportunities}, snapshots {e.snapshots}",
            f"  snapshot range        : {e.first_snapshot} .. {e.last_snapshot}",
            f"  as_of representation  : serial {e.as_of_serial_rows}, iso {e.as_of_iso_rows}, "
            f"invalid {e.as_of_invalid_rows}, time-of-day discarded {e.as_of_fractional_rows}",
            f"  duplicate grain keys  : {e.duplicate_grain_keys}",
        ]
        for f in self.identifying:
            out.append(f"  {f.column:<22}: {f.distinct} distinct, {f.nulls} null (counts only)")

        d = self.days_to_close
        out += ["", "== days_to_close =="]
        if not d.present:
            out.append("  column absent")
        else:
            out += [
                f"  non-null {d.non_null}, null {d.null}, fractional {d.fractional}",
                f"  negative {d.negative}, zero {d.zero}, positive {d.positive}",
                "  quantiles: " + ", ".join(f"p{k}={v:g}" for k, v in d.quantiles.items()),
            ]

        out += ["", "== status candidates (candidates only) =="]
        if not self.status_candidates:
            out.append("  none found")
        for c in self.status_candidates:
            shown = (
                ", ".join(f"{k}={v}" for k, v in c.values.items())
                if c.values is not None
                else f"{c.distinct} distinct values, not listed"
            )
            out.append(f"  {c.column}: {shown} (null {c.nulls})")
        for column, groups in self.days_by_candidate.items():
            out.append(f"  days_to_close by {column}:")
            for g in groups:
                out.append(
                    f"    {g.value:<22} rows {g.rows:>6} null {g.null:>5} neg {g.negative:>6} "
                    f"min {g.minimum} median {g.median} max {g.maximum}"
                )

        r = self.reconstruction
        out += ["", "== reconstruction: as_of + days_to_close =="]
        if not r.evaluable:
            out.append("  not evaluable: as_of or days_to_close absent")
        else:
            out += [
                f"  non-null {r.non_null}, null {r.null}; range {r.earliest} .. {r.latest}",
                f"  opportunities whose date moves: {r.opportunities_with_moving_date} of "
                f"{r.opportunities_multi_snapshot} seen in more than one snapshot",
            ]
            if r.pushed_opportunities is None:
                out.append("  push-count checks not evaluable: close_date_push_count absent")
            else:
                out += [
                    f"  pushed opportunities (counter > 0): {r.pushed_opportunities}, "
                    f"of which the date moves: {r.pushed_with_moving_date}",
                    f"  counter increases between snapshots: {r.counter_increases} -> date later "
                    f"{r.increases_with_later_date}, earlier {r.increases_with_earlier_date}, "
                    f"unchanged {r.increases_with_unchanged_date}",
                ]

        out += [
            "",
            "== agreement tests (validity failures withhold; reconciliation failures warn) ==",
        ]
        for line in self.agreement:
            out.append(
                f"  {line.test_id:<38} {line.role:<15} {line.status:<12} "
                f"checked {line.checked_rows} "
                f"disagree {line.disagreeing_rows} {line.detail}".rstrip()
            )

        q = self.eoq_close_diff
        out += ["", "== eoq_close_diff =="]
        if not q.present:
            out.append("  column absent")
        else:
            out += [
                f"  non-null {q.non_null}, negative {q.negative}, zero {q.zero}, "
                f"positive {q.positive}, range {q.minimum} .. {q.maximum}",
            ]
            for expr, (hit, total) in q.conventions.items():
                out.append(f"  equals {expr:<34}: {hit} of {total}")
            if q.delta_histogram:
                deltas = ", ".join(f"{k}: {v}" for k, v in q.delta_histogram.items())
                out.append(
                    f"  offset from ({q.closest_convention}), rows by delta: {deltas}"
                )

        b = self.cd_in_qtr_boundaries
        out += ["", "== CD_in_qtr disagreements vs the calendar =="]
        if not b.evaluable:
            out.append("  not evaluable: CD_in_qtr or the reconstructed close date is absent")
        else:
            out += [
                f"  disagreeing rows {b.disagreeing}: flag says out / calendar says in "
                f"{b.flag_out_calendar_in}, flag says in / calendar says out "
                f"{b.flag_in_calendar_out}",
                f"  on a quarter's first day {b.on_quarter_first_day}, last day "
                f"{b.on_quarter_last_day}, interior {b.interior}",
            ]

        out += ["", "== 0/1 style flags =="]
        for flag in self.flags:
            shown = (
                ", ".join(f"{k}={v}" for k, v in flag.values.items())
                if flag.values is not None
                else "not listed"
            )
            out.append(f"  {flag.column}: {flag.distinct} distinct: {shown}")

        out += ["", "== stamped quarter labels =="]
        for ql in self.quarter_labels:
            shapes = ", ".join(f"{k}={v}" for k, v in ql.shapes.items())
            out.append(
                f"  {ql.column}: {ql.distinct} distinct; shapes {shapes}; "
                f"{ql.matching_assumed_format} of {ql.non_null} match {ASSUMED_QUARTER_LABEL}"
            )
        if not self.quarter_labels:
            out.append("  no stamped quarter label column present")

        if self.not_evaluable:
            out += ["", "== not evaluable here =="]
            out += [f"  {item}" for item in self.not_evaluable]
        return "\n".join(out)


# ---------------------------------------------------------------------------


def _label(uri: str) -> str:
    """The file name only, so a path or bucket never lands in a report."""
    return Path(uri.rstrip("/")).name or "export"


def _one(conn: duckdb.DuckDBPyConnection, sql: str) -> tuple:
    return conn.execute(sql).fetchone()


def _counts(conn: duckdb.DuckDBPyConnection, ident: str) -> tuple[int, int]:
    """Distinct and null counts for one column. Never a value."""
    return _one(
        conn,
        f"SELECT COUNT(DISTINCT {ident}), COUNT(*) FILTER (WHERE {ident} IS NULL) FROM raw",
    )


def _iso(value: object) -> str | None:
    return None if value is None else str(value)[:10]


def probe_export(
    uri: str | TableSource,
    *,
    as_of_encoding: DateEncoding = DateEncoding.EXCEL_SERIAL,
    settings: Settings | None = None,
) -> ProbeReport:
    """Probe one export and return aggregates only.

    `uri` is a local or `s3://` CSV/Parquet path, a partitioned Parquet
    directory, or a Delta table. A plain string has its format inferred from
    its shape (ARCHITECTURE 12.19; `TableSource.infer`); an `s3://` location
    that is not obviously `.csv`/`.parquet` cannot be inferred and must be
    passed as an explicit `TableSource(format=..., uri=...)`.

    `as_of_encoding` is explicit and defaults to the export registry's own
    declaration for `as_of`. Nothing about the encoding is detected.
    """
    settings = settings or get_settings()
    fiscal_start = settings.fiscal_year_start_month
    source = uri if isinstance(uri, TableSource) else TableSource.infer(uri)
    conn = connect_for(source)
    conn.execute(f"CREATE VIEW raw AS SELECT * FROM {scan_sql(source)}")
    columns = {r[0]: r[1] for r in conn.execute("DESCRIBE SELECT * FROM raw").fetchall()}
    not_evaluable: list[str] = []

    for grain in ("as_of", "opp_id"):
        if grain not in columns:
            raise ValueError(f"the export has no {grain!r} column, so it cannot be probed")

    # -- the working view: as_of decoded, close_date rebuilt -----------------
    has_days = "days_to_close" in columns
    close = (
        _close_date_expression("as_of", "days_to_close", as_of_encoding) if has_days else None
    )
    replace = f"* REPLACE ({date_sql('as_of', as_of_encoding)} AS as_of)"
    conn.execute(
        f"CREATE VIEW t AS SELECT {replace}"
        + (f", {close} AS close_date" if close else "")
        + " FROM raw"
    )

    # -- extent --------------------------------------------------------------
    rows, opps, snaps, lo, hi = _one(
        conn,
        "SELECT COUNT(*), COUNT(DISTINCT opp_id), COUNT(DISTINCT as_of), MIN(as_of), MAX(as_of) "
        "FROM t",
    )
    duplicates = _one(
        conn,
        "SELECT COUNT(*) FROM (SELECT as_of, opp_id FROM t GROUP BY 1, 2 HAVING COUNT(*) > 1)",
    )[0]
    raw_text = f"NULLIF(TRIM(CAST({quote_ident('as_of')} AS VARCHAR)), '')"
    serial_re = quote_literal(r"^[+-]?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][+-]?[0-9]+)?$")
    invalid, serial, iso_rows, fractional = _one(
        conn,
        f"SELECT COUNT(*) FILTER (WHERE {raw_text} IS NOT NULL AND "
        f"{date_sql('as_of', as_of_encoding)} IS NULL), "
        f"COUNT(*) FILTER (WHERE regexp_matches({raw_text}, {serial_re})), "
        f"COUNT(*) FILTER (WHERE {raw_text} IS NOT NULL "
        f"AND NOT regexp_matches({raw_text}, {serial_re})), "
        f"COUNT(*) FILTER (WHERE regexp_matches({raw_text}, {serial_re}) "
        f"AND TRY_CAST({raw_text} AS DOUBLE) <> FLOOR(TRY_CAST({raw_text} AS DOUBLE))) "
        "FROM raw",
    )
    extent = Extent(
        rows=int(rows),
        distinct_opportunities=int(opps),
        snapshots=int(snaps),
        first_snapshot=_iso(lo),
        last_snapshot=_iso(hi),
        as_of_serial_rows=int(serial),
        as_of_iso_rows=int(iso_rows),
        as_of_invalid_rows=int(invalid),
        as_of_fractional_rows=int(fractional),
        duplicate_grain_keys=int(duplicates),
    )

    identifying = []
    for name in sorted(columns):
        if name.lower() in IDENTIFYING_NAMES:
            ident = quote_ident(name)
            distinct, nulls = _counts(conn, ident)
            identifying.append(
                IdentifyingField(column=name, distinct=int(distinct), nulls=int(nulls))
            )

    # -- days_to_close -------------------------------------------------------
    if has_days:
        days = "TRY_CAST(days_to_close AS DOUBLE)"
        non_null, null, frac, neg, zero, pos = _one(
            conn,
            f"SELECT COUNT({days}), COUNT(*) FILTER (WHERE {days} IS NULL), "
            f"COUNT(*) FILTER (WHERE {days} <> FLOOR({days})), "
            f"COUNT(*) FILTER (WHERE {days} < 0), COUNT(*) FILTER (WHERE {days} = 0), "
            f"COUNT(*) FILTER (WHERE {days} > 0) FROM raw",
        )
        qs = _one(conn, f"SELECT quantile_cont({days}, {list(QUANTILES)}) FROM raw")[0] or []
        dtc = DaysToClose(
            present=True,
            non_null=int(non_null),
            null=int(null),
            fractional=int(frac),
            negative=int(neg),
            zero=int(zero),
            positive=int(pos),
            quantiles={f"{int(q * 100)}": float(v) for q, v in zip(QUANTILES, qs, strict=True)},
        )
    else:
        dtc = DaysToClose(present=False)
        not_evaluable.append("days_to_close is absent: the close date cannot be reconstructed")

    # -- status candidates and days_to_close behaviour by candidate ----------
    candidates: list[StatusCandidate] = []
    by_candidate: dict[str, list[DaysByGroup]] = {}
    for name, dtype in columns.items():
        if not STATUS_HINT.search(name) or STATUS_EXCLUDE.search(name):
            continue
        if dtype.upper() not in {"VARCHAR", "BOOLEAN"}:
            continue
        ident = quote_ident(name)
        distinct, nulls = _counts(conn, ident)
        if int(distinct) > 50:
            continue
        values = None
        if int(distinct) <= MAX_LISTED_VALUES:
            values = {
                str(v): int(n)
                for v, n in conn.execute(
                    f"SELECT CAST({ident} AS VARCHAR), COUNT(*) FROM raw "
                    f"WHERE {ident} IS NOT NULL GROUP BY 1 ORDER BY 2 DESC, 1"
                ).fetchall()
            }
        candidates.append(
            StatusCandidate(column=name, distinct=int(distinct), nulls=int(nulls), values=values)
        )
        if has_days and values is not None:
            groups = conn.execute(
                f"SELECT CAST({ident} AS VARCHAR), COUNT(*), "
                f"COUNT(*) FILTER (WHERE TRY_CAST(days_to_close AS DOUBLE) IS NULL), "
                f"COUNT(*) FILTER (WHERE TRY_CAST(days_to_close AS DOUBLE) < 0), "
                f"MIN(TRY_CAST(days_to_close AS DOUBLE)), "
                f"MEDIAN(TRY_CAST(days_to_close AS DOUBLE)), "
                f"MAX(TRY_CAST(days_to_close AS DOUBLE)) "
                f"FROM raw WHERE {ident} IS NOT NULL GROUP BY 1 ORDER BY 2 DESC, 1"
            ).fetchall()
            by_candidate[name] = [
                DaysByGroup(
                    value=str(g[0]), rows=int(g[1]), null=int(g[2]), negative=int(g[3]),
                    minimum=g[4], median=g[5], maximum=g[6],
                )
                for g in groups
            ]

    # -- reconstruction ------------------------------------------------------
    if has_days:
        nn, nl, early, late = _one(
            conn,
            "SELECT COUNT(close_date), COUNT(*) FILTER (WHERE close_date IS NULL), "
            "MIN(close_date), MAX(close_date) FROM t",
        )
        moving, multi = _one(
            conn,
            "SELECT COUNT(*) FILTER (WHERE d > 1), COUNT(*) FROM ("
            "SELECT opp_id, COUNT(DISTINCT close_date) AS d FROM t GROUP BY 1 "
            "HAVING COUNT(DISTINCT as_of) > 1)",
        )
        pushed = moved_pushed = inc = later = earlier = same = None
        if "close_date_push_count" in columns:
            pc = "TRY_CAST(close_date_push_count AS BIGINT)"
            pushed, moved_pushed = _one(
                conn,
                f"SELECT COUNT(*), COUNT(*) FILTER (WHERE d > 1) FROM ("
                f"SELECT opp_id, COUNT(DISTINCT close_date) AS d FROM t "
                f"WHERE opp_id IN (SELECT opp_id FROM t WHERE {pc} > 0) GROUP BY 1 "
                f"HAVING COUNT(DISTINCT as_of) > 1)",
            )
            inc, later, earlier, same = _one(
                conn,
                f"WITH s AS (SELECT close_date, {pc} AS pc, "
                f"LAG(close_date) OVER w AS prev_date, LAG({pc}) OVER w AS prev_pc "
                f"FROM t WINDOW w AS (PARTITION BY opp_id ORDER BY as_of)) "
                f"SELECT COUNT(*) FILTER (WHERE pc > prev_pc), "
                f"COUNT(*) FILTER (WHERE pc > prev_pc AND close_date > prev_date), "
                f"COUNT(*) FILTER (WHERE pc > prev_pc AND close_date < prev_date), "
                f"COUNT(*) FILTER (WHERE pc > prev_pc AND close_date = prev_date) FROM s",
            )
        reconstruction = Reconstruction(
            evaluable=True,
            non_null=int(nn),
            null=int(nl),
            earliest=_iso(early),
            latest=_iso(late),
            opportunities_with_moving_date=int(moving),
            opportunities_multi_snapshot=int(multi),
            pushed_opportunities=None if pushed is None else int(pushed),
            pushed_with_moving_date=None if moved_pushed is None else int(moved_pushed),
            counter_increases=None if inc is None else int(inc),
            increases_with_later_date=None if later is None else int(later),
            increases_with_earlier_date=None if earlier is None else int(earlier),
            increases_with_unchanged_date=None if same is None else int(same),
        )
    else:
        reconstruction = Reconstruction(evaluable=False)

    # -- agreement tests, samples off ----------------------------------------
    agreement_lines: list[AgreementLine] = []
    if has_days:
        report = evaluate_reconstruction_agreement(
            conn,
            "t",
            dataset_id="probe",
            reconstructed={CanonicalColumn.CLOSE_DATE},
            fiscal_year_start_month=fiscal_start,
            max_samples=0,
        )
        for result in report.results:
            status = (
                "skipped" if result.skipped
                else "passed" if result.passed
                else "FAILED" if result.failed
                else "inconclusive"
            )
            agreement_lines.append(
                AgreementLine(
                    test_id=result.test_id,
                    role=result.role.value,
                    checked_rows=result.checked_rows,
                    disagreeing_rows=result.disagreeing_rows,
                    status=status,
                    detail=result.skip_reason if result.skipped else "",
                )
            )
            if result.skipped:
                not_evaluable.append(f"{result.test_id}: {result.skip_reason}")

    # -- eoq_close_diff ------------------------------------------------------
    if "eoq_close_diff" in columns:
        e = "TRY_CAST(eoq_close_diff AS DOUBLE)"
        nn, neg, zero, pos, lo_e, hi_e = _one(
            conn,
            f"SELECT COUNT({e}), COUNT(*) FILTER (WHERE {e} < 0), COUNT(*) FILTER (WHERE {e} = 0), "
            f"COUNT(*) FILTER (WHERE {e} > 0), MIN({e}), MAX({e}) FROM raw",
        )
        conventions: dict[str, tuple[int, int]] = {}
        closest: str | None = None
        histogram: dict[str, int] = {}
        if has_days and "days_to_eoq" in columns:
            for expr in EOQ_CONVENTIONS:
                sql = expr.replace("days_to_eoq", "TRY_CAST(days_to_eoq AS DOUBLE)").replace(
                    "days_to_close", "TRY_CAST(days_to_close AS DOUBLE)"
                )
                hit, total = _one(
                    conn,
                    f"SELECT COUNT(*) FILTER (WHERE {e} = {sql}), COUNT(*) "
                    f"FROM raw WHERE {e} IS NOT NULL AND {sql} IS NOT NULL",
                )
                conventions[expr] = (int(hit), int(total))
            closest = max(conventions, key=lambda k: conventions[k][0])
            base = closest.replace("days_to_eoq", "TRY_CAST(days_to_eoq AS DOUBLE)").replace(
                "days_to_close", "TRY_CAST(days_to_close AS DOUBLE)"
            )
            histogram = {
                f"{int(d):+d}": int(n)
                for d, n in conn.execute(
                    f"SELECT {e} - ({base}), COUNT(*) FROM raw "
                    f"WHERE {e} IS NOT NULL AND ({base}) IS NOT NULL "
                    f"GROUP BY 1 ORDER BY 2 DESC LIMIT 6"
                ).fetchall()
            }
        else:
            not_evaluable.append("eoq_close_diff convention: days_to_eoq or days_to_close absent")
        eoq = EoqDiagnostics(
            present=True, non_null=int(nn), negative=int(neg), zero=int(zero),
            positive=int(pos), minimum=lo_e, maximum=hi_e, conventions=conventions,
            closest_convention=closest, delta_histogram=histogram,
        )
    else:
        eoq = EoqDiagnostics(present=False)
        not_evaluable.append("eoq_close_diff is absent: its convention cannot be checked")

    # -- where CD_in_qtr disagrees, relative to the quarter's edges ------------
    boundaries = BoundaryBreakdown(evaluable=False)
    if has_days and "CD_in_qtr" in columns:
        start = int(fiscal_start)
        in_flag = "CAST(CD_in_qtr AS BOOLEAN)"
        calendar_in = (
            f"({fiscal_quarter_label_sql('close_date', start)} = "
            f"{fiscal_quarter_label_sql('as_of', start)})"
        )
        first_day = f"(DAY(close_date) = 1 AND ((MONTH(close_date) - {start}) % 12 + 12) % 3 = 0)"
        nxt = "(close_date + 1)"
        last_day = f"(DAY({nxt}) = 1 AND ((MONTH({nxt}) - {start}) % 12 + 12) % 3 = 0)"
        row = _one(
            conn,
            f"SELECT COUNT(*) FILTER (WHERE {in_flag} <> {calendar_in}), "
            f"COUNT(*) FILTER (WHERE NOT {in_flag} AND {calendar_in}), "
            f"COUNT(*) FILTER (WHERE {in_flag} AND NOT {calendar_in}), "
            f"COUNT(*) FILTER (WHERE {in_flag} <> {calendar_in} AND {first_day}), "
            f"COUNT(*) FILTER (WHERE {in_flag} <> {calendar_in} AND {last_day}) "
            f"FROM t WHERE close_date IS NOT NULL AND CD_in_qtr IS NOT NULL",
        )
        total, out_in, in_out, first_n, last_n = (int(v) for v in row)
        boundaries = BoundaryBreakdown(
            evaluable=True,
            disagreeing=total,
            flag_out_calendar_in=out_in,
            flag_in_calendar_out=in_out,
            on_quarter_first_day=first_n,
            on_quarter_last_day=last_n,
            interior=max(0, total - first_n - last_n),
        )

    # -- flags: report the value set before any convention is asserted -------
    flags = []
    for name in ("CD_in_qtr", "CD_in_past"):
        if name not in columns:
            continue
        ident = quote_ident(name)
        distinct = int(_one(conn, f"SELECT COUNT(DISTINCT {ident}) FROM raw")[0])
        values = None
        if distinct <= MAX_FLAG_VALUES:
            values = {
                str(v): int(n)
                for v, n in conn.execute(
                    f"SELECT CAST({ident} AS VARCHAR), COUNT(*) FROM raw "
                    f"WHERE {ident} IS NOT NULL GROUP BY 1 ORDER BY 1"
                ).fetchall()
            }
        flags.append(FlagDiagnostics(column=name, distinct=distinct, values=values))

    # -- stamped quarter labels: shape only ----------------------------------
    quarter_labels = []
    for name in ("close_date_qtr", "as_of_qtr"):
        if name not in columns:
            continue
        ident = quote_ident(name)
        nn = int(_one(conn, f"SELECT COUNT({ident}) FROM raw")[0])
        distinct = int(_one(conn, f"SELECT COUNT(DISTINCT {ident}) FROM raw")[0])
        shapes = {
            str(s): int(n)
            for s, n in conn.execute(
                f"SELECT regexp_replace(CAST({ident} AS VARCHAR), '[0-9]', 'd', 'g'), COUNT(*) "
                f"FROM raw WHERE {ident} IS NOT NULL GROUP BY 1 ORDER BY 2 DESC LIMIT 6"
            ).fetchall()
        }
        matching = int(
            _one(
                conn,
                f"SELECT COUNT(*) FILTER (WHERE regexp_matches(CAST({ident} AS VARCHAR), "
                f"{quote_literal(ASSUMED_QUARTER_PATTERN)})) FROM raw",
            )[0]
        )
        quarter_labels.append(
            QuarterLabels(
                column=name, distinct=distinct, shapes=shapes,
                matching_assumed_format=matching, non_null=nn,
            )
        )

    return ProbeReport(
        source=_label(source.uri),
        assumptions=ProbeAssumptions(
            as_of_encoding=as_of_encoding.value,
            excel_epoch=EXCEL_EPOCH,
            fiscal_year_start_month=fiscal_start,
            quarter_label_format=ASSUMED_QUARTER_LABEL,
        ),
        extent=extent,
        identifying=identifying,
        days_to_close=dtc,
        status_candidates=candidates,
        days_by_candidate=by_candidate,
        reconstruction=reconstruction,
        agreement=agreement_lines,
        eoq_close_diff=eoq,
        cd_in_qtr_boundaries=boundaries,
        flags=flags,
        quarter_labels=quarter_labels,
        not_evaluable=not_evaluable,
    )


__all__ = ["ProbeAccessError", "ProbeReport", "probe_export", "WHOLE_DAYS_PATTERN"]
