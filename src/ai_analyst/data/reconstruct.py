"""Reconstructing a canonical column a tenant did not export (ARCHITECTURE 12.15).

The production export carries no per-snapshot close date. It carries
`days_to_close`, confirmed by the project owner as the snapshot's own expected
close date minus `as_of`, not derived from `terminal_date`. The close date is
therefore rebuildable as `as_of + days_to_close`.

Two rules govern doing that:

* **Nothing pretends the source contained the column.** A reconstructed column
  is recorded as `DerivationRule.RECONSTRUCTED` with its expression, its source
  columns, and the agreement tests that checked it.
* **The reconstruction is checked, not assumed.** If an agreement test fails,
  the physical column still exists but the `expected_close_date` concept stays
  unavailable, so no metric can read a date that is demonstrably wrong.

The derivation itself comes from the export registry's coverage record, not
from this module guessing a formula. A registry that declares no derivation
gets no reconstruction.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from ai_analyst.contracts.agreement import AgreementKind, AgreementRole, AgreementTest
from ai_analyst.contracts.columns import ColumnRegistry, CoverageStatus
from ai_analyst.contracts.concepts import BusinessConcept
from ai_analyst.contracts.schema import (
    CANONICAL_COLUMNS,
    CanonicalColumn,
    DataType,
    DateEncoding,
    MappingProposal,
)
from ai_analyst.data.conform import date_sql, quote_ident, quote_literal

# The one derivation this module knows how to build. Written exactly as the
# export registry states it, so a registry declaring something else is not
# quietly reinterpreted.
CLOSE_DATE_DERIVATION = "as_of + days_to_close"


class Reconstruction(BaseModel):
    """A canonical column rebuilt from other source columns."""

    model_config = ConfigDict(frozen=True)

    target: CanonicalColumn
    dtype: DataType
    # SQL over the *raw source* columns, used in the conform SELECT.
    expression_sql: str
    # Human-readable derivation, as the registry declared it.
    derivation: str
    sources: tuple[str, ...]
    note: str = ""
    # Who declared the derivation. A reconstruction nobody accountable declared
    # is `UNDECLARED` and never available (ARCHITECTURE 13.1 rule 1).
    declared_by: str | None = None

    @property
    def agreement_test_ids(self) -> tuple[str, ...]:
        """Every test that checks this reconstruction, including the per-opportunity one.

        Scoped to the target, so a future derivation does not inherit the close
        date's tests, and it includes `close_date_moves_for_pushed_deals`, which
        is built separately because it is evaluated per opportunity rather than
        per row. Recording three of four would under-report the provenance of
        exactly the discriminating check.
        """
        if self.target is not CanonicalColumn.CLOSE_DATE:
            return ()
        return (
            *(t.id for t in close_date_agreement_tests()),
            close_date_movement_test().id,
        )


# A whole-number day count as text, with an optional `.0` because a DOUBLE column
# prints 5 as 5.0. Anything fractional is refused rather than rounded.
WHOLE_DAYS_PATTERN = r"^[+-]?[0-9]+(\.0*)?$"
# DuckDB's DATE spans roughly +/-5.8 million years, and adding more than this
# many days to a real date would raise instead of yielding NULL. Two million days
# is about 5,470 years, far outside any real horizon.
MAX_ABS_DAYS = 2_000_000


def _close_date_expression(
    as_of_source: str,
    days_source: str,
    as_of_encoding: DateEncoding = DateEncoding.ISO,
) -> str:
    """`as_of + days_to_close`, as DuckDB SQL over raw source columns.

    `as_of` goes through `conform.date_sql`, the same conversion every other date
    uses. Without that, a serial `as_of` casts to NULL and every reconstructed
    close date is NULL, so every agreement test would report zero rows and the
    concept would be withheld for a reason that had nothing to do with the data.

    `days_to_close` must be a whole number. A fractional value is NULL, not
    rounded: DuckDB rounds on cast, so `TRY_CAST('2.6' AS INTEGER)` is 3, which
    would move a close date by a day and never announce it. Whether the real
    column is ever fractional is an open question (`days_to_close_edge_cases`),
    and the answer to an open question is not a guess.
    """
    as_of = date_sql(as_of_source, as_of_encoding)
    text = f"NULLIF(TRIM(CAST({quote_ident(days_source)} AS VARCHAR)), '')"
    number = f"TRY_CAST({text} AS DOUBLE)"
    days = (
        f"CASE WHEN regexp_matches({text}, {quote_literal(WHOLE_DAYS_PATTERN)}) "
        f"AND ABS({number}) <= {MAX_ABS_DAYS} THEN CAST({number} AS INTEGER) END"
    )
    return f"({as_of} + {days})"


def plan_reconstructions(
    registry: ColumnRegistry | None,
    mapping: MappingProposal,
    source_columns: list[str],
    date_encodings: dict[str, DateEncoding] | None = None,
) -> dict[CanonicalColumn, Reconstruction]:
    """Which canonical columns can be rebuilt for this dataset.

    Only columns the registry declares `RECONSTRUCTIBLE`, whose declared
    derivation this module implements, and whose source columns are present.
    """
    if registry is None:
        return {}
    available = set(source_columns)
    bound = set(mapping.by_canonical())
    plans: dict[CanonicalColumn, Reconstruction] = {}

    for record in registry.coverage:
        if record.status is not CoverageStatus.RECONSTRUCTIBLE:
            continue
        target = CanonicalColumn(record.canonical)
        if target in bound:
            continue  # the export carried it after all; never override a real column
        if target is not CanonicalColumn.CLOSE_DATE:
            continue  # no other derivation is implemented
        if (record.derivation or "").strip() != CLOSE_DATE_DERIVATION:
            continue  # the registry declares something this module cannot build

        as_of_mapping = mapping.by_canonical().get(CanonicalColumn.AS_OF)
        days_source = record.source or "days_to_close"
        if as_of_mapping is None or days_source not in available:
            continue

        plans[target] = Reconstruction(
            target=target,
            dtype=CANONICAL_COLUMNS[target].dtype,
            expression_sql=_close_date_expression(
                as_of_mapping.source_column,
                days_source,
                (date_encodings or {}).get(as_of_mapping.source_column, DateEncoding.ISO),
            ),
            derivation=CLOSE_DATE_DERIVATION,
            sources=(as_of_mapping.source_column, days_source),
            declared_by=f"export_registry:{registry.name}",
            note=(
                "Rebuilt because the export carries a horizon in days rather than a "
                "date. The source did not contain a close date."
            ),
        )
    return plans


def fiscal_quarter_label_sql(column: str, fiscal_year_start_month: int) -> str:
    """The fiscal quarter label of a DATE column, as `FYyyyy-Qn`.

    Mirrors `semantic.calendar.FiscalCalendar` exactly: the fiscal year is named
    for the calendar year it begins in, so a date before the start month belongs
    to the previous fiscal year.
    """
    d = quote_ident(column)
    start = int(fiscal_year_start_month)
    year = f"(CASE WHEN MONTH({d}) >= {start} THEN YEAR({d}) ELSE YEAR({d}) - 1 END)"
    # FLOOR, not CAST: DuckDB's `/` is true division and CAST rounds to
    # nearest, so month 6 would land in Q3 instead of Q2. This is the same
    # rounding trap that makes TRY_CAST('1.5' AS BIGINT) return 2 (5.15).
    quarter = f"FLOOR((((MONTH({d}) - {start}) % 12 + 12) % 12) / 3) + 1"
    ordinal = f"CAST(CAST({quarter} AS INTEGER) AS VARCHAR)"
    return f"('FY' || CAST({year} AS VARCHAR) || '-Q' || {ordinal})"


def close_date_agreement_tests(fiscal_year_start_month: int = 1) -> list[AgreementTest]:
    """Every check the export can make on a reconstructed close date.

    Each is independent of the derivation itself, so agreement is genuine
    corroboration rather than a restatement. A test whose columns are absent is
    skipped, and a skipped test never counts as a pass.
    """
    close = quote_ident(CanonicalColumn.CLOSE_DATE.value)
    label = fiscal_quarter_label_sql(CanonicalColumn.CLOSE_DATE.value, fiscal_year_start_month)
    as_of_label = fiscal_quarter_label_sql(
        CanonicalColumn.AS_OF.value, fiscal_year_start_month
    )
    concept = BusinessConcept.EXPECTED_CLOSE_DATE

    days = quote_ident("days_to_close")
    return [
        AgreementTest(
            id="close_date_is_structurally_valid",
            kind=AgreementKind.COLUMNS_SATISFY_RELATIONSHIP,
            concept=concept,
            assertion=(
                "Every row with a days_to_close value reconstructs to a real calendar "
                "date: the horizon is a whole number of days within range and as_of "
                "decoded under its declared encoding."
            ),
            predicate_sql=f"{close} IS NOT NULL",
            scope_sql=f"NULLIF(TRIM(CAST({days} AS VARCHAR)), '') IS NOT NULL",
            requires_columns=(CanonicalColumn.CLOSE_DATE.value, "days_to_close"),
            sample_columns=(CanonicalColumn.OPP_ID.value, CanonicalColumn.AS_OF.value),
            role=AgreementRole.VALIDITY,
        ),
        AgreementTest(
            id="close_date_matches_close_date_qtr",
            kind=AgreementKind.COLUMN_CONSISTENT_WITH_CONCEPT,
            concept=concept,
            assertion=(
                "The fiscal quarter of the reconstructed close date equals the "
                "exported close_date_qtr label."
            ),
            predicate_sql=f"{label} = {quote_ident('close_date_qtr')}",
            # The stamped label's convention is undeclared, and comparing it
            # needs the fiscal calendar.
            role=AgreementRole.RECONCILIATION,
            calendar_dependent=True,
            scope_sql=f"{close} IS NOT NULL AND {quote_ident('close_date_qtr')} IS NOT NULL",
            requires_columns=(CanonicalColumn.CLOSE_DATE.value, "close_date_qtr"),
            sample_columns=(
                CanonicalColumn.OPP_ID.value,
                CanonicalColumn.AS_OF.value,
                CanonicalColumn.CLOSE_DATE.value,
                "close_date_qtr",
            ),
        ),
        AgreementTest(
            id="close_date_matches_cd_in_qtr",
            kind=AgreementKind.COLUMN_CONSISTENT_WITH_CONCEPT,
            concept=concept,
            assertion=(
                "CD_in_qtr agrees with whether the reconstructed close date falls "
                "in the snapshot's own fiscal quarter."
            ),
            predicate_sql=(
                f"CAST({quote_ident('CD_in_qtr')} AS BOOLEAN) = ({label} = {as_of_label})"
            ),
            role=AgreementRole.RECONCILIATION,
            calendar_dependent=True,
            scope_sql=f"{close} IS NOT NULL AND {quote_ident('CD_in_qtr')} IS NOT NULL",
            requires_columns=(CanonicalColumn.CLOSE_DATE.value, "CD_in_qtr"),
            sample_columns=(
                CanonicalColumn.OPP_ID.value,
                CanonicalColumn.CLOSE_DATE.value,
                "CD_in_qtr",
            ),
        ),
        AgreementTest(
            id="close_date_matches_eoq_close_diff",
            kind=AgreementKind.COLUMNS_SATISFY_RELATIONSHIP,
            concept=concept,
            assertion=(
                "eoq_close_diff equals days_to_eoq minus days_to_close, the "
                "quarter-end offset of the reconstructed close date."
            ),
            role=AgreementRole.RECONCILIATION,
            predicate_sql=(
                f"CAST({quote_ident('eoq_close_diff')} AS BIGINT) = "
                f"CAST({quote_ident('days_to_eoq')} AS BIGINT) - "
                f"CAST({quote_ident('days_to_close')} AS BIGINT)"
            ),
            scope_sql=(
                f"{quote_ident('eoq_close_diff')} IS NOT NULL AND "
                f"{quote_ident('days_to_eoq')} IS NOT NULL AND "
                f"{quote_ident('days_to_close')} IS NOT NULL"
            ),
            requires_columns=("eoq_close_diff", "days_to_eoq", "days_to_close"),
            sample_columns=("opp_id", "eoq_close_diff", "days_to_eoq", "days_to_close"),
        ),
    ]


def close_date_movement_test() -> AgreementTest:
    """The discriminating test: a pushed deal's close date must move.

    If `days_to_close` were measured against `terminal_date` rather than the
    snapshot's own close date, `as_of + days_to_close` would be the *same* date
    at every snapshot, every slip would vanish, and every slip metric would
    return zero while looking healthy. An opportunity the export says was pushed
    is exactly the case that separates the two readings.

    Unlike the row-level tests this one is evaluated per opportunity, so it is
    built by `run_close_date_movement_test` rather than by a row predicate.
    """
    return AgreementTest(
        id="close_date_moves_for_pushed_deals",
        kind=AgreementKind.COLUMNS_SATISFY_RELATIONSHIP,
        concept=BusinessConcept.EXPECTED_CLOSE_DATE,
        assertion=(
            "Where close_date_push_count increases between two adjacent "
            "snapshots, the reconstructed close date moves between them."
        ),
        predicate_sql="TRUE",
        role=AgreementRole.VALIDITY,
        requires_columns=(
            CanonicalColumn.CLOSE_DATE.value,
            CanonicalColumn.OPP_ID.value,
            "close_date_push_count",
        ),
        sample_columns=(CanonicalColumn.OPP_ID.value,),
    )


def run_close_date_movement_test(
    conn, scan: str, available_columns: set[str], *, max_samples: int = 5
):
    """Evaluate the movement test, per adjacent snapshot pair.

    `close_date_push_count` is **cumulative since the opportunity was created**,
    so its absolute value says nothing about the export window. An opportunity
    created three years before an eight-quarter export can carry a count of 5
    and never move its close date inside the window: every one of those pushes
    happened before the first snapshot. Scoping on "count above zero" and then
    demanding movement therefore fails such an opportunity for a push the
    export does not contain, which is a false negative about the reading, not a
    finding about the data.

    The scope is instead every adjacent snapshot pair where the counter
    **actually increased** inside the window. That increase is an event the
    export does describe, and it is exactly the case that discriminates the two
    readings of `days_to_close`: if the horizon were measured against
    `terminal_date` rather than the snapshot's own close date, the
    reconstructed date would be identical at both snapshots of that pair, every
    slip would vanish, and every slip metric would return zero while looking
    healthy.

    With no within-window increase the test is **skipped**, never failed: the
    export simply does not contain the case, and an unverified reading is not a
    disproved one.

    The comparison is normalized to calendar dates. The close date is already a
    DATE by the time it is written, so this changes nothing today; it is
    written explicitly so the check stays date-based if a tenant's close date
    ever arrives as a timestamp, where an 11:00 time of day would otherwise
    make two equal dates compare unequal.
    """
    from ai_analyst.contracts.agreement import AgreementResult, AgreementSample

    test = close_date_movement_test()
    missing = [c for c in test.requires_columns if c not in available_columns]
    if missing:
        return AgreementResult(
            test_id=test.id,
            assertion=test.assertion,
            kind=test.kind,
            concept=test.concept,
            role=test.role,
            calendar_dependent=test.calendar_dependent,
            skipped=True,
            skip_reason=f"columns absent from this dataset: {', '.join(sorted(missing))}",
        )

    opp = quote_ident(CanonicalColumn.OPP_ID.value)
    close = f"CAST({quote_ident(CanonicalColumn.CLOSE_DATE.value)} AS DATE)"
    pushed = f"TRY_CAST({quote_ident('close_date_push_count')} AS BIGINT)"
    # One row per adjacent snapshot pair, carrying both sides of the pair.
    pairs = (
        f"SELECT {opp} AS opp_id, as_of, {close} AS close_date, {pushed} AS pc, "
        f"LAG({close}) OVER w AS prev_close, "
        f"LAG({pushed}) OVER w AS prev_pc, "
        f"LAG(as_of) OVER w AS prev_as_of "
        f"FROM {scan} "
        f"WINDOW w AS (PARTITION BY {opp} ORDER BY as_of)"
    )
    # The scope: pairs where the cumulative counter rose inside the window.
    scoped = (
        f"SELECT * FROM ({pairs}) "
        f"WHERE prev_pc IS NOT NULL AND pc IS NOT NULL AND pc > prev_pc"
    )
    checked, stuck = conn.execute(
        f"SELECT COUNT(*), "
        f"COUNT(*) FILTER (WHERE close_date IS NOT DISTINCT FROM prev_close) "
        f"FROM ({scoped})"
    ).fetchone()

    if not int(checked):
        return AgreementResult(
            test_id=test.id,
            assertion=test.assertion,
            kind=test.kind,
            concept=test.concept,
            role=test.role,
            calendar_dependent=test.calendar_dependent,
            skipped=True,
            skip_reason=(
                "close_date_push_count never increases between adjacent snapshots "
                "in this export, so the window contains no push to check. The "
                "counter is cumulative since creation, and a nonzero value alone "
                "does not mean the push happened inside the window."
            ),
        )

    samples: tuple[AgreementSample, ...] = ()
    if int(stuck):
        rows = conn.execute(
            f"SELECT opp_id, prev_as_of, as_of, prev_pc, pc FROM ({scoped}) "
            f"WHERE close_date IS NOT DISTINCT FROM prev_close LIMIT {int(max_samples)}"
        ).fetchall()
        samples = tuple(
            AgreementSample(
                values={
                    "opp_id": str(r[0]),
                    "from_as_of": str(r[1]),
                    "to_as_of": str(r[2]),
                    "push_count": f"{r[3]} -> {r[4]}",
                }
            )
            for r in rows
        )

    return AgreementResult(
        test_id=test.id,
        assertion=test.assertion,
        kind=test.kind,
        concept=test.concept,
        role=test.role,
            calendar_dependent=test.calendar_dependent,
        checked_rows=int(checked),
        disagreeing_rows=int(stuck),
        samples=samples,
        note=(
            "Scoped to adjacent snapshot pairs where close_date_push_count "
            "actually increased, because the counter is cumulative since "
            "creation and a push before the export window is not checkable. A "
            "close date that does not move across such a pair would mean "
            "days_to_close was measured against the terminal date, which would "
            "make every slip metric silently return zero."
        ),
    )


def evaluate_reconstruction_agreement(
    conn,
    scan: str,
    *,
    dataset_id: str,
    reconstructed: set[CanonicalColumn],
    fiscal_year_start_month: int = 1,
    max_samples: int = 5,
):
    """Run every agreement test that checks a reconstructed column.

    Only reconstructed columns are checked. A dataset that carried its own close
    date has nothing to corroborate, so no test runs and no report is produced,
    rather than tests that could only pass trivially.
    """
    from ai_analyst.contracts.agreement import AgreementReport
    from ai_analyst.data.agreement import run_agreement_tests

    if CanonicalColumn.CLOSE_DATE not in reconstructed:
        return AgreementReport(
            dataset_id=dataset_id, fiscal_year_start_month=fiscal_year_start_month
        )

    available = {r[0] for r in conn.execute(f"DESCRIBE SELECT * FROM {scan}").fetchall()}
    report = run_agreement_tests(
        conn,
        scan,
        close_date_agreement_tests(fiscal_year_start_month),
        available,
        dataset_id=dataset_id,
        fiscal_year_start_month=fiscal_year_start_month,
        max_samples=max_samples,
    )
    movement = run_close_date_movement_test(conn, scan, available, max_samples=max_samples)
    results = tuple(
        _with_eoq_diagnostics(conn, scan, r) if r.test_id == EOQ_TEST_ID else r
        for r in report.results
    )
    return AgreementReport(
        dataset_id=dataset_id,
        fiscal_year_start_month=fiscal_year_start_month,
        results=(*results, movement),
    )


EOQ_TEST_ID = "close_date_matches_eoq_close_diff"


def _with_eoq_diagnostics(conn, scan: str, result):
    """Attach the offset histogram of eoq_close_diff, when it disagrees.

    The histogram is what an owner needs to decide the field's convention, and
    it is reported rather than used: a "+1 on 38% of rows" pattern is evidence
    about a convention, never a reason to pass the test (13.1 rule 6).
    """
    if not result.failed:
        return result
    offset = (
        f"CAST({quote_ident('eoq_close_diff')} AS BIGINT) - "
        f"(CAST({quote_ident('days_to_eoq')} AS BIGINT) - "
        f"CAST({quote_ident('days_to_close')} AS BIGINT))"
    )
    rows = conn.execute(
        f"SELECT {offset} AS d, COUNT(*) FROM {scan} "
        f"WHERE {offset} IS NOT NULL GROUP BY 1 ORDER BY 2 DESC LIMIT 10"
    ).fetchall()
    histogram = {f"{int(d):+d}": int(n) for d, n in rows}
    return result.model_copy(update={"diagnostics": histogram})
