"""Executing agreement tests in DuckDB (ARCHITECTURE 12.3).

Three properties are load-bearing and each is implemented deliberately:

* A predicate that evaluates to NULL is **out of scope**, counted neither as
  agreement nor disagreement. Without this a column that is entirely null would
  read as universal agreement, which is the most misleading pass available.
* A test whose input columns are absent is **skipped, never passed**.
* The disagreeing row count is always reported, so a near-miss is visible.
"""

from __future__ import annotations

import duckdb

from ai_analyst.contracts.agreement import (
    AgreementReport,
    AgreementResult,
    AgreementSample,
    AgreementTest,
)
from ai_analyst.data.conform import quote_ident


def _stringify(value: object) -> str | None:
    return None if value is None else str(value)


def _where(test: AgreementTest) -> str:
    return f"WHERE {test.scope_sql}" if test.scope_sql else ""


def _samples(
    conn: duckdb.DuckDBPyConnection, scan: str, test: AgreementTest, limit: int
) -> tuple[AgreementSample, ...]:
    if not test.sample_columns or limit <= 0:
        return ()
    selected = ", ".join(quote_ident(c) for c in test.sample_columns)
    scope = f"({test.scope_sql}) AND " if test.scope_sql else ""
    rows = conn.execute(
        f"SELECT {selected} FROM {scan} "
        f"WHERE {scope}NOT ({test.predicate_sql}) LIMIT {int(limit)}"
    ).fetchall()
    return tuple(
        AgreementSample(values=dict(zip(test.sample_columns, map(_stringify, row), strict=True)))
        for row in rows
    )


def run_agreement_test(
    conn: duckdb.DuckDBPyConnection,
    scan: str,
    test: AgreementTest,
    available_columns: set[str],
    *,
    max_samples: int = 5,
) -> AgreementResult:
    """Evaluate one agreement test, or skip it when its inputs are absent."""
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

    checked, disagreeing = conn.execute(
        f"SELECT COUNT(*) FILTER (WHERE ({test.predicate_sql}) IS NOT NULL), "
        f"       COUNT(*) FILTER (WHERE ({test.predicate_sql}) IS FALSE) "
        f"FROM {scan} {_where(test)}"
    ).fetchone()

    result = AgreementResult(
        test_id=test.id,
        assertion=test.assertion,
        kind=test.kind,
        concept=test.concept,
        role=test.role,
            calendar_dependent=test.calendar_dependent,
        checked_rows=int(checked),
        disagreeing_rows=int(disagreeing),
        samples=(
            _samples(conn, scan, test, max_samples) if int(disagreeing) else ()
        ),
    )
    return result


def run_agreement_tests(
    conn: duckdb.DuckDBPyConnection,
    scan: str,
    tests: list[AgreementTest],
    available_columns: set[str],
    *,
    dataset_id: str,
    max_samples: int = 5,
    fiscal_year_start_month: int | None = None,
) -> AgreementReport:
    """Evaluate every test against one dataset."""
    return AgreementReport(
        dataset_id=dataset_id,
        fiscal_year_start_month=fiscal_year_start_month,
        results=tuple(
            run_agreement_test(conn, scan, t, available_columns, max_samples=max_samples)
            for t in tests
        ),
    )
