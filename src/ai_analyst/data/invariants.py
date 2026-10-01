"""Declared family invariants as reconciliation evidence (WP5).

A tenant declares a relationship a family of columns should satisfy. It is
evaluated through the ordinary agreement machinery with role RECONCILIATION:
a violation is reported with its checked and disagreeing row counts and never
repaired, never promoted to validity, and never rewrites a value. A test whose
columns are absent is skipped, not passed.
"""

from __future__ import annotations

import duckdb

from ai_analyst.contracts.agreement import (
    AgreementKind,
    AgreementReport,
    AgreementResult,
    AgreementRole,
    AgreementTest,
)
from ai_analyst.contracts.tenant import FamilyInvariant, FamilyInvariantKind
from ai_analyst.data.agreement import run_agreement_test
from ai_analyst.data.conform import quote_ident

TEST_PREFIX = "family_invariant:"


def _lifetime_last(windows: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(w for w in windows if w != "lifetime") + tuple(
        w for w in windows if w == "lifetime"
    )


def invariant_test(inv: FamilyInvariant, available: set[str]) -> AgreementTest:
    """Compile one declared invariant into a RECONCILIATION agreement test."""
    common = {
        "id": f"{TEST_PREFIX}{inv.id}",
        "kind": AgreementKind.COLUMNS_SATISFY_RELATIONSHIP,
        "role": AgreementRole.RECONCILIATION,
    }
    if inv.kind is FamilyInvariantKind.WINDOW_MONOTONE:
        columns = [inv.template.replace("{w}", w) for w in _lifetime_last(inv.windows)]
        present = [c for c in columns if c in available]
        used = present if len(present) >= 2 else columns  # <2 present: skip, never pass
        chain = " AND ".join(
            f"({quote_ident(a)} <= {quote_ident(b)})" for a, b in zip(used, used[1:], strict=False)
        )
        return AgreementTest(
            **common,
            assertion=f"{inv.template} is non-decreasing across windows {', '.join(inv.windows)}",
            predicate_sql=chain,
            requires_columns=tuple(used),
            sample_columns=tuple(used),
        )
    if inv.kind is FamilyInvariantKind.PARTS_SUM:
        windows = inv.windows or ("",)
        terms, cols = [], []
        for w in windows:
            parts = [p.replace("{w}", w) for p in inv.parts]
            whole = inv.whole.replace("{w}", w)
            cols += [*parts, whole]
            total = " + ".join(quote_ident(p) for p in parts)
            terms.append(f"(({total}) = {quote_ident(whole)})")
        return AgreementTest(
            **common,
            assertion=f"{' + '.join(inv.parts)} = {inv.whole}",
            predicate_sql=" AND ".join(terms),
            requires_columns=tuple(dict.fromkeys(cols)),
            sample_columns=tuple(dict.fromkeys(cols))[:6],
        )
    return AgreementTest(
        **common,
        assertion=f"{inv.label} is null wherever {inv.mask} = 0",
        predicate_sql=f"{quote_ident(inv.label)} IS NULL",
        scope_sql=f"{quote_ident(inv.mask)} = 0",
        requires_columns=(inv.label, inv.mask),
        sample_columns=(inv.mask, inv.label),
    )


def evaluate_family_invariants(
    conn: duckdb.DuckDBPyConnection,
    scan: str,
    invariants: list[FamilyInvariant],
    *,
    dataset_id: str,
    max_samples: int = 5,
) -> tuple[AgreementResult, ...]:
    """Evaluate declared invariants over `scan`. Evidence only; data is untouched."""
    available = {r[0] for r in conn.execute(f"DESCRIBE SELECT * FROM {scan}").fetchall()}
    return tuple(
        run_agreement_test(
            conn, scan, invariant_test(inv, available), available, max_samples=max_samples
        )
        for inv in invariants
    )


def with_invariants(
    report: AgreementReport, results: tuple[AgreementResult, ...]
) -> AgreementReport:
    """A report extended with invariant results, replacing any earlier ones."""
    kept = tuple(r for r in report.results if not r.test_id.startswith(TEST_PREFIX))
    return report.model_copy(update={"results": kept + results})
