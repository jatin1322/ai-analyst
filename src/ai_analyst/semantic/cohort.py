"""Cohort and trace primitives (ARCHITECTURE 5.4).

A cohort is a set of opportunities frozen at one snapshot. Once fixed, no later
snapshot may add a member: an opportunity that first appears after the cohort
snapshot is not in the cohort, however far forward the trace runs. This is what
makes "what percentage of opening pipeline eventually closed" correct, because
it follows those specific opportunities rather than comparing two independent
aggregates.

`trace` resolves each member's fate through the **status concept**, not through
a stage label and not through a terminal outcome column. Status is the
authoritative five-valued state (5.13); a terminal column is the reconciliation
target for a trace, never an input to one, and a tenant that has no terminal
column can still be traced.

Terminal state is read as **ever reached by** the through-snapshot, not as the
state at the through-snapshot. An opportunity that closed won and then dropped
out of later snapshots is won, not vanished. `vanished` is reserved for
opportunities that left the data without ever reaching a terminal state.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from ai_analyst.contracts.binding import ColumnPurpose
from ai_analyst.contracts.concepts import BusinessConcept
from ai_analyst.contracts.plan import Filter
from ai_analyst.contracts.status import OpportunityStatus
from ai_analyst.semantic.resolver import ConceptResolver
from ai_analyst.semantic.sql import build_with, filters_sql, literal

WON = "won"
LOST = "lost"
OPEN = "open"
EXCLUDED = "excluded"
UNKNOWN = "unknown"
VANISHED = "vanished"

# The six states a traced opportunity can be in, in precedence order.
#
# Precedence matters because an opportunity's status changes across snapshots
# and "ever reached" is not mutually exclusive: a deal can be open at one
# snapshot, excluded at another, and won at a third. The order below is the
# one deterministic answer, and it reads as an argument:
#
#   won, lost   a terminal business outcome outranks everything. It is the fact
#               the question is usually about, and it is irreversible.
#   excluded    a deliberate exclusion (deleted, disqualified) outranks
#               liveness: the record was removed from the funnel on purpose.
#   open        still live at the through-snapshot.
#   unknown     seen, but never in a state the status mapping could read. Not
#               the same as vanished, and never silently folded into `open`.
#   vanished    left the data without ever reaching a terminal state. This is
#               the `other_removed` case of the bridge and must stay visible.
TERMINAL_STATES: tuple[str, ...] = (WON, LOST, EXCLUDED, OPEN, UNKNOWN, VANISHED)


@dataclass(frozen=True)
class CohortSpec:
    """A cohort definition: one snapshot plus the predicates that select it."""

    as_of: date
    filters: tuple[Filter, ...] = ()
    extra_predicate: str = "TRUE"

    def sql(self, scan: str) -> str:
        """The frozen membership query, evaluated once at `as_of`."""
        predicate = filters_sql(list(self.filters))
        return (
            f"SELECT *\n"
            f"FROM {scan}\n"
            f"WHERE as_of = {literal(self.as_of)}\n"
            f"  AND ({predicate})\n"
            f"  AND ({self.extra_predicate})"
        )


def cohort(
    as_of: date,
    filters: list[Filter] | None = None,
    extra_predicate: str = "TRUE",
) -> CohortSpec:
    """Freeze a cohort at one snapshot (ARCHITECTURE 5.4).

    Membership is fixed here and never recomputed downstream.
    """
    return CohortSpec(
        as_of=as_of,
        filters=tuple(filters or ()),
        extra_predicate=extra_predicate,
    )


def _status_case(status_sql: str) -> dict[str, str]:
    """Per-state `ever reached` flags, built from the status concept.

    Status values are compared case-insensitively against `OpportunityStatus`,
    which is what the ingestion layer already wrote into the status column.
    """

    def ever(state: OpportunityStatus) -> str:
        matches = f"LOWER(CAST({status_sql} AS VARCHAR)) = {literal(state.value)}"
        return f"COALESCE(BOOL_OR({matches}), FALSE)"

    return {
        "ever_won": ever(OpportunityStatus.WON),
        "ever_lost": ever(OpportunityStatus.LOST),
        "ever_excluded": ever(OpportunityStatus.EXCLUDED),
        "ever_open": ever(OpportunityStatus.OPEN),
    }


def trace_sql(
    scan: str,
    cohort_spec: CohortSpec,
    through: date,
    resolver: ConceptResolver,
    measure_concept: BusinessConcept = BusinessConcept.AMOUNT,
    dimensions: tuple[str, ...] = (),
) -> str:
    """Classify each frozen cohort member's fate by `through`.

    The measure is the **cohort's own** amount, read at the cohort snapshot and
    fixed when the cohort was fixed. Reading it at the through-snapshot would
    let a later amount change silently restate the cohort's size.

    The six states partition the cohort exactly: every branch of the CASE is
    mutually exclusive by precedence and the final ELSE is total, so the counts
    always sum back to the cohort size and nothing is double counted.
    """
    opp = resolver.resolve(BusinessConcept.OPPORTUNITY_ID, field_name="cohort_trace")
    status = resolver.resolve(BusinessConcept.OPPORTUNITY_STATUS, field_name="cohort_trace")
    measure = resolver.resolve(measure_concept, field_name="cohort_trace")

    flags = _status_case(status.sql("w"))
    flag_select = "".join(f"    {sql} AS {name},\n" for name, sql in flags.items())

    dim_cols = [
        resolver.reference(d, purpose=ColumnPurpose.DIMENSION, field_name="dimensions")[0] or d
        for d in dimensions
    ]
    dim_select = "".join(f'    c."{d}" AS "{d}",\n' for d in dim_cols)
    dim_tail = "".join(f', "{d}"' for d in dim_cols)
    dim_head = "".join(f'"{d}", ' for d in dim_cols)

    ctes = [
        ("cohort_rows", cohort_spec.sql(scan)),
        (
            "window_rows",
            f"SELECT w.*\n"
            f"FROM {scan} w\n"
            f"JOIN cohort_rows c ON {opp.sql('c')} = {opp.sql('w')}\n"
            f"WHERE w.as_of BETWEEN {literal(cohort_spec.as_of)} AND {literal(through)}",
        ),
        (
            "outcomes",
            f"SELECT\n"
            f"    {opp.sql('c')} AS opp_id,\n"
            f"{dim_select}"
            f"    {measure.measure_sql('c')} AS cohort_measure,\n"
            f"{flag_select}"
            f"    MAX(w.as_of) AS last_seen\n"
            f"FROM cohort_rows c\n"
            f"LEFT JOIN window_rows w ON {opp.sql('w')} = {opp.sql('c')}\n"
            f"GROUP BY {opp.sql('c')}{dim_tail}, cohort_measure",
        ),
        (
            "classified",
            f"SELECT\n"
            f"    opp_id{dim_tail},\n"
            f"    cohort_measure,\n"
            f"    CASE\n"
            f"        WHEN ever_won THEN {literal(WON)}\n"
            f"        WHEN ever_lost THEN {literal(LOST)}\n"
            f"        WHEN ever_excluded THEN {literal(EXCLUDED)}\n"
            f"        WHEN last_seen >= {literal(through)} AND ever_open "
            f"THEN {literal(OPEN)}\n"
            f"        WHEN last_seen >= {literal(through)} THEN {literal(UNKNOWN)}\n"
            f"        ELSE {literal(VANISHED)}\n"
            f"    END AS terminal_state\n"
            f"FROM outcomes",
        ),
    ]

    body = (
        f"SELECT {dim_head}terminal_state,\n"
        f"       COUNT(*) AS opportunity_count,\n"
        f"       COALESCE(SUM(cohort_measure), 0) AS cohort_amount\n"
        f"FROM classified\n"
        f"GROUP BY terminal_state{dim_tail}\n"
        f"ORDER BY terminal_state{dim_tail}"
    )
    return build_with(ctes, body)
