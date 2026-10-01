"""The transition primitive (ARCHITECTURE 5.4).

`transition(snap_a, snap_b, field)` answers *what changed between two snapshots,
per opportunity*. It is the primitive behind slippage, stage progression,
amount movement, and forecast-category drift, and it is deliberately computed
from raw snapshot history rather than read from a precomputed counter.

That last point is the rule of 5.9 applied here. An export may carry a
`close_date_push_count`; this module does not read it. A counter cumulative
since creation cannot be reproduced from a bounded export window, so the two
disagree by construction on opportunities older than the window, and the
recomputation is the one that says what happened *inside* the window. Where the
counter exists it is a reconciliation target, never an input.

A transition is a FULL OUTER JOIN on the grain, so an opportunity present in
only one of the two snapshots is representable rather than dropped: `appeared`
and `disappeared` are transitions too, and the bridge needs both.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum

from ai_analyst.contracts.concepts import BusinessConcept
from ai_analyst.contracts.plan import Filter
from ai_analyst.semantic.resolver import ConceptResolver
from ai_analyst.semantic.sql import build_with, filters_sql, literal, snapshot_rows_cte


class TransitionField(StrEnum):
    """Which attribute a transition tracks.

    Each maps to a concept, never to a column name, so a tenant whose stage
    column is `SalesStage` needs no change here.
    """

    CLOSE_DATE = "close_date"
    STAGE = "stage"
    AMOUNT = "amount"
    FORECAST_CATEGORY = "forecast_category"
    STATUS = "status"

    @property
    def concept(self) -> BusinessConcept:
        return _FIELD_CONCEPTS[self]

    @property
    def is_monetary(self) -> bool:
        return self is TransitionField.AMOUNT


_FIELD_CONCEPTS: dict[TransitionField, BusinessConcept] = {
    TransitionField.CLOSE_DATE: BusinessConcept.EXPECTED_CLOSE_DATE,
    TransitionField.STAGE: BusinessConcept.STAGE,
    TransitionField.AMOUNT: BusinessConcept.AMOUNT,
    TransitionField.FORECAST_CATEGORY: BusinessConcept.FORECAST_CATEGORY,
    TransitionField.STATUS: BusinessConcept.OPPORTUNITY_STATUS,
}


class TransitionKind(StrEnum):
    """What happened to one opportunity between the two snapshots."""

    UNCHANGED = "unchanged"
    CHANGED = "changed"
    APPEARED = "appeared"
    DISAPPEARED = "disappeared"


@dataclass(frozen=True)
class TransitionSpec:
    """One transition query: two snapshots and the field to compare."""

    field: TransitionField
    from_as_of: date
    to_as_of: date
    filters: tuple[Filter, ...] = ()
    # Restrict to opportunities present in the earlier snapshot. The bridge
    # uses this; a standalone transition query usually does not.
    require_before: bool = False
    changed_only: bool = False

    @property
    def snapshot_pair(self) -> tuple[date, date]:
        return (self.from_as_of, self.to_as_of)


TRANSITION_COLUMNS: tuple[str, ...] = (
    "opp_id",
    "field",
    "before",
    "after",
    "changed",
    "transition",
    "from_as_of",
    "to_as_of",
)


def transition_sql(scan: str, spec: TransitionSpec, resolver: ConceptResolver) -> str:
    """Per-opportunity before/after for one field across two snapshots.

    The comparison is `IS DISTINCT FROM`, not `<>`. A null-to-value move is a
    real change, and `<>` would return NULL for it and quietly drop the row
    from the changed set, understating every movement metric that uses this.
    """
    resolution = resolver.resolve(
        spec.field.concept, load_bearing=True, field_name="transition"
    )
    opp = resolver.resolve(
        BusinessConcept.OPPORTUNITY_ID, load_bearing=True, field_name="transition"
    )

    value = resolution.measure_sql if spec.field.is_monetary else resolution.sql
    before, after = value("a"), value("b")
    opp_a, opp_b = opp.sql("a"), opp.sql("b")
    predicate = filters_sql(list(spec.filters))

    join = "INNER" if spec.require_before else "FULL OUTER"
    changed = f"({before} IS DISTINCT FROM {after})"
    kind = (
        f"CASE\n"
        f"        WHEN {opp_a} IS NULL THEN {literal(TransitionKind.APPEARED.value)}\n"
        f"        WHEN {opp_b} IS NULL THEN {literal(TransitionKind.DISAPPEARED.value)}\n"
        f"        WHEN {changed} THEN {literal(TransitionKind.CHANGED.value)}\n"
        f"        ELSE {literal(TransitionKind.UNCHANGED.value)}\n"
        f"    END"
    )

    ctes = [
        ("snapshot_a", snapshot_rows_cte(scan, spec.from_as_of, predicate)),
        ("snapshot_b", snapshot_rows_cte(scan, spec.to_as_of, predicate)),
    ]
    where = f"\nWHERE {changed}" if spec.changed_only else ""
    body = (
        f"SELECT\n"
        f"    COALESCE({opp_a}, {opp_b}) AS opp_id,\n"
        f"    {literal(spec.field.value)} AS field,\n"
        f"    {before} AS before,\n"
        f"    {after} AS after,\n"
        f"    {changed} AS changed,\n"
        f"    {kind} AS transition,\n"
        f"    {literal(spec.from_as_of)} AS from_as_of,\n"
        f"    {literal(spec.to_as_of)} AS to_as_of\n"
        f"FROM snapshot_a a\n"
        f"{join} JOIN snapshot_b b ON {opp_a} = {opp_b}"
        f"{where}\n"
        f"ORDER BY opp_id"
    )
    return build_with(ctes, body)


def close_date_movement_sql(
    scan: str,
    resolver: ConceptResolver,
    from_as_of: date,
    to_as_of: date,
    filters: tuple[Filter, ...] = (),
) -> str:
    """Close-date before/after for every opportunity in the earlier snapshot.

    The shared input to slippage and pull-in. Both are period predicates over
    the same two columns, so computing them from one CTE keeps them consistent
    by construction: an opportunity cannot be counted as both.
    """
    spec = TransitionSpec(
        field=TransitionField.CLOSE_DATE,
        from_as_of=from_as_of,
        to_as_of=to_as_of,
        filters=filters,
        require_before=True,
    )
    return transition_sql(scan, spec, resolver)
