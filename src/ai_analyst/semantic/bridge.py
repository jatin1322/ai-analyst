"""The pipeline bridge (ARCHITECTURE 5.2).

Five of the six question patterns are slices of one identity:

    opening_pipeline(P)
      + created_in_period
      + pulled_in
      + amount_increased
      - amount_decreased
      - closed_won
      - closed_lost
      - slipped_out
      - other_removed
      = ending_pipeline(P)

**Every term is computed independently, including `other_removed`.** If
`other_removed` were the residual needed to close the identity, the bridge
would balance by construction and the invariant would verify nothing. Here each
term is a set predicate over the two snapshots, and the identity is a genuine
check on nine separately computed quantities. A failure means a real bug.

How the terms stay exhaustive without becoming residuals
--------------------------------------------------------
The pipeline at a snapshot is the set of opportunities that are **open** and
whose **close date falls in P**. Call those sets `O_A` at the opening snapshot
and `O_B` at the closing one. Every opportunity in `O_A ∪ O_B` is in exactly
one of three positions, and each position is classified by its own predicate:

* **survivor** (in both): contributes `amount_increased` or `amount_decreased`,
  never both, computed per opportunity so offsetting moves do not cancel.
* **entrant** (in `O_B` only): `created_in_period` when its creation falls in P,
  otherwise `pulled_in`.
* **leaver** (in `O_A` only): `closed_won`, then `closed_lost`, then
  `slipped_out`, then `other_removed` — first match wins.

The three positions partition the union and the predicates within each position
are mutually exclusive and total, so the identity holds exactly rather than
approximately. The monetary tolerance of 0.01 exists for DECIMAL scale, not to
absorb a classification gap.

`pulled_in` and `other_removed` each report their composition, because both are
buckets a reader should be able to look inside. A large `other_removed` makes
every derived rate suspect (7.3), and a `pulled_in` that is mostly arrivals
rather than close-date moves means something different from what the name
suggests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from ai_analyst.contracts.concepts import BusinessConcept
from ai_analyst.contracts.plan import CreationBasis, Filter, SlipBasis
from ai_analyst.contracts.status import OpportunityStatus
from ai_analyst.semantic.calendar import ResolvedPeriod
from ai_analyst.semantic.resolver import ConceptResolver
from ai_analyst.semantic.sql import CompilationError, build_with, filters_sql, literal

# The nine terms, in identity order, with the sign each carries.
OPENING = "opening_pipeline"
CREATED = "created_in_period"
PULLED_IN = "pulled_in"
AMOUNT_INCREASED = "amount_increased"
AMOUNT_DECREASED = "amount_decreased"
CLOSED_WON = "closed_won"
CLOSED_LOST = "closed_lost"
SLIPPED_OUT = "slipped_out"
OTHER_REMOVED = "other_removed"
ENDING = "ending_pipeline"

BRIDGE_COMPONENTS: tuple[str, ...] = (
    OPENING,
    CREATED,
    PULLED_IN,
    AMOUNT_INCREASED,
    AMOUNT_DECREASED,
    CLOSED_WON,
    CLOSED_LOST,
    SLIPPED_OUT,
    OTHER_REMOVED,
    ENDING,
)

# +1 adds to the pipeline, -1 removes from it. `opening` and `ending` are the
# two sides of the identity and carry no sign of their own.
COMPONENT_SIGNS: dict[str, int] = {
    CREATED: 1,
    PULLED_IN: 1,
    AMOUNT_INCREASED: 1,
    AMOUNT_DECREASED: -1,
    CLOSED_WON: -1,
    CLOSED_LOST: -1,
    SLIPPED_OUT: -1,
    OTHER_REMOVED: -1,
}

# ARCHITECTURE 5.2: the bridge must balance to within 0.01. This is a DECIMAL
# scale tolerance and nothing more. It is never widened to make a real
# classification gap balance, because that is precisely the bug the invariant
# exists to catch.
BRIDGE_TOLERANCE = Decimal("0.01")


class BridgeComponent(BaseModel):
    """One term of the identity."""

    model_config = ConfigDict(frozen=True)

    name: str
    amount: Decimal
    opportunity_count: int
    sign: int = 0
    # What this bucket is made of, for the terms that are buckets.
    composition: dict[str, int] = Field(default_factory=dict)

    @property
    def signed_amount(self) -> Decimal:
        return self.amount * self.sign


class BridgeResult(BaseModel):
    """The full decomposition of one period, with its balance check."""

    model_config = ConfigDict(frozen=True)

    dataset_id: str
    period_label: str
    period_start: date
    period_end: date
    opening_as_of: date
    closing_as_of: date
    components: tuple[BridgeComponent, ...]
    measure_concept: BusinessConcept
    assumptions: tuple[str, ...] = ()

    def by_name(self) -> dict[str, BridgeComponent]:
        return {c.name: c for c in self.components}

    def amount(self, name: str) -> Decimal:
        return self.by_name()[name].amount

    def count(self, name: str) -> int:
        return self.by_name()[name].opportunity_count

    @property
    def movement(self) -> Decimal:
        """The sum of the signed middle terms."""
        return sum(
            (c.signed_amount for c in self.components if c.sign), start=Decimal("0")
        )

    @property
    def implied_ending(self) -> Decimal:
        return self.amount(OPENING) + self.movement

    @property
    def residual(self) -> Decimal:
        """Opening plus movement, minus the independently measured ending.

        This is the check, not a term. It is never added back into the bridge.
        """
        return self.implied_ending - self.amount(ENDING)

    @property
    def balances(self) -> bool:
        return abs(self.residual) <= BRIDGE_TOLERANCE


class BridgeImbalance(AssertionError):
    """The bridge identity did not close. Always a bug, never a tolerance."""


def check_balance(result: BridgeResult) -> None:
    """The bridge invariant (ARCHITECTURE 5.2, 8.3). A failure fails the turn."""
    if not result.balances:
        raise BridgeImbalance(
            f"bridge for {result.period_label} does not balance: "
            f"{result.amount(OPENING)} opening + {result.movement} movement = "
            f"{result.implied_ending}, but ending measures {result.amount(ENDING)} "
            f"(residual {result.residual}, tolerance {BRIDGE_TOLERANCE}). "
            "One of the nine terms is wrong; the identity is not adjusted to fit."
        )


@dataclass
class BridgeSpec:
    """Everything the bridge needs that is not a concept binding."""

    period: ResolvedPeriod
    opening_as_of: date
    closing_as_of: date
    filters: tuple[Filter, ...] = ()
    creation_basis: CreationBasis = CreationBasis.CREATED_DATE
    slip_basis: SlipBasis = SlipBasis.PERIOD_MOVE
    measure_concept: BusinessConcept = BusinessConcept.AMOUNT
    assumptions: list[str] = field(default_factory=list)


def _open_status(status_sql: str) -> str:
    """Whether a row is open pipeline, from the authoritative status concept.

    Never inferred from a stage label here: a vocabulary can encode an outcome
    in a label, and `SFDCDELETED` would read as open pipeline (5.13).
    """
    return f"LOWER(CAST({status_sql} AS VARCHAR)) = {literal(OpportunityStatus.OPEN.value)}"


def bridge_sql(scan: str, spec: BridgeSpec, resolver: ConceptResolver) -> str:
    """Compile the whole decomposition as one query, one row per term."""
    opp = resolver.resolve(BusinessConcept.OPPORTUNITY_ID, field_name="bridge")
    status = resolver.resolve(BusinessConcept.OPPORTUNITY_STATUS, field_name="bridge")
    close = resolver.resolve(BusinessConcept.EXPECTED_CLOSE_DATE, field_name="bridge")
    measure = resolver.resolve(spec.measure_concept, field_name="bridge")

    p_start, p_end = literal(spec.period.start), literal(spec.period.end)
    predicate = filters_sql(list(spec.filters))

    def pipeline_rows(as_of: date) -> str:
        """Open rows whose close date falls in the period, at one snapshot."""
        return (
            f"SELECT {opp.sql()} AS opp_id, {measure.measure_sql()} AS measure, "
            f"{close.sql()} AS close_date\n"
            f"FROM {scan}\n"
            f"WHERE as_of = {literal(as_of)}\n"
            f"  AND ({predicate})\n"
            f"  AND {_open_status(status.sql())}\n"
            f"  AND {close.sql()} BETWEEN {p_start} AND {p_end}"
        )

    # Creation basis (5.3 ambiguity 1). `created_date` is the default. First
    # appearance answers a different question, so it is used only when the plan
    # selects it; a creation-date basis without that concept is refused here as
    # well as at the gate, never quietly replaced.
    if spec.creation_basis is CreationBasis.CREATED_DATE and not resolver.has(
        BusinessConcept.CREATED_DATE, load_bearing=False
    ):
        raise CompilationError(
            "creation basis 'created_date' needs the created_date concept, which this "
            "dataset does not have; select 'first_seen' explicitly"
        )
    if spec.creation_basis is CreationBasis.CREATED_DATE:
        created = resolver.resolve(
            BusinessConcept.CREATED_DATE, load_bearing=False, field_name="bridge"
        )
        creation_cte = (
            f"SELECT DISTINCT {opp.sql()} AS opp_id, {created.sql()} AS created_on\n"
            f"FROM {scan}\n"
            f"WHERE {created.sql()} IS NOT NULL"
        )
        spec.assumptions.append(
            f"'Created in period' uses the {created.column!r} column."
        )
    else:
        creation_cte = (
            f"SELECT {opp.sql()} AS opp_id, MIN(as_of) AS created_on\n"
            f"FROM {scan}\n"
            f"GROUP BY 1"
        )
        spec.assumptions.append(
            "'Created in period' uses first appearance in a snapshot, not a "
            "creation date; an opportunity created before the export window "
            "appears as created at its first snapshot."
        )

    slipped = (
        f"b_close > {p_end}"
        if spec.slip_basis is SlipBasis.PERIOD_MOVE
        else "b_close > a.close_date"
    )

    ctes = [
        ("pipe_open", pipeline_rows(spec.opening_as_of)),
        ("pipe_close", pipeline_rows(spec.closing_as_of)),
        ("creation", creation_cte),
        (
            # State at the closing snapshot, and whether a terminal state was
            # reached anywhere in the window. A deal that closed won and then
            # dropped out of later snapshots is won, not vanished.
            "fate",
            f"SELECT {opp.sql()} AS opp_id,\n"
            f"    COALESCE(BOOL_OR(as_of = {literal(spec.closing_as_of)}), FALSE)\n"
            f"        AS present_at_close,\n"
            f"    MAX(CASE WHEN as_of = {literal(spec.closing_as_of)} "
            f"THEN {close.sql()} END) AS b_close,\n"
            f"    COALESCE(BOOL_OR(LOWER(CAST({status.sql()} AS VARCHAR)) = "
            f"{literal(OpportunityStatus.WON.value)}), FALSE) AS ever_won,\n"
            f"    COALESCE(BOOL_OR(LOWER(CAST({status.sql()} AS VARCHAR)) = "
            f"{literal(OpportunityStatus.LOST.value)}), FALSE) AS ever_lost\n"
            f"FROM {scan}\n"
            f"WHERE as_of > {literal(spec.opening_as_of)} "
            f"AND as_of <= {literal(spec.closing_as_of)}\n"
            f"  AND ({predicate})\n"
            f"GROUP BY 1",
        ),
        (
            # State at the opening snapshot for *every* opportunity, not only
            # those already in the period's pipeline. Without this, an entrant
            # that was present at A with a later close date is indistinguishable
            # from one that did not exist yet, and `pulled_in` cannot say which
            # it is made of.
            "opening_state",
            f"SELECT {opp.sql()} AS opp_id, {close.sql()} AS a_close_any,\n"
            f"    {_open_status(status.sql())} AS a_open\n"
            f"FROM {scan}\n"
            f"WHERE as_of = {literal(spec.opening_as_of)} AND ({predicate})",
        ),
        (
            "positions",
            "SELECT\n"
            "    COALESCE(a.opp_id, b.opp_id) AS opp_id,\n"
            "    a.opp_id IS NOT NULL AS in_open,\n"
            "    b.opp_id IS NOT NULL AS in_close,\n"
            "    COALESCE(a.measure, 0) AS a_measure,\n"
            "    COALESCE(b.measure, 0) AS b_measure,\n"
            "    a.close_date AS a_close,\n"
            "    f.b_close AS b_close,\n"
            "    COALESCE(f.present_at_close, FALSE) AS present_at_close,\n"
            "    COALESCE(f.ever_won, FALSE) AS ever_won,\n"
            "    COALESCE(f.ever_lost, FALSE) AS ever_lost,\n"
            "    c.created_on AS created_on,\n"
            "    o.a_close_any AS a_close_any,\n"
            "    o.opp_id IS NOT NULL AS present_at_open,\n"
            "    COALESCE(o.a_open, FALSE) AS a_open\n"
            "FROM pipe_open a\n"
            "FULL OUTER JOIN pipe_close b ON a.opp_id = b.opp_id\n"
            "LEFT JOIN fate f ON f.opp_id = COALESCE(a.opp_id, b.opp_id)\n"
            "LEFT JOIN creation c ON c.opp_id = COALESCE(a.opp_id, b.opp_id)\n"
            "LEFT JOIN opening_state o ON o.opp_id = COALESCE(a.opp_id, b.opp_id)",
        ),
        (
            # One bucket per opportunity. The CASE is ordered and total, so no
            # opportunity lands in two terms and none lands in none.
            "classified",
            f"SELECT\n"
            f"    opp_id, a_measure, b_measure,\n"
            f"    CASE\n"
            f"        WHEN in_open AND in_close THEN\n"
            f"            CASE WHEN b_measure > a_measure THEN {literal(AMOUNT_INCREASED)}\n"
            f"                 WHEN b_measure < a_measure THEN {literal(AMOUNT_DECREASED)}\n"
            f"                 ELSE 'unchanged' END\n"
            f"        WHEN in_close THEN\n"
            f"            CASE WHEN created_on BETWEEN {p_start} AND {p_end}\n"
            f"                 THEN {literal(CREATED)} ELSE {literal(PULLED_IN)} END\n"
            f"        WHEN ever_won THEN {literal(CLOSED_WON)}\n"
            f"        WHEN ever_lost THEN {literal(CLOSED_LOST)}\n"
            f"        WHEN present_at_close AND b_close IS NOT NULL AND {slipped}\n"
            f"             THEN {literal(SLIPPED_OUT)}\n"
            f"        ELSE {literal(OTHER_REMOVED)}\n"
            f"    END AS component,\n"
            f"    CASE\n"
            f"        WHEN NOT in_open AND in_close AND NOT present_at_open\n"
            f"             THEN 'arrived_after_opening'\n"
            f"        WHEN NOT in_open AND in_close AND a_close_any IS NOT NULL\n"
            f"             AND NOT (a_close_any BETWEEN {p_start} AND {p_end})\n"
            f"             THEN 'close_date_moved_into_period'\n"
            f"        WHEN NOT in_open AND in_close AND NOT a_open THEN 'reopened'\n"
            f"        WHEN NOT present_at_close THEN 'absent_at_close'\n"
            f"        ELSE 'present_not_open'\n"
            f"    END AS composition\n"
            f"FROM positions",
        ),
        (
            # Movement terms. A survivor contributes the absolute size of its
            # own move, so an increase on one deal and a decrease on another do
            # not silently net to nothing.
            "movement",
            f"SELECT component,\n"
            f"       SUM(ABS(b_measure - a_measure)) AS amount,\n"
            f"       COUNT(*) AS opportunity_count,\n"
            f"       '' AS composition\n"
            f"FROM classified\n"
            f"WHERE component IN ({literal(AMOUNT_INCREASED)}, {literal(AMOUNT_DECREASED)})\n"
            f"GROUP BY 1",
        ),
        (
            "entrants",
            f"SELECT component, SUM(b_measure) AS amount, COUNT(*) AS opportunity_count,\n"
            f"       '' AS composition\n"
            f"FROM classified\n"
            f"WHERE component IN ({literal(CREATED)}, {literal(PULLED_IN)})\n"
            f"GROUP BY 1",
        ),
        (
            "leavers",
            f"SELECT component, SUM(a_measure) AS amount, COUNT(*) AS opportunity_count,\n"
            f"       '' AS composition\n"
            f"FROM classified\n"
            f"WHERE component IN ({literal(CLOSED_WON)}, {literal(CLOSED_LOST)}, "
            f"{literal(SLIPPED_OUT)}, {literal(OTHER_REMOVED)})\n"
            f"GROUP BY 1",
        ),
        (
            "endpoints",
            f"SELECT {literal(OPENING)} AS component, "
            f"COALESCE(SUM(measure), 0) AS amount, COUNT(*) AS opportunity_count, "
            f"'' AS composition FROM pipe_open\n"
            f"UNION ALL\n"
            f"SELECT {literal(ENDING)}, COALESCE(SUM(measure), 0), COUNT(*), '' "
            f"FROM pipe_close",
        ),
        (
            "terms",
            "SELECT * FROM endpoints\n"
            "UNION ALL SELECT * FROM movement\n"
            "UNION ALL SELECT * FROM entrants\n"
            "UNION ALL SELECT * FROM leavers",
        ),
    ]

    # Left-join the fixed component list so a term with no rows reports zero
    # rather than vanishing from the result.
    names = ", ".join(f"({literal(n)}, {i})" for i, n in enumerate(BRIDGE_COMPONENTS))
    body = (
        f"SELECT n.component AS component,\n"
        f"       COALESCE(t.amount, 0) AS amount,\n"
        f"       COALESCE(t.opportunity_count, 0) AS opportunity_count\n"
        f"FROM (VALUES {names}) AS n(component, ordinal)\n"
        f"LEFT JOIN terms t ON t.component = n.component\n"
        f"ORDER BY n.ordinal"
    )
    return build_with(ctes, body)


def composition_sql(scan: str, spec: BridgeSpec, resolver: ConceptResolver) -> str:
    """What `pulled_in` and `other_removed` are actually made of.

    Reported beside the bridge rather than folded into it, because both names
    promise more than the predicate delivers and a reader should be able to
    check.
    """
    inner = bridge_sql(scan, spec, resolver)
    # Reuse the classification CTEs by re-running the same query shape and
    # grouping one level lower.
    head, _, _ = inner.partition("\nSELECT n.component")
    return (
        f"{head}\n"
        f"SELECT component, composition, COUNT(*) AS opportunity_count\n"
        f"FROM classified\n"
        f"WHERE component IN ({literal(PULLED_IN)}, {literal(OTHER_REMOVED)})\n"
        f"GROUP BY 1, 2\n"
        f"ORDER BY 1, 2"
    )
