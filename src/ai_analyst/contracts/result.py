"""Result contracts.

`ResultSet` is designed against its consumer, the provenance scanner, which
addresses values as (query_id, row_index, column_name) (ARCHITECTURE §8.4).
Every result also carries the snapshot dates that were actually resolved, never
just the rule that was requested (ARCHITECTURE §5.1).
"""

from __future__ import annotations

import uuid
from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from ai_analyst.contracts.schema import DataType

type QueryId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,63}$")]

# A result value. `ResultSet` is held in memory and persisted as Parquet
# (ARCHITECTURE 7.1), never as JSON, so Decimal and date keep their types.
# Serializing one to JSON would route a Decimal through a union that resolves
# it to str; if that ever becomes a real path, it needs an explicit encoder.
type CellValue = str | int | float | Decimal | bool | date | None


def new_query_id() -> str:
    """Generate a query id that is valid as an answer reference token."""
    return f"q{uuid.uuid4().hex[:12]}"


class TrustTier(StrEnum):
    """How much the system vouches for a result (ARCHITECTURE 12.7).

    A: registry metric on confirmed bindings, through a compiled plan.
    B: executed and checked, but a definition or a binding is not established.
    C: insufficient evidence. No number is emitted at all.

    **The tier is computed, never chosen.** It is the weakest of a result's
    inputs, assigned after execution. `weakest` is that rule, in one place.
    """

    A = "A"
    B = "B"
    C = "C"

    @property
    def emits_a_number(self) -> bool:
        """Tier C is an abstention: nothing executes and no value is reported."""
        return self is not TrustTier.C

    @classmethod
    def weakest(cls, tiers) -> TrustTier:
        """The weakest tier among the inputs. A result is never stronger."""
        order = {cls.A: 0, cls.B: 1, cls.C: 2}
        return max(list(tiers) or [cls.A], key=lambda t: order[t])


class TrustFactorKind(StrEnum):
    """Every input that can lower a result's trust (ARCHITECTURE 13.10)."""

    # The analytical path taken.
    SEMANTIC_PATH = "semantic_path"
    INVESTIGATION_PATH = "investigation_path"
    GUARDED_SQL = "guarded_sql"
    # Bindings and columns.
    BINDING_INFERRED = "binding_inferred"
    USAGE_GRANT = "usage_grant"
    LINEAGE_UNCONFIRMED = "lineage_unconfirmed"
    STATUS_NOT_AUTHORITATIVE = "status_not_authoritative"
    UNRESOLVED_QUESTION = "unresolved_question"
    # Reconstructed concepts (13.1).
    RECONSTRUCTION_UNVERIFIED = "reconstruction_unverified"
    RECONSTRUCTION_UNCORROBORATED = "reconstruction_uncorroborated"
    RECONCILIATION_WARNING = "reconciliation_warning"
    # Time.
    CALENDAR_UNRESOLVED = "calendar_unresolved"
    SNAPSHOT_DRIFT = "snapshot_drift"
    RETROSPECTIVE_READ = "retrospective_read"
    # Nothing may execute, or the result is void.
    UNRESOLVED_AMBIGUITY = "unresolved_ambiguity"
    CONCEPT_UNAVAILABLE = "concept_unavailable"
    SANITY_FAILED = "sanity_failed"


# The ceiling each factor imposes. This table *is* the trust model: a factor
# carries no tier of its own, so nothing that constructs a factor, including a
# model's structured output, can choose what the factor costs.
FACTOR_CEILINGS: dict[TrustFactorKind, TrustTier] = {
    TrustFactorKind.SEMANTIC_PATH: TrustTier.A,
    TrustFactorKind.INVESTIGATION_PATH: TrustTier.B,
    TrustFactorKind.GUARDED_SQL: TrustTier.B,
    TrustFactorKind.BINDING_INFERRED: TrustTier.B,
    TrustFactorKind.USAGE_GRANT: TrustTier.A,
    TrustFactorKind.LINEAGE_UNCONFIRMED: TrustTier.B,
    TrustFactorKind.STATUS_NOT_AUTHORITATIVE: TrustTier.B,
    TrustFactorKind.UNRESOLVED_QUESTION: TrustTier.B,
    TrustFactorKind.RECONSTRUCTION_UNVERIFIED: TrustTier.B,
    TrustFactorKind.RECONSTRUCTION_UNCORROBORATED: TrustTier.B,
    TrustFactorKind.RECONCILIATION_WARNING: TrustTier.B,
    TrustFactorKind.CALENDAR_UNRESOLVED: TrustTier.B,
    TrustFactorKind.SNAPSHOT_DRIFT: TrustTier.B,
    TrustFactorKind.RETROSPECTIVE_READ: TrustTier.B,
    TrustFactorKind.UNRESOLVED_AMBIGUITY: TrustTier.C,
    TrustFactorKind.CONCEPT_UNAVAILABLE: TrustTier.C,
    TrustFactorKind.SANITY_FAILED: TrustTier.C,
}


class TrustFactor(BaseModel):
    """One reason a result is trusted less, or one fact disclosed about it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: TrustFactorKind
    # What the factor is about: a concept, a column, a test id, a plan field.
    subject: str = ""
    # A deterministic, templated sentence. Never model prose.
    reason: str

    @property
    def tier(self) -> TrustTier:
        return FACTOR_CEILINGS[self.kind]

    @property
    def lowers_trust(self) -> bool:
        return self.tier is not TrustTier.A

    @property
    def is_disclosure(self) -> bool:
        """Tier-A factors that must still be stated, such as a usage grant."""
        return self.kind is TrustFactorKind.USAGE_GRANT


class TrustAssessment(BaseModel):
    """The computed trust of one result or one answer (ARCHITECTURE 13.10).

    The tier is **derived** from the factors through `FACTOR_CEILINGS` and is
    not a field. There is no argument, attribute, or JSON key through which a
    caller could supply it, and `extra="forbid"` makes an attempt to pass one
    an error rather than a silent no-op.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    factors: tuple[TrustFactor, ...] = ()

    @property
    def tier(self) -> TrustTier:
        return TrustTier.weakest([f.tier for f in self.factors])

    @property
    def reasons(self) -> list[str]:
        """Why the tier is below A: what a tier-B answer must name."""
        return [f.reason for f in self.factors if f.lowers_trust]

    @property
    def disclosures(self) -> list[str]:
        """Every sentence an answer must carry: the reasons, then disclosures."""
        return [f.reason for f in self.factors if f.lowers_trust or f.is_disclosure]

    def combine(self, *others: TrustAssessment) -> TrustAssessment:
        """An answer's assessment: the union of its results' factors."""
        merged = list(self.factors)
        for other in others:
            merged.extend(f for f in other.factors if f not in merged)
        return TrustAssessment(factors=tuple(merged))


class SnapshotRule(StrEnum):
    """Snapshot selection rules (ARCHITECTURE §5.1)."""

    AS_OF_EXACT = "as_of_exact"
    PERIOD_OPEN = "period_open"
    PERIOD_CLOSE = "period_close"
    LATEST = "latest"
    LATEST_IN_PERIOD = "latest_in_period"
    ALL = "all"


class ResolvedSnapshot(BaseModel):
    """Which snapshot a rule actually landed on, and how far off it was."""

    model_config = ConfigDict(frozen=True)

    rule: SnapshotRule
    requested_boundary: date | None = None
    resolved_as_of: date
    drift_days: int = 0
    within_tolerance: bool = True


class ValueKind(StrEnum):
    """How a result value is rendered. Set by the compiler, never by a model."""

    MONEY = "money"
    RATIO = "ratio"
    COUNT = "count"
    QUANTITY = "quantity"
    DATE = "date"
    TEXT = "text"
    BOOLEAN = "boolean"


class ResultColumn(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    dtype: DataType
    kind: ValueKind | None = None


class TemporalRelation(StrEnum):
    """How a value's snapshot relates to what an analysis may know (13.11 #4)."""

    BACKWARD = "backward"                  # read at an earlier snapshot: history
    CONTEMPORANEOUS = "contemporaneous"    # read at the analysis snapshot itself
    LATER = "later"                        # read after the horizon: hindsight
    RETROSPECTIVE_TERMINAL = "retrospective_terminal"  # an outcome, whenever read

    @property
    def safe_for_prospective(self) -> bool:
        return self in (TemporalRelation.BACKWARD, TemporalRelation.CONTEMPORANEOUS)


class AttributionRead(BaseModel):
    """One dimension or feature, and the snapshot it is attributed from.

    Recorded by the gate and carried into the compilation metadata, so the
    prospective later-snapshot guard is a property of the compiled artifact and
    not only of a helper that happened to run.
    """

    model_config = ConfigDict(frozen=True)

    field: str
    column: str
    rule: str
    read_as_of: date
    horizon: date | None = None
    relation: TemporalRelation


class CompilationMetadata(BaseModel):
    """How one compiled query was produced.

    Enough to re-derive the SQL and to audit what the compiler was allowed to
    read. `permitted_columns` is the stance-derived allowlist: under a
    prospective stance a contaminated column is not merely discouraged, it was
    never reachable, and this records what the reachable set actually was.
    """

    model_config = ConfigDict(frozen=True)

    pattern: str
    stance: str
    dataset_id: str
    # Concept to the physical column it resolved to.
    concept_columns: dict[str, str] = Field(default_factory=dict)
    # Metrics compiled into this query, by registry name.
    metrics: tuple[str, ...] = ()
    # Columns the stance allowed the compiler to reach.
    permitted_columns: tuple[str, ...] = ()
    knowledge_cutoff: date | None = None
    # Monetary expressions, recorded so the DECIMAL boundary is auditable
    # without re-reading the SQL.
    monetary_expressions: tuple[str, ...] = ()
    fiscal_year_start_month: int = 1
    # "semantic" or "investigation": which analytical path compiled this.
    path: str = "semantic"
    # Usage grants the compiler relied on, with their provenance (13.2).
    usage_grants: tuple[str, ...] = ()
    # Every attributed dimension or feature and the snapshot it was read at.
    attributions: tuple[AttributionRead, ...] = ()
    # Typed comparisons compiled into this query (13.7).
    comparisons: tuple[str, ...] = ()


class ResultSet(BaseModel):
    """A materialized, addressable analytical result."""

    model_config = ConfigDict(extra="forbid")

    query_id: QueryId
    columns: list[ResultColumn]
    rows: list[list[CellValue]] = Field(default_factory=list)
    # The computed assessment. The tier is read from it, never set.
    trust: TrustAssessment = Field(default_factory=TrustAssessment)
    resolved_snapshots: list[ResolvedSnapshot] = Field(default_factory=list)
    compiled_sql: str | None = None
    spec_id: str | None = None
    truncated: bool = False
    warnings: list[str] = Field(default_factory=list)
    # Which dataset produced this. A result is meaningless without it once more
    # than one tenant exists.
    dataset_id: str | None = None
    # Every ambiguity resolved on the caller's behalf, stated rather than
    # implied: which snapshot a rule landed on, which measure was used, which
    # attribution rule applied.
    assumptions: list[str] = Field(default_factory=list)
    # Deterministic record of how the SQL was produced, for later provenance.
    compilation: CompilationMetadata | None = None

    @property
    def trust_tier(self) -> TrustTier:
        return self.trust.tier

    @property
    def trust_reasons(self) -> list[str]:
        return self.trust.reasons

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    @property
    def is_empty(self) -> bool:
        return not self.rows

    def cell(self, row: int, column: str) -> CellValue:
        """Look up one value the way an answer reference token addresses it."""
        try:
            index = self.column_names.index(column)
        except ValueError as exc:
            raise KeyError(
                f"column {column!r} not in result {self.query_id}; "
                f"available: {self.column_names}"
            ) from exc
        if not 0 <= row < len(self.rows):
            raise IndexError(
                f"row {row} out of range for result {self.query_id} "
                f"with {len(self.rows)} rows"
            )
        return self.rows[row][index]

    @model_validator(mode="after")
    def _rows_match_columns(self) -> ResultSet:
        width = len(self.columns)
        for i, row in enumerate(self.rows):
            if len(row) != width:
                raise ValueError(
                    f"row {i} has {len(row)} values but there are {width} columns"
                )
        return self
