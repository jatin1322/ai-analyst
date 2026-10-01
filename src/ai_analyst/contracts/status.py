"""Opportunity status: the authoritative open / won / lost state.

Status and stage are different things and this module keeps them apart.

`Stage` is a snapshot attribute describing where an opportunity sits in the
sales process. It is used for progression, regression, dwell and distribution
analysis. It is **not** the authoritative source of whether an opportunity is
open, won, or lost, because a stage vocabulary can encode outcomes in ways
keyword inference cannot read. In the production export, `6 - Order Placed`
sounds final but is open pipeline, which only the tenant can say, and both
`SFDCDELETED` and `not available` would be counted as open pipeline.

Status resolution is therefore configurable, with a strategy chain that records
which strategy was used and whether it was authoritative.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class OpportunityStatus(StrEnum):
    """The authoritative state of an opportunity in one snapshot."""

    OPEN = "open"
    WON = "won"
    LOST = "lost"
    EXCLUDED = "excluded"
    UNKNOWN = "unknown"

    @property
    def is_closed(self) -> bool:
        return self in (OpportunityStatus.WON, OpportunityStatus.LOST)

    @property
    def is_won(self) -> bool:
        return self is OpportunityStatus.WON

    @property
    def counts_as_pipeline(self) -> bool:
        """Whether a row in this state may enter a pipeline measure at all."""
        return self is OpportunityStatus.OPEN


class StageClass(StrEnum):
    """What a stage label represents, independent of status."""

    PROGRESSION = "progression"
    TERMINAL_WON = "terminal_won"
    TERMINAL_LOST = "terminal_lost"
    INVALID = "invalid"


class StatusStrategy(StrEnum):
    """How status was resolved, in descending order of trust."""

    AUTHORITATIVE_COLUMN = "authoritative_column"
    # A tenant declared what every stage value means (WP5). Authoritative
    # because a person wrote it down; a value nobody declared is EXCLUDED.
    DECLARED_STAGE_MAP = "declared_stage_map"
    MAPPED_FLAGS = "mapped_flags"
    STAGE_KEYWORD = "stage_keyword"
    UNRESOLVED = "unresolved"

    @property
    def is_authoritative(self) -> bool:
        return self in (
            StatusStrategy.AUTHORITATIVE_COLUMN,
            StatusStrategy.DECLARED_STAGE_MAP,
        )


class StatusMapping(BaseModel):
    """Maps raw values of a status column onto `OpportunityStatus`.

    Matching is case-insensitive and whitespace-trimmed. Any value not listed
    resolves to `UNKNOWN`, which is surfaced rather than guessed at.
    """

    model_config = ConfigDict(frozen=True)

    won_values: tuple[str, ...] = ()
    lost_values: tuple[str, ...] = ()
    open_values: tuple[str, ...] = ()
    excluded_values: tuple[str, ...] = ()

    @staticmethod
    def _norm(value: str) -> str:
        return value.strip().lower()

    def resolve(self, raw: str | None) -> OpportunityStatus:
        if raw is None:
            return OpportunityStatus.UNKNOWN
        value = self._norm(raw)
        if value in {self._norm(v) for v in self.excluded_values}:
            return OpportunityStatus.EXCLUDED
        if value in {self._norm(v) for v in self.won_values}:
            return OpportunityStatus.WON
        if value in {self._norm(v) for v in self.lost_values}:
            return OpportunityStatus.LOST
        if value in {self._norm(v) for v in self.open_values}:
            return OpportunityStatus.OPEN
        return OpportunityStatus.UNKNOWN

    @property
    def all_values(self) -> tuple[str, ...]:
        return (
            self.won_values + self.lost_values + self.open_values + self.excluded_values
        )

    @model_validator(mode="after")
    def _no_value_in_two_buckets(self) -> StatusMapping:
        seen: set[str] = set()
        for value in self.all_values:
            key = self._norm(value)
            if key in seen:
                raise ValueError(f"status value {value!r} is mapped to more than one status")
            seen.add(key)
        return self


class StatusResolution(BaseModel):
    """Which strategy resolved status for one dataset, and how much to trust it."""

    strategy: StatusStrategy
    column: str | None = None
    mapping: StatusMapping | None = None
    unmapped_values: list[str] = Field(default_factory=list)
    # Declared stage map only: the exact stage-to-status lookup used, and how
    # many rows fell outside it (they are EXCLUDED, never open).
    stage_status_map: dict[str, OpportunityStatus] | None = None
    unmapped_rows: int = 0
    note: str = ""

    @property
    def is_authoritative(self) -> bool:
        return self.strategy.is_authoritative

    @property
    def requires_confirmation(self) -> bool:
        return not self.is_authoritative or bool(self.unmapped_values) or self.unmapped_rows > 0

    @model_validator(mode="after")
    def _column_strategies_name_a_column(self) -> StatusResolution:
        needs_column = self.strategy in (
            StatusStrategy.AUTHORITATIVE_COLUMN,
            StatusStrategy.MAPPED_FLAGS,
            StatusStrategy.DECLARED_STAGE_MAP,
        )
        if needs_column and not self.column:
            raise ValueError(f"strategy {self.strategy} requires a source column")
        if self.strategy is StatusStrategy.AUTHORITATIVE_COLUMN and self.mapping is None:
            raise ValueError("an authoritative status column requires a value mapping")
        if self.strategy is StatusStrategy.DECLARED_STAGE_MAP and self.stage_status_map is None:
            raise ValueError("a declared stage map resolution must carry the map")
        return self


# Stage labels that carry no sales-process meaning. A row in one of these is not
# open pipeline, and counting it as such is a silent overstatement.
INVALID_STAGE_VALUES: tuple[str, ...] = ("sfdcdeleted", "not available", "n/a", "unknown", "")

WON_STAGE_KEYWORDS: tuple[str, ...] = ("won", "win")
LOST_STAGE_KEYWORDS: tuple[str, ...] = (
    "lost",
    "disqualif",
    "no decision",
    "abandoned",
    "cancelled",
    "canceled",
    "dead",
)


def stage_class(stage: str | None) -> StageClass:
    """Classify a stage label. Never a substitute for authoritative status."""
    if stage is None:
        return StageClass.INVALID
    normalized = stage.strip().lower()
    if normalized in INVALID_STAGE_VALUES:
        return StageClass.INVALID
    if any(k in normalized for k in LOST_STAGE_KEYWORDS):
        return StageClass.TERMINAL_LOST
    if any(k in normalized for k in WON_STAGE_KEYWORDS):
        return StageClass.TERMINAL_WON
    return StageClass.PROGRESSION


def status_from_stage(stage: str | None) -> OpportunityStatus:
    """Last-resort status inference. Always recorded as non-authoritative."""
    match stage_class(stage):
        case StageClass.TERMINAL_WON:
            return OpportunityStatus.WON
        case StageClass.TERMINAL_LOST:
            return OpportunityStatus.LOST
        case StageClass.INVALID:
            return OpportunityStatus.EXCLUDED
        case _:
            return OpportunityStatus.OPEN
