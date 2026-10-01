"""Dataset profile contracts (ARCHITECTURE §7.3).

Numeric extremes are stored as exact strings rather than floats. Money must
never round-trip through a float anywhere in this system.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from ai_analyst.contracts.columns import MonetaryStatus
from ai_analyst.contracts.schema import DataType
from ai_analyst.contracts.status import StatusStrategy


class SnapshotInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    as_of: date
    row_count: int


class TopValue(BaseModel):
    model_config = ConfigDict(frozen=True)

    value: str | None
    count: int


class ProfileKind(StrEnum):
    """The shape of a column's observed values.

    This describes what the data looks like, not what the column means. A
    column's meaning lives in its classification and is never changed by this.
    """

    CATEGORICAL = "categorical"
    NUMERIC = "numeric"
    DATE = "date"
    BOOLEAN = "boolean"
    TEXT = "text"
    EMPTY = "empty"


class ColumnProfile(BaseModel):
    """Observed statistics for one column. Never a statement of meaning.

    `min_value` and `max_value` are exact strings so a Decimal never passes
    through a float. `mean`, `median`, and `stddev` are only computed for
    integer and floating features; a DECIMAL money column reports its exact
    extremes and nothing else, so no amount is ever averaged as a float.

    For a column with declared sentinel values, every statistic is computed over
    the observed values only. `null_count` counts SQL nulls, `sentinel_count`
    counts rows holding a sentinel (missing history), and `observed_count` is
    what is left. The three are distinct on purpose.

    Text content is never recorded: text columns carry length statistics only.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    dtype: DataType
    kind: ProfileKind = ProfileKind.CATEGORICAL
    row_count: int
    null_count: int
    distinct_count: int
    min_value: str | None = None
    max_value: str | None = None
    top_values: list[TopValue] = Field(default_factory=list)
    # True when top values were withheld because the column has too many
    # distinct values to list compactly.
    high_cardinality: bool = False
    mean: float | None = None
    median: float | None = None
    stddev: float | None = None
    mean_length: float | None = None
    max_length: int | None = None
    sentinel_values: tuple[float, ...] = ()
    sentinel_count: int = 0
    observed_count: int | None = None
    # Why summary statistics may be absent (ARCHITECTURE 12.16). Money and
    # unknown-status columns report exact extremes and no float mean.
    monetary: MonetaryStatus = MonetaryStatus.NON_MONETARY

    @property
    def null_rate(self) -> float:
        return 0.0 if self.row_count == 0 else self.null_count / self.row_count

    @property
    def has_sentinel_declaration(self) -> bool:
        return bool(self.sentinel_values)

    @property
    def has_float_summary(self) -> bool:
        return self.mean is not None or self.median is not None or self.stddev is not None

    @property
    def summary_withheld_reason(self) -> str | None:
        """Why a numeric column reports no float mean, median or deviation.

        Structural numeric profiling (counts, exact extremes) is always given.
        Float summaries are withheld for money, and for a column nobody has
        established as non-money, so an absent mean is never a mystery.
        """
        if self.kind is not ProfileKind.NUMERIC or self.has_float_summary:
            return None
        if self.monetary is MonetaryStatus.MONETARY:
            return "monetary column: exact extremes only, never a float summary"
        if self.monetary is MonetaryStatus.UNKNOWN:
            return "monetary status unknown: not established as non-monetary"
        return "decimal storage: exact extremes only"


class StageClassification(BaseModel):
    """One observed stage label and its inferred terminal semantics."""

    model_config = ConfigDict(frozen=True)

    stage: str
    row_count: int
    is_closed: bool
    is_won: bool
    inferred: bool = True


class StageVocabulary(BaseModel):
    """Observed stages plus the closed/won mapping, flagged for confirmation."""

    model_config = ConfigDict(frozen=True)

    stages: list[StageClassification] = Field(default_factory=list)
    requires_confirmation: bool = True

    @property
    def labels(self) -> list[str]:
        return [s.stage for s in self.stages]

    @property
    def won_labels(self) -> list[str]:
        return [s.stage for s in self.stages if s.is_won]

    @property
    def lost_labels(self) -> list[str]:
        return [s.stage for s in self.stages if s.is_closed and not s.is_won]

    @property
    def open_labels(self) -> list[str]:
        return [s.stage for s in self.stages if not s.is_closed]


class TextColumnProfile(BaseModel):
    """Catalogue entry for a free-text column (ARCHITECTURE 5.11).

    Catalogue only. No content is summarized, embedded, or retrieved. Text
    columns are never dimensions or measures, and admit only null and non-null
    filters, because grouping by free text produces a meaningless
    high-cardinality result.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    row_count: int
    null_count: int
    distinct_count: int
    mean_length: float
    max_length: int
    appears_append_only: bool | None = None
    # "classification": the registry declares the column as text.
    # "detected": nothing declares it; its shape merely looks like text.
    basis: str = "detected"
    note: str = ""

    @property
    def distinct_ratio(self) -> float:
        populated = self.row_count - self.null_count
        return 0.0 if populated == 0 else self.distinct_count / populated


class LifecycleStats(BaseModel):
    """How opportunities behave across snapshots.

    `vanished_without_terminal_state` is the population behind the
    `other_removed` bridge term (ARCHITECTURE §5.2). A large value here makes
    every derived rate suspect, so it is surfaced at profiling time.
    """

    model_config = ConfigDict(frozen=True)

    distinct_opportunities: int
    snapshot_count: int
    min_snapshots_per_opportunity: int
    max_snapshots_per_opportunity: int
    median_snapshots_per_opportunity: float
    present_in_all_snapshots: int
    vanished_without_terminal_state: int
    vanished_sample_opp_ids: list[str] = Field(default_factory=list)


class DataQualityFlags(BaseModel):
    model_config = ConfigDict(frozen=True)

    negative_amount_rows: int = 0
    close_date_before_created_date_rows: int = 0
    cast_failure_rows: dict[str, int] = Field(default_factory=dict)

    @property
    def any_flagged(self) -> bool:
        return bool(
            self.negative_amount_rows
            or self.close_date_before_created_date_rows
            or any(self.cast_failure_rows.values())
        )


class StatusProfile(BaseModel):
    """How opportunity status was resolved, and what values resulted.

    Makes a missing authoritative status source visible in the profile rather
    than only in the schema. When status was inferred from stage labels this says
    so, because every rate derived from it inherits that.
    """

    model_config = ConfigDict(frozen=True)

    strategy: StatusStrategy
    authoritative: bool
    configured_column: str | None = None
    distribution: dict[str, int] = Field(default_factory=dict)
    unmapped_values: list[str] = Field(default_factory=list)
    requires_confirmation: bool = True
    note: str = ""


class GrainCheck(BaseModel):
    """Result of the (as_of, opp_id) uniqueness assertion."""

    model_config = ConfigDict(frozen=True)

    passed: bool
    total_rows: int
    distinct_keys: int
    duplicate_key_count: int = 0


class DatasetProfile(BaseModel):
    """Everything known about an ingested dataset, computed once and persisted."""

    dataset_id: str
    row_count: int
    snapshots: list[SnapshotInfo] = Field(default_factory=list)
    columns: list[ColumnProfile] = Field(default_factory=list)
    stage_vocabulary: StageVocabulary
    lifecycle: LifecycleStats
    grain: GrainCheck
    quality: DataQualityFlags = Field(default_factory=DataQualityFlags)
    text_columns: list[TextColumnProfile] = Field(default_factory=list)
    status: StatusProfile | None = None
    # Fiscal boundary drift is computed by the semantic calendar, which is a
    # later milestone. Left unset until then rather than guessed at.
    snapshot_drift_days: dict[str, int] | None = None

    @property
    def snapshot_dates(self) -> list[date]:
        return [s.as_of for s in self.snapshots]

    @property
    def min_as_of(self) -> date | None:
        return min(self.snapshot_dates) if self.snapshots else None

    @property
    def max_as_of(self) -> date | None:
        return max(self.snapshot_dates) if self.snapshots else None

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def column(self, name: str) -> ColumnProfile:
        for c in self.columns:
            if c.name == name:
                return c
        raise KeyError(f"column {name} not profiled")
