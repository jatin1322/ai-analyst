"""Canonical schema contract and column mapping.

The canonical schema is the vocabulary every downstream layer speaks. Real
uploads use arbitrary headers, so a `ColumnMapping` translates source headers
onto canonical names before anything else runs (ARCHITECTURE §1.4, §7.2).
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai_analyst.contracts.snapshot_policy import CaptureResolution
from ai_analyst.contracts.status import StatusResolution


class DataType(StrEnum):
    """Canonical storage types. Maps one-to-one onto DuckDB types."""

    DATE = "DATE"
    VARCHAR = "VARCHAR"
    DECIMAL = "DECIMAL"
    BOOLEAN = "BOOLEAN"
    INTEGER = "INTEGER"
    BIGINT = "BIGINT"
    DOUBLE = "DOUBLE"

    @property
    def duckdb_type(self) -> str:
        if self is DataType.DECIMAL:
            return "DECIMAL(18,2)"
        return self.value


class DateEncoding(StrEnum):
    """How a source column stores a date (ARCHITECTURE 12.19).

    Declared per source column, never detected. A numeric column is not a date
    because its values look like one: 45973 is an integer until somebody with
    authority says the column is an Excel serial.

    `EXCEL_SERIAL` columns also accept ISO date text, so a file that mixes the
    two representations still converts. Nothing else is accepted: an 8-digit
    integer such as 20250101 is out of the serial range and fails loudly rather
    than being guessed at.
    """

    ISO = "iso"
    EXCEL_SERIAL = "excel_serial"


# The Excel 1900 date system, as a serial number counted from this base date.
# It is 1899-12-30, not 1900-01-01, because Excel inherited Lotus 1-2-3's belief
# that 1900 was a leap year and so counts a nonexistent 1900-02-29 as serial 60.
# For every serial from 61 (1900-03-01) onward, base + serial is the true date.
# Earlier serials are ambiguous and are refused, as is anything past 9999-12-31.
EXCEL_EPOCH = "1899-12-30"
EXCEL_MIN_SERIAL = 61
EXCEL_MAX_SERIAL = 2_958_465


class CanonicalColumn(StrEnum):
    """Every column name the system is allowed to reason about."""

    AS_OF = "as_of"
    OPP_ID = "opp_id"
    CLOSE_DATE = "close_date"
    STAGE = "stage"
    AMOUNT = "amount"
    CREATED_DATE = "created_date"
    STATUS = "status"
    IS_CLOSED = "is_closed"
    IS_WON = "is_won"
    FORECAST_CATEGORY = "forecast_category"
    ARR = "arr"
    SEGMENT = "segment"
    REGION = "region"
    INDUSTRY = "industry"
    OWNER_ID = "owner_id"
    ACCOUNT_ID = "account_id"
    OPP_NAME = "opp_name"
    PROBABILITY = "probability"


class ColumnSpec(BaseModel):
    """Static definition of one canonical column."""

    model_config = ConfigDict(frozen=True)

    name: CanonicalColumn
    dtype: DataType
    required: bool
    description: str
    derivable: bool = False


CANONICAL_COLUMNS: dict[CanonicalColumn, ColumnSpec] = {
    spec.name: spec
    for spec in [
        ColumnSpec(
            name=CanonicalColumn.AS_OF,
            dtype=DataType.DATE,
            required=True,
            description="Snapshot date. Half of the primary key.",
        ),
        ColumnSpec(
            name=CanonicalColumn.OPP_ID,
            dtype=DataType.VARCHAR,
            required=True,
            description="Stable opportunity identifier. Half of the primary key.",
        ),
        ColumnSpec(
            name=CanonicalColumn.CLOSE_DATE,
            dtype=DataType.DATE,
            required=False,
            description=(
                "The close date recorded in this snapshot. Not required for "
                "ingestion: an export may carry it as a horizon in days instead, "
                "and a tenant without it is still a usable dataset."
            ),
        ),
        ColumnSpec(
            name=CanonicalColumn.STAGE,
            dtype=DataType.VARCHAR,
            required=False,
            description="Sales stage label. An attribute, never authoritative for status.",
        ),
        ColumnSpec(
            name=CanonicalColumn.AMOUNT,
            dtype=DataType.DECIMAL,
            required=False,
            description="Deal value in the reporting currency. Monetary.",
        ),
        ColumnSpec(
            name=CanonicalColumn.CREATED_DATE,
            dtype=DataType.DATE,
            required=False,
            description="Creation date. Falls back to first-seen snapshot when absent.",
        ),
        ColumnSpec(
            name=CanonicalColumn.STATUS,
            dtype=DataType.VARCHAR,
            required=False,
            derivable=True,
            description=(
                "Authoritative open, won, or lost state. Preferred over inferring "
                "the state from the stage label."
            ),
        ),
        ColumnSpec(
            name=CanonicalColumn.IS_CLOSED,
            dtype=DataType.BOOLEAN,
            required=False,
            derivable=True,
            description="Whether the opportunity is in a terminal stage.",
        ),
        ColumnSpec(
            name=CanonicalColumn.IS_WON,
            dtype=DataType.BOOLEAN,
            required=False,
            derivable=True,
            description="Whether the opportunity closed won.",
        ),
        ColumnSpec(
            name=CanonicalColumn.FORECAST_CATEGORY,
            dtype=DataType.VARCHAR,
            required=False,
            description="Commit, Best Case, Pipeline, or Omitted.",
        ),
        ColumnSpec(
            name=CanonicalColumn.ARR,
            dtype=DataType.DECIMAL,
            required=False,
            description="Annual recurring revenue. Falls back to amount when absent.",
        ),
        ColumnSpec(
            name=CanonicalColumn.SEGMENT,
            dtype=DataType.VARCHAR,
            required=False,
            description="Customer segment. Mutable across snapshots.",
        ),
        ColumnSpec(
            name=CanonicalColumn.REGION,
            dtype=DataType.VARCHAR,
            required=False,
            description="Sales region. Mutable across snapshots.",
        ),
        ColumnSpec(
            name=CanonicalColumn.INDUSTRY,
            dtype=DataType.VARCHAR,
            required=False,
            description="Account industry.",
        ),
        ColumnSpec(
            name=CanonicalColumn.OWNER_ID,
            dtype=DataType.VARCHAR,
            required=False,
            description="Opportunity owner. Mutable across snapshots.",
        ),
        ColumnSpec(
            name=CanonicalColumn.ACCOUNT_ID,
            dtype=DataType.VARCHAR,
            required=False,
            description="Account identifier.",
        ),
        ColumnSpec(
            name=CanonicalColumn.OPP_NAME,
            dtype=DataType.VARCHAR,
            required=False,
            description="Human-readable opportunity name.",
        ),
        ColumnSpec(
            name=CanonicalColumn.PROBABILITY,
            dtype=DataType.DECIMAL,
            required=False,
            description="Win probability. Used only when present.",
        ),
    ]
}

# The true ingestion minimum is the grain and nothing else (ARCHITECTURE 12.15).
# A tenant missing amount or a close date is still a dataset; what it cannot do
# is answer questions that need them, and that is decided at concept resolution
# by `ConceptBindings.check_operation`, not by refusing the file.
REQUIRED_COLUMNS: tuple[CanonicalColumn, ...] = tuple(
    name for name, spec in CANONICAL_COLUMNS.items() if spec.required
)

GRAIN_COLUMNS: tuple[CanonicalColumn, ...] = (CanonicalColumn.AS_OF, CanonicalColumn.OPP_ID)


class MappingConfidence(StrEnum):
    """How a source header was matched to a canonical column."""

    EXACT = "exact"
    ALIAS = "alias"
    FUZZY = "fuzzy"
    USER = "user"


class ColumnMapping(BaseModel):
    """One source header bound to one canonical column."""

    model_config = ConfigDict(frozen=True)

    source_column: str
    canonical_column: CanonicalColumn
    confidence: MappingConfidence
    score: float = Field(ge=0.0, le=1.0, default=1.0)

    @property
    def requires_confirmation(self) -> bool:
        return self.confidence is MappingConfidence.FUZZY


class MappingProposal(BaseModel):
    """Result of proposing a mapping. Never applied without validation.

    `mappings` holds only bindings nobody has to second-guess: an exact
    canonical name, a known alias, a user override, or an export registry's own
    coverage record. **A fuzzy name match is never in `mappings`.** It lands in
    `fuzzy_candidates`, which is a list of proposals for a human to confirm and
    is never conformed.

    That separation exists because of a real failure: on the production headers
    the fuzzy matcher bound `close_date_qtr`, a quarter label, to the close
    date. Once the close date stopped being required for ingestion
    (ARCHITECTURE 12.15), refusing only *required* fuzzy matches would have let
    exactly that binding through silently. A guess never conforms.
    """

    mappings: list[ColumnMapping] = Field(default_factory=list)
    unmapped_source_columns: list[str] = Field(default_factory=list)
    missing_required: list[CanonicalColumn] = Field(default_factory=list)
    # Fuzzy name matches. Surfaced as proposals and as inferred concept
    # bindings; never applied to the conformed table.
    fuzzy_candidates: list[ColumnMapping] = Field(default_factory=list)

    @property
    def requires_confirmation(self) -> bool:
        return bool(self.fuzzy_candidates) or any(
            m.requires_confirmation for m in self.mappings
        )

    @property
    def unconfirmed_required(self) -> list[ColumnMapping]:
        """Required columns a fuzzy match would have satisfied.

        Ingestion refuses these outright. An optional column is simply left
        unbound, with the candidate reported.
        """
        return [
            m for m in self.fuzzy_candidates if m.canonical_column in REQUIRED_COLUMNS
        ]

    @property
    def is_complete(self) -> bool:
        return not self.missing_required and not self.unconfirmed_required

    def by_canonical(self) -> dict[CanonicalColumn, ColumnMapping]:
        return {m.canonical_column: m for m in self.mappings}

    @model_validator(mode="after")
    def _no_duplicate_targets(self) -> MappingProposal:
        seen: set[CanonicalColumn] = set()
        for mapping in self.mappings:
            if mapping.canonical_column in seen:
                raise ValueError(
                    f"canonical column {mapping.canonical_column} mapped more than once"
                )
            seen.add(mapping.canonical_column)
        return self


class DerivationRule(StrEnum):
    """How a derivable column's values were produced."""

    OBSERVED = "observed"
    STATUS_COLUMN = "status_column"
    STAGE_KEYWORD = "stage_keyword"
    # Computed from other source columns by a declared derivation, such as a
    # close date rebuilt from a snapshot date and a horizon in days. Never a
    # silent cast: the expression and its sources are recorded (12.15).
    RECONSTRUCTED = "reconstructed"


class DerivedColumn(BaseModel):
    """Provenance for a column the system computed rather than read.

    Later layers must be able to tell an observed value from an inferred one.
    """

    model_config = ConfigDict(frozen=True)

    column: CanonicalColumn
    rule: DerivationRule
    note: str
    requires_confirmation: bool = True
    # For a RECONSTRUCTED column: the derivation as written, the source columns
    # it read, and the agreement tests that checked it. Recorded so an answer
    # can say the value was rebuilt rather than read.
    expression: str | None = None
    sources: tuple[str, ...] = ()
    agreement_test_ids: tuple[str, ...] = ()
    # Who declared a RECONSTRUCTED derivation. None means nobody accountable
    # did, and the concept is `UNDECLARED` (ARCHITECTURE 13.1).
    declared_by: str | None = None

    @property
    def is_reconstructed(self) -> bool:
        return self.rule is DerivationRule.RECONSTRUCTED


class DateConversion(BaseModel):
    """Provenance for a date column converted from another representation.

    Measured at ingestion against the raw values, so a reader can tell that
    `as_of` was decoded from Excel serials rather than read as dates, and how
    much of it needed decoding. Time of day is discarded: the conformed type is
    DATE, so a fractional serial keeps its whole-day part only.
    """

    model_config = ConfigDict(frozen=True)

    source_column: str
    column: str
    encoding: DateEncoding
    epoch: str = EXCEL_EPOCH
    serial_rows: int = 0
    iso_rows: int = 0
    null_rows: int = 0
    # Serial rows carrying a time-of-day fraction that was truncated.
    fractional_rows: int = 0
    min_date: date | None = None
    max_date: date | None = None
    note: str = ""

    @property
    def converted_from_serial(self) -> bool:
        return self.encoding is DateEncoding.EXCEL_SERIAL and self.serial_rows > 0


class DatasetSchema(BaseModel):
    """The conformed schema of one ingested dataset."""

    dataset_id: str
    source_path: str
    mapping: MappingProposal
    columns: list[ColumnSpec]
    derived_columns: list[DerivedColumn] = Field(default_factory=list)
    # Source cells that held text but did not survive their cast. Measured
    # during ingestion, where the raw text is still available.
    cast_failures: dict[str, int] = Field(default_factory=dict)
    # How open, won, and lost were determined for this dataset.
    status_resolution: StatusResolution | None = None
    # Source columns carried through unmapped, preserved as text so they can
    # be classified and catalogued (ARCHITECTURE 5.10).
    discovered_columns: list[str] = Field(default_factory=list)
    # Storage type detected for each discovered column at ingestion. Integers
    # are BIGINT, other numerics DOUBLE. Canonical money stays DECIMAL(18,2);
    # a discovered money-valued column is DOUBLE and must be cast before it is
    # ever used as a measure. A column that is not uniformly numeric, boolean
    # or date stays VARCHAR.
    discovered_types: dict[str, DataType] = Field(default_factory=dict)
    # Date columns decoded from a declared non-ISO representation, with counts.
    date_conversions: list[DateConversion] = Field(default_factory=list)
    # Set when the tenant declared `latest_capture` and ingestion de-duplicated
    # same-day captures. None means the strict default (any duplicate failed).
    capture_resolution: CaptureResolution | None = None

    @property
    def column_names(self) -> list[CanonicalColumn]:
        return [c.name for c in self.columns]

    def has(self, column: CanonicalColumn) -> bool:
        return column in set(self.column_names)

    @property
    def requires_confirmation(self) -> bool:
        status_unconfirmed = (
            self.status_resolution is not None
            and self.status_resolution.requires_confirmation
        )
        return (
            self.mapping.requires_confirmation
            or any(d.requires_confirmation for d in self.derived_columns)
            or status_unconfirmed
        )

    @property
    def status_is_authoritative(self) -> bool:
        return (
            self.status_resolution is not None
            and self.status_resolution.is_authoritative
        )

    @model_validator(mode="after")
    def _grain_present(self) -> DatasetSchema:
        present = set(self.column_names)
        missing = [c for c in GRAIN_COLUMNS if c not in present]
        if missing:
            raise ValueError(f"schema is missing grain columns: {missing}")
        return self
