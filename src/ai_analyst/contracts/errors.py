"""Typed error models and the exceptions that carry them.

Errors are data, not strings. The grain violation in particular must carry the
offending keys so ingestion can report them (ARCHITECTURE §7.2).
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from ai_analyst.contracts.schema import CanonicalColumn


class ErrorCode(StrEnum):
    MISSING_REQUIRED_COLUMNS = "missing_required_columns"
    GRAIN_VIOLATION = "grain_violation"
    NULL_GRAIN_KEY = "null_grain_key"
    CAPTURE_RESOLUTION_FAILED = "capture_resolution_failed"
    UNREADABLE_SOURCE = "unreadable_source"
    AMBIGUOUS_MAPPING = "ambiguous_mapping"
    UNCONFIRMED_REQUIRED_MAPPING = "unconfirmed_required_mapping"
    STATUS_CONFIG_INVALID = "status_config_invalid"
    INVALID_DATE_VALUES = "invalid_date_values"
    DATE_ENCODING_INVALID = "date_encoding_invalid"
    DATASET_INCONSISTENT = "dataset_inconsistent"
    PROFILING_FAILED = "profiling_failed"
    EXECUTION_FAILED = "execution_failed"
    VALIDATION_FAILED = "validation_failed"
    PLAN_REJECTED = "plan_rejected"


class ErrorDetail(BaseModel):
    """Base for every structured error payload."""

    model_config = ConfigDict(frozen=True)

    code: ErrorCode
    message: str


class MissingRequiredColumns(ErrorDetail):
    code: ErrorCode = ErrorCode.MISSING_REQUIRED_COLUMNS
    missing: list[CanonicalColumn]
    available_source_columns: list[str] = Field(default_factory=list)


class DuplicateKey(BaseModel):
    model_config = ConfigDict(frozen=True)

    as_of: date
    opp_id: str
    row_count: int


class GrainViolation(ErrorDetail):
    """(as_of, opp_id) was not unique. Ingestion stops here."""

    code: ErrorCode = ErrorCode.GRAIN_VIOLATION
    duplicate_key_count: int
    offending_row_count: int
    samples: list[DuplicateKey] = Field(default_factory=list)
    sample_limit: int = 20


class NullGrainKey(ErrorDetail):
    """as_of or opp_id was null or failed to parse."""

    code: ErrorCode = ErrorCode.NULL_GRAIN_KEY
    null_as_of_rows: int = 0
    null_opp_id_rows: int = 0


class CaptureResolutionFailed(ErrorDetail):
    """A declared capture policy could not be applied. Ingestion stops here."""

    code: ErrorCode = ErrorCode.CAPTURE_RESOLUTION_FAILED
    reason: str
    unresolvable_groups: int = 0
    samples: list[DuplicateKey] = Field(default_factory=list)


class InvalidDateValues(ErrorDetail):
    """A column declared as dates held values that are not dates (12.19).

    Raised rather than counted: a declared date column that quietly produced
    nulls would look like missing data instead of a wrong declaration.
    """

    code: ErrorCode = ErrorCode.INVALID_DATE_VALUES
    column: str
    encoding: str
    invalid_rows: int
    # Distinct offending raw values, capped. Dates are not identifying.
    samples: list[str] = Field(default_factory=list)


class DateEncodingInvalid(ErrorDetail):
    """A date encoding was declared for a column that cannot carry one."""

    code: ErrorCode = ErrorCode.DATE_ENCODING_INVALID
    column: str
    reason: str


class UnreadableSource(ErrorDetail):
    code: ErrorCode = ErrorCode.UNREADABLE_SOURCE
    path: str
    reason: str


class AmbiguousMapping(ErrorDetail):
    code: ErrorCode = ErrorCode.AMBIGUOUS_MAPPING
    canonical_column: CanonicalColumn
    candidates: list[str]


class FuzzyBinding(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_column: str
    canonical_column: CanonicalColumn
    score: float


class UnconfirmedRequiredMapping(ErrorDetail):
    """A required canonical column is bound only by an unconfirmed fuzzy match.

    A near-miss on an optional dimension is a recoverable annoyance. A near-miss
    on a required column corrupts every metric downstream, so it is refused
    until a human confirms it.
    """

    code: ErrorCode = ErrorCode.UNCONFIRMED_REQUIRED_MAPPING
    bindings: list[FuzzyBinding]


class StatusConfigInvalid(ErrorDetail):
    """The configured authoritative status column or value mapping is unusable."""

    code: ErrorCode = ErrorCode.STATUS_CONFIG_INVALID
    column: str | None = None
    reason: str = ""


class DatasetInconsistent(ErrorDetail):
    """Persisted schema, classifications, and profile no longer describe one dataset."""

    code: ErrorCode = ErrorCode.DATASET_INCONSISTENT
    dataset_id: str
    only_in_registry: list[str] = Field(default_factory=list)
    only_in_profile: list[str] = Field(default_factory=list)


class ProfilingFailed(ErrorDetail):
    code: ErrorCode = ErrorCode.PROFILING_FAILED
    reason: str


class ExecutionError(ErrorDetail):
    code: ErrorCode = ErrorCode.EXECUTION_FAILED
    sql: str | None = None
    reason: str = ""


class ValidationFailure(ErrorDetail):
    """A post-execution invariant did not hold (ARCHITECTURE §8.3)."""

    code: ErrorCode = ErrorCode.VALIDATION_FAILED
    check_name: str
    expected: str
    observed: str


class PlanRejection(ErrorDetail):
    """Structured rejection fed back to the planner (ARCHITECTURE §8.2)."""

    code: ErrorCode = ErrorCode.PLAN_REJECTED
    spec_id: str | None = None
    field: str | None = None
    reason: str = ""
    suggestions: list[str] = Field(default_factory=list)


class AnalystError(Exception):
    """Base exception. Always carries a structured detail payload."""

    def __init__(self, detail: ErrorDetail) -> None:
        super().__init__(detail.message)
        self.detail = detail


class IngestionError(AnalystError):
    """Raised anywhere in the ingestion pipeline."""


class MappingError(AnalystError):
    """Raised when source headers cannot be mapped onto the canonical schema."""


class ProfilingError(AnalystError):
    """Raised when a dataset cannot be profiled."""
