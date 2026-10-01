"""Error models.

ARCHITECTURE §7.2 requires ingestion to report the offending duplicate keys, so
the payload has to carry them.
"""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from ai_analyst.contracts.errors import (
    AnalystError,
    DuplicateKey,
    ErrorCode,
    GrainViolation,
    IngestionError,
    MissingRequiredColumns,
    PlanRejection,
    ValidationFailure,
)
from ai_analyst.contracts.schema import CanonicalColumn


def test_grain_violation_carries_offending_keys():
    detail = GrainViolation(
        message="not unique",
        duplicate_key_count=2,
        offending_row_count=4,
        samples=[
            DuplicateKey(as_of=date(2025, 1, 1), opp_id="OPP-001", row_count=2),
            DuplicateKey(as_of=date(2025, 2, 1), opp_id="OPP-002", row_count=2),
        ],
    )
    assert detail.code is ErrorCode.GRAIN_VIOLATION
    assert [s.opp_id for s in detail.samples] == ["OPP-001", "OPP-002"]


def test_missing_required_columns_enumerates_what_is_missing():
    detail = MissingRequiredColumns(
        message="missing",
        missing=[CanonicalColumn.STAGE, CanonicalColumn.CLOSE_DATE],
        available_source_columns=["a", "b"],
    )
    assert CanonicalColumn.STAGE in detail.missing


def test_exceptions_carry_the_structured_detail():
    detail = MissingRequiredColumns(message="boom", missing=[CanonicalColumn.AS_OF])
    with pytest.raises(IngestionError) as excinfo:
        raise IngestionError(detail)
    assert excinfo.value.detail is detail
    assert str(excinfo.value) == "boom"
    assert isinstance(excinfo.value, AnalystError)


def test_validation_failure_records_expected_and_observed():
    failure = ValidationFailure(
        message="bridge did not balance",
        check_name="bridge_balance",
        expected="0.00",
        observed="1250.00",
    )
    assert failure.code is ErrorCode.VALIDATION_FAILED


def test_plan_rejection_can_carry_suggestions():
    rejection = PlanRejection(
        message="unknown filter value",
        spec_id="q1",
        field="filters[0].values[0]",
        reason="'Enterprise' not found",
        suggestions=["ENT"],
    )
    assert rejection.suggestions == ["ENT"]


def test_error_details_are_immutable():
    detail = PlanRejection(message="x")
    with pytest.raises(ValidationError):
        detail.message = "y"
