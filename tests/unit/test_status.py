"""Authoritative opportunity status (ARCHITECTURE 5.13)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ai_analyst.config import Settings
from ai_analyst.contracts.status import (
    OpportunityStatus,
    StageClass,
    StatusMapping,
    StatusResolution,
    StatusStrategy,
    stage_class,
    status_from_stage,
)

PRODUCTION_STAGES = [
    "0 - Qualification",
    "1 - Discovery",
    "2 - Solution Design",
    "3 - Proposal",
    "4 - Negotiation",
    "5 - Commit",
    "6 - Order Placed",
    "Closed Won",
    "Closed Lost",
    "not available",
    "SFDCDELETED",
]


@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        ("0 - Qualification", StageClass.PROGRESSION),
        ("5 - Commit", StageClass.PROGRESSION),
        ("6 - Order Placed", StageClass.PROGRESSION),
        ("Closed Won", StageClass.TERMINAL_WON),
        ("Closed Lost", StageClass.TERMINAL_LOST),
        ("SFDCDELETED", StageClass.INVALID),
        ("not available", StageClass.INVALID),
        (None, StageClass.INVALID),
    ],
)
def test_production_stage_vocabulary_classifies(stage, expected):
    assert stage_class(stage) is expected


def test_deleted_and_unavailable_rows_are_excluded_not_counted_as_open():
    # Counting a deleted record as open pipeline overstates the number silently.
    for stage in ("SFDCDELETED", "not available"):
        assert status_from_stage(stage) is OpportunityStatus.EXCLUDED
        assert not status_from_stage(stage).counts_as_pipeline


def test_final_sounding_stage_shows_why_stage_inference_is_not_authoritative():
    # Keyword inference reads a final-sounding stage as open, but only the tenant
    # knows whether it is open or won. This is the concrete reason status must
    # come from a dedicated column or a declared stage map.
    assert status_from_stage("6 - Order Placed") is OpportunityStatus.OPEN


def test_every_production_stage_resolves_without_error():
    assert len({status_from_stage(s) for s in PRODUCTION_STAGES}) >= 3


def test_status_flags_are_consistent():
    assert OpportunityStatus.WON.is_closed and OpportunityStatus.WON.is_won
    assert OpportunityStatus.LOST.is_closed and not OpportunityStatus.LOST.is_won
    assert not OpportunityStatus.OPEN.is_closed
    assert OpportunityStatus.OPEN.counts_as_pipeline
    assert not OpportunityStatus.EXCLUDED.counts_as_pipeline
    assert not OpportunityStatus.UNKNOWN.counts_as_pipeline


def test_status_mapping_resolves_case_insensitively():
    mapping = StatusMapping(won_values=("W",), lost_values=("L",), open_values=("O",))
    assert mapping.resolve("w") is OpportunityStatus.WON
    assert mapping.resolve(" L ") is OpportunityStatus.LOST
    assert mapping.resolve("o") is OpportunityStatus.OPEN


def test_unmapped_status_values_resolve_to_unknown_not_a_guess():
    mapping = StatusMapping(won_values=("W",), lost_values=("L",))
    assert mapping.resolve("something else") is OpportunityStatus.UNKNOWN
    assert mapping.resolve(None) is OpportunityStatus.UNKNOWN
    assert not OpportunityStatus.UNKNOWN.counts_as_pipeline


def test_a_value_cannot_mean_two_statuses():
    with pytest.raises(ValidationError, match="more than one status"):
        StatusMapping(won_values=("W",), lost_values=("w",))


def test_only_an_authoritative_column_is_authoritative():
    authoritative = StatusResolution(
        strategy=StatusStrategy.AUTHORITATIVE_COLUMN,
        column="status",
        mapping=StatusMapping(won_values=("W",)),
    )
    inferred = StatusResolution(strategy=StatusStrategy.STAGE_KEYWORD)
    assert authoritative.is_authoritative
    assert not authoritative.requires_confirmation
    assert not inferred.is_authoritative
    assert inferred.requires_confirmation


def test_authoritative_strategy_requires_a_column_and_mapping():
    with pytest.raises(ValidationError, match="requires a source column"):
        StatusResolution(strategy=StatusStrategy.AUTHORITATIVE_COLUMN)
    with pytest.raises(ValidationError, match="requires a value mapping"):
        StatusResolution(strategy=StatusStrategy.AUTHORITATIVE_COLUMN, column="s")


def test_status_source_is_configurable():
    assert Settings().status_mapping() is None
    configured = Settings(
        status_column="opp_status",
        status_won_values=("W",),
        status_lost_values=("L",),
        status_open_values=("O",),
        status_excluded_values=("DELETED",),
    )
    mapping = configured.status_mapping()
    assert mapping is not None
    assert mapping.resolve("W") is OpportunityStatus.WON
    assert mapping.resolve("DELETED") is OpportunityStatus.EXCLUDED
