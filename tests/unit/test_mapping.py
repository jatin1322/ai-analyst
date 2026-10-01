"""Column mapping from source headers onto the canonical schema."""

from __future__ import annotations

import pytest

from ai_analyst.config import Settings
from ai_analyst.contracts.errors import MappingError, MissingRequiredColumns
from ai_analyst.contracts.schema import CanonicalColumn, MappingConfidence
from ai_analyst.data.mapping import (
    assert_mappable,
    mapping_from_overrides,
    normalize,
    propose_mapping,
)

TINY_HEADERS = [
    "snapshot_date",
    "opportunity_id",
    "opportunity_name",
    "created",
    "expected_close_date",
    "sales_stage",
    "deal_amount",
    "annual_recurring_revenue",
    "customer_segment",
    "sales_region",
    "owner",
    "forecast_cat",
]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("As Of Date", "as_of_date"),
        ("  Snapshot-Date  ", "snapshot_date"),
        ("OPP__ID", "opp_id"),
        ("amount($)", "amount"),
    ],
)
def test_normalize(raw, expected):
    assert normalize(raw) == expected


def test_fixture_headers_map_completely(settings: Settings):
    proposal = propose_mapping(TINY_HEADERS, settings)
    assert proposal.is_complete
    assert proposal.missing_required == []
    assert proposal.unmapped_source_columns == []

    by_canonical = proposal.by_canonical()
    assert by_canonical[CanonicalColumn.AS_OF].source_column == "snapshot_date"
    assert by_canonical[CanonicalColumn.OPP_ID].source_column == "opportunity_id"
    assert by_canonical[CanonicalColumn.CLOSE_DATE].source_column == "expected_close_date"
    assert by_canonical[CanonicalColumn.STAGE].source_column == "sales_stage"
    assert by_canonical[CanonicalColumn.AMOUNT].source_column == "deal_amount"
    assert by_canonical[CanonicalColumn.ARR].source_column == "annual_recurring_revenue"
    assert by_canonical[CanonicalColumn.SEGMENT].source_column == "customer_segment"


def test_exact_canonical_names_win(settings: Settings):
    proposal = propose_mapping(
        ["as_of", "opp_id", "close_date", "stage", "amount"], settings
    )
    assert all(
        m.confidence is MappingConfidence.EXACT for m in proposal.mappings
    )
    assert not proposal.requires_confirmation


def test_alias_matches_are_not_flagged_for_confirmation(settings: Settings):
    proposal = propose_mapping(TINY_HEADERS, settings)
    assert not proposal.requires_confirmation


def test_fuzzy_match_is_flagged_for_confirmation(settings: Settings):
    proposal = propose_mapping(
        ["snapshot_date", "opportunity_idx", "close_date", "stage", "amount"], settings
    )
    # A fuzzy match is a proposal, never an applied binding (ARCHITECTURE 12.2).
    assert proposal.fuzzy_candidates, "expected a near-miss header to fuzzy match"
    assert all(m.confidence is not MappingConfidence.FUZZY for m in proposal.mappings)
    assert proposal.requires_confirmation


def test_fuzzy_never_displaces_an_alias_hit(settings: Settings):
    # "opportunity_id" is an alias for opp_id; "opportunity_idx" must not steal it.
    proposal = propose_mapping(
        ["snapshot_date", "opportunity_id", "opportunity_idx", "close_date", "stage", "amount"],
        settings,
    )
    opp_id = proposal.by_canonical()[CanonicalColumn.OPP_ID]
    assert opp_id.source_column == "opportunity_id"
    assert opp_id.confidence is MappingConfidence.ALIAS


def test_unrecognized_headers_are_reported_not_forced(settings: Settings):
    proposal = propose_mapping(
        ["snapshot_date", "opportunity_id", "close_date", "stage", "amount", "zzz_internal_xyz"],
        settings,
    )
    assert "zzz_internal_xyz" in proposal.unmapped_source_columns


def test_missing_required_columns_are_listed(settings: Settings):
    # Only the grain is required now (ARCHITECTURE 12.15), so a file with both
    # grain columns is complete however little else it carries.
    proposal = propose_mapping(["snapshot_date", "opportunity_id"], settings)
    assert proposal.is_complete
    assert proposal.missing_required == []

    without_grain = propose_mapping(["stage", "amount"], settings)
    assert not without_grain.is_complete
    assert set(without_grain.missing_required) == {
        CanonicalColumn.AS_OF,
        CanonicalColumn.OPP_ID,
    }


def test_assert_mappable_fails_loudly_with_a_typed_error(settings: Settings):
    headers = ["stage", "amount"]
    proposal = propose_mapping(headers, settings)
    with pytest.raises(MappingError) as excinfo:
        assert_mappable(proposal, headers)
    detail = excinfo.value.detail
    assert isinstance(detail, MissingRequiredColumns)
    assert CanonicalColumn.OPP_ID in detail.missing
    assert "never guesses a primary key" in detail.message


def test_assert_mappable_passes_on_a_complete_proposal(settings: Settings):
    proposal = propose_mapping(TINY_HEADERS, settings)
    assert_mappable(proposal, TINY_HEADERS)


def test_user_overrides_are_authoritative():
    proposal = mapping_from_overrides(
        ["a", "b", "c", "d", "e", "spare"],
        {
            "a": CanonicalColumn.AS_OF,
            "b": CanonicalColumn.OPP_ID,
            "c": CanonicalColumn.CLOSE_DATE,
            "d": CanonicalColumn.STAGE,
            "e": CanonicalColumn.AMOUNT,
        },
    )
    assert proposal.is_complete
    assert not proposal.requires_confirmation
    assert proposal.unmapped_source_columns == ["spare"]
    assert all(m.confidence is MappingConfidence.USER for m in proposal.mappings)


def test_overrides_referencing_absent_columns_are_rejected():
    with pytest.raises(MappingError, match="not present in source"):
        mapping_from_overrides(["a"], {"nope": CanonicalColumn.AS_OF})
