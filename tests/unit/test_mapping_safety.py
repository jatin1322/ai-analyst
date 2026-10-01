"""Mapping safety on the real production headers.

On the real 129 headers the fuzzy matcher bound `close_date_qtr` (a quarter
label) to the required close date, and never found `new_amount`. Together they
would have let the production export ingest with an all-null close date.
"""

from __future__ import annotations

import pytest

from ai_analyst.contracts.errors import (
    MappingError,
    UnconfirmedRequiredMapping,
)
from ai_analyst.contracts.opportunity_snapshot_v1 import (
    ALL_COLUMNS,
    OPPORTUNITY_SNAPSHOT_V1,
)
from ai_analyst.contracts.schema import CanonicalColumn, MappingConfidence
from ai_analyst.data.dataset import register_dataset
from ai_analyst.data.mapping import (
    assert_mappable,
    mapping_from_registry,
    propose_mapping,
)
from ai_analyst.data.store import DuckDBStore
from tests.fixtures.production_shape import write_production_csv

HEADERS = list(ALL_COLUMNS)


def test_new_amount_is_a_confirmed_alias_for_amount():
    proposal = propose_mapping(["as_of", "opp_id", "close_date", "stage", "new_amount"])
    assert proposal.by_canonical()[CanonicalColumn.AMOUNT].source_column == "new_amount"
    assert proposal.is_complete


def test_a_fuzzy_match_never_conforms_whatever_column_it_targets():
    # The defect this guards: on the real headers the matcher bound
    # `close_date_qtr`, a quarter label, to the close date at 0.83. Once the
    # close date stopped being required for ingestion (ARCHITECTURE 12.15),
    # refusing only *required* fuzzy matches would have let that through
    # silently. A guess never conforms, required or not.
    proposal = propose_mapping(HEADERS)
    assert CanonicalColumn.CLOSE_DATE not in proposal.by_canonical()
    candidates = {m.canonical_column: m.source_column for m in proposal.fuzzy_candidates}
    assert candidates[CanonicalColumn.CLOSE_DATE] == "close_date_qtr"
    # The source column survives as a discovered column rather than being
    # consumed by the guess.
    assert "close_date_qtr" in proposal.unmapped_source_columns
    assert proposal.requires_confirmation


def test_ingestion_refuses_a_required_column_bound_only_by_a_guess():
    # The anti-guessing rule is unchanged for the grain, which is what is
    # required now. A near-miss on opp_id stops ingestion outright.
    headers = ["as_of", "opportunity_idx", "stage", "new_amount"]
    proposal = propose_mapping(headers)
    with pytest.raises(MappingError) as excinfo:
        assert_mappable(proposal, headers)
    detail = excinfo.value.detail
    assert isinstance(detail, UnconfirmedRequiredMapping)
    assert detail.bindings[0].source_column == "opportunity_idx"
    assert detail.bindings[0].canonical_column is CanonicalColumn.OPP_ID
    assert "mapping_overrides" in detail.message


def test_a_fuzzy_match_on_an_optional_column_is_still_only_flagged():
    proposal = propose_mapping(
        ["as_of", "opp_id", "close_date", "stage", "amount", "OwnerID"]
    )
    fuzzy = proposal.fuzzy_candidates
    assert [m.canonical_column for m in fuzzy] == [CanonicalColumn.OWNER_ID]
    assert all(m.confidence is not MappingConfidence.FUZZY for m in proposal.mappings)
    # Optional and unconfirmed, so ingestion proceeds with the column unbound.
    assert not proposal.unconfirmed_required
    assert proposal.is_complete
    assert proposal.unconfirmed_required == []
    assert proposal.is_complete


def test_registry_mapping_binds_exactly_the_documented_columns():
    mapping = mapping_from_registry(OPPORTUNITY_SNAPSHOT_V1, HEADERS)
    bound = {m.canonical_column.value: m.source_column for m in mapping.mappings}
    assert bound == {
        "as_of": "as_of",
        "opp_id": "opp_id",
        "amount": "new_amount",
        "stage": "Stage",
        "forecast_category": "ForecastCategory",
        "owner_id": "OwnerID",
        "account_id": "account_id",
        "probability": "Probability",
    }
    assert "close_date_qtr" in mapping.unmapped_source_columns
    assert not mapping.unconfirmed_required


def test_registry_mapping_never_uses_fuzzy_matching():
    mapping = mapping_from_registry(OPPORTUNITY_SNAPSHOT_V1, HEADERS)
    assert all(m.confidence is not MappingConfidence.FUZZY for m in mapping.mappings)


def test_registry_mapping_leaves_close_date_unbound_when_the_export_lacks_it():
    mapping = mapping_from_registry(OPPORTUNITY_SNAPSHOT_V1, HEADERS)
    assert CanonicalColumn.CLOSE_DATE not in mapping.by_canonical()
    # No longer a blocker: it is reconstructed at ingestion (ARCHITECTURE 12.15).
    assert mapping.missing_required == []


def test_registry_mapping_accepts_an_exact_canonical_header():
    mapping = mapping_from_registry(OPPORTUNITY_SNAPSHOT_V1, [*HEADERS, "close_date"])
    close = mapping.by_canonical()[CanonicalColumn.CLOSE_DATE]
    assert close.source_column == "close_date"
    assert close.confidence is MappingConfidence.EXACT
    assert mapping.missing_required == []


def test_the_production_export_now_ingests_with_a_reconstructed_close_date(
    tmp_path, settings
):
    # The export carries a horizon in days rather than a date. It is rebuilt as
    # as_of + days_to_close, and the rebuild is recorded rather than disguised.
    csv = tmp_path / "no_close_date.csv"
    write_production_csv(csv, include_close_date=False)
    dataset = register_dataset(
        csv,
        "nocd",
        column_registry=OPPORTUNITY_SNAPSHOT_V1,
        settings=settings,
        store=DuckDBStore(settings),
    )
    derived = {d.column: d for d in dataset.schema.derived_columns}
    close = derived[CanonicalColumn.CLOSE_DATE]
    assert close.is_reconstructed
    assert close.expression == "as_of + days_to_close"
    assert close.sources == ("as_of", "days_to_close")
    assert "carried no close_date column" in close.note


def test_without_a_registry_the_quarter_label_still_never_becomes_a_close_date(
    tmp_path, settings
):
    # Same file, no registry, so no reconstruction is documented and nothing
    # can rebuild the close date. The file ingests, and the important part is
    # what does *not* happen: close_date_qtr does not become close_date.
    csv = tmp_path / "no_close_date.csv"
    write_production_csv(csv, include_close_date=False)
    dataset = register_dataset(
        csv,
        "guess",
        mapping_overrides=None,
        settings=settings,
        store=DuckDBStore(settings),
    )
    assert CanonicalColumn.CLOSE_DATE not in {c.name for c in dataset.schema.columns}
    assert CanonicalColumn.CLOSE_DATE not in dataset.schema.mapping.by_canonical()
    assert "close_date_qtr" in dataset.schema.discovered_columns
