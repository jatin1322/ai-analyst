"""Canonical schema and mapping contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ai_analyst.contracts.schema import (
    CANONICAL_COLUMNS,
    GRAIN_COLUMNS,
    REQUIRED_COLUMNS,
    CanonicalColumn,
    ColumnMapping,
    DatasetSchema,
    DataType,
    DerivationRule,
    DerivedColumn,
    MappingConfidence,
    MappingProposal,
)


def test_the_ingestion_minimum_is_the_grain_and_nothing_else():
    # ARCHITECTURE 12.15: a tenant missing amount, stage, or a close date is
    # still a dataset. What it cannot do is answer questions that need them,
    # and that is decided at concept resolution, not by refusing the file.
    assert set(REQUIRED_COLUMNS) == {CanonicalColumn.AS_OF, CanonicalColumn.OPP_ID}
    for optional in (
        CanonicalColumn.CLOSE_DATE,
        CanonicalColumn.STAGE,
        CanonicalColumn.AMOUNT,
    ):
        assert not CANONICAL_COLUMNS[optional].required


def test_grain_is_as_of_and_opp_id():
    assert GRAIN_COLUMNS == (CanonicalColumn.AS_OF, CanonicalColumn.OPP_ID)


def test_every_canonical_column_has_a_spec():
    assert set(CANONICAL_COLUMNS) == set(CanonicalColumn)


def test_decimal_maps_to_fixed_precision():
    assert DataType.DECIMAL.duckdb_type == "DECIMAL(18,2)"
    assert DataType.DATE.duckdb_type == "DATE"


def test_only_status_and_its_flags_are_derivable():
    derivable = {n for n, s in CANONICAL_COLUMNS.items() if s.derivable}
    assert derivable == {
        CanonicalColumn.STATUS,
        CanonicalColumn.IS_CLOSED,
        CanonicalColumn.IS_WON,
    }


def test_fuzzy_mapping_requires_confirmation():
    fuzzy = ColumnMapping(
        source_column="oppty_identifier",
        canonical_column=CanonicalColumn.OPP_ID,
        confidence=MappingConfidence.FUZZY,
        score=0.9,
    )
    exact = ColumnMapping(
        source_column="opp_id",
        canonical_column=CanonicalColumn.OPP_ID,
        confidence=MappingConfidence.EXACT,
    )
    assert fuzzy.requires_confirmation
    assert not exact.requires_confirmation


def test_proposal_rejects_two_sources_for_one_canonical_column():
    with pytest.raises(ValidationError, match="mapped more than once"):
        MappingProposal(
            mappings=[
                ColumnMapping(
                    source_column="a",
                    canonical_column=CanonicalColumn.AMOUNT,
                    confidence=MappingConfidence.EXACT,
                ),
                ColumnMapping(
                    source_column="b",
                    canonical_column=CanonicalColumn.AMOUNT,
                    confidence=MappingConfidence.ALIAS,
                ),
            ]
        )


def test_schema_rejects_missing_grain_columns():
    proposal = MappingProposal(
        mappings=[
            ColumnMapping(
                source_column="amount",
                canonical_column=CanonicalColumn.AMOUNT,
                confidence=MappingConfidence.EXACT,
            )
        ],
        missing_required=[CanonicalColumn.AS_OF],
    )
    with pytest.raises(ValidationError, match="missing grain columns"):
        DatasetSchema(
            dataset_id="d",
            source_path="/tmp/x.csv",
            mapping=proposal,
            columns=[CANONICAL_COLUMNS[CanonicalColumn.AMOUNT]],
        )


def test_schema_requires_confirmation_when_a_flag_was_inferred():
    proposal = MappingProposal(
        mappings=[
            ColumnMapping(
                source_column=c.value,
                canonical_column=c,
                confidence=MappingConfidence.EXACT,
            )
            for c in GRAIN_COLUMNS
        ]
    )
    schema = DatasetSchema(
        dataset_id="d",
        source_path="/tmp/x.csv",
        mapping=proposal,
        columns=[CANONICAL_COLUMNS[c] for c in GRAIN_COLUMNS],
        derived_columns=[
            DerivedColumn(
                column=CanonicalColumn.IS_WON,
                rule=DerivationRule.STAGE_KEYWORD,
                note="inferred",
            )
        ],
    )
    assert schema.requires_confirmation
    assert schema.has(CanonicalColumn.AS_OF)
    assert not schema.has(CanonicalColumn.ARR)
