"""The stable analytical ontology (ARCHITECTURE 12.1).

A concept is an idea, not a column. These tests hold that line: nothing in the
ontology may name a tenant's physical column, because the moment one does, one
tenant's schema has been baked into the vocabulary every other tenant speaks.
"""

from __future__ import annotations

import pytest

from ai_analyst.contracts.concepts import (
    CONCEPTS,
    GRAIN_CONCEPTS,
    MONETARY_CONCEPTS,
    RETROSPECTIVE_CONCEPTS,
    AnalyticalOperation,
    BusinessConcept,
    ConceptCardinality,
    ConceptDefinition,
    ConceptRole,
    SemanticType,
    concepts_required_for,
    definition,
)
from ai_analyst.contracts.schema import DataType


def test_every_concept_is_defined_exactly_once():
    assert set(CONCEPTS) == set(BusinessConcept)
    for concept, spec in CONCEPTS.items():
        assert spec.concept is concept


def test_every_concept_carries_the_metadata_the_milestone_requires():
    for spec in CONCEPTS.values():
        assert spec.concept.value  # stable identifier
        assert spec.display_name.strip()
        assert len(spec.definition) > 30, f"{spec.concept}: definition is too thin"
        assert isinstance(spec.semantic_type, SemanticType)
        assert isinstance(spec.expected_dtype, DataType)
        assert isinstance(spec.role, ConceptRole)
        # "whether it is required for any known analytical operation"
        assert isinstance(spec.is_required_for_any_operation, bool)


def test_no_concept_names_a_physical_column():
    # The whole point of the ontology. If `new_amount` or `ForecastCategory`
    # appeared here, one tenant's headers would have become the vocabulary.
    # Tenant headers that are unambiguously physical. "stage" is not in the
    # list: it is a genuine concept word as well as one tenant's column name.
    physical = (
        "new_amount",
        "ForecastCategory",
        "opp_id",
        "as_of",
        "close_date_qtr",
        "terminal_fate",
        "OwnerID",
    )
    blob = " ".join(
        f"{s.concept.value} {s.display_name} {s.definition}" for s in CONCEPTS.values()
    )
    for name in physical:
        assert name not in blob, f"the ontology names a physical column: {name}"


def test_the_grain_concepts_are_required_for_every_operation():
    for concept in GRAIN_CONCEPTS:
        assert set(definition(concept).required_for) == set(AnalyticalOperation)


def test_win_rate_needs_status_and_not_stage():
    # ARCHITECTURE 5.13: stage is an attribute, never authoritative for a rate.
    needed = set(concepts_required_for(AnalyticalOperation.WIN_RATE))
    assert BusinessConcept.OPPORTUNITY_STATUS in needed
    assert BusinessConcept.STAGE not in needed


def test_segment_analysis_requires_a_segment_concept():
    needed = concepts_required_for(AnalyticalOperation.SEGMENT_ANALYSIS)
    assert BusinessConcept.CUSTOMER_SEGMENT in needed


def test_money_concepts_are_declared_semantically():
    # Declared on the concept, never inferred from a numeric shape. Two decimal
    # places make a ratio, not money (ARCHITECTURE 12.16).
    assert {BusinessConcept.AMOUNT, BusinessConcept.TERMINAL_AMOUNT} == MONETARY_CONCEPTS
    assert definition(BusinessConcept.AMOUNT).semantic_type is SemanticType.MONEY
    assert definition(BusinessConcept.STAGE).semantic_type is SemanticType.CATEGORY


def test_retrospective_concepts_are_exactly_the_terminal_ones():
    assert {
        BusinessConcept.TERMINAL_OUTCOME,
        BusinessConcept.TERMINAL_DATE,
        BusinessConcept.TERMINAL_AMOUNT,
    } == RETROSPECTIVE_CONCEPTS


def test_quarter_is_computed_rather_than_bound():
    # ARCHITECTURE 12.1: a stamped quarter label may have been derived from the
    # final close date, so the calendar computes it and never reads it.
    quarter = definition(BusinessConcept.QUARTER)
    assert quarter.computed
    assert quarter.derived_from is BusinessConcept.SNAPSHOT_DATE


def test_narrative_text_is_the_only_many_cardinality_concept():
    many = [c for c, s in CONCEPTS.items() if s.cardinality is ConceptCardinality.MANY]
    assert many == [BusinessConcept.NARRATIVE_TEXT]


def test_a_computed_concept_must_name_its_source():
    with pytest.raises(ValueError, match="must name what it derives from"):
        ConceptDefinition(
            concept=BusinessConcept.QUARTER,
            display_name="Quarter",
            definition="x" * 40,
            semantic_type=SemanticType.PERIOD,
            expected_dtype=DataType.VARCHAR,
            role=ConceptRole.TEMPORAL,
            computed=True,
        )
