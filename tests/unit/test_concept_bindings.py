"""Concept bindings and the rule that a guess never confirms (ARCHITECTURE 12.2).

The central test in this file is
`test_name_evidence_can_never_produce_a_confirmed_binding`. Everything else
supports it. It exists because the failure it prevents already happened once:
the fuzzy matcher bound `close_date_qtr`, a quarter label, to the required
close date, and nothing in the type system objected.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ai_analyst.contracts.agreement import (
    AgreementKind,
    AgreementReport,
    AgreementResult,
)
from ai_analyst.contracts.binding import (
    BindingEvidence,
    BindingStatus,
    ConceptBinding,
    ConceptBindings,
    EvidenceKind,
)
from ai_analyst.contracts.concepts import AnalyticalOperation, BusinessConcept
from ai_analyst.contracts.tenant import TenantProfile

DECLARATIVE = (
    EvidenceKind.USER_CONFIRMATION,
    EvidenceKind.TENANT_CONFIG,
    EvidenceKind.EXPORT_REGISTRY,
    EvidenceKind.DOCUMENTATION,
)
INFERENTIAL = (
    EvidenceKind.EXACT_NAME,
    EvidenceKind.ALIAS,
    EvidenceKind.FUZZY_NAME,
    EvidenceKind.VALUE_PATTERN,
    EvidenceKind.TYPE_SHAPE,
    EvidenceKind.AGREEMENT_TEST,
    EvidenceKind.DERIVATION,
)


def _binding(status: BindingStatus, evidence: tuple[BindingEvidence, ...], **kwargs):
    return ConceptBinding(
        dataset_id="d",
        concept=BusinessConcept.AMOUNT,
        columns=kwargs.pop("columns", ("amount",)),
        status=status,
        evidence=evidence,
        **kwargs,
    )


def test_only_declarative_evidence_claims_it_can_confirm():
    assert all(k.can_confirm for k in DECLARATIVE)
    assert not any(k.can_confirm for k in INFERENTIAL)


@pytest.mark.parametrize("kind", INFERENTIAL)
def test_name_evidence_can_never_produce_a_confirmed_binding(kind: EvidenceKind):
    # The rule of ARCHITECTURE 12.2, enforced by the type rather than by
    # discipline. No quantity of inferential evidence promotes to confirmed.
    with pytest.raises(ValidationError, match="Name similarity never confirms"):
        _binding(
            BindingStatus.CONFIRMED,
            (BindingEvidence(kind=kind, detail="looks right", confidence=1.0),),
        )


def test_a_pile_of_inferential_evidence_still_cannot_confirm():
    with pytest.raises(ValidationError, match="never confirms"):
        _binding(
            BindingStatus.CONFIRMED,
            tuple(
                BindingEvidence(kind=k, detail="looks right", confidence=1.0)
                for k in INFERENTIAL
            ),
        )


@pytest.mark.parametrize("kind", DECLARATIVE)
def test_a_declaration_confirms(kind: EvidenceKind):
    binding = _binding(
        BindingStatus.CONFIRMED,
        (BindingEvidence(kind=kind, detail="declared", source="tenant.json"),),
    )
    assert binding.status is BindingStatus.CONFIRMED
    assert binding.confirming_evidence
    assert binding.status.admissible_in_semantic_path


def test_an_inferred_binding_is_usable_but_not_in_the_semantic_path():
    binding = _binding(
        BindingStatus.INFERRED,
        (BindingEvidence(kind=EvidenceKind.ALIAS, detail="header is an alias"),),
    )
    assert binding.is_available
    assert not binding.status.admissible_in_semantic_path


def test_an_ambiguous_binding_binds_nothing_and_names_its_rivals():
    binding = ConceptBinding(
        dataset_id="d",
        concept=BusinessConcept.CUSTOMER_SEGMENT,
        status=BindingStatus.AMBIGUOUS,
        alternatives=("segment", "account_segment"),
    )
    assert binding.columns == ()
    assert not binding.is_available
    assert len(binding.alternatives) == 2

    with pytest.raises(ValidationError, match="must list the rival candidates"):
        ConceptBinding(
            dataset_id="d",
            concept=BusinessConcept.CUSTOMER_SEGMENT,
            status=BindingStatus.AMBIGUOUS,
            alternatives=("segment",),
        )


def test_an_ambiguous_concept_may_not_also_be_bound():
    with pytest.raises(ValidationError, match="its candidates are"):
        ConceptBinding(
            dataset_id="d",
            concept=BusinessConcept.CUSTOMER_SEGMENT,
            columns=("segment",),
            status=BindingStatus.AMBIGUOUS,
            alternatives=("segment", "account_segment"),
        )


def test_an_unavailable_concept_is_explicit_and_binds_nothing():
    binding = ConceptBinding(
        dataset_id="d",
        concept=BusinessConcept.CUSTOMER_SEGMENT,
        status=BindingStatus.UNAVAILABLE,
        note="no candidate column",
    )
    assert binding.columns == ()
    assert not binding.is_available
    assert binding.confidence == 0.0

    with pytest.raises(ValidationError, match="must bind no column"):
        ConceptBinding(
            dataset_id="d",
            concept=BusinessConcept.CUSTOMER_SEGMENT,
            columns=("segment",),
            status=BindingStatus.UNAVAILABLE,
        )


def test_a_bound_status_must_actually_name_a_column():
    with pytest.raises(ValidationError, match="must name a column"):
        _binding(BindingStatus.INFERRED, (), columns=())


def test_evidence_records_what_was_weighed_including_the_contrary():
    against = BindingEvidence(
        kind=EvidenceKind.AGREEMENT_TEST,
        detail="reconstruction disagrees with close_date_qtr on 15 rows",
        source="close_date_matches_close_date_qtr",
        confidence=0.73,
        supports=False,
    )
    binding = _binding(
        BindingStatus.INFERRED,
        (BindingEvidence(kind=EvidenceKind.ALIAS, detail="alias", confidence=0.6), against),
    )
    # Contrary evidence is kept, not dropped, and does not raise confidence.
    assert against in binding.evidence
    assert binding.confidence == 0.6
    assert not against.can_confirm


def test_a_one_cardinality_concept_refuses_two_columns():
    with pytest.raises(ValidationError, match="binds one column"):
        _binding(
            BindingStatus.INFERRED,
            (BindingEvidence(kind=EvidenceKind.ALIAS, detail="a"),),
            columns=("amount", "arr"),
        )


def test_narrative_text_may_bind_many_columns():
    binding = ConceptBinding(
        dataset_id="d",
        concept=BusinessConcept.NARRATIVE_TEXT,
        columns=("ManagerNotes", "NextStep", "SENotes"),
        status=BindingStatus.INFERRED,
        evidence=(BindingEvidence(kind=EvidenceKind.VALUE_PATTERN, detail="prose"),),
    )
    assert len(binding.columns) == 3


def test_an_agreement_result_becomes_non_confirming_evidence():
    passing = AgreementResult(
        test_id="t",
        assertion="a",
        kind=AgreementKind.COLUMN_CONSISTENT_WITH_CONCEPT,
        concept=BusinessConcept.EXPECTED_CLOSE_DATE,
        checked_rows=100,
        disagreeing_rows=0,
    )
    evidence = passing.as_evidence()
    assert evidence.supports
    assert not evidence.can_confirm, "a passing agreement test is evidence, not proof"


def test_a_concept_may_be_bound_only_once_per_dataset():
    duplicate = _binding(
        BindingStatus.INFERRED,
        (BindingEvidence(kind=EvidenceKind.ALIAS, detail="a"),),
    )
    with pytest.raises(ValidationError, match="bound more than once"):
        ConceptBindings(dataset_id="d", bindings=(duplicate, duplicate))


def test_an_operation_is_unsatisfied_when_a_concept_is_only_inferred():
    # A semantic metric needs a confirmed binding (ARCHITECTURE 12.6), so an
    # inferred one reads as missing here and the operation is hidden.
    bindings = ConceptBindings(
        dataset_id="d",
        bindings=(
            ConceptBinding(
                dataset_id="d",
                concept=BusinessConcept.OPPORTUNITY_ID,
                columns=("opp_id",),
                status=BindingStatus.CONFIRMED,
                evidence=(BindingEvidence(kind=EvidenceKind.TENANT_CONFIG, detail="d"),),
            ),
            ConceptBinding(
                dataset_id="d",
                concept=BusinessConcept.SNAPSHOT_DATE,
                columns=("as_of",),
                status=BindingStatus.CONFIRMED,
                evidence=(BindingEvidence(kind=EvidenceKind.TENANT_CONFIG, detail="d"),),
            ),
            ConceptBinding(
                dataset_id="d",
                concept=BusinessConcept.OPPORTUNITY_STATUS,
                columns=("status",),
                status=BindingStatus.INFERRED,
                evidence=(BindingEvidence(kind=EvidenceKind.VALUE_PATTERN, detail="stage"),),
            ),
        ),
    )
    check = bindings.check_operation(AnalyticalOperation.WIN_RATE)
    assert not check.satisfied
    assert BusinessConcept.OPPORTUNITY_STATUS in check.missing
    assert "opportunity_status concept unavailable" in check.reason


def test_monetary_columns_come_from_the_concept_not_the_shape():
    bindings = ConceptBindings(
        dataset_id="d",
        bindings=(
            ConceptBinding(
                dataset_id="d",
                concept=BusinessConcept.AMOUNT,
                columns=("ARR",),
                status=BindingStatus.CONFIRMED,
                evidence=(BindingEvidence(kind=EvidenceKind.TENANT_CONFIG, detail="d"),),
            ),
        ),
    )
    assert bindings.monetary_columns() == {"ARR"}


def test_a_tenant_declaration_respects_cardinality_and_records_absence():
    tenant = TenantProfile(
        tenant_id="acme",
        concept_columns={BusinessConcept.AMOUNT: ("ARR",)},
        absent_concepts=(BusinessConcept.CUSTOMER_SEGMENT,),
        source="confirmed by the RevOps lead",
    )
    assert tenant.declared(BusinessConcept.AMOUNT) == ("ARR",)
    assert tenant.declares_absent(BusinessConcept.CUSTOMER_SEGMENT)

    with pytest.raises(ValidationError, match="binds one column"):
        TenantProfile(
            tenant_id="acme",
            concept_columns={BusinessConcept.AMOUNT: ("ARR", "new_amount")},
        )
    with pytest.raises(ValidationError, match="both present and absent"):
        TenantProfile(
            tenant_id="acme",
            concept_columns={BusinessConcept.AMOUNT: ("ARR",)},
            absent_concepts=(BusinessConcept.AMOUNT,),
        )


def test_a_report_knows_when_a_concept_is_contradicted():
    failing = AgreementResult(
        test_id="t",
        assertion="a",
        kind=AgreementKind.COLUMN_CONSISTENT_WITH_CONCEPT,
        concept=BusinessConcept.EXPECTED_CLOSE_DATE,
        checked_rows=100,
        disagreeing_rows=15,
    )
    report = AgreementReport(dataset_id="d", results=(failing,))
    assert report.concept_is_contradicted(BusinessConcept.EXPECTED_CLOSE_DATE)
    assert not report.concept_is_contradicted(BusinessConcept.AMOUNT)
