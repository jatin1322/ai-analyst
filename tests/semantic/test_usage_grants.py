"""Purpose-scoped usage grants (ARCHITECTURE 13.2).

Three questions that one flag used to answer together:

* concept confirmation: does this column mean concept X? (a declaration)
* physical classification: what is it, and when is it knowable? (a registry)
* usability grant: may it be read *for this purpose, under this stance*?

A valid tenant declaration issues a grant scoped to the concept's own
purposes, with the concept's timing. Everything else stays fail-closed.
"""

from __future__ import annotations

from decimal import Decimal

from ai_analyst.contracts.binding import BindingStatus, ColumnPurpose, EvidenceKind
from ai_analyst.contracts.columns import Availability
from ai_analyst.contracts.concepts import BusinessConcept as Concept
from ai_analyst.contracts.plan import (
    AnalysisPattern,
    AnalysisSpec,
    AnalysisStance,
    Filter,
    FilterOp,
    Period,
    PeriodKind,
    SnapshotSelection,
)
from ai_analyst.contracts.rejection import PlanRejection, RejectionCode
from ai_analyst.contracts.result import SnapshotRule, TrustFactorKind
from ai_analyst.contracts.tenant import TenantProfile
from ai_analyst.semantic.resolver import ConceptResolver
from tests.semantic.conftest import CUSTOM_CSV, TINY_TENANT, build_engine

Q1 = Period(kind=PeriodKind.FISCAL_QUARTER, label="FY2025-Q1")


def spec(spec_id: str, **kwargs) -> AnalysisSpec:
    kwargs.setdefault("pattern", AnalysisPattern.POINT_IN_TIME)
    kwargs.setdefault("period", Q1)
    kwargs.setdefault("snapshot", SnapshotSelection(rule=SnapshotRule.PERIOD_OPEN))
    return AnalysisSpec(id=spec_id, **kwargs)


def _tenant(**overrides) -> TenantProfile:
    columns = {**TINY_TENANT.concept_columns, **overrides}
    return TenantProfile(
        tenant_id="custom",
        source="owner@tenant, 2026-09-23",
        concept_columns=columns,
        fiscal_year_start_month=1,
    )


def _resolver(engine, stance=AnalysisStance.PROSPECTIVE) -> ConceptResolver:
    return ConceptResolver(
        dataset_id=engine.dataset_id,
        registry=engine.registry,
        bindings=engine.bindings,
        stance=stance,
    )


def test_a_valid_declaration_issues_a_grant_with_the_concepts_purposes(custom_declared):
    grant = custom_declared.bindings.grant_for("enterprise_amount", Concept.AMOUNT)
    assert grant is not None
    assert grant.purposes == frozenset({ColumnPurpose.MEASURE, ColumnPurpose.FILTER})
    assert grant.availability is Availability.AS_OF_FACT


def test_the_grant_retains_its_provenance(custom_declared):
    grant = custom_declared.bindings.grant_for("enterprise_amount", Concept.AMOUNT)
    assert grant.source.kind is EvidenceKind.TENANT_CONFIG
    assert grant.source.source == "tests fixture"
    assert "tenant declaration" in grant.disclosure


def test_the_column_itself_stays_unclassified(custom_declared):
    column = custom_declared.registry.get("enterprise_amount")
    assert not column.classification.classified
    assert column.is_quarantined


def test_the_grant_licenses_a_filter_on_the_concept(custom_declared):
    # Q1 opening pipeline at 2025-01-01, enterprise_amount >= 25000:
    #   OPP-001 50000, OPP-007 30000 qualify; OPP-005 20000 does not.
    #   50000 + 30000 = 80000
    result = custom_declared.one(
        spec("f", metrics=["opening_pipeline"],
             filters=[Filter(column="amount", op=FilterOp.GTE, values=[25000])])
    )
    assert result.cell(0, "opening_pipeline") == Decimal("80000.00")


def test_the_grant_does_not_license_a_dimension(custom_declared):
    outcome = custom_declared.gate(
        spec("d", metrics=["opening_pipeline"], dimensions=["amount"])
    )
    assert outcome.validation.has(RejectionCode.GRANT_PURPOSE_NOT_PERMITTED)


def test_the_grant_does_not_license_a_feature(custom_declared):
    outcome = custom_declared.gate(
        spec("r", pattern=AnalysisPattern.RANKED_LIST, metrics=["deal_count"],
             features=["amount"], limit=5,
             snapshot=SnapshotSelection(rule=SnapshotRule.LATEST))
    )
    assert outcome.validation.has(RejectionCode.GRANT_PURPOSE_NOT_PERMITTED)


def test_the_raw_header_never_sees_the_grant(custom_declared):
    outcome = custom_declared.gate(
        spec("f", metrics=["opening_pipeline"],
             filters=[Filter(column="enterprise_amount", op=FilterOp.GTE, values=[1])])
    )
    assert outcome.validation.has(RejectionCode.COLUMN_UNCLASSIFIED)


def test_using_a_grant_is_disclosed_as_a_trust_factor(custom_declared):
    result = custom_declared.one(spec("m", metrics=["opening_pipeline"]))
    kinds = {f.kind for f in result.trust.factors}
    assert TrustFactorKind.USAGE_GRANT in kinds
    assert any("enterprise_amount" in d for d in result.trust.disclosures)


def test_an_undeclared_dataset_issues_no_grants(undeclared, custom):
    assert undeclared.bindings.grants == ()
    assert custom.bindings.grant_for("enterprise_amount", Concept.AMOUNT) is None


def test_an_inferred_binding_never_issues_a_grant(custom):
    # The custom dataset has no tenant declaration for enterprise_amount at all.
    binding = custom.bindings.get(Concept.AMOUNT)
    assert binding.columns != ("enterprise_amount",)
    assert all(g.source.kind is EvidenceKind.TENANT_CONFIG for g in custom.bindings.grants)


def test_a_declaration_whose_type_cannot_hold_the_concept_is_rejected(tmp_path):
    engine = build_engine(
        CUSTOM_CSV, "badtype", tmp_path, tenant=_tenant(**{Concept.AMOUNT: ("owner",)})
    )
    binding = engine.bindings.get(Concept.AMOUNT)
    assert binding.status is BindingStatus.UNAVAILABLE
    assert "declaration_conflict" in binding.caveats
    assert "cannot hold amount" in binding.note
    outcome = engine.gate(spec("m", metrics=["opening_pipeline"]))
    assert outcome.validation.has(RejectionCode.DECLARATION_CONFLICT)


def test_a_declaration_cannot_overrule_a_known_contaminated_classification(
    tmp_path, production_csv
):
    from ai_analyst.contracts.opportunity_snapshot_v1 import OPPORTUNITY_SNAPSHOT_V1

    tenant = TenantProfile(
        tenant_id="p", source="x", concept_columns={Concept.AMOUNT: ("terminal_amount",)}
    )
    engine = build_engine(
        production_csv, "conflict", tmp_path, tenant=tenant,
        column_registry=OPPORTUNITY_SNAPSHOT_V1,
    )
    binding = engine.bindings.get(Concept.AMOUNT)
    assert binding.status is BindingStatus.UNAVAILABLE
    assert "future_contaminated" in binding.note
    assert engine.bindings.grant_for("terminal_amount", Concept.AMOUNT) is None


def test_grant_availability_follows_the_concept_not_the_column(tmp_path):
    engine = build_engine(
        CUSTOM_CSV, "retro", tmp_path,
        tenant=_tenant(**{Concept.TERMINAL_AMOUNT: ("enterprise_amount",)}),
    )
    grant = engine.bindings.grant_for("enterprise_amount", Concept.TERMINAL_AMOUNT)
    assert grant.availability is Availability.FUTURE_CONTAMINATED

    prospective = _resolver(engine).try_resolve(Concept.TERMINAL_AMOUNT)
    assert isinstance(prospective, PlanRejection)
    assert prospective.code is RejectionCode.RETROSPECTIVE_CONCEPT_IN_PROSPECTIVE

    retro = _resolver(engine, AnalysisStance.RETROSPECTIVE).try_resolve(Concept.TERMINAL_AMOUNT)
    assert not isinstance(retro, PlanRejection)
    assert retro.grant is grant
