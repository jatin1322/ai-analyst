"""Tenant column classifications (ARCHITECTURE 13.2, CLAUDE.md 2.3 and 22).

`TenantProfile.column_classifications` lets a tenant say what an unknown column
*is* and *when it is knowable*, reusing `ColumnClassification`. A confirmed,
valid classification issues a *generic* usage grant, persisted in the ledger.
It never confirms a concept binding, never overrules a registry, and never makes
a column a measure concept; an inferred one issues nothing.

`enterprise_amount` in `custom_column.csv` is `deal_amount / 2` and is not
placed by any registry. Q1 cohort at 2025-01-01 (open, any close date):

    opp      enterprise_amount   stage 01-01 -> 03-31
    OPP-001  50000               Negotiation -> Negotiation
    OPP-002  25000               Proposal    -> Closed Won
    OPP-003  37500               Discovery   -> Discovery
    OPP-004  100000              Qualification -> Qualification
    OPP-005  20000               Discovery   -> Discovery
    OPP-007  30000               Negotiation -> Closed Lost
"""

from __future__ import annotations

import csv
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from ai_analyst.contracts.binding import BindingStatus, ColumnPurpose, GrantKind
from ai_analyst.contracts.columns import (
    Availability,
    ColumnCategory,
    ColumnClassification,
    Disposition,
    MonetaryStatus,
    QuarantineCode,
    QuarantineReason,
)
from ai_analyst.contracts.concepts import BusinessConcept as Concept
from ai_analyst.contracts.investigation import Grouping, Operation, Variable
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
from ai_analyst.contracts.rejection import RejectionCode
from ai_analyst.contracts.result import SnapshotRule, TrustFactorKind
from ai_analyst.contracts.tenant import ColumnDeclaration, TenantProfile
from ai_analyst.data import grants as grants_module
from ai_analyst.data.dataset import load_dataset
from ai_analyst.data.grants import ledger_path, load_grants
from ai_analyst.data.understanding import understand
from tests.semantic.conftest import CUSTOM_CSV, TINY_TENANT, build_engine
from tests.semantic.test_investigation import F, Op, derived, gate, plan, rows, run

Q1 = Period(kind=PeriodKind.FISCAL_QUARTER, label="FY2025-Q1")
PROSP, RETRO = AnalysisStance.PROSPECTIVE, AnalysisStance.RETROSPECTIVE
GTE_25K = Filter(column="enterprise_amount", op=FilterOp.GTE, values=[25000])


def classification(
    name: str = "enterprise_amount",
    *,
    availability: Availability = Availability.AS_OF_FACT,
    monetary: MonetaryStatus | None = MonetaryStatus.MONETARY,
    category: ColumnCategory = ColumnCategory.DEAL_FEATURE,
    disposition: Disposition | None = None,
    quarantine: QuarantineReason | None = None,
) -> ColumnClassification:
    if disposition is None:
        disposition = (
            Disposition.DIRECT
            if availability is Availability.AS_OF_FACT
            else Disposition.USE_WITH_PROOF
        )
    return ColumnClassification(
        name=name, category=category, availability=availability,
        disposition=disposition, monetary=monetary, quarantine=quarantine,
    )


def declared(
    *classifications: ColumnClassification, status: BindingStatus = BindingStatus.CONFIRMED
) -> TenantProfile:
    return TenantProfile(
        tenant_id="acme",
        source="owner@acme",
        concept_columns=TINY_TENANT.concept_columns,
        column_classifications=[
            ColumnDeclaration(column=c.name, classification=c, status=status,
                              source="owner@acme, 2026-09-23")
            for c in classifications
        ],
        fiscal_year_start_month=1,
    )


def spec(spec_id: str = "a", **kwargs) -> AnalysisSpec:
    kwargs.setdefault("pattern", AnalysisPattern.POINT_IN_TIME)
    kwargs.setdefault("period", Q1)
    kwargs.setdefault("metrics", ["opening_pipeline"])
    kwargs.setdefault("snapshot", SnapshotSelection(rule=SnapshotRule.PERIOD_OPEN))
    return AnalysisSpec(id=spec_id, **kwargs)


def sum_plan(**kwargs):
    """Sum of enterprise_amount per opportunity, grouped by 'stage changed'."""
    return plan(
        variables=[Variable(id="e", column="enterprise_amount"),
                   derived("moved", F.CHANGED, Concept.STAGE)],
        grouping=[Grouping(variable="moved")],
        operation=Operation(kind=Op.SUM, measure="e"),
        **kwargs,
    )


@pytest.fixture
def money(tmp_path: Path):
    return build_engine(CUSTOM_CSV, "cls", tmp_path, tenant=declared(classification()))


@pytest.fixture
def tier_csv(tmp_path: Path) -> Path:
    """The custom fixture plus a discovered text column, `partner_tier`."""
    reader = list(csv.DictReader(CUSTOM_CSV.open(encoding="utf-8")))
    for row in reader:
        gold = row["opportunity_id"] in ("OPP-001", "OPP-004")
        row["partner_tier"] = "Gold" if gold else "Silver"
    path = tmp_path / "tier.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(reader[0]))
        writer.writeheader()
        writer.writerows(reader)
    return path


# ============================================================================
# 1. Persistence, and 2. reload
# ============================================================================


def test_a_tenant_profile_with_classifications_round_trips():
    tenant = declared(classification())
    again = TenantProfile.model_validate_json(tenant.model_dump_json())
    assert again == tenant
    assert again.declarations_fingerprint == tenant.declarations_fingerprint


def test_a_classification_change_changes_the_declarations_fingerprint():
    one = declared(classification())
    two = declared(classification(availability=Availability.UNKNOWN))
    assert one.declarations_fingerprint != two.declarations_fingerprint


def test_a_classification_is_persisted_as_a_generic_grant(money):
    ledger = load_grants("cls", tenant=declared(classification()), registry=money.registry,
                         settings=money.settings)
    (grant,) = [g for g in ledger.grants if g.kind is GrantKind.GENERIC]
    assert grant.column == "enterprise_amount"
    assert grant.concept is None
    assert grant.monetary is True
    assert grant.availability is Availability.AS_OF_FACT
    assert grant.purposes == frozenset(
        {ColumnPurpose.DIMENSION, ColumnPurpose.FILTER, ColumnPurpose.FEATURE,
         ColumnPurpose.MEASURE}
    )
    assert grant.source.source == "owner@acme, 2026-09-23"


def test_the_runtime_reloads_the_generic_grant_from_disk(money):
    tenant = declared(classification())
    bindings = understand(load_dataset("cls", money.settings), tenant=tenant,
                          settings=money.settings).bindings
    assert bindings.generic_grant_for("enterprise_amount") == money.bindings.generic_grant_for(
        "enterprise_amount"
    )


def test_a_duplicate_classification_is_refused():
    with pytest.raises(ValidationError):
        declared(classification(), classification())


# ============================================================================
# 3. Classification -> binding behaviour
# ============================================================================


def test_a_classification_never_confirms_a_concept_binding(money):
    # Declared money, named like an amount, confirmed: still not the amount concept.
    binding = money.bindings.get(Concept.AMOUNT)
    assert "enterprise_amount" not in binding.columns
    assert money.bindings.grant_for("enterprise_amount", Concept.AMOUNT) is None
    # Opening pipeline is still measured in deal_amount: 100000 + 40000 + 60000.
    assert money.one(spec()).cell(0, "opening_pipeline") == Decimal("200000.00")


def test_a_classification_cannot_overrule_a_registry_classification(tmp_path):
    engine = build_engine(CUSTOM_CSV, "overrule", tmp_path,
                          tenant=declared(classification("forecast_category", monetary=None)))
    assert engine.bindings.generic_grant_for("forecast_category") is None
    ledger = load_grants("overrule", tenant=declared(classification("forecast_category",
                                                                     monetary=None)),
                         registry=engine.registry, settings=engine.settings)
    (rejected,) = ledger.rejected
    assert "already classified by the registry" in rejected.reason


# ============================================================================
# 4. An inferred classification does not become a confirmed binding or grant
# ============================================================================


def test_an_inferred_classification_issues_nothing(tmp_path):
    tenant = declared(classification(), status=BindingStatus.INFERRED)
    engine = build_engine(CUSTOM_CSV, "inferred", tmp_path, tenant=tenant)
    assert engine.bindings.generic_grant_for("enterprise_amount") is None
    ledger = load_grants("inferred", tenant=tenant, registry=engine.registry,
                         settings=engine.settings)
    (rejected,) = ledger.rejected
    assert rejected.kind is GrantKind.GENERIC
    assert "proposal" in rejected.reason
    outcome = engine.gate(spec(filters=[GTE_25K]))
    assert outcome.validation.has(RejectionCode.COLUMN_UNCLASSIFIED)


@pytest.mark.parametrize(
    "status", [BindingStatus.AMBIGUOUS, BindingStatus.UNAVAILABLE]
)
def test_a_declaration_is_only_ever_confirmed_or_inferred(status):
    with pytest.raises(ValidationError):
        declared(classification(), status=status)


def test_a_declaration_cannot_declare_unclassified():
    unclassified = classification().model_copy(update={"classified": False})
    with pytest.raises(ValidationError):
        declared(unclassified)


# ============================================================================
# 5. Explicit declaration: what a confirmed classification may and may not do
# ============================================================================


@pytest.mark.parametrize(
    ("bad", "reason"),
    [
        (
            classification(
                disposition=Disposition.QUARANTINE,
                quarantine=QuarantineReason(
                    code=QuarantineCode.UNDOCUMENTED_LINEAGE, detail="d", resolution="r"
                ),
            ),
            "quarantines",
        ),
        (classification(category=ColumnCategory.TEXT, monetary=None), "text column"),
    ],
)
def test_a_confirmed_but_unusable_classification_is_rejected_with_a_reason(
    tmp_path, bad, reason
):
    tenant = declared(bad)
    engine = build_engine(CUSTOM_CSV, "bad", tmp_path, tenant=tenant)
    assert engine.bindings.generic_grant_for("enterprise_amount") is None
    ledger = load_grants("bad", tenant=tenant, registry=engine.registry,
                         settings=engine.settings)
    assert reason in ledger.rejected[0].reason


def test_a_text_column_cannot_be_declared_money(tmp_path, tier_csv):
    tenant = declared(classification("partner_tier"))
    engine = build_engine(tier_csv, "tier_money", tmp_path, tenant=tenant)
    assert engine.bindings.generic_grant_for("partner_tier") is None
    ledger = load_grants("tier_money", tenant=tenant, registry=engine.registry,
                         settings=engine.settings)
    assert "cannot be declared money" in ledger.rejected[0].reason


def test_a_classification_of_a_missing_column_is_recorded_not_granted(tmp_path):
    tenant = declared(classification("no_such_column"))
    engine = build_engine(CUSTOM_CSV, "missing", tmp_path, tenant=tenant)
    ledger = load_grants("missing", tenant=tenant, registry=engine.registry,
                         settings=engine.settings)
    assert ledger.grants and all(g.kind is GrantKind.CONCEPT for g in ledger.grants)
    assert ledger.rejected[0].reason == "no such column in this dataset"


# ============================================================================
# 6. Temporal enforcement
# ============================================================================


@pytest.mark.parametrize(
    "availability", [Availability.FUTURE_CONTAMINATED, Availability.UNKNOWN]
)
def test_an_unsafe_classification_is_unreadable_prospectively(tmp_path, availability):
    engine = build_engine(CUSTOM_CSV, "unsafe", tmp_path,
                          tenant=declared(classification(availability=availability)))
    outcome = engine.gate(spec(filters=[GTE_25K], stance=PROSP))
    assert outcome.validation.has(RejectionCode.COLUMN_NOT_KNOWABLE_AT_SNAPSHOT)
    assert gate(engine, sum_plan(stance=PROSP)).validation.has(
        RejectionCode.COLUMN_NOT_KNOWABLE_AT_SNAPSHOT
    )


def test_an_unsafe_classification_is_readable_retrospectively_and_disclosed(tmp_path):
    engine = build_engine(
        CUSTOM_CSV, "retro", tmp_path,
        tenant=declared(classification(availability=Availability.FUTURE_CONTAMINATED)),
    )
    result = engine.one(spec(filters=[GTE_25K], stance=RETRO))
    kinds = {f.kind for f in result.trust.factors}
    assert {TrustFactorKind.USAGE_GRANT, TrustFactorKind.RETROSPECTIVE_READ} <= kinds


def test_a_backward_derived_classification_is_readable_prospectively(tmp_path):
    engine = build_engine(
        CUSTOM_CSV, "backward", tmp_path,
        tenant=declared(classification(availability=Availability.BACKWARD_DERIVED)),
    )
    assert engine.gate(spec(filters=[GTE_25K], stance=PROSP)).ok


# ============================================================================
# 7. Monetary DECIMAL behaviour
# ============================================================================


def test_a_monetary_classification_sums_as_exact_decimal(money):
    # Grouped by stage changed 01-01 -> 03-31, values at the cohort snapshot:
    #   changed:   OPP-002 25000 + OPP-007 30000                      =  55000
    #   unchanged: OPP-001 50000 + 003 37500 + 004 100000 + 005 20000 = 207500
    result = run(money, sum_plan())
    assert rows(result) == [
        {"moved": "false", "total": Decimal("207500.00"), "units": 4},
        {"moved": "true", "total": Decimal("55000.00"), "units": 2},
    ]
    assert 'CAST("c"."enterprise_amount" AS DECIMAL(18,2))' in result.compiled_sql


def test_a_non_monetary_classification_does_not_cross_the_money_boundary(tmp_path):
    engine = build_engine(
        CUSTOM_CSV, "plain", tmp_path,
        tenant=declared(classification(monetary=MonetaryStatus.NON_MONETARY)),
    )
    result = run(engine, sum_plan())
    assert "DECIMAL(18,2)" not in result.compiled_sql


def test_a_monetary_classification_reports_no_float_statistics(money):
    grant = money.bindings.generic_grant_for("enterprise_amount")
    assert grant.monetary
    from ai_analyst.data.money import declared_monetary_columns

    tenant = declared(classification())
    assert "enterprise_amount" in declared_monetary_columns(tenant, money.dataset.schema)


# ============================================================================
# 8. Generic usage with a grant
# ============================================================================


def test_a_generic_grant_licenses_a_filter(money):
    # Q1 opening pipeline at 01-01 where enterprise_amount >= 25000:
    #   OPP-001 (50000) 100000, OPP-007 (30000) 60000; OPP-005 (20000) excluded.
    result = money.one(spec(filters=[GTE_25K]))
    assert result.cell(0, "opening_pipeline") == Decimal("160000.00")
    assert TrustFactorKind.USAGE_GRANT in {f.kind for f in result.trust.factors}
    assert any("enterprise_amount" in d for d in result.trust.disclosures)


def test_a_generic_grant_on_a_text_column_licenses_a_dimension_not_a_measure(
    tmp_path, tier_csv
):
    tenant = declared(classification("partner_tier", monetary=None))
    engine = build_engine(tier_csv, "tier", tmp_path, tenant=tenant)
    grant = engine.bindings.generic_grant_for("partner_tier")
    assert ColumnPurpose.MEASURE not in grant.purposes

    # Q1 opening pipeline by partner_tier: Gold OPP-001 100000;
    # Silver OPP-005 40000 + OPP-007 60000 = 100000.
    result = engine.one(spec(dimensions=["partner_tier"]))
    assert {result.cell(i, "partner_tier"): result.cell(i, "opening_pipeline")
            for i in range(result.row_count)} == {
        "Gold": Decimal("100000.00"),
        "Silver": Decimal("100000.00"),
    }
    summed = plan(
        variables=[Variable(id="t", column="partner_tier"),
                   derived("moved", F.CHANGED, Concept.STAGE)],
        grouping=[Grouping(variable="moved")],
        operation=Operation(kind=Op.SUM, measure="t"),
    )
    assert gate(engine, summed).validation.has(RejectionCode.GRANT_PURPOSE_NOT_PERMITTED)


# ============================================================================
# 9. Mutation: classification alone cannot bypass semantic authorization
# ============================================================================


def test_mutation_a_classification_without_its_persisted_grant_authorizes_nothing(money):
    """The tenant profile still carries the classification; only the grant is
    removed. The column is unreadable again: the classification itself
    authorizes nothing, the persisted grant does."""
    ledger_path("cls", money.settings).unlink()
    tenant = declared(classification())
    assert tenant.declaration_for("enterprise_amount") is not None
    money.bindings = understand(load_dataset("cls", money.settings), tenant=tenant,
                                settings=money.settings).bindings
    assert money.gate(spec(filters=[GTE_25K])).validation.has(
        RejectionCode.COLUMN_UNCLASSIFIED
    )


def test_mutation_the_inferred_check_is_what_withholds_the_grant(tmp_path, monkeypatch):
    """The control for section 4: disable the issuing check and an inferred
    classification would release the column. The check is load-bearing."""
    tenant = declared(classification(), status=BindingStatus.INFERRED)
    assert build_engine(CUSTOM_CSV, "held", tmp_path / "held",
                        tenant=tenant).bindings.generic_grant_for("enterprise_amount") is None

    monkeypatch.setattr(grants_module, "_classification_rejection", lambda d, r: None)
    leaked = build_engine(CUSTOM_CSV, "leaked", tmp_path / "leaked", tenant=tenant)
    assert leaked.bindings.generic_grant_for("enterprise_amount") is not None


def test_the_amount_concept_never_resolves_through_a_generic_grant(money):
    """The amount concept is bound by its declaration, never by a classification."""
    from ai_analyst.semantic.resolver import ConceptResolver

    resolver = ConceptResolver(dataset_id="cls", registry=money.registry,
                               bindings=money.bindings, stance=PROSP)
    resolution = resolver.resolve(Concept.AMOUNT, field_name="metrics")
    assert "enterprise_amount" not in resolution.measure_sql()
