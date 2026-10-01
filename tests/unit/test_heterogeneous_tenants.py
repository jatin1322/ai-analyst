"""Different tenants, different physical schemas (ARCHITECTURE 12.10).

One tenant calls the amount `new_amount`, another calls it `ARR`, a third
splits it across segment columns. None of that may change the vocabulary a
metric is written in, and none of it may be resolved by resemblance.
"""

from __future__ import annotations

from ai_analyst.config import Settings
from ai_analyst.contracts.binding import BindingStatus, EvidenceKind
from ai_analyst.contracts.concepts import BusinessConcept
from ai_analyst.contracts.schema import CanonicalColumn
from ai_analyst.contracts.tenant import TenantProfile
from ai_analyst.data.binding import build_bindings
from ai_analyst.data.ingest import ingest
from ai_analyst.data.profiler import profile_dataset

HEADERS = (
    "as_of,opp_id,ARR,terminal_fate,final_outcome,segment,Stage\n"
    "2025-01-01,O-1,1000.00,W,won,enterprise,Closed Won\n"
    "2025-02-01,O-1,1000.00,W,won,enterprise,Closed Won\n"
    "2025-01-01,O-2,2000.00,L,lost,mid market,Closed Lost\n"
    "2025-02-01,O-2,2500.00,L,lost,mid market,Closed Lost\n"
)


def _ingest(tmp_path, settings: Settings, tenant: TenantProfile | None = None):
    source = tmp_path / "tenant_b.csv"
    source.write_text(HEADERS, encoding="utf-8")
    result = ingest(source, "tenant_b", settings=settings)
    profile = profile_dataset(
        "tenant_b", result.schema, settings=settings, registry=result.registry
    )
    bindings = build_bindings(result.schema, result.registry, profile, tenant=tenant)
    return result, profile, bindings


def test_a_tenant_with_a_different_amount_column_still_ingests(
    tmp_path, settings: Settings
):
    result, _, bindings = _ingest(tmp_path, settings)
    assert result.row_count == 4
    # `ARR` is a known alias, so it conforms to the canonical arr column. The
    # header differing from every other tenant's changes nothing downstream.
    arr = result.schema.mapping.by_canonical()[CanonicalColumn.ARR]
    assert arr.source_column == "ARR"
    assert bindings.get(BusinessConcept.OPPORTUNITY_ID).columns == ("opp_id",)


def test_without_a_declaration_the_amount_concept_is_not_confirmed(
    tmp_path, settings: Settings
):
    # `ARR` is a canonical alias, which is name evidence and can never confirm.
    _, _, bindings = _ingest(tmp_path, settings)
    amount = bindings.get(BusinessConcept.AMOUNT)
    assert amount.status is not BindingStatus.CONFIRMED
    assert not amount.confirming_evidence


def test_a_tenant_declaration_confirms_the_amount_concept(tmp_path, settings: Settings):
    tenant = TenantProfile(
        tenant_id="tenant_b",
        concept_columns={BusinessConcept.AMOUNT: ("ARR",)},
        source="confirmed by the RevOps lead, 2026-09-22",
    )
    _, _, bindings = _ingest(tmp_path, settings, tenant)
    amount = bindings.get(BusinessConcept.AMOUNT)
    assert amount.status is BindingStatus.CONFIRMED
    # Declared as the tenant's own header; resolved to the conformed column.
    assert amount.columns == ("arr",)
    assert amount.confirming_evidence[0].kind is EvidenceKind.TENANT_CONFIG
    assert "RevOps lead" in amount.confirming_evidence[0].source
    # And it is money because the concept says so, not because of its values.
    assert amount.columns[0] in bindings.monetary_columns()


def test_two_rival_candidates_make_a_concept_ambiguous(tmp_path, settings: Settings):
    # `terminal_fate` and `final_outcome` both look like the terminal outcome.
    # Picking the higher-scoring name is exactly what must not happen.
    _, _, bindings = _ingest(tmp_path, settings)
    outcome = bindings.get(BusinessConcept.TERMINAL_OUTCOME)
    assert outcome.status is BindingStatus.AMBIGUOUS
    assert outcome.columns == ()
    assert set(outcome.alternatives) == {"terminal_fate", "final_outcome"}


def test_a_declaration_resolves_an_ambiguity(tmp_path, settings: Settings):
    tenant = TenantProfile(
        tenant_id="tenant_b",
        concept_columns={BusinessConcept.TERMINAL_OUTCOME: ("terminal_fate",)},
    )
    _, _, bindings = _ingest(tmp_path, settings, tenant)
    outcome = bindings.get(BusinessConcept.TERMINAL_OUTCOME)
    assert outcome.status is BindingStatus.CONFIRMED
    assert outcome.columns == ("terminal_fate",)


def test_a_tenant_may_declare_a_concept_absent(tmp_path, settings: Settings):
    # Declaring an absence stops the inference layer proposing a wrong column.
    tenant = TenantProfile(
        tenant_id="tenant_b",
        absent_concepts=(BusinessConcept.CUSTOMER_SEGMENT,),
    )
    _, _, bindings = _ingest(tmp_path, settings, tenant)
    segment = bindings.get(BusinessConcept.CUSTOMER_SEGMENT)
    assert segment.status is BindingStatus.UNAVAILABLE
    assert segment.columns == ()
    assert "declares this concept absent" in segment.note


def test_a_concept_no_column_satisfies_is_explicitly_unavailable(
    tmp_path, settings: Settings
):
    _, _, bindings = _ingest(tmp_path, settings)
    # This tenant exports no forecast category at all.
    forecast = bindings.get(BusinessConcept.FORECAST_CATEGORY)
    assert forecast.status is BindingStatus.UNAVAILABLE
    assert forecast.columns == ()


def test_the_same_concept_vocabulary_serves_both_tenants(tmp_path, settings: Settings):
    _, _, bindings = _ingest(tmp_path, settings)
    # Whatever the headers, every concept is resolved to some status.
    assert {b.concept for b in bindings.bindings} == set(BusinessConcept)
