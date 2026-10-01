"""The persisted usage-grant ledger (ARCHITECTURE 13.2, CLAUDE.md 2.3).

Grants are issued once, at registration, persisted as `grants.json`, and loaded
by the runtime. These tests pin the four properties that make a grant
auditable: it has a stable identity, it is tied to one dataset and one tenant,
loading it re-validates everything it depends on, and the runtime uses exactly
what was persisted rather than rebuilding grants from the tenant profile.

The fixture is `custom_column.csv`, whose `enterprise_amount` column no
registry places, declared by the tenant as its amount concept.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from ai_analyst.config import Settings
from ai_analyst.contracts.binding import (
    ColumnPurpose,
    GrantKind,
    UsageGrant,
    grant_identity,
)
from ai_analyst.contracts.columns import (
    Availability,
    ColumnCategory,
    ColumnClassification,
    Disposition,
)
from ai_analyst.contracts.concepts import BusinessConcept as Concept
from ai_analyst.contracts.dataset import DatasetRegistry
from ai_analyst.contracts.grants import GrantLedger, GrantLedgerError, GrantLedgerErrorCode
from ai_analyst.contracts.plan import (
    AnalysisPattern,
    AnalysisSpec,
    Period,
    PeriodKind,
    SnapshotSelection,
)
from ai_analyst.contracts.rejection import RejectionCode
from ai_analyst.contracts.result import SnapshotRule
from ai_analyst.contracts.tenant import ColumnDeclaration, TenantProfile
from ai_analyst.data.dataset import load_dataset
from ai_analyst.data.grants import issue_grants, ledger_path, load_grants, save_grants
from ai_analyst.data.understanding import understand
from tests.semantic.conftest import CUSTOM_CSV, TINY_TENANT, build_engine

Q1 = Period(kind=PeriodKind.FISCAL_QUARTER, label="FY2025-Q1")

TENANT = TenantProfile(
    tenant_id="acme",
    source="owner@acme, 2026-09-23",
    concept_columns={**TINY_TENANT.concept_columns, Concept.AMOUNT: ("enterprise_amount",)},
    fiscal_year_start_month=1,
)


def spec(spec_id: str, **kwargs) -> AnalysisSpec:
    kwargs.setdefault("pattern", AnalysisPattern.POINT_IN_TIME)
    kwargs.setdefault("period", Q1)
    kwargs.setdefault("snapshot", SnapshotSelection(rule=SnapshotRule.PERIOD_OPEN))
    return AnalysisSpec(id=spec_id, **kwargs)


@pytest.fixture
def engine(tmp_path: Path):
    return build_engine(CUSTOM_CSV, "acme_ds", tmp_path, tenant=TENANT)


def _amount_grant(ledger: GrantLedger) -> UsageGrant:
    (grant,) = [g for g in ledger.grants if g.column == "enterprise_amount"]
    return grant


def _load(engine, tenant: TenantProfile = TENANT, registry=None, dataset_id=None):
    return load_grants(
        dataset_id or engine.dataset_id,
        tenant=tenant,
        registry=registry or engine.registry,
        settings=engine.settings,
    )


def _rewrite(path: Path, edit) -> None:
    data = json.loads(path.read_text())
    edit(data)
    path.write_text(json.dumps(data, indent=2))


# ============================================================================
# 1. Creation
# ============================================================================


def test_a_declaration_creates_a_typed_scoped_grant_bound_to_dataset_and_tenant(engine):
    ledger = _load(engine)
    grant = _amount_grant(ledger)
    assert grant.kind is GrantKind.CONCEPT
    assert grant.concept is Concept.AMOUNT
    assert grant.dataset_id == "acme_ds"
    assert grant.tenant_id == "acme"
    assert grant.purposes == frozenset({ColumnPurpose.MEASURE, ColumnPurpose.FILTER})
    assert grant.availability is Availability.AS_OF_FACT


def test_a_grant_identity_is_derived_from_its_content_and_is_stable(engine):
    grant = _amount_grant(_load(engine))
    assert grant.grant_id == grant_identity(
        "acme_ds", "acme", GrantKind.CONCEPT, "enterprise_amount", Concept.AMOUNT,
        frozenset({ColumnPurpose.MEASURE, ColumnPurpose.FILTER}), Availability.AS_OF_FACT,
    )
    assert grant.grant_id.startswith("g") and len(grant.grant_id) == 17
    # The same content on another dataset is a different grant.
    other = grant.model_copy(update={"dataset_id": "other", "grant_id": ""})
    assert UsageGrant.model_validate(other.model_dump()).grant_id != grant.grant_id


def test_issuing_is_pure_and_deterministic(engine):
    one = issue_grants(engine.dataset_id, engine.registry, TENANT)
    two = issue_grants(engine.dataset_id, engine.registry, TENANT)
    assert one == two
    assert one.model_dump_json() == two.model_dump_json()


# ============================================================================
# 2. Persistence
# ============================================================================


def test_registration_persists_the_ledger_beside_the_dataset(engine):
    path = ledger_path(engine.dataset_id, engine.settings)
    assert path.exists()
    assert path.parent == engine.settings.dataset_dir(engine.dataset_id)
    on_disk = json.loads(path.read_text())
    assert on_disk["schema_version"] == 1
    assert on_disk["dataset_id"] == "acme_ds"
    assert on_disk["tenant_id"] == "acme"
    assert on_disk["declarations_fingerprint"] == TENANT.declarations_fingerprint


def test_saving_is_byte_for_byte_deterministic(engine):
    path = ledger_path(engine.dataset_id, engine.settings)
    first = path.read_bytes()
    save_grants(issue_grants(engine.dataset_id, engine.registry, TENANT), engine.settings)
    assert path.read_bytes() == first


def test_purposes_are_serialized_in_sorted_order(engine):
    on_disk = json.loads(ledger_path(engine.dataset_id, engine.settings).read_text())
    for grant in on_disk["grants"]:
        assert grant["purposes"] == sorted(grant["purposes"])


def test_re_registering_without_a_tenant_discards_the_ledger(engine, tmp_path):
    from ai_analyst.data.dataset import register_dataset

    path = ledger_path(engine.dataset_id, engine.settings)
    assert path.exists()
    register_dataset(CUSTOM_CSV, engine.dataset_id, settings=engine.settings, tenant=None)
    assert not path.exists()


# ============================================================================
# 3. Reload, and 4. round-trip equality
# ============================================================================


def test_the_runtime_uses_exactly_the_persisted_grants(engine):
    assert engine.bindings.grants == _load(engine).grants


def test_the_ledger_round_trips_exactly(engine):
    ledger = _load(engine)
    again = GrantLedger.model_validate_json(ledger.model_dump_json())
    assert again == ledger
    assert again.model_dump_json(indent=2) == ledger.model_dump_json(indent=2)


def test_the_runtime_loads_the_ledger_and_never_reconstructs_it(engine):
    # Remove the amount grant from the persisted ledger, keeping the ledger
    # valid. If the runtime rebuilt grants from the tenant profile it would
    # silently restore it; it must not.
    path = ledger_path(engine.dataset_id, engine.settings)
    ledger = _load(engine)
    trimmed = GrantLedger.canonical(
        dataset_id=ledger.dataset_id,
        tenant_id=ledger.tenant_id,
        declarations_fingerprint=ledger.declarations_fingerprint,
        grants=[g for g in ledger.grants if g.column != "enterprise_amount"],
        rejected=list(ledger.rejected),
    )
    save_grants(trimmed, engine.settings)

    dataset = load_dataset(engine.dataset_id, engine.settings)
    bindings = understand(dataset, tenant=TENANT, settings=engine.settings).bindings
    assert bindings.grant_for("enterprise_amount", Concept.AMOUNT) is None
    assert path.exists()


def test_a_missing_ledger_fails_closed_to_no_grants(engine):
    ledger_path(engine.dataset_id, engine.settings).unlink()
    dataset = load_dataset(engine.dataset_id, engine.settings)
    bindings = understand(dataset, tenant=TENANT, settings=engine.settings).bindings
    assert bindings.grants == ()


# ============================================================================
# 5. Dataset and tenant mismatch
# ============================================================================


def test_a_ledger_copied_to_another_dataset_is_refused(engine):
    source = ledger_path(engine.dataset_id, engine.settings)
    target = ledger_path("elsewhere", engine.settings)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())
    with pytest.raises(GrantLedgerError) as exc:
        _load(engine, dataset_id="elsewhere")
    assert exc.value.code is GrantLedgerErrorCode.DATASET_MISMATCH


def test_a_ledger_for_another_tenant_is_refused(engine):
    intruder = TENANT.model_copy(update={"tenant_id": "globex"})
    with pytest.raises(GrantLedgerError) as exc:
        _load(engine, tenant=intruder)
    assert exc.value.code is GrantLedgerErrorCode.TENANT_MISMATCH


def test_a_ledger_issued_under_other_declarations_is_stale(engine):
    changed = TENANT.model_copy(
        update={"concept_columns": {**TENANT.concept_columns, Concept.AMOUNT: ("deal_amount",)}}
    )
    with pytest.raises(GrantLedgerError) as exc:
        _load(engine, tenant=changed)
    assert exc.value.code is GrantLedgerErrorCode.STALE


def test_the_runtime_refuses_a_mismatched_ledger_rather_than_ignoring_it(engine):
    dataset = load_dataset(engine.dataset_id, engine.settings)
    intruder = TENANT.model_copy(update={"tenant_id": "globex"})
    with pytest.raises(GrantLedgerError):
        understand(dataset, tenant=intruder, settings=engine.settings)


def test_a_grant_for_another_dataset_cannot_sit_in_a_ledger(engine):
    ledger = _load(engine)
    foreign = _amount_grant(ledger).model_copy(update={"dataset_id": "other", "grant_id": ""})
    foreign = UsageGrant.model_validate(foreign.model_dump())
    with pytest.raises(ValueError, match="belongs to"):
        GrantLedger(
            dataset_id=ledger.dataset_id,
            tenant_id=ledger.tenant_id,
            declarations_fingerprint=ledger.declarations_fingerprint,
            grants=(foreign,),
        )


# ============================================================================
# 6. Corrupted ledgers
# ============================================================================


@pytest.mark.parametrize(
    "corrupt",
    [
        pytest.param(lambda p: p.write_text("{not json"), id="not-json"),
        pytest.param(lambda p: p.write_text("{}"), id="empty-object"),
        pytest.param(
            lambda p: _rewrite(p, lambda d: d.update(schema_version=2)), id="future-schema"
        ),
        pytest.param(
            lambda p: _rewrite(p, lambda d: d.update(unexpected="x")), id="unknown-field"
        ),
        pytest.param(
            lambda p: _rewrite(p, lambda d: d.update(grants=list(reversed(d["grants"])))),
            id="non-canonical-order",
        ),
        pytest.param(
            lambda p: _rewrite(p, lambda d: d["grants"][0].update(availability="unknown")),
            id="edited-content-without-id",
        ),
        pytest.param(
            lambda p: _rewrite(p, lambda d: d["grants"][0].update(grant_id="g0000000000000000")),
            id="forged-id",
        ),
    ],
)
def test_a_corrupted_ledger_is_refused(engine, corrupt):
    corrupt(ledger_path(engine.dataset_id, engine.settings))
    with pytest.raises(GrantLedgerError) as exc:
        _load(engine)
    assert exc.value.code is GrantLedgerErrorCode.CORRUPTED


# ============================================================================
# 7. Scope enforcement after reload
# ============================================================================


def _reloaded_engine(engine):
    """The engine rebuilt from disk, so everything below runs on reloaded grants."""
    dataset = load_dataset(engine.dataset_id, engine.settings)
    engine.bindings = understand(dataset, tenant=TENANT, settings=engine.settings).bindings
    return engine


def test_the_reloaded_grant_still_licenses_its_own_purpose(engine):
    # Q1 opening pipeline in enterprise_amount (deal_amount / 2), open with a
    # close date in Q1 at 2025-01-01: OPP-001 50000 + OPP-005 20000 + OPP-007
    # 30000 = 100000.
    result = _reloaded_engine(engine).one(spec("m", metrics=["opening_pipeline"]))
    assert result.cell(0, "opening_pipeline") == Decimal("100000.00")


def test_the_reloaded_grant_does_not_license_another_purpose(engine):
    outcome = _reloaded_engine(engine).gate(
        spec("d", metrics=["opening_pipeline"], dimensions=["amount"])
    )
    assert outcome.validation.has(RejectionCode.GRANT_PURPOSE_NOT_PERMITTED)


def test_the_reloaded_concept_grant_never_releases_the_raw_header(engine):
    from ai_analyst.contracts.plan import Filter, FilterOp

    outcome = _reloaded_engine(engine).gate(
        spec("f", metrics=["opening_pipeline"],
             filters=[Filter(column="enterprise_amount", op=FilterOp.GTE, values=[1])])
    )
    assert outcome.validation.has(RejectionCode.COLUMN_UNCLASSIFIED)


# ============================================================================
# 8. Declaration contradiction after reload
# ============================================================================


def _reclassified(registry: DatasetRegistry, name: str, **classification) -> DatasetRegistry:
    """The registry as if a later export registry had classified `name`."""
    columns = []
    for column in registry.columns:
        if column.name == name:
            column = column.model_copy(
                update={"classification": ColumnClassification(name=name, **classification)}
            )
        columns.append(column)
    return registry.model_copy(update={"columns": columns})


def test_a_grant_contradicted_by_a_later_classification_is_refused_on_load(engine):
    registry = _reclassified(
        engine.registry,
        "enterprise_amount",
        category=ColumnCategory.OUTCOME,
        availability=Availability.FUTURE_CONTAMINATED,
        disposition=Disposition.USE_WITH_PROOF,
    )
    with pytest.raises(GrantLedgerError) as exc:
        _load(engine, registry=registry)
    assert exc.value.code is GrantLedgerErrorCode.CONTRADICTED
    assert "future_contaminated" in str(exc.value)


def test_a_grant_whose_column_disappeared_is_refused_on_load(engine):
    registry = engine.registry.model_copy(
        update={"columns": [c for c in engine.registry.columns if c.name != "enterprise_amount"]}
    )
    with pytest.raises(GrantLedgerError) as exc:
        _load(engine, registry=registry)
    assert exc.value.code is GrantLedgerErrorCode.SCHEMA_DRIFT


def test_a_generic_grant_on_a_column_a_registry_now_classifies_is_contradicted(tmp_path):
    tenant = TenantProfile(
        tenant_id="acme",
        source="owner@acme",
        concept_columns=TINY_TENANT.concept_columns,
        column_classifications=[
            ColumnDeclaration(
                column="enterprise_amount",
                classification=ColumnClassification(
                    name="enterprise_amount",
                    category=ColumnCategory.DEAL_FEATURE,
                    availability=Availability.AS_OF_FACT,
                    disposition=Disposition.DIRECT,
                ),
                source="owner@acme",
            )
        ],
        fiscal_year_start_month=1,
    )
    engine = build_engine(CUSTOM_CSV, "generic_ds", tmp_path, tenant=tenant)
    assert engine.bindings.generic_grant_for("enterprise_amount") is not None
    registry = _reclassified(
        engine.registry,
        "enterprise_amount",
        category=ColumnCategory.DEAL_FEATURE,
        availability=Availability.AS_OF_FACT,
        disposition=Disposition.DIRECT,
    )
    with pytest.raises(GrantLedgerError) as exc:
        _load(engine, tenant=tenant, registry=registry)
    assert exc.value.code is GrantLedgerErrorCode.CONTRADICTED


# ============================================================================
# 9. Mutation: an out-of-scope purpose is rejected even after persistence
# ============================================================================


def test_mutation_a_persisted_grant_widened_to_an_unlicensed_purpose_is_refused(engine):
    """Widen the persisted grant to DIMENSION and forge a matching id.

    The forged id defeats the content hash, so this proves the purpose rule
    itself is re-applied on load: the amount concept never licenses a
    dimension, whatever the file says.
    """
    path = ledger_path(engine.dataset_id, engine.settings)
    grant = _amount_grant(_load(engine))
    widened = grant.purposes | {ColumnPurpose.DIMENSION}
    forged = grant_identity(
        grant.dataset_id, grant.tenant_id, grant.kind, grant.column, grant.concept,
        widened, grant.availability,
    )

    def widen(data):
        for g in data["grants"]:
            if g["grant_id"] == grant.grant_id:
                g["purposes"] = sorted(p.value for p in widened)
                g["grant_id"] = forged
        data["grants"].sort(key=lambda g: g["grant_id"])

    _rewrite(path, widen)
    with pytest.raises(GrantLedgerError) as exc:
        _load(engine)
    assert exc.value.code is GrantLedgerErrorCode.CORRUPTED
    assert "does not license" in str(exc.value)


def test_mutation_without_the_purpose_check_the_widened_grant_would_leak(engine, monkeypatch):
    """Disable the resolver's purpose check and show the gate then admits the read.

    This is the control for the test above: it demonstrates the purpose check is
    the load-bearing guard for a reloaded grant, not an incidental rejection.
    """
    reloaded = _reloaded_engine(engine)
    guarded = reloaded.gate(spec("d", metrics=["opening_pipeline"], dimensions=["amount"]))
    assert guarded.validation.has(RejectionCode.GRANT_PURPOSE_NOT_PERMITTED)

    monkeypatch.setattr(UsageGrant, "permits", lambda self, purpose: True)
    mutated = reloaded.gate(spec("d", metrics=["opening_pipeline"], dimensions=["amount"]))
    assert not mutated.validation.has(RejectionCode.GRANT_PURPOSE_NOT_PERMITTED)


def test_settings_are_isolated_per_test(tmp_path):
    # Guard against a ledger leaking between tests through a shared data root.
    settings = Settings(data_root=tmp_path / "data")
    assert not ledger_path("acme_ds", settings).exists()
