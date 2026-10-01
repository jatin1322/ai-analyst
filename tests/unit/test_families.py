"""Column families and generic classification proposals (WP4).

Names only propose. These tests pin that inference groups columns by structure,
that an unconfirmed proposal never makes a column readable, and that a confirmed
family declaration goes through the ordinary per-column grant rules.
"""

from __future__ import annotations

import pytest

from ai_analyst.config import Settings
from ai_analyst.contracts.binding import BindingStatus, GrantKind
from ai_analyst.contracts.columns import (
    Availability,
    ColumnCategory,
    ColumnClassification,
    Disposition,
)
from ai_analyst.contracts.tenant import FamilyDeclaration, TenantProfile
from ai_analyst.data.dataset import register_dataset
from ai_analyst.data.families import infer_families, onboarding_report
from ai_analyst.data.grants import issue_grants
from ai_analyst.synthetic.feature_store import (
    ALL_COLUMNS,
    GeneratorConfig,
    generate,
    write_csv,
)

EMAIL = "email_count_(?:7d|14d|30d|60d|90d|180d|lifetime)"


@pytest.fixture(scope="module")
def proposals():
    return {p.name: p for p in infer_families(list(ALL_COLUMNS))}


def _template(name: str = "email_count") -> ColumnClassification:
    return ColumnClassification(
        name=name,
        category=ColumnCategory.HISTORICAL_FEATURE,
        availability=Availability.BACKWARD_DERIVED,
        disposition=Disposition.USE_WITH_PROOF,
        requires_confirmation=False,
    )


def _tenant(status: BindingStatus = BindingStatus.CONFIRMED, pattern: str = EMAIL):
    return TenantProfile(
        tenant_id="acme",
        family_declarations=[
            FamilyDeclaration(
                family="email_count",
                pattern=pattern,
                classification=_template(),
                status=status,
                source="owner@acme, 2026-09-30",
            )
        ],
    )


@pytest.fixture(scope="module")
def registered(tmp_path_factory):
    root = tmp_path_factory.mktemp("families")
    data = generate(
        GeneratorConfig(
            seed=3,
            n_opportunities=20,
            n_quarters=1,
            dst_dates=[],
            dst_duplicates_per_date=0,
            conflict_pairs=0,
        )
    )
    csv = write_csv(data, root / "snap.csv")
    settings = Settings(data_root=root / "data")
    dataset = register_dataset(csv, "fs", settings=settings)
    return dataset, settings


def test_windowed_and_directional_columns_collapse_into_families(proposals):
    email = proposals["email_count"]
    assert len(email.members) == 7
    assert email.windows == ("7d", "14d", "30d", "60d", "90d", "180d", "lifetime")
    tone = proposals["tone_avg"]
    assert tone.directions == ("ALL", "IN", "OUT")
    assert tone.windows == ("30d", "90d")
    assert len(tone.members) == 6
    assert set(proposals["*_updated_days"].fields) == {
        "stage",
        "amount",
        "close_date",
        "next_steps",
    }
    # Every column lands in exactly one group; singletons stay singletons.
    assert sorted(m for p in proposals.values() for m in p.members) == sorted(ALL_COLUMNS)
    assert proposals["days_since_last_email"].is_singleton


def test_label_split_and_etl_columns_get_leakage_and_metadata_proposals(proposals):
    win = proposals["win_label"]
    assert set(win.members) == {"win_label", "win_label_mask"}
    for p in (win, proposals["slip_label"], proposals["train"]):
        assert p.proposed.availability is Availability.FUTURE_CONTAMINATED
        assert p.proposed.disposition is Disposition.QUARANTINE
        assert p.proposed.reason
    for etl in ("pipeline_version", "config_hash", "run_id", "scored_at"):
        assert proposals[etl].proposed.category is ColumnCategory.METADATA
    activity = proposals["email_count"].proposed
    assert activity.availability is Availability.BACKWARD_DERIVED
    assert activity.disposition is Disposition.USE_WITH_PROOF
    assert "unconfirmed" in activity.reason
    assert proposals["owner_id"].proposed.availability is Availability.UNKNOWN
    assert all(p.proposal_only and p.proposed.proposal_only for p in proposals.values())


def test_an_unconfirmed_proposal_never_makes_a_column_readable(registered):
    dataset, _ = registered
    registry = dataset.registry
    # Inference proposes BACKWARD_DERIVED for the email family, but with no
    # declaration the registry still fails closed for every member.
    for name in (f"email_count_{w}" for w in ("7d", "lifetime")):
        column = registry.get(name)
        assert not column.classification.classified
        assert column.is_quarantined
        assert column not in registry.prospectively_usable()
    assert registry.check_requirements(["email_count_7d"]).quarantined == ["email_count_7d"]


def test_a_confirmed_family_issues_a_generic_grant_per_member(registered):
    dataset, _ = registered
    ledger = issue_grants("fs", dataset.registry, _tenant())
    granted = {g.column for g in ledger.grants if g.kind is GrantKind.GENERIC}
    expected = {c for c in ALL_COLUMNS if c.startswith("email_count_")}
    assert len(expected) == 7
    assert granted == expected
    assert not ledger.rejected


def test_an_inferred_family_issues_nothing(registered):
    dataset, _ = registered
    ledger = issue_grants("fs", dataset.registry, _tenant(BindingStatus.INFERRED))
    assert not ledger.grants
    assert len(ledger.rejected) == 7
    assert all("proposal" in r.reason for r in ledger.rejected)


def test_a_family_never_overrules_a_registry_classification(registered):
    # `stage` is classified by the registry; the family cannot release or alter it.
    dataset, _ = registered
    ledger = issue_grants("fs", dataset.registry, _tenant(pattern="stage"))
    assert not ledger.grants
    assert "already classified" in ledger.rejected[0].reason


def test_changing_a_family_declaration_changes_the_fingerprint():
    base = _tenant().declarations_fingerprint
    assert base == _tenant().declarations_fingerprint
    assert base != _tenant(pattern="email_count_7d").declarations_fingerprint
    assert base != _tenant(BindingStatus.INFERRED).declarations_fingerprint
    assert base != TenantProfile(tenant_id="acme").declarations_fingerprint


def test_report_counts_are_right_and_carry_no_values(proposals):
    report = onboarding_report(list(proposals.values()))
    families = [p for p in proposals.values() if not p.is_singleton]
    assert report.startswith(
        f"{len(ALL_COLUMNS)} columns -> {len(proposals)} groups "
        f"({len(families)} families, {len(proposals) - len(families)} singletons)"
    )
    assert "email_count (7 columns; 7 windows" in report
    assert "proposal" in report.lower()
    # Names, counts and reasons only: no synthetic cell value appears.
    assert "synthetic-fs" not in report
    assert "Closed Won" not in report
