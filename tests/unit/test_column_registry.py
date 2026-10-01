"""Column classification registry.

These test completeness and internal consistency, not whether an individual
call is correct. Correctness of a name-based assignment can only be settled by
looking at the data, which is why every entry carries requires_confirmation.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ai_analyst.contracts.columns import (
    Availability,
    ColumnCategory,
    ColumnClassification,
    ColumnRegistry,
    CoverageStatus,
    Disposition,
)
from ai_analyst.contracts.opportunity_snapshot_v1 import (
    ALL_COLUMNS,
    FAMILIES,
    INDIVIDUAL,
)
from ai_analyst.contracts.opportunity_snapshot_v1 import (
    OPPORTUNITY_SNAPSHOT_V1 as REGISTRY,
)

EXPECTED_COLUMN_COUNT = 129


def test_every_column_is_classified_exactly_once():
    assert len(ALL_COLUMNS) == EXPECTED_COLUMN_COUNT
    assert len(set(ALL_COLUMNS)) == EXPECTED_COLUMN_COUNT
    assert len(REGISTRY.columns) == EXPECTED_COLUMN_COUNT
    assert {c.name for c in REGISTRY.columns} == set(ALL_COLUMNS)


def test_registry_rejects_a_duplicate_classification():
    entry = ColumnClassification(
        name="dup",
        category=ColumnCategory.METADATA,
        availability=Availability.AS_OF_FACT,
        disposition=Disposition.DIRECT,
    )
    with pytest.raises(ValidationError, match="classified more than once"):
        ColumnRegistry(name="x", columns=[entry, entry])


def test_unknown_availability_fails_closed_like_contamination():
    # The entire safety property of ARCHITECTURE 5.8.
    assert not Availability.UNKNOWN.safe_for_prospective
    assert not Availability.FUTURE_CONTAMINATED.safe_for_prospective
    assert Availability.AS_OF_FACT.safe_for_prospective
    assert Availability.BACKWARD_DERIVED.safe_for_prospective


def test_no_unsafe_column_is_dispositioned_for_direct_use():
    for column in REGISTRY.columns:
        if not column.availability.safe_for_prospective:
            assert column.disposition is not Disposition.DIRECT, column.name


def test_direct_disposition_is_rejected_at_construction_for_unsafe_columns():
    with pytest.raises(ValidationError, match="dispositioned DIRECT"):
        ColumnClassification(
            name="terminal_fate",
            category=ColumnCategory.OUTCOME,
            availability=Availability.FUTURE_CONTAMINATED,
            disposition=Disposition.DIRECT,
        )


def test_quarantined_columns_are_never_prospectively_usable():
    for column in REGISTRY.quarantined():
        assert not column.usable_prospectively, column.name


def test_family_patterns_match_exactly_what_they_claim():
    by_family: dict[str, list[str]] = {}
    for column in REGISTRY.columns:
        if column.family:
            by_family.setdefault(column.family, []).append(column.name)

    assert len(by_family["field_update_recency"]) == 22
    assert all(n.endswith("_updated_days") for n in by_family["field_update_recency"])

    assert len(by_family["rep_aggregate"]) == 23
    assert all(n.startswith("rep_") for n in by_family["rep_aggregate"])

    assert len(by_family["terminal_outcome"]) == 5
    assert all(n.startswith("terminal_") for n in by_family["terminal_outcome"])

    # Families and individual assignments must partition the column list.
    assert sum(len(v) for v in by_family.values()) + len(INDIVIDUAL) == EXPECTED_COLUMN_COUNT


def test_individual_assignments_do_not_collide_with_families():
    for entry in INDIVIDUAL:
        for family in FAMILIES:
            assert not family.matches(entry.name), f"{entry.name} shadows {family.name}"


def test_all_terminal_columns_are_future_contaminated():
    # These are the outcome labels. Using one prospectively is the worst
    # available failure, so the classification must be unambiguous.
    for name in (
        "terminal_date",
        "terminal_fate",
        "terminal_amount",
        "terminal_quarter_eoq",
        "terminal_date_qtr",
    ):
        column = REGISTRY.get(name)
        assert column.availability is Availability.FUTURE_CONTAMINATED
        assert not column.usable_prospectively


def test_rep_aggregates_are_preserved_with_unconfirmed_lineage():
    # Confirmed to be frozen through the previous quarter, which is what takes
    # them out of quarantine. Preserved rather than recomputed, per project
    # direction, but no individual lookback has been documented yet.
    rep = [c for c in REGISTRY.columns if c.name.startswith("rep_")]
    assert len(rep) == 23
    assert all(c.disposition is Disposition.USE_WITH_PROOF for c in rep)
    assert all(c.availability is Availability.BACKWARD_DERIVED for c in rep)
    assert all(c.lineage is not None and not c.lineage.confirmed for c in rep)
    assert not any(c.recomputable for c in rep)


def test_the_six_named_feature_columns_reflect_the_confirmations():
    expected = {
        # Movement flags stay recomputable within the export window.
        "is_pushed_out_deal": Disposition.RECOMPUTE,
        "is_pulled_in_deal": Disposition.RECOMPUTE,
        # Cumulative since creation, so an opportunity older than the export
        # window has history the system cannot see. Recomputing undercounts.
        "close_date_push_count": Disposition.USE_WITH_PROOF,
        "stage_changes_count": Disposition.USE_WITH_PROOF,
        # Frozen through the prior quarter, preserved rather than recomputed.
        "rep_win_rate": Disposition.USE_WITH_PROOF,
        "historical_win_rate_at_stage": Disposition.USE_WITH_PROOF,
    }
    for name, disposition in expected.items():
        assert REGISTRY.get(name).disposition is disposition, name


def test_cumulative_counters_are_backward_derived_and_not_recomputed():
    for name in ("close_date_push_count", "close_date_pull_count", "stage_changes_count"):
        column = REGISTRY.get(name)
        assert column.availability is Availability.BACKWARD_DERIVED, name
        assert column.usable_prospectively, name
        assert not column.recomputable, name


def test_close_date_family_is_snapshot_state_not_terminal_derived():
    days_to_close = REGISTRY.get("days_to_close")
    assert days_to_close.availability is Availability.AS_OF_FACT
    assert not days_to_close.requires_confirmation
    for name in ("CD_in_qtr", "CD_in_past", "eoq_close_diff", "close_date_qtr"):
        assert REGISTRY.get(name).availability is Availability.AS_OF_FACT, name


def test_account_time_since_features_carry_sentinel_and_leakage_check():
    # -999999 entering a mean would be catastrophic and silent, and the feature
    # is defined through terminal_date so no-look-ahead must be verified.
    for name in ("account_ti_first_won", "account_ti_first_loss"):
        column = REGISTRY.get(name)
        assert column.sentinels == (-999999.0,), name
        assert column.leakage_check, name
        assert column.has_sentinel


def test_qtr_segment_is_quarter_phase_not_customer_segment():
    column = REGISTRY.get("qtr_segment")
    assert column.category is ColumnCategory.QUARTER
    assert not column.requires_confirmation
    assert "NOT a customer segment" in column.note


def test_stage_is_not_authoritative_for_status():
    column = REGISTRY.get("Stage")
    assert column.category is ColumnCategory.SNAPSHOT_STATE
    assert "NOT authoritative" in column.note


def test_information_class_is_derived_and_never_contradicts_the_axes():
    from ai_analyst.contracts.columns import InformationClass

    for column in REGISTRY.columns:
        info = column.information_class
        if column.category is ColumnCategory.OUTCOME:
            assert info is InformationClass.RETROSPECTIVE_OUTCOME, column.name
        if info is InformationClass.RETROSPECTIVE_OUTCOME:
            assert not column.usable_prospectively, column.name


def test_crm_narrative_fields_classify_as_text_when_present():
    # Confirmed to exist upstream, absent from this 129-column export.
    for name in ("ManagerNotes", "SENotes", "Why_Now", "NextStep", "Why_Us"):
        classified = REGISTRY.classify_unknown(name)
        assert classified.category is ColumnCategory.TEXT, name
        assert classified.family == "crm_narrative"


def test_an_unrecognised_column_fails_closed():
    unknown = REGISTRY.classify_unknown("some_new_column_nobody_documented")
    assert not unknown.usable_prospectively
    assert unknown.availability is Availability.UNKNOWN


def test_train_flag_is_quarantined_metadata():
    # Subsetting on a train/test split would silently change every number.
    train = REGISTRY.get("train")
    assert train.category is ColumnCategory.METADATA
    assert train.disposition is Disposition.QUARANTINE


def test_update_recency_columns_are_backward_derived_and_usable():
    recency = [c for c in REGISTRY.columns if c.family == "field_update_recency"]
    assert all(c.availability is Availability.BACKWARD_DERIVED for c in recency)
    assert all(c.usable_prospectively for c in recency)


def test_no_text_column_is_present_in_this_export():
    # The narrative fields were not exported; only their recency derivatives
    # were. ARCHITECTURE 5.11 therefore has nothing to catalogue here.
    assert REGISTRY.in_category(ColumnCategory.TEXT) == []


def test_recomputable_columns_are_marked_for_recomputation():
    for column in REGISTRY.columns:
        if column.recomputable:
            assert column.disposition in (
                Disposition.RECOMPUTE,
                Disposition.USE_WITH_PROOF,
            ), column.name


def test_only_owner_confirmed_assignments_are_settled():
    confirmed = {c.name for c in REGISTRY.columns if not c.requires_confirmation}
    assert confirmed == {
        "opp_id",
        "as_of",
        "new_amount",
        "days_to_close",
        "qtr_segment",
        "Stage",
    }


def test_canonical_coverage_is_recorded_for_every_required_field():
    for required in ("as_of", "opp_id", "close_date", "stage", "amount"):
        REGISTRY.coverage_for(required)


def test_close_date_is_reconstructible_not_mapped():
    # The blocking finding: the canonical schema requires close_date and this
    # export has no such column.
    coverage = REGISTRY.coverage_for("close_date")
    assert coverage.status is CoverageStatus.RECONSTRUCTIBLE
    assert coverage.source == "days_to_close"
    assert coverage.derivation == "as_of + days_to_close"
    assert not coverage.verified


def test_created_date_is_reconstructible_from_age():
    coverage = REGISTRY.coverage_for("created_date")
    assert coverage.status is CoverageStatus.RECONSTRUCTIBLE
    assert coverage.derivation == "as_of - age"


def test_segment_region_and_industry_are_absent():
    # Segment analysis is not answerable from this export.
    for name in ("segment", "region", "industry"):
        assert REGISTRY.coverage_for(name).status is CoverageStatus.ABSENT


def test_no_coverage_record_claims_an_unverified_mapping_is_verified():
    for record in REGISTRY.coverage:
        if record.verified:
            assert record.status is CoverageStatus.MAPPED, record.canonical


def test_prospectively_usable_set_excludes_every_outcome_column():
    usable = {c.name for c in REGISTRY.prospectively_usable()}
    for column in REGISTRY.in_category(ColumnCategory.OUTCOME):
        assert column.name not in usable


def test_information_class_distribution_matches_the_architecture_document():
    # ARCHITECTURE 5.14 publishes these counts. Pinning them here stops the
    # document and the registry from drifting apart.
    from ai_analyst.contracts.columns import InformationClass

    distribution = {
        info.value: len(REGISTRY.in_information_class(info))
        for info in InformationClass
        if REGISTRY.in_information_class(info)
    }
    assert distribution == {
        "identity": 5,
        "snapshot_state": 16,
        "temporal_context": 18,
        "historical_feature": 84,
        "retrospective_outcome": 5,
        "metadata": 1,
    }
    assert sum(distribution.values()) == EXPECTED_COLUMN_COUNT


def test_prospective_and_quarantine_counts_match_the_architecture_document():
    assert len(REGISTRY.prospectively_usable()) == 95
    assert len(REGISTRY.quarantined()) == 19


def test_information_class_says_what_a_column_is_not_when_it_is_safe():
    # A column with a settled category but an unstated window still reports
    # what it is, rather than vanishing into an unclassified bucket.
    from ai_analyst.contracts.columns import InformationClass

    percentile = REGISTRY.get("deal_amount_percentile_overall")
    assert percentile.information_class is InformationClass.SNAPSHOT_STATE
    assert not percentile.usable_prospectively


def test_every_registry_column_is_classified():
    from ai_analyst.contracts.columns import InformationClass

    assert all(c.classified for c in REGISTRY.columns)
    assert REGISTRY.in_information_class(InformationClass.UNCLASSIFIED) == []
