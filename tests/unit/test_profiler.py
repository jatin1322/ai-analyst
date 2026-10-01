"""Dataset profiling.

Every expected value here is hand-computed from tests/fixtures/tiny/README.md,
never read back from the code under test.
"""

from __future__ import annotations

from datetime import date

import pytest

from ai_analyst.config import Settings
from ai_analyst.contracts.errors import ProfilingError
from ai_analyst.contracts.profile import DatasetProfile
from ai_analyst.contracts.schema import CanonicalColumn, DatasetSchema
from ai_analyst.data.profiler import load_profile, profile_dataset
from ai_analyst.data.store import DuckDBStore
from tests.conftest import TINY_DATASET_ID


def test_row_count(tiny_profile: DatasetProfile):
    assert tiny_profile.row_count == 40  # 6+7+7+6+7+7


def test_snapshot_inventory_and_per_snapshot_row_counts(tiny_profile: DatasetProfile):
    assert [(s.as_of, s.row_count) for s in tiny_profile.snapshots] == [
        (date(2025, 1, 1), 6),
        (date(2025, 2, 1), 7),
        (date(2025, 3, 31), 7),
        (date(2025, 4, 1), 6),
        (date(2025, 5, 1), 7),
        (date(2025, 6, 30), 7),
    ]
    assert tiny_profile.min_as_of == date(2025, 1, 1)
    assert tiny_profile.max_as_of == date(2025, 6, 30)


def test_null_counts(tiny_profile: DatasetProfile):
    # OPP-005 has 3 rows with blank arr; OPP-006 has 2 rows with blank region.
    assert tiny_profile.column(CanonicalColumn.ARR).null_count == 3
    assert tiny_profile.column(CanonicalColumn.REGION).null_count == 2
    assert tiny_profile.column(CanonicalColumn.OPP_ID).null_count == 0


def test_null_rate_is_derived_from_the_counts(tiny_profile: DatasetProfile):
    assert tiny_profile.column(CanonicalColumn.ARR).null_rate == 3 / 40


def test_distinct_counts(tiny_profile: DatasetProfile):
    assert tiny_profile.column(CanonicalColumn.OPP_ID).distinct_count == 8
    assert tiny_profile.column(CanonicalColumn.AS_OF).distinct_count == 6
    # Enterprise, Mid-Market, SMB
    assert tiny_profile.column(CanonicalColumn.SEGMENT).distinct_count == 3
    # Discovery, Qualification, Proposal, Negotiation, Closed Won, Closed Lost
    assert tiny_profile.column(CanonicalColumn.STAGE).distinct_count == 6


def test_min_max_are_exact_strings_not_floats(tiny_profile: DatasetProfile):
    amount = tiny_profile.column(CanonicalColumn.AMOUNT)
    assert amount.min_value == "40000.00"   # OPP-005
    assert amount.max_value == "250000.00"  # OPP-004 at 2025-03-31
    assert isinstance(amount.min_value, str)


def test_date_extremes(tiny_profile: DatasetProfile):
    close = tiny_profile.column(CanonicalColumn.CLOSE_DATE)
    assert close.min_value == "2025-02-28"  # OPP-007
    assert close.max_value == "2025-09-15"  # OPP-006
    created = tiny_profile.column(CanonicalColumn.CREATED_DATE)
    assert created.min_value == "2024-10-01"  # OPP-007
    assert created.max_value == "2025-04-20"  # OPP-006


def test_top_values_are_ranked_for_categorical_columns(tiny_profile: DatasetProfile):
    segment = tiny_profile.column(CanonicalColumn.SEGMENT)
    counts = {t.value: t.count for t in segment.top_values}
    # Enterprise: OPP-001 (6) + OPP-004 (6) + OPP-006 (2) + OPP-003 from 2025-03-31 (4)
    assert counts["Enterprise"] == 18
    # Mid-Market: OPP-002 (6) + OPP-003 first two snapshots (2) + OPP-008 (5)
    assert counts["Mid-Market"] == 13
    # SMB: OPP-005 (3) + OPP-007 (6)
    assert counts["SMB"] == 9
    assert sum(counts.values()) == 40


def test_numeric_columns_get_no_top_values(tiny_profile: DatasetProfile):
    assert tiny_profile.column(CanonicalColumn.AMOUNT).top_values == []


def test_stage_vocabulary_and_row_counts(tiny_profile: DatasetProfile):
    vocab = tiny_profile.stage_vocabulary
    counts = {s.stage: s.row_count for s in vocab.stages}
    assert counts == {
        "Negotiation": 10,
        "Discovery": 8,
        "Proposal": 6,
        "Qualification": 6,
        "Closed Won": 6,
        "Closed Lost": 4,
    }
    assert sum(counts.values()) == 40


def test_stage_closed_won_mapping_is_inferred_and_flagged(tiny_profile: DatasetProfile):
    vocab = tiny_profile.stage_vocabulary
    assert vocab.won_labels == ["Closed Won"]
    assert vocab.lost_labels == ["Closed Lost"]
    assert sorted(vocab.open_labels) == [
        "Discovery",
        "Negotiation",
        "Proposal",
        "Qualification",
    ]
    # A wrong closed/won mapping corrupts every rate, so it must be confirmed.
    assert vocab.requires_confirmation
    assert all(s.inferred for s in vocab.stages)


def test_lifecycle_statistics(tiny_profile: DatasetProfile):
    life = tiny_profile.lifecycle
    assert life.distinct_opportunities == 8
    assert life.snapshot_count == 6
    assert life.min_snapshots_per_opportunity == 2  # OPP-006
    assert life.max_snapshots_per_opportunity == 6
    # sorted [2,3,5,6,6,6,6,6]; median is the mean of the 4th and 5th values
    assert life.median_snapshots_per_opportunity == 6.0
    # OPP-001, 002, 003, 004, 007 appear in all six snapshots
    assert life.present_in_all_snapshots == 5


def test_vanished_without_terminal_state_is_the_other_removed_population(
    tiny_profile: DatasetProfile,
):
    # ARCHITECTURE §5.2: this term must be computed, not inferred as a residual.
    life = tiny_profile.lifecycle
    assert life.vanished_without_terminal_state == 1
    assert life.vanished_sample_opp_ids == ["OPP-005"]


def test_closed_opportunities_are_not_counted_as_vanished(tiny_profile: DatasetProfile):
    # OPP-007 closed lost and stayed in every later snapshot.
    assert "OPP-007" not in tiny_profile.lifecycle.vanished_sample_opp_ids


def test_grain_check_passes_on_the_clean_fixture(tiny_profile: DatasetProfile):
    grain = tiny_profile.grain
    assert grain.passed
    assert grain.total_rows == 40
    assert grain.distinct_keys == 40
    assert grain.duplicate_key_count == 0


def test_quality_flags_are_clean_on_the_fixture(tiny_profile: DatasetProfile):
    quality = tiny_profile.quality
    assert quality.negative_amount_rows == 0
    assert quality.close_date_before_created_date_rows == 0
    # The three blank arr cells are legitimate nulls, not failed casts. A clean
    # dataset must not report data quality problems it does not have.
    assert quality.cast_failure_rows == {}
    assert not quality.any_flagged
    # The nulls are still reported, by the field that actually means nulls.
    assert tiny_profile.column(CanonicalColumn.ARR).null_count == 3


def test_fiscal_drift_is_left_unset_until_the_semantic_calendar_exists(
    tiny_profile: DatasetProfile,
):
    assert tiny_profile.snapshot_drift_days is None


def test_profile_is_persisted_and_reloadable(
    tiny_profile: DatasetProfile, settings: Settings
):
    assert settings.profile_path(TINY_DATASET_ID).exists()
    assert load_profile(TINY_DATASET_ID, settings) == tiny_profile


def test_persisted_profile_keeps_money_exact(
    tiny_profile: DatasetProfile, settings: Settings
):
    reloaded = load_profile(TINY_DATASET_ID, settings)
    assert reloaded.column(CanonicalColumn.AMOUNT).max_value == "250000.00"


def test_profiling_an_uningested_dataset_fails_cleanly(
    tiny_schema: DatasetSchema, settings: Settings, store: DuckDBStore
):
    with pytest.raises(ProfilingError, match="no canonical parquet"):
        profile_dataset("never-ingested", tiny_schema, settings=settings, store=store)


def test_top_k_is_configurable(
    tiny_schema: DatasetSchema, settings: Settings
):
    limited = settings.model_copy(update={"top_k_values": 2})
    profile = profile_dataset(
        TINY_DATASET_ID, tiny_schema, settings=limited, store=DuckDBStore(limited)
    )
    assert len(profile.column(CanonicalColumn.STAGE).top_values) == 2
