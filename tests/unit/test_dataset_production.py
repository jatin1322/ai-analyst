"""The dataset-scoped registry and complete profile, on production-shaped data.

Expected values are computed independently, straight from the generator's own
rows with plain Python, never read back from the profiler being tested.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from decimal import Decimal

import pytest

from ai_analyst.contracts.columns import (
    Availability,
    InformationClass,
    QuarantineCode,
)
from ai_analyst.contracts.dataset import ColumnOrigin, RequirementState
from ai_analyst.contracts.opportunity_snapshot_v1 import QUARANTINE_REASONS
from ai_analyst.contracts.profile import ProfileKind
from ai_analyst.contracts.schema import DataType
from tests.fixtures.production_shape import TEXT_COLUMNS, production_rows

ROWS = production_rows()
N_ROWS = len(ROWS)


def _values(column: str) -> list[str]:
    return [r[column] for r in ROWS]


def _non_blank(column: str) -> list[str]:
    return [v for v in _values(column) if v != ""]


@pytest.fixture
def dataset(production_dataset):
    return production_dataset[0]


@pytest.fixture
def registry(dataset):
    return dataset.registry


@pytest.fixture
def profile(dataset):
    return dataset.profile


# --- 1. canonical columns are registered ------------------------------------


def test_canonical_columns_are_registered(registry):
    canonical = {c.name: c for c in registry.by_origin(ColumnOrigin.CANONICAL)}
    assert set(canonical) == {
        "as_of",
        "opp_id",
        "close_date",
        "stage",
        "amount",
        "forecast_category",
        "owner_id",
        "account_id",
        "probability",
    }
    # Canonical names, with the source header they were mapped from recorded.
    assert canonical["amount"].source_name == "new_amount"
    assert canonical["stage"].source_name == "Stage"
    assert canonical["owner_id"].source_name == "OwnerID"


def test_a_canonical_column_carries_the_classification_of_its_source(registry):
    amount = registry.get("amount")
    assert amount.classification.name == "amount"
    assert amount.information_class is InformationClass.SNAPSHOT_STATE
    # The owner confirmed new_amount is the deal amount.
    assert not amount.classification.requires_confirmation
    assert amount.dtype is DataType.DECIMAL

    stage = registry.get("stage")
    assert "NOT authoritative" in stage.classification.note


def test_derived_columns_are_registered_as_derived(registry):
    derived = {c.name for c in registry.by_origin(ColumnOrigin.DERIVED)}
    assert derived == {"status", "is_closed", "is_won"}


def test_only_the_grain_is_declared_non_nullable(registry):
    assert {c.name for c in registry.columns if not c.nullable} == {"as_of", "opp_id"}


# --- 2 and 3. discovered columns are registered and all profiled ------------


def test_discovered_columns_are_registered(registry, dataset):
    discovered = registry.by_origin(ColumnOrigin.DISCOVERED)
    names = {c.name for c in discovered}
    assert names == set(dataset.schema.discovered_columns)
    assert {"rep_win_rate", "days_to_close", "terminal_fate", "ManagerNotes", "train"} <= names
    assert len(names) == 133
    assert all(c.source_name == c.name for c in discovered)


def test_every_registered_column_has_a_profile(registry, profile):
    assert set(profile.column_names) == set(registry.names)
    assert len(profile.column_names) == len(registry.names) == 145
    assert all(c.row_count == N_ROWS for c in profile.columns)


def test_every_discovered_column_receives_a_typed_profile(registry, profile):
    by_name = {c.name: c for c in profile.columns}
    for column in registry.by_origin(ColumnOrigin.DISCOVERED):
        observed = by_name[column.name]
        assert observed.dtype is column.dtype, column.name
        assert observed.kind is not None
        assert observed.null_count + (observed.observed_count or 0) <= N_ROWS


# --- 4. text columns --------------------------------------------------------


@pytest.mark.parametrize("name", TEXT_COLUMNS)
def test_narrative_fields_are_registered_and_profiled_as_text(name, registry, profile):
    column = registry.get(name)
    assert column.information_class is InformationClass.TEXT
    assert column.dtype is DataType.VARCHAR

    observed = profile.column(name)
    assert observed.kind is ProfileKind.TEXT
    assert observed.null_count == _values(name).count("")
    lengths = [len(v) for v in _non_blank(name)]
    assert observed.max_length == max(lengths)
    assert observed.mean_length == pytest.approx(statistics.fmean(lengths))
    assert observed.distinct_count == len(set(_non_blank(name)))


def test_text_profiles_carry_length_statistics_and_no_content(profile):
    payload = profile.model_dump_json()
    for name in TEXT_COLUMNS:
        observed = profile.column(name)
        assert observed.min_value is None
        assert observed.max_value is None
        assert observed.top_values == []
    # No narrative content leaks into the profile.
    assert "confirmed" not in payload
    assert "sponsor" not in payload


def test_declared_text_stays_text_even_when_its_shape_would_not_qualify(registry, profile):
    # NextStep values are a few characters long, far below the text-detection
    # threshold. It is text because the classification says so, and the profile
    # does not overrule that.
    next_step = profile.column("NextStep")
    assert next_step.mean_length < 40
    assert registry.get("NextStep").information_class is InformationClass.TEXT
    catalogue = {t.name: t for t in profile.text_columns}
    assert set(catalogue) == set(TEXT_COLUMNS)
    assert all(t.basis == "classification" for t in catalogue.values())


# --- 5. numeric, date, boolean and categorical columns ----------------------


def test_integer_features_are_bigint_with_exact_extremes(profile):
    observed = profile.column("days_to_close")
    values = [int(v) for v in _values("days_to_close")]
    assert observed.dtype is DataType.BIGINT
    assert observed.kind is ProfileKind.NUMERIC
    assert observed.min_value == str(min(values))
    assert observed.max_value == str(max(values))
    assert observed.mean == pytest.approx(statistics.fmean(values))
    assert observed.median == pytest.approx(statistics.median(values))
    assert observed.stddev == pytest.approx(statistics.stdev(values))
    assert observed.null_count == 0


def test_fractional_features_are_double(profile):
    observed = profile.column("rep_win_rate")
    values = [float(v) for v in _values("rep_win_rate")]
    assert observed.dtype is DataType.DOUBLE
    assert observed.mean == pytest.approx(statistics.fmean(values))
    assert float(observed.min_value) == pytest.approx(min(values))
    assert float(observed.max_value) == pytest.approx(max(values))


def test_zero_one_flags_are_integers_not_booleans(profile):
    assert profile.column("is_pushed_out_deal").dtype is DataType.BIGINT
    assert profile.column("train").dtype is DataType.BIGINT


def test_money_keeps_exact_extremes_and_reports_no_float_average(profile):
    amount = profile.column("amount")
    values = [Decimal(v) for v in _values("new_amount")]
    assert amount.dtype is DataType.DECIMAL
    assert amount.min_value == format(min(values), "f")
    assert amount.max_value == format(max(values), "f")
    # A DECIMAL column is never averaged, because that would route money
    # through a float.
    assert amount.mean is None
    assert amount.median is None
    assert amount.stddev is None


def test_date_columns_report_null_count_and_extremes(profile):
    observed = profile.column("terminal_date")
    present = sorted(_non_blank("terminal_date"))
    assert observed.dtype is DataType.DATE
    assert observed.kind is ProfileKind.DATE
    assert observed.null_count == N_ROWS - len(present)
    assert observed.min_value == present[0]
    assert observed.max_value == present[-1]


def test_true_false_columns_are_boolean_with_counts(profile):
    observed = profile.column("CD_in_qtr")
    counts = Counter(_values("CD_in_qtr"))
    assert observed.dtype is DataType.BOOLEAN
    assert observed.kind is ProfileKind.BOOLEAN
    assert {t.value: t.count for t in observed.top_values} == {
        "true": counts["true"],
        "false": counts["false"],
    }


def test_categorical_columns_list_top_values(profile):
    observed = profile.column("Type")
    counts = Counter(_values("Type"))
    assert observed.kind is ProfileKind.CATEGORICAL
    assert observed.distinct_count == len(counts)
    assert {t.value: t.count for t in observed.top_values} == dict(counts)
    assert not observed.high_cardinality


def test_null_rate_counts_only_genuine_blanks(profile):
    observed = profile.column("terminal_date")
    assert observed.null_rate == observed.null_count / N_ROWS


# --- 6. sentinels are column-specific ---------------------------------------


@pytest.mark.parametrize("name", ["account_ti_first_won", "account_ti_first_loss"])
def test_declared_sentinels_are_excluded_from_every_statistic(name, registry, profile):
    raw = [int(v) for v in _values(name)]
    real = [v for v in raw if v != -999999]
    observed = profile.column(name)

    assert registry.get(name).classification.sentinels == (-999999.0,)
    assert observed.sentinel_values == (-999999.0,)
    # Sentinel, null and observed are three different things.
    assert observed.sentinel_count == len(raw) - len(real)
    assert observed.null_count == 0
    assert observed.observed_count == len(real)

    assert observed.min_value == str(min(real))
    assert observed.max_value == str(max(real))
    assert observed.mean == pytest.approx(statistics.fmean(real))
    assert observed.median == pytest.approx(statistics.median(real))
    # The unmasked mean would be wildly different, which is the whole hazard.
    assert observed.mean != pytest.approx(statistics.fmean(raw))
    assert observed.distinct_count == len(set(real))


def test_an_undeclared_column_holding_the_sentinel_value_is_not_masked(registry, profile):
    # -999999 is only a sentinel where the column's metadata says so. Here it is
    # an ordinary value and must stay in the statistics.
    name = "days_since_pulled_into_qtr"
    raw = [int(v) for v in _values(name)]
    assert -999999 in raw

    assert not registry.get(name).has_sentinel
    observed = profile.column(name)
    assert observed.sentinel_values == ()
    assert observed.sentinel_count == 0
    assert observed.min_value == "-999999"
    assert observed.mean == pytest.approx(statistics.fmean(raw))
    assert observed.observed_count == len(raw)


def test_only_the_two_declared_columns_carry_sentinels(registry):
    assert {c.name for c in registry.sentinel_columns()} == {
        "account_ti_first_won",
        "account_ti_first_loss",
    }


# --- 8. missing authoritative status is visible ------------------------------


def test_missing_authoritative_status_is_visible_everywhere(dataset, registry, profile):
    assert not dataset.status_is_authoritative
    assert dataset.schema.requires_confirmation

    assert profile.status is not None
    assert not profile.status.authoritative
    assert profile.status.configured_column is None
    assert profile.status.requires_confirmation
    assert "NOT AUTHORITATIVE" in profile.status.note
    assert profile.status.strategy.value == "stage_keyword"
    assert sum(profile.status.distribution.values()) == N_ROWS

    open_ids = {u.id for u in registry.open_unresolved()}
    assert "authoritative_status" in open_ids


def test_invalid_stage_labels_are_excluded_not_counted_as_open(profile):
    stages = Counter(_values("Stage"))
    invalid = stages["SFDCDELETED"] + stages["not available"]
    assert profile.status.distribution["excluded"] == invalid


def test_stage_vocabulary_is_never_reported_as_confirmed(profile):
    assert profile.stage_vocabulary.requires_confirmation
    assert all(s.inferred for s in profile.stage_vocabulary.stages)


# --- 9. quarantined columns remain quarantined, with reasons ----------------


def test_the_nineteen_quarantined_columns_are_unchanged(registry):
    assert {c.name for c in registry.quarantined()} == set(QUARANTINE_REASONS)
    assert len(registry.quarantined()) == 19


def test_every_quarantined_column_states_why(registry):
    for column in registry.quarantined():
        reason = column.quarantine
        assert reason is not None, column.name
        assert reason.detail and reason.resolution


@pytest.mark.parametrize(
    ("name", "code"),
    [
        ("train", QuarantineCode.NON_ANALYTIC_METADATA),
        ("opp_commit_to_close_days", QuarantineCode.AMBIGUOUS_DEFINITION),
        ("deal_amount_percentile_overall", QuarantineCode.UNSTATED_REFERENCE_POPULATION),
        ("deal_amount_rank_pct", QuarantineCode.UNSTATED_REFERENCE_POPULATION),
        ("deal_amount_vs_stage_avg", QuarantineCode.UNSTATED_REFERENCE_POPULATION),
        ("deal_amount_vs_rep_avg", QuarantineCode.UNSTATED_REFERENCE_POPULATION),
        ("activity_density", QuarantineCode.UNDEFINED_COMPOSITE),
        ("eoq_urgency_score", QuarantineCode.UNDEFINED_COMPOSITE),
    ],
)
def test_named_quarantines_keep_their_reason(name, code, registry):
    column = registry.get(name)
    assert column.is_quarantined
    assert column.quarantine.code is code
    assert not column.classification.usable_prospectively


def test_a_quarantined_column_is_reported_as_such_to_a_requirement_check(dataset):
    check = dataset.check_requirements(["train", "amount", "opp_commit_to_close_days"])
    states = {r.name: r.state for r in check.requirements}
    assert states["train"] is RequirementState.QUARANTINED
    assert states["opp_commit_to_close_days"] is RequirementState.QUARANTINED
    assert states["amount"] is RequirementState.AVAILABLE
    assert not check.satisfiable_prospectively
    assert not check.satisfiable_retrospectively
    reasons = {r.name: r.quarantine.code for r in check.requirements if r.quarantine}
    assert reasons["train"] is QuarantineCode.NON_ANALYTIC_METADATA


# --- 6 (registry). information timing is exposed ----------------------------


def test_outcome_columns_are_retrospective_and_not_knowable_at_the_snapshot(dataset, registry):
    outcomes = [c for c in registry.columns if c.name.startswith("terminal_")]
    assert len(outcomes) == 5
    for column in outcomes:
        assert column.information_class is InformationClass.RETROSPECTIVE_OUTCOME
        assert column.classification.availability is Availability.FUTURE_CONTAMINATED
        assert not column.classification.knowable_at_snapshot
    check = dataset.check_requirements(["terminal_fate", "amount"])
    assert check.not_knowable == ["terminal_fate"]
    assert not check.satisfiable_prospectively
    assert check.satisfiable_retrospectively


def test_the_four_information_kinds_are_all_distinguishable(registry):
    kinds = {
        name: registry.get(name).information_class
        for name in ("amount", "close_date_push_count", "terminal_fate", "ManagerNotes")
    }
    assert kinds == {
        "amount": InformationClass.SNAPSHOT_STATE,
        "close_date_push_count": InformationClass.HISTORICAL_FEATURE,
        "terminal_fate": InformationClass.RETROSPECTIVE_OUTCOME,
        "ManagerNotes": InformationClass.TEXT,
    }


# --- 14. unresolved items are data ------------------------------------------


def test_every_open_item_is_recorded_as_data(registry):
    # Prose is not a record. A later layer enumerates these; anything living
    # only in a document would read as an all-clear here.
    open_items = {u.id: u for u in registry.open_unresolved()}
    assert set(open_items) == {
        "opp_commit_to_close_days_definition",
        "rep_aggregate_lineage",
        "authoritative_status",
        # The gaps the real-data inspection left around the reconstruction.
        "days_to_close_edge_cases",
        "eoq_close_diff_convention",
        "fiscal_year_start_month",
        "date_encoding_declarations",
        "quarter_boundary_day_convention",
    }
    assert len(open_items["rep_aggregate_lineage"].columns) == 23
    assert open_items["opp_commit_to_close_days_definition"].columns == (
        "opp_commit_to_close_days",
    )
    assert all(not u.resolved for u in open_items.values())


def test_rep_aggregates_carry_their_unresolved_lineage_as_a_caveat(dataset):
    # Their classification is unchanged from the previous milestone; what is new
    # is that a consumer can now see the lineage question attached to each one.
    check = dataset.check_requirements(["rep_win_rate", "rep_pushout_rate"])
    for requirement in check.requirements:
        assert requirement.state is RequirementState.AVAILABLE_WITH_CAVEATS
        assert "rep_aggregate_lineage" in requirement.caveats
        assert "lineage_unconfirmed" in requirement.caveats


# --- compactness ------------------------------------------------------------


def test_the_profile_stays_within_a_per_column_size_budget(production_dataset):
    # Measured at ~230 bytes per column and ~59 KB in total for 145 columns
    # (about 15k tokens). The budget sits close to that, so a regression such as
    # top values appearing on text columns or a raised top_k fails here instead
    # of passing through loose headroom.
    dataset, settings = production_dataset
    compact = [
        json.dumps(
            c.model_dump(mode="json", exclude_none=True, exclude_defaults=True),
            separators=(",", ":"),
        )
        for c in dataset.profile.columns
    ]
    per_column = sum(len(c) for c in compact) / len(compact)
    assert per_column < 300, f"{per_column:.0f} bytes per column"
    assert len(dataset.profile.model_dump_json()) < 70_000

    for column in dataset.profile.columns:
        assert len(column.top_values) <= settings.top_k_values
        if column.kind is ProfileKind.TEXT:
            assert column.top_values == []
    text = settings.profile_path(dataset.dataset_id).read_text(encoding="utf-8")
    assert json.loads(text)["row_count"] == N_ROWS
