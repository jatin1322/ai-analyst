"""Authoritative status configuration on a dataset (ARCHITECTURE 5.13).

The production status column's real name and values are unknown and must not be
guessed, so these tests use a clearly synthetic column.
"""

from __future__ import annotations

from collections import Counter

import pytest

from ai_analyst.config import Settings
from ai_analyst.contracts.errors import MappingError, StatusConfigInvalid
from ai_analyst.contracts.opportunity_snapshot_v1 import OPPORTUNITY_SNAPSHOT_V1
from ai_analyst.contracts.schema import CanonicalColumn
from ai_analyst.contracts.status import StatusMapping, StatusStrategy
from ai_analyst.data.dataset import load_dataset, register_dataset
from ai_analyst.data.ingest import IngestionError
from ai_analyst.data.store import DuckDBStore
from tests.conftest import TINY_CSV
from tests.fixtures.production_shape import (
    STATUS_COLUMN,
    STATUS_VALUES,
    production_rows,
    write_production_csv,
)


def _config(tmp_path, **overrides) -> Settings:
    base = {
        "data_root": tmp_path / "data",
        "status_column": STATUS_COLUMN,
        "status_won_values": (STATUS_VALUES["won"],),
        "status_lost_values": (STATUS_VALUES["lost"],),
        "status_open_values": (STATUS_VALUES["open"],),
        "status_excluded_values": (STATUS_VALUES["excluded"],),
    }
    return Settings(**{**base, **overrides})


@pytest.fixture
def status_csv(tmp_path):
    path = tmp_path / "with_status.csv"
    write_production_csv(path, status_column=True)
    return path


def _register(csv, settings, **kwargs):
    return register_dataset(
        csv,
        "status_ds",
        column_registry=OPPORTUNITY_SNAPSHOT_V1,
        settings=settings,
        store=DuckDBStore(settings),
        **kwargs,
    )


def test_a_configured_status_column_becomes_authoritative(status_csv, tmp_path):
    dataset = _register(status_csv, _config(tmp_path))

    assert dataset.status_is_authoritative
    resolution = dataset.schema.status_resolution
    assert resolution.strategy is StatusStrategy.AUTHORITATIVE_COLUMN
    assert resolution.column == STATUS_COLUMN
    assert dataset.profile.status.authoritative
    assert dataset.profile.status.configured_column == STATUS_COLUMN


def test_status_distribution_follows_the_authoritative_column_not_the_stage(
    status_csv, tmp_path
):
    dataset = _register(status_csv, _config(tmp_path))
    expected = Counter(
        {"O": "open", "W": "won", "L": "lost", "DELETED": "excluded"}.get(
            r[STATUS_COLUMN], "unknown"
        )
        for r in production_rows(status_column=True)
    )
    assert dataset.profile.status.distribution == dict(expected)


def test_values_outside_the_mapping_are_visible_not_guessed(status_csv, tmp_path):
    dataset = _register(status_csv, _config(tmp_path))

    assert dataset.schema.status_resolution.unmapped_values == ["PENDING"]
    assert dataset.profile.status.unmapped_values == ["PENDING"]
    assert dataset.profile.status.distribution["unknown"] == 1
    assert dataset.schema.requires_confirmation
    ids = {u.id for u in dataset.registry.open_unresolved()}
    assert "status_unmapped_values" in ids
    assert "authoritative_status" not in ids


def test_a_fully_mapped_status_column_raises_nothing_to_resolve(status_csv, tmp_path):
    settings = _config(tmp_path, status_open_values=("O", "PENDING"))
    dataset = _register(status_csv, settings)

    assert dataset.schema.status_resolution.unmapped_values == []
    assert not dataset.schema.requires_confirmation
    assert "authoritative_status" not in {u.id for u in dataset.registry.open_unresolved()}
    assert "status_unmapped_values" not in {u.id for u in dataset.registry.open_unresolved()}


def test_status_configuration_is_persisted_and_reloaded(status_csv, tmp_path):
    settings = _config(tmp_path)
    original = _register(status_csv, settings)

    reloaded = load_dataset("status_ds", settings)
    assert reloaded.schema.status_resolution == original.schema.status_resolution
    assert reloaded.schema.status_resolution.mapping == StatusMapping(
        won_values=("W",), lost_values=("L",), open_values=("O",), excluded_values=("DELETED",)
    )
    assert reloaded.status_is_authoritative
    assert reloaded.profile.status == original.profile.status


def test_the_status_source_is_not_also_preserved_as_a_discovered_column(status_csv, tmp_path):
    dataset = _register(status_csv, _config(tmp_path))
    assert STATUS_COLUMN not in dataset.schema.discovered_columns
    assert dataset.registry.has("status")


def test_the_authoritative_column_wins_over_the_stage_label(status_csv, tmp_path):
    # '6 - Order Placed' reads as open to keyword inference. Under an
    # authoritative column a row's status comes from that column alone.
    settings = _config(tmp_path)
    dataset = _register(status_csv, settings)
    store = DuckDBStore(settings)
    with store.connect() as conn:
        rows = conn.execute(
            f"SELECT stage, status FROM {store.snapshots_scan('status_ds')} "
            "WHERE stage = 'Closed Won'"
        ).fetchall()
    assert rows and all(status == "won" for _, status in rows)
    assert dataset.status_is_authoritative


# --- configuration errors are loud, never a silent fallback ------------------


def test_a_configured_column_missing_from_the_source_is_an_error(tmp_path):
    csv = tmp_path / "no_status.csv"
    write_production_csv(csv)
    with pytest.raises(IngestionError) as excinfo:
        _register(csv, _config(tmp_path))
    detail = excinfo.value.detail
    assert isinstance(detail, StatusConfigInvalid)
    assert detail.reason == "column_not_in_source"
    assert detail.column == STATUS_COLUMN


def test_an_empty_value_mapping_is_an_error(status_csv, tmp_path):
    settings = _config(
        tmp_path,
        status_won_values=(),
        status_lost_values=(),
        status_open_values=(),
        status_excluded_values=("DELETED",),
    )
    with pytest.raises(IngestionError) as excinfo:
        _register(status_csv, settings)
    assert excinfo.value.detail.reason == "empty_value_mapping"


def test_a_column_bound_to_status_without_a_mapping_is_an_error(status_csv, tmp_path):
    # Bound by an explicit override, with no configured values.
    settings = Settings(data_root=tmp_path / "data")
    with pytest.raises(IngestionError) as excinfo:
        _register(
            status_csv,
            settings,
            mapping_overrides={
                "opp_id": CanonicalColumn.OPP_ID,
                "as_of": CanonicalColumn.AS_OF,
                "close_date": CanonicalColumn.CLOSE_DATE,
                "Stage": CanonicalColumn.STAGE,
                "new_amount": CanonicalColumn.AMOUNT,
                STATUS_COLUMN: CanonicalColumn.STATUS,
            },
        )
    assert excinfo.value.detail.reason == "column_without_value_mapping"


def test_a_value_mapping_with_no_status_column_is_an_error(tmp_path):
    settings = Settings(data_root=tmp_path / "data")
    with pytest.raises(IngestionError) as excinfo:
        register_dataset(
            TINY_CSV,
            "tiny_status",
            status_mapping=StatusMapping(won_values=("W",)),
            settings=settings,
            store=DuckDBStore(settings),
        )
    assert excinfo.value.detail.reason == "mapping_without_column"


def test_status_errors_are_not_mapping_errors(status_csv, tmp_path):
    with pytest.raises(IngestionError) as excinfo:
        _register(status_csv, _config(tmp_path, status_column="does_not_exist"))
    assert not isinstance(excinfo.value, MappingError)
