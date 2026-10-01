"""Persistence, classification/profile separation, and fail-closed behaviour."""

from __future__ import annotations

import hashlib
import json

import duckdb
import pytest

from ai_analyst.config import Settings
from ai_analyst.contracts.columns import (
    Availability,
    ColumnCategory,
    ColumnClassification,
    Disposition,
    InformationClass,
    QuarantineCode,
)
from ai_analyst.contracts.dataset import ColumnOrigin
from ai_analyst.contracts.errors import GrainViolation
from ai_analyst.contracts.opportunity_snapshot_v1 import OPPORTUNITY_SNAPSHOT_V1
from ai_analyst.contracts.profile import ProfileKind
from ai_analyst.contracts.schema import CanonicalColumn, DataType
from ai_analyst.data.column_profiler import profile_column
from ai_analyst.data.dataset import (
    DatasetInconsistentError,
    load_dataset,
    register_dataset,
)
from ai_analyst.data.ingest import IngestionError, ingest
from ai_analyst.data.profiler import profile_dataset
from ai_analyst.data.store import DuckDBStore
from tests.conftest import TINY_CSV
from tests.fixtures.production_shape import write_production_csv

REQUIRED_OVERRIDES = {
    "opp_id": CanonicalColumn.OPP_ID,
    "as_of": CanonicalColumn.AS_OF,
    "close_date": CanonicalColumn.CLOSE_DATE,
    "Stage": CanonicalColumn.STAGE,
    "new_amount": CanonicalColumn.AMOUNT,
}


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _register(csv, settings, dataset_id="ds", **kwargs):
    return register_dataset(
        csv,
        dataset_id,
        column_registry=kwargs.pop("column_registry", OPPORTUNITY_SNAPSHOT_V1),
        settings=settings,
        store=DuckDBStore(settings),
        **kwargs,
    )


# --- 8 and 10. persistence --------------------------------------------------


def test_schema_classifications_and_profile_are_persisted_beside_the_dataset(
    production_dataset,
):
    dataset, settings = production_dataset
    directory = settings.dataset_dir(dataset.dataset_id)
    assert settings.schema_path(dataset.dataset_id).exists()
    assert settings.classifications_path(dataset.dataset_id).exists()
    assert settings.profile_path(dataset.dataset_id).exists()
    assert (directory / "canonical" / "snapshots").exists()


def test_metadata_survives_save_and_load(production_dataset):
    dataset, settings = production_dataset
    reloaded = load_dataset(dataset.dataset_id, settings)
    assert reloaded.schema == dataset.schema
    assert reloaded.registry == dataset.registry
    assert reloaded.profile == dataset.profile


def test_loading_never_reruns_the_profiler(production_dataset, monkeypatch):
    dataset, settings = production_dataset

    def boom(*args, **kwargs):
        raise AssertionError("the profiler must not run when loading a dataset")

    monkeypatch.setattr("ai_analyst.data.dataset.profile_dataset", boom)
    monkeypatch.setattr("ai_analyst.data.profiler.profile_dataset", boom)
    reloaded = load_dataset(dataset.dataset_id, settings)
    assert reloaded.is_profiled
    assert reloaded.registry.get("account_ti_first_won").has_sentinel


def test_quarantine_reasons_and_unresolved_items_survive_reload(production_dataset):
    dataset, settings = production_dataset
    reloaded = load_dataset(dataset.dataset_id, settings)
    train = reloaded.registry.get("train")
    assert train.quarantine.code is QuarantineCode.NON_ANALYTIC_METADATA
    assert {u.id for u in reloaded.registry.open_unresolved()} == {
        "opp_commit_to_close_days_definition",
        "rep_aggregate_lineage",
        "authoritative_status",
        "days_to_close_edge_cases",
        "eoq_close_diff_convention",
        "fiscal_year_start_month",
        "date_encoding_declarations",
        "quarter_boundary_day_convention",
    }


def test_a_dataset_is_loadable_before_it_is_profiled(tmp_path):
    settings = Settings(data_root=tmp_path / "data")
    csv = tmp_path / "p.csv"
    write_production_csv(csv)
    ingest(
        csv,
        "unprofiled",
        column_registry=OPPORTUNITY_SNAPSHOT_V1,
        settings=settings,
        store=DuckDBStore(settings),
    )
    dataset = load_dataset("unprofiled", settings)
    assert not dataset.is_profiled
    assert dataset.registry.has("rep_win_rate")


def test_reingestion_discards_the_stale_profile(tmp_path):
    settings = Settings(data_root=tmp_path / "data")
    csv = tmp_path / "p.csv"
    write_production_csv(csv)
    _register(csv, settings)
    assert settings.profile_path("ds").exists()

    ingest(
        csv,
        "ds",
        column_registry=OPPORTUNITY_SNAPSHOT_V1,
        settings=settings,
        store=DuckDBStore(settings),
    )
    assert not settings.profile_path("ds").exists()
    assert not load_dataset("ds", settings).is_profiled


def test_a_profile_that_no_longer_matches_the_registry_is_refused(tmp_path):
    settings = Settings(data_root=tmp_path / "data")
    csv = tmp_path / "p.csv"
    write_production_csv(csv)
    _register(csv, settings)

    path = settings.profile_path("ds")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["columns"] = [c for c in payload["columns"] if c["name"] != "rep_win_rate"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DatasetInconsistentError) as excinfo:
        load_dataset("ds", settings)
    assert excinfo.value.detail.only_in_registry == ["rep_win_rate"]


# --- 3. classification and profile must not drift ---------------------------


def test_profiling_never_modifies_the_persisted_classification(tmp_path):
    settings = Settings(data_root=tmp_path / "data")
    csv = tmp_path / "p.csv"
    write_production_csv(csv)
    store = DuckDBStore(settings)
    result = ingest(
        csv, "ds", column_registry=OPPORTUNITY_SNAPSHOT_V1, settings=settings, store=store
    )
    before_file = _sha(settings.classifications_path("ds"))
    before_registry = result.registry.model_dump_json()

    profile_dataset("ds", result.schema, registry=result.registry, settings=settings, store=store)

    assert _sha(settings.classifications_path("ds")) == before_file
    assert result.registry.model_dump_json() == before_registry


def test_an_observed_text_shape_does_not_reclassify_a_column(tmp_path):
    # With no export registry, ManagerNotes is unclassified. Its values look
    # like text, and the profile says so, but the registry must not follow.
    settings = Settings(data_root=tmp_path / "data")
    csv = tmp_path / "p.csv"
    write_production_csv(csv)
    dataset = _register(csv, settings, column_registry=None, mapping_overrides=REQUIRED_OVERRIDES)

    column = dataset.registry.get("ManagerNotes")
    assert column.information_class is InformationClass.UNCLASSIFIED
    assert not column.classification.classified

    observed = dataset.profile.column("ManagerNotes")
    assert observed.kind is ProfileKind.TEXT
    catalogue = {t.name: t for t in dataset.profile.text_columns}
    assert catalogue["ManagerNotes"].basis == "detected"


def test_a_declared_category_is_not_inferred_from_a_column_name(tmp_path):
    # `rep_win_rate` looks like a rep feature, but with no registry nothing may
    # say so. The name alone classifies nothing.
    settings = Settings(data_root=tmp_path / "data")
    csv = tmp_path / "p.csv"
    write_production_csv(csv)
    dataset = _register(csv, settings, column_registry=None, mapping_overrides=REQUIRED_OVERRIDES)
    column = dataset.registry.get("rep_win_rate")
    assert column.information_class is InformationClass.UNCLASSIFIED
    assert column.classification.family is None


# --- fail closed on an unrecognised dataset ---------------------------------


def test_every_discovered_column_of_an_unrecognised_dataset_fails_closed(tmp_path):
    settings = Settings(data_root=tmp_path / "data")
    csv = tmp_path / "p.csv"
    write_production_csv(csv)
    dataset = _register(csv, settings, column_registry=None, mapping_overrides=REQUIRED_OVERRIDES)
    registry = dataset.registry

    discovered = registry.by_origin(ColumnOrigin.DISCOVERED)
    assert len(discovered) > 100
    assert registry.export_registry is None
    for column in discovered:
        assert column.quarantine.code is QuarantineCode.UNCLASSIFIED_COLUMN, column.name
        assert not column.classification.usable_prospectively
        assert column.information_class is InformationClass.UNCLASSIFIED
    # Only canonical and derived columns remain usable.
    usable = {c.name for c in registry.prospectively_usable()}
    assert usable and not (usable & {c.name for c in discovered})
    assert dataset.check_requirements(["rep_win_rate"]).quarantined == ["rep_win_rate"]


def test_the_tiny_fixture_gets_the_canonical_contract_classification(tmp_path):
    settings = Settings(data_root=tmp_path / "data")
    dataset = register_dataset(
        TINY_CSV, "tiny", settings=settings, store=DuckDBStore(settings)
    )
    registry = dataset.registry
    assert registry.export_registry is None
    assert registry.by_origin(ColumnOrigin.DISCOVERED) == []
    assert len(registry.columns) > 0
    assert registry.get("as_of").information_class is InformationClass.IDENTITY
    assert registry.get("amount").classification.availability is Availability.AS_OF_FACT
    assert registry.get("close_date").classification.disposition is Disposition.DIRECT
    assert all(not c.is_quarantined for c in registry.columns)
    assert {u.id for u in registry.open_unresolved()} == {"authoritative_status"}


# --- 11. grain validation still holds through the new path ------------------


def test_a_duplicate_grain_key_still_hard_fails_and_persists_nothing(tmp_path):
    settings = Settings(data_root=tmp_path / "data")
    csv = tmp_path / "dup.csv"
    write_production_csv(csv, duplicate_first_row=True)
    with pytest.raises(IngestionError) as excinfo:
        _register(csv, settings, dataset_id="dup")

    detail = excinfo.value.detail
    assert isinstance(detail, GrainViolation)
    assert detail.duplicate_key_count == 1
    assert detail.samples[0].opp_id == "OPP-000"
    assert not settings.classifications_path("dup").exists()
    assert not settings.schema_path("dup").exists()


# --- typed sources ----------------------------------------------------------


def test_a_typed_parquet_source_yields_the_same_types_as_the_csv(
    production_dataset, production_csv, tmp_path
):
    csv_dataset, _ = production_dataset
    parquet = tmp_path / "typed.parquet"
    conn = duckdb.connect()
    conn.execute(
        f"COPY (SELECT * FROM read_csv('{production_csv}', header = true, sample_size = -1)) "
        f"TO '{parquet}' (FORMAT PARQUET)"
    )
    conn.close()

    settings = Settings(data_root=tmp_path / "data")
    from_parquet = _register(parquet, settings, dataset_id="typed")

    assert from_parquet.schema.discovered_types == csv_dataset.schema.discovered_types
    assert from_parquet.profile.column("amount").min_value == csv_dataset.profile.column(
        "amount"
    ).min_value
    assert from_parquet.profile.column("account_ti_first_won").sentinel_count == (
        csv_dataset.profile.column("account_ti_first_won").sentinel_count
    )


# --- compactness ------------------------------------------------------------


def test_high_cardinality_columns_do_not_list_top_values(tmp_path):
    settings = Settings(data_root=tmp_path / "data", profile_max_top_value_cardinality=5)
    csv = tmp_path / "p.csv"
    write_production_csv(csv)
    dataset = _register(csv, settings)

    stage = dataset.profile.column("stage")
    assert stage.distinct_count > 5
    assert stage.high_cardinality
    assert stage.top_values == []

    type_column = dataset.profile.column("Type")
    assert not type_column.high_cardinality
    assert len(type_column.top_values) == 3


# --- the column view keeps the two concepts apart ---------------------------


def test_a_column_view_holds_classification_and_profile_separately(production_dataset):
    dataset, _ = production_dataset
    view = dataset.column("account_ti_first_won")
    assert view.column.classification.sentinels == (-999999.0,)
    assert view.profile.sentinel_count > 0
    assert view.profile.name == view.column.name
    assert not hasattr(view.profile, "classification")
    assert not hasattr(view.column, "mean")


# --- sentinel edge cases ----------------------------------------------------


def _numeric_classification(name: str, sentinels: tuple[float, ...]) -> ColumnClassification:
    return ColumnClassification(
        name=name,
        category=ColumnCategory.ACCOUNT_FEATURE,
        availability=Availability.BACKWARD_DERIVED,
        disposition=Disposition.USE_WITH_PROOF,
        sentinels=sentinels,
    )


def test_a_column_of_only_sentinels_has_no_observed_statistics(settings):
    conn = duckdb.connect()
    conn.execute("CREATE TABLE t AS SELECT * FROM (VALUES (-999999), (-999999), (-999999)) v(x)")
    profile, _ = profile_column(
        conn, "t", "x", DataType.BIGINT, 3, _numeric_classification("x", (-999999.0,)), settings
    )
    assert profile.sentinel_count == 3
    assert profile.observed_count == 0
    assert profile.null_count == 0
    assert profile.min_value is None
    assert profile.max_value is None
    assert profile.mean is None


def test_nulls_and_sentinels_are_counted_separately(settings):
    conn = duckdb.connect()
    conn.execute("CREATE TABLE t AS SELECT * FROM (VALUES (-999999), (NULL), (4), (6)) v(x)")
    profile, _ = profile_column(
        conn, "t", "x", DataType.BIGINT, 4, _numeric_classification("x", (-999999.0,)), settings
    )
    assert (profile.null_count, profile.sentinel_count, profile.observed_count) == (1, 1, 2)
    assert profile.mean == 5.0


def test_without_a_declaration_the_same_value_is_ordinary(settings):
    conn = duckdb.connect()
    conn.execute("CREATE TABLE t AS SELECT * FROM (VALUES (-999999), (4), (6)) v(x)")
    profile, _ = profile_column(conn, "t", "x", DataType.BIGINT, 3, None, settings)
    assert profile.sentinel_count == 0
    assert profile.min_value == "-999999"
