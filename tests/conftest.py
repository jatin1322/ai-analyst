"""Shared fixtures.

Every test runs against an isolated `data_root` under tmp_path so no test can
observe another's artifacts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_analyst.config import Settings, reset_settings
from ai_analyst.contracts.profile import DatasetProfile
from ai_analyst.contracts.schema import DatasetSchema
from ai_analyst.data.ingest import IngestionResult, ingest
from ai_analyst.data.profiler import profile_dataset
from ai_analyst.data.store import DuckDBStore

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "tiny"

TINY_CSV = FIXTURE_DIR / "snapshots.csv"
DUPLICATE_KEY_CSV = FIXTURE_DIR / "duplicate_key.csv"
MISSING_REQUIRED_CSV = FIXTURE_DIR / "missing_required.csv"
NULL_KEY_CSV = FIXTURE_DIR / "null_key.csv"
BAD_VALUES_CSV = FIXTURE_DIR / "bad_amount.csv"

TINY_DATASET_ID = "tiny"


@pytest.fixture(autouse=True)
def _reset_settings_singleton() -> None:
    reset_settings()
    yield
    reset_settings()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_root=tmp_path / "data")


@pytest.fixture
def store(settings: Settings) -> DuckDBStore:
    return DuckDBStore(settings)


@pytest.fixture
def ingested(settings: Settings, store: DuckDBStore) -> IngestionResult:
    return ingest(TINY_CSV, TINY_DATASET_ID, settings=settings, store=store)


@pytest.fixture
def tiny_schema(ingested: IngestionResult) -> DatasetSchema:
    return ingested.schema


@pytest.fixture
def tiny_profile(
    ingested: IngestionResult, settings: Settings, store: DuckDBStore
) -> DatasetProfile:
    return profile_dataset(
        ingested.dataset_id, ingested.schema, settings=settings, store=store
    )


# ---------------------------------------------------------------------------
# Production-shaped synthetic dataset (no real data). Built once per session
# because registering 142 columns is the slowest thing in the suite. Tests that
# use it must treat it as read-only.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def production_csv(tmp_path_factory) -> Path:
    from tests.fixtures.production_shape import write_production_csv

    path = tmp_path_factory.mktemp("production_shape") / "production.csv"
    write_production_csv(path)
    return path


@pytest.fixture(scope="session")
def production_dataset(production_csv: Path, tmp_path_factory):
    from ai_analyst.contracts.opportunity_snapshot_v1 import OPPORTUNITY_SNAPSHOT_V1
    from ai_analyst.data.dataset import register_dataset

    settings = Settings(data_root=tmp_path_factory.mktemp("production_data") / "data")
    dataset = register_dataset(
        production_csv,
        "production",
        column_registry=OPPORTUNITY_SNAPSHOT_V1,
        settings=settings,
    )
    return dataset, settings
