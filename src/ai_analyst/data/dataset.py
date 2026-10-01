"""A dataset as one object: schema, classifications, profile (ARCHITECTURE 5.15).

    Dataset
      |- schema           how the file mapped onto the canonical contract
      |- registry         what every column means and when it may be read
      `- profile          what values actually occur

The three are persisted as separate files and are only ever joined by column
name. Loading reconstructs them from disk and never re-runs the profiler.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ai_analyst.config import Settings, get_settings
from ai_analyst.contracts.agreement import AgreementReport
from ai_analyst.contracts.columns import ColumnRegistry
from ai_analyst.contracts.dataset import DatasetColumn, DatasetRegistry, RequirementCheck
from ai_analyst.contracts.errors import (
    DatasetInconsistent,
    IngestionError,
    UnreadableSource,
)
from ai_analyst.contracts.profile import ColumnProfile, DatasetProfile
from ai_analyst.contracts.schema import CanonicalColumn, DatasetSchema, DateEncoding
from ai_analyst.contracts.source import TableSource
from ai_analyst.contracts.status import StatusMapping
from ai_analyst.contracts.tenant import TenantProfile
from ai_analyst.data.grants import discard_grants, issue_grants, save_grants
from ai_analyst.data.ingest import ingest, load_schema
from ai_analyst.data.money import declared_monetary_columns
from ai_analyst.data.profiler import load_profile, profile_dataset
from ai_analyst.data.store import DuckDBStore


@dataclass(frozen=True)
class ColumnView:
    """One column's classification beside its observations, kept separate."""

    column: DatasetColumn
    profile: ColumnProfile | None


class DatasetInconsistentError(IngestionError):
    """Persisted metadata no longer describes one dataset."""


@dataclass(frozen=True)
class Dataset:
    dataset_id: str
    schema: DatasetSchema
    registry: DatasetRegistry
    profile: DatasetProfile | None = None
    # Agreement tests over reconstructed columns, persisted at ingestion so a
    # failure stays visible in the dataset's metadata. None when nothing was
    # reconstructed.
    agreement: AgreementReport | None = None

    def __post_init__(self) -> None:
        if self.profile is not None:
            verify_consistent(self.registry, self.profile)

    @property
    def is_profiled(self) -> bool:
        return self.profile is not None

    @property
    def status_is_authoritative(self) -> bool:
        return self.schema.status_is_authoritative

    def column(self, name: str) -> ColumnView:
        observed = None
        if self.profile is not None:
            observed = next((c for c in self.profile.columns if c.name == name), None)
        return ColumnView(column=self.registry.get(name), profile=observed)

    def check_requirements(self, required: list[str]) -> RequirementCheck:
        return self.registry.check_requirements(required)

    def canonical(self, column: CanonicalColumn) -> DatasetColumn:
        return self.registry.get(column.value)


def verify_consistent(registry: DatasetRegistry, profile: DatasetProfile) -> None:
    """Refuse a profile that does not describe the columns the registry lists."""
    in_registry, in_profile = set(registry.names), set(profile.column_names)
    if in_registry != in_profile:
        raise DatasetInconsistentError(
            DatasetInconsistent(
                message=(
                    "the persisted profile and the column registry describe different "
                    "columns. Re-profile the dataset."
                ),
                dataset_id=registry.dataset_id,
                only_in_registry=sorted(in_registry - in_profile),
                only_in_profile=sorted(in_profile - in_registry),
            )
        )


def load_registry(dataset_id: str, settings: Settings | None = None) -> DatasetRegistry:
    settings = settings or get_settings()
    path = settings.classifications_path(dataset_id)
    if not path.exists():
        raise IngestionError(
            UnreadableSource(
                message=f"no column registry persisted for dataset {dataset_id!r}",
                path=str(path),
                reason="not_found",
            )
        )
    return DatasetRegistry.model_validate_json(path.read_text(encoding="utf-8"))


def load_agreement(dataset_id: str, settings: Settings | None = None) -> AgreementReport | None:
    """The persisted agreement report, or None when nothing was reconstructed."""
    settings = settings or get_settings()
    path = settings.agreement_path(dataset_id)
    if not path.exists():
        return None
    return AgreementReport.model_validate_json(path.read_text(encoding="utf-8"))


def load_dataset(dataset_id: str, settings: Settings | None = None) -> Dataset:
    """Reconstruct a dataset's metadata from disk, without profiling anything."""
    settings = settings or get_settings()
    profile = (
        load_profile(dataset_id, settings) if settings.profile_path(dataset_id).exists() else None
    )
    return Dataset(
        dataset_id=dataset_id,
        schema=load_schema(dataset_id, settings),
        registry=load_registry(dataset_id, settings),
        profile=profile,
        agreement=load_agreement(dataset_id, settings),
    )


def register_dataset(
    source_path: str | Path | TableSource,
    dataset_id: str,
    *,
    column_registry: ColumnRegistry | None = None,
    mapping_overrides: dict[str, CanonicalColumn] | None = None,
    status_mapping: StatusMapping | None = None,
    date_encodings: dict[str, DateEncoding] | None = None,
    tenant: TenantProfile | None = None,
    settings: Settings | None = None,
    store: DuckDBStore | None = None,
) -> Dataset:
    """Ingest, classify, and profile a source (a file or a `TableSource`) into a
    complete dataset."""
    settings = settings or get_settings()
    store = store or DuckDBStore(settings)
    result = ingest(
        source_path,
        dataset_id,
        mapping_overrides=mapping_overrides,
        status_mapping=status_mapping,
        column_registry=column_registry,
        date_encodings=date_encodings,
        snapshot_policy=tenant.snapshot_policy if tenant is not None else None,
        stage_status_map=tenant.stage_status_map if tenant is not None else None,
        family_invariants=list(tenant.family_invariants) if tenant is not None else None,
        settings=settings,
        store=store,
    )
    profile = profile_dataset(
        dataset_id,
        result.schema,
        registry=result.registry,
        settings=settings,
        store=store,
        declared_monetary=declared_monetary_columns(tenant, result.schema),
    )
    # Grants are issued once, here, where the tenant's declarations meet the
    # dataset, and persisted. The runtime loads them; it never rebuilds them.
    # Re-registering without a tenant discards any ledger for the old data.
    if tenant is not None:
        save_grants(issue_grants(dataset_id, result.registry, tenant), settings)
    else:
        discard_grants(dataset_id, settings)
    return Dataset(
        dataset_id=dataset_id,
        schema=result.schema,
        registry=result.registry,
        profile=profile,
        agreement=result.agreement,
    )
