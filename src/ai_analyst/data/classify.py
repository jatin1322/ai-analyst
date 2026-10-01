"""Build the dataset-scoped registry for one ingested dataset (ARCHITECTURE 5.15).

Inputs are the dataset's schema and, optionally, an export registry describing
the shape of the file it came from. Nothing here reads data or infers meaning
from a column name beyond what the export registry and its family patterns
already document. A column nobody documented comes back unclassified and
quarantined, so an unrecognised column can never quietly become usable.
"""

from __future__ import annotations

from ai_analyst.contracts.columns import (
    Availability,
    ColumnCategory,
    ColumnClassification,
    ColumnRegistry,
    Disposition,
    MonetaryStatus,
    UnresolvedItem,
)
from ai_analyst.contracts.dataset import ColumnOrigin, DatasetColumn, DatasetRegistry
from ai_analyst.contracts.schema import (
    CANONICAL_COLUMNS,
    GRAIN_COLUMNS,
    CanonicalColumn,
    DatasetSchema,
    DataType,
)
from ai_analyst.contracts.status import StatusStrategy

_IDENTITY = ColumnCategory.IDENTITY
_STATE = ColumnCategory.SNAPSHOT_STATE

# The documented canonical contract (ARCHITECTURE 1.4). These are semantic
# assignments made by the contract itself, not inferred from any header.
_CANONICAL_CATEGORY: dict[CanonicalColumn, ColumnCategory] = {
    CanonicalColumn.AS_OF: _IDENTITY,
    CanonicalColumn.OPP_ID: _IDENTITY,
    CanonicalColumn.OWNER_ID: _IDENTITY,
    CanonicalColumn.ACCOUNT_ID: _IDENTITY,
    CanonicalColumn.OPP_NAME: _IDENTITY,
    CanonicalColumn.CREATED_DATE: ColumnCategory.TEMPORAL,
    CanonicalColumn.CLOSE_DATE: _STATE,
    CanonicalColumn.STAGE: _STATE,
    CanonicalColumn.AMOUNT: _STATE,
    CanonicalColumn.ARR: _STATE,
    CanonicalColumn.STATUS: _STATE,
    CanonicalColumn.IS_CLOSED: _STATE,
    CanonicalColumn.IS_WON: _STATE,
    CanonicalColumn.FORECAST_CATEGORY: _STATE,
    CanonicalColumn.SEGMENT: _STATE,
    CanonicalColumn.REGION: _STATE,
    CanonicalColumn.INDUSTRY: _STATE,
    CanonicalColumn.PROBABILITY: _STATE,
}


# Canonical columns that hold money (ARCHITECTURE 12.16). Enumerated rather than
# inferred from the storage type, because `probability` is DECIMAL too and is not
# money. Storage type never decides this.
CANONICAL_MONETARY: frozenset[CanonicalColumn] = frozenset(
    {CanonicalColumn.AMOUNT, CanonicalColumn.ARR}
)


def canonical_default(canonical: CanonicalColumn) -> ColumnClassification:
    """The contract's own classification of a canonical column."""
    spec = CANONICAL_COLUMNS[canonical]
    return ColumnClassification(
        name=canonical.value,
        category=_CANONICAL_CATEGORY[canonical],
        availability=Availability.AS_OF_FACT,
        disposition=Disposition.DIRECT,
        note=spec.description,
        requires_confirmation=False,
        monetary=(
            MonetaryStatus.MONETARY
            if canonical in CANONICAL_MONETARY
            else MonetaryStatus.NON_MONETARY
        ),
    )


def _classify_discovered(
    name: str, export_registry: ColumnRegistry | None
) -> ColumnClassification:
    """Classify a discovered column from the export registry, else fail closed."""
    if export_registry is None:
        return ColumnRegistry(name="none", columns=[]).classify_unknown(name)
    return export_registry.classify(name)


def _status_unresolved(schema: DatasetSchema) -> list[UnresolvedItem]:
    """Make the state of the authoritative status source visible as data."""
    resolution = schema.status_resolution
    derived = ("status", "is_closed", "is_won")
    if resolution is None or not resolution.is_authoritative:
        return [
            UnresolvedItem(
                id="authoritative_status",
                subject="Authoritative opportunity status source",
                question=(
                    "Which column is the authoritative source of open, won, and lost, "
                    "and how do its values map to open, won, lost, and excluded? None "
                    "is configured, so status falls back to stage keyword inference."
                ),
                columns=derived,
                blocks=(
                    "Trusting any won, lost, or open count. Stage inference is not "
                    "authoritative: it reads '6 - Order Placed' as open and cannot "
                    "read an outcome encoded in a label."
                ),
            )
        ]
    if resolution.strategy is StatusStrategy.DECLARED_STAGE_MAP:
        if not (resolution.unmapped_values or resolution.unmapped_rows):
            return []
        return [
            UnresolvedItem(
                id="status_unmapped_values",
                subject="Stage values with no declared status",
                question=(
                    f"{resolution.unmapped_rows} rows carry stage values that are not in "
                    f"the declared stage-to-status map: {resolution.unmapped_values}. "
                    "They are EXCLUDED, never counted as open pipeline."
                ),
                columns=derived,
                blocks="Counting those rows as open, won, or lost.",
            )
        ]
    if resolution.unmapped_values:
        return [
            UnresolvedItem(
                id="status_unmapped_values",
                subject="Status values with no mapping",
                question=(
                    "The authoritative status column contains values that are not in the "
                    f"configured mapping: {resolution.unmapped_values}. They resolve to "
                    "'unknown' rather than being guessed."
                ),
                columns=derived,
                blocks="Counting those rows as open, won, or lost.",
            )
        ]
    return []


def build_dataset_registry(
    schema: DatasetSchema, export_registry: ColumnRegistry | None = None
) -> DatasetRegistry:
    """Create the registry for one ingested dataset."""
    by_canonical = schema.mapping.by_canonical()
    columns: list[DatasetColumn] = []

    for spec in schema.columns:
        canonical = spec.name
        mapped = by_canonical.get(canonical)
        source = mapped.source_column if mapped else None

        exported = (
            export_registry.by_name().get(source)
            if export_registry is not None and source is not None
            else None
        )
        classification = (
            exported.model_copy(update={"name": canonical.value})
            if exported is not None
            else canonical_default(canonical)
        )
        columns.append(
            DatasetColumn(
                name=canonical.value,
                origin=ColumnOrigin.CANONICAL if mapped else ColumnOrigin.DERIVED,
                source_name=source,
                dtype=spec.dtype,
                nullable=canonical not in GRAIN_COLUMNS,
                classification=classification,
            )
        )

    for name in schema.discovered_columns:
        columns.append(
            DatasetColumn(
                name=name,
                origin=ColumnOrigin.DISCOVERED,
                source_name=name,
                dtype=schema.discovered_types.get(name, DataType.VARCHAR),
                classification=_classify_discovered(name, export_registry),
            )
        )

    present = {c.name for c in columns}
    exported_unresolved = [
        item
        for item in (export_registry.unresolved if export_registry else [])
        if any(col in present for col in item.columns)
    ]

    return DatasetRegistry(
        dataset_id=schema.dataset_id,
        export_registry=export_registry.name if export_registry else None,
        columns=columns,
        unresolved=[*exported_unresolved, *_status_unresolved(schema)],
    )
