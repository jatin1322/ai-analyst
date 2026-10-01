"""The dataset-scoped registry (ARCHITECTURE 5.15).

`OPPORTUNITY_SNAPSHOT_V1` describes an export *shape* and is a static reference.
A `DatasetRegistry` describes one ingested dataset: the columns it actually
contains, canonical and discovered, each with its classification. It is created
at ingestion, persisted beside the dataset, and reloadable without re-profiling.

Classification and profile are deliberately two different objects:

* the registry answers *what a column means and when it may be read*;
* the profile answers *what values actually occur*.

They are joined by column name and never merged. Nothing observed writes into a
classification field, which is what stops a profile from redefining a column.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai_analyst.contracts.columns import (
    ColumnCategory,
    ColumnClassification,
    InformationClass,
    QuarantineReason,
    UnresolvedItem,
)
from ai_analyst.contracts.schema import DataType


class ColumnOrigin(StrEnum):
    """How a column came to exist in the dataset."""

    CANONICAL = "canonical"
    DERIVED = "derived"
    DISCOVERED = "discovered"


class DatasetColumn(BaseModel):
    """One column of one dataset: structure plus classification, never observation.

    `nullable` is declared structure, not an observed fact. Only the grain
    columns are non-nullable, because ingestion hard-fails on a null grain key.
    Observed null counts live in the profile.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    origin: ColumnOrigin
    source_name: str | None = None
    dtype: DataType
    nullable: bool = True
    classification: ColumnClassification

    @model_validator(mode="after")
    def _classification_describes_this_column(self) -> DatasetColumn:
        if self.classification.name != self.name:
            raise ValueError(
                f"classification is for {self.classification.name!r}, not {self.name!r}"
            )
        return self

    @property
    def category(self) -> ColumnCategory:
        return self.classification.category

    @property
    def information_class(self) -> InformationClass:
        return self.classification.information_class

    @property
    def quarantine(self) -> QuarantineReason | None:
        return self.classification.quarantine

    @property
    def is_quarantined(self) -> bool:
        return self.classification.quarantine is not None

    @property
    def has_sentinel(self) -> bool:
        return self.classification.has_sentinel


class RequirementState(StrEnum):
    """Whether one required column can be used, and if not, why."""

    AVAILABLE = "available"
    AVAILABLE_WITH_CAVEATS = "available_with_caveats"
    QUARANTINED = "quarantined"
    NOT_KNOWABLE_AT_SNAPSHOT = "not_knowable_at_snapshot"
    MISSING = "missing"


class ColumnRequirement(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    state: RequirementState
    quarantine: QuarantineReason | None = None
    blockers: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class RequirementCheck(BaseModel):
    """The answer to "can these columns be used?", without knowing about metrics.

    A metric layer decides which metrics to hide by asking this question of the
    columns it needs. The registry itself knows nothing about metrics.
    """

    requirements: list[ColumnRequirement]

    def _named(self, state: RequirementState) -> list[str]:
        return [r.name for r in self.requirements if r.state is state]

    @property
    def missing(self) -> list[str]:
        return self._named(RequirementState.MISSING)

    @property
    def quarantined(self) -> list[str]:
        return self._named(RequirementState.QUARANTINED)

    @property
    def not_knowable(self) -> list[str]:
        return self._named(RequirementState.NOT_KNOWABLE_AT_SNAPSHOT)

    @property
    def caveated(self) -> list[str]:
        return self._named(RequirementState.AVAILABLE_WITH_CAVEATS)

    @property
    def satisfiable_prospectively(self) -> bool:
        """Every column exists, is not quarantined, and was knowable at as_of."""
        return not (self.missing or self.quarantined or self.not_knowable)

    @property
    def satisfiable_retrospectively(self) -> bool:
        """Hindsight may read contaminated columns, but never quarantined ones."""
        return not (self.missing or self.quarantined)


class DatasetRegistry(BaseModel):
    """Every column of one dataset with its classification."""

    model_config = ConfigDict(frozen=True)

    dataset_id: str
    export_registry: str | None = None
    columns: list[DatasetColumn]
    unresolved: list[UnresolvedItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_names(self) -> DatasetRegistry:
        names = [c.name for c in self.columns]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"columns registered more than once: {duplicates}")
        return self

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.columns]

    def by_name(self) -> dict[str, DatasetColumn]:
        return {c.name: c for c in self.columns}

    def has(self, name: str) -> bool:
        return name in self.by_name()

    def get(self, name: str) -> DatasetColumn:
        found = self.by_name().get(name)
        if found is None:
            raise KeyError(f"column {name!r} is not registered in dataset {self.dataset_id!r}")
        return found

    def by_origin(self, origin: ColumnOrigin) -> list[DatasetColumn]:
        return [c for c in self.columns if c.origin == origin]

    def in_information_class(self, info: InformationClass) -> list[DatasetColumn]:
        return [c for c in self.columns if c.information_class is info]

    def prospectively_usable(self) -> list[DatasetColumn]:
        return [c for c in self.columns if c.classification.usable_prospectively]

    def quarantined(self) -> list[DatasetColumn]:
        return [c for c in self.columns if c.is_quarantined]

    def text_columns(self) -> list[DatasetColumn]:
        return self.in_information_class(InformationClass.TEXT)

    def sentinel_columns(self) -> list[DatasetColumn]:
        return [c for c in self.columns if c.has_sentinel]

    def unclassified(self) -> list[DatasetColumn]:
        return self.in_information_class(InformationClass.UNCLASSIFIED)

    def unresolved_for(self, name: str) -> list[UnresolvedItem]:
        return [u for u in self.unresolved if name in u.columns and not u.resolved]

    def open_unresolved(self) -> list[UnresolvedItem]:
        return [u for u in self.unresolved if not u.resolved]

    def check_requirements(self, required: list[str]) -> RequirementCheck:
        """Report whether each named column is present, usable, or withheld."""
        results: list[ColumnRequirement] = []
        for name in required:
            column = self.by_name().get(name)
            if column is None:
                results.append(ColumnRequirement(name=name, state=RequirementState.MISSING))
                continue

            classification = column.classification
            blockers = classification.prospective_blockers

            if column.is_quarantined:
                state = RequirementState.QUARANTINED
                caveats: list[str] = []
            elif not classification.knowable_at_snapshot:
                state = RequirementState.NOT_KNOWABLE_AT_SNAPSHOT
                caveats = []
            else:
                caveats = [u.id for u in self.unresolved_for(name)]
                lineage = classification.lineage
                if lineage is not None and not lineage.confirmed:
                    caveats.append("lineage_unconfirmed")
                state = (
                    RequirementState.AVAILABLE_WITH_CAVEATS
                    if caveats
                    else RequirementState.AVAILABLE
                )

            results.append(
                ColumnRequirement(
                    name=name,
                    state=state,
                    quarantine=column.quarantine,
                    blockers=blockers,
                    caveats=caveats,
                )
            )
        return RequirementCheck(requirements=results)
