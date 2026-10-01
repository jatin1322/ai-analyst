"""Column classification: category, availability, and trust disposition.

Implements ARCHITECTURE 5.7 to 5.10 for a real opportunity-snapshot export.

Two orthogonal axes:

* **Availability** says *when* a value is knowable, and gates prospective
  analysis (5.8). `UNKNOWN` is treated exactly like `FUTURE_CONTAMINATED`, so
  the system fails closed on undocumented features.
* **Disposition** says *how much to trust* a value, and decides whether the
  semantic layer reads it or recomputes it (5.9).

Every classification in `OPPORTUNITY_SNAPSHOT_V1` was assigned from the column
name alone. Not one row of this dataset has been inspected, so every entry
carries `requires_confirmation=True` unless it is structurally certain, such as
the grain columns. This registry is a proposal to be verified, not a finding.
"""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai_analyst.contracts.schema import DateEncoding


class ColumnCategory(StrEnum):
    """What a column is (ARCHITECTURE 5.7)."""

    IDENTITY = "identity"
    SNAPSHOT_STATE = "snapshot_state"
    OUTCOME = "outcome"
    TEMPORAL = "temporal"
    QUARTER = "quarter"
    DERIVED_TEMPORAL = "derived_temporal"
    HISTORICAL_FEATURE = "historical_feature"
    REP_FEATURE = "rep_feature"
    ACCOUNT_FEATURE = "account_feature"
    DEAL_FEATURE = "deal_feature"
    TEXT = "text"
    METADATA = "metadata"


class Availability(StrEnum):
    """When a value is knowable (ARCHITECTURE 5.8)."""

    AS_OF_FACT = "as_of_fact"
    BACKWARD_DERIVED = "backward_derived"
    FUTURE_CONTAMINATED = "future_contaminated"
    UNKNOWN = "unknown"

    @property
    def safe_for_prospective(self) -> bool:
        """Unknown fails closed, exactly like contaminated."""
        return self in (Availability.AS_OF_FACT, Availability.BACKWARD_DERIVED)


class Disposition(StrEnum):
    """How much the semantic layer trusts a value (ARCHITECTURE 5.9)."""

    DIRECT = "direct"
    RECOMPUTE = "recompute"
    USE_WITH_PROOF = "use_with_proof"
    QUARANTINE = "quarantine"


class InformationClass(StrEnum):
    """The headline grouping the project asked to be made explicit.

    Derived from category and availability rather than stored, so the two can
    never drift apart.
    """

    IDENTITY = "identity"
    SNAPSHOT_STATE = "snapshot_state"
    HISTORICAL_FEATURE = "historical_feature"
    RETROSPECTIVE_OUTCOME = "retrospective_outcome"
    TEMPORAL_CONTEXT = "temporal_context"
    TEXT = "text"
    METADATA = "metadata"
    UNCLASSIFIED = "unclassified"


class FeatureLineage(BaseModel):
    """How a precomputed feature was produced.

    A feature without a documented lineage cannot be shown to be free of
    look-ahead leakage, so `confirmed` stays false until someone documents it.
    """

    model_config = ConfigDict(frozen=True)

    description: str = ""
    lookback: str = ""
    frozen_through: str = ""
    confirmed: bool = False


class MonetaryStatus(StrEnum):
    """Whether a column holds money (ARCHITECTURE 12.16).

    Money is decided **semantically**, from a classification or a concept
    binding, and never from numeric shape. Two decimal places do not make a
    column money; a ratio has two decimal places too.

    The three values differ in what profiling may report:

    * `MONETARY`: exact extremes only. No float mean, median, or deviation.
    * `NON_MONETARY`: full summary statistics. Safe, because it is not money.
    * `UNKNOWN`: no summary statistics either. Not because it is money, but
      because nobody has established that it is not, and a float mean over an
      unrecognized amount column is exactly the silent error this avoids.
    """

    MONETARY = "monetary"
    NON_MONETARY = "non_monetary"
    UNKNOWN = "unknown"

    @property
    def allows_float_summary(self) -> bool:
        return self is MonetaryStatus.NON_MONETARY


class QuarantineCode(StrEnum):
    """Why a column is withheld from analysis (ARCHITECTURE 5.15).

    A code rather than prose, so a later layer can enumerate, group, and report
    quarantined columns without parsing sentences.
    """

    UNSTATED_REFERENCE_POPULATION = "unstated_reference_population"
    UNSTATED_WINDOW = "unstated_window"
    UNDEFINED_COMPOSITE = "undefined_composite"
    AMBIGUOUS_DEFINITION = "ambiguous_definition"
    UNDOCUMENTED_LINEAGE = "undocumented_lineage"
    NON_ANALYTIC_METADATA = "non_analytic_metadata"
    UNCLASSIFIED_COLUMN = "unclassified_column"


class QuarantineReason(BaseModel):
    """The explicit reason one column is quarantined, and what would release it."""

    model_config = ConfigDict(frozen=True)

    code: QuarantineCode
    detail: str
    resolution: str


class UnresolvedItem(BaseModel):
    """A documented open question that affects how columns may be used.

    Recorded as data so a later layer can enumerate what is still unknown,
    rather than discovering it in prose. Nothing here is assumed; an unresolved
    item stays unresolved until someone answers it.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    subject: str
    question: str
    columns: tuple[str, ...] = ()
    blocks: str = ""
    resolved: bool = False


class ColumnClassification(BaseModel):
    """One column's proposed classification."""

    model_config = ConfigDict(frozen=True)

    name: str
    category: ColumnCategory
    availability: Availability
    disposition: Disposition
    recomputable: bool = False
    note: str = ""
    requires_confirmation: bool = True
    family: str | None = None
    # Values that encode "no value" and would corrupt any aggregate they enter.
    sentinels: tuple[float, ...] = ()
    lineage: FeatureLineage | None = None
    # A stated assertion that would prove this column does not look ahead.
    # Prose today; no executor stands behind it yet.
    leakage_check: str | None = None
    # False only for a column no registry or family could place.
    classified: bool = True
    # Declared monetary status. Left unset, a *classified* column is taken as
    # non-monetary: whoever classified it would have said so otherwise. An
    # unclassified column is always UNKNOWN, whatever this field holds, because
    # nothing about it has been established. See `monetary_status`.
    monetary: MonetaryStatus | None = None
    # Present exactly when disposition is QUARANTINE.
    quarantine: QuarantineReason | None = None

    @property
    def information_class(self) -> InformationClass:
        """The headline grouping, derived from the category.

        This answers *what a column is*, not *when it is safe to read*.
        Availability answers the second question and `usable_prospectively` is
        the gate. Keeping them apart means a column with a settled category but
        an undocumented window still reports what it is.

        The one place availability participates is `derived_temporal`, which
        genuinely spans two kinds: values computable from `as_of` and the
        calendar alone are temporal context, while values needing the
        opportunity's prior history are historical features.
        """
        if not self.classified:
            return InformationClass.UNCLASSIFIED
        match self.category:
            case ColumnCategory.IDENTITY:
                return InformationClass.IDENTITY
            case ColumnCategory.METADATA:
                return InformationClass.METADATA
            case ColumnCategory.TEXT:
                return InformationClass.TEXT
            case ColumnCategory.OUTCOME:
                return InformationClass.RETROSPECTIVE_OUTCOME
            case ColumnCategory.SNAPSHOT_STATE | ColumnCategory.DEAL_FEATURE:
                return InformationClass.SNAPSHOT_STATE
            case ColumnCategory.TEMPORAL | ColumnCategory.QUARTER:
                return InformationClass.TEMPORAL_CONTEXT
            case ColumnCategory.DERIVED_TEMPORAL:
                return (
                    InformationClass.TEMPORAL_CONTEXT
                    if self.availability is Availability.AS_OF_FACT
                    else InformationClass.HISTORICAL_FEATURE
                )
            case _:
                return InformationClass.HISTORICAL_FEATURE

    @property
    def has_sentinel(self) -> bool:
        return bool(self.sentinels)

    @property
    def monetary_status(self) -> MonetaryStatus:
        """The effective monetary status of this column.

        Fails closed: a column nothing could classify is UNKNOWN, so its values
        never reach a float summary statistic.
        """
        if not self.classified:
            return MonetaryStatus.UNKNOWN
        return self.monetary or MonetaryStatus.NON_MONETARY

    @property
    def is_monetary(self) -> bool:
        return self.monetary_status is MonetaryStatus.MONETARY

    @property
    def usable_prospectively(self) -> bool:
        return (
            self.availability.safe_for_prospective
            and self.disposition is not Disposition.QUARANTINE
        )

    @property
    def knowable_at_snapshot(self) -> bool:
        """Whether the value could have been known at its own `as_of`.

        Independent of quarantine: a quarantined column may still be believed to
        be backward-looking, but is withheld for an unrelated reason.
        """
        return self.availability.safe_for_prospective

    @property
    def prospective_blockers(self) -> list[str]:
        """Every reason this column is not usable in a prospective analysis."""
        blockers: list[str] = []
        if self.quarantine is not None:
            blockers.append(f"quarantined:{self.quarantine.code.value}")
        if not self.availability.safe_for_prospective:
            blockers.append(f"availability:{self.availability.value}")
        return blockers

    @model_validator(mode="after")
    def _quarantine_reason_matches_disposition(self) -> ColumnClassification:
        quarantined = self.disposition is Disposition.QUARANTINE
        if quarantined and self.quarantine is None:
            raise ValueError(
                f"{self.name}: a quarantined column must state its quarantine reason"
            )
        if not quarantined and self.quarantine is not None:
            raise ValueError(
                f"{self.name}: a quarantine reason is only valid on a QUARANTINE column"
            )
        return self

    @model_validator(mode="after")
    def _contaminated_is_never_direct(self) -> ColumnClassification:
        unsafe = not self.availability.safe_for_prospective
        if unsafe and self.disposition is Disposition.DIRECT:
            raise ValueError(
                f"{self.name}: a column that is {self.availability} cannot be "
                "dispositioned DIRECT; it must be recomputed, proven, or quarantined"
            )
        return self


class ColumnFamily(BaseModel):
    """A pattern-matched group of columns sharing one classification (5.10)."""

    model_config = ConfigDict(frozen=True)

    name: str
    pattern: str
    category: ColumnCategory
    availability: Availability
    disposition: Disposition
    recomputable: bool = False
    note: str = ""
    lineage: FeatureLineage | None = None
    sentinels: tuple[float, ...] = ()
    quarantine: QuarantineReason | None = None
    monetary: MonetaryStatus | None = None

    def matches(self, column: str) -> bool:
        return re.match(self.pattern, column) is not None

    def classify(self, column: str) -> ColumnClassification:
        return ColumnClassification(
            name=column,
            category=self.category,
            availability=self.availability,
            disposition=self.disposition,
            recomputable=self.recomputable,
            note=self.note,
            family=self.name,
            lineage=self.lineage,
            sentinels=self.sentinels,
            quarantine=self.quarantine,
            monetary=self.monetary,
        )


class CoverageStatus(StrEnum):
    MAPPED = "mapped"
    RECONSTRUCTIBLE = "reconstructible"
    DERIVABLE = "derivable"
    ABSENT = "absent"


class CanonicalCoverage(BaseModel):
    """Whether one canonical field is available in a given export.

    The canonical schema stays the cross-dataset contract. This record says how
    a particular export satisfies it, rather than redefining it per dataset.
    """

    model_config = ConfigDict(frozen=True)

    canonical: str
    status: CoverageStatus
    source: str | None = None
    derivation: str | None = None
    note: str = ""
    verified: bool = False


class ColumnRegistry(BaseModel):
    """A complete proposed classification for one export shape."""

    name: str
    columns: list[ColumnClassification]
    families: list[ColumnFamily] = Field(default_factory=list)
    coverage: list[CanonicalCoverage] = Field(default_factory=list)
    unresolved: list[UnresolvedItem] = Field(default_factory=list)
    # Columns that hold money, named explicitly (ARCHITECTURE 12.16). A family
    # cannot carry this, because families mix: `terminal_amount` is money and
    # `terminal_fate` is not, though both match the same pattern.
    monetary_columns: frozenset[str] = Field(default_factory=frozenset)
    # Source columns whose dates are stored in a declared non-ISO form, such as
    # Excel serial numbers (ARCHITECTURE 12.19). Keyed by *source* header,
    # because conformance happens before any renaming. Declared, never detected.
    date_encodings: dict[str, DateEncoding] = Field(default_factory=dict)

    def by_name(self) -> dict[str, ColumnClassification]:
        return {c.name: c for c in self.columns}

    def get(self, name: str) -> ColumnClassification:
        found = self.by_name().get(name)
        if found is None:
            raise KeyError(f"column {name!r} is not classified in registry {self.name!r}")
        return found

    def in_category(self, category: ColumnCategory) -> list[ColumnClassification]:
        return [c for c in self.columns if c.category is category]

    def prospectively_usable(self) -> list[ColumnClassification]:
        """The only columns a prospective analysis may read (5.8)."""
        return [c for c in self.columns if c.usable_prospectively]

    def quarantined(self) -> list[ColumnClassification]:
        return [c for c in self.columns if c.disposition is Disposition.QUARANTINE]

    def needing_recomputation(self) -> list[ColumnClassification]:
        return [c for c in self.columns if c.disposition is Disposition.RECOMPUTE]

    def unconfirmed(self) -> list[ColumnClassification]:
        return [c for c in self.columns if c.requires_confirmation]

    def in_information_class(self, info: InformationClass) -> list[ColumnClassification]:
        return [c for c in self.columns if c.information_class is info]

    def declared_monetary(self) -> list[ColumnClassification]:
        """Columns declared to hold money (ARCHITECTURE 12.16)."""
        return [self.classify(c.name) for c in self.columns if self.classify(c.name).is_monetary]

    def with_sentinels(self) -> list[ColumnClassification]:
        """Columns whose sentinel values must be masked before aggregation."""
        return [c for c in self.columns if c.has_sentinel]

    def needing_leakage_check(self) -> list[ColumnClassification]:
        return [c for c in self.columns if c.leakage_check]

    def unconfirmed_lineage(self) -> list[ColumnClassification]:
        return [c for c in self.columns if c.lineage and not c.lineage.confirmed]

    def _with_monetary(self, classification: ColumnClassification) -> ColumnClassification:
        """Apply the registry's monetary declaration to one classification."""
        if classification.name in self.monetary_columns:
            return classification.model_copy(update={"monetary": MonetaryStatus.MONETARY})
        return classification

    def classify(self, name: str) -> ColumnClassification:
        """Classify one column: registry entry, then family, then fail closed."""
        known = self.by_name().get(name)
        base = known if known is not None else self.classify_unknown(name)
        return self._with_monetary(base)

    def classify_unknown(self, name: str) -> ColumnClassification:
        """Classify a column that is not in this registry.

        Family patterns are tried first. Anything else is UNKNOWN, which by
        §5.8 excludes it from prospective analysis: adding a column to an
        export can never silently change an answer.
        """
        for family in self.families:
            if family.matches(name):
                return family.classify(name)
        return ColumnClassification(
            name=name,
            category=ColumnCategory.METADATA,
            availability=Availability.UNKNOWN,
            disposition=Disposition.QUARANTINE,
            classified=False,
            note="Not present in the registry and matched no family.",
            quarantine=QuarantineReason(
                code=QuarantineCode.UNCLASSIFIED_COLUMN,
                detail=(
                    "No registry entry and no family pattern places this column, so "
                    "nothing is known about what it means or when it is knowable."
                ),
                resolution="Classify the column in the export registry.",
            ),
        )

    def coverage_for(self, canonical: str) -> CanonicalCoverage:
        for record in self.coverage:
            if record.canonical == canonical:
                return record
        raise KeyError(f"no coverage record for canonical column {canonical!r}")

    @model_validator(mode="after")
    def _each_column_classified_once(self) -> ColumnRegistry:
        names = [c.name for c in self.columns]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"columns classified more than once: {duplicates}")
        return self
