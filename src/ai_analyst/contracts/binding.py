"""Concept bindings: which physical columns satisfy a concept (ARCHITECTURE 12.2).

A binding connects one concept to one or more physical columns in one dataset,
and records *why*. The recording is the point. Without evidence a binding is an
assertion, and an assertion made from a column name is how `close_date_qtr`, a
quarter label, once bound to the required close date.

The rule this module enforces structurally:

> **Name similarity never confirms a binding, and no amount of inferential
> evidence ever promotes to confirmed.** Only a declaration does.

`EvidenceKind.can_confirm` is the whole mechanism. An inferred binding stays
inferred until a person, a tenant configuration, a reviewed export registry, or
supplied documentation says otherwise. An agreement test deliberately cannot
confirm either: agreement is evidence, never proof (12.3).
"""

from __future__ import annotations

import hashlib
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator

from ai_analyst.contracts.columns import Availability, FeatureLineage
from ai_analyst.contracts.concepts import (
    CONCEPTS,
    AnalyticalOperation,
    BusinessConcept,
    ConceptCardinality,
    ConceptRequirement,
    ConceptRole,
    concepts_required_for,
)
from ai_analyst.contracts.schema import DataType


class BindingStatus(StrEnum):
    """How settled a binding is, and therefore where it may be used."""

    CONFIRMED = "confirmed"
    INFERRED = "inferred"
    AMBIGUOUS = "ambiguous"
    UNAVAILABLE = "unavailable"

    @property
    def is_bound(self) -> bool:
        """Whether any column is attached. Ambiguous has candidates, not a binding."""
        return self in (BindingStatus.CONFIRMED, BindingStatus.INFERRED)

    @property
    def admissible_in_semantic_path(self) -> bool:
        """Only a confirmed binding may back a semantic metric (12.6)."""
        return self is BindingStatus.CONFIRMED


class EvidenceKind(StrEnum):
    """Where a piece of evidence came from.

    The split between the first four and the rest is the load-bearing line in
    this module: the first four are *declarations* by someone accountable, the
    rest are *observations* that can be coincidence.
    """

    USER_CONFIRMATION = "user_confirmation"
    TENANT_CONFIG = "tenant_config"
    EXPORT_REGISTRY = "export_registry"
    DOCUMENTATION = "documentation"
    # The one *verified* confirming kind, and it is verified rather than
    # asserted: ingestion hard-fails unless (as_of, opp_id) is unique across
    # every row, so a column that survived that check demonstrably functions as
    # half the grain. This is a proof about behaviour, not a claim about a name.
    GRAIN_ASSERTION = "grain_assertion"

    EXACT_NAME = "exact_name"
    ALIAS = "alias"
    FUZZY_NAME = "fuzzy_name"
    VALUE_PATTERN = "value_pattern"
    TYPE_SHAPE = "type_shape"
    AGREEMENT_TEST = "agreement_test"
    DERIVATION = "derivation"

    @property
    def can_confirm(self) -> bool:
        """Whether this evidence alone may establish a confirmed binding.

        `AGREEMENT_TEST` cannot: ARCHITECTURE 12.3 holds that a passing
        agreement test raises confidence and never proves a reading, because
        agreement can be coincidental and the export may not contain the case
        that would break it. `DERIVATION` cannot either: a derivation is only
        as good as the declaration that documents it, which arrives separately.
        """
        return self in _DECLARATIVE


_DECLARATIVE: frozenset[EvidenceKind] = frozenset(
    {
        EvidenceKind.USER_CONFIRMATION,
        EvidenceKind.TENANT_CONFIG,
        EvidenceKind.EXPORT_REGISTRY,
        EvidenceKind.DOCUMENTATION,
        EvidenceKind.GRAIN_ASSERTION,
    }
)


class BindingEvidence(BaseModel):
    """One reason to believe, or doubt, a binding."""

    model_config = ConfigDict(frozen=True)

    kind: EvidenceKind
    detail: str
    source: str = ""
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    # False for evidence that argues against the binding, which is recorded
    # rather than dropped so a reviewer sees what was weighed.
    supports: bool = True

    @property
    def can_confirm(self) -> bool:
        return self.supports and self.kind.can_confirm


class ConceptBinding(BaseModel):
    """One concept's resolution in one dataset."""

    model_config = ConfigDict(frozen=True)

    dataset_id: str
    concept: BusinessConcept
    columns: tuple[str, ...] = ()
    status: BindingStatus
    evidence: tuple[BindingEvidence, ...] = ()
    # Columns considered and not chosen. For an ambiguous binding these are
    # the rival candidates the agent must ask about.
    alternatives: tuple[str, ...] = ()
    caveats: tuple[str, ...] = ()
    lineage: FeatureLineage | None = None
    note: str = ""

    @property
    def definition(self):  # noqa: ANN201 - ConceptDefinition, avoids a cycle in docs
        return CONCEPTS[self.concept]

    @property
    def confidence(self) -> float:
        """The strongest supporting evidence. Zero when nothing supports it."""
        supporting = [e.confidence for e in self.evidence if e.supports]
        return max(supporting) if supporting else 0.0

    @property
    def column(self) -> str | None:
        """The single bound column, for a one-cardinality concept."""
        return self.columns[0] if len(self.columns) == 1 else None

    @property
    def is_available(self) -> bool:
        return self.status.is_bound

    @property
    def confirming_evidence(self) -> tuple[BindingEvidence, ...]:
        return tuple(e for e in self.evidence if e.can_confirm)

    @model_validator(mode="after")
    def _status_matches_columns(self) -> ConceptBinding:
        if self.status is BindingStatus.UNAVAILABLE and self.columns:
            raise ValueError(
                f"{self.concept}: an unavailable concept must bind no column, got {self.columns}"
            )
        if self.status.is_bound and not self.columns:
            raise ValueError(f"{self.concept}: a {self.status} binding must name a column")
        if self.status is BindingStatus.AMBIGUOUS and len(self.alternatives) < 2:
            raise ValueError(
                f"{self.concept}: an ambiguous binding must list the rival candidates"
            )
        if self.status is BindingStatus.AMBIGUOUS and self.columns:
            raise ValueError(
                f"{self.concept}: an ambiguous concept is not bound; its candidates are "
                "alternatives, not columns"
            )
        return self

    @model_validator(mode="after")
    def _only_a_declaration_confirms(self) -> ConceptBinding:
        """The rule of 12.2, enforced by the type rather than by discipline."""
        if self.status is BindingStatus.CONFIRMED and not self.confirming_evidence:
            kinds = ", ".join(sorted({e.kind.value for e in self.evidence})) or "none"
            raise ValueError(
                f"{self.concept}: a confirmed binding needs a declaration "
                f"(user, tenant config, export registry, or documentation); "
                f"the evidence offered was: {kinds}. Name similarity never confirms."
            )
        return self

    @model_validator(mode="after")
    def _cardinality_respected(self) -> ConceptBinding:
        one = CONCEPTS[self.concept].cardinality is ConceptCardinality.ONE
        if one and len(self.columns) > 1:
            raise ValueError(
                f"{self.concept}: binds one column, got {len(self.columns)}: {self.columns}"
            )
        return self


class ColumnPurpose(StrEnum):
    """What a plan is using a column *for*.

    A usage grant is scoped to purposes, so a column a tenant declared as its
    amount measure does not also become a free dimension or feature.
    """

    MEASURE = "measure"
    DIMENSION = "dimension"
    FILTER = "filter"
    FEATURE = "feature"
    TEMPORAL = "temporal"
    OUTCOME = "outcome"
    GRAIN = "grain"


# The purposes a concept's role licenses. A filter on a declared measure is
# still a read of that measure ("deals above $100k"), so MEASURE admits FILTER.
ROLE_PURPOSES: dict[ConceptRole, frozenset[ColumnPurpose]] = {
    ConceptRole.GRAIN: frozenset({ColumnPurpose.GRAIN, ColumnPurpose.FILTER}),
    ConceptRole.MEASURE: frozenset({ColumnPurpose.MEASURE, ColumnPurpose.FILTER}),
    ConceptRole.DIMENSION: frozenset(
        {ColumnPurpose.DIMENSION, ColumnPurpose.FILTER, ColumnPurpose.FEATURE}
    ),
    ConceptRole.TEMPORAL: frozenset({ColumnPurpose.TEMPORAL, ColumnPurpose.FILTER}),
    ConceptRole.OUTCOME: frozenset(
        {ColumnPurpose.OUTCOME, ColumnPurpose.DIMENSION, ColumnPurpose.FILTER}
    ),
    ConceptRole.NARRATIVE: frozenset(),
}

PRIMARY_PURPOSE: dict[ConceptRole, ColumnPurpose] = {
    ConceptRole.GRAIN: ColumnPurpose.GRAIN,
    ConceptRole.MEASURE: ColumnPurpose.MEASURE,
    ConceptRole.DIMENSION: ColumnPurpose.DIMENSION,
    ConceptRole.TEMPORAL: ColumnPurpose.TEMPORAL,
    ConceptRole.OUTCOME: ColumnPurpose.OUTCOME,
    ConceptRole.NARRATIVE: ColumnPurpose.FEATURE,
}


class GrantKind(StrEnum):
    """What a usage grant releases (ARCHITECTURE 13.2)."""

    # A column readable *as one concept*, issued by a concept declaration.
    CONCEPT = "concept"
    # A column readable generically, issued by a tenant column classification.
    GENERIC = "generic"


_NUMERIC_TYPES = frozenset({DataType.DECIMAL, DataType.DOUBLE, DataType.BIGINT, DataType.INTEGER})
GENERIC_PURPOSES = frozenset(
    {ColumnPurpose.DIMENSION, ColumnPurpose.FILTER, ColumnPurpose.FEATURE}
)


def grant_identity(
    dataset_id: str,
    tenant_id: str,
    kind: GrantKind,
    column: str,
    concept: BusinessConcept | None,
    purposes: frozenset[ColumnPurpose],
    availability: Availability,
) -> str:
    """A stable identity from a grant's content. Same content, same id, always."""
    payload = "|".join(
        [
            dataset_id,
            tenant_id,
            kind.value,
            column,
            concept.value if concept else "*",
            ",".join(sorted(p.value for p in purposes)),
            availability.value,
        ]
    )
    return "g" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class UsageGrant(BaseModel):
    """Permission to read one column for stated purposes (ARCHITECTURE 13.2).

    Issued only by an explicit, valid tenant statement, and persisted as its own
    artifact (`contracts/grants.py`). The grant separates three things one flag
    used to conflate: the declaration says what the column means; the column's
    own registry classification may still say it is unknown; and the grant says
    it may nonetheless be read, for these purposes, with this timing.

    A `CONCEPT` grant applies only to reads *through its concept*. A `GENERIC`
    grant applies to reads of the physical column, but never as a measure of a
    concept: money still needs a concept binding. Neither applies to a purpose
    it does not list.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    grant_id: str = ""
    dataset_id: str
    tenant_id: str
    kind: GrantKind
    column: str
    column_dtype: DataType
    concept: BusinessConcept | None = None
    purposes: frozenset[ColumnPurpose]
    # For a CONCEPT grant, inherited from the concept definition, never from
    # the column. For a GENERIC grant, the tenant's declared classification.
    availability: Availability
    # True when the tenant classified the column as money. A monetary read goes
    # through the DECIMAL boundary whatever the storage type.
    monetary: bool = False
    source: BindingEvidence

    @field_serializer("purposes")
    def _sorted_purposes(self, purposes: frozenset[ColumnPurpose]) -> list[str]:
        return sorted(p.value for p in purposes)

    @property
    def computed_id(self) -> str:
        return grant_identity(
            self.dataset_id,
            self.tenant_id,
            self.kind,
            self.column,
            self.concept,
            self.purposes,
            self.availability,
        )

    @model_validator(mode="after")
    def _consistent(self) -> UsageGrant:
        if not self.purposes:
            raise ValueError(f"grant on {self.column!r} lists no purpose")
        if self.kind is GrantKind.CONCEPT:
            if self.concept is None:
                raise ValueError("a concept grant must name its concept")
            definition = CONCEPTS[self.concept]
            if not self.purposes <= ROLE_PURPOSES[definition.role]:
                raise ValueError(
                    f"grant on {self.column!r} lists purposes the {self.concept.value} "
                    "concept does not license"
                )
            expected = (
                Availability.FUTURE_CONTAMINATED
                if definition.retrospective
                else Availability.AS_OF_FACT
            )
            if self.availability is not expected:
                raise ValueError(
                    f"a {self.concept.value} grant carries the concept's timing "
                    f"({expected.value}), not {self.availability.value}"
                )
        else:
            if self.concept is not None:
                raise ValueError("a generic grant names no concept")
            allowed = GENERIC_PURPOSES | (
                {ColumnPurpose.MEASURE} if self.column_dtype in _NUMERIC_TYPES else set()
            )
            if not self.purposes <= allowed:
                raise ValueError(
                    f"generic grant on {self.column!r} lists purposes its type cannot support"
                )
        if self.monetary and self.column_dtype not in _NUMERIC_TYPES:
            raise ValueError(f"{self.column!r} is {self.column_dtype.value} and cannot be money")
        if self.grant_id and self.grant_id != self.computed_id:
            raise ValueError(
                f"grant {self.grant_id!r} does not match its own content; it was altered"
            )
        if not self.grant_id:
            object.__setattr__(self, "grant_id", self.computed_id)
        return self

    def permits(self, purpose: ColumnPurpose) -> bool:
        return purpose in self.purposes

    @property
    def disclosure(self) -> str:
        if self.kind is GrantKind.CONCEPT:
            return (
                f"{self.concept.value} read from {self.column!r} under a tenant "
                f"declaration ({self.source.source}); the column is otherwise unclassified"
            )
        return (
            f"{self.column!r} read under a tenant column classification "
            f"({self.source.source}); no export registry classifies it"
        )


def grant_for_declaration(
    concept: BusinessConcept,
    column: str,
    evidence: BindingEvidence,
    *,
    dataset_id: str,
    tenant_id: str,
    dtype: DataType,
) -> UsageGrant:
    """The grant a valid concept declaration issues. Purposes and timing come from the concept."""
    definition = CONCEPTS[concept]
    return UsageGrant(
        dataset_id=dataset_id,
        tenant_id=tenant_id,
        kind=GrantKind.CONCEPT,
        column=column,
        column_dtype=dtype,
        concept=concept,
        purposes=ROLE_PURPOSES[definition.role],
        availability=(
            Availability.FUTURE_CONTAMINATED
            if definition.retrospective
            else Availability.AS_OF_FACT
        ),
        monetary=definition.is_monetary,
        source=evidence,
    )


class ReconstructionVerdict(StrEnum):
    """The standing of a reconstructed concept after its tests ran (13.1)."""

    VALID = "valid"
    VALID_WITH_WARNINGS = "valid_with_warnings"
    CONTRADICTED = "contradicted"
    UNDECLARED = "undeclared"

    @property
    def is_available(self) -> bool:
        return self in (ReconstructionVerdict.VALID, ReconstructionVerdict.VALID_WITH_WARNINGS)


class ReconstructionAssessment(BaseModel):
    """A verdict with the test ids that produced it."""

    model_config = ConfigDict(frozen=True)

    concept: BusinessConcept
    verdict: ReconstructionVerdict
    contradicted: tuple[str, ...] = ()
    unverified: tuple[str, ...] = ()
    reconciliation_warnings: tuple[str, ...] = ()
    # Validity tests treated as reconciliation because the calendar is unresolved.
    demoted: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()

    @property
    def caveats(self) -> tuple[str, ...]:
        """Binding caveats, one per warning, in a form the trust model reads."""
        out = [f"reconstruction_unverified:{t}" for t in self.unverified]
        out += [f"reconciliation_warning:{t}" for t in self.reconciliation_warnings]
        if self.verdict is ReconstructionVerdict.VALID_WITH_WARNINGS and not out:
            out.append("reconstruction_uncorroborated")
        return tuple(out)


class ConceptBindings(BaseModel):
    """Every concept's resolution for one dataset."""

    model_config = ConfigDict(frozen=True)

    dataset_id: str
    bindings: tuple[ConceptBinding, ...]
    # The 13.1 verdict for every reconstructed concept, kept beside the bindings
    # so a card, a tool, or the trust model can report it without re-deriving.
    reconstruction_verdicts: tuple[ReconstructionAssessment, ...] = ()
    # Purpose-scoped permissions issued by explicit tenant declarations (13.2).
    grants: tuple[UsageGrant, ...] = ()

    @model_validator(mode="after")
    def _one_binding_per_concept(self) -> ConceptBindings:
        seen = [b.concept for b in self.bindings]
        duplicates = sorted({c.value for c in seen if seen.count(c) > 1})
        if duplicates:
            raise ValueError(f"concepts bound more than once: {duplicates}")
        return self

    def grant_for(self, column: str, concept: BusinessConcept) -> UsageGrant | None:
        """The concept grant for this column and concept, if one was issued."""
        return next(
            (
                g
                for g in self.grants
                if g.kind is GrantKind.CONCEPT and g.column == column and g.concept is concept
            ),
            None,
        )

    def generic_grant_for(self, column: str) -> UsageGrant | None:
        """The generic grant for a physical column, if one was issued."""
        return next(
            (g for g in self.grants if g.kind is GrantKind.GENERIC and g.column == column),
            None,
        )

    def by_concept(self) -> dict[BusinessConcept, ConceptBinding]:
        return {b.concept: b for b in self.bindings}

    def get(self, concept: BusinessConcept) -> ConceptBinding:
        found = self.by_concept().get(concept)
        if found is None:
            raise KeyError(f"concept {concept.value!r} was not resolved for {self.dataset_id!r}")
        return found

    def status_of(self, concept: BusinessConcept) -> BindingStatus:
        found = self.by_concept().get(concept)
        return found.status if found else BindingStatus.UNAVAILABLE

    def columns_for(self, concept: BusinessConcept) -> tuple[str, ...]:
        found = self.by_concept().get(concept)
        return found.columns if found else ()

    def _in_status(self, status: BindingStatus) -> list[ConceptBinding]:
        return [b for b in self.bindings if b.status is status]

    def confirmed(self) -> list[ConceptBinding]:
        return self._in_status(BindingStatus.CONFIRMED)

    def inferred(self) -> list[ConceptBinding]:
        return self._in_status(BindingStatus.INFERRED)

    def ambiguous(self) -> list[ConceptBinding]:
        return self._in_status(BindingStatus.AMBIGUOUS)

    def unavailable(self) -> list[ConceptBinding]:
        return self._in_status(BindingStatus.UNAVAILABLE)

    def available(self) -> list[ConceptBinding]:
        return [b for b in self.bindings if b.is_available]

    def monetary_columns(self) -> set[str]:
        """Every column bound to a monetary concept.

        This is the semantic answer to "is this money", and the only one the
        profiler is allowed to use. A numeric shape never makes a column money.
        """
        return {
            c
            for b in self.bindings
            if CONCEPTS[b.concept].is_monetary
            for c in b.columns
        }

    def check_operation(self, operation: AnalyticalOperation) -> ConceptRequirement:
        """Whether one analytical operation's concepts are all confirmed.

        A semantic metric needs a confirmed binding (12.6), so an inferred one
        counts as missing here and is reported as such.
        """
        missing: list[BusinessConcept] = []
        caveated: list[BusinessConcept] = []
        for concept in concepts_required_for(operation):
            if CONCEPTS[concept].computed:
                continue
            binding = self.by_concept().get(concept)
            if binding is None or not binding.status.admissible_in_semantic_path:
                missing.append(concept)
            elif binding.caveats:
                caveated.append(concept)
        return ConceptRequirement(
            operation=operation,
            satisfied=not missing,
            missing=tuple(missing),
            caveated=tuple(caveated),
        )
