"""Concept resolution: the one place a concept becomes a physical column.

Everything above this module speaks concepts. Everything below it speaks
columns. Nothing in the semantic engine may name a physical column directly,
which is what lets one metric definition serve a tenant whose amount column is
`new_amount` and one whose amount column is `ARR` (ARCHITECTURE 12.1).

Three gates run here, in this order, before any SQL exists:

1. **Binding.** Is the concept resolved for this dataset at all, and is the
   binding strong enough for the job? A concept a metric is *built on* needs a
   confirmed binding; a dimension or filter may run on an inferred one and caps
   the result at trust tier B (12.6, 12.7).
2. **Stance.** Under a prospective stance the contaminated columns are not
   merely discouraged, they are *unreachable*: `permitted_columns` never
   contains them, so nothing downstream can plan, group by, or filter on one
   (5.8, 12.8).
3. **Money.** A monetary concept resolves through `monetary_measure_sql` and
   nothing else, so the DECIMAL conversion is a declared boundary visible in
   the compiled SQL rather than a float that happened to be summed (12.16).

The resolver raises rather than returning a half-answer. A caller that wants to
*ask* instead of demand uses `try_resolve`, which returns the rejection as data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from ai_analyst.contracts.binding import (
    PRIMARY_PURPOSE,
    BindingStatus,
    ColumnPurpose,
    ConceptBindings,
    GrantKind,
    UsageGrant,
)
from ai_analyst.contracts.columns import Availability
from ai_analyst.contracts.concepts import CONCEPTS, BusinessConcept, SemanticType
from ai_analyst.contracts.dataset import DatasetRegistry
from ai_analyst.contracts.plan import AnalysisStance
from ai_analyst.contracts.rejection import PlanRejection, RejectionCode
from ai_analyst.contracts.result import (
    TrustAssessment,
    TrustFactor,
    TrustFactorKind,
    TrustTier,
)
from ai_analyst.contracts.schema import DataType
from ai_analyst.data.conform import quote_ident
from ai_analyst.data.money import monetary_measure_sql

# Short names a plan may use for a concept in a dimension, filter, or feature.
# Any concept's own value ("amount", "customer_segment") is accepted too.
CONCEPT_ALIASES: dict[str, BusinessConcept] = {
    "segment": BusinessConcept.CUSTOMER_SEGMENT,
    "owner": BusinessConcept.OWNER_ID,
    "stage": BusinessConcept.STAGE,
    "forecast_category": BusinessConcept.FORECAST_CATEGORY,
    "account": BusinessConcept.ACCOUNT_ID,
    "status": BusinessConcept.OPPORTUNITY_STATUS,
    "close_date": BusinessConcept.EXPECTED_CLOSE_DATE,
}


def concept_named(name: str) -> BusinessConcept | None:
    """The concept a plan reference names, or None for a physical column."""
    if name in CONCEPT_ALIASES:
        return CONCEPT_ALIASES[name]
    try:
        return BusinessConcept(name)
    except ValueError:
        return None


class ResolutionError(ValueError):
    """A concept or column could not be resolved. Carries the rejection."""

    def __init__(self, rejection: PlanRejection) -> None:
        super().__init__(rejection.message)
        self.rejection = rejection


@dataclass(frozen=True)
class ColumnResolution:
    """One concept resolved to one physical column of one dataset."""

    concept: BusinessConcept
    column: str
    dtype: DataType
    binding_status: BindingStatus
    semantic_type: SemanticType
    availability: Availability
    caveats: tuple[str, ...] = ()
    # Set when the column was reachable only through a tenant usage grant.
    grant: UsageGrant | None = None

    @property
    def is_monetary(self) -> bool:
        return self.semantic_type is SemanticType.MONEY

    def sql(self, alias: str | None = None) -> str:
        """The column reference. Not a measure: money goes through `measure_sql`."""
        ident = quote_ident(self.column)
        return f"{quote_ident(alias)}.{ident}" if alias else ident

    def measure_sql(self, alias: str | None = None) -> str:
        """The column as an aggregatable measure.

        A monetary concept is cast to fixed-scale DECIMAL here and nowhere else.
        """
        if self.is_monetary:
            return monetary_measure_sql(self.column, self.dtype, alias=alias)
        return self.sql(alias)


@dataclass
class ConceptResolver:
    """Resolves concepts to columns for one dataset under one stance."""

    dataset_id: str
    registry: DatasetRegistry
    bindings: ConceptBindings
    stance: AnalysisStance = AnalysisStance.PROSPECTIVE
    knowledge_cutoff: date | None = None
    spec_id: str | None = None
    # Every concept actually resolved, in resolution order. The compilation
    # record is built from this rather than from a re-scan of the SQL.
    resolved: dict[BusinessConcept, ColumnResolution] = field(default_factory=dict)
    # Structured trust factors, collected as concepts and columns are read.
    # The tier is derived from these by `TrustAssessment`; nothing here sets it.
    trust_factors: list[TrustFactor] = field(default_factory=list)
    # Generic grants relied on for physical-column reads, by column.
    generic_used: dict[str, UsageGrant] = field(default_factory=dict)

    # ---------------------------------------------------------------- stance

    @property
    def permitted_columns(self) -> frozenset[str]:
        """The stance-derived column allowlist.

        Prospective keeps only columns knowable at their own `as_of`.
        Retrospective admits outcome columns but still refuses quarantined and
        unclassified ones: hindsight is not a licence to read a column nobody
        could define.
        """
        permitted: set[str] = set()
        for column in self.registry.columns:
            classification = column.classification
            if column.is_quarantined or not classification.classified:
                continue
            if (
                self.stance is AnalysisStance.PROSPECTIVE
                and not classification.knowable_at_snapshot
            ):
                continue
            permitted.add(column.name)
        for grant in self.bindings.grants:
            if grant.kind is GrantKind.GENERIC and (
                self.stance is AnalysisStance.RETROSPECTIVE
                or grant.availability.safe_for_prospective
            ):
                permitted.add(grant.column)
        return frozenset(permitted)

    def generic_access(
        self, name: str, purpose: ColumnPurpose | None, *, field_name: str
    ) -> tuple[PlanRejection | None, UsageGrant | None]:
        """Whether a physical column may be read generically, and under which grant.

        A registry-classified column is judged by its classification alone. An
        unclassified one is readable only under a *generic* grant from a tenant
        column classification, only for the grant's purposes, and only when the
        grant's declared timing is safe for the stance. A concept grant never
        applies here, and no purpose at all means no grant is consulted.
        """
        rejection = self._column_rejection(name, field_name=field_name)
        if rejection is None or rejection.code is not RejectionCode.COLUMN_UNCLASSIFIED:
            return rejection, None
        grant = self.bindings.generic_grant_for(name)
        if grant is None:
            return rejection, None
        if purpose is None or not grant.permits(purpose):
            return (
                PlanRejection(
                    code=RejectionCode.GRANT_PURPOSE_NOT_PERMITTED,
                    message=(
                        f"{name!r} is readable under a tenant classification only for "
                        f"{', '.join(sorted(p.value for p in grant.purposes))}"
                        + (f"; not as a {purpose.value}" if purpose else "")
                    ),
                    spec_id=self.spec_id,
                    field=field_name,
                    column=name,
                ),
                None,
            )
        if self.stance is AnalysisStance.PROSPECTIVE and not (
            grant.availability.safe_for_prospective
        ):
            return (
                PlanRejection(
                    code=RejectionCode.COLUMN_NOT_KNOWABLE_AT_SNAPSHOT,
                    message=(
                        f"{name!r} is declared {grant.availability.value} and cannot be "
                        "read under a prospective stance"
                    ),
                    spec_id=self.spec_id,
                    field=field_name,
                    column=name,
                    remedy="Use a retrospective stance if hindsight is intended.",
                ),
                None,
            )
        return None, grant

    def column_is_monetary(self, name: str) -> bool:
        """Whether a physical column is money, by classification or tenant grant."""
        if self.registry.has(name) and self.registry.get(name).classification.is_monetary:
            return True
        grant = self.bindings.generic_grant_for(name)
        return bool(grant and grant.monetary)

    def _column_rejection(self, name: str, *, field_name: str) -> PlanRejection | None:
        """Why this physical column may not be read, or None when it may."""
        column = self.registry.by_name().get(name)
        if column is None:
            return PlanRejection(
                code=RejectionCode.UNKNOWN_COLUMN,
                message=f"column {name!r} does not exist in dataset {self.dataset_id!r}",
                spec_id=self.spec_id,
                field=field_name,
                column=name,
            )
        classification = column.classification
        if not classification.classified:
            return PlanRejection(
                code=RejectionCode.COLUMN_UNCLASSIFIED,
                message=(
                    f"column {name!r} is unclassified, so nothing is established "
                    "about when its values are knowable"
                ),
                spec_id=self.spec_id,
                field=field_name,
                column=name,
                remedy="Classify the column in the dataset's registry.",
            )
        if column.is_quarantined:
            reason = column.quarantine
            return PlanRejection(
                code=RejectionCode.COLUMN_QUARANTINED,
                message=f"column {name!r} is quarantined: {reason.detail}",
                spec_id=self.spec_id,
                field=field_name,
                column=name,
                remedy=reason.resolution,
            )
        if (
            self.stance is AnalysisStance.PROSPECTIVE
            and not classification.knowable_at_snapshot
        ):
            return PlanRejection(
                code=RejectionCode.COLUMN_NOT_KNOWABLE_AT_SNAPSHOT,
                message=(
                    f"column {name!r} is {classification.availability.value} and cannot "
                    f"be read under a prospective stance"
                ),
                spec_id=self.spec_id,
                field=field_name,
                column=name,
                remedy="Use a retrospective stance if hindsight is intended.",
            )
        return None

    def inspect_column(
        self,
        name: str,
        *,
        field_name: str = "columns",
        purpose: ColumnPurpose | None = None,
    ) -> PlanRejection | None:
        """Check a physical column for *generic* use and record what reading it costs.

        Generic means reached by its physical name. A concept grant never applies
        here: it is scoped to reads through its concept (13.2). A generic grant,
        from a tenant column classification, applies for its purposes only.

        The disclosure is the half that is easy to forget. Retrospective
        analysis is *allowed* to read a contaminated column, but 12.7 requires
        that doing so be disclosed and cap the tier at B.
        """
        rejection, grant = self.generic_access(name, purpose, field_name=field_name)
        if rejection is not None:
            return rejection
        if grant is not None:
            self.generic_used[name] = grant
            self._note(TrustFactorKind.USAGE_GRANT, name, grant.disclosure)
            if not grant.availability.safe_for_prospective:
                self._note(
                    TrustFactorKind.RETROSPECTIVE_READ,
                    name,
                    f"column {name!r} is declared {grant.availability.value} and was "
                    "read under a retrospective stance",
                )
            return None
        classification = self.registry.get(name).classification
        if self.stance is AnalysisStance.RETROSPECTIVE and not (
            classification.availability.safe_for_prospective
        ):
            self._note(
                TrustFactorKind.RETROSPECTIVE_READ,
                name,
                f"column {name!r} is {classification.availability.value} and was "
                "read under a retrospective stance",
            )
        self._note_lineage(name)
        return None

    def reference(
        self, name: str, *, purpose: ColumnPurpose, field_name: str
    ) -> tuple[str | None, PlanRejection | None]:
        """Resolve a plan reference to a physical column, or say why not.

        A concept name resolves through its binding, and so may use a usage
        grant within its purposes. Anything else is a physical column and is
        judged by generic rules alone, so a granted column named by its raw
        header is still blocked.
        """
        concept = concept_named(name)
        if concept is not None:
            outcome = self.try_resolve(
                concept, load_bearing=False, field_name=field_name, purpose=purpose
            )
            if isinstance(outcome, PlanRejection):
                return None, outcome
            return outcome.column, None
        if not self.registry.has(name):
            return None, PlanRejection(
                code=RejectionCode.UNKNOWN_COLUMN,
                message=(
                    f"{field_name} reference {name!r} is neither a concept nor a column "
                    f"of dataset {self.dataset_id!r}"
                ),
                spec_id=self.spec_id,
                field=field_name,
                column=name,
            )
        rejection = self.inspect_column(name, field_name=field_name, purpose=purpose)
        return (None, rejection) if rejection else (name, None)

    def check_column(self, name: str, *, field_name: str = "columns") -> None:
        """Raise unless this physical column may be read under the stance."""
        rejection = self.inspect_column(name, field_name=field_name)
        if rejection is not None:
            raise ResolutionError(rejection)

    # --------------------------------------------------------------- concepts

    def _note(self, kind: TrustFactorKind, subject: str, reason: str) -> None:
        factor = TrustFactor(kind=kind, subject=subject, reason=reason)
        if factor not in self.trust_factors:
            self.trust_factors.append(factor)

    def _note_lineage(self, name: str) -> None:
        lineage = self.registry.get(name).classification.lineage
        if lineage is not None and not lineage.confirmed:
            self._note(
                TrustFactorKind.LINEAGE_UNCONFIRMED,
                name,
                f"column {name!r} is a precomputed feature whose lineage is unconfirmed",
            )

    def _note_caveat(self, concept: BusinessConcept, caveat: str) -> None:
        """Map a binding caveat onto the trust factor it implies."""
        head, _, tail = caveat.partition(":")
        subject = tail or concept.value
        match head:
            case "authoritative_status":
                kind = TrustFactorKind.STATUS_NOT_AUTHORITATIVE
                reason = (
                    f"{concept.value}: authoritative_status is unresolved; status is "
                    "inferred from stage keywords"
                )
            case "lineage_unconfirmed":
                kind = TrustFactorKind.LINEAGE_UNCONFIRMED
                reason = f"{concept.value}: lineage_unconfirmed"
            case "reconstruction_unverified":
                kind = TrustFactorKind.RECONSTRUCTION_UNVERIFIED
                reason = f"{concept.value} is reconstructed and {tail} could not verify it"
            case "reconstruction_uncorroborated":
                kind = TrustFactorKind.RECONSTRUCTION_UNCORROBORATED
                reason = (
                    f"{concept.value} is reconstructed and no validity test "
                    "corroborated it; it rests on the declaration alone"
                )
            case "reconciliation_warning":
                kind = TrustFactorKind.RECONCILIATION_WARNING
                reason = (
                    f"{concept.value} disagrees with {tail}, whose convention is "
                    "undeclared"
                )
            case _:
                kind = TrustFactorKind.UNRESOLVED_QUESTION
                reason = f"{concept.value}: open question {caveat}"
        self._note(kind, subject, reason)

    def try_resolve(
        self,
        concept: BusinessConcept,
        *,
        load_bearing: bool = True,
        field_name: str = "metrics",
        purpose: ColumnPurpose | None = None,
    ) -> ColumnResolution | PlanRejection:
        """Resolve a concept, returning the rejection as data rather than raising.

        `purpose` says what the read is for. It matters only when the column is
        reachable solely through a usage grant, which licenses the concept's
        own purposes and nothing else.
        """
        definition = CONCEPTS[concept]
        purpose = purpose or PRIMARY_PURPOSE[definition.role]

        # A retrospective concept under a prospective stance is the leakage
        # failure of 5.8, and it is refused before the binding is even read.
        if definition.retrospective and self.stance is AnalysisStance.PROSPECTIVE:
            return PlanRejection(
                code=RejectionCode.RETROSPECTIVE_CONCEPT_IN_PROSPECTIVE,
                message=(
                    f"{concept.value} describes what eventually happened and cannot "
                    "be read under a prospective stance"
                ),
                spec_id=self.spec_id,
                field=field_name,
                concept=concept,
                remedy="Use a retrospective stance if hindsight is intended.",
            )

        binding = self.bindings.by_concept().get(concept)
        status = binding.status if binding else BindingStatus.UNAVAILABLE

        if status is BindingStatus.AMBIGUOUS:
            return PlanRejection(
                code=RejectionCode.CONCEPT_AMBIGUOUS,
                message=(
                    f"{concept.value} has rival candidates and no declaration: "
                    f"{', '.join(binding.alternatives)}"
                ),
                spec_id=self.spec_id,
                field=field_name,
                concept=concept,
                remedy="Declare which column satisfies the concept in the tenant profile.",
            )
        if status is BindingStatus.UNAVAILABLE:
            note = (binding.note if binding else "") or ""
            withheld = "withheld" in note.lower() or "agreement" in note.lower()
            conflict = bool(binding and "declaration_conflict" in binding.caveats)
            return PlanRejection(
                code=(
                    RejectionCode.DECLARATION_CONFLICT
                    if conflict
                    else RejectionCode.CONCEPT_WITHHELD
                    if withheld
                    else RejectionCode.CONCEPT_UNAVAILABLE
                ),
                message=f"{concept.value} concept unavailable" + (f": {note}" if note else ""),
                spec_id=self.spec_id,
                field=field_name,
                concept=concept,
            )
        if load_bearing and status is not BindingStatus.CONFIRMED:
            return PlanRejection(
                code=RejectionCode.CONCEPT_NOT_CONFIRMED,
                message=(
                    f"{concept.value} is bound to {binding.columns[0]!r} by inference "
                    "only, which is not enough for a metric built on it"
                ),
                spec_id=self.spec_id,
                field=field_name,
                concept=concept,
                column=binding.columns[0],
                remedy="Declare the column for this concept in the tenant profile.",
            )

        name = binding.columns[0]
        column = self.registry.get(name)
        availability = column.classification.availability
        grant: UsageGrant | None = None

        generic = self._column_rejection(name, field_name=field_name)
        if generic is not None:
            # Reachable only through a grant, and only for its purposes.
            grant = self.bindings.grant_for(name, concept)
            if grant is None or generic.code is RejectionCode.UNKNOWN_COLUMN:
                return generic
            if not grant.permits(purpose):
                return PlanRejection(
                    code=RejectionCode.GRANT_PURPOSE_NOT_PERMITTED,
                    message=(
                        f"{name!r} is readable only as {concept.value} for "
                        f"{', '.join(sorted(p.value for p in grant.purposes))}; "
                        f"it cannot be used as a {purpose.value}"
                    ),
                    spec_id=self.spec_id,
                    field=field_name,
                    concept=concept,
                    column=name,
                    remedy="Classify the column in the registry to use it generically.",
                )
            if self.stance is AnalysisStance.PROSPECTIVE and not (
                grant.availability.safe_for_prospective
            ):
                return generic
            availability = grant.availability
            self._note(TrustFactorKind.USAGE_GRANT, name, grant.disclosure)
        else:
            self.inspect_column(name, field_name=field_name)

        resolution = ColumnResolution(
            concept=concept,
            column=name,
            dtype=column.dtype,
            binding_status=status,
            semantic_type=definition.semantic_type,
            availability=availability,
            caveats=tuple(binding.caveats),
            grant=grant,
        )
        if status is not BindingStatus.CONFIRMED:
            self._note(
                TrustFactorKind.BINDING_INFERRED,
                concept.value,
                f"{concept.value} is bound to {name!r} by inference, not by a declaration",
            )
        for caveat in binding.caveats:
            if not caveat.startswith("quarantined:"):
                self._note_caveat(concept, caveat)
        if self.stance is AnalysisStance.RETROSPECTIVE and definition.retrospective:
            self._note(
                TrustFactorKind.RETROSPECTIVE_READ,
                concept.value,
                f"{concept.value} is retrospective and was read under a retrospective stance",
            )
        self.resolved[concept] = resolution
        return resolution

    def resolve(
        self,
        concept: BusinessConcept,
        *,
        load_bearing: bool = True,
        field_name: str = "metrics",
        purpose: ColumnPurpose | None = None,
    ) -> ColumnResolution:
        """Resolve a concept or raise `ResolutionError` carrying the rejection."""
        outcome = self.try_resolve(
            concept, load_bearing=load_bearing, field_name=field_name, purpose=purpose
        )
        if isinstance(outcome, PlanRejection):
            raise ResolutionError(outcome)
        return outcome

    def has(
        self,
        concept: BusinessConcept,
        *,
        load_bearing: bool = True,
        purpose: ColumnPurpose | None = None,
    ) -> bool:
        """Whether a concept is resolvable, without recording trust reasons."""
        probe = ConceptResolver(
            dataset_id=self.dataset_id,
            registry=self.registry,
            bindings=self.bindings,
            stance=self.stance,
            knowledge_cutoff=self.knowledge_cutoff,
        )
        outcome = probe.try_resolve(concept, load_bearing=load_bearing, purpose=purpose)
        return not isinstance(outcome, PlanRejection)

    # ------------------------------------------------------------------ sql

    def sql(self, concept: BusinessConcept, alias: str | None = None, **kwargs) -> str:
        return self.resolve(concept, **kwargs).sql(alias)

    def measure_sql(self, concept: BusinessConcept, alias: str | None = None, **kwargs) -> str:
        return self.resolve(concept, **kwargs).measure_sql(alias)

    @property
    def monetary_expressions(self) -> tuple[str, ...]:
        """Every DECIMAL boundary conversion this resolver produced."""
        return tuple(
            r.measure_sql() for r in self.resolved.values() if r.is_monetary
        )

    @property
    def concept_columns(self) -> dict[str, str]:
        return {c.value: r.column for c, r in self.resolved.items()}

    @property
    def grants_used(self) -> tuple[UsageGrant, ...]:
        concept = [r.grant for r in self.resolved.values() if r.grant is not None]
        return (*concept, *self.generic_used.values())

    @property
    def trust(self) -> TrustAssessment:
        """Everything resolved so far, as an assessment. The tier is derived."""
        return TrustAssessment(factors=tuple(self.trust_factors))

    @property
    def trust_tier(self) -> TrustTier:
        return self.trust.tier

    @property
    def trust_reasons(self) -> list[str]:
        return self.trust.disclosures
