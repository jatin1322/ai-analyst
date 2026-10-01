"""Resolving concepts to a dataset's physical columns (ARCHITECTURE 12.2).

Evidence is gathered from four declarative sources and several inferential
ones, and the status follows from which kinds were found:

* a declaration (tenant config, export registry, user override, documentation)
  yields `confirmed`;
* name or shape evidence alone yields `inferred`, permanently;
* two or more rival candidates with no declaration yield `ambiguous`;
* nothing at all yields `unavailable`.

The promotion rule is enforced by `ConceptBinding` itself, so a mistake here
raises rather than silently confirming a guess.
"""

from __future__ import annotations

from ai_analyst.contracts.agreement import (
    AgreementReport,
    ReconstructionAssessment,
    ReconstructionVerdict,
)
from ai_analyst.contracts.binding import (
    BindingEvidence,
    BindingStatus,
    ConceptBinding,
    ConceptBindings,
    EvidenceKind,
)
from ai_analyst.contracts.columns import Availability, QuarantineCode
from ai_analyst.contracts.concepts import (
    CONCEPTS,
    GRAIN_CONCEPTS,
    BusinessConcept,
    ConceptCardinality,
    SemanticType,
)
from ai_analyst.contracts.dataset import DatasetRegistry
from ai_analyst.contracts.grants import GrantLedger
from ai_analyst.contracts.profile import DatasetProfile
from ai_analyst.contracts.schema import (
    CanonicalColumn,
    DatasetSchema,
    DataType,
    DerivationRule,
    MappingConfidence,
)
from ai_analyst.contracts.status import StatusStrategy
from ai_analyst.contracts.tenant import TenantProfile

# Which canonical column carries which concept, once conformance has run.
CANONICAL_CONCEPTS: dict[CanonicalColumn, BusinessConcept] = {
    CanonicalColumn.OPP_ID: BusinessConcept.OPPORTUNITY_ID,
    CanonicalColumn.AS_OF: BusinessConcept.SNAPSHOT_DATE,
    CanonicalColumn.CLOSE_DATE: BusinessConcept.EXPECTED_CLOSE_DATE,
    CanonicalColumn.CREATED_DATE: BusinessConcept.CREATED_DATE,
    CanonicalColumn.STAGE: BusinessConcept.STAGE,
    CanonicalColumn.AMOUNT: BusinessConcept.AMOUNT,
    CanonicalColumn.STATUS: BusinessConcept.OPPORTUNITY_STATUS,
    CanonicalColumn.FORECAST_CATEGORY: BusinessConcept.FORECAST_CATEGORY,
    CanonicalColumn.SEGMENT: BusinessConcept.CUSTOMER_SEGMENT,
    CanonicalColumn.OWNER_ID: BusinessConcept.OWNER_ID,
    CanonicalColumn.ACCOUNT_ID: BusinessConcept.ACCOUNT_ID,
}

# Names that *suggest* a concept. Inference only: a hit here can never confirm,
# which is the whole point of keeping it separate from the alias table used for
# canonical mapping. `close_date_qtr` once matched a close date this way.
CONCEPT_NAME_HINTS: dict[BusinessConcept, tuple[str, ...]] = {
    BusinessConcept.TERMINAL_OUTCOME: ("terminal_fate", "terminal_outcome", "final_outcome"),
    BusinessConcept.TERMINAL_DATE: ("terminal_date", "actual_close_date", "closed_date"),
    BusinessConcept.TERMINAL_AMOUNT: ("terminal_amount", "booked_amount", "final_amount"),
    BusinessConcept.CUSTOMER_SEGMENT: ("segment", "customer_segment", "account_segment"),
    BusinessConcept.ACCOUNT_ID: ("account_id", "account", "account_name"),
    BusinessConcept.OWNER_ID: ("owner_id", "ownerid", "owner", "rep"),
    BusinessConcept.FORECAST_CATEGORY: ("forecastcategory", "forecastcategoryname"),
}


def _normalize(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum() or ch == "_")


def _mapping_evidence(
    confidence: MappingConfidence, source: str, has_export_registry: bool
) -> BindingEvidence:
    """Express how a canonical mapping was made as typed evidence."""
    if confidence is MappingConfidence.USER:
        kind = (
            EvidenceKind.EXPORT_REGISTRY if has_export_registry else EvidenceKind.USER_CONFIRMATION
        )
        detail = (
            "The export registry's coverage record names this source column."
            if has_export_registry
            else "Bound by an explicit user override."
        )
    elif confidence is MappingConfidence.EXACT:
        kind, detail = EvidenceKind.EXACT_NAME, "The source header equals the canonical name."
    elif confidence is MappingConfidence.ALIAS:
        kind, detail = EvidenceKind.ALIAS, "The source header is a known alias."
    else:
        kind, detail = EvidenceKind.FUZZY_NAME, "A fuzzy header match. Never applied."
    return BindingEvidence(kind=kind, detail=detail, source=source)


def _status_of(evidence: list[BindingEvidence]) -> BindingStatus:
    supporting = [e for e in evidence if e.supports]
    if any(e.can_confirm for e in supporting):
        return BindingStatus.CONFIRMED
    return BindingStatus.INFERRED if supporting else BindingStatus.UNAVAILABLE


def _tenant_evidence(tenant: TenantProfile, concept: BusinessConcept) -> BindingEvidence:
    return BindingEvidence(
        kind=EvidenceKind.TENANT_CONFIG,
        detail=f"Declared by tenant {tenant.tenant_id}.",
        source=tenant.source or tenant.tenant_id,
    )


def _text_columns(registry: DatasetRegistry, profile: DatasetProfile | None) -> tuple[
    list[str], list[str]
]:
    """Text columns the registry classifies, and ones only the profile detected."""
    classified = [c.name for c in registry.text_columns()]
    detected: list[str] = []
    if profile is not None:
        known = set(classified)
        detected = [
            t.name
            for t in profile.text_columns
            if t.name not in known and getattr(t, "basis", "") == "detected"
        ]
    return classified, detected


def _candidates_by_name(
    concept: BusinessConcept, registry: DatasetRegistry, taken: set[str]
) -> list[str]:
    hints = {_normalize(h) for h in CONCEPT_NAME_HINTS.get(concept, ())}
    if not hints:
        return []
    return [
        c.name
        for c in registry.columns
        if c.name not in taken and _normalize(c.name) in hints
    ]


def _reconstruction_evidence(
    schema: DatasetSchema, concept: BusinessConcept
) -> list[BindingEvidence]:
    """Evidence contributed by a declared reconstruction."""
    evidence: list[BindingEvidence] = []
    for derived in schema.derived_columns:
        if derived.rule is not DerivationRule.RECONSTRUCTED:
            continue
        if CANONICAL_CONCEPTS.get(derived.column) is not concept:
            continue
        evidence.append(
            BindingEvidence(
                kind=EvidenceKind.EXPORT_REGISTRY,
                detail=(
                    f"The export registry declares this column is rebuildable as "
                    f"`{derived.expression}` from {', '.join(derived.sources)}."
                ),
                source="export_registry_coverage",
            )
        )
        evidence.append(
            BindingEvidence(
                kind=EvidenceKind.DERIVATION,
                detail=f"Value rebuilt at ingestion as `{derived.expression}`.",
                source="conform",
                confidence=0.9,
            )
        )
    return evidence


def build_bindings(
    schema: DatasetSchema,
    registry: DatasetRegistry,
    profile: DatasetProfile | None = None,
    tenant: TenantProfile | None = None,
    agreement: AgreementReport | None = None,
    grants: GrantLedger | None = None,
) -> ConceptBindings:
    """Resolve every concept for one dataset.

    Usage grants are not created here. They are issued once, persisted, and
    passed in as a validated `GrantLedger` (`data/grants.py`); without one, a
    declared binding is still confirmed but releases no unclassified column.
    """
    dataset_id = registry.dataset_id
    present = set(registry.names)
    # A tenant declares its own header, which conformance may have renamed:
    # a tenant saying "amount is ARR" must still resolve after `ARR` became the
    # canonical `arr`. Source headers therefore resolve to conformed names.
    by_source = {
        c.source_name: c.name
        for c in registry.columns
        if c.source_name and c.source_name not in present
    }
    by_canonical = schema.mapping.by_canonical()
    has_registry = registry.export_registry is not None
    bindings: list[ConceptBinding] = []
    taken: set[str] = set()
    verdicts: dict[BusinessConcept, ReconstructionAssessment] = {}

    classified_text, detected_text = _text_columns(registry, profile)

    for concept, concept_def in CONCEPTS.items():
        evidence: list[BindingEvidence] = []
        columns: list[str] = []
        alternatives: list[str] = []
        caveats: list[str] = []
        note = ""
        # True when the column was rebuilt rather than read. Such a concept is
        # available only if its required agreement tests passed.
        reconstructed = False

        # A computed concept binds no column; the calendar produces it.
        if concept_def.computed:
            bindings.append(
                ConceptBinding(
                    dataset_id=dataset_id,
                    concept=concept,
                    status=BindingStatus.UNAVAILABLE,
                    note=(
                        f"Computed from {concept_def.derived_from.value} by the fiscal "
                        "calendar, never bound to a stamped label."
                    ),
                )
            )
            continue

        # 1. A tenant declaration outranks everything and can confirm.
        if tenant is not None and tenant.declares_absent(concept):
            bindings.append(
                ConceptBinding(
                    dataset_id=dataset_id,
                    concept=concept,
                    status=BindingStatus.UNAVAILABLE,
                    evidence=(_tenant_evidence(tenant, concept),),
                    note="The tenant declares this concept absent from the dataset.",
                )
            )
            continue

        declared = [
            by_source.get(c, c)
            for c in (tenant.declared(concept) if tenant else ())
            if c in present or c in by_source
        ]
        if declared:
            declaration = _tenant_evidence(tenant, concept)  # type: ignore[arg-type]
            conflicts = [
                reason
                for name in declared
                if (reason := declaration_conflict(concept, registry.by_name().get(name)))
            ]
            if conflicts:
                # A declaration can release an unknown; it cannot overrule a known
                # classification. The conflict is surfaced, and nothing binds.
                bindings.append(
                    ConceptBinding(
                        dataset_id=dataset_id,
                        concept=concept,
                        status=BindingStatus.UNAVAILABLE,
                        evidence=(declaration,),
                        caveats=("declaration_conflict",),
                        note=f"Declaration rejected: {'; '.join(conflicts)}",
                    )
                )
                continue
            columns = declared
            evidence.append(declaration)

        # 2. The grain is verified rather than declared: ingestion asserts
        #    (as_of, opp_id) uniqueness and refuses the file otherwise.
        if not columns and concept in GRAIN_CONCEPTS:
            canonical = next(k for k, v in CANONICAL_CONCEPTS.items() if v is concept)
            if canonical.value in present:
                columns = [canonical.value]
                mapped = by_canonical.get(canonical)
                evidence.append(
                    BindingEvidence(
                        kind=EvidenceKind.GRAIN_ASSERTION,
                        detail=(
                            "Ingestion asserted that (as_of, opp_id) is unique across "
                            "every row, so this column functions as half the grain."
                        ),
                        source=mapped.source_column if mapped else canonical.value,
                    )
                )
                note = (
                    f"Conformed from source column {mapped.source_column!r}."
                    if mapped
                    else ""
                )

        # 3. The canonical mapping, for concepts carried by a canonical column.
        if not columns:
            canonical = next(
                (k for k, v in CANONICAL_CONCEPTS.items() if v is concept), None
            )
            if canonical is not None and canonical.value in present:
                mapped = by_canonical.get(canonical)
                if mapped is not None:
                    columns = [canonical.value]
                    evidence.append(
                        _mapping_evidence(mapped.confidence, mapped.source_column, has_registry)
                    )
                    note = f"Conformed from source column {mapped.source_column!r}."
                else:
                    recon = _reconstruction_evidence(schema, concept)
                    if recon:
                        reconstructed = True
                        columns = [canonical.value]
                        evidence.extend(recon)
                        note = "Rebuilt at ingestion; the export carried no such column."
                    elif canonical is CanonicalColumn.STATUS:
                        columns, evidence, note, caveats = _status_binding(schema)

        # 4. Text is many-cardinality and comes from classification, then detection.
        if concept is BusinessConcept.NARRATIVE_TEXT:
            if classified_text:
                columns = classified_text
                evidence.append(
                    BindingEvidence(
                        kind=EvidenceKind.EXPORT_REGISTRY,
                        detail=f"{len(classified_text)} columns classified as text.",
                        source="classifications.json",
                    )
                )
            if detected_text:
                if not columns:
                    columns = detected_text
                evidence.append(
                    BindingEvidence(
                        kind=EvidenceKind.VALUE_PATTERN,
                        detail=(
                            f"{len(detected_text)} further columns look like prose by "
                            "length and distinctness, but nothing classifies them."
                        ),
                        source="profile",
                        confidence=0.5,
                    )
                )

        # 5. Name hints, inference only, and ambiguous when they collide.
        if not columns:
            hits = _candidates_by_name(concept, registry, taken)
            if len(hits) == 1:
                columns = hits
                evidence.append(
                    BindingEvidence(
                        kind=EvidenceKind.EXACT_NAME,
                        detail=f"Column {hits[0]!r} matches a known name for this concept.",
                        source=hits[0],
                        confidence=0.6,
                    )
                )
            elif len(hits) > 1:
                alternatives = hits
                evidence.append(
                    BindingEvidence(
                        kind=EvidenceKind.EXACT_NAME,
                        detail=f"{len(hits)} columns match names for this concept.",
                        source=", ".join(hits),
                        confidence=0.4,
                    )
                )

        # 6. A fuzzy candidate the mapper refused to apply is still a proposal.
        if not columns and not alternatives:
            canonical = next(
                (k for k, v in CANONICAL_CONCEPTS.items() if v is concept), None
            )
            fuzzy = [
                m for m in schema.mapping.fuzzy_candidates if m.canonical_column is canonical
            ]
            if fuzzy:
                columns = [fuzzy[0].source_column]
                evidence.append(
                    BindingEvidence(
                        kind=EvidenceKind.FUZZY_NAME,
                        detail=(
                            f"Header {fuzzy[0].source_column!r} resembles this concept "
                            f"(score {fuzzy[0].score}). Never conformed; confirm before use."
                        ),
                        source=fuzzy[0].source_column,
                        confidence=min(fuzzy[0].score, 0.5),
                    )
                )

        status = (
            BindingStatus.AMBIGUOUS
            if alternatives
            else _status_of(evidence)
        )
        if status is BindingStatus.UNAVAILABLE:
            columns = []

        # 7. Agreement tests corroborate or contradict, never confirm. A rebuilt
        #    column is judged by its verdict (ARCHITECTURE 13.1): a failed
        #    validity test or a missing declaration withholds it; an
        #    unverifiable validity test or a failed reconciliation test leaves it
        #    usable with caveats that cap trust at B.
        if columns and (agreement is not None or reconstructed):
            promoted = tenant.promoted_tests if tenant else frozenset()
            calendar_resolved = bool(tenant and tenant.calendar_resolved)
            if agreement is not None:
                for result in agreement.for_concept(concept):
                    evidence.append(result.as_evidence())
            if reconstructed:
                if agreement is None:
                    status, columns = BindingStatus.UNAVAILABLE, []
                    note = (
                        "Withheld: this column was reconstructed and no agreement "
                        "tests were run, so not even its structure was checked."
                    )
                else:
                    assessment = agreement.assess(
                        concept,
                        declared=_reconstruction_declared(schema, concept),
                        promoted=promoted,
                        calendar_resolved=calendar_resolved,
                    )
                    verdicts[concept] = assessment
                    if not assessment.verdict.is_available:
                        status, columns = BindingStatus.UNAVAILABLE, []
                        note = _withheld_note(assessment)
                    else:
                        caveats.extend(assessment.caveats)
                        note = (
                            f"{note} Verdict: {assessment.verdict.value}."
                            + (f" {'; '.join(assessment.reasons)}." if assessment.reasons else "")
                        ).strip()
            elif agreement.concept_is_contradicted(
                concept, promoted=promoted, calendar_resolved=calendar_resolved
            ):
                status, columns = BindingStatus.UNAVAILABLE, []
                note = "Withheld: a validity agreement test contradicts this binding."

        # 8. Caveats follow the column's classification.
        for name in columns:
            column = registry.by_name().get(name)
            if column is None:
                continue
            if column.is_quarantined and column.quarantine is not None:
                caveats.append(f"quarantined:{column.quarantine.code.value}")
            lineage = column.classification.lineage
            if lineage is not None and not lineage.confirmed:
                caveats.append("lineage_unconfirmed")
            for item in registry.unresolved_for(name):
                caveats.append(item.id)

        if len(columns) > 1 and CONCEPTS[concept].cardinality is ConceptCardinality.ONE:
            columns = columns[:1]

        taken.update(columns)
        bindings.append(
            ConceptBinding(
                dataset_id=dataset_id,
                concept=concept,
                columns=tuple(columns),
                status=status,
                evidence=tuple(evidence),
                alternatives=tuple(alternatives),
                caveats=tuple(dict.fromkeys(caveats)),
                note=note,
            )
        )

    return ConceptBindings(
        dataset_id=dataset_id,
        bindings=tuple(bindings),
        reconstruction_verdicts=tuple(verdicts.values()),
        grants=grants.grants if grants is not None else (),
    )


# Storage types that can hold each semantic type. A declaration binding a
# concept to a column that cannot hold it is invalid, not coerced.
_NUMERIC = frozenset({DataType.DECIMAL, DataType.DOUBLE, DataType.BIGINT, DataType.INTEGER})
_ADMISSIBLE_TYPES: dict[SemanticType, frozenset[DataType]] = {
    SemanticType.MONEY: _NUMERIC,
    SemanticType.QUANTITY: _NUMERIC,
    SemanticType.DATE: frozenset({DataType.DATE}),
}


def declaration_conflict(concept: BusinessConcept, column) -> str | None:
    """Why a tenant declaration of `column` as `concept` is invalid, or None.

    Checked when the declaration is loaded (ARCHITECTURE 13.2). Three ways to
    fail, each a real contradiction rather than a style preference:

    * the column cannot hold the concept's semantic type;
    * a registry classified the column as knowable only in hindsight, and the
      concept asserts it is knowable at its own snapshot;
    * a registry quarantined the column for a *known* reason. Being merely
      unclassified is not a known reason: that is exactly what a declaration
      may resolve.
    """
    if column is None:
        return None
    definition = CONCEPTS[concept]
    admissible = _ADMISSIBLE_TYPES.get(definition.semantic_type)
    if admissible is not None and column.dtype not in admissible:
        return (
            f"{column.name!r} is stored as {column.dtype.value}, which cannot hold "
            f"{concept.value} ({definition.semantic_type.value})"
        )
    classification = column.classification
    if not classification.classified:
        return None
    if (
        classification.availability is Availability.FUTURE_CONTAMINATED
        and not definition.retrospective
    ):
        return (
            f"{column.name!r} is classified future_contaminated, but {concept.value} "
            "is defined as knowable at its own snapshot"
        )
    if column.is_quarantined and column.quarantine.code is not QuarantineCode.UNCLASSIFIED_COLUMN:
        return (
            f"{column.name!r} is quarantined ({column.quarantine.code.value}): "
            f"{column.quarantine.detail}"
        )
    return None


def _reconstruction_declared(schema: DatasetSchema, concept: BusinessConcept) -> bool:
    """Whether an accountable source declared the derivation of this concept."""
    return any(
        d.is_reconstructed
        and d.declared_by
        and CANONICAL_CONCEPTS.get(d.column) is concept
        for d in schema.derived_columns
    )


def _withheld_note(assessment: ReconstructionAssessment) -> str:
    """Why a reconstructed concept is unavailable, naming the evidence."""
    if assessment.verdict is ReconstructionVerdict.UNDECLARED:
        return (
            "Withheld: the reconstruction is undeclared. Nobody accountable stated "
            "the derivation, and a rebuilt value is only as good as that statement."
        )
    return (
        f"Withheld: contradicted by {', '.join(assessment.contradicted)} "
        f"({'; '.join(assessment.reasons)}). A wrong value is worse than none."
    )


def _status_binding(
    schema: DatasetSchema,
) -> tuple[list[str], list[BindingEvidence], str, list[str]]:
    """Status is configuration, and how it resolved decides the binding."""
    resolution = schema.status_resolution
    column = CanonicalColumn.STATUS.value
    if resolution is None:
        return [], [], "No status could be resolved.", []
    if resolution.is_authoritative:
        return (
            [column],
            [
                BindingEvidence(
                    kind=EvidenceKind.TENANT_CONFIG,
                    detail=(
                        f"Resolved from a tenant-declared stage-to-status map on "
                        f"{resolution.column!r} (exact lookup)."
                        if resolution.strategy is StatusStrategy.DECLARED_STAGE_MAP
                        else f"Resolved from the configured authoritative column "
                        f"{resolution.column!r}."
                    ),
                    source="status configuration",
                )
            ],
            f"Authoritative status from {resolution.column!r}.",
            ["status_unmapped_values"]
            if (resolution.unmapped_values or resolution.unmapped_rows)
            else [],
        )
    return (
        [column],
        [
            BindingEvidence(
                kind=EvidenceKind.VALUE_PATTERN,
                detail=(
                    "Inferred from stage keywords, which is explicitly "
                    "non-authoritative: a vocabulary can encode an outcome in a "
                    "label this classifier cannot read."
                ),
                source=f"strategy={resolution.strategy.value}",
                confidence=0.4,
            )
        ],
        "Stage-keyword fallback. Not authoritative.",
        ["authoritative_status"],
    )


__all__ = [
    "CANONICAL_CONCEPTS",
    "CONCEPT_NAME_HINTS",
    "build_bindings",
]
