"""Issuing, persisting and loading usage grants (ARCHITECTURE 13.2).

Grants are issued once, when a tenant's declarations are registered against a
dataset, and persisted as `grants.json` beside the dataset. The runtime loads
that artifact; it never silently rebuilds grants from a tenant profile, which is
what makes the grant set auditable: what a result relied on is exactly what was
issued and saved.

Loading re-validates everything a grant depends on and fails loudly on any
mismatch. A ledger is never partially applied.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from ai_analyst.config import Settings, get_settings
from ai_analyst.contracts.binding import (
    GENERIC_PURPOSES,
    BindingEvidence,
    ColumnPurpose,
    EvidenceKind,
    GrantKind,
    UsageGrant,
    grant_for_declaration,
)
from ai_analyst.contracts.columns import (
    ColumnCategory,
    Disposition,
    MonetaryStatus,
)
from ai_analyst.contracts.dataset import DatasetRegistry
from ai_analyst.contracts.grants import (
    GrantLedger,
    GrantLedgerError,
    GrantLedgerErrorCode,
    GrantRejection,
)
from ai_analyst.contracts.schema import DataType
from ai_analyst.contracts.tenant import ColumnDeclaration, TenantProfile
from ai_analyst.data.binding import declaration_conflict

_NUMERIC = frozenset({DataType.DECIMAL, DataType.DOUBLE, DataType.BIGINT, DataType.INTEGER})


def _resolve_header(registry: DatasetRegistry, header: str) -> str | None:
    """A tenant names its own header; conformance may have renamed it."""
    if registry.has(header):
        return header
    return next((c.name for c in registry.columns if c.source_name == header), None)


def _classification_rejection(
    declaration: ColumnDeclaration, registry: DatasetRegistry
) -> str | None:
    """Why a column classification may not issue a generic grant, or None."""
    if not declaration.is_declared:
        return "an inferred classification is a proposal and never grants anything"
    column = registry.get(declaration.column)
    if column.classification.classified:
        return (
            f"{declaration.column!r} is already classified by the registry "
            f"({column.classification.availability.value}); a declaration can release "
            "an unknown column, never overrule a known classification"
        )
    classification = declaration.classification
    if classification.disposition is Disposition.QUARANTINE:
        return "the tenant's own classification quarantines the column"
    if classification.category is ColumnCategory.TEXT:
        return "a text column is catalogued only and never read"
    if classification.monetary_status is MonetaryStatus.MONETARY and column.dtype not in _NUMERIC:
        return f"a {column.dtype.value} column cannot be declared money"
    return None


def _generic_grant(
    declaration: ColumnDeclaration, dtype: DataType, dataset_id: str, tenant: TenantProfile
) -> UsageGrant:
    classification = declaration.classification
    purposes = set(GENERIC_PURPOSES)
    if dtype in _NUMERIC:
        purposes.add(ColumnPurpose.MEASURE)
    return UsageGrant(
        dataset_id=dataset_id,
        tenant_id=tenant.tenant_id,
        kind=GrantKind.GENERIC,
        column=declaration.column,
        column_dtype=dtype,
        purposes=frozenset(purposes),
        availability=classification.availability,
        monetary=classification.monetary_status is MonetaryStatus.MONETARY,
        source=BindingEvidence(
            kind=EvidenceKind.TENANT_CONFIG,
            detail=f"Column classified by tenant {tenant.tenant_id}.",
            source=declaration.source,
        ),
    )


def issue_grants(
    dataset_id: str, registry: DatasetRegistry, tenant: TenantProfile
) -> GrantLedger:
    """Every grant a tenant's declarations support on this dataset. Pure and deterministic."""
    grants: list[UsageGrant] = []
    rejected: list[GrantRejection] = []

    for concept, headers in sorted(tenant.concept_columns.items()):
        evidence = BindingEvidence(
            kind=EvidenceKind.TENANT_CONFIG,
            detail=f"Declared by tenant {tenant.tenant_id}.",
            source=tenant.source or tenant.tenant_id,
        )
        for header in headers:
            name = _resolve_header(registry, header)
            if name is None:
                rejected.append(
                    GrantRejection(
                        kind=GrantKind.CONCEPT,
                        column=header,
                        concept=concept,
                        reason="no such column in this dataset",
                    )
                )
                continue
            column = registry.get(name)
            conflict = declaration_conflict(concept, column)
            if conflict:
                rejected.append(
                    GrantRejection(
                        kind=GrantKind.CONCEPT, column=name, concept=concept, reason=conflict
                    )
                )
                continue
            grants.append(
                grant_for_declaration(
                    concept,
                    name,
                    evidence,
                    dataset_id=dataset_id,
                    tenant_id=tenant.tenant_id,
                    dtype=column.dtype,
                )
            )

    headers = [*registry.names,*(c.source_name for c in registry.columns if c.source_name)]
    family_declarations, family_conflicts = tenant.expanded_declarations(headers)
    for header, reason in sorted(family_conflicts.items()):
        rejected.append(GrantRejection(kind=GrantKind.GENERIC, column=header, reason=reason))
    for family in tenant.family_declarations:
        if not any(family.matches(h) for h in headers):
            rejected.append(
                GrantRejection(
                    kind=GrantKind.GENERIC,
                    column=family.family,
                    reason="family pattern matches no column in this dataset",
                )
            )
    seen: set[str] = set()
    for declaration in [*tenant.column_classifications, *family_declarations]:
        name = _resolve_header(registry, declaration.column)
        if name is not None:
            if name in seen:
                continue
            seen.add(name)
        if name is None:
            rejected.append(
                GrantRejection(
                    kind=GrantKind.GENERIC,
                    column=declaration.column,
                    reason="no such column in this dataset",
                )
            )
            continue
        declaration = declaration.model_copy(
            update={
                "column": name,
                "classification": declaration.classification.model_copy(update={"name": name}),
            }
        )
        reason = _classification_rejection(declaration, registry)
        if reason:
            rejected.append(GrantRejection(kind=GrantKind.GENERIC, column=name, reason=reason))
            continue
        grants.append(
            _generic_grant(declaration, registry.get(name).dtype, dataset_id, tenant)
        )

    return GrantLedger.canonical(
        dataset_id=dataset_id,
        tenant_id=tenant.tenant_id,
        declarations_fingerprint=tenant.declarations_fingerprint,
        grants=grants,
        rejected=rejected,
    )


def ledger_path(dataset_id: str, settings: Settings) -> Path:
    return settings.dataset_dir(dataset_id) / "grants.json"


def save_grants(ledger: GrantLedger, settings: Settings | None = None) -> Path:
    """Persist a ledger. The same ledger always produces the same bytes."""
    settings = settings or get_settings()
    path = ledger_path(ledger.dataset_id, settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ledger.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def discard_grants(dataset_id: str, settings: Settings | None = None) -> None:
    ledger_path(dataset_id, settings or get_settings()).unlink(missing_ok=True)


def _still_valid(grant: UsageGrant, registry: DatasetRegistry) -> None:
    """Re-check a persisted grant against the dataset as it is now."""
    if not registry.has(grant.column):
        raise GrantLedgerError(
            GrantLedgerErrorCode.SCHEMA_DRIFT,
            f"grant {grant.grant_id} names {grant.column!r}, which this dataset no longer has",
        )
    column = registry.get(grant.column)
    if column.dtype is not grant.column_dtype:
        raise GrantLedgerError(
            GrantLedgerErrorCode.SCHEMA_DRIFT,
            f"grant {grant.grant_id} was issued for a {grant.column_dtype.value} column; "
            f"{grant.column!r} is now {column.dtype.value}",
        )
    if grant.kind is GrantKind.CONCEPT:
        conflict = declaration_conflict(grant.concept, column)  # type: ignore[arg-type]
        if conflict:
            raise GrantLedgerError(GrantLedgerErrorCode.CONTRADICTED, conflict)
    elif column.classification.classified:
        raise GrantLedgerError(
            GrantLedgerErrorCode.CONTRADICTED,
            f"{grant.column!r} is now classified by the registry "
            f"({column.classification.availability.value}); its generic grant no longer applies",
        )


def load_grants(
    dataset_id: str,
    *,
    tenant: TenantProfile,
    registry: DatasetRegistry,
    settings: Settings | None = None,
) -> GrantLedger | None:
    """The persisted ledger for this dataset and tenant, validated, or None if none exists."""
    settings = settings or get_settings()
    path = ledger_path(dataset_id, settings)
    if not path.exists():
        return None
    try:
        ledger = GrantLedger.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValidationError, ValueError) as exc:
        raise GrantLedgerError(
            GrantLedgerErrorCode.CORRUPTED, f"{path} is not a valid grant ledger: {exc}"
        ) from exc
    if ledger.dataset_id != dataset_id:
        raise GrantLedgerError(
            GrantLedgerErrorCode.DATASET_MISMATCH,
            f"the ledger at {path} was issued for dataset {ledger.dataset_id!r}, "
            f"not {dataset_id!r}",
        )
    if ledger.tenant_id != tenant.tenant_id:
        raise GrantLedgerError(
            GrantLedgerErrorCode.TENANT_MISMATCH,
            f"the ledger was issued for tenant {ledger.tenant_id!r}, not {tenant.tenant_id!r}",
        )
    if ledger.declarations_fingerprint != tenant.declarations_fingerprint:
        raise GrantLedgerError(
            GrantLedgerErrorCode.STALE,
            "the tenant's declarations changed after these grants were issued; "
            "re-issue them before use",
        )
    for grant in ledger.grants:
        _still_valid(grant, registry)
    return ledger
