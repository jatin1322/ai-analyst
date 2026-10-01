"""The persisted usage-grant ledger (ARCHITECTURE 13.2, CLAUDE.md 2.3).

A grant must be explicit, typed, scoped, persisted and auditable. The ledger is
the persisted form: every grant issued for one dataset under one tenant's
declarations, plus every declaration that was refused and why.

Three identities tie a ledger to what it was issued for, and each is checked
on load:

* `dataset_id`: a ledger for one dataset is never applied to another;
* `tenant_id`: grants belong to the tenant who declared them;
* `declarations_fingerprint`: a hash of the tenant's declarations at issue
  time, so a ledger issued under different declarations is detected as stale
  rather than silently applied.

Each grant carries its own content-derived id, so an edited grant fails to load.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai_analyst.contracts.binding import GrantKind, UsageGrant
from ai_analyst.contracts.concepts import BusinessConcept

LEDGER_SCHEMA_VERSION = 1


class GrantRejection(BaseModel):
    """A declaration that did not become a grant, kept for the audit trail."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: GrantKind
    column: str
    concept: BusinessConcept | None = None
    reason: str


class GrantLedger(BaseModel):
    """Every usage grant for one dataset under one tenant's declarations."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = LEDGER_SCHEMA_VERSION
    dataset_id: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    declarations_fingerprint: str = Field(min_length=64, max_length=64)
    grants: tuple[UsageGrant, ...] = ()
    rejected: tuple[GrantRejection, ...] = ()

    @model_validator(mode="after")
    def _internally_consistent(self) -> GrantLedger:
        for grant in self.grants:
            if grant.dataset_id != self.dataset_id or grant.tenant_id != self.tenant_id:
                raise ValueError(
                    f"grant {grant.grant_id} belongs to {grant.tenant_id}/{grant.dataset_id}, "
                    f"not to this ledger's {self.tenant_id}/{self.dataset_id}"
                )
        ids = [g.grant_id for g in self.grants]
        if len(ids) != len(set(ids)):
            raise ValueError("a grant appears in the ledger more than once")
        if ids != sorted(ids):
            raise ValueError("grants must be stored in canonical order")
        keys = [(g.kind, g.column, g.concept) for g in self.grants]
        if len(keys) != len(set(keys)):
            raise ValueError("two grants release the same column for the same thing")
        return self

    @classmethod
    def canonical(
        cls,
        *,
        dataset_id: str,
        tenant_id: str,
        declarations_fingerprint: str,
        grants: list[UsageGrant],
        rejected: list[GrantRejection],
    ) -> GrantLedger:
        """Build a ledger in canonical order, so saving it is deterministic."""
        return cls(
            dataset_id=dataset_id,
            tenant_id=tenant_id,
            declarations_fingerprint=declarations_fingerprint,
            grants=tuple(sorted(grants, key=lambda g: g.grant_id)),
            rejected=tuple(
                sorted(rejected, key=lambda r: (r.kind.value, r.column, str(r.concept)))
            ),
        )


class GrantLedgerErrorCode(StrEnum):
    CORRUPTED = "corrupted"
    DATASET_MISMATCH = "dataset_mismatch"
    TENANT_MISMATCH = "tenant_mismatch"
    STALE = "stale"
    SCHEMA_DRIFT = "schema_drift"
    CONTRADICTED = "contradicted"


class GrantLedgerError(ValueError):
    """A persisted ledger that must not be applied. Fails loudly, never partially."""

    def __init__(self, code: GrantLedgerErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
