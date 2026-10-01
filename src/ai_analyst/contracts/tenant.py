"""Per-tenant declarations (ARCHITECTURE 12.14).

This is the only place a human states what a tenant's columns mean. It is the
declarative half of the binding model: a `TenantProfile` can confirm a binding
because someone accountable wrote it down, where a name match never can.

Deliberately small. It holds what cannot be discovered, not a copy of what can.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai_analyst.contracts.binding import BindingStatus
from ai_analyst.contracts.columns import ColumnClassification
from ai_analyst.contracts.concepts import (
    CONCEPTS,
    BusinessConcept,
    ConceptCardinality,
    SemanticType,
)
from ai_analyst.contracts.snapshot_policy import SnapshotPolicy
from ai_analyst.contracts.status import OpportunityStatus


class ColumnDeclaration(BaseModel):
    """A tenant's explicit classification of one physical column (ARCHITECTURE 13.2).

    Reuses `ColumnClassification`, the one classification type in the system,
    rather than introducing a second ontology. A declaration says what a column
    *is* and *when it is knowable*; it never says which concept it satisfies,
    so it can never confirm a concept binding. That is `concept_columns`' job.

    `status` follows the binding rules: `confirmed` is an accountable statement
    and may issue a generic usage grant; `inferred` is a proposal recorded for a
    human to confirm, and issues nothing.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    column: str
    classification: ColumnClassification
    semantic_type: SemanticType | None = None
    status: BindingStatus = BindingStatus.CONFIRMED
    # Who stated it, and on what evidence. Persisted with the declaration.
    source: str = Field(min_length=1)

    @model_validator(mode="after")
    def _coherent(self) -> ColumnDeclaration:
        if self.classification.name != self.column:
            raise ValueError(
                f"classification is for {self.classification.name!r}, not {self.column!r}"
            )
        if self.status not in (BindingStatus.CONFIRMED, BindingStatus.INFERRED):
            raise ValueError("a column declaration is either confirmed or inferred")
        if not self.classification.classified:
            raise ValueError(f"{self.column!r}: a declaration cannot declare 'unclassified'")
        return self

    @property
    def is_declared(self) -> bool:
        return self.status is BindingStatus.CONFIRMED


class FamilyDeclaration(BaseModel):
    """A tenant's explicit classification of a whole family of columns (WP4).

    It is shorthand for one `ColumnDeclaration` per matched column and nothing
    more: `expand` produces those declarations, and the ordinary grant rules
    then apply to each unchanged (a declaration never overrules a registry
    classification, `inferred` grants nothing, text is never read).

    `classification` is a template whose `name` is the family name; each member
    gets a copy renamed to the column. `pattern` is matched with `re.fullmatch`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    family: str = Field(min_length=1)
    pattern: str = Field(min_length=1)
    classification: ColumnClassification
    semantic_type: SemanticType | None = None
    status: BindingStatus = BindingStatus.CONFIRMED
    source: str = Field(min_length=1)

    @model_validator(mode="after")
    def _coherent(self) -> FamilyDeclaration:
        try:
            re.compile(self.pattern)
        except re.error as exc:
            raise ValueError(f"{self.family!r}: invalid member pattern: {exc}") from exc
        if self.classification.name != self.family:
            raise ValueError(
                f"template classification is for {self.classification.name!r}, "
                f"not family {self.family!r}"
            )
        if self.status not in (BindingStatus.CONFIRMED, BindingStatus.INFERRED):
            raise ValueError("a family declaration is either confirmed or inferred")
        if not self.classification.classified:
            raise ValueError(f"{self.family!r}: a declaration cannot declare 'unclassified'")
        return self

    def matches(self, column: str) -> bool:
        return re.fullmatch(self.pattern, column) is not None

    def expand(self, columns: Iterable[str]) -> list[ColumnDeclaration]:
        """One declaration per matching column, in the order given."""
        return [
            ColumnDeclaration(
                column=c,
                classification=self.classification.model_copy(
                    update={"name": c, "family": self.family}
                ),
                semantic_type=self.semantic_type,
                status=self.status,
                source=self.source,
            )
            for c in columns
            if self.matches(c)
        ]


class FamilyInvariantKind(StrEnum):
    # x_7d <= x_14d <= ... <= x_lifetime over a family's windowed members.
    WINDOW_MONOTONE = "window_monotone"
    # sum(parts) = whole, per window (or once, when no window applies).
    PARTS_SUM = "parts_sum"
    # label IS NULL wherever mask = 0.
    MASK_IMPLIES_NULL = "mask_implies_null"


class FamilyInvariant(BaseModel):
    """A tenant-declared relationship that a family of columns should satisfy.

    Evidence only: an invariant is evaluated as a RECONCILIATION agreement test
    and a violation is reported, never repaired. Column names are built from
    templates so one declaration covers every window:

    * `window_monotone`: `template` is a stem like `email_count_{w}`;
      windows are taken from `windows` in order, `lifetime` last.
    * `parts_sum`: `parts` and `whole` are templates using `{w}` over `windows`
      (or plain names when `windows` is empty).
    * `mask_implies_null`: `label` is null wherever `mask` = 0 (the synthetic
      generator's convention: win_label is null iff win_label_mask is 0).

    Column names are conformed names (a discovered column keeps its source name).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    kind: FamilyInvariantKind
    template: str = ""
    windows: tuple[str, ...] = ()
    parts: tuple[str, ...] = ()
    whole: str = ""
    label: str = ""
    mask: str = ""
    source: str = Field(min_length=1)

    @model_validator(mode="after")
    def _coherent(self) -> FamilyInvariant:
        if self.kind is FamilyInvariantKind.WINDOW_MONOTONE:
            if "{w}" not in self.template or len(self.windows) < 2:
                raise ValueError(
                    f"{self.id}: window_monotone needs a {{w}} template and 2+ windows"
                )
        elif self.kind is FamilyInvariantKind.PARTS_SUM:
            if len(self.parts) < 2 or not self.whole:
                raise ValueError(f"{self.id}: parts_sum needs 2+ parts and a whole")
        elif not (self.label and self.mask):
            raise ValueError(f"{self.id}: mask_implies_null needs a label and a mask")
        return self


class TenantProfile(BaseModel):
    """What a tenant has declared about its own dataset."""

    model_config = ConfigDict(frozen=True)

    tenant_id: str
    # Concept to the physical column(s) that satisfy it. A declaration here is
    # confirming evidence, so it is also an accountable statement.
    concept_columns: dict[BusinessConcept, tuple[str, ...]] = Field(default_factory=dict)
    # Concepts the tenant states it does not have. Declaring an absence stops
    # the inference layer from proposing a wrong candidate for it.
    absent_concepts: tuple[BusinessConcept, ...] = ()
    fiscal_year_start_month: int | None = None
    # How same-day captures of one opportunity are handled at ingestion. Not
    # part of `declarations_fingerprint`: it changes which rows exist, not what
    # any column may be used for, so it cannot make a usage grant stale.
    snapshot_policy: SnapshotPolicy = Field(default_factory=SnapshotPolicy)
    # Declared semantics of an ancillary field an agreement test compares
    # against, keyed by test id, e.g. "close_date_matches_eoq_close_diff":
    # "eoq_close_diff = days_to_eoq - days_to_close, both inclusive". A
    # declaration promotes the test from reconciliation to validity, so a
    # disagreement becomes a contradiction (ARCHITECTURE 13.1 rule 4).
    agreement_declarations: dict[str, str] = Field(default_factory=dict)
    # Explicit classifications of physical columns no export registry places.
    # A declaration here issues a *generic* usage grant; it never binds a concept.
    column_classifications: list[ColumnDeclaration] = Field(default_factory=list)
    # Classifications applied to every column matching a family pattern. Each
    # expands to per-column declarations, so the same grant rules apply.
    family_declarations: list[FamilyDeclaration] = Field(default_factory=list)
    # Declared stage-to-status map (WP5): what every stage value means. When
    # set, status is derived from stage by EXACT lookup and is authoritative;
    # a stage value not listed is EXCLUDED, never open. Unset keeps the
    # non-authoritative stage-keyword fallback. Not part of the grant
    # fingerprint: it decides status, not what any column may be used for.
    stage_status_map: dict[str, OpportunityStatus] | None = None
    # Declared invariants over families of columns (WP5), evaluated as
    # reconciliation evidence; they never rewrite data.
    family_invariants: list[FamilyInvariant] = Field(default_factory=list)
    # Free-text provenance for the declaration, such as who confirmed it.
    source: str = ""
    notes: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _stage_map_is_usable(self) -> TenantProfile:
        if self.stage_status_map is not None and OpportunityStatus.UNKNOWN in (
            self.stage_status_map.values()
        ):
            raise ValueError("stage_status_map may not map a stage to 'unknown'")
        ids = [i.id for i in self.family_invariants]
        if len(set(ids)) != len(ids):
            raise ValueError("family invariant ids must be unique")
        return self

    @model_validator(mode="after")
    def _declarations_respect_cardinality(self) -> TenantProfile:
        for concept, columns in self.concept_columns.items():
            if not columns:
                raise ValueError(f"{concept}: a declaration must name at least one column")
            if (
                CONCEPTS[concept].cardinality is ConceptCardinality.ONE
                and len(columns) > 1
            ):
                raise ValueError(
                    f"{concept}: binds one column, but {len(columns)} were declared"
                )
        declared = [d.column for d in self.column_classifications]
        repeated = sorted({c for c in declared if declared.count(c) > 1})
        if repeated:
            raise ValueError(f"columns classified more than once: {repeated}")
        families = [f.family for f in self.family_declarations]
        repeated_families = sorted({f for f in families if families.count(f) > 1})
        if repeated_families:
            raise ValueError(f"families declared more than once: {repeated_families}")
        overlap = set(self.concept_columns) & set(self.absent_concepts)
        if overlap:
            names = ", ".join(sorted(c.value for c in overlap))
            raise ValueError(f"declared both present and absent: {names}")
        return self

    @property
    def declarations_fingerprint(self) -> str:
        """A stable hash of everything a usage grant can be issued from.

        A persisted grant ledger records it, so a ledger issued under different
        declarations is detected as stale rather than silently applied.
        """
        payload = {
            "tenant_id": self.tenant_id,
            "concept_columns": {
                c.value: list(cols) for c, cols in sorted(self.concept_columns.items())
            },
            "column_classifications": sorted(
                (d.model_dump(mode="json") for d in self.column_classifications),
                key=lambda d: d["column"],
            ),
        }
        # Only present when used, so ledgers issued before families existed
        # keep their fingerprint.
        if self.family_declarations:
            payload["family_declarations"] = sorted(
                (d.model_dump(mode="json") for d in self.family_declarations),
                key=lambda d: d["family"],
            )
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def expanded_declarations(
        self, columns: Iterable[str]
    ) -> tuple[list[ColumnDeclaration], dict[str, str]]:
        """Family declarations expanded over `columns`.

        Returns the per-column declarations and, for any column that two
        families both claim, a reason it was withheld from both: never silently
        pick one of several plausible classifications. A column with an
        explicit `column_classifications` entry keeps that entry and is skipped.
        """
        names = list(dict.fromkeys(columns))
        explicit = {d.column for d in self.column_classifications}
        claimed: dict[str, list[str]] = {}
        for family in self.family_declarations:
            for column in names:
                if column not in explicit and family.matches(column):
                    claimed.setdefault(column, []).append(family.family)
        conflicts = {
            c: "claimed by several family declarations: " + ", ".join(sorted(fams))
            for c, fams in claimed.items()
            if len(fams) > 1
        }
        declarations: list[ColumnDeclaration] = []
        for family in self.family_declarations:
            declarations.extend(
                family.expand(
                    c
                    for c in names
                    if c not in conflicts and family.family in claimed.get(c, ())
                )
            )
        return declarations, conflicts

    def declaration_for(self, column: str) -> ColumnDeclaration | None:
        return next((d for d in self.column_classifications if d.column == column), None)

    @property
    def calendar_resolved(self) -> bool:
        return self.fiscal_year_start_month is not None

    @property
    def promoted_tests(self) -> frozenset[str]:
        return frozenset(self.agreement_declarations)

    def declared(self, concept: BusinessConcept) -> tuple[str, ...]:
        return self.concept_columns.get(concept, ())

    def declares_absent(self, concept: BusinessConcept) -> bool:
        return concept in self.absent_concepts

    @classmethod
    def load(cls, path: Path) -> TenantProfile:
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")
