"""Onboarding: profile a source, PROPOSE declarations, and APPROVE them (WP6).

The tool never confirms anything. `propose` profiles a source (aggregates only:
column names, types, null and distinct counts; the one exception is the value
list of the stage column, which a stage-to-status map cannot be drafted without)
and writes a draft declaration in which every proposed item is marked
`inferred` or `needs_review`. A human edits the draft, flipping items to
`confirmed`. `approve` then refuses the draft while any load-bearing item is
not confirmed (grain, amount, stage, the stage-to-status map, and a proposed
capture policy) and only then registers the dataset with the resulting
`TenantProfile`, so usage grants are issued by the existing path from the human
declaration and never from a proposal.

No example pack is applied here: a column registry is built only from what the
human confirmed.

An optional LLM proposer is out of scope for now and deliberately absent.
"""

from __future__ import annotations

import json
import re
from enum import StrEnum
from pathlib import Path

import duckdb
from pydantic import BaseModel, ConfigDict, Field

from ai_analyst.config import Settings
from ai_analyst.contracts.binding import BindingStatus
from ai_analyst.contracts.columns import (
    Availability,
    CanonicalCoverage,
    ColumnCategory,
    ColumnClassification,
    ColumnRegistry,
    CoverageStatus,
    Disposition,
    QuarantineCode,
    QuarantineReason,
)
from ai_analyst.contracts.concepts import BusinessConcept
from ai_analyst.contracts.schema import CanonicalColumn
from ai_analyst.contracts.snapshot_policy import IntradayPolicy, SnapshotPolicy
from ai_analyst.contracts.source import SourceFormat, TableSource
from ai_analyst.contracts.status import OpportunityStatus, status_from_stage
from ai_analyst.contracts.tenant import (
    FamilyDeclaration,
    FamilyInvariant,
    FamilyInvariantKind,
    TenantProfile,
)
from ai_analyst.data.agreement import run_agreement_test
from ai_analyst.data.binding import CANONICAL_CONCEPTS
from ai_analyst.data.conform import quote_ident
from ai_analyst.data.dataset import Dataset, register_dataset
from ai_analyst.data.families import infer_families
from ai_analyst.data.invariants import invariant_test
from ai_analyst.data.mapping import normalize, propose_mapping
from ai_analyst.data.reconstruct import CLOSE_DATE_DERIVATION
from ai_analyst.data.sources import connect_for, scan_sql
from ai_analyst.data.store import DuckDBStore

DRAFT_VERSION = 1
MAX_STAGE_VALUES = 200
MAX_GRAIN_COLUMNS = 150
MAX_CANDIDATES = 20
_WINDOW_SUFFIX = re.compile(r"^(?P<stem>.+)_(?P<w>\d+d|lifetime)$")


class ReviewStatus(StrEnum):
    """Where a proposed item stands. Only a human writes `confirmed`."""

    INFERRED = "inferred"
    NEEDS_REVIEW = "needs_review"
    CONFIRMED = "confirmed"


class OnboardingRefused(Exception):
    """A draft cannot be approved. `problems` names every reason."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ColumnStats(_Model):
    """Aggregates only. No values."""

    name: str
    type: str
    null_count: int
    distinct_count: int
    numeric_like: bool = False
    temporal_like: bool = False


class SourceProfile(_Model):
    row_count: int
    columns: list[ColumnStats]


class GrainCandidate(_Model):
    id_column: str
    as_of_column: str
    # Rows beyond the first per key: 0 means the pair is unique.
    duplicate_rows_raw: int
    duplicate_rows_day: int
    # Groups holding more than one row when the timestamp is floored to a date.
    duplicate_groups_day: int = 0
    name_hinted: bool = False

    @property
    def unique_raw(self) -> bool:
        return self.duplicate_rows_raw == 0

    @property
    def unique_after_flooring(self) -> bool:
        return self.duplicate_rows_day == 0


class GrainProposal(_Model):
    id_column: str | None = None
    as_of_column: str | None = None
    status: ReviewStatus = ReviewStatus.NEEDS_REVIEW
    evidence: str = ""
    candidates: list[GrainCandidate] = Field(default_factory=list)


class ColumnProposal(_Model):
    column: str | None = None
    status: ReviewStatus = ReviewStatus.NEEDS_REVIEW
    evidence: str = ""
    # Names only, as options for a human. Never values.
    candidates: list[str] = Field(default_factory=list)


class StageStatusEntry(_Model):
    """One stage value and the status it would map to. A proposal until confirmed."""

    value: str
    rows: int
    proposed_status: OpportunityStatus
    proposed_by: str = "keyword_heuristic"
    status: ReviewStatus = ReviewStatus.NEEDS_REVIEW


class StageStatusDraft(_Model):
    column: str | None = None
    entries: list[StageStatusEntry] = Field(default_factory=list)
    null_rows: int = 0
    note: str = ""


class CapturePolicyProposal(_Model):
    policy: SnapshotPolicy = Field(default_factory=SnapshotPolicy)
    status: ReviewStatus = ReviewStatus.NEEDS_REVIEW
    evidence: str = ""
    proposed: bool = False


class CloseDateProposal(_Model):
    source_column: str | None = None
    derivation: str = CLOSE_DATE_DERIVATION
    status: ReviewStatus = ReviewStatus.NEEDS_REVIEW
    evidence: str = ""


class CalendarProposal(_Model):
    fiscal_year_start_month: int | None = None
    status: ReviewStatus = ReviewStatus.NEEDS_REVIEW
    note: str = "Never inferred. A human states the fiscal year start month."


class InvariantProposal(_Model):
    invariant: FamilyInvariant
    status: ReviewStatus = ReviewStatus.INFERRED
    checked_rows: int
    disagreeing_rows: int


class FamilySummary(_Model):
    name: str
    member_count: int
    windows: list[str] = Field(default_factory=list)
    proposed: str


class OnboardingDraft(_Model):
    version: int = DRAFT_VERSION
    source: TableSource
    tenant_id: str
    profile: SourceProfile
    grain: GrainProposal
    amount: ColumnProposal
    stage: ColumnProposal
    stage_status_map: StageStatusDraft
    capture_policy: CapturePolicyProposal
    close_date_reconstruction: CloseDateProposal
    calendar: CalendarProposal
    concepts: dict[str, ColumnProposal] = Field(default_factory=dict)
    invariants: list[InvariantProposal] = Field(default_factory=list)
    families: list[FamilySummary] = Field(default_factory=list)
    # TenantProfile-shaped. Bindings and the status map are empty here: they are
    # filled from the human-confirmed items above at approval. Family
    # declarations carry status `inferred`, which issues no grants.
    tenant: TenantProfile
    warnings: list[str] = Field(default_factory=list)

    def save(self, path: Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> OnboardingDraft:
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))

    # -- approval -----------------------------------------------------------

    def approval_problems(self) -> list[str]:
        """Everything that still blocks approval. Empty means approvable."""
        problems: list[str] = []
        confirmed = ReviewStatus.CONFIRMED

        def need(item: str, status: ReviewStatus, ready: bool = True, why: str = "") -> None:
            if status is not confirmed:
                problems.append(f"{item} is still {status.value}: a human must confirm it")
            elif not ready:
                problems.append(f"{item}: {why}")

        need(
            "grain",
            self.grain.status,
            bool(self.grain.id_column and self.grain.as_of_column),
            "confirmed but names no id and as_of column",
        )
        no_column = "confirmed but names no column"
        need("amount", self.amount.status, bool(self.amount.column), no_column)
        need("stage", self.stage.status, bool(self.stage.column), no_column)

        smap = self.stage_status_map
        if not smap.entries:
            problems.append("stage_status_map has no entries: every stage value must be mapped")
        pending = [e.value for e in smap.entries if e.status is not confirmed]
        if pending:
            problems.append(
                f"stage_status_map has {len(pending)} entries not confirmed "
                f"(still needs_review/inferred): {pending[:5]}"
            )
        unknown = [e.value for e in smap.entries if e.proposed_status is OpportunityStatus.UNKNOWN]
        if unknown:
            problems.append(
                f"stage_status_map maps {len(unknown)} values to 'unknown': map each to "
                "open/won/lost/excluded, or remove it so it is excluded"
            )
        if smap.entries and smap.column != self.stage.column:
            problems.append(
                f"stage_status_map was drafted for column {smap.column!r} but the stage "
                f"column is {self.stage.column!r}: re-run the proposal with --stage-column"
            )
        if self.capture_policy.proposed:
            need("capture_policy", self.capture_policy.status)
        return problems

    def registration(self) -> Registration:
        problems = self.approval_problems()
        if problems:
            raise OnboardingRefused(problems)
        confirmed = ReviewStatus.CONFIRMED
        overrides: dict[str, CanonicalColumn] = {
            self.grain.as_of_column: CanonicalColumn.AS_OF,  # type: ignore[dict-item]
            self.grain.id_column: CanonicalColumn.OPP_ID,  # type: ignore[dict-item]
            self.amount.column: CanonicalColumn.AMOUNT,  # type: ignore[dict-item]
            self.stage.column: CanonicalColumn.STAGE,  # type: ignore[dict-item]
        }
        if len(overrides) != 4:
            raise OnboardingRefused(["grain, amount and stage must be four distinct columns"])
        used = set(overrides.values())

        concept_columns: dict[BusinessConcept, tuple[str, ...]] = {
            BusinessConcept.AMOUNT: (self.amount.column,),  # type: ignore[dict-item]
            BusinessConcept.STAGE: (self.stage.column,),  # type: ignore[dict-item]
        }
        for canonical_name, proposal in self.concepts.items():
            if proposal.status is not confirmed or not proposal.column:
                continue
            canonical = CanonicalColumn(canonical_name)
            if canonical in used or proposal.column in overrides:
                raise OnboardingRefused(
                    [f"{canonical_name}: column {proposal.column!r} or concept already bound"]
                )
            overrides[proposal.column] = canonical
            used.add(canonical)
            concept = CANONICAL_CONCEPTS.get(canonical)
            if concept is not None and concept not in (
                BusinessConcept.OPPORTUNITY_STATUS,
                BusinessConcept.OPPORTUNITY_ID,
                BusinessConcept.SNAPSHOT_DATE,
            ):
                concept_columns[concept] = (proposal.column,)

        registry: ColumnRegistry | None = None
        cd = self.close_date_reconstruction
        if (
            cd.status is confirmed
            and cd.source_column
            and CanonicalColumn.CLOSE_DATE not in used
        ):
            registry = ColumnRegistry(
                name=f"declared:{self.tenant_id}",
                columns=[],
                coverage=[
                    CanonicalCoverage(
                        canonical=CanonicalColumn.CLOSE_DATE.value,
                        status=CoverageStatus.RECONSTRUCTIBLE,
                        source=cd.source_column,
                        derivation=cd.derivation,
                        note="Declared by a human at onboarding approval.",
                    )
                ],
            )

        cal = self.calendar
        fiscal = (
            cal.fiscal_year_start_month
            if cal.status is confirmed and cal.fiscal_year_start_month
            else self.tenant.fiscal_year_start_month
        )
        base = self.tenant.model_dump(mode="python")
        base.update(
            concept_columns=concept_columns,
            stage_status_map={e.value: e.proposed_status for e in self.stage_status_map.entries},
            snapshot_policy=(
                self.capture_policy.policy
                if self.capture_policy.status is confirmed
                else self.tenant.snapshot_policy
            ),
            fiscal_year_start_month=fiscal,
            family_invariants=[
                p.invariant for p in self.invariants if p.status is confirmed
            ],
            source=f"onboarding approval of {self.source.uri}",
        )
        return Registration(
            source=self.source,
            tenant=TenantProfile.model_validate(base),
            mapping_overrides=overrides,
            column_registry=registry,
        )


class Registration(BaseModel):
    """What `register_dataset` needs, built only from confirmed items."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source: TableSource
    tenant: TenantProfile
    mapping_overrides: dict[str, CanonicalColumn]
    column_registry: ColumnRegistry | None = None


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------

_NUMERIC_TYPES = ("TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "UBIGINT", "UINTEGER")
_FLOAT_TYPES = ("FLOAT", "DOUBLE", "DECIMAL")


def _profile(conn: duckdb.DuckDBPyConnection, read_expr: str) -> SourceProfile:
    described = conn.execute(f"DESCRIBE SELECT * FROM {read_expr}").fetchall()
    names = [r[0] for r in described]
    types = {r[0]: str(r[1]) for r in described}
    parts = ["COUNT(*)"]
    for name in names:
        c = quote_ident(name)
        as_text = f"CAST({c} AS VARCHAR)"
        parts += [
            f"COUNT({c})",
            f"COUNT(DISTINCT {c})",
            f"COUNT(TRY_CAST({as_text} AS DOUBLE))",
            f"COUNT(TRY_CAST({as_text} AS TIMESTAMP))",
        ]
    row = conn.execute(f"SELECT {', '.join(parts)} FROM {read_expr}").fetchone()
    n = int(row[0])
    stats: list[ColumnStats] = []
    for i, name in enumerate(names):
        nn, distinct, num, ts = (int(v) for v in row[1 + 4 * i : 5 + 4 * i])
        dtype = types[name]
        temporal = dtype.startswith(("DATE", "TIMESTAMP")) or (
            dtype == "VARCHAR" and nn > 0 and ts == nn and num < nn
        )
        numeric = not temporal and dtype != "BOOLEAN" and nn > 0 and (
            dtype.startswith(_NUMERIC_TYPES + _FLOAT_TYPES) or (dtype == "VARCHAR" and num == nn)
        )
        stats.append(
            ColumnStats(
                name=name,
                type=dtype,
                null_count=n - nn,
                distinct_count=distinct,
                numeric_like=numeric,
                temporal_like=temporal,
            )
        )
    return SourceProfile(row_count=n, columns=stats)


def _ts(column: str) -> str:
    return f"TRY_CAST(CAST({quote_ident(column)} AS VARCHAR) AS TIMESTAMP)"


def _propose_grain(
    conn: duckdb.DuckDBPyConnection,
    read_expr: str,
    profile: SourceProfile,
    hints: dict[str, CanonicalColumn],
) -> GrainProposal:
    n = profile.row_count
    order = {c.name: i for i, c in enumerate(profile.columns)}
    temporal = [c.name for c in profile.columns if c.temporal_like and c.null_count == 0]
    others = [
        c.name
        for c in profile.columns
        if not c.temporal_like
        and c.null_count == 0
        and 2 <= c.distinct_count < n  # a column unique on its own is a row id, not a key part
    ][:MAX_GRAIN_COLUMNS]
    if not temporal or not others or n == 0:
        return GrainProposal(
            evidence="no (id, timestamp) column pair could be searched: need a fully-populated "
            "date/timestamp column and a fully-populated key-like column"
        )
    found: list[GrainCandidate] = []
    for t in temporal:
        aggs = []
        for o in others:
            key = f"CAST({quote_ident(o)} AS VARCHAR)"
            aggs += [
                f"COUNT(DISTINCT ({key}, {_ts(t)}))",
                f"COUNT(DISTINCT ({key}, CAST({_ts(t)} AS DATE)))",
            ]
        row = conn.execute(f"SELECT {', '.join(aggs)} FROM {read_expr}").fetchone()
        for i, o in enumerate(others):
            found.append(
                GrainCandidate(
                    id_column=o,
                    as_of_column=t,
                    duplicate_rows_raw=n - int(row[2 * i]),
                    duplicate_rows_day=n - int(row[2 * i + 1]),
                    name_hinted=(
                        hints.get(o) is CanonicalColumn.OPP_ID
                        or hints.get(t) is CanonicalColumn.AS_OF
                    ),
                )
            )
    found.sort(
        key=lambda g: (
            g.duplicate_rows_day,
            g.duplicate_rows_raw,
            not g.name_hinted,
            order[g.id_column] + order[g.as_of_column],
        )
    )
    top = found[:MAX_CANDIDATES]
    for g in top[:5]:
        if g.duplicate_rows_day:
            groups = conn.execute(
                f"SELECT COUNT(*) FROM (SELECT 1 FROM {read_expr} "
                f"GROUP BY {quote_ident(g.id_column)}, CAST({_ts(g.as_of_column)} AS DATE) "
                "HAVING COUNT(*) > 1)"
            ).fetchone()[0]
            g.duplicate_groups_day = int(groups)
    best = top[0]
    return GrainProposal(
        id_column=best.id_column,
        as_of_column=best.as_of_column,
        status=ReviewStatus.INFERRED,
        evidence=(
            f"({best.id_column}, {best.as_of_column}): {best.duplicate_rows_raw} duplicate rows "
            f"at full precision, {best.duplicate_rows_day} after flooring to a date. Ranked by "
            "uniqueness, then name hints (which never confirm), then column order."
        ),
        candidates=top,
    )


def _propose_capture(grain: GrainProposal) -> CapturePolicyProposal:
    if grain.id_column is None or not grain.candidates:
        return CapturePolicyProposal()
    best = grain.candidates[0]
    if best.duplicate_rows_day == 0:
        return CapturePolicyProposal(evidence="grain is unique per day; no policy needed")
    if best.duplicate_rows_raw == 0:
        return CapturePolicyProposal(
            policy=SnapshotPolicy(intraday_policy=IntradayPolicy.LATEST_CAPTURE),
            status=ReviewStatus.NEEDS_REVIEW,
            proposed=True,
            evidence=(
                f"{best.duplicate_groups_day} (as_of, opp) groups hold more than one capture on "
                f"one day ({best.duplicate_rows_day} extra rows), distinguishable by timestamp. "
                "latest_capture keeps the latest and reports what was dropped."
            ),
        )
    return CapturePolicyProposal(
        evidence=(
            "duplicate keys remain even at full timestamp precision; no declared policy can "
            "resolve them and ingestion will fail loudly"
        )
    )


def _stage_draft(
    conn: duckdb.DuckDBPyConnection, read_expr: str, column: str | None
) -> StageStatusDraft:
    if column is None:
        return StageStatusDraft(
            note="no stage column proposed: re-run with --stage-column NAME to list its values"
        )
    ident = quote_ident(column)
    rows = conn.execute(
        f"SELECT CAST({ident} AS VARCHAR), COUNT(*) FROM {read_expr} "
        "GROUP BY 1 ORDER BY 2 DESC, 1"
    ).fetchall()
    entries = [
        StageStatusEntry(value=v, rows=int(c), proposed_status=status_from_stage(v))
        for v, c in rows
        if v is not None
    ][:MAX_STAGE_VALUES]
    nulls = sum(int(c) for v, c in rows if v is None)
    note = (
        "proposed_status comes from stage keyword heuristics, which are non-authoritative "
        "(they read '6 - Order Placed' as open). Edit each value's proposed_status to the "
        "true status and set its status to confirmed. Values left out are EXCLUDED."
    )
    if len(rows) - (1 if nulls else 0) > MAX_STAGE_VALUES:
        note += f" Truncated to the {MAX_STAGE_VALUES} most frequent values."
    return StageStatusDraft(column=column, entries=entries, null_rows=nulls, note=note)


def _family_summaries(names: list[str]) -> tuple[list[FamilySummary], list[FamilyDeclaration]]:
    summaries: list[FamilySummary] = []
    declarations: list[FamilyDeclaration] = []
    for p in infer_families(names):
        if p.is_singleton:
            continue
        c = p.proposed
        summaries.append(
            FamilySummary(
                name=p.name,
                member_count=len(p.members),
                windows=list(p.windows),
                proposed=f"{c.category.value}/{c.availability.value}/{c.disposition.value}",
            )
        )
        declarations.append(
            FamilyDeclaration(
                family=p.name,
                pattern=p.member_pattern,
                classification=ColumnClassification(
                    name=p.name,
                    category=ColumnCategory(c.category),
                    availability=Availability(c.availability),
                    disposition=Disposition(c.disposition),
                    note=c.reason,
                    quarantine=(
                        QuarantineReason(
                            code=QuarantineCode.UNCLASSIFIED_COLUMN,
                            detail=f"proposed from column names only: {c.reason}",
                            resolution="a human confirms this family's classification",
                        )
                        if Disposition(c.disposition) is Disposition.QUARANTINE
                        else None
                    ),
                ),
                status=BindingStatus.INFERRED,
                source="onboarding proposal from column names; unconfirmed",
            )
        )
    return summaries, declarations


# ---------------------------------------------------------------------------
# Invariant proposals: only those that hold on the data
# ---------------------------------------------------------------------------


def _window_key(w: str) -> tuple[int, int]:
    return (1, 0) if w == "lifetime" else (0, int(w[:-1]))


def _holds(
    conn: duckdb.DuckDBPyConnection, inv: FamilyInvariant, numeric: set[str]
) -> tuple[bool, int, int]:
    result = run_agreement_test(
        conn, "_onb_numeric", invariant_test(inv, numeric), numeric, max_samples=0
    )
    return (
        (not result.skipped and result.checked_rows > 0 and result.disagreeing_rows == 0),
        result.checked_rows,
        result.disagreeing_rows,
    )


def _propose_invariants(
    conn: duckdb.DuckDBPyConnection, read_expr: str, profile: SourceProfile
) -> list[InvariantProposal]:
    numeric = {c.name for c in profile.columns if c.numeric_like}
    if not numeric:
        return []
    cols = ", ".join(
        f"TRY_CAST(CAST({quote_ident(c)} AS VARCHAR) AS DOUBLE) AS {quote_ident(c)}"
        for c in sorted(numeric)
    )
    conn.execute(f"CREATE OR REPLACE TEMP VIEW _onb_numeric AS SELECT {cols} FROM {read_expr}")

    stems: dict[str, list[str]] = {}
    for name in sorted(numeric):
        m = _WINDOW_SUFFIX.match(name)
        if m:
            stems.setdefault(m["stem"], []).append(m["w"])
    windowed = {s: tuple(sorted(w, key=_window_key)) for s, w in stems.items()}

    proposals: list[InvariantProposal] = []

    def keep(inv: FamilyInvariant) -> None:
        ok, checked, bad = _holds(conn, inv, numeric)
        if ok:
            proposals.append(
                InvariantProposal(invariant=inv, checked_rows=checked, disagreeing_rows=bad)
            )

    for stem, windows in sorted(windowed.items()):
        if len(windows) >= 2:
            keep(
                FamilyInvariant(
                    id=f"{stem}_window_monotone",
                    kind=FamilyInvariantKind.WINDOW_MONOTONE,
                    template=f"{stem}_{{w}}",
                    windows=windows,
                    source="onboarding proposal; holds on the observed data",
                )
            )

    # parts_sum: a + b = c per window, over stems sharing one window set.
    by_windows: dict[tuple[str, ...], list[str]] = {}
    for stem, windows in windowed.items():
        by_windows.setdefault(windows, []).append(stem)
    for windows, group in by_windows.items():
        group = sorted(group)[:12]
        w0 = windows[0]
        combos = [
            (a, b, c)
            for i, a in enumerate(group)
            for b in group[i + 1 :]
            for c in group
            if c not in (a, b)
        ]
        if not combos:
            continue
        aggs = []
        for a, b, c in combos:
            qa, qb, qc = (quote_ident(f"{s}_{w0}") for s in (a, b, c))
            aggs += [
                f"COUNT(*) FILTER (WHERE ({qa} + {qb}) <> {qc})",
                f"COUNT(*) FILTER (WHERE ({qa} + {qb}) = {qc} AND {qc} <> 0)",
            ]
        row = conn.execute(f"SELECT {', '.join(aggs)} FROM _onb_numeric").fetchone()
        for i, (a, b, c) in enumerate(combos):
            if row[2 * i] == 0 and row[2 * i + 1] > 0:
                keep(
                    FamilyInvariant(
                        id=f"{a}+{b}={c}_parts_sum",
                        kind=FamilyInvariantKind.PARTS_SUM,
                        parts=(f"{a}_{{w}}", f"{b}_{{w}}"),
                        whole=f"{c}_{{w}}",
                        windows=windows,
                        source="onboarding proposal; holds on the observed data",
                    )
                )

    names = {c.name for c in profile.columns}
    for name in sorted(names):
        m = re.match(r"^(?P<label>.+_label)_mask$", name)
        if m and m["label"] in numeric and name in numeric:
            keep(
                FamilyInvariant(
                    id=f"{m['label']}_mask_implies_null",
                    kind=FamilyInvariantKind.MASK_IMPLIES_NULL,
                    label=m["label"],
                    mask=name,
                    source="onboarding proposal; holds on the observed data",
                )
            )
    return proposals


# ---------------------------------------------------------------------------
# Propose
# ---------------------------------------------------------------------------


def _column_proposal(
    canonical: CanonicalColumn,
    exact: dict[CanonicalColumn, tuple[str, str]],
) -> ColumnProposal:
    if canonical in exact:
        column, how = exact[canonical]
        return ColumnProposal(
            column=column,
            status=ReviewStatus.INFERRED,
            evidence=f"{how} header match. A name never confirms a binding.",
        )
    return ColumnProposal(evidence="no header suggests this; a human names the column")


def propose(
    source: TableSource,
    *,
    tenant_id: str | None = None,
    stage_column: str | None = None,
) -> OnboardingDraft:
    """Profile `source` and draft a declaration. Nothing in the draft is confirmed.

    `stage_column` is a human-directed choice of which column's values to list
    for the stage-to-status map; without it only a header-suggested stage column
    is listed.
    """
    conn = connect_for(source)
    try:
        read_expr = scan_sql(source)
        profile = _profile(conn, read_expr)
        names = [c.name for c in profile.columns]
        proposal = propose_mapping(names)
        hints = {m.source_column: m.canonical_column for m in proposal.mappings}
        exact = {
            m.canonical_column: (m.source_column, m.confidence.value) for m in proposal.mappings
        }
        for m in proposal.fuzzy_candidates:
            exact.setdefault(m.canonical_column, (m.source_column, "fuzzy"))

        grain = _propose_grain(conn, read_expr, profile, hints)
        capture = _propose_capture(grain)

        amount = _column_proposal(CanonicalColumn.AMOUNT, exact)
        if amount.column is None:
            taken = {grain.id_column, grain.as_of_column}
            amount = amount.model_copy(
                update={
                    "candidates": [
                        c.name for c in profile.columns if c.numeric_like and c.name not in taken
                    ][:MAX_CANDIDATES]
                }
            )

        stage = _column_proposal(CanonicalColumn.STAGE, exact)
        if stage_column is not None:
            if stage_column not in names:
                raise ValueError(f"--stage-column {stage_column!r} is not a column of the source")
            stage = ColumnProposal(
                column=stage_column,
                status=ReviewStatus.INFERRED,
                evidence="named by the operator; its meaning is unconfirmed",
            )
        elif stage.column is None:
            stage = stage.model_copy(
                update={
                    "candidates": [
                        c.name
                        for c in profile.columns
                        if c.type == "VARCHAR"
                        and not c.temporal_like
                        and not c.numeric_like
                        and 2 <= c.distinct_count <= 50
                    ][:MAX_CANDIDATES]
                }
            )
        stage_map = _stage_draft(conn, read_expr, stage.column)

        by_name = {c.name: c for c in profile.columns}
        close = CloseDateProposal()
        if CanonicalColumn.CLOSE_DATE not in exact:
            for n in names:
                if normalize(n) == "days_to_close" and by_name[n].numeric_like:
                    close = CloseDateProposal(
                        source_column=n,
                        evidence=(
                            f"no close-date column found and {n!r} looks like a day "
                            "horizon. Reconstruction must be declared by a human."
                        ),
                    )
                    break

        skip = {
            CanonicalColumn.AS_OF,
            CanonicalColumn.OPP_ID,
            CanonicalColumn.AMOUNT,
            CanonicalColumn.STAGE,
            CanonicalColumn.IS_CLOSED,
            CanonicalColumn.IS_WON,
            CanonicalColumn.STATUS,
        }
        concepts = {
            canonical.value: _column_proposal(canonical, exact)
            for canonical in exact
            if canonical not in skip
        }

        invariants = _propose_invariants(conn, read_expr, profile)
        summaries, declarations = _family_summaries(names)
    finally:
        conn.close()

    warnings: list[str] = []
    if grain.id_column is None:
        warnings.append("no grain candidate found; a human must name the id and as_of columns")
    if profile.row_count == 0:
        warnings.append("the source has no rows")
    if stage_map.null_rows:
        warnings.append(f"{stage_map.null_rows} rows have a null stage; they will be excluded")

    tid = tenant_id or Path(source.uri.rstrip("/")).stem or "tenant"
    return OnboardingDraft(
        source=source,
        tenant_id=tid,
        profile=profile,
        grain=grain,
        amount=amount,
        stage=stage,
        stage_status_map=stage_map,
        capture_policy=capture,
        close_date_reconstruction=close,
        calendar=CalendarProposal(),
        concepts=concepts,
        invariants=invariants,
        families=summaries,
        tenant=TenantProfile(
            tenant_id=tid,
            family_declarations=declarations,
            source="onboarding draft: nothing confirmed",
        ),
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Approve
# ---------------------------------------------------------------------------


def approve(
    draft: OnboardingDraft,
    dataset_id: str,
    *,
    settings: Settings | None = None,
    store: DuckDBStore | None = None,
) -> tuple[Dataset, Registration]:
    """Register a dataset from a human-approved draft, or refuse with every reason."""
    registration = draft.registration()
    dataset = register_dataset(
        registration.source,
        dataset_id,
        column_registry=registration.column_registry,
        mapping_overrides=registration.mapping_overrides,
        tenant=registration.tenant,
        settings=settings,
        store=store,
    )
    return dataset, registration


def summarize(draft: OnboardingDraft) -> str:
    """A console summary: names, types and counts. No column values."""
    p = draft.profile
    lines = [
        f"source: {draft.source.uri} ({draft.source.format.value})",
        f"{p.row_count} rows, {len(p.columns)} columns",
        "",
        "grain candidates (id, as_of; duplicate rows raw/after flooring to a date):",
    ]
    for g in draft.grain.candidates[:5]:
        lines.append(
            f"  {g.id_column}, {g.as_of_column}: {g.duplicate_rows_raw}/{g.duplicate_rows_day}"
            + (f" ({g.duplicate_groups_day} groups)" if g.duplicate_groups_day else "")
        )
    if draft.capture_policy.proposed:
        lines.append(f"capture policy proposed: latest_capture. {draft.capture_policy.evidence}")
    lines.append(
        f"amount: {draft.amount.column or 'unknown'} ({draft.amount.status.value}); "
        f"stage: {draft.stage.column or 'unknown'} ({draft.stage.status.value})"
    )
    lines.append(
        f"stage_status_map: {len(draft.stage_status_map.entries)} values, all needs_review"
    )
    lines.append(f"{len(draft.families)} families, {len(draft.invariants)} invariants that hold")
    lines += [f"warning: {w}" for w in draft.warnings]
    lines.append("")
    lines.append("Nothing is confirmed. Edit the draft, mark items confirmed, then --approve.")
    return "\n".join(lines)


__all__ = [
    "OnboardingDraft",
    "OnboardingRefused",
    "Registration",
    "ReviewStatus",
    "SourceFormat",
    "approve",
    "propose",
    "summarize",
]
