"""The analyst context card (ARCHITECTURE 12.5).

The persisted profile is roughly 15k tokens for 145 columns and must never be
injected into a prompt. The card is a *projection* of it, in three tiers:

* **Tier 0** is concepts, not columns, plus a name-only column index. It is the
  cached prompt prefix and is budgeted in tokens at build time.
* **Tier 1** is one column's profile and classification, returned on request.
* **Tier 2** is rows, which never enter the default context at all and are
  reachable only through an explicit, capped inspection path.

Tier 0 lists what is *missing* as prominently as what is present, because an
unanswerable question should be refused from the card rather than from a failed
query.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from ai_analyst.contracts.agreement import AgreementResult
from ai_analyst.contracts.binding import BindingStatus
from ai_analyst.contracts.columns import MonetaryStatus
from ai_analyst.contracts.concepts import BusinessConcept
from ai_analyst.contracts.profile import ColumnProfile, ProfileKind
from ai_analyst.contracts.schema import DataType
from ai_analyst.contracts.status import StatusStrategy

# Rough characters-per-token for English prose and identifiers. Deliberately
# conservative: a real tokenizer is not a dependency of this layer, and a
# budget that under-counts would be worthless.
CHARS_PER_TOKEN = 3.6


def estimate_tokens(text: str) -> int:
    """A conservative token estimate for a budget check."""
    return int(len(text) / CHARS_PER_TOKEN) + 1


def _sample(names: tuple[str, ...], limit: int = 8) -> str:
    """The first few names, with a pointer to the index for the rest."""
    shown = ", ".join(names[:limit])
    remaining = len(names) - limit
    return shown if remaining <= 0 else f"{shown}, +{remaining} more in the index"


class ConceptCard(BaseModel):
    """One concept's line in tier 0."""

    model_config = ConfigDict(frozen=True)

    concept: BusinessConcept
    display_name: str
    definition: str
    columns: tuple[str, ...] = ()
    status: BindingStatus
    alternatives: tuple[str, ...] = ()
    caveats: tuple[str, ...] = ()
    note: str = ""

    def render(self) -> str:
        if self.status is BindingStatus.AMBIGUOUS:
            target = f"AMBIGUOUS between {', '.join(self.alternatives)}"
        elif self.status is BindingStatus.UNAVAILABLE:
            target = "UNAVAILABLE"
        else:
            target = ", ".join(self.columns)
        line = f"  {self.concept.value:<20} -> {target:<28} {self.status.value}"
        if self.caveats:
            line += f" [{'; '.join(self.caveats)}]"
        elif self.status is BindingStatus.UNAVAILABLE and self.note:
            line += f" ({self.note})"
        return line


class TextFieldCard(BaseModel):
    """A narrative field, named and measured. Never its content."""

    model_config = ConfigDict(frozen=True)

    name: str
    classified: bool
    knowable_at_snapshot: bool
    null_rate: float = 0.0
    mean_length: float | None = None
    max_length: int | None = None


class QualityWarning(BaseModel):
    """Something about the data a reader must know before trusting a number."""

    model_config = ConfigDict(frozen=True)

    code: str
    detail: str


class ColumnDetail(BaseModel):
    """Tier 1: everything known about one column, on request."""

    model_config = ConfigDict(frozen=True)

    name: str
    origin: str
    source_name: str | None = None
    dtype: DataType
    category: str
    information_class: str
    availability: str
    disposition: str
    knowable_at_snapshot: bool
    quarantined: bool = False
    quarantine_code: str | None = None
    quarantine_detail: str | None = None
    monetary: MonetaryStatus = MonetaryStatus.NON_MONETARY
    lineage_confirmed: bool | None = None
    unresolved: tuple[str, ...] = ()
    bound_to: BusinessConcept | None = None
    profile: ColumnProfile | None = None

    def render(self) -> str:
        lines = [
            f"{self.name} ({self.dtype.value}, {self.origin})",
            f"  means   : {self.category} / {self.information_class}",
            f"  readable: {self.availability}, disposition {self.disposition}",
        ]
        if self.source_name and self.source_name != self.name:
            lines.append(f"  source  : {self.source_name}")
        if self.bound_to is not None:
            lines.append(f"  concept : {self.bound_to.value}")
        if self.quarantined:
            lines.append(f"  WITHHELD: {self.quarantine_code} - {self.quarantine_detail}")
        if self.monetary is not MonetaryStatus.NON_MONETARY:
            lines.append(f"  money   : {self.monetary.value} (no float summary statistics)")
        if self.unresolved:
            lines.append(f"  open    : {', '.join(self.unresolved)}")
        p = self.profile
        if p is not None:
            lines.append(
                f"  values  : {p.row_count} rows, {p.null_count} null, "
                f"{p.distinct_count} distinct"
            )
            if p.min_value is not None:
                lines.append(f"  range   : {p.min_value} .. {p.max_value}")
            if p.mean is not None:
                lines.append(f"  mean    : {p.mean}")
            if p.sentinel_count:
                lines.append(
                    f"  sentinel: {p.sentinel_count} rows hold "
                    f"{list(p.sentinel_values)} and are excluded from statistics"
                )
            if p.kind is ProfileKind.TEXT:
                lines.append(
                    f"  text    : mean length {p.mean_length}, max {p.max_length}. "
                    "No content is stored."
                )
            elif p.top_values:
                shown = ", ".join(f"{t.value}={t.count}" for t in p.top_values[:8])
                lines.append(f"  top     : {shown}")
        return "\n".join(lines)


class AnalystContext(BaseModel):
    """Tier 0: the dataset as the model first sees it."""

    model_config = ConfigDict(frozen=True)

    dataset_id: str
    row_count: int
    snapshot_count: int
    first_snapshot: date | None = None
    last_snapshot: date | None = None
    grain: tuple[str, str] = ("snapshot_date", "opportunity_id")
    fiscal_year_start_month: int = 1

    concepts: tuple[ConceptCard, ...] = ()
    dimensions: tuple[str, ...] = ()
    measures: tuple[str, ...] = ()
    text_fields: tuple[TextFieldCard, ...] = ()

    column_count: int = 0
    class_counts: dict[str, int] = Field(default_factory=dict)
    quarantined_count: int = 0
    # Names only, grouped by information class. Dropped above a width
    # threshold, at which point the agent reaches names through a listing tool.
    column_index: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    index_degraded: bool = False

    status_strategy: StatusStrategy | None = None
    status_is_authoritative: bool = False
    open_questions: tuple[str, ...] = ()
    quality_warnings: tuple[QualityWarning, ...] = ()
    agreement: tuple[AgreementResult, ...] = ()
    # One line per reconstructed concept: its 13.1 verdict and why.
    reconstruction_verdicts: tuple[str, ...] = ()
    # False when the fiscal year start is the configured default rather than a
    # tenant declaration. Quarter boundaries then rest on an assumption.
    fiscal_calendar_resolved: bool = False

    def available_concepts(self) -> list[ConceptCard]:
        return [c for c in self.concepts if c.status.is_bound]

    def unavailable_concepts(self) -> list[ConceptCard]:
        return [c for c in self.concepts if c.status is BindingStatus.UNAVAILABLE]

    def ambiguous_concepts(self) -> list[ConceptCard]:
        return [c for c in self.concepts if c.status is BindingStatus.AMBIGUOUS]

    def render(self) -> str:
        """The card as it enters the prompt."""
        span = (
            f"{self.first_snapshot} .. {self.last_snapshot}"
            if self.first_snapshot
            else "unknown"
        )
        out = [
            f"dataset: {self.dataset_id}   rows: {self.row_count}   "
            f"snapshots: {self.snapshot_count} ({span})",
            f"grain: ({self.grain[0]}, {self.grain[1]})   "
            f"fiscal year starts month {self.fiscal_year_start_month}"
            + ("" if self.fiscal_calendar_resolved else " (UNRESOLVED: configured default)"),
            "",
            "concepts",
        ]
        out.extend(c.render() for c in self.concepts)

        # Names are listed once, in the column index. Here we give the count
        # and a sample, so the card says what is *available* without paying for
        # the same 90 identifiers twice.
        if self.dimensions:
            out += ["", f"dimensions ({len(self.dimensions)}): {_sample(self.dimensions)}"]
        if self.measures:
            out += [f"measures   ({len(self.measures)}): {_sample(self.measures)}"]
        if self.text_fields:
            names = tuple(t.name for t in self.text_fields)
            out += [
                f"text       ({len(names)}): {_sample(names)} - catalogued only, "
                "no content"
            ]

        counts = " | ".join(f"{k} {v}" for k, v in sorted(self.class_counts.items()))
        out += ["", f"columns: {self.column_count}   {counts}"]
        out += [f"quarantined: {self.quarantined_count} (withheld from analysis)"]

        if self.index_degraded:
            out += [
                "column index withheld: this dataset is too wide for the card. "
                "Use the column listing tool with a class or name filter."
            ]
        elif self.column_index:
            for information_class, names in self.column_index.items():
                out.append(f"  {information_class}: {', '.join(names)}")

        strategy = self.status_strategy.value if self.status_strategy else "unresolved"
        authority = "authoritative" if self.status_is_authoritative else "NOT authoritative"
        out += ["", f"status: resolved by {strategy} ({authority})"]

        if self.agreement:
            out += ["", "agreement tests"]
            out.extend(
                f"  {r.test_id} [{r.role.value}]: {r.summary}" for r in self.agreement
            )
        if self.reconstruction_verdicts:
            out += ["", "reconstructions"]
            out.extend(f"  {line}" for line in self.reconstruction_verdicts)
        if self.quality_warnings:
            out += ["", "data quality"]
            out.extend(f"  {w.code}: {w.detail}" for w in self.quality_warnings)
        if self.open_questions:
            out += ["", f"open questions: {', '.join(self.open_questions)}"]
        return "\n".join(out)

    @property
    def estimated_tokens(self) -> int:
        return estimate_tokens(self.render())
