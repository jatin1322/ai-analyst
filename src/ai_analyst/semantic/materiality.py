"""The clarification materiality probe (ARCHITECTURE 13.12).

When a question has two or more plausible readings, the probe answers one
question deterministically: *would the reading change the answer?* It runs each
reading as a typed edit of the base plan, through the full gate and executor,
and compares the results.

* `MATERIAL`: the readings disagree beyond the threshold, or produce different
  groups. Ask the user.
* `NOT_MATERIAL`: every reading gives the same numbers within the threshold.
  A documented default may be used, with disclosure.
* `INCONCLUSIVE`: the probe could not safely evaluate the ambiguity. Ask.

What the probe will not do is as important as what it does:

* **It never chooses.** The result has no field naming a winning reading.
* **It never substitutes a concept.** A reading may change the snapshot rule,
  the period, a filter, or a documented 5.3 option, and nothing else. A reading
  that changes the metric, the stance, the dimensions or the measure is refused
  as outside the probe's rules and returns `INCONCLUSIVE`.
* **It never evaluates an unconfirmed binding.** Comparing two candidate
  columns for a concept would mean computing a metric on a binding nobody has
  confirmed. Binding ambiguities are `INCONCLUSIVE` by rule: a declaration
  resolves them, not a comparison of numbers.
* **It is bounded.** At most four readings, each result at most fifty rows,
  and every reading under the base plan's own stance and knowledge horizon.
  There is no per-query time limit yet: `query_timeout_seconds` is configured
  but not enforced by the executor.
"""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from ai_analyst.config import Settings
from ai_analyst.contracts.plan import AnalysisPlan
from ai_analyst.contracts.result import ResultSet, TrustTier
from ai_analyst.contracts.session import (
    ChangePeriod,
    ChangeSnapshotRule,
    EditOperation,
    PlanEdit,
    RemoveFilter,
    SetAnalysisOption,
    SetFilter,
)
from ai_analyst.session.edits import EditConflict, apply_edit

MAX_INTERPRETATIONS = 4
MAX_RESULT_ROWS = 50


class Materiality(StrEnum):
    MATERIAL = "material"
    NOT_MATERIAL = "not_material"
    INCONCLUSIVE = "inconclusive"


class AmbiguityKind(StrEnum):
    SNAPSHOT = "snapshot"
    PERIOD = "period"
    FILTER = "filter"
    ANALYSIS_OPTION = "analysis_option"
    CONCEPT_BINDING = "concept_binding"


# The only edits a reading may make, per kind of ambiguity. Anything else would
# change the question rather than choose between readings of it.
ALLOWED_OPERATIONS: dict[AmbiguityKind, tuple[type, ...]] = {
    AmbiguityKind.SNAPSHOT: (ChangeSnapshotRule,),
    AmbiguityKind.PERIOD: (ChangePeriod,),
    AmbiguityKind.FILTER: (SetFilter, RemoveFilter),
    AmbiguityKind.ANALYSIS_OPTION: (SetAnalysisOption,),
    AmbiguityKind.CONCEPT_BINDING: (),
}


class Interpretation(BaseModel):
    """One reading of an ambiguous question, as a typed edit of the base plan."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str = Field(min_length=1)
    # Empty means the base plan exactly as it is.
    operations: list[EditOperation] = Field(default_factory=list)


class MaterialityProbe(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    ambiguity: AmbiguityKind
    interpretations: list[Interpretation] = Field(min_length=2, max_length=MAX_INTERPRETATIONS)
    # Readings within this relative difference are the same answer. Structural,
    # never a result.
    relative_threshold: Decimal = Field(default=Decimal("0.01"), ge=0, le=1)


class ProbeReading(BaseModel):
    model_config = ConfigDict(frozen=True)

    label: str
    row_count: int
    trust_tier: TrustTier
    # Group key to metric cells, as exact strings. Evidence, never an answer.
    cells: dict[str, dict[str, str | None]]


class MaterialityResult(BaseModel):
    """The verdict and its evidence. Deliberately has no 'chosen reading' field."""

    model_config = ConfigDict(frozen=True)

    verdict: Materiality
    ambiguity: AmbiguityKind
    reason: str
    readings: tuple[ProbeReading, ...] = ()
    largest_relative_difference: Decimal | None = None
    executed: int = 0


def _inconclusive(probe: MaterialityProbe, reason: str, **kw) -> MaterialityResult:
    return MaterialityResult(
        verdict=Materiality.INCONCLUSIVE, ambiguity=probe.ambiguity, reason=reason, **kw
    )


def outside_rules(probe: MaterialityProbe) -> str | None:
    """Why a probe asks for something it may not evaluate, or None."""
    if probe.ambiguity is AmbiguityKind.CONCEPT_BINDING:
        return (
            "a binding ambiguity changes which column carries a concept; evaluating "
            "it would compute a metric on an unconfirmed binding, so a declaration "
            "resolves it, not a comparison"
        )
    allowed = ALLOWED_OPERATIONS[probe.ambiguity]
    for interpretation in probe.interpretations:
        for operation in interpretation.operations:
            if not isinstance(operation, allowed):
                return (
                    f"reading {interpretation.label!r} makes a {operation.op} edit, which "
                    f"changes the question rather than resolving a {probe.ambiguity.value} "
                    "ambiguity"
                )
    return None


def _cells(result: ResultSet) -> dict[str, dict[str, str | None]]:
    """Rows keyed by their non-numeric (group) columns; numeric cells as exact text."""
    numeric = {"money", "count", "quantity", "ratio"}
    keys = [c.name for c in result.columns if c.kind is None or c.kind.value not in numeric]
    values = [c.name for c in result.columns if c.kind is not None and c.kind.value in numeric]
    out: dict[str, dict[str, str | None]] = {}
    for index in range(result.row_count):
        key = "|".join(str(result.cell(index, k)) for k in keys) or "*"
        out[key] = {
            v: None if result.cell(index, v) is None else str(result.cell(index, v))
            for v in values
        }
    return out


def _difference(a: str | None, b: str | None) -> Decimal | None:
    """Relative difference of two exact values; None when they are equal-null."""
    if a is None and b is None:
        return Decimal(0)
    if a is None or b is None:
        return None  # one reading has a value the other lacks: material
    x, y = Decimal(a), Decimal(b)
    scale = max(abs(x), abs(y))
    return Decimal(0) if scale == 0 else abs(x - y) / scale


def run_materiality_probe(
    probe: MaterialityProbe,
    base: AnalysisPlan,
    *,
    engine,
    settings: Settings | None = None,
) -> MaterialityResult:
    """Evaluate every reading and say whether the choice between them matters.

    `engine` supplies the gate and the executor (anything with `gate(*specs)`,
    `store`, `scan`, `dataset_id` and `calendar`). Every reading runs under the
    base plan's stance and knowledge horizon, because the probe's edits cannot
    change either.
    """
    from ai_analyst.semantic.execute import AbstentionRequired, run_plan

    reason = outside_rules(probe)
    if reason:
        return _inconclusive(probe, reason)
    if len(base.specs) != 1:
        return _inconclusive(probe, "the probe evaluates single-spec plans only")

    readings: list[ProbeReading] = []
    for interpretation in probe.interpretations:
        try:
            plan = (
                apply_edit(
                    base,
                    PlanEdit(base_plan_id=base.plan_id, operations=interpretation.operations),
                ).plan
                if interpretation.operations
                else base
            )
        except EditConflict as exc:
            return _inconclusive(probe, f"reading {interpretation.label!r}: {exc}")
        outcome = engine.gate(*plan.specs)
        if not outcome.ok:
            codes = ", ".join(sorted({r.code.value for r in outcome.validation.rejections}))
            return _inconclusive(
                probe,
                f"reading {interpretation.label!r} cannot be evaluated ({codes})",
                readings=tuple(readings),
                executed=len(readings),
            )
        try:
            with engine.store.connect() as conn:
                (result,) = run_plan(
                    conn, engine.scan, plan, outcome, dataset_id=engine.dataset_id,
                    calendar=engine.calendar, settings=settings or engine.settings,
                )
        except AbstentionRequired:
            return _inconclusive(probe, f"reading {interpretation.label!r} is tier C")
        if result.row_count > MAX_RESULT_ROWS or result.truncated:
            return _inconclusive(
                probe, f"reading {interpretation.label!r} is too large to compare"
            )
        readings.append(
            ProbeReading(
                label=interpretation.label,
                row_count=result.row_count,
                trust_tier=result.trust_tier,
                cells=_cells(result),
            )
        )

    largest = Decimal(0)
    first = readings[0]
    for other in readings[1:]:
        if set(first.cells) != set(other.cells):
            return MaterialityResult(
                verdict=Materiality.MATERIAL,
                ambiguity=probe.ambiguity,
                reason="the readings produce different groups",
                readings=tuple(readings),
                executed=len(readings),
            )
        for key, cells in first.cells.items():
            for column, value in cells.items():
                difference = _difference(value, other.cells[key].get(column))
                if difference is None:
                    return MaterialityResult(
                        verdict=Materiality.MATERIAL,
                        ambiguity=probe.ambiguity,
                        reason=f"{column} has a value under one reading and none under another",
                        readings=tuple(readings),
                        executed=len(readings),
                    )
                largest = max(largest, difference)
    material = largest > probe.relative_threshold
    return MaterialityResult(
        verdict=Materiality.MATERIAL if material else Materiality.NOT_MATERIAL,
        ambiguity=probe.ambiguity,
        reason=(
            f"the readings differ by up to {largest:.6f} relative, "
            f"{'above' if material else 'within'} the threshold of {probe.relative_threshold}"
        ),
        readings=tuple(readings),
        largest_relative_difference=largest,
        executed=len(readings),
    )
