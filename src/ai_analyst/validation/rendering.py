"""Deterministic rendering of reference tokens (ARCHITECTURE 8.4, 13.14).

The responder writes `{{q1.r0.opening_pipeline}}`; this module decides what
that becomes. Values are formatted by the kind the compiler assigned to the
column, so money renders as currency and a ratio as a percentage without the
model ever writing, rounding, or formatting a number.

Failures are loud. An unknown result, an unknown column, a row out of range, or
a malformed token is an error, never a blank: a token that resolves to nothing
is a model inventing a reference, and emitting the sentence without it would
change what the sentence claims.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from enum import StrEnum

from ai_analyst.contracts.answer import (
    AnswerDraft,
    ComparisonClaim,
    ComparisonKind,
    RenderedAnswer,
    ResultTableRef,
    Segment,
    SegmentSource,
)
from ai_analyst.contracts.result import ResultSet, TrustAssessment, ValueKind

TOKEN = re.compile(r"\{\{\s*([^{}]*?)\s*\}\}")
CELL = re.compile(r"^(q\d+)\.r(\d+)\.([A-Za-z_][A-Za-z0-9_]*)$")
META = re.compile(r"^(q\d+)\.meta\.([a-z_]+)$")

# Metadata a token may address, beside the cells.
META_FIELDS = ("resolved_as_of", "requested_boundary", "drift_days", "row_count")

# Comparative words that assert a direction between two numbers. A draft using
# one must carry a typed claim, which the renderer checks.
COMPARATIVES = re.compile(
    r"\b(fell|fall|falls|rose|rise|rises|higher|lower|more|fewer|less|greater|"
    r"exceed(ed|s)?|increase(d|s)?|decrease(d|s)?|declin(e|ed|es)|grew|grow|grows|"
    r"drop(ped|s)?|up from|down from)\b",
    re.IGNORECASE,
)


class RenderErrorCode(StrEnum):
    MALFORMED_TOKEN = "malformed_token"
    UNKNOWN_RESULT = "unknown_result"
    UNKNOWN_COLUMN = "unknown_column"
    ROW_OUT_OF_RANGE = "row_out_of_range"
    UNKNOWN_METADATA = "unknown_metadata"
    COMPARISON_FALSE = "comparison_false"
    UNSUPPORTED_COMPARATIVE = "unsupported_comparative"
    BAD_ASSUMPTION = "bad_assumption"


class RenderError(ValueError):
    def __init__(self, code: RenderErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass
class ResultRegistry:
    """Results addressable by per-turn alias: q1, q2, ... in registration order."""

    results: dict[str, ResultSet] = field(default_factory=dict)

    def register(self, result: ResultSet) -> str:
        alias = f"q{len(self.results) + 1}"
        self.results[alias] = result
        return alias

    def get(self, alias: str) -> ResultSet:
        if alias not in self.results:
            raise RenderError(
                RenderErrorCode.UNKNOWN_RESULT,
                f"no result registered as {alias!r}; registered: {sorted(self.results)}",
            )
        return self.results[alias]


# ---------------------------------------------------------------- formatting


def _group(integer: int) -> str:
    return f"{integer:,}"


def format_value(value: object, kind: ValueKind | None, currency: str = "$") -> str:
    """Render one value by its kind. The only place a number becomes text."""
    if value is None:
        return "no value"
    if isinstance(value, bool) or kind is ValueKind.BOOLEAN:
        return "yes" if value else "no"
    if isinstance(value, date) or kind is ValueKind.DATE:
        return value.isoformat() if isinstance(value, date) else str(value)
    if kind is ValueKind.TEXT:
        return str(value)
    number = value if isinstance(value, Decimal) else Decimal(str(value))
    sign = "-" if number < 0 else ""
    magnitude = abs(number)
    match kind:
        case ValueKind.MONEY:
            return f"{sign}{currency}{magnitude.quantize(Decimal('0.01')):,}"
        case ValueKind.RATIO:
            percent = (magnitude * 100).quantize(Decimal("0.1"))
            return f"{sign}{percent}%"
        case ValueKind.COUNT:
            return f"{sign}{_group(int(magnitude))}"
        case _:
            text = format(magnitude.normalize(), "f")
            whole, _, frac = text.partition(".")
            return f"{sign}{_group(int(whole))}" + (f".{frac}" if frac else "")


# ---------------------------------------------------------------- resolution


@dataclass(frozen=True)
class Resolved:
    reference: str
    raw: object
    text: str


def resolve_token(reference: str, registry: ResultRegistry, currency: str = "$") -> Resolved:
    """One token, to its raw value and its rendered text."""
    cell = CELL.match(reference)
    if cell:
        alias, row, column = cell.group(1), int(cell.group(2)), cell.group(3)
        result = registry.get(alias)
        if column not in result.column_names:
            raise RenderError(
                RenderErrorCode.UNKNOWN_COLUMN,
                f"{reference!r}: result {alias} has no column {column!r}",
            )
        if not 0 <= row < result.row_count:
            raise RenderError(
                RenderErrorCode.ROW_OUT_OF_RANGE,
                f"{reference!r}: result {alias} has {result.row_count} row(s)",
            )
        raw = result.cell(row, column)
        kind = next(c.kind for c in result.columns if c.name == column)
        return Resolved(reference, raw, format_value(raw, kind, currency))
    meta = META.match(reference)
    if meta:
        alias, name = meta.group(1), meta.group(2)
        result = registry.get(alias)
        snapshots = result.resolved_snapshots
        values = {
            "resolved_as_of": snapshots[0].resolved_as_of if snapshots else None,
            "requested_boundary": snapshots[0].requested_boundary if snapshots else None,
            "drift_days": snapshots[0].drift_days if snapshots else None,
            "row_count": result.row_count,
        }
        if name not in values:
            raise RenderError(
                RenderErrorCode.UNKNOWN_METADATA,
                f"{reference!r}: metadata is one of {', '.join(META_FIELDS)}",
            )
        raw = values[name]
        kind = ValueKind.DATE if isinstance(raw, date) else ValueKind.COUNT
        return Resolved(reference, raw, format_value(raw, kind, currency))
    raise RenderError(
        RenderErrorCode.MALFORMED_TOKEN,
        f"{{{{{reference}}}}} is not a reference: expected qN.rK.column or qN.meta.field",
    )


def _substitute(text: str, registry: ResultRegistry, currency: str) -> list[Segment]:
    segments: list[Segment] = []
    position = 0
    for match in TOKEN.finditer(text):
        if match.start() > position:
            segments.append(
                Segment(text=text[position : match.start()], source=SegmentSource.MODEL)
            )
        resolved = resolve_token(match.group(1), registry, currency)
        segments.append(
            Segment(text=resolved.text, source=SegmentSource.RESULT, reference=resolved.reference)
        )
        position = match.end()
    if position < len(text):
        segments.append(Segment(text=text[position:], source=SegmentSource.MODEL))
    return segments


def _comparable(value: object) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool) or value is None:
        raise RenderError(
            RenderErrorCode.COMPARISON_FALSE, "a comparison needs two numeric values"
        )
    return Decimal(str(value))


def check_claim(claim: ComparisonClaim, registry: ResultRegistry) -> None:
    """Verify one comparative claim against the raw values it names."""
    left = _comparable(resolve_token(claim.left.strip("{} "), registry).raw)
    right = _comparable(resolve_token(claim.right.strip("{} "), registry).raw)
    holds = {
        ComparisonKind.LESS_THAN: left < right,
        ComparisonKind.GREATER_THAN: left > right,
        ComparisonKind.EQUAL: left == right,
    }[claim.kind]
    if not holds:
        raise RenderError(
            RenderErrorCode.COMPARISON_FALSE,
            f"the draft claims {claim.left} {claim.kind.value} {claim.right}, "
            f"but the values are {left} and {right}",
        )


def render_table(ref: ResultTableRef, registry: ResultRegistry, currency: str = "$") -> str:
    """A markdown table drawn from a result, formatted by column kind."""
    result = registry.get(ref.result)
    columns = ref.columns or result.column_names
    for column in columns:
        if column not in result.column_names:
            raise RenderError(
                RenderErrorCode.UNKNOWN_COLUMN,
                f"table on {ref.result} asks for column {column!r}, which it does not have",
            )
    kinds = {c.name: c.kind for c in result.columns}
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for index in range(min(result.row_count, ref.max_rows)):
        cells = [format_value(result.cell(index, c), kinds[c], currency) for c in columns]
        lines.append("| " + " | ".join(cells) + " |")
    if result.row_count > ref.max_rows:
        lines.append(f"({result.row_count - ref.max_rows} more rows not shown)")
    return "\n".join(lines)


def mandatory_disclosures(
    trust: TrustAssessment, *, retrospective: bool, association: bool
) -> list[str]:
    """What an answer must say whatever the draft says. Appended, not requested."""
    out: list[str] = []
    if retrospective:
        out.append("This answer uses hindsight: it reads outcomes observed after the snapshot.")
    if association:
        out.append(
            "This is an association in this dataset, not evidence that one thing "
            "causes another."
        )
    out.extend(trust.disclosures)
    return list(dict.fromkeys(out))


def render(
    draft: AnswerDraft,
    registry: ResultRegistry,
    trust: TrustAssessment,
    *,
    assumptions: list[str] = (),
    warnings: list[str] = (),
    retrospective: bool = False,
    association: bool = False,
    currency: str = "$",
) -> RenderedAnswer:
    """Turn a draft into an answer. Every value in it comes from a result."""
    prose = draft.headline + (f"\n\n{draft.explanation}" if draft.explanation else "")
    if COMPARATIVES.search(prose) and not draft.comparisons:
        raise RenderError(
            RenderErrorCode.UNSUPPORTED_COMPARATIVE,
            "the draft states a direction between values without a comparison claim",
        )
    for claim in draft.comparisons:
        check_claim(claim, registry)

    segments = _substitute(prose, registry, currency)
    if draft.breakdown is not None:
        segments.append(
            Segment(
                text="\n\n" + render_table(draft.breakdown, registry, currency),
                source=SegmentSource.RESULT,
                reference=f"table:{draft.breakdown.result}",
            )
        )

    chosen = []
    for index in draft.assumptions:
        if not 0 <= index < len(assumptions):
            raise RenderError(
                RenderErrorCode.BAD_ASSUMPTION,
                f"assumption {index} does not exist; {len(assumptions)} were supplied",
            )
        chosen.append(assumptions[index])
    notes = [*chosen, *warnings, *mandatory_disclosures(
        trust, retrospective=retrospective, association=association
    )]
    if notes:
        segments.append(
            Segment(
                text="\n\n" + "\n".join(f"- {n}" for n in dict.fromkeys(notes)),
                source=SegmentSource.SYSTEM,
            )
        )
    return RenderedAnswer(
        segments=segments,
        trust_tier=trust.tier,
        references=[s.reference for s in segments if s.reference],
    )
