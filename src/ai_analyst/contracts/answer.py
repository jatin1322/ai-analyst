"""Answer contracts: the responder's draft and what the renderer makes of it.

ARCHITECTURE 13.14 and 13.15. The responder will write prose containing
reference tokens such as `{{q1.r0.opening_pipeline}}`, never a value. The
renderer, which is deterministic code, substitutes the values, draws tables from
results, verifies comparative claims, and appends the disclosures the trust
model requires. The provenance scanner then checks every numeral in the text
and fails any it cannot trace.

A rendered answer is kept as **segments**, each labelled with where its text
came from. That labelling is the provenance: a numeral inside a segment the
renderer produced from a result cell is traced by construction, and a numeral
in model prose must be traced by the scanner or the answer fails.
"""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from ai_analyst.contracts.result import TrustTier


class ComparisonKind(StrEnum):
    LESS_THAN = "less_than"
    GREATER_THAN = "greater_than"
    EQUAL = "equal"


class ComparisonClaim(BaseModel):
    """A comparative statement about two referenced values, checked before emitting.

    "Pipeline fell" is a claim about two numbers. The numeral scanner cannot
    see a correct number with the wrong direction; this can.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: ComparisonKind
    left: str
    right: str


class ResultTableRef(BaseModel):
    """A table the renderer draws from a result. The model never retypes rows."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    result: str
    columns: list[str] | None = None
    max_rows: int = Field(default=20, ge=1, le=50)


class AnswerDraft(BaseModel):
    """What the responder returns. Prose with tokens, plus typed references."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    headline: str = Field(min_length=1)
    explanation: str | None = None
    breakdown: ResultTableRef | None = None
    # Indexes into the assumptions the renderer was given; the text itself is
    # inserted by the renderer, so a model cannot reword an assumption.
    assumptions: list[int] = Field(default_factory=list)
    comparisons: list[ComparisonClaim] = Field(default_factory=list)
    follow_up_suggestions: list[str] = Field(default_factory=list)


class SegmentSource(StrEnum):
    """Who produced a piece of the rendered text."""

    MODEL = "model"      # the responder's prose, around the tokens
    RESULT = "result"    # a value substituted from a registered result cell
    SYSTEM = "system"    # deterministic metadata: disclosures, table layout


class Segment(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    source: SegmentSource
    # For a RESULT segment, the token it came from.
    reference: str | None = None


class RenderedAnswer(BaseModel):
    segments: list[Segment]
    trust_tier: TrustTier
    references: list[str] = Field(default_factory=list)

    @property
    def text(self) -> str:
        return "".join(s.text for s in self.segments)


class NumeralSource(StrEnum):
    RESULT = "result"            # substituted from a result cell
    SYSTEM = "system"            # deterministic metadata the renderer wrote
    USER = "user"                # a literal the user supplied in the question
    STRUCTURAL = "structural"    # a year, a quarter label, an ordinal, a limit
    UNVERIFIED = "unverified"    # none of the above: the answer fails


class NumeralFinding(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    source: NumeralSource
    value: Decimal | None = None
    reason: str = ""


class ProvenanceReport(BaseModel):
    """Every numeral in a rendered answer, and where it came from."""

    findings: list[NumeralFinding] = Field(default_factory=list)

    @property
    def unverified(self) -> list[NumeralFinding]:
        return [f for f in self.findings if f.source is NumeralSource.UNVERIFIED]

    @property
    def ok(self) -> bool:
        return not self.unverified

    @property
    def coverage(self) -> float:
        """Share of numerals traced to something. The headline metric of 8.4."""
        if not self.findings:
            return 1.0
        return 1 - len(self.unverified) / len(self.findings)
