"""The provenance scanner (ARCHITECTURE 8.4, 13.15).

Every numeral in a rendered answer must trace to something. A numeral in a
segment the renderer produced from a result cell traces by construction, as
does one in the deterministic metadata the renderer appended. A numeral in the
model's own prose must be one of:

* a literal the user supplied in the question ("$100k" matches 100000);
* structural: a year, a quarter label, a small ordinal, or the N of a
  "top N" that equals the plan's limit;

and anything else is **unverified**, which fails the answer. There is no
tolerance and no "close enough": a model writing 515,001 beside a result of
515,000 has invented a number.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from ai_analyst.contracts.answer import (
    NumeralFinding,
    NumeralSource,
    ProvenanceReport,
    RenderedAnswer,
    SegmentSource,
)

# A number as it may appear in prose: sign, currency, grouping, decimals, and a
# percent sign or a magnitude suffix.
NUMERAL = re.compile(
    r"(?<![\w.])"
    r"(?P<sign>[-−])?"
    r"(?P<currency>[$€£])?"
    r"(?P<digits>\d{1,3}(?:,\d{3})+|\d+)"
    r"(?P<fraction>\.\d+)?"
    r"(?P<unit>%|[kKmMbB](?![A-Za-z]))?"
)
# Masked before numerals are extracted, because they are labels, not quantities.
QUARTER_LABEL = re.compile(r"\bFY\d{4}(?:-Q[1-4])?\b|\bQ[1-4]\b")
ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
ORDINAL = re.compile(r"\b(\d{1,2})(st|nd|rd|th)\b", re.IGNORECASE)
TOP_N = re.compile(r"\b(?:top|bottom|first|last)\s+(\d+)\b", re.IGNORECASE)

_MULTIPLIERS = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}


def parse_numeral(match: re.Match) -> Decimal | None:
    """The numeric value a prose numeral denotes, with its sign and magnitude."""
    try:
        value = Decimal(match.group("digits").replace(",", "") + (match.group("fraction") or ""))
    except InvalidOperation:
        return None
    unit = (match.group("unit") or "").lower()
    if unit in _MULTIPLIERS:
        value *= _MULTIPLIERS[unit]
    if match.group("sign"):
        value = -value
    return value


def user_values(question: str) -> set[Decimal]:
    """Every number the user wrote, normalized: "$100k" and "100,000" are 100000."""
    values: set[Decimal] = set()
    for match in NUMERAL.finditer(question):
        value = parse_numeral(match)
        if value is not None:
            values.add(value)
            values.add(abs(value))
    return values


def _mask(text: str, pattern: re.Pattern) -> str:
    return pattern.sub(lambda m: " " * len(m.group(0)), text)


def scan(
    answer: RenderedAnswer,
    *,
    question: str,
    limit: int | None = None,
    known_dates: frozenset[str] = frozenset(),
) -> ProvenanceReport:
    """Classify every numeral in a rendered answer by where it came from."""
    findings: list[NumeralFinding] = []
    supplied = user_values(question)

    for segment in answer.segments:
        if segment.source is not SegmentSource.MODEL:
            source = (
                NumeralSource.RESULT
                if segment.source is SegmentSource.RESULT
                else NumeralSource.SYSTEM
            )
            findings.extend(
                NumeralFinding(text=m.group(0), source=source, value=parse_numeral(m))
                for m in NUMERAL.finditer(segment.text)
            )
            continue

        text = segment.text
        for match in ISO_DATE.finditer(text):
            known = match.group(0) in known_dates or match.group(0) in question
            findings.append(
                NumeralFinding(
                    text=match.group(0),
                    source=NumeralSource.STRUCTURAL if known else NumeralSource.UNVERIFIED,
                    reason="a resolved snapshot date" if known else "a date no result supplied",
                )
            )
        text = _mask(text, ISO_DATE)
        for match in QUARTER_LABEL.finditer(text):
            findings.append(
                NumeralFinding(text=match.group(0), source=NumeralSource.STRUCTURAL,
                               reason="a period label")
            )
        text = _mask(text, QUARTER_LABEL)
        for match in ORDINAL.finditer(text):
            findings.append(
                NumeralFinding(text=match.group(0), source=NumeralSource.STRUCTURAL,
                               reason="an ordinal")
            )
        text = _mask(text, ORDINAL)
        for match in TOP_N.finditer(text):
            n = int(match.group(1))
            allowed = limit is not None and n == limit
            findings.append(
                NumeralFinding(
                    text=match.group(0),
                    source=NumeralSource.STRUCTURAL if allowed else NumeralSource.UNVERIFIED,
                    value=Decimal(n),
                    reason=(
                        "the plan's limit" if allowed else "a count that is not the plan's limit"
                    ),
                )
            )
        text = _mask(text, TOP_N)

        for match in NUMERAL.finditer(text):
            value = parse_numeral(match)
            raw = match.group(0)
            if value is not None and value in supplied:
                findings.append(
                    NumeralFinding(text=raw, source=NumeralSource.USER, value=value,
                                   reason="supplied by the user")
                )
            elif (
                value is not None
                and not match.group("currency")
                and not match.group("unit")
                and not match.group("fraction")
                and 1900 <= value <= 2100
                and len(match.group("digits")) == 4
            ):
                findings.append(
                    NumeralFinding(text=raw, source=NumeralSource.STRUCTURAL, value=value,
                                   reason="a year")
                )
            else:
                findings.append(
                    NumeralFinding(text=raw, source=NumeralSource.UNVERIFIED, value=value,
                                   reason="not substituted from a result and not supplied")
                )
    return ProvenanceReport(findings=findings)
