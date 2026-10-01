"""The renderer inserts every number; the scanner verifies every numeral.

ARCHITECTURE 8.4, 13.14, 13.15. The responder is not built, so drafts here are
hand-written, exactly as a responder would emit them: prose with reference
tokens. The results they reference are real, produced by the engine.

Values referenced below, hand-computed from the tiny fixture:
    Q2 opening pipeline by owner (2025-04-01): U-101 350000, U-103 165000
    Q1 win rate at 2025-03-31: won 1, won + lost 2, ratio 0.5
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from ai_analyst.contracts.answer import (
    AnswerDraft,
    ComparisonClaim,
    ComparisonKind,
    NumeralSource,
    ResultTableRef,
    SegmentSource,
)
from ai_analyst.contracts.plan import (
    AnalysisPattern,
    AnalysisSpec,
    Period,
    PeriodKind,
    SnapshotSelection,
)
from ai_analyst.contracts.result import SnapshotRule, TrustTier, ValueKind
from ai_analyst.validation.provenance import scan, user_values
from ai_analyst.validation.rendering import (
    RenderError,
    RenderErrorCode,
    ResultRegistry,
    format_value,
    render,
)

Q1 = Period(kind=PeriodKind.FISCAL_QUARTER, label="FY2025-Q1")
Q2 = Period(kind=PeriodKind.FISCAL_QUARTER, label="FY2025-Q2")


@pytest.fixture
def registry(tiny) -> ResultRegistry:
    by_owner = tiny.one(
        AnalysisSpec(
            id="byowner", pattern=AnalysisPattern.POINT_IN_TIME, metrics=["opening_pipeline"],
            dimensions=["owner"], period=Q2,
            snapshot=SnapshotSelection(rule=SnapshotRule.PERIOD_OPEN),
        )
    )
    win_rate = tiny.one(
        AnalysisSpec(
            id="wr", pattern=AnalysisPattern.RATE, metrics=["win_rate"], period=Q1,
            snapshot=SnapshotSelection(rule=SnapshotRule.PERIOD_CLOSE),
        )
    )
    registry = ResultRegistry()
    assert registry.register(by_owner) == "q1"
    assert registry.register(win_rate) == "q2"
    return registry


def draft(headline: str, **kwargs) -> AnswerDraft:
    return AnswerDraft(headline=headline, **kwargs)


def rendered(registry, text: str, **kwargs):
    return render(draft(text, **kwargs), registry, registry.get("q1").trust)


# ============================================================================
# The renderer inserts values
# ============================================================================


def test_a_token_renders_to_the_cell_formatted_by_its_kind(registry):
    answer = rendered(
        registry,
        "Q2 opening pipeline for {{q1.r0.owner_id}} was {{q1.r0.opening_pipeline}}.",
    )
    assert answer.text.startswith("Q2 opening pipeline for U-101 was $350,000.00.")
    assert [s.source for s in answer.segments][:4] == [
        SegmentSource.MODEL,
        SegmentSource.RESULT,
        SegmentSource.MODEL,
        SegmentSource.RESULT,
    ]


def test_a_ratio_renders_as_a_percentage(registry):
    answer = rendered(registry, "Q1 win rate was {{q2.r0.ratio}}.")
    assert "Q1 win rate was 50.0%." in answer.text


def test_metadata_is_addressable(registry):
    answer = rendered(registry, "Measured at {{q1.meta.resolved_as_of}}.")
    assert "Measured at 2025-04-01." in answer.text


def test_a_table_is_drawn_from_the_result_not_retyped(registry):
    answer = rendered(
        registry, "Q2 opening pipeline by owner:", breakdown=ResultTableRef(result="q1")
    )
    assert "| U-101 | $350,000.00 |" in answer.text
    assert "| U-103 | $165,000.00 |" in answer.text


@pytest.mark.parametrize(
    ("token", "code"),
    [
        ("{{q9.r0.opening_pipeline}}", RenderErrorCode.UNKNOWN_RESULT),
        ("{{q1.r9.opening_pipeline}}", RenderErrorCode.ROW_OUT_OF_RANGE),
        ("{{q1.r0.no_such_column}}", RenderErrorCode.UNKNOWN_COLUMN),
        ("{{opening_pipeline}}", RenderErrorCode.MALFORMED_TOKEN),
        ("{{q1.meta.secret}}", RenderErrorCode.UNKNOWN_METADATA),
    ],
)
def test_an_invalid_token_fails_rather_than_rendering_blank(registry, token, code):
    with pytest.raises(RenderError) as exc:
        rendered(registry, f"The answer is {token}.")
    assert exc.value.code is code


def test_a_comparative_without_a_claim_is_refused(registry):
    with pytest.raises(RenderError) as exc:
        rendered(registry, "{{q1.r1.owner_id}} had lower pipeline than {{q1.r0.owner_id}}.")
    assert exc.value.code is RenderErrorCode.UNSUPPORTED_COMPARATIVE


def test_a_true_comparison_claim_renders(registry):
    claim = ComparisonClaim(
        kind=ComparisonKind.LESS_THAN,
        left="q1.r1.opening_pipeline",
        right="q1.r0.opening_pipeline",
    )
    answer = rendered(
        registry,
        "{{q1.r1.owner_id}} had lower pipeline than {{q1.r0.owner_id}}.",
        comparisons=[claim],
    )
    assert "U-103 had lower pipeline than U-101." in answer.text


def test_a_false_comparison_claim_is_refused(registry):
    """A correct number with the wrong direction: the gap the numeral scan cannot see."""
    claim = ComparisonClaim(
        kind=ComparisonKind.GREATER_THAN,
        left="q1.r1.opening_pipeline",
        right="q1.r0.opening_pipeline",
    )
    with pytest.raises(RenderError) as exc:
        rendered(registry, "{{q1.r1.owner_id}} had more pipeline.", comparisons=[claim])
    assert exc.value.code is RenderErrorCode.COMPARISON_FALSE


def test_trust_disclosures_are_appended_whatever_the_draft_says(registry):
    answer = rendered(registry, "Opening pipeline was {{q1.r0.opening_pipeline}}.")
    assert answer.trust_tier is TrustTier.B
    assert "authoritative_status" in answer.text
    assert answer.segments[-1].source is SegmentSource.SYSTEM


def test_retrospective_and_association_disclosures_are_mandatory(registry):
    answer = render(
        draft("Result: {{q1.r0.opening_pipeline}}."), registry, registry.get("q1").trust,
        retrospective=True, association=True,
    )
    assert "uses hindsight" in answer.text
    assert "not evidence that one thing causes another" in answer.text


def test_an_assumption_is_selected_by_index_and_inserted_verbatim(registry):
    assumptions = registry.get("q1").assumptions
    answer = render(
        draft("Result: {{q1.r0.opening_pipeline}}.", assumptions=[0]), registry,
        registry.get("q1").trust, assumptions=assumptions,
    )
    assert assumptions[0] in answer.text
    with pytest.raises(RenderError) as exc:
        render(draft("x", assumptions=[99]), registry, registry.get("q1").trust,
               assumptions=assumptions)
    assert exc.value.code is RenderErrorCode.BAD_ASSUMPTION


@pytest.mark.parametrize(
    ("value", "kind", "expected"),
    [
        (Decimal("350000.00"), ValueKind.MONEY, "$350,000.00"),
        (Decimal("-250000.5"), ValueKind.MONEY, "-$250,000.50"),
        (Decimal("0.5"), ValueKind.RATIO, "50.0%"),
        (Decimal("-0.25"), ValueKind.RATIO, "-25.0%"),
        (1234567, ValueKind.COUNT, "1,234,567"),
        (Decimal("0.392792"), ValueKind.QUANTITY, "0.392792"),
        (date(2025, 4, 1), ValueKind.DATE, "2025-04-01"),
        (None, ValueKind.MONEY, "no value"),
        (True, ValueKind.BOOLEAN, "yes"),
    ],
)
def test_format_value(value, kind, expected):
    assert format_value(value, kind) == expected


# ============================================================================
# The scanner verifies every numeral
# ============================================================================


def check(registry, text: str, *, question: str = "", limit=None, known_dates=frozenset()):
    return scan(rendered(registry, text), question=question, limit=limit, known_dates=known_dates)


def test_valid_references_verify(registry):
    report = check(registry, "{{q1.r0.owner_id}} had {{q1.r0.opening_pipeline}}.")
    assert report.ok
    assert {f.source for f in report.findings} <= {NumeralSource.RESULT, NumeralSource.SYSTEM}
    assert report.coverage == 1.0


def test_an_invented_number_fails(registry):
    report = check(registry, "Opening pipeline was about 515,000 in total.")
    assert not report.ok
    assert [f.text for f in report.unverified] == ["515,000"]


def test_an_invented_number_beside_a_real_one_fails(registry):
    """No tolerance: 350,001 next to a result of 350,000 is an invention."""
    report = check(registry, "{{q1.r0.opening_pipeline}}, roughly 350,001.")
    assert [f.text for f in report.unverified] == ["350,001"]


def test_a_user_supplied_threshold_verifies(registry):
    report = check(
        registry, "Showing deals above $100k: {{q1.r0.opening_pipeline}}.",
        question="Only show deals above $100k.",
    )
    assert report.ok
    assert any(f.source is NumeralSource.USER and f.value == 100000 for f in report.findings)


def test_the_same_threshold_written_differently_still_verifies(registry):
    report = check(
        registry, "Filtered to amounts of at least 100,000.",
        question="Only show deals above $100k.",
    )
    assert report.ok


def test_a_year_is_structural(registry):
    report = check(registry, "In 2025 the pipeline was {{q1.r0.opening_pipeline}}.")
    assert report.ok
    assert any(f.source is NumeralSource.STRUCTURAL and f.text == "2025" for f in report.findings)


def test_a_quarter_label_is_structural(registry):
    report = check(registry, "For FY2025-Q2 and Q3 we have {{q1.r0.opening_pipeline}}.")
    assert report.ok


def test_a_top_n_matching_the_limit_is_structural(registry):
    assert check(registry, "The top 5 deals follow.", limit=5).ok


def test_a_top_n_not_matching_the_limit_fails(registry):
    report = check(registry, "The top 7 deals follow.", limit=5)
    assert not report.ok


def test_an_invented_percentage_fails_and_a_referenced_one_passes(registry):
    assert not check(registry, "Roughly 12% of deals slipped.").ok
    assert check(registry, "The win rate was {{q2.r0.ratio}}.").ok


def test_an_invented_negative_number_fails(registry):
    report = check(registry, "The change was -250,000.")
    assert [f.value for f in report.unverified] == [Decimal("-250000")]


def test_an_invented_currency_amount_fails(registry):
    report = check(registry, "Pipeline was $515,000.00.")
    assert [f.text for f in report.unverified] == ["$515,000.00"]


def test_an_invented_decimal_fails(registry):
    assert not check(registry, "The correlation was 0.39.").ok


def test_a_date_must_be_a_resolved_snapshot_date(registry):
    assert check(registry, "As of 2025-04-01.", known_dates=frozenset({"2025-04-01"})).ok
    assert not check(registry, "As of 2025-05-17.", known_dates=frozenset({"2025-04-01"})).ok


def test_numbers_the_renderer_wrote_are_traced_by_construction(registry):
    """A negative, a decimal and a currency value, all from results, all verified."""
    report = check(
        registry,
        "{{q1.r0.opening_pipeline}} and {{q2.r0.ratio}} of {{q2.r0.denominator}}.",
    )
    assert report.ok


def test_user_values_are_normalized():
    assert user_values("deals above $100k or 1.5m, and 20%") >= {
        Decimal("100000"), Decimal("1500000"), Decimal("20"),
    }
