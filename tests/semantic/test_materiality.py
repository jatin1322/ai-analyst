"""The clarification materiality probe (ARCHITECTURE 13.12).

Given two to four readings of an ambiguous question, each a typed edit of the
base plan, the probe runs every reading through the full gate and executor and
says whether the choice matters: MATERIAL, NOT_MATERIAL or INCONCLUSIVE. It
never chooses a reading, never substitutes a concept, and never evaluates a
binding ambiguity.

Hand-computed from `tests/fixtures/tiny/snapshots.csv`:

    deal_count at 2025-01-01 = 6 (OPP-001..005, 007); at 2025-03-31 = 7 (adds 008)
    Q1 opening pipeline at 2025-01-01 by owner, owners unchanged through 03-31:
        U-101 OPP-001 100000; U-104 OPP-005 40000; U-102 OPP-007 60000
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from ai_analyst.contracts.plan import (
    AnalysisPattern,
    AnalysisPlan,
    AnalysisSpec,
    AnalysisStance,
    Attribution,
    Period,
    PeriodKind,
    SnapshotSelection,
)
from ai_analyst.contracts.result import SnapshotRule
from ai_analyst.contracts.session import (
    AddDimension,
    ChangeMetrics,
    ChangePeriod,
    ChangeSnapshotRule,
    ChangeStance,
    SetAnalysisOption,
)
from ai_analyst.semantic import materiality as materiality_module
from ai_analyst.semantic.materiality import (
    ALLOWED_OPERATIONS,
    AmbiguityKind,
    Interpretation,
    Materiality,
    MaterialityProbe,
    MaterialityResult,
    run_materiality_probe,
)
from tests.semantic.conftest import tool_context

Q1 = Period(kind=PeriodKind.FISCAL_QUARTER, label="FY2025-Q1")
Q2 = Period(kind=PeriodKind.FISCAL_QUARTER, label="FY2025-Q2")
PROSP, RETRO = AnalysisStance.PROSPECTIVE, AnalysisStance.RETROSPECTIVE


def base(**kwargs) -> AnalysisPlan:
    kwargs.setdefault("pattern", AnalysisPattern.POINT_IN_TIME)
    kwargs.setdefault("period", Q1)
    kwargs.setdefault("metrics", ["deal_count"])
    kwargs.setdefault("snapshot", SnapshotSelection(rule=SnapshotRule.PERIOD_OPEN))
    return AnalysisPlan(question_restatement="q", specs=[AnalysisSpec(id="a", **kwargs)])


def reading(label: str, *operations) -> Interpretation:
    return Interpretation(label=label, operations=list(operations))


def snapshot_rule(rule: SnapshotRule) -> ChangeSnapshotRule:
    return ChangeSnapshotRule(snapshot=SnapshotSelection(rule=rule))


def attribution(rule: Attribution) -> SetAnalysisOption:
    return SetAnalysisOption(option="attribution", value=rule.value)


OPEN_OR_CLOSE = MaterialityProbe(
    ambiguity=AmbiguityKind.SNAPSHOT,
    interpretations=[
        reading("at quarter open"),
        reading("at quarter close", snapshot_rule(SnapshotRule.PERIOD_CLOSE)),
    ],
)
OWNER_WHEN = MaterialityProbe(
    ambiguity=AmbiguityKind.ANALYSIS_OPTION,
    interpretations=[
        reading("owner at quarter open", attribution(Attribution.PERIOD_OPEN)),
        reading("owner at quarter close", attribution(Attribution.AT_CLOSE)),
    ],
)


def probe(engine, p: MaterialityProbe, plan: AnalysisPlan) -> MaterialityResult:
    return run_materiality_probe(p, plan, engine=engine)


# ============================================================================
# 1. Material
# ============================================================================


def test_readings_that_differ_are_material(tiny):
    result = probe(tiny, OPEN_OR_CLOSE, base())
    assert result.verdict is Materiality.MATERIAL
    assert [r.cells["*"]["deal_count"] for r in result.readings] == ["6", "7"]
    # |6 - 7| / 7 = 0.142857..., above the default 0.01.
    assert result.largest_relative_difference == Decimal(1) / Decimal(7)
    assert result.executed == 2


def test_readings_with_different_groups_are_material(tiny):
    # Q1 opening pipeline by stage at open vs at close: OPP-007 moves from
    # Negotiation to Closed Lost, so the set of groups differs.
    stage_when = MaterialityProbe(
        ambiguity=AmbiguityKind.ANALYSIS_OPTION,
        interpretations=[
            reading("stage at open", attribution(Attribution.PERIOD_OPEN)),
            reading("stage at close", attribution(Attribution.AT_CLOSE)),
        ],
    )
    plan = base(metrics=["opening_pipeline"], dimensions=["stage"], stance=RETRO)
    result = probe(tiny, stage_when, plan)
    assert result.verdict is Materiality.MATERIAL
    assert result.reason == "the readings produce different groups"


# ============================================================================
# 2. Not material
# ============================================================================


def test_readings_that_agree_are_not_material(tiny):
    plan = base(metrics=["opening_pipeline"], dimensions=["owner"], stance=RETRO)
    result = probe(tiny, OWNER_WHEN, plan)
    assert result.verdict is Materiality.NOT_MATERIAL
    assert result.largest_relative_difference == Decimal(0)
    first, second = result.readings
    assert first.cells == second.cells == {
        "U-101": {"opening_pipeline": "100000.00"},
        "U-102": {"opening_pipeline": "60000.00"},
        "U-104": {"opening_pipeline": "40000.00"},
    }


def test_a_difference_within_the_threshold_is_not_material(tiny):
    wide = OPEN_OR_CLOSE.model_copy(update={"relative_threshold": Decimal("0.2")})
    result = probe(tiny, wide, base())
    assert result.verdict is Materiality.NOT_MATERIAL
    assert "within the threshold" in result.reason


# ============================================================================
# 3. Inconclusive
# ============================================================================


def test_a_binding_ambiguity_is_inconclusive_by_rule(tiny):
    binding = MaterialityProbe(
        ambiguity=AmbiguityKind.CONCEPT_BINDING,
        interpretations=[reading("a"), reading("b")],
    )
    result = probe(tiny, binding, base())
    assert result.verdict is Materiality.INCONCLUSIVE
    assert "declaration resolves it" in result.reason
    assert result.executed == 0


def test_a_reading_the_gate_refuses_is_inconclusive(tiny):
    bad = MaterialityProbe(
        ambiguity=AmbiguityKind.PERIOD,
        interpretations=[
            reading("Q1"),
            reading("FY2019-Q1", ChangePeriod(period=Period(kind=PeriodKind.FISCAL_QUARTER,
                                                           label="FY2019-Q1"))),
        ],
    )
    result = probe(tiny, bad, base())
    assert result.verdict is Materiality.INCONCLUSIVE
    assert "cannot be evaluated" in result.reason


def test_an_edit_that_cannot_apply_is_inconclusive(tiny):
    bad = MaterialityProbe(
        ambiguity=AmbiguityKind.ANALYSIS_OPTION,
        interpretations=[reading("a"), reading("b", SetAnalysisOption(option="attribution",
                                                                      value="whenever"))],
    )
    assert probe(tiny, bad, base()).verdict is Materiality.INCONCLUSIVE


# ============================================================================
# 4. Bounded execution
# ============================================================================


def test_at_most_four_readings():
    with pytest.raises(ValidationError):
        MaterialityProbe(
            ambiguity=AmbiguityKind.SNAPSHOT,
            interpretations=[reading(str(i)) for i in range(5)],
        )


def test_at_least_two_readings():
    with pytest.raises(ValidationError):
        MaterialityProbe(ambiguity=AmbiguityKind.SNAPSHOT, interpretations=[reading("one")])


def test_execution_stops_at_the_first_reading_that_cannot_run(tiny):
    stops = MaterialityProbe(
        ambiguity=AmbiguityKind.PERIOD,
        interpretations=[
            reading("Q1"),
            reading("FY2019-Q1", ChangePeriod(period=Period(kind=PeriodKind.FISCAL_QUARTER,
                                                           label="FY2019-Q1"))),
            reading("Q2", ChangePeriod(period=Q2)),
        ],
    )
    result = probe(tiny, stops, base())
    assert result.verdict is Materiality.INCONCLUSIVE
    assert result.executed == 1
    assert [r.label for r in result.readings] == ["Q1"]


def test_a_result_larger_than_the_row_bound_is_not_compared(tiny, monkeypatch):
    monkeypatch.setattr(materiality_module, "MAX_RESULT_ROWS", 2)
    plan = base(metrics=["opening_pipeline"], dimensions=["owner"], stance=RETRO)
    result = probe(tiny, OWNER_WHEN, plan)
    assert result.verdict is Materiality.INCONCLUSIVE
    assert "too large" in result.reason


def test_a_multi_spec_plan_is_not_probed(tiny):
    two = AnalysisPlan(
        question_restatement="q",
        specs=[base().specs[0], base().specs[0].model_copy(update={"id": "b"})],
    )
    assert probe(tiny, OPEN_OR_CLOSE, two).verdict is Materiality.INCONCLUSIVE


# ============================================================================
# 5. Deterministic repeated output
# ============================================================================


def test_the_same_probe_gives_the_same_result(tiny):
    one = probe(tiny, OPEN_OR_CLOSE, base())
    two = probe(tiny, OPEN_OR_CLOSE, base())
    assert one == two
    assert one.model_dump_json() == two.model_dump_json()


# ============================================================================
# 6. No semantic substitution
# ============================================================================


@pytest.mark.parametrize(
    ("ambiguity", "operation"),
    [
        (AmbiguityKind.SNAPSHOT, ChangeMetrics(metrics=["opening_pipeline"])),
        (AmbiguityKind.SNAPSHOT, AddDimension(dimension="owner")),
        (AmbiguityKind.FILTER, ChangeStance(stance=RETRO)),
        (AmbiguityKind.FILTER, ChangePeriod(period=Q2)),
        (AmbiguityKind.PERIOD, snapshot_rule(SnapshotRule.PERIOD_CLOSE)),
    ],
)
def test_a_reading_that_changes_the_question_is_refused_unrun(tiny, ambiguity, operation):
    p = MaterialityProbe(
        ambiguity=ambiguity, interpretations=[reading("as asked"), reading("other", operation)]
    )
    result = probe(tiny, p, base())
    assert result.verdict is Materiality.INCONCLUSIVE
    assert "changes the question" in result.reason
    assert result.executed == 0


def test_a_binding_ambiguity_admits_no_operation_at_all():
    assert ALLOWED_OPERATIONS[AmbiguityKind.CONCEPT_BINDING] == ()


def test_the_result_never_names_a_chosen_reading():
    fields = set(MaterialityResult.model_fields)
    assert not {"chosen", "choice", "selected", "winner", "recommended"} & fields


# ============================================================================
# 7. Prospective temporal safety
# ============================================================================


def test_a_reading_past_the_knowledge_cutoff_is_inconclusive(tiny):
    horizon = MaterialityProbe(
        ambiguity=AmbiguityKind.PERIOD,
        interpretations=[reading("Q1"), reading("Q2", ChangePeriod(period=Q2))],
    )
    plan = base(stance=PROSP, knowledge_cutoff=date(2025, 2, 1))
    result = probe(tiny, horizon, plan)
    assert result.verdict is Materiality.INCONCLUSIVE
    assert result.executed == 1


def test_a_later_attribution_reading_under_a_prospective_stance_is_inconclusive(tiny):
    plan = base(metrics=["opening_pipeline"], dimensions=["owner"], stance=PROSP)
    result = probe(tiny, OWNER_WHEN, plan)
    assert result.verdict is Materiality.INCONCLUSIVE
    assert "stance_violation" in result.reason


def test_the_tool_refuses_a_base_wider_than_the_session_scope(tiny):
    ctx = tool_context(tiny, stance=PROSP)
    retro = base(metrics=["opening_pipeline"], dimensions=["owner"], stance=RETRO)
    from ai_analyst.agent.tools.surface import probe_materiality

    result = probe_materiality(ctx, OWNER_WHEN, retro)
    assert result.verdict is Materiality.INCONCLUSIVE
    assert result.executed == 0


def test_the_tool_tightens_every_reading_to_the_session_horizon(tiny):
    ctx = tool_context(tiny, stance=PROSP, horizon=date(2025, 2, 1))
    horizon = MaterialityProbe(
        ambiguity=AmbiguityKind.PERIOD,
        interpretations=[reading("Q1"), reading("Q2", ChangePeriod(period=Q2))],
    )
    from ai_analyst.agent.tools.surface import probe_materiality

    result = probe_materiality(ctx, horizon, base(stance=PROSP))
    assert result.verdict is Materiality.INCONCLUSIVE


# ============================================================================
# 8. Mutation: an ambiguity cannot be auto-resolved outside the probe rules
# ============================================================================


def test_mutation_without_the_rules_a_substituted_metric_would_be_evaluated(
    tiny, monkeypatch
):
    """Disable the rule check and a reading that swaps the metric runs.

    With the rule in place nothing executes; without it both readings execute
    and are compared, which is exactly the semantic substitution the rules
    exist to prevent. The rules are the load-bearing guard.
    """
    swap = MaterialityProbe(
        ambiguity=AmbiguityKind.SNAPSHOT,
        interpretations=[
            reading("deal count"),
            reading("opening pipeline", ChangeMetrics(metrics=["opening_pipeline"])),
        ],
    )
    guarded = probe(tiny, swap, base())
    assert (guarded.verdict, guarded.executed) == (Materiality.INCONCLUSIVE, 0)

    monkeypatch.setattr(materiality_module, "outside_rules", lambda p: None)
    mutated = probe(tiny, swap, base())
    assert mutated.executed == 2


def test_mutation_without_the_binding_rule_a_binding_probe_would_run(tiny, monkeypatch):
    binding = MaterialityProbe(
        ambiguity=AmbiguityKind.CONCEPT_BINDING,
        interpretations=[reading("a"), reading("b")],
    )
    assert probe(tiny, binding, base()).executed == 0
    monkeypatch.setattr(materiality_module, "outside_rules", lambda p: None)
    assert probe(tiny, binding, base()).executed == 2
