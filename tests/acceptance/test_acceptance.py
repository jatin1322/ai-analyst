"""Deterministic acceptance suite: question to provenance, end to end.

Every case walks the whole deterministic chain the LLM layer will sit on:

    question -> plan -> gate -> compiled SQL -> DuckDB -> ResultSet
             -> trust -> render -> provenance scan

There is no planner and no responder yet (both are the next milestone). Each
case therefore supplies what they will produce: a hand-written typed plan in
place of the planner, and a hand-written draft with reference tokens in place
of the responder. Everything after that is the real engine.

Expected values are hand-computed from the fixture CSVs, with the arithmetic in
each case. Expected trust tiers are derived from the expected trust factors
through `FACTOR_CEILINGS`, never read back from a result.

Fixtures:
  * tiny  - tests/fixtures/tiny/snapshots.csv with its tenant declaration.
  * moves - tests/fixtures/tiny/bridge_moves.csv, built so every bridge term is
            non-zero (tiny's created and pulled-in terms are zero in both
            quarters, which would make those cases vacuous).
  * granted - custom_column.csv with the tenant declaring enterprise_amount
            (deal_amount / 2) as its amount concept.

Tiny's status is inferred from stage keywords, so every result that reads
status carries STATUS_NOT_AUTHORITATIVE and is at most tier B. That is the
fixture being honest, not a defect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from ai_analyst.contracts.answer import AnswerDraft, NumeralSource
from ai_analyst.contracts.concepts import BusinessConcept as C
from ai_analyst.contracts.investigation import (
    DerivedFeature,
    DerivedFeatureRef,
    EvidenceRequirements,
    Grouping,
    Hypothesis,
    HypothesisKind,
    InvestigationPlan,
    Operation,
    Population,
    StatisticalOperation,
    Variable,
)
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
from ai_analyst.contracts.rejection import RejectionCode
from ai_analyst.contracts.result import (
    FACTOR_CEILINGS,
    ResultSet,
    SnapshotRule,
    TrustFactorKind,
    TrustTier,
)
from ai_analyst.semantic.compiler import UnvalidatedPlan, compile_plan
from ai_analyst.semantic.investigation import run_investigation, validate_investigation
from ai_analyst.validation.provenance import scan
from ai_analyst.validation.rendering import ResultRegistry, format_value, render
from tests.semantic.conftest import (
    CUSTOM_CSV,
    MOVES_CSV,
    TINY_CSV,
    TINY_TENANT,
    Engine,
    build_engine,
)

Q1 = Period(kind=PeriodKind.FISCAL_QUARTER, label="FY2025-Q1")
Q2 = Period(kind=PeriodKind.FISCAL_QUARTER, label="FY2025-Q2")
OPEN = SnapshotSelection(rule=SnapshotRule.PERIOD_OPEN)
CLOSE = SnapshotSelection(rule=SnapshotRule.PERIOD_CLOSE)
PROSP, RETRO = AnalysisStance.PROSPECTIVE, AnalysisStance.RETROSPECTIVE

SEMANTIC = TrustFactorKind.SEMANTIC_PATH
INVESTIGATION = TrustFactorKind.INVESTIGATION_PATH
STATUS = TrustFactorKind.STATUS_NOT_AUTHORITATIVE
GRANT = TrustFactorKind.USAGE_GRANT


# ============================================================================
# Fixtures: one engine per dataset for the module
# ============================================================================


@pytest.fixture(scope="module")
def engines(tmp_path_factory) -> dict[str, Engine]:
    from ai_analyst.contracts.tenant import TenantProfile

    root = tmp_path_factory.mktemp("acceptance")
    granted = TenantProfile(
        tenant_id="custom",
        source="tests fixture",
        concept_columns={**TINY_TENANT.concept_columns, C.AMOUNT: ("enterprise_amount",)},
        fiscal_year_start_month=1,
    )
    return {
        "tiny": build_engine(TINY_CSV, "acc_tiny", root / "tiny"),
        "moves": build_engine(MOVES_CSV, "acc_moves", root / "moves"),
        "granted": build_engine(CUSTOM_CSV, "acc_granted", root / "granted", tenant=granted),
    }


# ============================================================================
# The chain
# ============================================================================


def tier_of(factors: set[TrustFactorKind]) -> TrustTier:
    """The weakest ceiling among the factors: C is weaker than B, B than A."""
    return max((FACTOR_CEILINGS[f] for f in factors), key=lambda t: t.value)


@dataclass
class Case:
    name: str
    dataset: str
    question: str
    # The planner's output, hand-written until the planner exists.
    plan: AnalysisPlan | InvestigationPlan
    # The responder's output, hand-written: prose and reference tokens only.
    draft: str
    # Hand-computed cells: {row index: {column: value}}.
    expected: dict[int, dict[str, object]]
    # The trust factors the inputs must produce. The tier follows from these.
    factors: set[TrustFactorKind]
    # Snapshots the result must report having read.
    snapshots: list[date] = field(default_factory=list)


@dataclass
class Trace:
    result: ResultSet
    compiled_sql: str
    rendered_text: str
    report: object


def run_chain(engine: Engine, case: Case) -> Trace:
    """Every deterministic stage, each checked as it is passed."""
    # plan -> gate
    if isinstance(case.plan, InvestigationPlan):
        outcome = validate_investigation(
            case.plan, dataset_id=engine.dataset_id, registry=engine.registry,
            bindings=engine.bindings, snapshots=engine.snapshots, calendar=engine.calendar,
        )
        assert outcome.ok, [r.message for r in outcome.validation.rejections]
        # gate -> compiled -> DuckDB -> ResultSet
        with engine.store.connect() as conn:
            result = run_investigation(conn, engine.scan, outcome, dataset_id=engine.dataset_id,
                                       calendar=engine.calendar)
        compiled_sql = result.compiled_sql
        retrospective = case.plan.stance is RETRO
    else:
        outcome = engine.gate(*case.plan.specs, question=case.question)
        assert outcome.ok, [r.message for r in outcome.validation.rejections]
        (compiled,) = compile_plan(engine.scan, case.plan, outcome)
        from ai_analyst.semantic.execute import run_plan

        with engine.store.connect() as conn:
            (result,) = run_plan(conn, engine.scan, case.plan, outcome,
                                 dataset_id=engine.dataset_id, calendar=engine.calendar,
                                 settings=engine.settings)
        compiled_sql = compiled.sql
        # What ran is exactly what was compiled.
        assert result.compiled_sql == compiled_sql
        retrospective = case.plan.specs[0].stance is RETRO

    # ResultSet: hand-computed cells.
    for index, cells in case.expected.items():
        for column, value in cells.items():
            assert result.cell(index, column) == value, (case.name, index, column)

    # Snapshots: every one read is reported, as a date, not only as a rule.
    reported = [s.resolved_as_of for s in result.resolved_snapshots]
    for expected_date in case.snapshots:
        assert expected_date in reported, (case.name, reported)

    # Trust: computed from the inputs, and equal to what those inputs imply.
    kinds = {f.kind for f in result.trust.factors}
    assert kinds == case.factors, (case.name, kinds)
    assert result.trust_tier is tier_of(case.factors)
    assert result.trust_tier is not TrustTier.C

    # Render: the responder's draft, tokens substituted by the renderer.
    registry = ResultRegistry()
    assert registry.register(result) == "q1"
    rendered = render(
        AnswerDraft(headline=case.draft), registry, result.trust,
        assumptions=list(result.assumptions), warnings=list(result.warnings),
        retrospective=retrospective,
        association=isinstance(case.plan, InvestigationPlan),
    )
    # Every value the draft cites appears exactly as the renderer formats it.
    kinds_by_column = {c.name: c.kind for c in result.columns}
    for token in [r for r in rendered.references if r.startswith("q1.r")]:
        _, row, column = token.split(".")
        value = result.cell(int(row[1:]), column)
        assert format_value(value, kinds_by_column[column]) in rendered.text

    # Provenance: every numeral traces to a result, the system, the user, or
    # structure. An unsupported number fails the answer.
    report = scan(rendered, question=case.question,
                  known_dates=frozenset(d.isoformat() for d in reported))
    assert report.ok, [(f.text, f.reason) for f in report.unverified]
    assert any(f.source is NumeralSource.RESULT for f in report.findings)
    return Trace(result=result, compiled_sql=compiled_sql, rendered_text=rendered.text,
                 report=report)


def point(metrics, period=Q2, snapshot=OPEN, **kwargs) -> AnalysisPlan:
    kwargs.setdefault("pattern", AnalysisPattern.POINT_IN_TIME)
    return AnalysisPlan(
        question_restatement="acceptance",
        specs=[AnalysisSpec(id="s", metrics=list(metrics), period=period, snapshot=snapshot,
                            **kwargs)],
    )


# ============================================================================
# The ten cases
# ============================================================================

CASES = [
    # 1. Opening pipeline. Q2 open is 2025-04-01; open with close in Q2:
    #    OPP-001 100000 + OPP-003 75000 + OPP-004 250000 + OPP-008 90000 = 515000.
    Case(
        name="opening pipeline",
        dataset="tiny",
        question="What was Q2 opening pipeline?",
        plan=point(["opening_pipeline"]),
        draft="Q2 opening pipeline was {{q1.r0.opening_pipeline}}, "
              "measured at the snapshot of {{q1.meta.resolved_as_of}}.",
        expected={0: {"opening_pipeline": Decimal("515000.00")}},
        factors={SEMANTIC, STATUS},
        snapshots=[date(2025, 4, 1)],
    ),
    # 2. Ending pipeline. Q2 close is 2025-06-30; open with close in Q2:
    #    OPP-004 (close 06-20) 180000 only. OPP-001 moved to 08-15; 003, 008 won.
    Case(
        name="ending pipeline",
        dataset="tiny",
        question="What was Q2 ending pipeline?",
        plan=point(["ending_pipeline"], snapshot=CLOSE),
        draft="Q2 ending pipeline was {{q1.r0.ending_pipeline}}.",
        expected={0: {"ending_pipeline": Decimal("180000.00")}},
        factors={SEMANTIC, STATUS},
        snapshots=[date(2025, 6, 30)],
    ),
    # 3. Created after quarter start (moves, Q1). Entrants to the closing
    #    pipeline: B-003 (pulled in) and B-004, created 2025-02-05, inside Q1,
    #    40000 at 03-31.
    Case(
        name="created after quarter start",
        dataset="moves",
        question="How much Q1 pipeline was created after the quarter started?",
        plan=point(["created_pipeline"], period=Q1, snapshot=CLOSE),
        draft="Pipeline created during Q1 was {{q1.r0.created_pipeline}}.",
        expected={0: {"created_pipeline": Decimal("40000.00")}},
        factors={SEMANTIC, STATUS},
        snapshots=[date(2025, 1, 1), date(2025, 3, 31)],
    ),
    # 4. Slipped (tiny, Q2). Leavers of the opening pipeline: OPP-001 still open
    #    with close moved 05-15 -> 08-15: slipped, at its opening amount 100000.
    #    OPP-003 and OPP-008 closed won.
    Case(
        name="slipped",
        dataset="tiny",
        question="How much Q2 pipeline slipped out of the quarter?",
        plan=point(["slipped_pipeline"], snapshot=CLOSE),
        draft="{{q1.r0.slipped_pipeline}} of Q2 pipeline slipped to a later quarter.",
        expected={0: {"slipped_pipeline": Decimal("100000.00")}},
        factors={SEMANTIC, STATUS},
        snapshots=[date(2025, 4, 1), date(2025, 6, 30)],
    ),
    # 5. Pulled in (moves, Q1). B-003 close 05-10 at 01-01 moves to 03-20 at
    #    03-31, still open, created before the quarter: 30000.
    Case(
        name="pulled in",
        dataset="moves",
        question="How much pipeline was pulled into Q1?",
        plan=point(["pulled_in_pipeline"], period=Q1, snapshot=CLOSE),
        draft="{{q1.r0.pulled_in_pipeline}} was pulled into Q1.",
        expected={0: {"pulled_in_pipeline": Decimal("30000.00")}},
        factors={SEMANTIC, STATUS},
        snapshots=[date(2025, 1, 1), date(2025, 3, 31)],
    ),
    # 6. Win rate (tiny, Q1 at 2025-03-31, closed-only). Close in Q1 and closed:
    #    won OPP-002 (close 03-25), lost OPP-007 (close 02-28): 1 / 2 = 0.5.
    Case(
        name="win rate",
        dataset="tiny",
        question="What was the Q1 win rate?",
        plan=point(["win_rate"], period=Q1, snapshot=CLOSE, pattern=AnalysisPattern.RATE),
        draft="The Q1 win rate was {{q1.r0.ratio}}: {{q1.r0.numerator}} won of "
              "{{q1.r0.denominator}} closed.",
        expected={0: {"numerator": 1, "denominator": 2, "ratio": Decimal("0.5")}},
        factors={SEMANTIC, STATUS},
        snapshots=[date(2025, 3, 31)],
    ),
    # 7. Cohort (tiny, Q1, retrospective). Open at 01-01, fate at 03-31, amount
    #    at 01-01: lost OPP-007 60000; open 001, 003, 004, 005 =
    #    100000 + 75000 + 200000 + 40000 = 415000; won OPP-002 50000.
    #    Rows ordered by state: lost, open, won.
    Case(
        name="cohort",
        dataset="tiny",
        question="What happened to the deals that were open at the start of Q1?",
        plan=point(["cohort_fate"], period=Q1, pattern=AnalysisPattern.COHORT_TRACE,
                   stance=RETRO),
        draft="Of the Q1 opening cohort, {{q1.r2.opportunity_count}} deal worth "
              "{{q1.r2.cohort_amount}} was won and {{q1.r0.opportunity_count}} worth "
              "{{q1.r0.cohort_amount}} was lost; {{q1.r1.opportunity_count}} worth "
              "{{q1.r1.cohort_amount}} were still open.",
        expected={
            0: {"terminal_state": "lost", "opportunity_count": 1,
                "cohort_amount": Decimal("60000.00")},
            1: {"terminal_state": "open", "opportunity_count": 4,
                "cohort_amount": Decimal("415000.00")},
            2: {"terminal_state": "won", "opportunity_count": 1,
                "cohort_amount": Decimal("50000.00")},
        },
        factors={SEMANTIC, STATUS},
        snapshots=[date(2025, 1, 1), date(2025, 3, 31)],
    ),
    # 10. A usage-grant result (granted, Q1). enterprise_amount = deal_amount / 2,
    #     read through the tenant's declaration of the amount concept:
    #     OPP-001 50000 + OPP-005 20000 + OPP-007 30000 = 100000.
    Case(
        name="usage grant",
        dataset="granted",
        question="What was Q1 opening pipeline in enterprise terms?",
        plan=point(["opening_pipeline"], period=Q1),
        draft="Q1 opening pipeline was {{q1.r0.opening_pipeline}}.",
        expected={0: {"opening_pipeline": Decimal("100000.00")}},
        factors={SEMANTIC, STATUS, GRANT},
        snapshots=[date(2025, 1, 1)],
    ),
]

# 8. A non-registry investigation (tiny, Q1 cohort at 01-01, retrospective).
#    "Did deals whose forecast category changed during Q1 end up differently?"
#    No registry metric answers it. Forecast category 01-01 -> 03-31:
#    OPP-002 Best Case -> Commit, OPP-007 Commit -> Omitted: changed (2);
#    OPP-001, 003, 004, 005 unchanged (4).
INVESTIGATION_CASE = Case(
    name="non-registry investigation",
    dataset="tiny",
    question="How many Q1 deals had their forecast category change during the quarter?",
    plan=InvestigationPlan(
        question_restatement="Count the Q1 opening cohort by whether forecast category "
                             "changed during the quarter.",
        hypotheses=[Hypothesis(statement="forecast changes are common",
                               kind=HypothesisKind.ASSOCIATION)],
        stance=RETRO,
        population=Population(period=Q1, cohort=OPEN, window_end=CLOSE),
        variables=[Variable(id="fc_changed", derived=DerivedFeatureRef(
            feature=DerivedFeature.CHANGED, concept=C.FORECAST_CATEGORY))],
        grouping=[Grouping(variable="fc_changed")],
        operation=Operation(kind=StatisticalOperation.COUNT),
        evidence=EvidenceRequirements(min_group_support=1),
    ),
    draft="{{q1.r1.units}} deals changed forecast category during Q1 and "
          "{{q1.r0.units}} did not.",
    expected={0: {"fc_changed": "false", "units": 4}, 1: {"fc_changed": "true", "units": 2}},
    factors={INVESTIGATION, STATUS},
    snapshots=[date(2025, 1, 1), date(2025, 3, 31)],
)


@pytest.mark.parametrize("case", [*CASES, INVESTIGATION_CASE], ids=lambda c: c.name)
def test_acceptance_case(engines, case):
    trace = run_chain(engines[case.dataset], case)
    # The chain is deterministic end to end.
    again = run_chain(engines[case.dataset], case)
    assert again.compiled_sql == trace.compiled_sql
    assert again.rendered_text == trace.rendered_text


def test_the_usage_grant_is_disclosed_in_the_answer(engines):
    (case,) = [c for c in CASES if c.name == "usage grant"]
    trace = run_chain(engines["granted"], case)
    assert "enterprise_amount" in trace.rendered_text
    assert trace.result.compilation.usage_grants


def test_the_investigation_is_capped_at_b_and_says_it_is_an_association(engines):
    trace = run_chain(engines["tiny"], INVESTIGATION_CASE)
    assert trace.result.trust_tier is TrustTier.B


# 9. A prospective temporal-safety query. "Break Q1 opening pipeline down by
#    each deal's stage at quarter close" is hindsight under a prospective
#    stance: the stage at close reveals OPP-007 as Closed Lost. The chain must
#    stop at the gate with a structured rejection, and no number may be emitted.
def test_acceptance_prospective_temporal_safety(engines):
    engine = engines["tiny"]
    question = "Break down Q1 opening pipeline by each deal's stage at quarter close."
    plan = point(["opening_pipeline"], period=Q1, dimensions=["stage"],
                 attribution=Attribution.AT_CLOSE, stance=PROSP)
    outcome = engine.gate(*plan.specs, question=question)
    assert not outcome.ok
    (rejection,) = outcome.validation.rejections
    assert rejection.code is RejectionCode.STANCE_VIOLATION
    assert rejection.field == "attribution"
    assert "2025-03-31" in rejection.message and "2025-01-01" in rejection.message
    assert rejection.remedy
    # Nothing downstream will run: the compiler refuses an unvalidated plan.
    with pytest.raises(UnvalidatedPlan):
        compile_plan(engine.scan, plan, outcome)


def test_acceptance_the_same_question_is_answerable_retrospectively(engines):
    """The control for case 9: hindsight, stated as such, is allowed."""
    question = "Break down Q1 opening pipeline by each deal's stage at quarter close."
    case = Case(
        name="retrospective control",
        dataset="tiny",
        question=question,
        plan=point(["opening_pipeline"], period=Q1, dimensions=["stage"],
                   attribution=Attribution.AT_CLOSE, stance=RETRO),
        # Stage at 03-31: Closed Lost OPP-007 60000; Discovery OPP-005 40000;
        # Negotiation OPP-001 100000. Rows ordered by stage.
        draft="Closed Lost deals held {{q1.r0.opening_pipeline}} of Q1 opening pipeline.",
        expected={0: {"stage": "Closed Lost", "opening_pipeline": Decimal("60000.00")},
                  1: {"stage": "Discovery", "opening_pipeline": Decimal("40000.00")},
                  2: {"stage": "Negotiation", "opening_pipeline": Decimal("100000.00")}},
        factors={SEMANTIC, STATUS},
        snapshots=[date(2025, 1, 1), date(2025, 3, 31)],
    )
    trace = run_chain(engines["tiny"], case)
    assert trace.result.compilation.attributions[0].read_as_of == date(2025, 3, 31)


def test_an_unsupported_number_fails_the_chain(engines):
    """The provenance stage is live: a literal the responder invented fails."""
    (case,) = [c for c in CASES if c.name == "opening pipeline"]
    bad = Case(**{**case.__dict__,
                  "draft": "Q2 opening pipeline was {{q1.r0.opening_pipeline}}, "
                           "against a plan of $600,000."})
    with pytest.raises(AssertionError, match="600,000"):
        run_chain(engines["tiny"], bad)


def test_every_acceptance_dataset_is_a_repository_fixture():
    for path in (TINY_CSV, MOVES_CSV, CUSTOM_CSV):
        assert Path(path).exists()
