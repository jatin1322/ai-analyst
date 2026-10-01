"""The deterministic tool surface and the planner context (ARCHITECTURE 13.4, 13.5).

No model is involved. These tests pin what a tool may show under a stance and
horizon, and that the default context stays small and row-free.

Hand-computed tiny-fixture facts used below. Rows at 2025-01-01 and 2025-02-01
(13 rows) carry four stages:
    Negotiation 4 (OPP-001 x2, OPP-007 x2)   Discovery 5 (003 x2, 005 x2, 008)
    Proposal 2 (OPP-002 x2)                   Qualification 2 (OPP-004 x2)
Across every snapshot there are six, including Closed Won and Closed Lost.
"""

from __future__ import annotations

from datetime import date

import pytest

from ai_analyst.agent.context import build_planner_context
from ai_analyst.agent.tools.surface import (
    inspect_column,
    inspect_concept,
    inspect_dataset,
    inspect_relationship,
    inspect_sample_rows,
    inspect_values,
    list_available_metrics,
    request_clarification,
    run_analysis_plan,
    run_investigation_plan,
)
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
    Period,
    PeriodKind,
    SnapshotSelection,
)
from ai_analyst.contracts.rejection import RejectionCode
from ai_analyst.contracts.result import SnapshotRule, TrustTier
from ai_analyst.contracts.tools import (
    ClarificationOption,
    ClarificationReason,
    ClarificationRequest,
)
from ai_analyst.data.understanding import understand
from ai_analyst.session.state import SessionState
from tests.semantic.conftest import tool_context

PROSP, RETRO = AnalysisStance.PROSPECTIVE, AnalysisStance.RETROSPECTIVE
Q1 = Period(kind=PeriodKind.FISCAL_QUARTER, label="FY2025-Q1")
FEB = date(2025, 2, 1)


def opening_plan(stance=PROSP, rule=SnapshotRule.PERIOD_OPEN) -> AnalysisPlan:
    return AnalysisPlan(
        question_restatement="Q1 opening pipeline",
        specs=[
            AnalysisSpec(
                id="a", pattern=AnalysisPattern.POINT_IN_TIME, metrics=["opening_pipeline"],
                period=Q1, snapshot=SnapshotSelection(rule=rule), stance=stance,
            )
        ],
    )


# ============================================================================
# inspect_dataset / inspect_concept
# ============================================================================


def test_inspect_dataset_is_structured_and_horizon_bounded(tiny):
    view = inspect_dataset(tool_context(tiny, horizon=FEB))
    assert view.snapshot_count == 2
    assert view.last_snapshot == FEB
    assert {c.concept for c in view.concepts} >= {C.AMOUNT, C.STAGE}
    assert all(isinstance(c.usable, bool) for c in view.columns)


def test_a_contaminated_column_is_listed_with_its_class_but_unusable(production):
    view = inspect_dataset(tool_context(production), name_pattern="terminal_*")
    fate = next(c for c in view.columns if c.name == "terminal_fate")
    assert not fate.usable
    assert fate.reason == RejectionCode.COLUMN_NOT_KNOWABLE_AT_SNAPSHOT.value
    assert fate.information_class == "retrospective_outcome"


def test_a_name_pattern_is_a_glob_and_nothing_else(tiny):
    with pytest.raises(ValueError):
        inspect_dataset(tool_context(tiny), name_pattern="x'; DROP TABLE t; --")


def test_inspect_concept_reports_usability_under_the_stance(production):
    prospective = inspect_concept(tool_context(production), C.TERMINAL_OUTCOME)
    retrospective = inspect_concept(tool_context(production, RETRO), C.TERMINAL_OUTCOME)
    assert not prospective.usable
    assert prospective.rejection is RejectionCode.RETROSPECTIVE_CONCEPT_IN_PROSPECTIVE
    assert retrospective.usable


def test_inspect_concept_carries_the_reconstruction_verdict(tmp_path):
    """An export with no close date column, so the concept is reconstructed."""
    from ai_analyst.contracts.opportunity_snapshot_v1 import OPPORTUNITY_SNAPSHOT_V1
    from tests.fixtures.production_shape import write_production_csv
    from tests.semantic.conftest import build_engine

    source = tmp_path / "no_close.csv"
    write_production_csv(source, include_close_date=False)
    engine = build_engine(
        source, "rebuilt", tmp_path, tenant=None, column_registry=OPPORTUNITY_SNAPSHOT_V1
    )
    view = inspect_concept(tool_context(engine), C.EXPECTED_CLOSE_DATE)
    assert view.reconstruction_verdict == "valid"


def test_inspect_concept_reports_grants(custom_declared):
    view = inspect_concept(tool_context(custom_declared), C.AMOUNT)
    assert view.columns == ("enterprise_amount",)
    assert view.grants and "enterprise_amount" in view.grants[0]


# ============================================================================
# inspect_column / inspect_values: stance and horizon
# ============================================================================


def test_a_prospective_column_view_withholds_future_values(production):
    view = inspect_column(tool_context(production), "terminal_fate")
    assert view.summary is None
    assert view.values_withheld_reason
    assert view.rejection is RejectionCode.COLUMN_NOT_KNOWABLE_AT_SNAPSHOT


def test_a_retrospective_column_view_shows_them(production):
    view = inspect_column(tool_context(production, RETRO), "terminal_fate")
    assert view.summary is not None


def test_the_horizon_bounds_what_a_profile_reveals(tiny):
    """The leakage the stored profile would allow: 'Closed Won' exists only after 02-01."""
    unbounded = inspect_column(tool_context(tiny), "stage")
    bounded = inspect_column(tool_context(tiny, horizon=FEB), "stage")
    assert unbounded.summary.distinct_count == 6
    assert bounded.summary.distinct_count == 4
    assert bounded.summary.rows_considered == 13


def test_inspect_values_is_horizon_bounded_and_registered(tiny):
    ctx = tool_context(tiny, horizon=FEB)
    view = inspect_values(ctx, "stage")
    assert {v.value: v.count for v in view.values} == {
        "Discovery": 5, "Negotiation": 4, "Proposal": 2, "Qualification": 2,
    }
    assert "Closed Won" not in {v.value for v in view.values}
    assert view.reference == "q1"
    # The numbers are citable: they live in a registered result.
    registered = ctx.results.get("q1")
    assert registered.trust_tier is TrustTier.B
    assert registered.cell(0, "count") == 5


def test_inspect_values_caps_top_k(tiny):
    view = inspect_values(tool_context(tiny), "opp_id", top_k=500)
    assert len(view.values) <= 50


def test_text_content_is_never_valued_or_sampled(production):
    ctx = tool_context(production, RETRO)
    assert inspect_values(ctx, "NextStep").values == ()
    assert "free-text" in inspect_values(ctx, "NextStep").values_withheld_reason
    sample = inspect_sample_rows(ctx, n=3, columns=["opp_id", "NextStep", "ManagerNotes"])
    assert sample.columns == ("opp_id",)
    assert set(sample.excluded) == {"NextStep", "ManagerNotes"}


def test_a_granted_column_is_not_a_browsing_target(custom_declared):
    view = inspect_values(tool_context(custom_declared), "enterprise_amount")
    assert view.values == ()
    assert view.values_withheld_reason
    column = inspect_column(tool_context(custom_declared), "enterprise_amount")
    assert column.grants == ("readable as amount for filter, measure",)
    assert column.summary is None


def test_inspect_relationship_is_bounded_registered_and_permitted_only(production, tiny):
    ctx = tool_context(tiny, horizon=FEB)
    evidence = inspect_relationship(ctx, "segment", "stage")
    assert evidence.support == 13
    assert sum(p.count for p in evidence.pairs) == 13
    assert evidence.reference and evidence.trust_tier is TrustTier.B
    with pytest.raises(PermissionError):
        inspect_relationship(tool_context(production), "stage", "terminal_fate")


# ============================================================================
# inspect_sample_rows
# ============================================================================


def test_samples_are_capped_by_configuration(tiny):
    view = inspect_sample_rows(tool_context(tiny), n=1000)
    assert view.limit == tiny.settings.max_sample_rows
    assert len(view.rows) <= view.limit


def test_samples_respect_the_horizon(tiny):
    view = inspect_sample_rows(tool_context(tiny, horizon=FEB), n=20, columns=["as_of", "opp_id"])
    assert {row[0] for row in view.rows} <= {"2025-01-01", "2025-02-01"}


def test_samples_never_include_a_forbidden_column(production):
    view = inspect_sample_rows(tool_context(production), n=3, columns=["opp_id", "terminal_fate"])
    assert "terminal_fate" not in view.columns
    assert "terminal_fate" in view.excluded


# ============================================================================
# list_available_metrics and the terminal tools
# ============================================================================


def test_the_catalog_names_what_blocks_each_unavailable_metric(undeclared):
    catalog = list_available_metrics(tool_context(undeclared))
    assert [m.name for m in catalog.available] == ["deal_count"]
    blocked = {m.name: m for m in catalog.unavailable}
    assert "amount" in blocked["opening_pipeline"].missing_concepts


def test_run_analysis_plan_returns_handles_not_cells(tiny):
    ctx = tool_context(tiny)
    outcome = run_analysis_plan(ctx, opening_plan())
    assert outcome.executed
    (summary,) = outcome.results
    assert summary.reference == "q1"
    assert summary.columns == ("opening_pipeline",)
    assert not hasattr(summary, "rows")
    assert ctx.results.get("q1").cell(0, "opening_pipeline") is not None


def test_a_plan_cannot_widen_the_session_stance(tiny):
    outcome = run_analysis_plan(tool_context(tiny), opening_plan(stance=RETRO))
    assert not outcome.executed
    assert outcome.validation.has(RejectionCode.STANCE_VIOLATION)


def test_the_session_horizon_tightens_a_plans_cutoff(tiny):
    # Q1 PERIOD_CLOSE resolves to 03-31, after the session's 02-01 horizon.
    outcome = run_analysis_plan(
        tool_context(tiny, horizon=FEB), opening_plan(rule=SnapshotRule.PERIOD_CLOSE)
    )
    assert outcome.validation.has(RejectionCode.KNOWLEDGE_CUTOFF_VIOLATION)


def _investigation(stance) -> InvestigationPlan:
    return InvestigationPlan(
        question_restatement="q",
        hypotheses=[Hypothesis(statement="h", kind=HypothesisKind.DISTRIBUTION)],
        stance=stance,
        population=Population(period=Q1),
        variables=[Variable(id="changed", derived=DerivedFeatureRef(
            feature=DerivedFeature.CHANGED, concept=C.STAGE))],
        grouping=[Grouping(variable="changed")],
        operation=Operation(kind=StatisticalOperation.COUNT),
        evidence=EvidenceRequirements(min_group_support=1),
    )


def test_run_investigation_executes_under_the_scope(tiny):
    ctx = tool_context(tiny)
    outcome = run_investigation_plan(ctx, _investigation(PROSP))
    assert outcome.executed
    assert outcome.results[0].trust_tier is TrustTier.B


def test_an_investigation_cannot_widen_the_session_stance(tiny):
    outcome = run_investigation_plan(tool_context(tiny), _investigation(RETRO))
    assert outcome.validation.has(RejectionCode.STANCE_VIOLATION)


# ============================================================================
# request_clarification
# ============================================================================


def test_alternatives_for_an_unavailable_concept_are_filled_deterministically(production):
    request = ClarificationRequest(
        reason=ClarificationReason.CONCEPT_UNAVAILABLE,
        concept=C.CUSTOMER_SEGMENT,
        question="Customer segment is not in this dataset. Break down by something else?",
        # A model's invented substitute, which must not survive.
        available_alternatives=["region", "customer_tier"],
    )
    filled = request_clarification(tool_context(production), request)
    assert "region" not in filled.available_alternatives
    assert "customer_tier" not in filled.available_alternatives
    assert "customer_segment" not in filled.available_alternatives
    assert {"owner_id", "stage", "forecast_category"} <= set(filled.available_alternatives)


def test_a_clarification_offers_two_to_four_options():
    with pytest.raises(ValueError):
        ClarificationRequest(
            reason=ClarificationReason.MISSING_PERIOD,
            question="Which quarter?",
            options=[ClarificationOption(label="Q1")],
        )


# ============================================================================
# The planner context
# ============================================================================


def _context(engine, **kwargs):
    understanding = understand(engine.dataset, settings=engine.settings)
    return build_planner_context(
        engine.dataset, understanding, tool_context(engine), **kwargs
    )


def test_the_default_context_fits_the_budget_at_realistic_width(production):
    """145 columns, the production export's width."""
    context = _context(production)
    assert len(production.registry.columns) >= 140
    assert context.estimated_tokens <= 2000
    assert context.within_budget


def test_the_default_context_carries_no_rows_and_no_text(production):
    text = _context(production).render()
    assert "OPP-000" not in text  # an opportunity id from the rows
    assert "(null)" not in text
    # Narrative columns are named, never valued.
    assert "catalogued only, no content" in text


def test_the_default_context_states_metrics_and_temporal_rules(production, undeclared):
    context = _context(production)
    assert context.catalog.startswith("metrics (prospective):")
    assert "retrospective only:" in context.temporal
    assert "terminal_outcome" in context.temporal
    # The production tenant declares its fiscal year; the undeclared one does not.
    assert "UNRESOLVED" not in context.temporal
    assert "UNRESOLVED" in _context(undeclared).temporal


def test_an_over_budget_context_degrades_the_column_index(production):
    context = _context(production, budget=1200)
    assert context.index_degraded
    assert "column index withheld" in context.card


def test_the_session_block_carries_the_active_plan(tiny):
    session = SessionState(session_id="s", dataset_id="tiny")
    session.record(opening_plan())
    context = _context(tiny, session=session)
    assert "opening_pipeline" in context.session
    assert "period FY2025-Q1" in context.session
