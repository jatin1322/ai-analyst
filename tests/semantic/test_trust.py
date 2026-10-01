"""Trust is computed from structured factors, never chosen (ARCHITECTURE 13.10).

The tests below prove two things. First, the tier follows the factor table for
every factor kind the architecture names. Second, nothing a model could emit
reaches the tier: there is no field for it, supplying one is an error, and the
assessment function takes no tier-shaped argument.
"""

from __future__ import annotations

import inspect
from datetime import date

import pytest
from pydantic import ValidationError

from ai_analyst.contracts.plan import (
    AnalysisPattern,
    AnalysisPlan,
    AnalysisSpec,
    Period,
    PeriodKind,
    SnapshotSelection,
)
from ai_analyst.contracts.result import (
    FACTOR_CEILINGS,
    ResultColumn,
    ResultSet,
    SnapshotRule,
    TrustAssessment,
    TrustFactor,
    TrustFactorKind,
    TrustTier,
)
from ai_analyst.contracts.schema import DataType
from ai_analyst.semantic.calendar import resolve_calendar
from ai_analyst.semantic.execute import AbstentionRequired, run_plan
from ai_analyst.semantic.resolver import ConceptResolver
from ai_analyst.semantic.trust import AnalysisPath, assess

K = TrustFactorKind
Q1 = Period(kind=PeriodKind.FISCAL_QUARTER, label="FY2025-Q1")


def factor(kind: TrustFactorKind) -> TrustFactor:
    return TrustFactor(kind=kind, subject="s", reason=f"{kind.value} reason")


# -- the table ----------------------------------------------------------------


def test_every_factor_kind_has_a_ceiling():
    assert set(FACTOR_CEILINGS) == set(TrustFactorKind)


@pytest.mark.parametrize(
    ("kinds", "expected"),
    [
        ((), TrustTier.A),
        ((K.SEMANTIC_PATH,), TrustTier.A),
        ((K.SEMANTIC_PATH, K.USAGE_GRANT), TrustTier.A),
        ((K.INVESTIGATION_PATH,), TrustTier.B),
        ((K.SEMANTIC_PATH, K.BINDING_INFERRED), TrustTier.B),
        ((K.SEMANTIC_PATH, K.LINEAGE_UNCONFIRMED), TrustTier.B),
        ((K.SEMANTIC_PATH, K.RECONCILIATION_WARNING), TrustTier.B),
        ((K.SEMANTIC_PATH, K.RECONSTRUCTION_UNVERIFIED), TrustTier.B),
        ((K.SEMANTIC_PATH, K.RETROSPECTIVE_READ), TrustTier.B),
        ((K.SEMANTIC_PATH, K.CALENDAR_UNRESOLVED), TrustTier.B),
        ((K.SEMANTIC_PATH, K.CONCEPT_UNAVAILABLE), TrustTier.C),
        ((K.INVESTIGATION_PATH, K.UNRESOLVED_AMBIGUITY), TrustTier.C),
        ((K.SEMANTIC_PATH, K.SANITY_FAILED), TrustTier.C),
    ],
)
def test_the_tier_is_the_weakest_factor(kinds, expected):
    assert TrustAssessment(factors=tuple(factor(k) for k in kinds)).tier is expected


def test_an_investigation_is_never_better_than_b_even_with_every_input_confirmed():
    assessment = TrustAssessment(factors=(factor(K.INVESTIGATION_PATH), factor(K.USAGE_GRANT)))
    assert assessment.tier is TrustTier.B


def test_a_tier_b_assessment_names_its_reasons():
    assessment = TrustAssessment(factors=(factor(K.SEMANTIC_PATH), factor(K.BINDING_INFERRED)))
    assert assessment.reasons == ["binding_inferred reason"]


def test_a_grant_is_disclosed_without_lowering_the_tier():
    assessment = TrustAssessment(factors=(factor(K.SEMANTIC_PATH), factor(K.USAGE_GRANT)))
    assert assessment.tier is TrustTier.A
    assert assessment.reasons == []
    assert assessment.disclosures == ["usage_grant reason"]


def test_an_answer_takes_the_weakest_of_its_results():
    a = TrustAssessment(factors=(factor(K.SEMANTIC_PATH),))
    b = TrustAssessment(factors=(factor(K.INVESTIGATION_PATH),))
    assert a.combine(b).tier is TrustTier.B


# -- a model cannot influence it ---------------------------------------------------


def test_an_assessment_has_no_tier_field_to_set():
    with pytest.raises(ValidationError):
        TrustAssessment.model_validate({"factors": [], "tier": "A"})


def test_a_factor_cannot_carry_its_own_tier():
    """A model emitting an investigation factor that claims to cost nothing."""
    with pytest.raises(ValidationError):
        TrustFactor.model_validate(
            {"kind": "investigation_path", "reason": "trust me", "tier": "A"}
        )


def test_a_result_set_refuses_a_supplied_tier():
    with pytest.raises(ValidationError):
        ResultSet(
            query_id="q1",
            columns=[ResultColumn(name="x", dtype=DataType.BIGINT)],
            trust_tier="A",
        )


def test_the_tier_cannot_be_assigned_after_construction():
    assessment = TrustAssessment(factors=(factor(K.INVESTIGATION_PATH),))
    with pytest.raises((AttributeError, ValidationError, TypeError)):
        assessment.tier = TrustTier.A  # type: ignore[misc]
    assert assessment.tier is TrustTier.B


def test_the_assessment_function_accepts_no_tier():
    parameters = set(inspect.signature(assess).parameters)
    assert not {p for p in parameters if "tier" in p or "trust" in p}


def test_a_model_supplied_factor_list_is_ignored_by_the_pipeline(tiny):
    """The pipeline computes factors from the resolver, not from anything handed in."""
    result = tiny.one(
        AnalysisSpec(
            id="a", pattern=AnalysisPattern.POINT_IN_TIME, metrics=["opening_pipeline"],
            period=Q1, snapshot=SnapshotSelection(rule=SnapshotRule.PERIOD_OPEN),
        )
    )
    # tiny derives status from stage keywords: the factor is always there.
    assert K.STATUS_NOT_AUTHORITATIVE in {f.kind for f in result.trust.factors}
    assert result.trust_tier is TrustTier.B


# -- computed from real inputs ---------------------------------------------------


def _resolver(engine) -> ConceptResolver:
    return ConceptResolver(
        dataset_id=engine.dataset_id, registry=engine.registry, bindings=engine.bindings
    )


def test_the_path_alone_sets_the_floor(tiny):
    resolver = _resolver(tiny)
    assert assess(path=AnalysisPath.SEMANTIC, resolver=resolver).tier is TrustTier.A
    assert assess(path=AnalysisPath.INVESTIGATION, resolver=resolver).tier is TrustTier.B
    assert assess(path=AnalysisPath.GUARDED_SQL, resolver=resolver).tier is TrustTier.B


def test_an_unresolved_calendar_caps_a_fiscal_period_but_not_a_month(tiny):
    resolver = _resolver(tiny)
    unresolved = resolve_calendar(None, tiny.settings)
    fiscal = assess(
        path=AnalysisPath.SEMANTIC, resolver=resolver, calendar=unresolved,
        period_kind=PeriodKind.FISCAL_QUARTER,
    )
    month = assess(
        path=AnalysisPath.SEMANTIC, resolver=resolver, calendar=unresolved,
        period_kind=PeriodKind.MONTH,
    )
    assert fiscal.tier is TrustTier.B
    assert K.CALENDAR_UNRESOLVED in {f.kind for f in fiscal.factors}
    assert month.tier is TrustTier.A


def test_drift_beyond_tolerance_caps_at_b(tiny):
    from ai_analyst.semantic.calendar import ResolvedPeriod

    period = ResolvedPeriod(
        kind=PeriodKind.CUSTOM, start=date(2025, 1, 1), end=date(2025, 4, 15), label="c"
    )
    snapshot = tiny.snapshots.resolve(SnapshotRule.PERIOD_CLOSE, period=period)
    assessment = assess(
        path=AnalysisPath.SEMANTIC, resolver=_resolver(tiny), snapshot=snapshot,
        max_drift_days=10,
    )
    assert K.SNAPSHOT_DRIFT in {f.kind for f in assessment.factors}
    assert assessment.tier is TrustTier.B


def test_an_unresolved_ambiguity_executes_nothing(tiny):
    spec = AnalysisSpec(
        id="a", pattern=AnalysisPattern.POINT_IN_TIME, metrics=["opening_pipeline"],
        period=Q1, snapshot=SnapshotSelection(rule=SnapshotRule.PERIOD_OPEN),
    )
    plan = AnalysisPlan(
        question_restatement="q", specs=[spec],
        unresolved_ambiguities=["which quarter did the user mean"],
    )
    outcome = tiny.gate(spec)
    with tiny.store.connect() as conn, pytest.raises(AbstentionRequired) as exc:
        run_plan(conn, tiny.scan, plan, outcome, dataset_id="tiny", calendar=tiny.calendar)
    assert exc.value.trust.tier is TrustTier.C


def test_a_reconciliation_warning_reaches_the_result_as_a_b_factor(tmp_path, production_csv):
    """The 13.1 verdict flows into trust through the binding caveats."""
    from ai_analyst.contracts.concepts import BusinessConcept
    from ai_analyst.contracts.opportunity_snapshot_v1 import OPPORTUNITY_SNAPSHOT_V1
    from ai_analyst.contracts.tenant import TenantProfile
    from tests.semantic.conftest import build_engine

    engine = build_engine(
        production_csv, "recon", tmp_path,
        tenant=TenantProfile(tenant_id="p", source="x", fiscal_year_start_month=1),
        column_registry=OPPORTUNITY_SNAPSHOT_V1,
    )
    binding = engine.bindings.get(BusinessConcept.EXPECTED_CLOSE_DATE)
    resolver = _resolver(engine)
    resolver.try_resolve(BusinessConcept.EXPECTED_CLOSE_DATE, load_bearing=False)
    kinds = {f.kind for f in resolver.trust_factors}
    # The synthetic export corroborates everything, so it is VALID with no
    # warning; the caveat mapping is exercised directly below.
    assert not [c for c in binding.caveats if c.startswith("reconciliation_warning")]
    assert K.RECONCILIATION_WARNING not in kinds

    resolver._note_caveat(
        BusinessConcept.EXPECTED_CLOSE_DATE,
        "reconciliation_warning:close_date_matches_eoq_close_diff",
    )
    assert resolver.trust_tier is TrustTier.B
