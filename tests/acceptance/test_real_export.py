"""Real-export acceptance: the ten cases on a real tenant export. Opt-in.

Skipped unless `AI_ANALYST_REAL_EXPORT` names a local parquet export shaped
like `opportunity_snapshot_v1` (the sibling tenant's export). The data is not
in the repository, and ingesting it takes several minutes, so it is never part
of the default run. Set `AI_ANALYST_REAL_EXPORT_DATA_ROOT` to keep the ingested
dataset between runs.

    AI_ANALYST_REAL_EXPORT=/path/to/export.parquet pytest tests/acceptance -q

Every expected value comes from `reference_export.Reference`, an independent
raw-SQL implementation over the parquet that imports nothing from the engine.
Nothing is printed: assertions compare aggregates, and a failure message names
a case and a column, never a row or an identifier.

What this suite does **not** resolve. The tenant profile below is a test
declaration, not a production one:

* `opportunity_status` is declared as the stage-derived status column so that
  status-dependent metrics can run at all. That does **not** resolve which
  production column is the authoritative status; every result keeps the
  STATUS_NOT_AUTHORITATIVE factor and is capped at tier B.
* `OpportunityOwnerDivision` is classified by the test, explicitly, as an as-of
  text dimension, to exercise a generic usage grant. It is not a confirmed
  production classification.
* The eoq_close_diff and CD_in_qtr conventions, the fiscal calendar and the
  date declarations stay unresolved; their factors are expected, not waived.

The export is almost static across its snapshots: in the sibling, only open
deals whose close date is the snapshot day move, a day at a time, so they never
leave a period that contains their snapshot. Bridge movement terms are
therefore genuinely zero here; non-zero bridge behaviour is covered by the
`moves` fixture in `test_acceptance.py`. Zero is still checked against the
reference.
"""

from __future__ import annotations

import os
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from ai_analyst.contracts.columns import (
    Availability,
    ColumnCategory,
    ColumnClassification,
    Disposition,
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
    Attribution,
    CreationBasis,
    Period,
    PeriodKind,
    SnapshotSelection,
)
from ai_analyst.contracts.rejection import RejectionCode
from ai_analyst.contracts.result import SnapshotRule, TrustFactorKind
from ai_analyst.contracts.tenant import ColumnDeclaration, TenantProfile
from ai_analyst.semantic.compiler import UnvalidatedPlan, compile_plan
from tests.acceptance.reference_export import Reference
from tests.acceptance.test_acceptance import Case, run_chain

EXPORT = os.environ.get("AI_ANALYST_REAL_EXPORT")
pytestmark = [
    pytest.mark.skipif(
        not EXPORT or not Path(EXPORT).exists(),
        reason="set AI_ANALYST_REAL_EXPORT to a local opportunity_snapshot_v1 parquet export",
    ),
    # WP9: opt-in only, and ingest takes several minutes per the README.
    pytest.mark.slow,
]

DATASET = "real_export_acceptance"
TEST_TENANT = TenantProfile(
    tenant_id="real_export_acceptance",
    source="tests/acceptance/test_real_export.py: test declaration, not a production one",
    concept_columns={
        C.AMOUNT: ("new_amount",),
        C.STAGE: ("Stage",),
        C.OWNER_ID: ("OwnerID",),
        # Test-only: the stage-derived status. Not the authoritative status.
        C.OPPORTUNITY_STATUS: ("status",),
    },
    column_classifications=[
        ColumnDeclaration(
            column="OpportunityOwnerDivision",
            classification=ColumnClassification(
                name="OpportunityOwnerDivision",
                category=ColumnCategory.SNAPSHOT_STATE,
                availability=Availability.AS_OF_FACT,
                disposition=Disposition.DIRECT,
            ),
            source="test declaration, not a production classification",
        )
    ],
)

SEMANTIC = TrustFactorKind.SEMANTIC_PATH
INVESTIGATION = TrustFactorKind.INVESTIGATION_PATH
STATUS = TrustFactorKind.STATUS_NOT_AUTHORITATIVE
GRANT = TrustFactorKind.USAGE_GRANT
# The reconstructed close date is VALID_WITH_WARNINGS on this export (13.1):
# two reconciliation tests disagree under undeclared conventions (CD_in_qtr,
# eoq_close_diff); the movement check could not run; and the edge-case,
# boundary-day, eoq and fiscal-calendar questions are open.
CLOSE_DATE = {
    TrustFactorKind.RECONCILIATION_WARNING,
    TrustFactorKind.RECONSTRUCTION_UNVERIFIED,
    TrustFactorKind.UNRESOLVED_QUESTION,
}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    from ai_analyst.config import Settings
    from ai_analyst.contracts.opportunity_snapshot_v1 import OPPORTUNITY_SNAPSHOT_V1
    from ai_analyst.data.dataset import load_dataset
    from ai_analyst.data.store import DuckDBStore
    from ai_analyst.data.understanding import understand
    from ai_analyst.semantic.calendar import resolve_calendar
    from ai_analyst.semantic.snapshots import SnapshotResolver
    from tests.semantic.conftest import Engine, build_engine

    kept = os.environ.get("AI_ANALYST_REAL_EXPORT_DATA_ROOT")
    root = Path(kept) if kept else tmp_path_factory.mktemp("real_export")
    settings = Settings(data_root=root / "data")
    try:
        dataset = load_dataset(DATASET, settings)
    except Exception:  # not ingested yet under this root
        engine = build_engine(EXPORT, DATASET, root, tenant=TEST_TENANT,
                              column_registry=OPPORTUNITY_SNAPSHOT_V1)
    else:
        store = DuckDBStore(settings)
        scan = store.snapshots_scan(DATASET)
        with store.connect() as conn:
            dates = [r[0] for r in conn.execute(
                f"SELECT DISTINCT as_of FROM {scan} ORDER BY 1").fetchall()]
        engine = Engine(
            dataset_id=DATASET, dataset=dataset,
            bindings=understand(dataset, tenant=TEST_TENANT, settings=settings).bindings,
            settings=settings, store=store, scan=scan, snapshots=SnapshotResolver(dates),
            calendar=resolve_calendar(TEST_TENANT, settings),
        )
    reference = Reference(EXPORT)
    # Engine and reference must agree on the snapshots before anything else.
    distinct = f"SELECT DISTINCT as_of FROM {engine.scan} ORDER BY 1"
    engine_dates = [r[0] for r in engine.query(distinct)]
    assert engine_dates == reference.snapshots()
    return engine, reference


def _window(reference: Reference) -> tuple[date, date, Period]:
    snapshots = reference.snapshots()
    first, last = snapshots[0], snapshots[-1]
    period = Period(kind=PeriodKind.CUSTOM, start=first, end=last,
                    label=f"{first.isoformat()}..{last.isoformat()}")
    return first, last, period


def _plan(period, metrics, snapshot_rule, **kwargs) -> AnalysisPlan:
    kwargs.setdefault("pattern", AnalysisPattern.POINT_IN_TIME)
    return AnalysisPlan(
        question_restatement="real-export acceptance",
        specs=[AnalysisSpec(id="s", metrics=metrics, period=period,
                            snapshot=SnapshotSelection(rule=snapshot_rule), **kwargs)],
    )


def _cases(reference: Reference) -> list[Case]:
    first, last, period = _window(reference)
    open_rule, close_rule = SnapshotRule.PERIOD_OPEN, SnapshotRule.PERIOD_CLOSE
    first_seen = {"creation_basis": CreationBasis.FIRST_SEEN}
    created, pulled = reference.entrants(first, last, first, last)
    won, closed = reference.win_rate(last, first, last)
    return [
        Case("opening pipeline", "real", "What was opening pipeline for the window?",
             _plan(period, ["opening_pipeline"], open_rule),
             "Opening pipeline was {{q1.r0.opening_pipeline}}.",
             {0: {"opening_pipeline": reference.pipeline(first, first, last)}},
             {SEMANTIC, STATUS} | CLOSE_DATE, [first]),
        Case("ending pipeline", "real", "What was ending pipeline for the window?",
             _plan(period, ["ending_pipeline"], close_rule),
             "Ending pipeline was {{q1.r0.ending_pipeline}}.",
             {0: {"ending_pipeline": reference.pipeline(last, first, last)}},
             {SEMANTIC, STATUS} | CLOSE_DATE, [last]),
        Case("created after start", "real", "How much pipeline was created in the window?",
             _plan(period, ["created_pipeline"], close_rule, **first_seen),
             "Created pipeline was {{q1.r0.created_pipeline}}.",
             {0: {"created_pipeline": created}},
             {SEMANTIC, STATUS} | CLOSE_DATE, [first, last]),
        Case("slipped", "real", "How much pipeline slipped out of the window?",
             _plan(period, ["slipped_pipeline"], close_rule, **first_seen),
             "Slipped pipeline was {{q1.r0.slipped_pipeline}}.",
             {0: {"slipped_pipeline": reference.slipped(first, last, first, last)}},
             {SEMANTIC, STATUS} | CLOSE_DATE, [first, last]),
        Case("pulled in", "real", "How much pipeline was pulled into the window?",
             _plan(period, ["pulled_in_pipeline"], close_rule, **first_seen),
             "Pulled-in pipeline was {{q1.r0.pulled_in_pipeline}}.",
             {0: {"pulled_in_pipeline": pulled}},
             {SEMANTIC, STATUS} | CLOSE_DATE, [first, last]),
        Case("win rate", "real", "What was the win rate for the window?",
             _plan(period, ["win_rate"], close_rule, pattern=AnalysisPattern.RATE),
             "{{q1.r0.numerator}} won of {{q1.r0.denominator}} closed.",
             {0: {"numerator": won, "denominator": closed}},
             {SEMANTIC, STATUS} | CLOSE_DATE, [last]),
    ]


def test_real_export_semantic_cases(world):
    engine, reference = world
    for case in _cases(reference):
        run_chain(engine, case)


def test_real_export_cohort(world):
    engine, reference = world
    first, last, period = _window(reference)
    trace = run_chain(engine, Case(
        "cohort", "real", "What happened to the deals open at the start of the window?",
        _plan(period, ["cohort_fate"], SnapshotRule.PERIOD_OPEN,
              pattern=AnalysisPattern.COHORT_TRACE, stance=AnalysisStance.RETROSPECTIVE),
        "{{q1.r0.opportunity_count}} deals are in the first fate group.",
        {}, {SEMANTIC, STATUS, TrustFactorKind.UNRESOLVED_QUESTION}, [first, last],
    ))
    result = trace.result
    got = {result.cell(i, "terminal_state"): (result.cell(i, "opportunity_count"),
                                              result.cell(i, "cohort_amount"))
           for i in range(result.row_count)}
    assert got == reference.cohort_fates(first, last)


def test_real_export_investigation(world):
    engine, reference = world
    first, last, period = _window(reference)
    plan = InvestigationPlan(
        question_restatement="The opening cohort by whether its close date moved.",
        hypotheses=[Hypothesis(statement="close dates move", kind=HypothesisKind.ASSOCIATION)],
        stance=AnalysisStance.RETROSPECTIVE,
        population=Population(period=period,
                              cohort=SnapshotSelection(rule=SnapshotRule.PERIOD_OPEN),
                              window_end=SnapshotSelection(rule=SnapshotRule.PERIOD_CLOSE)),
        variables=[Variable(id="moved", derived=DerivedFeatureRef(
            feature=DerivedFeature.CHANGED, concept=C.EXPECTED_CLOSE_DATE))],
        grouping=[Grouping(variable="moved")],
        operation=Operation(kind=StatisticalOperation.COUNT),
        evidence=EvidenceRequirements(min_group_support=1),
    )
    trace = run_chain(engine, Case(
        "investigation", "real", "How many open deals had their close date move?",
        plan, "{{q1.r0.units}} deals are in the first group.", {},
        {INVESTIGATION, STATUS} | CLOSE_DATE, [first, last],
    ))
    result = trace.result
    got = {result.cell(i, "moved"): result.cell(i, "units") for i in range(result.row_count)}
    assert got == reference.close_date_changed(first, last)


def test_real_export_prospective_temporal_safety(world):
    engine, reference = world
    first, last, period = _window(reference)
    plan = _plan(period, ["opening_pipeline"], SnapshotRule.PERIOD_OPEN,
                 dimensions=["owner"], attribution=Attribution.LATEST,
                 stance=AnalysisStance.PROSPECTIVE)
    outcome = engine.gate(*plan.specs)
    assert not outcome.ok
    (rejection,) = outcome.validation.rejections
    assert rejection.code is RejectionCode.STANCE_VIOLATION
    assert last.isoformat() in rejection.message and first.isoformat() in rejection.message
    with pytest.raises(UnvalidatedPlan):
        compile_plan(engine.scan, plan, outcome)


def test_real_export_usage_grant(world):
    engine, reference = world
    first, last, period = _window(reference)
    trace = run_chain(engine, Case(
        "usage grant", "real", "Opening pipeline by owner division?",
        _plan(period, ["opening_pipeline"], SnapshotRule.PERIOD_OPEN,
              dimensions=["OpportunityOwnerDivision"]),
        "The first division held {{q1.r0.opening_pipeline}}.",
        {}, {SEMANTIC, STATUS, GRANT} | CLOSE_DATE, [first],
    ))
    result = trace.result
    got = {result.cell(i, "OpportunityOwnerDivision"): result.cell(i, "opening_pipeline")
           for i in range(result.row_count)}
    expected = reference.pipeline_by("division", first, first, last)
    assert got == {k: v for k, v in expected.items()}
    assert sum(got.values(), Decimal(0)) == reference.pipeline(first, first, last)
