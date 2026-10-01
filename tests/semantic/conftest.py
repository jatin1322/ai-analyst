"""Fixtures for the semantic engine tests.

The tenant profile is load-bearing and deliberately visible here rather than
buried in a helper. Without a declaration every concept on the tiny fixture
resolves to `inferred` (a header match never confirms, 12.2) and every metric
built on one is withheld. `test_metrics.py` asserts that undeclared half; these
fixtures are the declared half.

The declaration names the fixture's **source headers**, not the conformed
names, because that is what a real tenant writes down. Resolving them through
conformance is part of what is being tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pytest

from ai_analyst.config import Settings
from ai_analyst.contracts.concepts import BusinessConcept as Concept
from ai_analyst.contracts.dataset import DatasetRegistry
from ai_analyst.contracts.plan import AnalysisPlan, AnalysisSpec
from ai_analyst.contracts.tenant import TenantProfile
from ai_analyst.data.dataset import Dataset, register_dataset
from ai_analyst.data.store import DuckDBStore
from ai_analyst.data.understanding import understand
from ai_analyst.semantic.calendar import FiscalCalendarResolution, resolve_calendar
from ai_analyst.semantic.gate import GateOutcome, validate_plan
from ai_analyst.semantic.snapshots import SnapshotResolver

FIXTURES = Path(__file__).parent.parent / "fixtures" / "tiny"
TINY_CSV = FIXTURES / "snapshots.csv"
MOVES_CSV = FIXTURES / "bridge_moves.csv"
CUSTOM_CSV = FIXTURES / "custom_column.csv"

# What the tiny fixture's owner has declared about its own columns. Source
# headers, as a tenant would write them.
TINY_TENANT = TenantProfile(
    tenant_id="tiny",
    source="tests/fixtures/tiny/README.md",
    concept_columns={
        Concept.AMOUNT: ("deal_amount",),
        Concept.EXPECTED_CLOSE_DATE: ("expected_close_date",),
        Concept.CREATED_DATE: ("created",),
        Concept.STAGE: ("sales_stage",),
        Concept.OPPORTUNITY_STATUS: ("status",),
        Concept.FORECAST_CATEGORY: ("forecast_cat",),
        Concept.CUSTOMER_SEGMENT: ("customer_segment",),
        Concept.OWNER_ID: ("owner",),
    },
    fiscal_year_start_month=1,
)


@dataclass
class Engine:
    """One dataset with everything the semantic engine needs to run on it."""

    dataset_id: str
    dataset: Dataset
    bindings: object
    settings: Settings
    store: DuckDBStore
    scan: str
    snapshots: SnapshotResolver
    calendar: FiscalCalendarResolution

    @property
    def registry(self) -> DatasetRegistry:
        return self.dataset.registry

    def gate(self, *specs: AnalysisSpec, question: str = "test") -> GateOutcome:
        plan = AnalysisPlan(question_restatement=question, specs=list(specs))
        return validate_plan(
            plan,
            dataset_id=self.dataset_id,
            registry=self.registry,
            bindings=self.bindings,
            snapshots=self.snapshots,
            calendar=self.calendar,
        )

    def run(self, *specs: AnalysisSpec, question: str = "test"):
        """Validate, compile, and execute. Returns the results."""
        from ai_analyst.semantic.execute import run_plan

        plan = AnalysisPlan(question_restatement=question, specs=list(specs))
        outcome = self.gate(*specs, question=question)
        assert outcome.ok, [r.message for r in outcome.validation.rejections]
        with self.store.connect() as conn:
            return run_plan(
                conn,
                self.scan,
                plan,
                outcome,
                dataset_id=self.dataset_id,
                calendar=self.calendar,
                settings=self.settings,
            )

    def one(self, spec: AnalysisSpec):
        """Run a single spec and return its result."""
        return self.run(spec)[0]

    def query(self, sql: str) -> list[tuple]:
        with self.store.connect() as conn:
            return conn.execute(sql).fetchall()


def build_engine(
    csv: Path,
    dataset_id: str,
    tmp_path: Path,
    tenant: TenantProfile | None = TINY_TENANT,
    column_registry=None,
) -> Engine:
    settings = Settings(data_root=tmp_path / "data")
    dataset = register_dataset(
        csv,
        dataset_id,
        settings=settings,
        tenant=tenant,
        column_registry=column_registry,
    )
    understanding = understand(dataset, tenant=tenant, settings=settings)
    store = DuckDBStore(settings)
    scan = store.snapshots_scan(dataset_id)
    with store.connect() as conn:
        dates = [
            r[0] for r in conn.execute(f"SELECT DISTINCT as_of FROM {scan} ORDER BY 1").fetchall()
        ]
    return Engine(
        dataset_id=dataset_id,
        dataset=dataset,
        bindings=understanding.bindings,
        settings=settings,
        store=store,
        scan=scan,
        snapshots=SnapshotResolver(dates),
        calendar=resolve_calendar(tenant, settings),
    )


# ---------------------------------------------------------------------------
# Session-scoped engines (WP9: `build_engine` re-ingests and re-runs the
# understanding layer, which is the slowest thing most tests here do).
#
# Session-scoped: treat as read-only. A test that rebinds, re-registers,
# deletes a ledger, or otherwise mutates an engine must build its own with
# `build_engine(..., tmp_path)`, as `test_grant_ledger.py` and
# `test_column_classifications.py` do (their local `engine`/`money`
# fixtures stay function-scoped for exactly this reason). `understand()`
# itself writes nothing to disk, so re-deriving bindings against a shared
# engine's dataset (as `_context` in `test_tools_and_context.py` does) is
# also read-only. Each engine gets its own `tmp_path_factory` directory so
# it never shares a data_root with another dataset id. Verified at runtime
# with a before/after fingerprint of each engine's on-disk files plus its
# settings, registry, bindings, snapshots, dataset, and calendar, run
# serially and under `-n auto`: no test in this package changed any of it.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def tiny(tmp_path_factory) -> Engine:
    """The tiny fixture with its tenant declaration: every concept confirmed."""
    return build_engine(TINY_CSV, "tiny", tmp_path_factory.mktemp("tiny"))


@pytest.fixture(scope="session")
def undeclared(tmp_path_factory) -> Engine:
    """The same data with no tenant declaration: every concept inferred."""
    return build_engine(
        TINY_CSV, "tiny_undeclared", tmp_path_factory.mktemp("undeclared"), tenant=None
    )


@pytest.fixture(scope="session")
def moves(tmp_path_factory) -> Engine:
    """The bridge fixture, built so all nine identity terms are non-zero."""
    return build_engine(MOVES_CSV, "moves", tmp_path_factory.mktemp("moves"))


@pytest.fixture(scope="session")
def production(tmp_path_factory, production_csv: Path) -> Engine:
    """The production-shaped synthetic export, which has terminal columns.

    The tiny fixture has no future-contaminated column at all, so it cannot
    exercise the stance gate. This one does: `terminal_fate` is classified
    `future_contaminated` and is not quarantined, which is exactly the case the
    prospective stance must refuse and the retrospective stance must allow.
    """
    from ai_analyst.contracts.opportunity_snapshot_v1 import OPPORTUNITY_SNAPSHOT_V1

    tenant = TenantProfile(
        tenant_id="production",
        source="tests fixture",
        concept_columns={
            Concept.AMOUNT: ("amount",),
            Concept.STAGE: ("stage",),
            Concept.TERMINAL_OUTCOME: ("terminal_fate",),
        },
        fiscal_year_start_month=1,
    )
    return build_engine(
        production_csv,
        "production_semantic",
        tmp_path_factory.mktemp("production_semantic"),
        tenant=tenant,
        column_registry=OPPORTUNITY_SNAPSHOT_V1,
    )


# Snapshot dates of the tiny fixture, named so tests read as arguments rather
# than as magic dates.
Q1_OPEN = date(2025, 1, 1)
Q1_CLOSE = date(2025, 3, 31)
Q2_OPEN = date(2025, 4, 1)
Q2_CLOSE = date(2025, 6, 30)


@pytest.fixture(scope="session")
def custom(tmp_path_factory) -> Engine:
    """The tiny data plus `enterprise_amount`, a genuinely discovered column.

    No canonical registry places it, so it is carried through unmapped and
    fails closed as quarantined and unclassified. This is the fixture for what
    a tenant-specific measure column costs (12.10).
    """
    return build_engine(CUSTOM_CSV, "custom", tmp_path_factory.mktemp("custom"))


@pytest.fixture(scope="session")
def custom_declared(tmp_path_factory) -> Engine:
    """The same data with the tenant declaring that column as its amount."""
    declared = TenantProfile(
        tenant_id="custom",
        source="tests fixture",
        concept_columns={
            **TINY_TENANT.concept_columns,
            Concept.AMOUNT: ("enterprise_amount",),
        },
        fiscal_year_start_month=1,
    )
    return build_engine(
        CUSTOM_CSV, "custom_declared", tmp_path_factory.mktemp("custom_declared"), tenant=declared
    )


def tool_context(engine: Engine, stance=None, horizon=None):
    """A deterministic tool context over an engine's dataset."""
    from ai_analyst.agent.tools.surface import ToolContext
    from ai_analyst.contracts.plan import AnalysisStance

    return ToolContext(
        dataset_id=engine.dataset_id,
        registry=engine.registry,
        bindings=engine.bindings,
        store=engine.store,
        settings=engine.settings,
        snapshots=engine.snapshots,
        calendar=engine.calendar,
        stance=stance or AnalysisStance.PROSPECTIVE,
        horizon=horizon,
        row_count=engine.dataset.profile.row_count,
    )
