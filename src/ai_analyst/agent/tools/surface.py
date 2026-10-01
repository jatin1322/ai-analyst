"""The deterministic tool surface (ARCHITECTURE 13.5).

Each function here is what a tool call will run. None is wired to a model yet.
Every one answers under a `ToolContext` that fixes the stance and the knowledge
horizon for the whole turn, so a tool cannot be talked into a wider view than
the plan it serves.

The safety-carrying details:

* **Stance-bounded visibility.** A column the stance forbids is listed with its
  classification and a reason, and its values are never computed.
* **Horizon-bounded profiles.** With a horizon set, every observation a tool
  reports is recomputed over `as_of <= horizon` instead of read from the
  stored profile, which describes every snapshot including later ones. A stage
  distribution over all snapshots already contains how many deals eventually
  closed won; reading it would let the future into the model's reasoning
  without touching a query.
* **Grants are concept-scoped.** A column readable only through a tenant usage
  grant is not an `inspect_values` target: its values are reached as its
  concept inside a plan, never browsed.
* **No narrative content.** Free-text columns are never valued or sampled.
* **Registered evidence.** Distributions and relationships are registered as
  results, so any number a later answer takes from them has a reference.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from datetime import date

from ai_analyst.config import Settings
from ai_analyst.contracts.binding import ColumnPurpose, ConceptBindings
from ai_analyst.contracts.columns import InformationClass
from ai_analyst.contracts.concepts import CONCEPTS, BusinessConcept
from ai_analyst.contracts.dataset import DatasetRegistry
from ai_analyst.contracts.investigation import InvestigationPlan
from ai_analyst.contracts.plan import AnalysisPlan, AnalysisStance
from ai_analyst.contracts.rejection import PlanRejection, PlanValidation, RejectionCode
from ai_analyst.contracts.result import (
    ResultColumn,
    ResultSet,
    TrustAssessment,
    TrustFactor,
    TrustFactorKind,
    ValueKind,
    new_query_id,
)
from ai_analyst.contracts.schema import DataType
from ai_analyst.contracts.tools import (
    ClarificationReason,
    ClarificationRequest,
    ColumnListing,
    ColumnSummary,
    ColumnView,
    ConceptSummary,
    ConceptView,
    DatasetView,
    MetricCatalog,
    MetricEntry,
    PairCount,
    RelationshipEvidence,
    ResultSummary,
    RunOutcome,
    SampleRowsView,
    ToolScope,
    UnavailableMetric,
    ValueCount,
    ValueDistribution,
)
from ai_analyst.data.conform import quote_ident
from ai_analyst.data.store import DuckDBStore
from ai_analyst.semantic.calendar import FiscalCalendarResolution
from ai_analyst.semantic.execute import run_plan
from ai_analyst.semantic.gate import validate_plan
from ai_analyst.semantic.investigation import run_investigation, validate_investigation
from ai_analyst.semantic.metrics import METRICS, metric_availability
from ai_analyst.semantic.resolver import ConceptResolver
from ai_analyst.semantic.snapshots import SnapshotResolver
from ai_analyst.validation.rendering import ResultRegistry

MAX_LISTED_COLUMNS = 200
MAX_TOP_K = 50
MAX_RELATIONSHIP_PAIRS = 50

_EVIDENCE_TRUST = TrustAssessment(
    factors=(
        TrustFactor(
            kind=TrustFactorKind.INVESTIGATION_PATH,
            subject="inspection",
            reason="descriptive evidence from an inspection tool, not a registry metric",
        ),
    )
)


@dataclass
class ToolContext:
    """Everything a tool may consult, and the scope it answers under."""

    dataset_id: str
    registry: DatasetRegistry
    bindings: ConceptBindings
    store: DuckDBStore
    settings: Settings
    snapshots: SnapshotResolver
    calendar: FiscalCalendarResolution
    stance: AnalysisStance = AnalysisStance.PROSPECTIVE
    horizon: date | None = None
    results: ResultRegistry = field(default_factory=ResultRegistry)
    row_count: int = 0

    @property
    def scope(self) -> ToolScope:
        return ToolScope(stance=self.stance, horizon=self.horizon)

    @property
    def scan(self) -> str:
        return self.store.snapshots_scan(self.dataset_id)

    def resolver(self) -> ConceptResolver:
        return ConceptResolver(
            dataset_id=self.dataset_id,
            registry=self.registry,
            bindings=self.bindings,
            stance=self.stance,
            knowledge_cutoff=self.horizon,
        )

    def horizon_sql(self) -> str:
        return f"as_of <= DATE '{self.horizon.isoformat()}'" if self.horizon else "TRUE"

    def generic_rejection(self, name: str) -> PlanRejection | None:
        """Whether inspecting a column is permitted. A tenant generic grant counts."""
        rejection, _ = self.resolver().generic_access(
            name, ColumnPurpose.FEATURE, field_name="tools"
        )
        return rejection


def _is_text(ctx: ToolContext, name: str) -> bool:
    return ctx.registry.get(name).information_class is InformationClass.TEXT


# ------------------------------------------------------------------ inspection


def inspect_dataset(
    ctx: ToolContext, class_filter: str | None = None, name_pattern: str | None = None
) -> DatasetView:
    """The dataset's shape, concepts, and a bounded, filterable column listing."""
    resolver = ctx.resolver()
    concepts = []
    for binding in ctx.bindings.bindings:
        outcome = resolver.try_resolve(binding.concept, load_bearing=False)
        rejected = isinstance(outcome, PlanRejection)
        concepts.append(
            ConceptSummary(
                concept=binding.concept,
                status=binding.status,
                columns=binding.columns,
                usable=not rejected,
                reason=outcome.message if rejected else "",
            )
        )
    listings = []
    for column in ctx.registry.columns:
        info = column.information_class.value
        if class_filter and info != class_filter:
            continue
        if name_pattern and not fnmatch.fnmatch(
            column.name.lower(), safe_name_pattern(name_pattern).lower()
        ):
            continue
        rejection = ctx.generic_rejection(column.name)
        listings.append(
            ColumnListing(
                name=column.name,
                information_class=info,
                usable=rejection is None,
                reason=rejection.code.value if rejection else "",
            )
        )
    dates = ctx.snapshots.dates
    visible = [d for d in dates if ctx.horizon is None or d <= ctx.horizon]
    return DatasetView(
        scope=ctx.scope,
        dataset_id=ctx.dataset_id,
        row_count=ctx.row_count,
        snapshot_count=len(visible),
        first_snapshot=visible[0] if visible else None,
        last_snapshot=visible[-1] if visible else None,
        fiscal_year_start_month=ctx.calendar.start_month,
        fiscal_calendar_resolved=ctx.calendar.is_resolved,
        concepts=tuple(concepts),
        columns=tuple(listings[:MAX_LISTED_COLUMNS]),
        columns_truncated=len(listings) > MAX_LISTED_COLUMNS,
        quarantined_count=len(ctx.registry.quarantined()),
    )


def inspect_concept(ctx: ToolContext, concept: BusinessConcept) -> ConceptView:
    """What a concept means, how it is bound here, and whether it is usable now."""
    definition = CONCEPTS[concept]
    binding = ctx.bindings.by_concept().get(concept)
    outcome = ctx.resolver().try_resolve(concept, load_bearing=False)
    rejected = isinstance(outcome, PlanRejection)
    verdict = next(
        (v.verdict.value for v in ctx.bindings.reconstruction_verdicts if v.concept is concept),
        None,
    )
    return ConceptView(
        scope=ctx.scope,
        concept=concept,
        display_name=definition.display_name,
        definition=definition.definition,
        semantic_type=definition.semantic_type.value,
        retrospective=definition.retrospective,
        status=binding.status if binding else ctx.bindings.status_of(concept),
        columns=binding.columns if binding else (),
        evidence_kinds=tuple(dict.fromkeys(e.kind.value for e in binding.evidence))
        if binding else (),
        caveats=binding.caveats if binding else (),
        alternatives=binding.alternatives if binding else (),
        note=binding.note if binding else "",
        usable=not rejected,
        rejection=outcome.code if rejected else None,
        reconstruction_verdict=verdict,
        grants=tuple(g.disclosure for g in ctx.bindings.grants if g.concept is concept),
    )


def _column_summary(ctx: ToolContext, name: str) -> ColumnSummary:
    column = quote_ident(name)
    with ctx.store.connect() as conn:
        rows, nulls, distinct, low, high = conn.execute(
            f"SELECT COUNT(*), COUNT(*) - COUNT({column}), COUNT(DISTINCT {column}), "
            f"CAST(MIN({column}) AS VARCHAR), CAST(MAX({column}) AS VARCHAR) "
            f"FROM {ctx.scan} WHERE {ctx.horizon_sql()}"
        ).fetchone()
    return ColumnSummary(
        rows_considered=int(rows),
        null_count=int(nulls),
        distinct_count=int(distinct),
        min=low,
        max=high,
    )


def _withheld(ctx: ToolContext, name: str) -> tuple[str, PlanRejection | None]:
    """Why a column's values may not be shown under this scope, or ''."""
    if _is_text(ctx, name):
        return "free-text column: catalogued only, content never read", None
    rejection = ctx.generic_rejection(name)
    if rejection is not None:
        return rejection.message, rejection
    return "", None


def inspect_column(ctx: ToolContext, name: str) -> ColumnView:
    """One column's classification, usability and, when permitted, its shape."""
    column = ctx.registry.get(name)
    classification = column.classification
    reason, rejection = _withheld(ctx, name)
    grants = tuple(
        f"readable as {g.concept.value} for {', '.join(sorted(p.value for p in g.purposes))}"
        for g in ctx.bindings.grants
        if g.column == name
    )
    return ColumnView(
        scope=ctx.scope,
        name=name,
        origin=column.origin.value,
        dtype=column.dtype.value,
        information_class=column.information_class.value,
        availability=classification.availability.value,
        disposition=classification.disposition.value,
        quarantine=column.quarantine.code.value if column.quarantine else None,
        usable=rejection is None and not reason,
        rejection=rejection.code if rejection else None,
        grants=grants,
        summary=None if reason else _column_summary(ctx, name),
        values_withheld_reason=reason,
    )


def _register(ctx: ToolContext, columns: list[tuple[str, DataType, ValueKind]], rows) -> str:
    result = ResultSet(
        query_id=new_query_id(),
        columns=[ResultColumn(name=n, dtype=d, kind=k) for n, d, k in columns],
        rows=[list(r) for r in rows],
        trust=_EVIDENCE_TRUST,
        dataset_id=ctx.dataset_id,
    )
    return ctx.results.register(result)


def inspect_values(ctx: ToolContext, name: str, top_k: int = 20) -> ValueDistribution:
    """A value distribution, capped, horizon-bounded, and registered as evidence."""
    top_k = max(1, min(top_k, MAX_TOP_K))
    reason, _ = _withheld(ctx, name)
    if reason:
        return ValueDistribution(scope=ctx.scope, column=name, values_withheld_reason=reason)
    column = quote_ident(name)
    with ctx.store.connect() as conn:
        rows, nulls, distinct = conn.execute(
            f"SELECT COUNT(*), COUNT(*) - COUNT({column}), COUNT(DISTINCT {column}) "
            f"FROM {ctx.scan} WHERE {ctx.horizon_sql()}"
        ).fetchone()
        counts = conn.execute(
            f"SELECT CAST({column} AS VARCHAR) AS value, COUNT(*) AS n FROM {ctx.scan} "
            f"WHERE {ctx.horizon_sql()} AND {column} IS NOT NULL "
            f"GROUP BY 1 ORDER BY n DESC, value LIMIT {top_k + 1}"
        ).fetchall()
    shown = counts[:top_k]
    reference = _register(
        ctx,
        [("value", DataType.VARCHAR, ValueKind.TEXT), ("count", DataType.BIGINT, ValueKind.COUNT)],
        shown,
    )
    return ValueDistribution(
        scope=ctx.scope,
        column=name,
        rows_considered=int(rows),
        null_count=int(nulls),
        distinct_count=int(distinct),
        values=tuple(ValueCount(value=v, count=int(n)) for v, n in shown),
        truncated=len(counts) > top_k,
        reference=reference,
    )


def inspect_relationship(ctx: ToolContext, a: str, b: str) -> RelationshipEvidence:
    """A contingency of two permitted columns, registered as tier-B evidence."""
    for name in (a, b):
        reason, _ = _withheld(ctx, name)
        if reason:
            raise PermissionError(f"{name!r} cannot be inspected under this scope: {reason}")
    ca, cb = quote_ident(a), quote_ident(b)
    with ctx.store.connect() as conn:
        pairs = conn.execute(
            f"SELECT CAST({ca} AS VARCHAR), CAST({cb} AS VARCHAR), COUNT(*) AS n "
            f"FROM {ctx.scan} WHERE {ctx.horizon_sql()} "
            f"GROUP BY 1, 2 ORDER BY n DESC, 1, 2 LIMIT {MAX_RELATIONSHIP_PAIRS + 1}"
        ).fetchall()
        (support,) = conn.execute(
            f"SELECT COUNT(*) FROM {ctx.scan} WHERE {ctx.horizon_sql()}"
        ).fetchone()
    shown = pairs[:MAX_RELATIONSHIP_PAIRS]
    reference = _register(
        ctx,
        [
            (a, DataType.VARCHAR, ValueKind.TEXT),
            (b, DataType.VARCHAR, ValueKind.TEXT),
            ("count", DataType.BIGINT, ValueKind.COUNT),
        ],
        shown,
    )
    return RelationshipEvidence(
        scope=ctx.scope,
        a=a,
        b=b,
        support=int(support),
        pairs=tuple(PairCount(a=str(x), b=str(y), count=int(n)) for x, y, n in shown),
        truncated=len(pairs) > MAX_RELATIONSHIP_PAIRS,
        reference=reference,
    )


def inspect_sample_rows(
    ctx: ToolContext, n: int = 5, columns: list[str] | None = None
) -> SampleRowsView:
    """A bounded sample: permitted, non-text columns only, within the horizon."""
    limit = max(1, min(n, ctx.settings.max_sample_rows))
    requested = columns or ctx.registry.names
    excluded: dict[str, str] = {}
    kept: list[str] = []
    for name in requested:
        if not ctx.registry.has(name):
            excluded[name] = "no such column"
            continue
        reason, _ = _withheld(ctx, name)
        if reason:
            excluded[name] = reason
        else:
            kept.append(name)
    if not kept:
        return SampleRowsView(scope=ctx.scope, columns=(), rows=(), excluded=excluded, limit=limit)
    select = ", ".join(f"CAST({quote_ident(c)} AS VARCHAR)" for c in kept)
    with ctx.store.connect() as conn:
        rows = conn.execute(
            f"SELECT {select} FROM {ctx.scan} WHERE {ctx.horizon_sql()} "
            f"ORDER BY as_of, {quote_ident('opp_id')} LIMIT {limit}"
        ).fetchall()
    return SampleRowsView(
        scope=ctx.scope,
        columns=tuple(kept),
        rows=tuple(tuple(r) for r in rows),
        excluded=excluded,
        limit=limit,
    )


def list_available_metrics(ctx: ToolContext) -> MetricCatalog:
    """The catalog under the scope's stance: what can be planned, and what cannot."""
    resolver = ctx.resolver()
    available, unavailable = [], []
    for definition in METRICS.values():
        availability = metric_availability(definition, resolver, ctx.stance)
        if availability.available:
            available.append(
                MetricEntry(
                    name=definition.name,
                    display_name=definition.display_name,
                    definition=definition.definition,
                    required_concepts=tuple(c.value for c in definition.required_concepts),
                    default_snapshot_rule=definition.default_snapshot_rule.value,
                    patterns=tuple(p.value for p in definition.patterns),
                )
            )
        else:
            unavailable.append(
                UnavailableMetric(
                    name=definition.name,
                    missing_concepts=tuple(c.value for c in availability.missing_concepts),
                    reason=availability.reason,
                )
            )
    return MetricCatalog(
        scope=ctx.scope, available=tuple(available), unavailable=tuple(unavailable)
    )


# ------------------------------------------------------------------- terminal


def _summaries(ctx: ToolContext, results: list[ResultSet]) -> tuple[ResultSummary, ...]:
    out = []
    for result in results:
        reference = ctx.results.register(result)
        out.append(
            ResultSummary(
                reference=reference,
                query_id=result.query_id,
                columns=tuple(result.column_names),
                row_count=result.row_count,
                trust_tier=result.trust_tier,
                trust_reasons=tuple(result.trust_reasons),
            )
        )
    return tuple(out)


def _scope_rejection(
    ctx: ToolContext, stance: AnalysisStance, spec_id: str
) -> PlanRejection | None:
    """A plan may narrow the session's scope, never widen it."""
    if ctx.stance is AnalysisStance.PROSPECTIVE and stance is AnalysisStance.RETROSPECTIVE:
        return PlanRejection(
            code=RejectionCode.STANCE_VIOLATION,
            message=(
                "the session is prospective; a plan cannot switch to hindsight on its own. "
                "Changing stance is an explicit session decision."
            ),
            spec_id=spec_id,
            field="stance",
        )
    return None


def _tightened(cutoff: date | None, horizon: date | None) -> date | None:
    """The tighter of a plan's cutoff and the session's horizon. Tightening is always safe."""
    if horizon is None:
        return cutoff
    return horizon if cutoff is None else min(cutoff, horizon)


def _refused(rejections: list[PlanRejection]) -> RunOutcome:
    return RunOutcome(validation=PlanValidation(plan_ok=False, rejections=rejections))


@dataclass(frozen=True)
class ScopedValidation:
    """A plan after the scope check and the gate, ready to execute or be refused.

    The one validation path shared by the run tools and the planner's submit
    path, so a planner can never validate a plan differently from how it will
    be executed.
    """

    plan: AnalysisPlan | InvestigationPlan
    validation: PlanValidation
    outcome: object | None = None

    @property
    def ok(self) -> bool:
        return self.outcome is not None and self.validation.plan_ok


def validate_analysis_plan(ctx: ToolContext, plan: AnalysisPlan) -> ScopedValidation:
    """Scope, then gate. The scope is not negotiable.

    A spec asking for a wider stance than the session's is refused before the
    gate sees it, and every spec's knowledge cutoff is tightened to the
    session's horizon.
    """
    refusals = [
        r for s in plan.specs if (r := _scope_rejection(ctx, s.stance, s.id)) is not None
    ]
    if refusals:
        return ScopedValidation(plan, PlanValidation(plan_ok=False, rejections=refusals))
    plan = plan.model_copy(
        update={
            "specs": [
                s.model_copy(
                    update={"knowledge_cutoff": _tightened(s.knowledge_cutoff, ctx.horizon)}
                )
                for s in plan.specs
            ]
        }
    )
    outcome = validate_plan(
        plan,
        dataset_id=ctx.dataset_id,
        registry=ctx.registry,
        bindings=ctx.bindings,
        snapshots=ctx.snapshots,
        calendar=ctx.calendar,
    )
    return ScopedValidation(plan, outcome.validation, outcome if outcome.ok else None)


def validate_investigation_plan(ctx: ToolContext, plan: InvestigationPlan) -> ScopedValidation:
    """Scope, then the investigation gate."""
    refusal = _scope_rejection(ctx, plan.stance, plan.plan_id)
    if refusal is not None:
        return ScopedValidation(plan, PlanValidation(plan_ok=False, rejections=[refusal]))
    plan = plan.model_copy(
        update={"knowledge_cutoff": _tightened(plan.knowledge_cutoff, ctx.horizon)}
    )
    outcome = validate_investigation(
        plan,
        dataset_id=ctx.dataset_id,
        registry=ctx.registry,
        bindings=ctx.bindings,
        snapshots=ctx.snapshots,
        calendar=ctx.calendar,
    )
    return ScopedValidation(plan, outcome.validation, outcome if outcome.ok else None)


def run_analysis_plan(ctx: ToolContext, plan: AnalysisPlan) -> RunOutcome:
    """Terminal: validate, compile and execute a semantic plan under the scope."""
    checked = validate_analysis_plan(ctx, plan)
    if not checked.ok:
        return RunOutcome(validation=checked.validation)
    with ctx.store.connect() as conn:
        results = run_plan(
            conn, ctx.scan, checked.plan, checked.outcome, dataset_id=ctx.dataset_id,
            calendar=ctx.calendar, settings=ctx.settings,
        )
    return RunOutcome(validation=checked.validation, results=_summaries(ctx, results))


def run_investigation_plan(ctx: ToolContext, plan: InvestigationPlan) -> RunOutcome:
    """Terminal: validate, compile and execute an investigation. At most tier B."""
    checked = validate_investigation_plan(ctx, plan)
    if not checked.ok:
        return RunOutcome(validation=checked.validation)
    with ctx.store.connect() as conn:
        result = run_investigation(
            conn, ctx.scan, checked.outcome, dataset_id=ctx.dataset_id,
            calendar=ctx.calendar, settings=ctx.settings,
        )
    return RunOutcome(validation=checked.validation, results=_summaries(ctx, [result]))


def available_alternatives(ctx: ToolContext) -> list[str]:
    """Bound dimension concepts usable now: the only substitutes an answer may list."""
    resolver = ctx.resolver()
    out = []
    for binding in ctx.bindings.bindings:
        definition = CONCEPTS[binding.concept]
        if definition.role.value != "dimension":
            continue
        if resolver.has(binding.concept, load_bearing=False, purpose=ColumnPurpose.DIMENSION):
            out.append(binding.concept.value)
    return out


def request_clarification(ctx: ToolContext, request: ClarificationRequest) -> ClarificationRequest:
    """Terminal: validate a clarification and fill its alternatives deterministically.

    For an unavailable concept the alternatives are replaced with the dataset's
    own usable dimension concepts, so a model can neither invent a substitute
    nor omit the real ones. The concept asked about is never among them.
    """
    if request.reason is not ClarificationReason.CONCEPT_UNAVAILABLE:
        return request
    alternatives = [a for a in available_alternatives(ctx) if a != request.concept.value]
    return request.model_copy(update={"available_alternatives": alternatives})


_SAFE_PATTERN = re.compile(r"^[\w*?\[\]-]{1,64}$")


def safe_name_pattern(pattern: str) -> str:
    """Name patterns are globs over column names; anything else is refused."""
    if not _SAFE_PATTERN.match(pattern):
        raise ValueError(f"{pattern!r} is not a column-name pattern")
    return pattern


class _ScopedEngine:
    """The gate and executor a materiality probe needs, under the tool scope."""

    def __init__(self, ctx: ToolContext) -> None:
        self.ctx = ctx
        self.store = ctx.store
        self.scan = ctx.scan
        self.dataset_id = ctx.dataset_id
        self.calendar = ctx.calendar
        self.settings = ctx.settings

    def gate(self, *specs):
        plan = AnalysisPlan(question_restatement="materiality probe", specs=list(specs))
        return validate_plan(
            plan,
            dataset_id=self.ctx.dataset_id,
            registry=self.ctx.registry,
            bindings=self.ctx.bindings,
            snapshots=self.ctx.snapshots,
            calendar=self.ctx.calendar,
        )


def probe_materiality(ctx: ToolContext, probe, base: AnalysisPlan):
    """Deterministic: would the choice between readings change the answer?

    Runs under the session scope. A base plan wider than the scope is refused,
    and every cutoff is tightened to the session horizon before any reading runs.
    """
    from ai_analyst.semantic.materiality import (
        AmbiguityKind,
        Materiality,
        MaterialityResult,
        run_materiality_probe,
    )

    refusals = [r for s in base.specs if (r := _scope_rejection(ctx, s.stance, s.id))]
    if refusals:
        return MaterialityResult(
            verdict=Materiality.INCONCLUSIVE,
            ambiguity=AmbiguityKind(probe.ambiguity),
            reason=refusals[0].message,
        )
    base = base.model_copy(
        update={
            "specs": [
                s.model_copy(
                    update={"knowledge_cutoff": _tightened(s.knowledge_cutoff, ctx.horizon)}
                )
                for s in base.specs
            ]
        }
    )
    return run_materiality_probe(probe, base, engine=_ScopedEngine(ctx), settings=ctx.settings)
