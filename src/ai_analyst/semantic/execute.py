"""Execution and post-execution checks (ARCHITECTURE 7.4, 8.3).

Runs a compiled query and materializes a `ResultSet` addressable as
(query_id, row, column), which is the shape the provenance scanner will need
(8.4). The scanner itself is a later milestone; what lands here is the metadata
it will depend on.

Two things happen after execution and before the result is handed back:

* **Sanity checks.** The bridge identity must close and a rate must be
  arithmetically possible. A failure raises rather than annotating: a result
  that fails its own invariant is a bug, and returning it with a warning
  attached is how a wrong number reaches a reader (8.3).
* **Trust tier assignment.** The tier is *computed* from the inputs and is the
  weakest of them (12.7). Nothing chooses it, and there is no argument to
  override it.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import duckdb

from ai_analyst.config import Settings, get_settings
from ai_analyst.contracts.plan import AnalysisPlan
from ai_analyst.contracts.result import (
    ResultColumn,
    ResultSet,
    TrustAssessment,
    TrustTier,
    ValueKind,
    new_query_id,
)
from ai_analyst.contracts.schema import DataType
from ai_analyst.semantic.bridge import (
    BRIDGE_COMPONENTS,
    COMPONENT_SIGNS,
    ENDING,
    OPENING,
    BridgeComponent,
    BridgeImbalance,
    BridgeResult,
)
from ai_analyst.semantic.calendar import FiscalCalendarResolution
from ai_analyst.semantic.compiler import CompiledQuery, compile_plan
from ai_analyst.semantic.gate import GateOutcome
from ai_analyst.semantic.rate import RateRow, check_rate
from ai_analyst.semantic.trust import AnalysisPath, assess

# DuckDB type names mapped onto the canonical storage types a result reports.
_DUCKDB_TYPES: dict[str, DataType] = {
    "DATE": DataType.DATE,
    "VARCHAR": DataType.VARCHAR,
    "BOOLEAN": DataType.BOOLEAN,
    "INTEGER": DataType.INTEGER,
    "BIGINT": DataType.BIGINT,
    "HUGEINT": DataType.BIGINT,
    "DOUBLE": DataType.DOUBLE,
    "FLOAT": DataType.DOUBLE,
}


def _result_type(duckdb_type: object) -> DataType:
    # DuckDB hands back a type object, not a string.
    name = str(duckdb_type).upper()
    if name.startswith("DECIMAL"):
        return DataType.DECIMAL
    return _DUCKDB_TYPES.get(name, DataType.VARCHAR)


def execute(
    conn: duckdb.DuckDBPyConnection,
    query: CompiledQuery,
    *,
    dataset_id: str,
    trust: TrustAssessment,
    resolved_snapshots=(),
    settings: Settings | None = None,
) -> ResultSet:
    """Run one compiled query and materialize its result.

    `trust` is a computed assessment from `semantic.trust.assess`, never a tier:
    the result's tier is read off it, and a sanity failure is recorded on it by
    raising rather than by downgrading a number that should not be emitted.
    """
    settings = settings or get_settings()
    cursor = conn.execute(query.sql)
    columns = [
        ResultColumn(
            name=name,
            dtype=_result_type(dtype),
            kind=query.column_kinds.get(name) or _default_kind(_result_type(dtype)),
        )
        for name, dtype in zip(
            [d[0] for d in cursor.description],
            [d[1] for d in cursor.description],
            strict=True,
        )
    ]
    limit = settings.max_result_rows
    rows = cursor.fetchmany(limit + 1)
    truncated = len(rows) > limit
    rows = rows[:limit]

    result = ResultSet(
        query_id=new_query_id(),
        columns=columns,
        rows=[list(r) for r in rows],
        trust=trust,
        resolved_snapshots=list(resolved_snapshots),
        compiled_sql=query.sql,
        spec_id=query.spec_id,
        truncated=truncated,
        warnings=list(query.warnings),
        dataset_id=dataset_id,
        assumptions=list(query.assumptions),
        compilation=query.metadata,
    )

    if query.is_bridge:
        check_bridge_result(bridge_result(result, query, dataset_id))
    if query.rate is not None:
        check_rate(rate_rows(result), query.rate)
    return result


def _default_kind(dtype: DataType) -> ValueKind:
    """A fallback rendering kind. Never MONEY: money is semantic, not a type."""
    return {
        DataType.DATE: ValueKind.DATE,
        DataType.BOOLEAN: ValueKind.BOOLEAN,
        DataType.VARCHAR: ValueKind.TEXT,
        DataType.BIGINT: ValueKind.COUNT,
        DataType.INTEGER: ValueKind.COUNT,
    }.get(dtype, ValueKind.QUANTITY)


class AbstentionRequired(RuntimeError):
    """The computed trust is C, so nothing may execute (ARCHITECTURE 12.7)."""

    def __init__(self, trust: TrustAssessment) -> None:
        super().__init__("; ".join(trust.reasons))
        self.trust = trust


def rate_rows(result: ResultSet) -> list[RateRow]:
    """Read a rate result back as typed rows for checking."""
    names = result.column_names
    group_columns = [n for n in names if n not in ("numerator", "denominator", "ratio")]
    out: list[RateRow] = []
    for index in range(result.row_count):
        ratio = result.cell(index, "ratio")
        out.append(
            RateRow(
                group=tuple(str(result.cell(index, g)) for g in group_columns),
                numerator=Decimal(str(result.cell(index, "numerator"))),
                denominator=Decimal(str(result.cell(index, "denominator"))),
                ratio=None if ratio is None else Decimal(str(ratio)),
            )
        )
    return out


def bridge_result(
    result: ResultSet,
    query: CompiledQuery,
    dataset_id: str,
    *,
    period_label: str = "",
    period_start: date | None = None,
    period_end: date | None = None,
    opening_as_of: date | None = None,
    closing_as_of: date | None = None,
) -> BridgeResult:
    """Read a bridge result back as the typed decomposition."""
    from ai_analyst.contracts.concepts import BusinessConcept

    amounts = {
        str(result.cell(i, "component")): (
            Decimal(str(result.cell(i, "amount"))),
            int(result.cell(i, "opportunity_count")),
        )
        for i in range(result.row_count)
    }
    components = tuple(
        BridgeComponent(
            name=name,
            amount=amounts.get(name, (Decimal("0"), 0))[0],
            opportunity_count=amounts.get(name, (Decimal("0"), 0))[1],
            sign=COMPONENT_SIGNS.get(name, 0),
        )
        for name in BRIDGE_COMPONENTS
    )
    snapshots = [s.resolved_as_of for s in result.resolved_snapshots]
    return BridgeResult(
        dataset_id=dataset_id,
        period_label=period_label or query.spec_id,
        period_start=period_start or (snapshots[0] if snapshots else date.min),
        period_end=period_end or (snapshots[-1] if snapshots else date.max),
        opening_as_of=opening_as_of or (snapshots[0] if snapshots else date.min),
        closing_as_of=closing_as_of or (snapshots[-1] if snapshots else date.max),
        components=components,
        measure_concept=BusinessConcept.AMOUNT,
        assumptions=query.assumptions,
    )


def check_bridge_result(result: BridgeResult) -> None:
    """The bridge invariant, run on every bridge result (5.2, 8.3)."""
    if not result.balances:
        raise BridgeImbalance(
            f"bridge does not balance: {result.amount(OPENING)} opening + "
            f"{result.movement} movement = {result.implied_ending}, but ending "
            f"measures {result.amount(ENDING)} (residual {result.residual})"
        )


def run_plan(
    conn: duckdb.DuckDBPyConnection,
    scan: str,
    plan: AnalysisPlan,
    outcome: GateOutcome,
    *,
    dataset_id: str,
    calendar: FiscalCalendarResolution | None = None,
    settings: Settings | None = None,
) -> list[ResultSet]:
    """Compile and execute a validated plan, returning one result per spec.

    Trust is assessed before execution. A tier-C assessment executes nothing.
    """
    results: list[ResultSet] = []
    for query in compile_plan(scan, plan, outcome):
        validated = outcome.specs[query.spec_id]
        trust = assess(
            path=AnalysisPath.SEMANTIC,
            resolver=validated.resolver,
            snapshot=validated.snapshot,
            max_drift_days=validated.spec.snapshot.max_drift_days,
            calendar=calendar,
            period_kind=validated.spec.period.kind,
            unresolved_ambiguities=tuple(plan.unresolved_ambiguities),
        )
        if trust.tier is TrustTier.C:
            raise AbstentionRequired(trust)
        results.append(
            execute(
                conn,
                query,
                dataset_id=dataset_id,
                trust=trust,
                resolved_snapshots=validated.snapshots_read(),
                settings=settings,
            )
        )
    return results
