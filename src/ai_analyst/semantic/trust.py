"""Trust assessment (ARCHITECTURE 12.7, 13.10).

The tier is computed, never chosen. This module turns the inputs of one result
into `TrustFactor`s, and `TrustAssessment` turns factors into a tier through a
fixed table. Nothing here accepts a tier, and nothing a planner or responder
emits reaches it: the only inputs are the path, what the resolver recorded while
resolving concepts and columns, the snapshot resolution, the fiscal calendar,
the plan's own unresolved ambiguities, and the sanity checks.
"""

from __future__ import annotations

from enum import StrEnum

from ai_analyst.contracts.plan import PeriodKind
from ai_analyst.contracts.result import TrustAssessment, TrustFactor, TrustFactorKind
from ai_analyst.semantic.calendar import FiscalCalendarResolution
from ai_analyst.semantic.resolver import ConceptResolver
from ai_analyst.semantic.snapshots import SnapshotResolution


class AnalysisPath(StrEnum):
    """Which analytical path produced a result."""

    SEMANTIC = "semantic"
    INVESTIGATION = "investigation"
    GUARDED_SQL = "guarded_sql"


_PATH_FACTORS: dict[AnalysisPath, TrustFactor] = {
    AnalysisPath.SEMANTIC: TrustFactor(
        kind=TrustFactorKind.SEMANTIC_PATH,
        subject="path",
        reason="computed by a registry metric through the deterministic compiler",
    ),
    AnalysisPath.INVESTIGATION: TrustFactor(
        kind=TrustFactorKind.INVESTIGATION_PATH,
        subject="path",
        reason=(
            "an ad-hoc investigation: executed and checked, but its definition is "
            "not a registry metric"
        ),
    ),
    AnalysisPath.GUARDED_SQL: TrustFactor(
        kind=TrustFactorKind.GUARDED_SQL,
        subject="path",
        reason="computed by guarded SQL outside the typed plan vocabulary",
    ),
}

# Period kinds whose boundaries come from the fiscal calendar. A month or a
# custom range does not depend on the fiscal year start.
_CALENDAR_PERIODS = frozenset(
    {PeriodKind.FISCAL_QUARTER, PeriodKind.FISCAL_YEAR, PeriodKind.RELATIVE}
)


def assess(
    *,
    path: AnalysisPath,
    resolver: ConceptResolver,
    snapshot: SnapshotResolution | None = None,
    max_drift_days: int | None = None,
    calendar: FiscalCalendarResolution | None = None,
    period_kind: PeriodKind | None = None,
    unresolved_ambiguities: tuple[str, ...] = (),
    sanity_failures: tuple[str, ...] = (),
) -> TrustAssessment:
    """Compute one result's trust from its inputs.

    Every argument is a fact the deterministic pipeline already holds. There is
    deliberately no parameter through which a tier, or a factor's cost, could be
    supplied.
    """
    factors: list[TrustFactor] = [_PATH_FACTORS[path], *resolver.trust_factors]

    if snapshot is not None:
        limit = max_drift_days
        over = not snapshot.within_tolerance or (
            limit is not None and snapshot.drift_days > limit
        )
        if over:
            factors.append(
                TrustFactor(
                    kind=TrustFactorKind.SNAPSHOT_DRIFT,
                    subject="snapshot",
                    reason=(
                        f"the resolved snapshot is {snapshot.drift_days} day(s) from the "
                        "requested boundary, beyond tolerance"
                    ),
                )
            )

    if (
        calendar is not None
        and not calendar.is_resolved
        and period_kind in _CALENDAR_PERIODS
    ):
        factors.append(
            TrustFactor(
                kind=TrustFactorKind.CALENDAR_UNRESOLVED,
                subject="fiscal_calendar",
                reason=calendar.assumption,
            )
        )

    factors.extend(
        TrustFactor(
            kind=TrustFactorKind.UNRESOLVED_AMBIGUITY,
            subject="plan",
            reason=f"unresolved ambiguity: {item}",
        )
        for item in unresolved_ambiguities
    )
    factors.extend(
        TrustFactor(kind=TrustFactorKind.SANITY_FAILED, subject="result", reason=item)
        for item in sanity_failures
    )
    return TrustAssessment(factors=tuple(dict.fromkeys(factors)))
