"""Snapshot selection rules (ARCHITECTURE 5.1).

Snapshot dates do not land on quarter boundaries, so selection is always
explicit. Every resolution reports the boundary requested, the snapshot
actually used, the drift between them, and whether that drift is within
tolerance. A substitution is always reported, never silent.

Drift beyond tolerance is a warning, not an error: the caller asked for the
nearest snapshot and got it. The genuine errors are the unresolvable cases,
where no snapshot satisfies the rule at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from ai_analyst.contracts.result import ResolvedSnapshot, SnapshotRule
from ai_analyst.semantic.calendar import ResolvedPeriod


class SnapshotResolutionError(ValueError):
    """Raised when no snapshot can satisfy a rule."""


@dataclass(frozen=True)
class SnapshotResolution:
    """The outcome of applying one snapshot rule."""

    rule: SnapshotRule
    resolved: tuple[date, ...]
    requested_boundary: date | None = None
    drift_days: int = 0
    within_tolerance: bool = True
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def as_of(self) -> date:
        """The single snapshot this rule resolved to."""
        if len(self.resolved) != 1:
            raise SnapshotResolutionError(
                f"{self.rule} resolved to {len(self.resolved)} snapshots, expected exactly one"
            )
        return self.resolved[0]

    def to_contract(self) -> list[ResolvedSnapshot]:
        """Convert to the contract carried on every ResultSet."""
        return [
            ResolvedSnapshot(
                rule=self.rule,
                requested_boundary=self.requested_boundary,
                resolved_as_of=resolved,
                drift_days=self.drift_days if len(self.resolved) == 1 else 0,
                within_tolerance=self.within_tolerance,
            )
            for resolved in self.resolved
        ]


class SnapshotResolver:
    """Resolves snapshot rules against a dataset's known snapshot dates."""

    def __init__(
        self, snapshot_dates: list[date], max_drift_days: int = 10
    ) -> None:
        if not snapshot_dates:
            raise SnapshotResolutionError("dataset has no snapshots")
        self.dates: tuple[date, ...] = tuple(sorted(set(snapshot_dates)))
        self.max_drift_days = max_drift_days

    @property
    def earliest(self) -> date:
        return self.dates[0]

    @property
    def latest(self) -> date:
        return self.dates[-1]

    def _nearest_on_or_before(self, boundary: date) -> date | None:
        candidates = [d for d in self.dates if d <= boundary]
        return candidates[-1] if candidates else None

    def _with_drift(
        self, rule: SnapshotRule, boundary: date, resolved: date
    ) -> SnapshotResolution:
        drift = (boundary - resolved).days
        within = drift <= self.max_drift_days
        warnings: list[str] = []
        if drift > 0:
            warnings.append(
                f"{rule.value} requested {boundary.isoformat()} but the nearest snapshot "
                f"on or before it is {resolved.isoformat()}, {drift} day(s) earlier"
            )
        if not within:
            warnings.append(
                f"snapshot drift of {drift} day(s) exceeds the tolerance of "
                f"{self.max_drift_days} day(s); treat this result with caution"
            )
        return SnapshotResolution(
            rule=rule,
            resolved=(resolved,),
            requested_boundary=boundary,
            drift_days=drift,
            within_tolerance=within,
            warnings=tuple(warnings),
        )

    def resolve(
        self,
        rule: SnapshotRule,
        period: ResolvedPeriod | None = None,
        explicit_date: date | None = None,
    ) -> SnapshotResolution:
        """Apply one snapshot rule."""
        match rule:
            case SnapshotRule.AS_OF_EXACT:
                if explicit_date is None:
                    raise SnapshotResolutionError("as_of_exact requires an explicit date")
                if explicit_date not in self.dates:
                    raise SnapshotResolutionError(
                        f"no snapshot exists on {explicit_date.isoformat()}; "
                        f"available snapshots run {self.earliest.isoformat()} to "
                        f"{self.latest.isoformat()}"
                    )
                return SnapshotResolution(
                    rule=rule, resolved=(explicit_date,), requested_boundary=explicit_date
                )

            case SnapshotRule.PERIOD_OPEN:
                boundary = self._require_period(rule, period).start
                resolved = self._nearest_on_or_before(boundary)
                if resolved is None:
                    raise SnapshotResolutionError(
                        f"period_open needs a snapshot on or before {boundary.isoformat()}, "
                        f"but the earliest snapshot is {self.earliest.isoformat()}"
                    )
                return self._with_drift(rule, boundary, resolved)

            case SnapshotRule.PERIOD_CLOSE:
                boundary = self._require_period(rule, period).end
                resolved = self._nearest_on_or_before(boundary)
                if resolved is None:
                    raise SnapshotResolutionError(
                        f"period_close needs a snapshot on or before {boundary.isoformat()}, "
                        f"but the earliest snapshot is {self.earliest.isoformat()}"
                    )
                return self._with_drift(rule, boundary, resolved)

            case SnapshotRule.LATEST:
                return SnapshotResolution(rule=rule, resolved=(self.latest,))

            case SnapshotRule.LATEST_IN_PERIOD:
                resolved_period = self._require_period(rule, period)
                inside = [
                    d for d in self.dates if resolved_period.start <= d <= resolved_period.end
                ]
                if not inside:
                    raise SnapshotResolutionError(
                        f"no snapshot falls inside {resolved_period.label} "
                        f"({resolved_period.start.isoformat()} to "
                        f"{resolved_period.end.isoformat()})"
                    )
                return SnapshotResolution(
                    rule=rule,
                    resolved=(inside[-1],),
                    requested_boundary=resolved_period.end,
                    drift_days=(resolved_period.end - inside[-1]).days,
                    within_tolerance=True,
                )

            case SnapshotRule.ALL:
                if period is None:
                    return SnapshotResolution(rule=rule, resolved=self.dates)
                inside = tuple(d for d in self.dates if period.start <= d <= period.end)
                if not inside:
                    raise SnapshotResolutionError(
                        f"no snapshot falls inside {period.label}"
                    )
                return SnapshotResolution(rule=rule, resolved=inside)

            case _:  # pragma: no cover - exhaustive over SnapshotRule
                raise SnapshotResolutionError(f"unsupported snapshot rule {rule}")

    def snapshots_between(
        self, start: date, end: date, *, inclusive_start: bool
    ) -> tuple[date, ...]:
        """Snapshot dates in a window. Used for 'ever reached' state checks."""
        if inclusive_start:
            return tuple(d for d in self.dates if start <= d <= end)
        return tuple(d for d in self.dates if start < d <= end)

    @staticmethod
    def _require_period(rule: SnapshotRule, period: ResolvedPeriod | None) -> ResolvedPeriod:
        if period is None:
            raise SnapshotResolutionError(f"{rule.value} requires a period")
        return period
