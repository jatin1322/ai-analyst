"""Fiscal calendar and period resolution (ARCHITECTURE 5.6).

Quarter labels resolve through this module, never through string parsing in the
compiler. "This quarter" resolves against the dataset's latest snapshot, not
against wall-clock today.

Fiscal year naming convention: a fiscal year is named for the calendar year in
which it **begins**. With a February start, FY2025 runs 2025-02-01 through
2026-01-31. With the default January start the convention is invisible, because
the fiscal and calendar years coincide.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from enum import StrEnum

from ai_analyst.config import Settings, get_settings
from ai_analyst.contracts.plan import Period, PeriodKind, RelativePeriod
from ai_analyst.contracts.tenant import TenantProfile

QUARTER_LABEL_RE = re.compile(r"^FY(\d{4})-Q([1-4])$")
YEAR_LABEL_RE = re.compile(r"^FY(\d{4})$")
MONTH_LABEL_RE = re.compile(r"^(\d{4})-(0[1-9]|1[0-2])$")


class CalendarError(ValueError):
    """Raised when a period cannot be resolved deterministically."""


def add_months(anchor: date, months: int) -> date:
    """Shift a date by whole months, clamping to the first of the month.

    Only ever called with day-one anchors, so no end-of-month clamping is
    needed and none is silently applied.
    """
    total = (anchor.year * 12 + anchor.month - 1) + months
    return date(total // 12, total % 12 + 1, 1)


@dataclass(frozen=True, order=True)
class FiscalQuarter:
    """One fiscal quarter, with concrete inclusive bounds."""

    fiscal_year: int
    quarter: int
    start: date
    end: date

    @property
    def label(self) -> str:
        return f"FY{self.fiscal_year}-Q{self.quarter}"

    def contains(self, day: date) -> bool:
        return self.start <= day <= self.end


@dataclass(frozen=True)
class ResolvedPeriod:
    """A period reduced to concrete inclusive bounds and a canonical label."""

    kind: PeriodKind
    start: date
    end: date
    label: str

    def contains(self, day: date) -> bool:
        return self.start <= day <= self.end


class FiscalCalendar:
    """Deterministic fiscal period arithmetic for one fiscal year start month."""

    def __init__(self, fiscal_year_start_month: int = 1) -> None:
        if not 1 <= fiscal_year_start_month <= 12:
            raise CalendarError(
                f"fiscal_year_start_month must be 1-12, got {fiscal_year_start_month}"
            )
        self.start_month = fiscal_year_start_month

    def __repr__(self) -> str:
        return f"FiscalCalendar(fiscal_year_start_month={self.start_month})"

    def fiscal_year_of(self, day: date) -> int:
        """The fiscal year a date belongs to, named for the year it begins."""
        return day.year if day.month >= self.start_month else day.year - 1

    def quarter_of(self, day: date) -> FiscalQuarter:
        """The fiscal quarter containing a date."""
        fiscal_year = self.fiscal_year_of(day)
        months_in = (day.month - self.start_month) % 12
        return self.quarter(fiscal_year, months_in // 3 + 1)

    def quarter(self, fiscal_year: int, quarter: int) -> FiscalQuarter:
        """Build a fiscal quarter from its ordinal position."""
        if not 1 <= quarter <= 4:
            raise CalendarError(f"quarter must be 1-4, got {quarter}")
        start = add_months(date(fiscal_year, self.start_month, 1), (quarter - 1) * 3)
        end = add_months(start, 3) - timedelta(days=1)
        return FiscalQuarter(fiscal_year=fiscal_year, quarter=quarter, start=start, end=end)

    def fiscal_year_bounds(self, fiscal_year: int) -> tuple[date, date]:
        start = date(fiscal_year, self.start_month, 1)
        return start, add_months(start, 12) - timedelta(days=1)

    def shift_quarter(self, quarter: FiscalQuarter, offset: int) -> FiscalQuarter:
        """Move a quarter forward or backward by `offset` quarters."""
        ordinal = quarter.fiscal_year * 4 + (quarter.quarter - 1) + offset
        return self.quarter(ordinal // 4, ordinal % 4 + 1)

    def next_quarter(self, quarter: FiscalQuarter) -> FiscalQuarter:
        return self.shift_quarter(quarter, 1)

    def previous_quarter(self, quarter: FiscalQuarter) -> FiscalQuarter:
        return self.shift_quarter(quarter, -1)

    def last_n_quarters(self, ending_at: FiscalQuarter, n: int) -> list[FiscalQuarter]:
        """The `n` quarters ending at and including `ending_at`, oldest first."""
        if n < 1:
            raise CalendarError(f"n must be at least 1, got {n}")
        return [self.shift_quarter(ending_at, -(n - 1 - i)) for i in range(n)]

    # Label parsing is the strict inverse of this calendar's own labelling. It
    # accepts only the canonical forms and raises on anything else, so no
    # fiscal period is ever inferred from free text.
    def parse_quarter_label(self, label: str) -> FiscalQuarter:
        match = QUARTER_LABEL_RE.match(label.strip())
        if not match:
            raise CalendarError(
                f"quarter label {label!r} is not canonical; expected FYYYYY-QN, e.g. FY2025-Q3"
            )
        return self.quarter(int(match.group(1)), int(match.group(2)))

    def parse_year_label(self, label: str) -> tuple[int, date, date]:
        match = YEAR_LABEL_RE.match(label.strip())
        if not match:
            raise CalendarError(
                f"fiscal year label {label!r} is not canonical; expected FYYYYY, e.g. FY2025"
            )
        fiscal_year = int(match.group(1))
        start, end = self.fiscal_year_bounds(fiscal_year)
        return fiscal_year, start, end

    def month_bounds(self, label: str) -> tuple[date, date]:
        match = MONTH_LABEL_RE.match(label.strip())
        if not match:
            raise CalendarError(
                f"month label {label!r} is not canonical; expected YYYY-MM, e.g. 2025-03"
            )
        start = date(int(match.group(1)), int(match.group(2)), 1)
        return start, add_months(start, 1) - timedelta(days=1)

    def to_resolved(self, quarter: FiscalQuarter) -> ResolvedPeriod:
        return ResolvedPeriod(
            kind=PeriodKind.FISCAL_QUARTER,
            start=quarter.start,
            end=quarter.end,
            label=quarter.label,
        )

    def resolve_periods(self, period: Period, anchor: date) -> list[ResolvedPeriod]:
        """Resolve a plan `Period` into one or more concrete periods.

        `anchor` is the dataset's latest snapshot date, never wall-clock today.
        A `last_n` relative period resolves to `n` periods, oldest first.
        """
        match period.kind:
            case PeriodKind.CUSTOM:
                return [
                    ResolvedPeriod(
                        kind=period.kind,
                        start=period.start,  # type: ignore[arg-type]
                        end=period.end,  # type: ignore[arg-type]
                        label=period.label or f"{period.start} to {period.end}",
                    )
                ]

            case PeriodKind.MONTH:
                if period.start and period.end:
                    start, end = period.start, period.end
                    label = period.label or f"{start:%Y-%m}"
                elif period.label:
                    start, end = self.month_bounds(period.label)
                    label = period.label
                else:
                    raise CalendarError("month period requires a label or explicit bounds")
                return [ResolvedPeriod(kind=period.kind, start=start, end=end, label=label)]

            case PeriodKind.FISCAL_YEAR:
                if period.label:
                    _, start, end = self.parse_year_label(period.label)
                    label = period.label
                elif period.start:
                    fiscal_year = self.fiscal_year_of(period.start)
                    start, end = self.fiscal_year_bounds(fiscal_year)
                    label = f"FY{fiscal_year}"
                else:
                    raise CalendarError(
                        "fiscal year period requires a label or a start date"
                    )
                return [ResolvedPeriod(kind=period.kind, start=start, end=end, label=label)]

            case PeriodKind.FISCAL_QUARTER:
                if period.label:
                    quarter = self.parse_quarter_label(period.label)
                elif period.start:
                    quarter = self.quarter_of(period.start)
                else:
                    raise CalendarError(
                        "fiscal quarter period requires a label or a start date"
                    )
                return [self.to_resolved(quarter)]

            case PeriodKind.RELATIVE:
                current = self.quarter_of(anchor)
                match period.relative:
                    case RelativePeriod.CURRENT:
                        return [self.to_resolved(current)]
                    case RelativePeriod.PREVIOUS:
                        return [self.to_resolved(self.previous_quarter(current))]
                    case RelativePeriod.LAST_N:
                        n = period.n or 1
                        return [self.to_resolved(q) for q in self.last_n_quarters(current, n)]
                    case _:  # pragma: no cover - guarded by the Period validator
                        raise CalendarError(f"unsupported relative period {period.relative}")

            case _:  # pragma: no cover - exhaustive over PeriodKind
                raise CalendarError(f"unsupported period kind {period.kind}")

    def resolve_period(self, period: Period, anchor: date) -> ResolvedPeriod:
        """Resolve to exactly one period. The most recent, for `last_n`."""
        return self.resolve_periods(period, anchor)[-1]


class FiscalCalendarSource(StrEnum):
    """Where a fiscal year start month came from."""

    TENANT_DECLARATION = "tenant_declaration"
    CONFIGURED_DEFAULT = "configured_default"

    @property
    def is_declared(self) -> bool:
        return self is FiscalCalendarSource.TENANT_DECLARATION


@dataclass(frozen=True)
class FiscalCalendarResolution:
    """Which fiscal calendar an analysis ran under, and whether anyone said so.

    The fiscal year start is **never inferred from data**. A quarter label in
    an export is not evidence: the label may itself have been derived from a
    calendar nobody wrote down, so reading it back would be circular. When no
    tenant declares one, the configured default is used and this object says
    the value is unresolved, so the assumption travels with every answer
    instead of disappearing into a setting.
    """

    calendar: FiscalCalendar
    source: FiscalCalendarSource
    tenant_id: str | None = None

    @property
    def start_month(self) -> int:
        return self.calendar.start_month

    @property
    def is_resolved(self) -> bool:
        """True only when a tenant actually declared the fiscal year start."""
        return self.source.is_declared

    @property
    def assumption(self) -> str:
        """The sentence an answer carries when the calendar is an assumption."""
        if self.is_resolved:
            return (
                f"Fiscal year starts in month {self.start_month}, as declared by "
                f"tenant {self.tenant_id}."
            )
        return (
            f"Fiscal year start month is unresolved for this dataset; the "
            f"configured default of month {self.start_month} was assumed. "
            f"Quarter boundaries depend on it."
        )


def resolve_calendar(
    tenant: TenantProfile | None = None, settings: Settings | None = None
) -> FiscalCalendarResolution:
    """Resolve the fiscal calendar for one tenant, explicitly.

    A tenant declaration wins. Otherwise the configured default is used and
    reported as unresolved, which is the state the production tenant is in.
    """
    settings = settings or get_settings()
    if tenant is not None and tenant.fiscal_year_start_month is not None:
        return FiscalCalendarResolution(
            calendar=FiscalCalendar(tenant.fiscal_year_start_month),
            source=FiscalCalendarSource.TENANT_DECLARATION,
            tenant_id=tenant.tenant_id,
        )
    return FiscalCalendarResolution(
        calendar=FiscalCalendar(settings.fiscal_year_start_month),
        source=FiscalCalendarSource.CONFIGURED_DEFAULT,
        tenant_id=tenant.tenant_id if tenant else None,
    )
