"""Declared snapshot granularity and capture de-duplication (WP2).

The conformed grain is `(as_of DATE, opp_id)`. Some exports capture an
opportunity more than once on the same calendar day (a daylight-saving hour
repeated, a re-run). Ingestion hard-fails on that by default; a tenant may
DECLARE how to resolve it. Nothing here is inferred from the data.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class SnapshotGranularity(StrEnum):
    # One snapshot per calendar day. The conformed `as_of` is a DATE.
    DAY = "day"
    # Every capture is its own snapshot. Declared but not yet supported by
    # ingestion: the conformed `as_of` is a DATE, so this fails loudly.
    CAPTURE = "capture"


class IntradayPolicy(StrEnum):
    # Two captures of one opportunity on one day are a grain violation.
    FAIL = "fail"
    # Keep the latest capture timestamp; report what was dropped.
    LATEST_CAPTURE = "latest_capture"


class SnapshotPolicy(BaseModel):
    """A tenant's declaration of how same-day captures are handled."""

    model_config = ConfigDict(frozen=True)

    snapshot_granularity: SnapshotGranularity = SnapshotGranularity.DAY
    intraday_policy: IntradayPolicy = IntradayPolicy.FAIL


class CaptureResolution(BaseModel):
    """What capture de-duplication did at ingestion. Reported, never hidden."""

    model_config = ConfigDict(frozen=True)

    policy: IntradayPolicy
    # The source column whose timestamp decided "latest".
    capture_column: str
    # (as_of, opp_id) groups that held more than one capture.
    duplicate_groups: int = 0
    rows_dropped: int = 0
    # Groups whose kept and dropped captures differ in any conformed column.
    conflicting_groups: int = 0

    # TODO(trust): conflicting_groups > 0 should cap results at tier B via a
    # dataset-level trust factor. Trust factors reach the resolver through
    # binding caveats, which is a larger change than this milestone; for now
    # the count is persisted on the schema so it is reported, not hidden.
    @property
    def has_conflicts(self) -> bool:
        return self.conflicting_groups > 0


__all__ = ["CaptureResolution", "IntradayPolicy", "SnapshotGranularity", "SnapshotPolicy"]
