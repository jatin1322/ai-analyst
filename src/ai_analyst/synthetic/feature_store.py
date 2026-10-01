"""Synthetic ML feature-store generator (WP7).

Reproduces the SHAPE of a real opportunity-snapshot feature store we profiled
(daily panel grain, windowed activity families, directional score families,
precomputed rep aggregates, leakage columns, ETL metadata) using entirely
GENERIC column names invented for this project. No real column name, table
name, or value from the source system appears anywhere below or in generated
output — see ``FORBIDDEN_NAME_FRAGMENTS`` for the names this module is not
allowed to reproduce, which the test suite checks against every generated
column name and value.

Everything here is deterministic given a seed: the same ``GeneratorConfig``
always produces byte-identical output. There are no calls to ``datetime.now``,
``random`` (module-level), ``hash()``, or set iteration anywhere in the
generation path; every random draw goes through a ``random.Random`` instance
seeded from the config, and every "current time" field is derived from the
config's own dates.

Design notes (why, not just what):

* Rows are built as plain Python lists/dicts (not DuckDB, not pandas/numpy —
  the venv has neither, and CLAUDE.md forbids adding dependencies). This
  matches the task's explicit allowance ("build with DuckDB or plain Python
  lists"). The default fixture size (~100 opportunities x ~2 quarters) stays
  small enough that this is fast; the "millions of rows" benchmark size is
  intentionally not exercised by tests. `generate()` still returns a fully
  materialized `list[tuple]` (the task's own API shape), so that size is
  not free: two passes over the opportunities (DST planning, then the real
  simulation — see `_plan_duplicates`), each opportunity's rows finalized
  and converted to tuples immediately rather than accumulated as dicts, and
  a duplicate-day row copied only once instead of the naive rebuild. This
  keeps memory roughly proportional to `GeneratedDataset.rows` itself
  rather than a multiple of it: measured ~780MB RSS / ~8s for ~670k rows at
  1,000 opportunities x 8 quarters, which scales close to linearly, so the
  documented 20,000-opportunity benchmark is estimated at ~15GB / ~2.5-3
  minutes — plausible on a real dev machine, not run here (measure before
  trusting the extrapolation on a memory-constrained one).
* Money is tracked as integer cents throughout the simulation and converted
  to a float dollar amount exactly once, at row-emission time, so repeated
  arithmetic never drifts through binary floating point (CLAUDE.md #9).
* Each opportunity gets its own ``random.Random(f"{seed}:{i}")`` instance, so
  an opportunity's simulated life is independent of how many other
  opportunities exist or which variant is being written. Variant column
  renaming happens only at write time (see ``RENAME_MAPS``); the generator
  itself never sees tenant-specific names, mirroring the product's own
  concept/binding separation (CLAUDE.md #2.1) even though this is a
  standalone demo generator, not the product's semantic engine.
* Planted effects are driven by an exact predicate over an already-emitted
  column (see ``GroundTruth.planted_effects``), not by a hidden flag, so a
  downstream test (or an interview panel) can recover the effect using only
  the columns that ship in the data.
"""

from __future__ import annotations

import csv
import hashlib
import random
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Literal

import duckdb
from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Forbidden real names (never emit these, in any casing/punctuation form)
# ---------------------------------------------------------------------------
#
# Stored as (length, sha256-of-normalized-fragment) fingerprints rather than
# plaintext, so this file — which is meant to prove no real internal name
# ever reaches generated output — does not itself carry those names. See
# `contains_forbidden_name` for the sliding-window check this backs.

_FORBIDDEN_FINGERPRINTS: frozenset[tuple[int, str]] = frozenset(
    {
        (8, "2a6f7a69245c798d76399794c22b230f84ca8f037ee19e0bb33a62d323f28001"),
        (6, "ed05456a53250b9f5bae1d7d04037415887de2ff90fdfbcac246c69d83ad11c8"),
        (9, "8a7eb2987c98cccad97760661323df3f7330f3c2bdbfa8a4de3723aff34270e3"),
        (10, "c9d184e4c31458568f2d7e0508e06525ea0534ae9b28cb74f6f5226b15f5f28b"),
        (4, "5de6ffaca893b4b2371483d7e60888e955f604ebd5c0a26881e75d618128f361"),
        (7, "f304480b685d104e9a9817e0f15a064a4773fadbed8df472e985bf04d20c5bf7"),
    }
)


def _normalize_for_scan(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def contains_forbidden_name(text: str) -> bool:
    """True if any forbidden (real, non-generic) name fragment is present.

    Checked by hash fingerprint over a sliding window of the normalized
    (lowercase, alnum-only) text, not by plaintext substring search, so
    this module never has to hold the literal names it must never
    reproduce.
    """
    norm = _normalize_for_scan(text)
    n = len(norm)
    for length, digest in _FORBIDDEN_FINGERPRINTS:
        if length > n:
            continue
        for start in range(n - length + 1):
            window = norm[start : start + length]
            if hashlib.sha256(window.encode("utf-8")).hexdigest() == digest:
                return True
    return False

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

WINDOWS: tuple[int, ...] = (7, 14, 30, 60, 90, 180)
WINDOW_LABELS: tuple[str, ...] = tuple(f"{w}d" for w in WINDOWS) + ("lifetime",)
DIRECTIONS: tuple[str, ...] = ("ALL", "IN", "OUT")
SCORE_WINDOWS: tuple[int, ...] = (30, 90)
RECENCY_FIELDS: tuple[str, ...] = ("stage", "amount", "close_date", "next_steps")

STAGES: tuple[str, ...] = (
    "1 - Discovery",
    "2 - Qualification",
    "3 - Proposal",
    "4 - Negotiation",
    "5 - Commit",
    "6 - Order Placed",
)
CLOSED_WON = "Closed Won"
CLOSED_LOST = "Closed Lost"
JUNK_STAGES: tuple[str, ...] = ("not available", "DELETED_IN_CRM")
FORECAST_CATEGORIES: tuple[str, ...] = ("Pipeline", "Best Case", "Commit", "Omitted")

PIPELINE_VERSION = "synthetic-fs-1.0"
RUN_ID = "synthetic-run"
CAPTURE_TIME = time(2, 0, 0)
DST_TIME_FIRST = time(3, 0, 0)
DST_TIME_SECOND = time(4, 0, 0)

# Planted-effect parameters (documented so a test can replicate them exactly).
BASE_PUSH_HAZARD = 0.004
SILENT_PUSH_HAZARD = 0.035
BASE_P_WIN = 0.15
MEETING_WIN_EFFECT = 0.15
MAX_MEETING_EFFECT_INPUT = 5  # meetings above this stop increasing p_win

# ---------------------------------------------------------------------------
# Column vocabulary
# ---------------------------------------------------------------------------

GRAIN_COLUMNS: tuple[str, ...] = ("opp_id", "as_of", "as_of_date", "as_of_qtr")
CORE_COLUMNS: tuple[str, ...] = (
    "account_id",
    "owner_id",
    "stage",
    "amount",
    "probability",
    "forecast_category",
    "days_to_close",
    "created_date",
)


def _window_columns() -> tuple[str, ...]:
    cols: list[str] = []
    for fam in ("email_count", "inbound_count", "outbound_count", "meeting_count"):
        cols.extend(f"{fam}_{label}" for label in WINDOW_LABELS)
    cols.append("days_since_last_email")
    return tuple(cols)


def _score_columns() -> tuple[str, ...]:
    cols: list[str] = []
    for direction in DIRECTIONS:
        cols.extend(f"{direction}_tone_avg_{w}d" for w in SCORE_WINDOWS)
        cols.extend(f"{direction}_objection_sum_{w}d" for w in SCORE_WINDOWS)
    return tuple(cols)


def _recency_columns() -> tuple[str, ...]:
    return tuple(f"{f}_updated_days" for f in RECENCY_FIELDS)


REP_COLUMNS: tuple[str, ...] = ("rep_win_rate", "rep_avg_cycle_days", "rep_open_deal_count")
LEAKAGE_COLUMNS: tuple[str, ...] = (
    "outcome_fate",
    "outcome_date",
    "win_label",
    "win_label_mask",
    "slip_label",
    "slip_label_mask",
    "train",
)
ETL_COLUMNS: tuple[str, ...] = ("pipeline_version", "config_hash", "run_id", "scored_at")

WINDOW_COLUMNS: tuple[str, ...] = _window_columns()
SCORE_COLUMNS: tuple[str, ...] = _score_columns()
RECENCY_COLUMNS: tuple[str, ...] = _recency_columns()

ALL_COLUMNS: tuple[str, ...] = (
    GRAIN_COLUMNS
    + CORE_COLUMNS
    + WINDOW_COLUMNS
    + SCORE_COLUMNS
    + RECENCY_COLUMNS
    + REP_COLUMNS
    + LEAKAGE_COLUMNS
    + ETL_COLUMNS
)

# DuckDB column types, used to load the written CSV without auto-detection
# (auto-detect is a non-determinism risk across DuckDB versions/files).
_INT_COLS = set(WINDOW_COLUMNS) - {"days_since_last_email"}


def _column_type(col: str) -> str:
    if col in ("as_of", "scored_at"):
        return "TIMESTAMP"
    if col in ("as_of_date", "created_date", "outcome_date"):
        return "DATE"
    if col == "amount":
        return "DOUBLE"
    if col == "probability":
        return "DOUBLE"
    if col == "days_to_close":
        return "BIGINT"
    if col in _INT_COLS or col == "days_since_last_email":
        return "BIGINT"
    if col.endswith("_tone_avg_30d") or col.endswith("_tone_avg_90d"):
        return "DOUBLE"
    if col.endswith("_objection_sum_30d") or col.endswith("_objection_sum_90d"):
        return "BIGINT"
    if col.endswith("_updated_days"):
        return "BIGINT"
    if col == "rep_win_rate":
        return "DOUBLE"
    if col == "rep_avg_cycle_days":
        return "DOUBLE"
    if col == "rep_open_deal_count":
        return "BIGINT"
    if col in ("win_label", "win_label_mask", "slip_label", "slip_label_mask"):
        return "BIGINT"
    if col == "train":
        return "BOOLEAN"
    return "VARCHAR"


COLUMN_TYPES: dict[str, str] = {c: _column_type(c) for c in ALL_COLUMNS}

# ---------------------------------------------------------------------------
# Tenant-variant header renames. Ground truth for tests only — the product
# under test must never read this mapping; it exists purely so this demo
# generator can prove that onboarding generalises across schemas.
# ---------------------------------------------------------------------------

RENAME_MAPS: dict[str, dict[str, str]] = {
    "alpha": {
        "opp_id": "deal_key",
        "as_of": "capture_ts",
        "amount": "booking_value",
        "stage": "pipeline_step",
        "owner_id": "rep_ref",
        "account_id": "customer_key",
        "created_date": "open_date",
        **{f"email_count_{label}": f"emails_{label}" for label in WINDOW_LABELS},
    },
    "beta": {
        "opp_id": "record_id",
        "as_of": "snapshot_ts",
        "amount": "deal_value",
        "stage": "status_code",
        "owner_id": "owner_ref",
        "account_id": "customer_ref",
        "created_date": "created_on",
        **{f"meeting_count_{label}": f"meetings_{label}" for label in WINDOW_LABELS},
    },
}


def variant_columns(variant: str) -> list[str]:
    """The emitted header for a variant, same order as ``ALL_COLUMNS``."""
    mapping = RENAME_MAPS.get(variant, {})
    return [mapping.get(c, c) for c in ALL_COLUMNS]


# ---------------------------------------------------------------------------
# Config / ground truth contracts
# ---------------------------------------------------------------------------


class GeneratorConfig(BaseModel):
    """Everything needed to reproduce a dataset byte-for-byte."""

    seed: int = 42
    n_opportunities: int = 100
    start_date: date = date(2025, 1, 1)
    n_quarters: int = 2
    dst_dates: list[date] = Field(default_factory=lambda: [date(2025, 3, 9)])
    dst_duplicates_per_date: int = 5
    conflict_pairs: int = 3
    variant: Literal["base", "alpha", "beta"] = "base"
    output_format: Literal["csv", "parquet", "hive_parquet"] = "csv"

    @property
    def panel_end(self) -> date:
        return self.start_date + timedelta(days=91 * self.n_quarters - 1)

    @model_validator(mode="after")
    def _validate(self) -> GeneratorConfig:
        if self.n_opportunities < 1:
            raise ValueError("n_opportunities must be >= 1")
        if self.n_quarters < 1:
            raise ValueError("n_quarters must be >= 1")
        if self.dst_duplicates_per_date < 0:
            raise ValueError("dst_duplicates_per_date must be >= 0")
        if self.conflict_pairs < 0:
            raise ValueError("conflict_pairs must be >= 0")
        for d in self.dst_dates:
            if not (self.start_date <= d <= self.panel_end):
                raise ValueError(f"dst_date {d} falls outside the panel window")
        return self

    def config_hash(self) -> str:
        canonical = (
            f"{self.seed}|{self.n_opportunities}|{self.start_date.isoformat()}|"
            f"{self.n_quarters}|{sorted(self.dst_dates)}|{self.dst_duplicates_per_date}|"
            f"{self.conflict_pairs}"
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class PlantedEffect(BaseModel):
    """A documented, recoverable effect baked into the simulation."""

    name: str
    description: str
    predicate: str
    parameters: dict[str, float]


class GroundTruth(BaseModel):
    """What was actually planted, so tests never have to re-derive it from
    the same SQL that would later check it (that would be circular)."""

    seed: int
    variant: str
    n_opportunities: int
    n_rows: int
    n_snapshot_days: int
    duplicate_groups: int
    conflicting_groups: int
    label_convention: str
    planted_effects: list[PlantedEffect]


@dataclass
class GeneratedDataset:
    """Column-name-agnostic rows (base names, ``ALL_COLUMNS`` order) plus
    the ground truth that was planted while generating them."""

    columns: tuple[str, ...]
    rows: list[tuple]
    ground_truth: GroundTruth


# ---------------------------------------------------------------------------
# Per-opportunity simulation
# ---------------------------------------------------------------------------


@dataclass
class _DayEvents:
    inbound: int = 0
    outbound: int = 0
    meetings: int = 0
    in_tone_sum: float = 0.0
    in_tone_n: int = 0
    out_tone_sum: float = 0.0
    out_tone_n: int = 0
    in_obj: int = 0
    out_obj: int = 0


@dataclass
class _CumTrack:
    """Prefix sums so any trailing window is an O(1) lookup."""

    values: list[float] = field(default_factory=lambda: [0.0])

    def push(self, v: float) -> None:
        self.values.append(self.values[-1] + v)

    def window(self, today_idx: int, w: int | None) -> float:
        # today_idx is 0-based index into the *emitted* value array; values[]
        # has a leading 0 sentinel, so values[today_idx + 1] is the cumulative
        # total through and including today.
        hi = self.values[today_idx + 1]
        if w is None:  # lifetime
            return hi
        lo_idx = max(0, today_idx + 1 - w)
        return hi - self.values[lo_idx]


def _fiscal_label(d: date) -> str:
    return f"{d.year}Q{(d.month - 1) // 3 + 1}"


def _rep_profile(seed: int, owner_idx: int) -> tuple[float, float, int]:
    """Precomputed-looking rep aggregates: plausible, constant per owner,
    lineage deliberately undocumented (CLAUDE.md #5 — these are exactly the
    kind of feature that must never be trusted just because it looks tidy).
    """
    rng = random.Random(f"{seed}:rep:{owner_idx}")
    win_rate = round(rng.uniform(0.15, 0.55), 4)
    avg_cycle = round(rng.uniform(35.0, 130.0), 1)
    open_count = rng.randint(1, 35)
    return win_rate, avg_cycle, open_count


def _simulate_opportunity(
    idx: int, config: GeneratorConfig, *, want_rows: bool = True
) -> tuple[list[dict], dict[date, str]]:
    """Simulate one opportunity's full daily life.

    Returns (per-day row dicts covering the emit window, {date: stage} map
    restricted to dst_dates that actually get emitted — used to decide DST
    duplicate eligibility).

    ``want_rows=False`` runs the identical day-by-day simulation (same RNG
    draws, in the same order, so it stays consistent with a later
    ``want_rows=True`` call for the same opportunity) but skips building the
    71-key row dict and its window/score sub-dicts, which is most of the
    per-day cost. `generate()` uses this to plan DST duplicates/conflicts in
    a first pass without holding every opportunity's rows in memory at once,
    then re-simulates with `want_rows=True` in a second pass that finalizes
    and emits rows immediately, one opportunity at a time.
    """
    rng = random.Random(f"{config.seed}:{idx}")
    opp_id = f"O-{idx:06d}"

    n_accounts = max(5, config.n_opportunities // 8)
    n_reps = max(3, config.n_opportunities // 15)
    account_id = f"A-{rng.randrange(n_accounts):05d}"
    owner_idx = rng.randrange(n_reps)
    owner_id = f"R-{owner_idx:04d}"
    rep_win_rate, rep_avg_cycle, rep_open_count = _rep_profile(config.seed, owner_idx)

    panel_len = 91 * config.n_quarters
    # Skewed toward "already in the panel before it opens" (most CRM pipelines
    # are not brand-new each quarter) so the default fixture size lands in the
    # "tens of thousands of rows" the task calls for, not a thin sliver of it.
    created_offset = rng.randint(-120, max(30, panel_len // 4))
    created_date = config.start_date + timedelta(days=created_offset)
    emit_start = max(created_date, config.start_date)
    if emit_start > config.panel_end:
        return [], {}

    baseline_cycle = rng.randint(45, 150)
    planned_close = created_date + timedelta(days=baseline_cycle)

    amount_cents = rng.randint(500_000, 50_000_000)  # $5,000 - $500,000

    inbound_active = True
    stage_last_changed = created_date
    amount_last_changed = created_date
    close_last_changed = created_date
    next_steps_last_changed = created_date

    inbound_track = _CumTrack()
    outbound_track = _CumTrack()
    meeting_track = _CumTrack()
    in_tone_sum_track = _CumTrack()
    in_tone_n_track = _CumTrack()
    out_tone_sum_track = _CumTrack()
    out_tone_n_track = _CumTrack()
    in_obj_track = _CumTrack()
    out_obj_track = _CumTrack()

    last_email_date: date | None = None
    closed = False
    outcome_fate: str | None = None
    outcome_date: date | None = None
    push_dates: list[date] = []
    prev_stage: str | None = None

    rows: list[dict] = []
    dst_stage_by_date: dict[date, str] = {}

    day_idx = -1
    today = created_date
    while today <= config.panel_end:
        day_idx += 1

        events = _DayEvents()
        if not closed:
            # Silent-spell toggle: independent per-day Markov chain.
            if inbound_active and rng.random() < 0.012:
                inbound_active = False
            elif not inbound_active and rng.random() < 0.05:
                inbound_active = True

            if inbound_active and rng.random() < 0.35:
                events.inbound = rng.randint(1, 3)
                events.in_tone_sum = sum(rng.uniform(-1.0, 1.0) for _ in range(events.inbound))
                events.in_tone_n = events.inbound
                events.in_obj = sum(1 for _ in range(events.inbound) if rng.random() < 0.15)
            if rng.random() < 0.30:
                events.outbound = rng.randint(1, 2)
                events.out_tone_sum = sum(rng.uniform(-1.0, 1.0) for _ in range(events.outbound))
                events.out_tone_n = events.outbound
                events.out_obj = sum(1 for _ in range(events.outbound) if rng.random() < 0.10)
            if rng.random() < 0.08:
                events.meetings = 1

        if events.inbound or events.outbound:
            last_email_date = today

        inbound_track.push(events.inbound)
        outbound_track.push(events.outbound)
        meeting_track.push(events.meetings)
        in_tone_sum_track.push(events.in_tone_sum)
        in_tone_n_track.push(events.in_tone_n)
        out_tone_sum_track.push(events.out_tone_sum)
        out_tone_n_track.push(events.out_tone_n)
        in_obj_track.push(events.in_obj)
        out_obj_track.push(events.out_obj)

        inbound_30 = inbound_track.window(day_idx, 30)

        # --- push decision (uses only already-emitted inbound_count_30d) ---
        if not closed:
            silent = inbound_30 == 0
            hazard = SILENT_PUSH_HAZARD if silent else BASE_PUSH_HAZARD
            if rng.random() < hazard:
                push_days = rng.randint(20, 60)
                planned_close = planned_close + timedelta(days=push_days)
                close_last_changed = today
                push_dates.append(today)

        # --- stage progression ---
        if not closed:
            elapsed = (today - created_date).days
            duration = max(1, (planned_close - created_date).days)
            progress = elapsed / duration
            stage_idx = min(len(STAGES) - 1, int(progress * len(STAGES)))
            new_stage = STAGES[stage_idx]
            if rng.random() < 0.01:
                new_stage = rng.choice(JUNK_STAGES)
            stage = new_stage

            # --- close decision (uses meeting_count_30d just computed) ---
            days_to_planned = (planned_close - today).days
            if days_to_planned <= 0:
                close_hazard = 0.35
            elif days_to_planned <= 14:
                close_hazard = 0.05
            else:
                close_hazard = 0.0
            # Deliberately no special-case at config.panel_end: opportunities
            # still open on the last day are left open (natural censoring),
            # which is what makes win_label_mask == 0 non-vacuous.
            if close_hazard > 0 and rng.random() < close_hazard:
                closed = True
                meetings_recent = meeting_track.window(day_idx, 30)
                capped = min(meetings_recent, MAX_MEETING_EFFECT_INPUT)
                p_win = BASE_P_WIN + MEETING_WIN_EFFECT * capped
                p_win = max(0.03, min(0.97, p_win))
                won = rng.random() < p_win
                stage = CLOSED_WON if won else CLOSED_LOST
                outcome_fate = "W" if won else "L"
                outcome_date = today
        else:
            stage = CLOSED_WON if outcome_fate == "W" else CLOSED_LOST

        if prev_stage is None or stage != prev_stage:
            stage_last_changed = today
        prev_stage = stage

        # --- amount drift (rare, only while open) ---
        if not closed and rng.random() < 0.015:
            factor = rng.uniform(0.85, 1.20)
            amount_cents = max(100_000, int(amount_cents * factor))
            amount_last_changed = today

        if rng.random() < 0.08:
            next_steps_last_changed = today

        # --- derived fields ---
        days_to_close = (
            (outcome_date - today).days  # type: ignore[union-attr]
            if closed
            else (planned_close - today).days
        )

        if closed:
            probability = 100.0 if outcome_fate == "W" else 0.0
        else:
            stage_pos = STAGES.index(stage) if stage in STAGES else 0
            probability = round(min(95.0, 10.0 + stage_pos * 15.0), 1)

        if probability >= 80:
            forecast_category = "Commit"
        elif probability >= 50:
            forecast_category = "Best Case"
        elif probability >= 20:
            forecast_category = "Pipeline"
        else:
            forecast_category = "Omitted"

        emit_today = today >= emit_start

        if want_rows and emit_today:
            days_since_last_email = (
                (today - last_email_date).days if last_email_date is not None else None
            )

            window_values: dict[str, float | int | None] = {}
            for label, w in zip(WINDOW_LABELS, WINDOWS + (None,), strict=True):
                inb = int(inbound_track.window(day_idx, w))
                outb = int(outbound_track.window(day_idx, w))
                window_values[f"inbound_count_{label}"] = inb
                window_values[f"outbound_count_{label}"] = outb
                window_values[f"email_count_{label}"] = inb + outb
                window_values[f"meeting_count_{label}"] = int(meeting_track.window(day_idx, w))

            score_values: dict[str, float | int | None] = {}
            for w in SCORE_WINDOWS:
                in_n = in_tone_n_track.window(day_idx, w)
                out_n = out_tone_n_track.window(day_idx, w)
                in_sum = in_tone_sum_track.window(day_idx, w)
                out_sum = out_tone_sum_track.window(day_idx, w)
                all_n = in_n + out_n
                all_sum = in_sum + out_sum
                score_values[f"IN_tone_avg_{w}d"] = round(in_sum / in_n, 4) if in_n else None
                score_values[f"OUT_tone_avg_{w}d"] = round(out_sum / out_n, 4) if out_n else None
                score_values[f"ALL_tone_avg_{w}d"] = round(all_sum / all_n, 4) if all_n else None
                score_values[f"IN_objection_sum_{w}d"] = int(in_obj_track.window(day_idx, w))
                score_values[f"OUT_objection_sum_{w}d"] = int(out_obj_track.window(day_idx, w))
                score_values[f"ALL_objection_sum_{w}d"] = int(
                    in_obj_track.window(day_idx, w) + out_obj_track.window(day_idx, w)
                )

            row = {
                "opp_id": opp_id,
                "as_of_date": today,
                "as_of_qtr": _fiscal_label(today),
                "account_id": account_id,
                "owner_id": owner_id,
                "stage": stage,
                "amount": round(amount_cents / 100, 2),
                "probability": probability,
                "forecast_category": forecast_category,
                "days_to_close": days_to_close,
                "created_date": created_date,
                "days_since_last_email": days_since_last_email,
                **window_values,
                **score_values,
                "stage_updated_days": (today - stage_last_changed).days,
                "amount_updated_days": (today - amount_last_changed).days,
                "close_date_updated_days": (today - close_last_changed).days,
                "next_steps_updated_days": (today - next_steps_last_changed).days,
                "rep_win_rate": rep_win_rate,
                "rep_avg_cycle_days": rep_avg_cycle,
                "rep_open_deal_count": rep_open_count,
                # Leakage columns are intentionally NOT filled causally here.
                # They are backfilled onto every row after the day loop ends
                # (see below) — that backfill, using outcome/push information
                # that postdates the row's own as_of, is what makes them
                # leakage columns rather than AS_OF_FACT columns.
                "outcome_fate": None,
                "outcome_date": None,
                "win_label": None,
                "win_label_mask": 0,
                "slip_label": None,
                "slip_label_mask": 0,
                "train": (idx % 5) != 0,  # split by opportunity, not row
            }
            rows.append(row)

        if emit_today and today in config.dst_dates:
            dst_stage_by_date[today] = stage

        today += timedelta(days=1)

    if want_rows:
        _backfill_leakage_columns(rows, closed, outcome_fate, outcome_date, push_dates)
    return rows, dst_stage_by_date


def _backfill_leakage_columns(
    rows: list[dict],
    closed: bool,
    outcome_fate: str | None,
    outcome_date: date | None,
    push_dates: list[date],
) -> None:
    """Retrospective backfill of the leakage columns (CLAUDE.md #5, #4.2:
    FUTURE_CONTAMINATED is exactly what these columns are for).

    outcome_fate/outcome_date/win_label are "known only in hindsight": if the
    opportunity closed anywhere within the panel, every one of its rows —
    including rows long before the close — gets the eventual outcome.

    slip_label answers "will this deal's close date slip again after this
    snapshot?" — a forward-looking label, matching its own mask: mask is 1
    once that question is answerable from the panel (either a later push
    has already been observed, or the deal has closed with no further push
    possible), and the label is then 1 iff such a later push occurred. A
    push occurring after panel end, for a still-open deal, stays
    undetermined (mask 0), since the panel cannot see past its own end.
    """
    win_mask = 1 if closed else 0
    win_value = (1 if outcome_fate == "W" else 0) if closed else None
    for row in rows:
        row["outcome_fate"] = outcome_fate if closed else None
        row["outcome_date"] = outcome_date if closed else None
        row["win_label"] = win_value
        row["win_label_mask"] = win_mask
        later_push_observed = any(pd > row["as_of_date"] for pd in push_dates)
        if later_push_observed:
            row["slip_label"] = 1
            row["slip_label_mask"] = 1
        elif closed:
            row["slip_label"] = 0
            row["slip_label_mask"] = 1
        else:
            row["slip_label"] = None
            row["slip_label_mask"] = 0


# ---------------------------------------------------------------------------
# Top-level generation
# ---------------------------------------------------------------------------


def _plan_duplicates(
    config: GeneratorConfig,
) -> tuple[set[tuple[date, int]], set[tuple[date, int]]]:
    """Decide which (dst_date, opp_idx) pairs get duplicated/conflicted.

    Runs every opportunity's simulation with ``want_rows=False`` — same RNG
    draws as the real pass, so results are consistent, but none of the
    per-day row dicts are built or held. This keeps the planning pass cheap
    in memory regardless of ``n_opportunities`` x ``n_quarters``, which
    matters once the real pass streams rows instead of collecting them all
    (see ``generate``).
    """
    dst_candidates: dict[date, list[int]] = {d: [] for d in config.dst_dates}
    for i in range(config.n_opportunities):
        _, dst_stage_by_date = _simulate_opportunity(i, config, want_rows=False)
        for d, stage in dst_stage_by_date.items():
            if stage in STAGES:
                dst_candidates[d].append(i)

    duplicate_keys: list[tuple[date, int]] = []  # (date, opp_idx) selected for duplication
    for d in config.dst_dates:
        eligible = dst_candidates[d]
        dup_rng = random.Random(f"{config.seed}:dst:{d.isoformat()}")
        k = min(config.dst_duplicates_per_date, len(eligible))
        chosen = dup_rng.sample(eligible, k=k) if k else []
        duplicate_keys.extend((d, i) for i in chosen)

    if config.conflict_pairs > len(duplicate_keys):
        raise ValueError(
            f"conflict_pairs ({config.conflict_pairs}) exceeds the number of duplicate "
            f"groups actually planted ({len(duplicate_keys)}); raise "
            "dst_duplicates_per_date or add more dst_dates."
        )
    duplicate_keys.sort(key=lambda t: (t[0], t[1]))
    conflict_rng = random.Random(f"{config.seed}:dst:conflicts")
    conflict_set = set(
        conflict_rng.sample(duplicate_keys, k=config.conflict_pairs)
        if config.conflict_pairs
        else []
    )
    return set(duplicate_keys), conflict_set


def generate(config: GeneratorConfig) -> GeneratedDataset:
    """Deterministically simulate the full panel described by ``config``.

    Two passes over the opportunities: `_plan_duplicates` decides DST
    duplicate/conflict placement first (cheaply, see its docstring), then
    each opportunity is re-simulated with `want_rows=True` and finalized
    (ETL metadata, `as_of`, DST duplication) immediately, one at a time, so
    only one opportunity's rows are ever alive at once alongside the
    growing output list — not two full copies of the whole panel.
    """
    config_hash = config.config_hash()
    scored_at = datetime.combine(config.panel_end, time(0, 0, 0))

    duplicate_key_set, conflict_set = _plan_duplicates(config)

    # Rows are converted to tuples (ALL_COLUMNS order) and the source dict
    # discarded immediately, rather than accumulating a list of finalized
    # dicts and converting in one bulk pass at the end — holding both the
    # dict list and the tuple list at once would be the last remaining
    # doubling of peak memory at the benchmark size.
    final_rows: list[tuple] = []
    for i in range(config.n_opportunities):
        opp_rows, _ = _simulate_opportunity(i, config, want_rows=True)
        for r in opp_rows:
            d = r["as_of_date"]
            r["pipeline_version"] = PIPELINE_VERSION
            r["config_hash"] = config_hash
            r["run_id"] = RUN_ID
            r["scored_at"] = scored_at
            if (d, i) in duplicate_key_set:
                r["as_of"] = datetime.combine(d, DST_TIME_FIRST)
                second = dict(r)
                second["as_of"] = datetime.combine(d, DST_TIME_SECOND)
                if (d, i) in conflict_set:
                    current_idx = STAGES.index(second["stage"])
                    second["stage"] = STAGES[(current_idx + 1) % len(STAGES)]
                final_rows.append(tuple(r.get(c) for c in ALL_COLUMNS))
                final_rows.append(tuple(second.get(c) for c in ALL_COLUMNS))
            else:
                r["as_of"] = datetime.combine(d, CAPTURE_TIME)
                final_rows.append(tuple(r.get(c) for c in ALL_COLUMNS))

    # No global sort needed: opp_id ("O-000000") sorts lexically in the same
    # order it was generated (i ascending), each opportunity's own rows are
    # already date-ordered, and within a duplicated day 03:00 < 04:00 — so
    # iterating i ascending, in generation order, already yields rows sorted
    # by (opp_id, as_of).

    conflicting_groups = len(conflict_set)
    n_snapshot_days = 91 * config.n_quarters

    ground_truth = GroundTruth(
        seed=config.seed,
        variant=config.variant,
        n_opportunities=config.n_opportunities,
        n_rows=len(final_rows),
        n_snapshot_days=n_snapshot_days,
        duplicate_groups=len(duplicate_key_set),
        conflicting_groups=conflicting_groups,
        label_convention=(
            "outcome_fate/outcome_date/win_label are backfilled retrospectively onto "
            "EVERY row of an opportunity that closes within the panel window (this is "
            "the deliberate leakage: these columns tell you, on a row from months "
            "before close, how the deal eventually turned out). win_label_mask==1 iff "
            "the opportunity closed within the panel; win_label is null iff the mask "
            "is 0 (still open at panel end). slip_label answers a forward-looking "
            "question -- 'does this deal's close date slip again after this snapshot?' "
            "-- so slip_label_mask==1 iff that is already answerable from the panel "
            "(a later push has been observed, or the deal has since closed with no "
            "further push possible), and slip_label is then 1 iff such a later push "
            "occurred, 0 otherwise; both are null/0 while still undetermined. train is "
            "a per-opportunity split flag, not per-row."
        ),
        planted_effects=[
            PlantedEffect(
                name="silent_inbound_push",
                description=(
                    "Opportunities with zero inbound email in the trailing 30 days "
                    "(inbound_count_30d == 0), on a day other than their creation day "
                    "(which trivially has no email history yet), are pushed (planned "
                    "close date moved out, visible as close_date_updated_days == 0) "
                    "at an elevated daily hazard vs. active opportunities."
                ),
                predicate=(
                    "as_of_date > created_date AND inbound_count_30d == 0 "
                    "AND close_date_updated_days == 0"
                ),
                parameters={
                    "base_push_hazard": BASE_PUSH_HAZARD,
                    "silent_push_hazard": SILENT_PUSH_HAZARD,
                },
            ),
            PlantedEffect(
                name="meetings_drive_wins",
                description=(
                    "On the day an opportunity closes, its win probability increases "
                    "with meeting_count_30d (capped), so meeting_count_30d on the row "
                    "where as_of_date == outcome_date predicts win_label."
                ),
                predicate="as_of_date == outcome_date",
                parameters={
                    "base_p_win": BASE_P_WIN,
                    "meeting_win_effect": MEETING_WIN_EFFECT,
                    "max_meeting_effect_input": float(MAX_MEETING_EFFECT_INPUT),
                },
            ),
        ],
    )

    return GeneratedDataset(columns=ALL_COLUMNS, rows=final_rows, ground_truth=ground_truth)


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def _format_value(v: object) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (date, datetime)):
        return v.isoformat()
    if isinstance(v, float):
        return repr(v)
    return str(v)


def write_csv(dataset: GeneratedDataset, path: Path, variant: str = "base") -> Path:
    """Write the dataset as CSV with the given variant's header names.

    Byte-deterministic: same dataset + variant always produce identical bytes
    (no locale-dependent formatting, no wall-clock content).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    header = variant_columns(variant)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in dataset.rows:
            writer.writerow(_format_value(v) for v in row)
    return path


def _duckdb_read_csv_columns(variant: str) -> str:
    header = variant_columns(variant)
    types = [COLUMN_TYPES[base] for base in ALL_COLUMNS]
    pairs = ", ".join(f"'{h}': '{t}'" for h, t in zip(header, types, strict=True))
    return "{" + pairs + "}"


def write_parquet(dataset: GeneratedDataset, csv_path: Path, out_path: Path, variant: str) -> Path:
    """Write a single parquet file, via DuckDB, from an already-written CSV.

    Going through CSV (rather than a giant SQL VALUES literal or pandas,
    which the venv doesn't have) keeps this dependency-free and reuses the
    exact bytes already validated for determinism.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    columns = _duckdb_read_csv_columns(variant)
    con = duckdb.connect(":memory:")
    try:
        con.execute(
            f"""
            COPY (
                SELECT * FROM read_csv(
                    '{csv_path.as_posix()}',
                    header = true,
                    columns = {columns}
                )
            ) TO '{out_path.as_posix()}' (FORMAT PARQUET)
            """
        )
    finally:
        con.close()
    return out_path


def write_hive_parquet(
    dataset: GeneratedDataset, csv_path: Path, out_dir: Path, variant: str
) -> Path:
    """Write a hive-partitioned-by-quarter parquet directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = _duckdb_read_csv_columns(variant)
    con = duckdb.connect(":memory:")
    try:
        con.execute(
            f"""
            COPY (
                SELECT * FROM read_csv(
                    '{csv_path.as_posix()}',
                    header = true,
                    columns = {columns}
                )
            ) TO '{out_dir.as_posix()}' (
                FORMAT PARQUET, PARTITION_BY (as_of_qtr), OVERWRITE_OR_IGNORE true
            )
            """
        )
    finally:
        con.close()
    return out_dir
