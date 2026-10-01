"""Deterministic synthetic data shaped like the production opportunity export.

Nothing here is real data. It carries the 129 production column names plus the
CRM narrative fields the project owner listed, so that classification and
profiling are exercised against the real column vocabulary: canonical columns
under non-canonical headers, numeric features, sentinel-valued history features,
outcome columns, dates, booleans, and free text.

Values are functions of (opportunity index, snapshot index); no randomness.
Column *names* select how a value is generated. That is fixture convenience
only, and nothing in the system under test infers meaning that way.

Two deliberate differences from the real export, both documented:

* a raw ``close_date`` column is included, because reconstructing it from
  ``days_to_close`` is not implemented yet (use ``include_close_date=False`` to
  see the loud failure instead);
* an optional ``synthetic_status`` column stands in for the authoritative status
  source, whose real name and values are unknown and must not be guessed.
"""

from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path

from ai_analyst.contracts.opportunity_snapshot_v1 import ALL_COLUMNS

SNAPSHOTS: tuple[date, ...] = (
    date(2025, 1, 1),
    date(2025, 2, 1),
    date(2025, 3, 31),
    date(2025, 4, 1),
    date(2025, 5, 1),
)
N_OPPS = 12

TEXT_COLUMNS: tuple[str, ...] = (
    "ManagerNotes",
    "SENotes",
    "AVPNotes",
    "Channel_Notes",
    "NextStep",
    "Why_Do_Anything",
    "Why_Now",
    "Why_Us",
    "Decision_Criteria",
    "POVNextSteps",
    "Paperwork_Process",
    "RVPNotes",
)

STATUS_COLUMN = "synthetic_status"
STATUS_VALUES = {"open": "O", "won": "W", "lost": "L", "excluded": "DELETED"}

STAGES = (
    "0 - Qualification",
    "1 - Discovery",
    "2 - Solution Design",
    "3 - Proposal",
    "4 - Negotiation",
    "5 - Commit",
    "6 - Order Placed",
)


def fiscal_quarter(day: date) -> str:
    return f"FY{day.year}-Q{(day.month - 1) // 3 + 1}"


def quarter_end(day: date) -> date:
    first = date(day.year, 3 * ((day.month - 1) // 3) + 1, 1)
    nxt = date(first.year + (first.month + 2) // 12, (first.month + 2) % 12 + 1, 1)
    return nxt - timedelta(days=1)


def _created(i: int) -> date:
    return date(2024, 10, 1) + timedelta(days=10 * i)


def _close(i: int, s: int) -> date:
    close = date(2025, 3, 15) + timedelta(days=20 * i)
    if i % 3 == 0 and s >= 2:
        close += timedelta(days=45)  # slips
    return close


def _stage(i: int, s: int) -> str:
    if i == 1 and s >= 2:
        return "Closed Won"
    if i == 2 and s >= 3:
        return "Closed Lost"
    if i == 3 and s == 4:
        return "SFDCDELETED"
    if i == 4 and s == 4:
        return "not available"
    return STAGES[(i + s) % len(STAGES)]


def _status(i: int, s: int) -> str:
    stage = _stage(i, s)
    if stage == "Closed Won":
        return STATUS_VALUES["won"]
    if stage == "Closed Lost":
        return STATUS_VALUES["lost"]
    if stage == "SFDCDELETED":
        return STATUS_VALUES["excluded"]
    if i == 6 and s == 4:
        return "PENDING"  # deliberately absent from any status mapping
    return STATUS_VALUES["open"]


def _amount(i: int, s: int) -> str:
    base = 10_000 * (i + 1) + (5_000 if i == 4 and s >= 2 else 0)
    return f"{base}.00"


def _decimal(i: int, s: int, mod: int, scale: float) -> str:
    return f"{scale * ((i * 7 + s * 3) % mod):.4f}"


def _text(column: str, i: int, s: int) -> str:
    if (i + s) % 4 == 0:
        return ""
    if column == "NextStep":
        return f"Call {i}-{s}"
    return (
        f"{column} entry for opportunity {i} at snapshot {s}. The customer confirmed "
        f"budget and named sponsor number {i * 13 + s}; the remaining gate is review "
        f"round {s + 1} with legal, targeted for the end of the month."
    )


def _value(column: str, i: int, s: int) -> str:
    as_of = SNAPSHOTS[s]
    close = _close(i, s)
    created = _created(i)
    stage = _stage(i, s)
    terminal_fate = {1: "W", 2: "L"}.get(i, "")
    terminal_date = close if terminal_fate else None

    special: dict[str, str] = {
        "opp_id": f"OPP-{i:03d}",
        "as_of": as_of.isoformat(),
        "account_id": f"ACC-{i % 5:02d}",
        "OwnerID": f"U-{100 + i % 4}",
        "RenewalManager": f"RM-{i % 3}",
        "Stage": stage,
        "ForecastCategory": ("Pipeline", "Best Case", "Commit", "Omitted")[(i + s) % 4],
        "Type": ("New Business", "Renewal", "Expansion")[i % 3],
        "pipe_type": ("Direct", "Partner")[i % 2],
        "new_amount": _amount(i, s),
        "Probability": str((10, 25, 50, 75, 90)[(i + s) % 5]),
        "close_date": close.isoformat(),
        "days_to_close": str((close - as_of).days),
        "age": str((as_of - created).days),
        "train": str(i % 2),
        "as_of_qtr": fiscal_quarter(as_of),
        "close_date_qtr": fiscal_quarter(close),
        "terminal_date_qtr": fiscal_quarter(terminal_date) if terminal_date else "",
        "as_of_qtr_plus_1": fiscal_quarter(as_of + timedelta(days=92)),
        "as_of_qtr_plus_2": fiscal_quarter(as_of + timedelta(days=183)),
        "as_of_qtr_plus_3": fiscal_quarter(as_of + timedelta(days=275)),
        "qtr_number": str((as_of.month - 1) // 3 + 1),
        "qtr_segment": ("early", "mid", "late")[min(2, ((as_of.month - 1) % 3))],
        "days_since_boq": str((as_of - date(as_of.year, 3 * ((as_of.month - 1) // 3) + 1, 1)).days),
        "days_to_eoq": str((quarter_end(as_of) - as_of).days),
        # Kept internally consistent with the close date so the reconstruction
        # agreement tests have a genuine positive control. The real export's
        # sign convention for this column is unconfirmed (ARCHITECTURE 12.15).
        "eoq_close_diff": str((quarter_end(as_of) - as_of).days - (close - as_of).days),
        "terminal_date": terminal_date.isoformat() if terminal_date else "",
        "terminal_quarter_eoq": quarter_end(terminal_date).isoformat() if terminal_date else "",
        "terminal_fate": terminal_fate,
        "terminal_amount": f"{10_000 * (i + 1)}.00" if terminal_fate == "W" else "",
        "deal_first_seen_close_date": _close(i, 0).isoformat(),
        "opp_first_commit_as_of": SNAPSHOTS[1].isoformat() if i % 2 == 0 and s >= 1 else "",
        "CD_in_qtr": "true" if fiscal_quarter(close) == fiscal_quarter(as_of) else "false",
        "CD_in_past": "true" if close < as_of else "false",
        "is_pushed_out_deal": str(int(i % 3 == 0 and s >= 2)),
        "is_pulled_in_deal": str(int(i == 5 and s >= 1)),
        "is_qtr_pushed_out_deal": str(int(i % 3 == 0 and s >= 2)),
        "is_qtr_pulled_in_deal": "0",
        "close_date_push_count": str(int(i % 3 == 0 and s >= 2)),
        "close_date_pull_count": str(int(i == 5 and s >= 1)),
        "stage_changes_count": str(s),
        "deal_size_bucket_num": str(1 + i % 5),
        "account_ti_first_won": "-999999" if i % 2 == 0 else str(30 + 10 * s + i),
        "account_ti_first_loss": "-999999" if i % 3 != 0 else str(15 + 5 * s + i),
        "prev_won": str(i % 3),
        "prev_loss": str(i % 2),
        # -999999 in a column that declares no sentinel: it must stay a real value.
        "days_since_pulled_into_qtr": "-999999" if i % 4 == 0 else str(3 * s + i),
    }
    if column in special:
        return special[column]
    if column in TEXT_COLUMNS:
        return _text(column, i, s)
    if column.endswith("_updated_days"):
        return "" if (i + s) % 6 == 0 else str((i * 3 + s * 5) % 40)
    if column.startswith("rep_"):
        return _decimal(i, s, 19, 0.05)
    if column.startswith(("is_", "deal_is_")):
        return str((i + s) % 2)
    if column.endswith("_count"):
        return str((i + s) % 5)
    return _decimal(i, s, 23, 0.125)


def production_headers(
    *, include_close_date: bool = True, status_column: bool = False
) -> list[str]:
    headers = list(ALL_COLUMNS) + list(TEXT_COLUMNS)
    if include_close_date:
        headers.append("close_date")
    if status_column:
        headers.append(STATUS_COLUMN)
    return headers


def production_rows(
    *, include_close_date: bool = True, status_column: bool = False
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for s in range(len(SNAPSHOTS)):
        for i in range(N_OPPS):
            if i == 5 and s > 2:
                continue  # vanishes without a terminal state
            if i == 8 and s < 2:
                continue  # created mid-window
            row = {
                c: _value(c, i, s)
                for c in production_headers(include_close_date=include_close_date)
            }
            if status_column:
                row[STATUS_COLUMN] = _status(i, s)
            rows.append(row)
    return rows


def write_production_csv(
    path: Path,
    *,
    include_close_date: bool = True,
    status_column: bool = False,
    duplicate_first_row: bool = False,
) -> list[str]:
    """Write the fixture as CSV and return its headers."""
    headers = production_headers(
        include_close_date=include_close_date, status_column=status_column
    )
    rows = production_rows(include_close_date=include_close_date, status_column=status_column)
    if duplicate_first_row:
        rows.append(dict(rows[0]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    return headers


# ---------------------------------------------------------------------------
# Excel serial dates (ARCHITECTURE 12.19)
# ---------------------------------------------------------------------------

EXCEL_BASE = date(1899, 12, 30)
# The date columns the export registry declares as Excel serials.
SERIAL_DATE_COLUMNS: tuple[str, ...] = (
    "as_of",
    "terminal_date",
    "terminal_quarter_eoq",
    "deal_first_seen_close_date",
    "opp_first_commit_as_of",
)


def excel_serial(day: date, fraction: str = "") -> str:
    """A date as an Excel serial number, optionally with a time-of-day fraction.

    Computed here from first principles (days since 1899-12-30) rather than by
    calling the code under test, so a conversion bug cannot agree with itself.
    """
    return f"{(day - EXCEL_BASE).days}{fraction}"


def write_serial_production_csv(
    path: Path,
    *,
    include_close_date: bool = False,
    as_of_fraction: str = ".4583333",
    columns: tuple[str, ...] = SERIAL_DATE_COLUMNS,
) -> list[str]:
    """The fixture with its date columns rewritten as Excel serial numbers.

    `as_of` gets a time-of-day fraction, like the real export's, so the
    truncation to a whole day is exercised on every row.
    """
    headers = production_headers(include_close_date=include_close_date)
    rows = production_rows(include_close_date=include_close_date)
    for row in rows:
        for column in columns:
            value = row.get(column, "")
            if value:
                fraction = as_of_fraction if column == "as_of" else ""
                row[column] = excel_serial(date.fromisoformat(value), fraction)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    return headers
