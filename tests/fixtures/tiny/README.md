# Tiny Fixture

Hand-built verification instrument for every later milestone (ARCHITECTURE §9.1).
Small enough that every expected value is provable by reading `snapshots.csv`.

**Shape:** 40 rows, 8 opportunities, 6 snapshots, 2 quarters of snapshots.

Source headers are deliberately non-canonical so the mapping layer is exercised
on the real path.

## Snapshots

| as_of | quarter | rows |
|---|---|---|
| 2025-01-01 | Q1 open | 6 |
| 2025-02-01 | Q1 | 7 |
| 2025-03-31 | Q1 close | 7 |
| 2025-04-01 | Q2 open | 6 |
| 2025-05-01 | Q2 | 7 |
| 2025-06-30 | Q2 close | 7 |

Snapshots span Q1 and Q2 2025. Close dates reach into Q3 2025, which is what
lets a single opportunity slip across three quarters inside a two-quarter
snapshot window.

## The hard cases

| Opportunity | Rows | Case |
|---|---|---|
| OPP-001 | 6 | **Slips twice.** close_date 2025-03-15 (Q1), then 2025-05-15 (Q2) at 2025-03-31, then 2025-08-15 (Q3) at 2025-06-30. |
| OPP-002 | 6 | **Pulled in.** close_date 2025-05-20 (Q2) moves to 2025-03-25 (Q1) at 2025-03-31, and closes won. |
| OPP-003 | 6 | **Segment change.** Mid-Market for the first two snapshots, Enterprise from 2025-03-31. Closes won in Q2. |
| OPP-004 | 6 | **Amount changes.** 200000 to 250000 at 2025-03-31, down to 180000 at 2025-06-30. Still open at the end. |
| OPP-005 | 3 | **Vanishes without a terminal state.** Last seen 2025-03-31 in Discovery. The `other_removed` case. Also the only rows with a null `arr`. |
| OPP-006 | 2 | **Created mid-quarter (Q2).** created 2025-04-20, first appears 2025-05-01. Null `region`. |
| OPP-007 | 6 | **Closed lost** in Q1 at 2025-03-31. |
| OPP-008 | 5 | **Created mid-quarter (Q1).** created 2025-01-15, absent from the first snapshot. Closes won in Q2. |

## Hand-computed expectations

| Quantity | Value | Derivation |
|---|---|---|
| Total rows | 40 | 6+7+7+6+7+7 |
| Distinct opportunities | 8 | OPP-001 through OPP-008 |
| Snapshots | 6 | |
| Median snapshots per opportunity | 6.0 | sorted [2,3,5,6,6,6,6,6], mean of 4th and 5th |
| Present in all 6 snapshots | 5 | OPP-001, 002, 003, 004, 007 |
| Vanished without terminal state | 1 | OPP-005 only |
| `arr` nulls | 3 | OPP-005 across its 3 rows |
| `region` nulls | 2 | OPP-006 across its 2 rows |
| Distinct stages | 6 | Discovery, Qualification, Proposal, Negotiation, Closed Won, Closed Lost |
| `amount` min / max | 40000.00 / 250000.00 | OPP-005 / OPP-004 at 2025-03-31 |
| `close_date` min / max | 2025-02-28 / 2025-09-15 | OPP-007 / OPP-006 |
| `created_date` min / max | 2024-10-01 / 2025-04-20 | OPP-007 / OPP-006 |
| Negative amounts | 0 | |
| close_date before created_date | 0 | |

Stage row counts: Negotiation 10, Discovery 8, Proposal 6, Qualification 6,
Closed Won 6, Closed Lost 4. Sums to 40.

## Negative fixtures

Kept in separate files so the main fixture never fails ingestion.

| File | Triggers |
|---|---|
| `duplicate_key.csv` | `GrainViolation`. Two duplicated keys across 4 offending rows. |
| `missing_required.csv` | `MissingRequiredColumns`. No stage or close date column. |
| `null_key.csv` | `NullGrainKey`. One `as_of` value that will not cast to a date. |
