"""An independent reference implementation for the real-export acceptance suite.

It deliberately imports nothing from `ai_analyst`. Every expected value in
`test_real_export.py` comes from here: plain DuckDB SQL over the raw parquet,
written from the stated semantics rather than from the engine's code, so a bug
in the engine cannot also be a bug in its expected answer.

Stated semantics, each one written down rather than inherited:

* Snapshot date: the export's `as_of` is an Excel serial with a time of day;
  the date is `1899-12-30 + FLOOR(as_of)` days (1900 date system).
* Close date: `snapshot date + days_to_close` days. Declared derivation.
* Amount: `new_amount`, read as text and cast to DECIMAL(18,2) before any
  aggregate, which is the project's declared ingestion rule (source data is read
  as text and cast at conformance). The route matters on real data: the sibling
  export has 1,576 amounts with more than two decimal places, and on 7 of them
  rounding the DOUBLE directly lands a cent away from rounding its text form.
  Which rounding the tenant intends is an open question
  (`sub_cent_amounts`); this reference follows the declared rule.
* Status: **inferred from the stage label, and not authoritative.** Null or an
  invalid label is excluded; a label containing a lost keyword is lost; one
  containing "won" or "win" is won; anything else is open. This mirrors the
  documented last-resort rule; it does not resolve which production column is
  the authoritative status, which remains an open convention.
* Pipeline at a snapshot: open rows whose close date falls in the period.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import duckdb

INVALID = ("sfdcdeleted", "not available", "n/a", "unknown", "")
LOST = ("lost", "disqualif", "no decision", "abandoned", "cancelled", "canceled", "dead")
WON = ("won", "win")


def _like_any(expr: str, words: tuple[str, ...]) -> str:
    return " OR ".join(f"{expr} LIKE '%{w}%'" for w in words)


class Reference:
    """Raw-SQL answers over one export file."""

    def __init__(self, path: str) -> None:
        self.con = duckdb.connect()
        stage = "LOWER(TRIM(CAST(Stage AS VARCHAR)))"
        invalid = ", ".join(f"'{v}'" for v in INVALID if v)
        self.con.execute(
            f"""
            CREATE VIEW rows AS
            SELECT
                DATE '1899-12-30' + CAST(FLOOR(as_of) AS INTEGER) AS d,
                CAST(opp_id AS VARCHAR) AS opp,
                CAST(CAST(new_amount AS VARCHAR) AS DECIMAL(18,2)) AS amount,
                DATE '1899-12-30' + CAST(FLOOR(as_of) AS INTEGER)
                    + CAST(days_to_close AS INTEGER) AS close_date,
                CAST(OwnerID AS VARCHAR) AS owner,
                CAST(OpportunityOwnerDivision AS VARCHAR) AS division,
                CASE
                    WHEN {stage} IS NULL OR {stage} IN ({invalid}) THEN 'excluded'
                    WHEN {_like_any(stage, LOST)} THEN 'lost'
                    WHEN {_like_any(stage, WON)} THEN 'won'
                    ELSE 'open'
                END AS status
            FROM read_parquet('{path}')
            """
        )

    def snapshots(self) -> list[date]:
        return [r[0] for r in self.con.execute("SELECT DISTINCT d FROM rows ORDER BY 1").fetchall()]

    def _one(self, sql: str):
        return self.con.execute(sql).fetchone()

    def _pipeline(self, at: date, start: date, end: date) -> str:
        return (
            f"SELECT opp, amount, close_date FROM rows WHERE d = DATE '{at}' "
            f"AND status = 'open' AND close_date BETWEEN DATE '{start}' AND DATE '{end}'"
        )

    def pipeline(self, at: date, start: date, end: date) -> Decimal:
        rows = self._pipeline(at, start, end)
        (total,) = self._one(f"SELECT COALESCE(SUM(amount), 0) FROM ({rows})")
        return Decimal(total)

    def pipeline_by(self, column: str, at: date, start: date, end: date) -> dict:
        rows = self.con.execute(
            f"SELECT {column}, SUM(amount) FROM rows WHERE d = DATE '{at}' AND status = 'open' "
            f"AND close_date BETWEEN DATE '{start}' AND DATE '{end}' GROUP BY 1"
        ).fetchall()
        return {k: Decimal(v) for k, v in rows}

    def entrants(self, a: date, b: date, start: date, end: date) -> tuple[Decimal, Decimal]:
        """(created, pulled_in): in the closing pipeline, not the opening one.

        Created means first seen in any snapshot inside the period; everything
        else that entered was pulled in.
        """
        inside = f"fs.f BETWEEN DATE '{start}' AND DATE '{end}'"
        created, pulled = self._one(
            f"""
            WITH pa AS ({self._pipeline(a, start, end)}),
                 pb AS ({self._pipeline(b, start, end)}),
                 first_seen AS (SELECT opp, MIN(d) AS f FROM rows GROUP BY 1)
            SELECT
                COALESCE(SUM(pb.amount) FILTER (WHERE {inside}), 0),
                COALESCE(SUM(pb.amount) FILTER (WHERE NOT ({inside})), 0)
            FROM pb JOIN first_seen fs USING (opp)
            WHERE pb.opp NOT IN (SELECT opp FROM pa)
            """
        )
        return Decimal(created), Decimal(pulled)

    def slipped(self, a: date, b: date, start: date, end: date) -> Decimal:
        """Left the opening pipeline, never won or lost in (a, b], close now after the period."""
        (total,) = self._one(
            f"""
            WITH pa AS ({self._pipeline(a, start, end)}),
                 pb AS ({self._pipeline(b, start, end)}),
                 later AS (
                     SELECT opp,
                            BOOL_OR(status = 'won') AS won, BOOL_OR(status = 'lost') AS lost,
                            MAX(close_date) FILTER (WHERE d = DATE '{b}') AS b_close
                     FROM rows WHERE d > DATE '{a}' AND d <= DATE '{b}' GROUP BY 1)
            SELECT COALESCE(SUM(pa.amount), 0)
            FROM pa JOIN later USING (opp)
            WHERE pa.opp NOT IN (SELECT opp FROM pb)
              AND NOT later.won AND NOT later.lost AND later.b_close > DATE '{end}'
            """
        )
        return Decimal(total)

    def win_rate(self, at: date, start: date, end: date) -> tuple[int, int]:
        """(won, won + lost) among closed rows whose close date falls in the period."""
        won, closed = self._one(
            f"SELECT COUNT(*) FILTER (WHERE status = 'won'), "
            f"COUNT(*) FILTER (WHERE status IN ('won', 'lost')) FROM rows "
            f"WHERE d = DATE '{at}' AND close_date BETWEEN DATE '{start}' AND DATE '{end}'"
        )
        return won, closed

    def cohort_fates(self, a: date, b: date) -> dict[str, tuple[int, Decimal]]:
        """Open at a, classified at b; amount as of a."""
        rows = self.con.execute(
            f"""
            WITH cohort AS (SELECT opp, amount FROM rows WHERE d = DATE '{a}' AND status = 'open'),
                 w AS (
                     SELECT opp,
                            BOOL_OR(status = 'won') AS won, BOOL_OR(status = 'lost') AS lost,
                            BOOL_OR(status = 'excluded') AS excluded,
                            BOOL_OR(status = 'open' AND d = DATE '{b}') AS open_at_b,
                            MAX(d) AS last_seen
                     FROM rows WHERE d BETWEEN DATE '{a}' AND DATE '{b}' GROUP BY 1)
            SELECT CASE WHEN w.won THEN 'won' WHEN w.lost THEN 'lost'
                        WHEN w.excluded THEN 'excluded'
                        WHEN w.last_seen >= DATE '{b}' AND w.open_at_b THEN 'open'
                        WHEN w.last_seen >= DATE '{b}' THEN 'unknown'
                        ELSE 'vanished' END AS state,
                   COUNT(*), SUM(cohort.amount)
            FROM cohort LEFT JOIN w USING (opp) GROUP BY 1
            """
        ).fetchall()
        return {state: (n, Decimal(total)) for state, n, total in rows}

    def close_date_changed(self, a: date, b: date) -> dict[str, int]:
        """The cohort open at a, by whether its close date differs at b."""
        rows = self.con.execute(
            f"""
            SELECT CASE WHEN x.close_date IS DISTINCT FROM y.close_date THEN 'true'
                        ELSE 'false' END, COUNT(*)
            FROM rows x JOIN rows y ON x.opp = y.opp AND y.d = DATE '{b}'
            WHERE x.d = DATE '{a}' AND x.status = 'open' GROUP BY 1
            """
        ).fetchall()
        return dict(rows)
