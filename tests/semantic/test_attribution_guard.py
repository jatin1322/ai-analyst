"""The prospective later-snapshot attribution guard (ARCHITECTURE 13.11 #4).

A dimension or feature is read from the snapshot its attribution rule names
(5.3 ambiguity 5), which need not be the analysis snapshot. Every such read is
classified:

* BACKWARD: read before the analysis snapshot. History; allowed.
* CONTEMPORANEOUS: read at the analysis snapshot. The state then; allowed.
* LATER: read after what a prospective analysis may know. Refused.
* RETROSPECTIVE_TERMINAL: an outcome column. Refused prospectively.

The guard sits in the gate and again in the compiler, which recomputes the
horizon from the spec rather than trusting the recorded relation.

Hand-computed values from `tests/fixtures/tiny/snapshots.csv`. Q1 opening
pipeline at 2025-01-01, open with a close date in Q1:
    OPP-001 100000  Enterprise  stage Negotiation (01-01), Negotiation (03-31)
    OPP-005  40000  SMB         stage Discovery   (01-01), Discovery   (03-31)
    OPP-007  60000  SMB         stage Negotiation (01-01), Closed Lost (03-31)
"""

from __future__ import annotations

import csv
import dataclasses
from datetime import date
from decimal import Decimal

import pytest

from ai_analyst.contracts.columns import Availability
from ai_analyst.contracts.plan import (
    AnalysisPattern,
    AnalysisSpec,
    AnalysisStance,
    Attribution,
    Period,
    PeriodKind,
    SnapshotSelection,
)
from ai_analyst.contracts.rejection import RejectionCode
from ai_analyst.contracts.result import (
    AttributionRead,
    SnapshotRule,
    TemporalRelation,
    TrustFactorKind,
)
from ai_analyst.semantic import compiler as compiler_module
from ai_analyst.semantic import gate as gate_module
from ai_analyst.semantic.compiler import compile_spec
from ai_analyst.semantic.gate import attribution_rejections, relation_of
from ai_analyst.semantic.sql import CompilationError
from tests.semantic.conftest import TINY_CSV, build_engine

Q1 = Period(kind=PeriodKind.FISCAL_QUARTER, label="FY2025-Q1")
PROSP, RETRO = AnalysisStance.PROSPECTIVE, AnalysisStance.RETROSPECTIVE


def spec(spec_id: str = "a", **kwargs) -> AnalysisSpec:
    kwargs.setdefault("pattern", AnalysisPattern.POINT_IN_TIME)
    kwargs.setdefault("period", Q1)
    kwargs.setdefault("metrics", ["opening_pipeline"])
    kwargs.setdefault("snapshot", SnapshotSelection(rule=SnapshotRule.PERIOD_OPEN))
    return AnalysisSpec(id=spec_id, **kwargs)


def by(result, key: str, value: str) -> dict:
    return {result.cell(i, key): result.cell(i, value) for i in range(result.row_count)}


# ============================================================================
# The relation of a read, as a pure function
# ============================================================================


@pytest.mark.parametrize(
    ("read", "availability", "expected"),
    [
        (date(2025, 1, 1), Availability.AS_OF_FACT, TemporalRelation.BACKWARD),
        (date(2025, 3, 31), Availability.AS_OF_FACT, TemporalRelation.CONTEMPORANEOUS),
        (date(2025, 4, 1), Availability.AS_OF_FACT, TemporalRelation.LATER),
        (date(2025, 1, 1), Availability.BACKWARD_DERIVED, TemporalRelation.BACKWARD),
        (date(2025, 1, 1), Availability.FUTURE_CONTAMINATED,
         TemporalRelation.RETROSPECTIVE_TERMINAL),
    ],
)
def test_a_read_is_classified_relative_to_the_horizon(read, availability, expected):
    assert relation_of(read, date(2025, 3, 31), availability) is expected


def test_only_backward_and_contemporaneous_reads_are_prospectively_safe():
    safe = {r for r in TemporalRelation if r.safe_for_prospective}
    assert safe == {TemporalRelation.BACKWARD, TemporalRelation.CONTEMPORANEOUS}


# ============================================================================
# 1. Valid same-snapshot attribution
# ============================================================================


def test_same_snapshot_attribution_is_contemporaneous_and_runs(tiny):
    outcome = tiny.gate(spec(dimensions=["segment"], stance=PROSP))
    assert outcome.ok
    (read,) = outcome.specs["a"].attributions
    assert read.relation is TemporalRelation.CONTEMPORANEOUS
    assert read.read_as_of == read.horizon == date(2025, 1, 1)

    result = tiny.one(spec(dimensions=["segment"], stance=PROSP))
    # Enterprise: OPP-001 100000. SMB: OPP-005 40000 + OPP-007 60000 = 100000.
    assert by(result, "segment", "opening_pipeline") == {
        "Enterprise": Decimal("100000.00"),
        "SMB": Decimal("100000.00"),
    }


def test_backward_attribution_on_an_as_of_column_is_allowed(tiny):
    # Ending pipeline at 03-31 attributed to owner at the period open (01-01).
    # Open at 03-31 with close in Q1: OPP-005 40000, owner U-104 at 01-01.
    outcome = tiny.gate(spec(metrics=["ending_pipeline"], dimensions=["owner"],
                             snapshot=SnapshotSelection(rule=SnapshotRule.PERIOD_CLOSE)))
    assert outcome.ok
    (read,) = outcome.specs["a"].attributions
    assert read.relation is TemporalRelation.BACKWARD
    assert (read.read_as_of, read.horizon) == (date(2025, 1, 1), date(2025, 3, 31))
    result = tiny.one(spec(metrics=["ending_pipeline"], dimensions=["owner"],
                           snapshot=SnapshotSelection(rule=SnapshotRule.PERIOD_CLOSE)))
    assert by(result, "owner_id", "ending_pipeline") == {"U-104": Decimal("40000.00")}


# ============================================================================
# 2. Valid backward-derived feature
# ============================================================================


def test_a_backward_derived_feature_read_before_the_horizon_is_allowed(production):
    outcome = production.gate(
        spec(metrics=["deal_count"], dimensions=["prev_won"], stance=PROSP,
             snapshot=SnapshotSelection(rule=SnapshotRule.PERIOD_CLOSE))
    )
    assert outcome.ok, [r.message for r in outcome.validation.rejections]
    (read,) = outcome.specs["a"].attributions
    assert read.column == "prev_won"
    assert read.relation is TemporalRelation.BACKWARD


# ============================================================================
# 3. Invalid later-snapshot dimension
# ============================================================================


def test_a_later_snapshot_dimension_is_refused_and_named(tiny):
    outcome = tiny.gate(spec(dimensions=["segment"], attribution=Attribution.AT_CLOSE,
                             stance=PROSP))
    assert not outcome.ok
    (rejection,) = [r for r in outcome.validation.rejections if r.field == "attribution"]
    assert rejection.code is RejectionCode.STANCE_VIOLATION
    assert rejection.column == "segment"
    # The field, the attempted snapshot and the allowed horizon, all named.
    assert "'segment'" in rejection.message
    assert "2025-03-31" in rejection.message  # attempted
    assert "nothing after 2025-01-01" in rejection.message  # horizon
    assert rejection.remedy


def test_a_knowledge_cutoff_narrows_the_horizon(tiny):
    # Ending pipeline at 03-31, cutoff 03-31 but attribution LATEST (06-30).
    outcome = tiny.gate(spec(metrics=["ending_pipeline"], dimensions=["owner"],
                             attribution=Attribution.LATEST, stance=PROSP,
                             knowledge_cutoff=date(2025, 3, 31),
                             snapshot=SnapshotSelection(rule=SnapshotRule.PERIOD_CLOSE)))
    (rejection,) = [r for r in outcome.validation.rejections if r.field == "attribution"]
    assert "2025-06-30" in rejection.message
    assert "nothing after 2025-03-31" in rejection.message


# ============================================================================
# 4. Invalid later-snapshot outcome
# ============================================================================


def test_a_later_stage_that_reveals_the_outcome_is_refused(tiny):
    # Stage at 03-31 reads OPP-007 as 'Closed Lost': the quarter's outcome,
    # attributed onto the quarter's opening pipeline.
    outcome = tiny.gate(spec(dimensions=["stage"], attribution=Attribution.AT_CLOSE,
                             stance=PROSP))
    assert outcome.validation.has(RejectionCode.STANCE_VIOLATION)


def test_an_outcome_column_is_refused_prospectively(production):
    outcome = production.gate(
        spec(metrics=["deal_count"], dimensions=["terminal_fate"], stance=PROSP,
             snapshot=SnapshotSelection(rule=SnapshotRule.PERIOD_CLOSE))
    )
    assert outcome.validation.has(RejectionCode.COLUMN_NOT_KNOWABLE_AT_SNAPSHOT)


def test_the_guard_itself_refuses_a_terminal_read_even_at_an_earlier_snapshot():
    # Defence in depth: the column resolver refuses an outcome column before
    # the guard sees it, so exercise the guard directly. A terminal value is
    # hindsight whichever snapshot it is read from.
    read = AttributionRead(
        field="fate", column="terminal_fate", rule="period_open",
        read_as_of=date(2025, 1, 1), horizon=date(2025, 3, 31),
        relation=TemporalRelation.RETROSPECTIVE_TERMINAL,
    )
    (rejection,) = attribution_rejections(spec(stance=PROSP), [read])
    assert rejection.code is RejectionCode.STANCE_VIOLATION
    assert "retrospective outcome" in rejection.message


# ============================================================================
# 5. Retrospective analysis is still allowed
# ============================================================================


def test_retrospective_later_attribution_is_allowed_and_disclosed(tiny):
    result = tiny.one(spec(dimensions=["stage"], attribution=Attribution.AT_CLOSE,
                           stance=RETRO))
    # Stage at 03-31 for the 01-01 opening pipeline:
    #   OPP-001 Negotiation 100000, OPP-005 Discovery 40000, OPP-007 Closed Lost 60000.
    assert by(result, "stage", "opening_pipeline") == {
        "Negotiation": Decimal("100000.00"),
        "Discovery": Decimal("40000.00"),
        "Closed Lost": Decimal("60000.00"),
    }
    (read,) = result.compilation.attributions
    assert read.relation is TemporalRelation.LATER
    assert read.horizon is None


def test_retrospective_outcome_dimension_is_allowed_and_disclosed(production):
    result = production.one(
        spec(metrics=["deal_count"], dimensions=["terminal_fate"], stance=RETRO,
             snapshot=SnapshotSelection(rule=SnapshotRule.PERIOD_CLOSE))
    )
    (read,) = result.compilation.attributions
    assert read.relation is TemporalRelation.RETROSPECTIVE_TERMINAL
    assert TrustFactorKind.RETROSPECTIVE_READ in {f.kind for f in result.trust.factors}


# ============================================================================
# 6. Mutation: removing the guard
# ============================================================================


def test_mutation_removing_the_gate_guard_leaves_the_compiler_guard(tiny, monkeypatch):
    monkeypatch.setattr(gate_module, "attribution_rejections", lambda spec, reads: [])
    later = spec(dimensions=["segment"], attribution=Attribution.AT_CLOSE, stance=PROSP)
    outcome = tiny.gate(later)
    assert outcome.ok  # the gate layer is gone
    with pytest.raises(CompilationError, match="prospective attribution guard"):
        compile_spec(tiny.scan, outcome.specs["a"])


def test_mutation_removing_both_guards_leaks_a_later_value(tiny, monkeypatch):
    """The control: with both layers removed the later value really is read.

    OPP-003 moved from Mid-Market to Enterprise at 03-31, but is not in Q1
    opening pipeline; OPP-007's stage became 'Closed Lost'. Without the guards
    the prospective answer names an outcome, which is the leak they prevent.
    """
    monkeypatch.setattr(gate_module, "attribution_rejections", lambda spec, reads: [])
    monkeypatch.setattr(compiler_module, "check_attribution", lambda validated: None)
    result = tiny.one(spec(dimensions=["stage"], attribution=Attribution.AT_CLOSE,
                           stance=PROSP))
    assert "Closed Lost" in by(result, "stage", "opening_pipeline")


# ============================================================================
# 7. The guard survives compilation
# ============================================================================


def test_the_compiled_sql_joins_the_attribution_snapshot(tiny):
    result = tiny.one(spec(metrics=["ending_pipeline"], dimensions=["owner"],
                           snapshot=SnapshotSelection(rule=SnapshotRule.PERIOD_CLOSE)))
    assert "DATE '2025-01-01'" in result.compiled_sql
    (read,) = result.compilation.attributions
    assert read.read_as_of == date(2025, 1, 1)


def test_a_tampered_validated_spec_cannot_compile_a_later_read(tiny):
    outcome = tiny.gate(spec(dimensions=["segment"], stance=PROSP))
    validated = outcome.specs["a"]
    (read,) = validated.attributions
    # Rewrite the spec to read at close, and forge the recorded read as safe.
    forged = read.model_copy(
        update={"read_as_of": date(2025, 3, 31), "relation": TemporalRelation.CONTEMPORANEOUS}
    )
    tampered = dataclasses.replace(
        validated,
        spec=validated.spec.model_copy(update={"attribution": Attribution.AT_CLOSE}),
        attributions=[forged],
    )
    with pytest.raises(CompilationError, match="after the horizon of 2025-01-01"):
        compile_spec(tiny.scan, tampered)


def test_a_prospective_attributed_answer_does_not_change_when_the_future_changes(
    tiny, tmp_path
):
    """The plan-path twin of the investigation leakage test: rewrite every
    snapshot after the horizon and require the prospective answer unchanged."""
    future = tmp_path / "future.csv"
    reader = list(csv.DictReader(TINY_CSV.open(encoding="utf-8")))
    for row in reader:
        if row["snapshot_date"] > "2025-01-01":
            row["customer_segment"] = "Rewritten"
            row["sales_stage"] = "Closed Lost"
            row["owner"] = "U-999"
    with future.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(reader[0]))
        writer.writeheader()
        writer.writerows(reader)
    rewritten = build_engine(future, "future", tmp_path / "f")

    question = spec(dimensions=["segment", "owner"], stance=PROSP)
    assert tiny.one(question).rows == rewritten.one(question).rows
