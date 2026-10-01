"""The dataset-scoped registry contract and its requirement-check surface."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ai_analyst.contracts.columns import (
    Availability,
    ColumnCategory,
    ColumnClassification,
    Disposition,
    FeatureLineage,
    QuarantineCode,
    QuarantineReason,
    UnresolvedItem,
)
from ai_analyst.contracts.dataset import (
    ColumnOrigin,
    DatasetColumn,
    DatasetRegistry,
    RequirementState,
)
from ai_analyst.contracts.schema import DataType

REASON = QuarantineReason(
    code=QuarantineCode.UNSTATED_REFERENCE_POPULATION,
    detail="population unstated",
    resolution="document it",
)


def _classification(name: str, **overrides) -> ColumnClassification:
    base = {
        "name": name,
        "category": ColumnCategory.SNAPSHOT_STATE,
        "availability": Availability.AS_OF_FACT,
        "disposition": Disposition.DIRECT,
    }
    return ColumnClassification(**{**base, **overrides})


def _column(name: str, **overrides) -> DatasetColumn:
    classification = overrides.pop("classification", None) or _classification(name)
    return DatasetColumn(
        name=name,
        origin=ColumnOrigin.DISCOVERED,
        dtype=DataType.DOUBLE,
        classification=classification,
        **overrides,
    )


def _registry() -> DatasetRegistry:
    return DatasetRegistry(
        dataset_id="d",
        columns=[
            _column("fine"),
            _column(
                "withheld",
                classification=_classification(
                    "withheld", disposition=Disposition.QUARANTINE, quarantine=REASON
                ),
            ),
            _column(
                "outcome",
                classification=_classification(
                    "outcome",
                    category=ColumnCategory.OUTCOME,
                    availability=Availability.FUTURE_CONTAMINATED,
                    disposition=Disposition.RECOMPUTE,
                ),
            ),
            _column(
                "caveated",
                classification=_classification(
                    "caveated",
                    availability=Availability.BACKWARD_DERIVED,
                    disposition=Disposition.USE_WITH_PROOF,
                    lineage=FeatureLineage(lookback="unconfirmed"),
                ),
            ),
            _column("listed"),
        ],
        unresolved=[
            UnresolvedItem(id="q1", subject="s", question="?", columns=("listed",)),
            UnresolvedItem(id="done", subject="s", question="?", columns=("fine",), resolved=True),
        ],
    )


def test_a_quarantined_column_must_state_its_reason():
    with pytest.raises(ValidationError, match="must state its quarantine reason"):
        _classification("x", disposition=Disposition.QUARANTINE)


def test_a_reason_is_only_valid_on_a_quarantined_column():
    with pytest.raises(ValidationError, match="only valid on a QUARANTINE column"):
        _classification("x", quarantine=REASON)


def test_registry_rejects_duplicate_columns():
    with pytest.raises(ValidationError, match="registered more than once"):
        DatasetRegistry(dataset_id="d", columns=[_column("a"), _column("a")])


def test_a_column_cannot_carry_another_columns_classification():
    with pytest.raises(ValidationError, match="not 'b'"):
        DatasetColumn(
            name="b",
            origin=ColumnOrigin.DISCOVERED,
            dtype=DataType.DOUBLE,
            classification=_classification("a"),
        )


def test_requirement_check_reports_every_state():
    check = _registry().check_requirements(
        ["fine", "withheld", "outcome", "caveated", "listed", "absent"]
    )
    states = {r.name: r.state for r in check.requirements}
    assert states == {
        "fine": RequirementState.AVAILABLE,
        "withheld": RequirementState.QUARANTINED,
        "outcome": RequirementState.NOT_KNOWABLE_AT_SNAPSHOT,
        "caveated": RequirementState.AVAILABLE_WITH_CAVEATS,
        "listed": RequirementState.AVAILABLE_WITH_CAVEATS,
        "absent": RequirementState.MISSING,
    }


def test_quarantine_reason_travels_with_the_requirement():
    check = _registry().check_requirements(["withheld"])
    (requirement,) = check.requirements
    assert requirement.quarantine == REASON
    assert requirement.blockers == ["quarantined:unstated_reference_population"]


def test_caveats_name_the_unresolved_item_and_unconfirmed_lineage():
    check = _registry().check_requirements(["caveated", "listed"])
    by_name = {r.name: r for r in check.requirements}
    assert by_name["caveated"].caveats == ["lineage_unconfirmed"]
    assert by_name["listed"].caveats == ["q1"]


def test_a_resolved_item_is_no_longer_a_caveat():
    assert _registry().check_requirements(["fine"]).requirements[0].caveats == []


def test_prospective_and_retrospective_satisfiability_differ():
    registry = _registry()
    hindsight = registry.check_requirements(["fine", "outcome"])
    assert not hindsight.satisfiable_prospectively
    assert hindsight.satisfiable_retrospectively
    assert hindsight.not_knowable == ["outcome"]


def test_quarantine_blocks_both_stances_and_missing_blocks_both():
    registry = _registry()
    for names in (["withheld"], ["absent"]):
        check = registry.check_requirements(names)
        assert not check.satisfiable_prospectively
        assert not check.satisfiable_retrospectively


def test_caveated_columns_remain_satisfiable():
    # Caveats inform the next layer. They do not withhold the column.
    check = _registry().check_requirements(["caveated"])
    assert check.satisfiable_prospectively
    assert check.caveated == ["caveated"]


def test_registry_selectors():
    registry = _registry()
    assert [c.name for c in registry.quarantined()] == ["withheld"]
    assert "withheld" not in {c.name for c in registry.prospectively_usable()}
    assert [u.id for u in registry.open_unresolved()] == ["q1"]
    assert [u.id for u in registry.unresolved_for("listed")] == ["q1"]
    assert registry.unresolved_for("fine") == []


def test_knowable_at_snapshot_is_independent_of_quarantine():
    withheld = _classification("w", disposition=Disposition.QUARANTINE, quarantine=REASON)
    assert withheld.knowable_at_snapshot
    assert not withheld.usable_prospectively
    assert withheld.prospective_blockers == ["quarantined:unstated_reference_population"]


def test_registry_round_trips_through_json():
    registry = _registry()
    assert DatasetRegistry.model_validate_json(registry.model_dump_json()) == registry
