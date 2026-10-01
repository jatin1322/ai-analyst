"""Reconstruction verdicts (ARCHITECTURE 13.1).

A rebuilt close date is judged by what its evidence can actually say:

* the derivation must be **declared** by an accountable source, or it is
  `UNDECLARED` and unavailable;
* a **validity** test (the structural check, the discriminating movement test,
  or an ancillary check whose semantics a tenant declared) that fails
  **contradicts** it, and it is unavailable;
* a validity test that could not run, or a **reconciliation** test against an
  undocumented ancillary field that fails, leaves it usable **with warnings**,
  which cap trust at B;
* otherwise it is `VALID`.

The previous rule required every test to pass. That made the reconstruction's
validity hostage to fields whose conventions nobody had written down, and on the
real sibling export it withheld a date the evidence did not contradict. These
tests pin the new rule, including the parts that must stay strict.
"""

from __future__ import annotations

import csv

import pytest

from ai_analyst.config import Settings
from ai_analyst.contracts.agreement import (
    AgreementKind,
    AgreementReport,
    AgreementResult,
    AgreementRole,
)
from ai_analyst.contracts.binding import BindingStatus, ReconstructionVerdict
from ai_analyst.contracts.concepts import BusinessConcept
from ai_analyst.contracts.opportunity_snapshot_v1 import OPPORTUNITY_SNAPSHOT_V1
from ai_analyst.contracts.schema import CanonicalColumn
from ai_analyst.contracts.tenant import TenantProfile
from ai_analyst.data.binding import build_bindings
from ai_analyst.data.dataset import load_dataset, register_dataset
from ai_analyst.data.ingest import ingest
from ai_analyst.data.store import DuckDBStore
from ai_analyst.data.understanding import understand
from tests.fixtures.production_shape import write_production_csv

CLOSE = BusinessConcept.EXPECTED_CLOSE_DATE
V, R = AgreementRole.VALIDITY, AgreementRole.RECONCILIATION


def _export(tmp_path, *, drop=(), edit=None, name="export.csv"):
    """The production-shaped fixture with no close_date column, optionally trimmed."""
    path = tmp_path / name
    write_production_csv(path, include_close_date=False)
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    keep = [c for c in rows[0] if c not in drop]
    for index, row in enumerate(rows):
        if edit:
            edit(index, row)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keep, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _dataset(tmp_path, settings, **kwargs):
    return register_dataset(
        _export(tmp_path, **kwargs),
        "gated",
        column_registry=OPPORTUNITY_SNAPSHOT_V1,
        settings=settings,
    )


def _understood(dataset, settings, tenant=None):
    return understand(dataset, tenant=tenant, settings=settings)


def _close_binding(dataset, settings, tenant=None):
    return _understood(dataset, settings, tenant).bindings.get(CLOSE)


def _verdict(dataset, settings, tenant=None):
    bindings = _understood(dataset, settings, tenant).bindings
    return next(v for v in bindings.reconstruction_verdicts if v.concept is CLOSE)


def _corrupt_close_date_qtr(index: int, row: dict[str, str]) -> None:
    if index < 3:
        row["close_date_qtr"] = "FY1999-Q1"


def _corrupt_eoq(index: int, row: dict[str, str]) -> None:
    if index < 3:
        row["eoq_close_diff"] = "123456"


def _push_every_snapshot(index: int, row: dict[str, str]) -> None:
    """A counter that rises at every snapshot, so a stable close date disagrees."""
    row["close_date_push_count"] = str(sum(int(c) for c in row["as_of"] if c.isdigit()))


def _fractional(index: int, row: dict[str, str]) -> None:
    if index == 0:
        row["days_to_close"] = "2.6"


# -- VALID ----------------------------------------------------------------


def test_every_test_passing_is_valid_and_confirmed(tmp_path, settings: Settings):
    dataset = _dataset(tmp_path, settings)
    assessment = _verdict(dataset, settings)
    assert assessment.verdict is ReconstructionVerdict.VALID
    assert assessment.caveats == ()
    binding = _close_binding(dataset, settings)
    assert binding.status is BindingStatus.CONFIRMED
    assert not [c for c in binding.caveats if c.startswith("reconc")]


# -- VALID_WITH_WARNINGS ----------------------------------------------------


def test_a_reconciliation_failure_warns_and_does_not_withhold(tmp_path, settings: Settings):
    dataset = _dataset(tmp_path, settings, edit=_corrupt_close_date_qtr)
    result = dataset.agreement.by_id()["close_date_matches_close_date_qtr"]
    assert result.failed and result.role is R
    assessment = _verdict(dataset, settings)
    assert assessment.verdict is ReconstructionVerdict.VALID_WITH_WARNINGS
    assert assessment.reconciliation_warnings == ("close_date_matches_close_date_qtr",)
    binding = _close_binding(dataset, settings)
    assert binding.status is BindingStatus.CONFIRMED
    assert "reconciliation_warning:close_date_matches_close_date_qtr" in binding.caveats


def test_an_eoq_disagreement_carries_its_offset_histogram(tmp_path, settings: Settings):
    dataset = _dataset(tmp_path, settings, edit=_corrupt_eoq)
    result = dataset.agreement.by_id()["close_date_matches_eoq_close_diff"]
    assert result.failed and result.role is R
    # The histogram is diagnostic, reported, and never a reason to pass.
    assert result.diagnostics
    assert sum(result.diagnostics.values()) >= result.disagreeing_rows
    assert _verdict(dataset, settings).verdict is ReconstructionVerdict.VALID_WITH_WARNINGS


def test_an_unrunnable_movement_test_is_unverified_not_failed(tmp_path, settings: Settings):
    dataset = _dataset(tmp_path, settings, drop=("close_date_push_count",))
    movement = dataset.agreement.by_id()["close_date_moves_for_pushed_deals"]
    assert movement.skipped and movement.role is V
    assessment = _verdict(dataset, settings)
    assert assessment.verdict is ReconstructionVerdict.VALID_WITH_WARNINGS
    assert assessment.unverified == ("close_date_moves_for_pushed_deals",)
    binding = _close_binding(dataset, settings)
    assert binding.status is BindingStatus.CONFIRMED
    assert "reconstruction_unverified:close_date_moves_for_pushed_deals" in binding.caveats


def test_a_skipped_reconciliation_test_changes_nothing(tmp_path, settings: Settings):
    dataset = _dataset(tmp_path, settings, drop=("close_date_qtr",))
    assert dataset.agreement.by_id()["close_date_matches_close_date_qtr"].skipped
    assert _verdict(dataset, settings).verdict is ReconstructionVerdict.VALID


# -- CONTRADICTED -----------------------------------------------------------


def test_a_failed_movement_test_contradicts(tmp_path, settings: Settings):
    dataset = _dataset(tmp_path, settings, edit=_push_every_snapshot)
    movement = dataset.agreement.by_id()["close_date_moves_for_pushed_deals"]
    assert movement.failed
    assessment = _verdict(dataset, settings)
    assert assessment.verdict is ReconstructionVerdict.CONTRADICTED
    assert assessment.contradicted == ("close_date_moves_for_pushed_deals",)
    binding = _close_binding(dataset, settings)
    assert binding.status is BindingStatus.UNAVAILABLE
    assert binding.columns == ()
    assert "contradicted by close_date_moves_for_pushed_deals" in binding.note


def test_a_structural_failure_contradicts(tmp_path, settings: Settings):
    dataset = _dataset(tmp_path, settings, edit=_fractional)
    structural = dataset.agreement.by_id()["close_date_is_structurally_valid"]
    assert structural.failed and structural.disagreeing_rows == 1
    assert _verdict(dataset, settings).verdict is ReconstructionVerdict.CONTRADICTED
    assert _close_binding(dataset, settings).status is BindingStatus.UNAVAILABLE


def test_a_declared_ancillary_semantic_promotes_its_test_to_validity(
    tmp_path, settings: Settings
):
    dataset = _dataset(tmp_path, settings, edit=_corrupt_eoq)
    tenant = TenantProfile(
        tenant_id="t",
        agreement_declarations={
            "close_date_matches_eoq_close_diff": "eoq_close_diff = days_to_eoq - days_to_close"
        },
    )
    assessment = _verdict(dataset, settings, tenant)
    assert assessment.verdict is ReconstructionVerdict.CONTRADICTED
    assert _close_binding(dataset, settings, tenant).status is BindingStatus.UNAVAILABLE


def test_a_calendar_dependent_test_cannot_be_promoted_while_the_calendar_is_unresolved(
    tmp_path, settings: Settings
):
    dataset = _dataset(tmp_path, settings, edit=_corrupt_close_date_qtr)
    declaration = {"close_date_matches_close_date_qtr": "label = fiscal quarter of close"}
    unresolved = TenantProfile(tenant_id="t", agreement_declarations=declaration)
    assessment = _verdict(dataset, settings, unresolved)
    assert assessment.verdict is ReconstructionVerdict.VALID_WITH_WARNINGS
    assert "close_date_matches_close_date_qtr" in assessment.demoted

    resolved = TenantProfile(
        tenant_id="t", agreement_declarations=declaration, fiscal_year_start_month=1
    )
    assert _verdict(dataset, settings, resolved).verdict is ReconstructionVerdict.CONTRADICTED


# -- UNDECLARED -------------------------------------------------------------


def test_an_undeclared_reconstruction_is_unavailable(tmp_path, settings: Settings):
    dataset = _dataset(tmp_path, settings)
    schema = dataset.schema.model_copy(
        update={
            "derived_columns": [
                d.model_copy(update={"declared_by": None}) if d.is_reconstructed else d
                for d in dataset.schema.derived_columns
            ]
        }
    )
    bindings = build_bindings(
        schema, dataset.registry, dataset.profile, agreement=dataset.agreement
    )
    binding = bindings.get(CLOSE)
    assert binding.status is BindingStatus.UNAVAILABLE
    assert "undeclared" in binding.note
    verdict = next(v for v in bindings.reconstruction_verdicts if v.concept is CLOSE)
    assert verdict.verdict is ReconstructionVerdict.UNDECLARED


def test_the_registry_records_who_declared_the_derivation(tmp_path, settings: Settings):
    dataset = _dataset(tmp_path, settings)
    derived = {d.column: d for d in dataset.schema.derived_columns}[CanonicalColumn.CLOSE_DATE]
    assert derived.declared_by == f"export_registry:{OPPORTUNITY_SNAPSHOT_V1.name}"


def test_a_reconstruction_with_no_report_at_all_is_withheld(tmp_path, settings: Settings):
    dataset = _dataset(tmp_path, settings)
    bindings = build_bindings(dataset.schema, dataset.registry, dataset.profile, agreement=None)
    binding = bindings.get(CLOSE)
    assert binding.status is BindingStatus.UNAVAILABLE
    assert "no agreement tests were run" in binding.note


# -- the verdict function, directly -------------------------------------------


def _result(test_id, role, *, checked=10, disagree=0, skipped=False, calendar=False):
    return AgreementResult(
        test_id=test_id,
        assertion="a",
        kind=AgreementKind.COLUMNS_SATISFY_RELATIONSHIP,
        concept=CLOSE,
        role=role,
        calendar_dependent=calendar,
        checked_rows=0 if skipped else checked,
        disagreeing_rows=disagree,
        skipped=skipped,
        skip_reason="absent" if skipped else "",
    )


@pytest.mark.parametrize(
    ("results", "declared", "expected"),
    [
        ((_result("s", V),), True, ReconstructionVerdict.VALID),
        ((_result("s", V), _result("r", R, disagree=4)), True,
         ReconstructionVerdict.VALID_WITH_WARNINGS),
        ((_result("s", V), _result("m", V, skipped=True)), True,
         ReconstructionVerdict.VALID_WITH_WARNINGS),
        ((_result("s", V), _result("m", V, disagree=1)), True,
         ReconstructionVerdict.CONTRADICTED),
        ((_result("s", V),), False, ReconstructionVerdict.UNDECLARED),
        # A failure alongside a missing declaration is still undeclared first.
        ((_result("m", V, disagree=1),), False, ReconstructionVerdict.UNDECLARED),
    ],
)
def test_each_verdict(results, declared, expected):
    report = AgreementReport(dataset_id="d", results=results)
    assert report.assess(CLOSE, declared=declared).verdict is expected


def test_vacuous_truth_is_refused():
    """No validity test corroborating anything is never plain VALID."""
    assessment = AgreementReport(dataset_id="d").assess(CLOSE, declared=True)
    assert assessment.verdict is ReconstructionVerdict.VALID_WITH_WARNINGS
    assert assessment.caveats == ("reconstruction_uncorroborated",)


def test_a_test_that_saw_no_rows_is_unverified_not_passed():
    report = AgreementReport(dataset_id="d", results=(_result("s", V, checked=0),))
    assessment = report.assess(CLOSE, declared=True)
    assert assessment.verdict is ReconstructionVerdict.VALID_WITH_WARNINGS
    assert assessment.unverified == ("s",)


def test_no_tolerance_turns_a_near_miss_into_a_pass():
    one_row = _result("m", V, checked=100_000, disagree=1)
    report = AgreementReport(dataset_id="d", results=(one_row,))
    assert report.assess(CLOSE, declared=True).verdict is ReconstructionVerdict.CONTRADICTED


def test_a_calendar_dependent_validity_test_is_demoted_until_the_calendar_resolves():
    labelled = _result("q", V, disagree=2, calendar=True)
    report = AgreementReport(dataset_id="d", results=(_result("s", V), labelled))
    unresolved = report.assess(CLOSE, declared=True, calendar_resolved=False)
    resolved = report.assess(CLOSE, declared=True, calendar_resolved=True)
    assert unresolved.verdict is ReconstructionVerdict.VALID_WITH_WARNINGS
    assert unresolved.demoted == ("q",)
    assert resolved.verdict is ReconstructionVerdict.CONTRADICTED


# -- visible in dataset metadata --------------------------------------------


def test_the_agreement_report_is_persisted_beside_the_dataset(tmp_path, settings: Settings):
    dataset = _dataset(tmp_path, settings)
    path = settings.agreement_path("gated")
    assert path.exists()
    reloaded = load_dataset("gated", settings)
    assert reloaded.agreement == dataset.agreement
    assert reloaded.agreement.fiscal_year_start_month == settings.fiscal_year_start_month


def test_a_verdict_survives_reload_without_re_running_anything(tmp_path, settings: Settings):
    _dataset(tmp_path, settings, edit=_push_every_snapshot)
    reloaded = load_dataset("gated", settings)
    failure = reloaded.agreement.by_id()["close_date_moves_for_pushed_deals"]
    assert failure.failed
    assert _close_binding(reloaded, settings).status is BindingStatus.UNAVAILABLE


def test_the_card_carries_roles_and_the_verdict(tmp_path, settings: Settings):
    dataset = _dataset(tmp_path, settings, edit=_corrupt_close_date_qtr)
    card = understand(dataset, settings=settings).card()
    assert "close_date_matches_close_date_qtr [reconciliation]: failed" in card
    assert "close_date_moves_for_pushed_deals [validity]: passed" in card
    assert "expected_close_date: valid_with_warnings" in card
    assert "UNRESOLVED: configured default" in card


def test_a_report_from_another_fiscal_calendar_is_re_evaluated(tmp_path, settings: Settings):
    dataset = _dataset(tmp_path, settings)
    assert _verdict(dataset, settings).verdict is ReconstructionVerdict.VALID

    # A tenant declaring a February year re-evaluates the quarter tests against
    # it. The stamped labels were built for January, so they now disagree. The
    # checks are reconciliation, so that is a warning rather than a withhold.
    february = TenantProfile(tenant_id="t", fiscal_year_start_month=2)
    understanding = understand(dataset, tenant=february, settings=settings)
    assert understanding.agreement.fiscal_year_start_month == 2
    assert understanding.agreement.failures
    assert understanding.bindings.get(CLOSE).status is BindingStatus.CONFIRMED
    verdict = understanding.bindings.reconstruction_verdicts[0]
    assert verdict.verdict is ReconstructionVerdict.VALID_WITH_WARNINGS


def test_a_dataset_that_carried_its_own_close_date_has_no_agreement_report(
    tmp_path, settings: Settings
):
    path = tmp_path / "with_close.csv"
    write_production_csv(path, include_close_date=True)
    dataset = register_dataset(
        path, "own", column_registry=OPPORTUNITY_SNAPSHOT_V1, settings=settings
    )
    assert dataset.agreement is None
    assert not settings.agreement_path("own").exists()


def test_reingesting_without_a_reconstruction_removes_the_stale_report(
    tmp_path, settings: Settings
):
    ingest(
        _export(tmp_path), "again", settings=settings, column_registry=OPPORTUNITY_SNAPSHOT_V1
    )
    assert settings.agreement_path("again").exists()

    with_close = tmp_path / "with_close.csv"
    write_production_csv(with_close, include_close_date=True)
    ingest(with_close, "again", settings=settings, column_registry=OPPORTUNITY_SNAPSHOT_V1)
    assert not settings.agreement_path("again").exists()


# -- days_to_close is never guessed ------------------------------------------


def _close_dates(settings: Settings, dataset_id: str) -> dict[int, object]:
    store = DuckDBStore(settings)
    with store.connect() as conn:
        rows = conn.execute(
            f"SELECT opp_id, as_of, close_date FROM {store.snapshots_scan(dataset_id)} "
            "ORDER BY as_of, opp_id"
        ).fetchall()
    return {i: r for i, r in enumerate(rows)}


def test_a_fractional_days_to_close_is_null_not_rounded(tmp_path, settings: Settings):
    # DuckDB rounds on cast, so '2.6' would become 3 and move the date a day with
    # no signal. Whether the real column is ever fractional is an open question
    # (days_to_close_edge_cases), and an open question is not answered by rounding.
    def fractional(index: int, row: dict[str, str]) -> None:
        if index == 0:
            row["days_to_close"] = "2.6"
        if index == 1:
            row["days_to_close"] = "7.0"

    _dataset(tmp_path, settings, edit=fractional)
    rows = _close_dates(settings, "gated")
    assert rows[0][2] is None
    # A whole number written with a trailing .0 is a whole number.
    assert rows[1][2] is not None
    assert (rows[1][2] - rows[1][1]).days == 7


def test_an_absurd_horizon_is_null_rather_than_a_crash(tmp_path, settings: Settings):
    def absurd(index: int, row: dict[str, str]) -> None:
        if index == 0:
            row["days_to_close"] = "999999999"

    _dataset(tmp_path, settings, edit=absurd)
    assert _close_dates(settings, "gated")[0][2] is None


def test_negative_horizons_are_kept_not_clamped(tmp_path, settings: Settings):
    # days_to_close is negative for a close date already past, and clamping it
    # would be a guess about a definition nobody has confirmed.
    def past(index: int, row: dict[str, str]) -> None:
        if index == 0:
            row["days_to_close"] = "-30"

    _dataset(tmp_path, settings, edit=past)
    row = _close_dates(settings, "gated")[0]
    assert (row[2] - row[1]).days == -30


# -- reconstruction metadata ---------------------------------------------------


def test_reconstruction_metadata_names_the_derivation_and_its_inputs(
    tmp_path, settings: Settings
):
    dataset = _dataset(tmp_path, settings)
    derived = {d.column: d for d in dataset.schema.derived_columns}[CanonicalColumn.CLOSE_DATE]
    assert derived.is_reconstructed
    assert derived.expression == "as_of + days_to_close"
    assert derived.sources == ("as_of", "days_to_close")
    assert set(derived.agreement_test_ids) == {r.test_id for r in dataset.agreement.results}
    assert derived.requires_confirmation


def test_the_close_date_tests_carry_their_declared_roles(tmp_path, settings: Settings):
    dataset = _dataset(tmp_path, settings)
    roles = {r.test_id: r.role for r in dataset.agreement.results}
    assert roles == {
        "close_date_is_structurally_valid": V,
        "close_date_moves_for_pushed_deals": V,
        "close_date_matches_close_date_qtr": R,
        "close_date_matches_cd_in_qtr": R,
        "close_date_matches_eoq_close_diff": R,
    }
    dependent = {r.test_id for r in dataset.agreement.results if r.calendar_dependent}
    assert dependent == {"close_date_matches_close_date_qtr", "close_date_matches_cd_in_qtr"}
