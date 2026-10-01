"""Onboarding: propose never confirms; approve refuses until a human has (WP6).

The generalization check: the same synthetic panel under renamed headers
(the `alpha` variant) onboards to the same answers once a human declares it.
The human's choices are written here explicitly, playing the reviewer; the
product never reads the generator's rename map.
"""

from __future__ import annotations

import pytest

from ai_analyst.config import Settings
from ai_analyst.contracts.source import SourceFormat, TableSource
from ai_analyst.contracts.status import OpportunityStatus, StatusStrategy
from ai_analyst.data.onboarding import (
    OnboardingDraft,
    OnboardingRefused,
    ReviewStatus,
    approve,
    propose,
)
from ai_analyst.data.store import DuckDBStore
from ai_analyst.synthetic.feature_store import (
    CLOSED_LOST,
    CLOSED_WON,
    GeneratorConfig,
    generate,
    write_csv,
)
from scripts.onboard import main as cli

CONFIRMED = ReviewStatus.CONFIRMED
# The reviewer's reading of the synthetic stage vocabulary.
STATUS_OF = {CLOSED_WON: OpportunityStatus.WON, CLOSED_LOST: OpportunityStatus.LOST}
JUNK = {"not available", "DELETED_IN_CRM"}


@pytest.fixture(scope="module")
def panel(tmp_path_factory):
    dataset = generate(GeneratorConfig(seed=3, n_opportunities=25, n_quarters=1))
    root = tmp_path_factory.mktemp("onboard")
    paths = {}
    for variant in ("base", "alpha"):
        paths[variant] = write_csv(dataset, root / f"{variant}.csv", variant=variant)
    return dataset.ground_truth, paths


def _source(path) -> TableSource:
    return TableSource(format=SourceFormat.CSV, uri=str(path))


def _review(draft: OnboardingDraft, *, id_col, as_of, amount, stage) -> OnboardingDraft:
    """Play the human reviewer: name the columns and confirm every load-bearing item."""
    draft.grain.id_column, draft.grain.as_of_column = id_col, as_of
    draft.grain.status = CONFIRMED
    draft.amount.column, draft.amount.status = amount, CONFIRMED
    draft.stage.column, draft.stage.status = stage, CONFIRMED
    for entry in draft.stage_status_map.entries:
        entry.proposed_status = STATUS_OF.get(
            entry.value,
            OpportunityStatus.EXCLUDED if entry.value in JUNK else OpportunityStatus.OPEN,
        )
        entry.status = CONFIRMED
    draft.capture_policy.status = CONFIRMED
    return draft


def _answers(dataset_id: str, settings: Settings) -> list[tuple]:
    store = DuckDBStore(settings)
    with store.connect() as conn:
        return conn.execute(
            f"SELECT status, COUNT(*), COUNT(DISTINCT opp_id), SUM(amount) "
            f"FROM {store.snapshots_scan(dataset_id)} GROUP BY 1 ORDER BY 1"
        ).fetchall()


# ------------------------------------------------------------------ propose


def test_a_proposal_confirms_nothing(panel):
    _, paths = panel
    draft = propose(_source(paths["base"]))
    assert draft.grain.status is ReviewStatus.INFERRED
    assert (draft.grain.id_column, draft.grain.as_of_column) == ("opp_id", "as_of")
    assert draft.amount.status is not CONFIRMED
    assert draft.stage.status is not CONFIRMED
    assert all(e.status is not CONFIRMED for e in draft.stage_status_map.entries)
    assert all(i.status is not CONFIRMED for i in draft.invariants)
    # Family declarations are inferred, which issues no grants.
    assert draft.tenant.family_declarations
    assert all(d.status.value == "inferred" for d in draft.tenant.family_declarations)
    assert not draft.tenant.concept_columns and draft.tenant.stage_status_map is None


def test_planted_double_captures_propose_a_capture_policy_for_review(panel):
    truth, paths = panel
    draft = propose(_source(paths["base"]))
    assert draft.capture_policy.proposed
    assert draft.capture_policy.status is ReviewStatus.NEEDS_REVIEW
    assert draft.grain.candidates[0].duplicate_groups_day == truth.duplicate_groups


def test_invariants_are_proposed_only_when_they_hold(panel):
    _, paths = panel
    draft = propose(_source(paths["base"]))
    assert draft.invariants
    assert all(p.disagreeing_rows == 0 and p.checked_rows > 0 for p in draft.invariants)


def test_the_draft_round_trips_through_json(panel, tmp_path):
    _, paths = panel
    draft = propose(_source(paths["base"]))
    draft.save(tmp_path / "d.json")
    assert OnboardingDraft.load(tmp_path / "d.json") == draft


# ------------------------------------------------------------------ approve


def test_an_unreviewed_draft_is_refused_with_every_reason(panel, tmp_path):
    _, paths = panel
    draft = propose(_source(paths["base"]))
    with pytest.raises(OnboardingRefused) as refused:
        approve(draft, "ds", settings=Settings(data_root=tmp_path))
    text = " ".join(refused.value.problems)
    for item in ("grain", "amount", "stage", "stage_status_map", "capture_policy"):
        assert item in text
    assert not (tmp_path / "canonical").exists()


def test_one_unconfirmed_stage_value_blocks_approval(panel, tmp_path):
    _, paths = panel
    draft = _review(propose(_source(paths["base"])),
                    id_col="opp_id", as_of="as_of", amount="amount", stage="stage")
    draft.stage_status_map.entries[-1].status = ReviewStatus.NEEDS_REVIEW
    with pytest.raises(OnboardingRefused, match="stage_status_map"):
        approve(draft, "ds", settings=Settings(data_root=tmp_path))


def test_a_stage_mapped_to_unknown_is_refused(panel, tmp_path):
    _, paths = panel
    draft = _review(propose(_source(paths["base"])),
                    id_col="opp_id", as_of="as_of", amount="amount", stage="stage")
    draft.stage_status_map.entries[0].proposed_status = OpportunityStatus.UNKNOWN
    with pytest.raises(OnboardingRefused, match="unknown"):
        approve(draft, "ds", settings=Settings(data_root=tmp_path))


def test_an_approved_draft_registers_with_the_declared_stage_map(panel, tmp_path):
    truth, paths = panel
    settings = Settings(data_root=tmp_path)
    draft = _review(propose(_source(paths["base"])),
                    id_col="opp_id", as_of="as_of", amount="amount", stage="stage")
    dataset, registration = approve(draft, "base", settings=settings)
    resolution = dataset.schema.status_resolution
    assert resolution.strategy is StatusStrategy.DECLARED_STAGE_MAP
    assert dataset.schema.capture_resolution.duplicate_groups == truth.duplicate_groups
    statuses = {row[0] for row in _answers("base", settings)}
    assert "unknown" not in statuses
    assert registration.tenant.stage_status_map


# ------------------------------------------------------------ generalization


def test_renamed_headers_onboard_to_the_same_answers(panel, tmp_path):
    _, paths = panel
    settings = Settings(data_root=tmp_path)
    base = _review(propose(_source(paths["base"])),
                   id_col="opp_id", as_of="as_of", amount="amount", stage="stage")
    # The renamed tenant: its stage column is named by the operator, since no
    # header suggests it; the reviewer then declares the same meanings.
    alpha = propose(_source(paths["alpha"]), stage_column="pipeline_step")
    assert alpha.stage.status is ReviewStatus.INFERRED
    alpha = _review(alpha, id_col="deal_key", as_of="capture_ts",
                    amount="booking_value", stage="pipeline_step")
    approve(base, "base", settings=settings)
    approve(alpha, "alpha", settings=settings)
    assert _answers("alpha", settings) == _answers("base", settings)


def test_cli_refuses_an_unreviewed_draft(panel, tmp_path, capsys):
    _, paths = panel
    out = tmp_path / "draft.json"
    assert cli(["propose", str(paths["base"]), "--out", str(out)]) == 0
    assert "Nothing is confirmed" in capsys.readouterr().out
    assert cli(["approve", str(out), "--dataset-id", "x"]) == 2
    assert "refused" in capsys.readouterr().err
