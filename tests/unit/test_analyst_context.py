"""The tiered analyst context card (ARCHITECTURE 12.4, 12.5).

The budget test is the point of this file. The persisted profile is roughly
15k tokens, which is why the card exists at all; a card that quietly grows back
toward that size has failed even if every field in it is correct.
"""

from __future__ import annotations

from ai_analyst.config import Settings
from ai_analyst.contracts.binding import BindingStatus
from ai_analyst.contracts.columns import InformationClass, MonetaryStatus
from ai_analyst.contracts.concepts import BusinessConcept
from ai_analyst.contracts.context import estimate_tokens
from ai_analyst.contracts.opportunity_snapshot_v1 import OPPORTUNITY_SNAPSHOT_V1
from ai_analyst.data.binding import build_bindings
from ai_analyst.data.ingest import ingest
from ai_analyst.data.profiler import profile_dataset
from ai_analyst.data.store import DuckDBStore
from ai_analyst.data.understanding import (
    build_context,
    column_detail,
    evaluate_agreement,
    list_columns,
)
from tests.fixtures.production_shape import write_production_csv


def _context(tmp_path, settings: Settings):
    source = tmp_path / "export.csv"
    write_production_csv(source, include_close_date=False)
    result = ingest(
        source, "client_xyz", settings=settings, column_registry=OPPORTUNITY_SNAPSHOT_V1
    )
    profile = profile_dataset(
        "client_xyz", result.schema, settings=settings, registry=result.registry
    )
    store = DuckDBStore(settings)
    with store.connect() as conn:
        report = evaluate_agreement(
            conn,
            store.snapshots_scan("client_xyz"),
            result.schema,
            dataset_id="client_xyz",
            settings=settings,
        )
    bindings = build_bindings(result.schema, result.registry, profile, agreement=report)
    context = build_context(result.schema, result.registry, profile, bindings, report, settings)
    return result, profile, bindings, context


def test_tier_zero_stays_within_its_token_budget(tmp_path, settings: Settings):
    _, profile, _, context = _context(tmp_path, settings)
    rendered = context.render()
    assert context.estimated_tokens <= settings.context_card_token_budget, rendered

    # And it is a genuine projection: an order of magnitude smaller than the
    # profile it is drawn from, not a reformatting of it.
    profile_tokens = estimate_tokens(profile.model_dump_json())
    assert context.estimated_tokens * 5 < profile_tokens


def test_tier_zero_names_concepts_rather_than_listing_every_column(
    tmp_path, settings: Settings
):
    _, _, _, context = _context(tmp_path, settings)
    rendered = context.render()
    assert "concepts" in rendered
    # Every concept gets a line; the 145 columns do not.
    assert len(context.concepts) == len(BusinessConcept)
    assert context.column_count == 145


def test_tier_zero_says_what_is_unavailable_as_plainly_as_what_is_not(
    tmp_path, settings: Settings
):
    # An unanswerable question should be refused from the card, not from a
    # failed query. This export has no segment column at all.
    _, _, _, context = _context(tmp_path, settings)
    unavailable = {c.concept for c in context.unavailable_concepts()}
    assert BusinessConcept.CUSTOMER_SEGMENT in unavailable
    assert "UNAVAILABLE" in context.render()


def test_tier_zero_carries_the_open_questions_and_quality_warnings(
    tmp_path, settings: Settings
):
    _, _, _, context = _context(tmp_path, settings)
    assert "authoritative_status" in context.open_questions
    assert "rep_aggregate_lineage" in context.open_questions
    codes = {w.code for w in context.quality_warnings}
    assert "status_not_authoritative" in codes


def test_tier_zero_reports_the_agreement_results(tmp_path, settings: Settings):
    _, _, _, context = _context(tmp_path, settings)
    assert context.agreement
    assert "agreement tests" in context.render()


def test_no_raw_row_ever_reaches_tier_zero(tmp_path, settings: Settings):
    # Tier 2 is rows, and tier 2 is never the default context.
    _, _, _, context = _context(tmp_path, settings)
    rendered = context.render()
    assert "OPP-000" not in rendered
    assert "OPP-001" not in rendered


def test_no_narrative_content_reaches_tier_zero(tmp_path, settings: Settings):
    _, profile, _, context = _context(tmp_path, settings)
    rendered = context.render()
    assert context.text_fields
    for field in context.text_fields:
        assert field.name in rendered or "more in the index" in rendered
    # The field names appear; a sentence of manager notes does not.
    assert "Reviewed the" not in rendered
    assert "catalogued only" in rendered


def test_the_column_index_degrades_on_a_wide_dataset(tmp_path, settings: Settings):
    narrow = settings.model_copy(update={"context_card_max_indexed_columns": 10})
    source = tmp_path / "export.csv"
    write_production_csv(source, include_close_date=False)
    result = ingest(source, "wide", settings=settings, column_registry=OPPORTUNITY_SNAPSHOT_V1)
    profile = profile_dataset("wide", result.schema, settings=settings, registry=result.registry)
    bindings = build_bindings(result.schema, result.registry, profile)
    context = build_context(result.schema, result.registry, profile, bindings, None, narrow)

    assert context.index_degraded
    assert context.column_index == {}
    assert "column index withheld" in context.render()
    # The counts survive, so the model still knows the shape of what it cannot see.
    assert context.class_counts
    assert context.estimated_tokens <= narrow.context_card_token_budget


def test_tier_one_returns_one_column_in_full(tmp_path, settings: Settings):
    result, profile, bindings, _ = _context(tmp_path, settings)
    detail = column_detail("account_ti_first_won", result.registry, profile, bindings)

    assert detail.name == "account_ti_first_won"
    assert detail.category == "account_feature"
    assert detail.availability == "backward_derived"
    assert detail.profile is not None
    assert detail.profile.sentinel_count > 0
    rendered = detail.render()
    assert "sentinel" in rendered
    assert "excluded from statistics" in rendered


def test_tier_one_explains_why_a_column_is_withheld(tmp_path, settings: Settings):
    result, profile, _, _ = _context(tmp_path, settings)
    detail = column_detail("train", result.registry, profile)
    assert detail.quarantined
    assert detail.quarantine_code == "non_analytic_metadata"
    assert "WITHHELD" in detail.render()


def test_tier_one_on_a_text_column_reports_length_and_no_content(
    tmp_path, settings: Settings
):
    result, profile, _, _ = _context(tmp_path, settings)
    detail = column_detail("ManagerNotes", result.registry, profile)
    rendered = detail.render()
    assert "mean length" in rendered
    assert "No content is stored" in rendered
    assert detail.profile is not None
    assert detail.profile.top_values == []


def test_listing_columns_is_how_a_wide_dataset_is_explored(tmp_path, settings: Settings):
    result, _, _, _ = _context(tmp_path, settings)
    text = list_columns(result.registry, InformationClass.TEXT)
    assert "ManagerNotes" in text
    assert all("Notes" in n or "Why" in n or "Next" in n or "Paperwork" in n or "Decision" in n
               or "Channel" in n for n in text)

    matched = list_columns(result.registry, pattern="rep_win")
    assert "rep_win_rate" in matched
    assert "account_id" not in matched


def test_the_card_reports_the_status_strategy_and_that_it_is_not_authoritative(
    tmp_path, settings: Settings
):
    _, _, _, context = _context(tmp_path, settings)
    assert not context.status_is_authoritative
    assert "NOT authoritative" in context.render()


def test_bindings_cover_the_four_statuses_the_milestone_requires(
    tmp_path, settings: Settings
):
    _, _, bindings, _ = _context(tmp_path, settings)
    statuses = {b.status for b in bindings.bindings}
    assert BindingStatus.CONFIRMED in statuses
    assert BindingStatus.INFERRED in statuses
    assert BindingStatus.UNAVAILABLE in statuses

    confirmed = bindings.get(BusinessConcept.AMOUNT)
    assert confirmed.status is BindingStatus.CONFIRMED
    inferred = bindings.get(BusinessConcept.TERMINAL_OUTCOME)
    assert inferred.status is BindingStatus.INFERRED
    assert inferred.columns == ("terminal_fate",)
    assert not inferred.confirming_evidence


def test_the_amount_concept_marks_its_column_monetary(tmp_path, settings: Settings):
    result, profile, bindings, _ = _context(tmp_path, settings)
    assert "amount" in bindings.monetary_columns()
    detail = column_detail("amount", result.registry, profile, bindings)
    assert detail.monetary is MonetaryStatus.MONETARY
    assert detail.bound_to is BusinessConcept.AMOUNT


def test_understand_runs_the_whole_layer_from_a_registered_dataset(
    tmp_path, settings: Settings
):
    from ai_analyst.contracts.concepts import AnalyticalOperation
    from ai_analyst.data.dataset import register_dataset
    from ai_analyst.data.understanding import understand

    source = tmp_path / "export.csv"
    write_production_csv(source, include_close_date=False)
    dataset = register_dataset(
        source, "whole", column_registry=OPPORTUNITY_SNAPSHOT_V1, settings=settings
    )
    understanding = understand(dataset, settings=settings)

    assert understanding.dataset_id == "whole"
    assert understanding.card()
    assert understanding.context.estimated_tokens <= settings.context_card_token_budget
    assert understanding.agreement.results

    # A metric layer asks the question this way and gets a reason, not a crash.
    unanswerable = understanding.unanswerable()
    assert AnalyticalOperation.SEGMENT_ANALYSIS in unanswerable
    assert unanswerable[AnalyticalOperation.SEGMENT_ANALYSIS] == (
        "customer_segment concept unavailable"
    )
    # Pipeline questions need an amount and a close date, and this export has
    # both once the close date is rebuilt.
    assert AnalyticalOperation.SNAPSHOT_PIPELINE not in unanswerable


def test_a_dataset_without_an_amount_reports_the_concept_as_unavailable(
    tmp_path, settings: Settings
):
    # "A metric requiring amount should later report 'amount concept
    # unavailable' rather than making dataset ingestion fail."
    from ai_analyst.contracts.concepts import AnalyticalOperation
    from ai_analyst.data.dataset import register_dataset
    from ai_analyst.data.understanding import understand

    source = tmp_path / "no_amount.csv"
    source.write_text(
        "as_of,opp_id,Stage\n2025-01-01,O-1,Closed Won\n2025-02-01,O-1,Closed Won\n",
        encoding="utf-8",
    )
    dataset = register_dataset(source, "noamount", settings=settings)
    understanding = understand(dataset, settings=settings)

    from ai_analyst.contracts.concepts import BusinessConcept as BC

    check = understanding.check(AnalyticalOperation.SNAPSHOT_PIPELINE)
    assert not check.satisfied
    assert BC.AMOUNT in check.missing
    assert "amount" in check.reason
    assert "concept unavailable" in check.reason
    # The grain is still confirmed, because ingestion verified it rather than
    # inferring it from a header.
    assert understanding.bindings.get(BC.OPPORTUNITY_ID).status.admissible_in_semantic_path


def test_tier_two_returns_bounded_rows_and_withholds_narrative(
    tmp_path, settings: Settings
):
    # Rows exist, but only down this explicit path, and never with content.
    from ai_analyst.data.dataset import register_dataset
    from ai_analyst.data.understanding import sample_rows

    source = tmp_path / "export.csv"
    write_production_csv(source, include_close_date=False)
    dataset = register_dataset(
        source, "tier2", column_registry=OPPORTUNITY_SNAPSHOT_V1, settings=settings
    )
    sample = sample_rows(dataset, limit=3, settings=settings)

    assert len(sample.rows) == 3
    assert "ManagerNotes" not in sample.columns
    assert "ManagerNotes" in sample.excluded_text_columns
    assert "train" in sample.excluded_quarantined_columns
    assert "opp_id" in sample.columns
    assert "narrative columns withheld" in sample.render()


def test_tier_two_respects_a_hard_row_cap(tmp_path, settings: Settings):
    from ai_analyst.data.dataset import register_dataset
    from ai_analyst.data.understanding import sample_rows

    source = tmp_path / "export.csv"
    write_production_csv(source, include_close_date=False)
    dataset = register_dataset(
        source, "capped", column_registry=OPPORTUNITY_SNAPSHOT_V1, settings=settings
    )
    sample = sample_rows(dataset, limit=10_000, settings=settings)
    assert sample.row_limit == settings.max_sample_rows
    assert len(sample.rows) <= settings.max_sample_rows
