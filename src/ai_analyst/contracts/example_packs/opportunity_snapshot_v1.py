"""Proposed classification for the production opportunity-snapshot export.

129 columns. Every assignment below was made from the column name alone; no row
of this dataset has been inspected. Entries are therefore proposals carrying
`requires_confirmation=True`, except the grain columns, whose role is fixed by
the project's own definition of the data.

Three structural findings drove the assignments:

1. **There is no per-snapshot close date column.** The concept plainly exists,
   because `close_date_push_count`, `cd_movement` and `eoq_close_diff` are
   meaningless without one. It is reconstructible as `as_of + days_to_close`,
   and that reconstruction has three independent cross-checks. Until it is
   verified, the bridge, transition and slip primitives cannot be written.
2. **Segment, region and industry are absent**, so segment analysis is not
   answerable from this export.
3. **No free-text column is present.** The 22 `*_updated_days` columns are
   recency derivatives of CRM narrative fields that were not exported.
"""

from __future__ import annotations

from ai_analyst.contracts.columns import (
    Availability,
    CanonicalCoverage,
    ColumnClassification,
    ColumnFamily,
    ColumnRegistry,
    CoverageStatus,
    Disposition,
    FeatureLineage,
    MonetaryStatus,
    QuarantineCode,
    QuarantineReason,
    UnresolvedItem,
)
from ai_analyst.contracts.schema import CanonicalColumn, DateEncoding

CATEGORY = __import__("ai_analyst.contracts.columns", fromlist=["ColumnCategory"]).ColumnCategory

_LINEAGE_UNCONFIRMED = FeatureLineage(
    description="Precomputed aggregate supplied with the export.",
    lookback="unconfirmed",
    frozen_through="previous quarter (frozen historical set)",
    confirmed=False,
)

FAMILIES: list[ColumnFamily] = [
    ColumnFamily(
        name="field_update_recency",
        pattern=r"^.+_updated_days$",
        category=CATEGORY.DERIVED_TEMPORAL,
        availability=Availability.BACKWARD_DERIVED,
        disposition=Disposition.USE_WITH_PROOF,
        recomputable=False,
        note=(
            "Days since the named CRM field last changed, as of this snapshot. "
            "Backward-looking by construction and a genuinely useful staleness "
            "signal. Not recomputable here, because the underlying narrative "
            "fields were not exported. Needs proof that the clock stops at as_of."
        ),
    ),
    ColumnFamily(
        name="rep_aggregate",
        pattern=r"^rep_",
        category=CATEGORY.REP_FEATURE,
        availability=Availability.BACKWARD_DERIVED,
        disposition=Disposition.USE_WITH_PROOF,
        recomputable=False,
        note=(
            "Rep-scoped aggregate. Confirmed to come in two flavours: frozen "
            "historical data through the previous quarter, and day-level current "
            "quarter data supplied separately for inference. The frozen basis "
            "cannot contain current-quarter outcomes, which is what takes these "
            "out of quarantine. Preserved rather than recomputed, per project "
            "direction. Per-feature lineage is still unconfirmed, and lookbacks "
            "must not be assumed uniform across the family."
        ),
        lineage=_LINEAGE_UNCONFIRMED,
    ),
    ColumnFamily(
        name="terminal_outcome",
        pattern=r"^terminal_",
        category=CATEGORY.OUTCOME,
        availability=Availability.FUTURE_CONTAMINATED,
        disposition=Disposition.RECOMPUTE,
        recomputable=True,
        note=(
            "Retrospective outcome label. Forbidden in prospective analysis. The "
            "trace primitive recomputes terminal state from snapshot history, and "
            "this column is the reconciliation target for that recomputation."
        ),
    ),
    ColumnFamily(
        name="crm_narrative",
        pattern=(
            r"^(ManagerNotes|SENotes|AVPNotes|RVPNotes|CSMNotes|Channel_Notes"
            r"|RetentionTeamNotes|NextStep|POVNextSteps|Why_Do_Anything|Why_Now"
            r"|Why_Us|Decision_Criteria|Major_Pain_Point|Paperwork_Process)$"
        ),
        category=CATEGORY.TEXT,
        availability=Availability.AS_OF_FACT,
        disposition=Disposition.USE_WITH_PROOF,
        note=(
            "CRM narrative field. Confirmed to exist upstream but absent from this "
            "129-column export, which carries only its *_updated_days recency "
            "derivative. Catalogued only: never a dimension, never a measure, and "
            "only null and non-null filters. An as-of fact if written in period; "
            "a field backfilled after the fact would leak the outcome, and "
            "append-only detection across snapshots is the evidence either way."
        ),
    ),
]


# Source columns stored as Excel serial numbers rather than ISO dates
# (ARCHITECTURE 12.19). Declared, never detected. The project owner confirmed that
# both real exports use serials for their date fields. Only `as_of` has been
# decoded against real rows (it yields plausible consecutive snapshot dates); the
# rest are declared from that confirmation and recorded as unverified in
# `date_encoding_declarations`. A wrong declaration fails ingestion loudly rather
# than producing wrong dates, so this is safe to declare ahead of verification.
DATE_ENCODINGS: dict[str, DateEncoding] = {
    "as_of": DateEncoding.EXCEL_SERIAL,
    "terminal_date": DateEncoding.EXCEL_SERIAL,
    "terminal_quarter_eoq": DateEncoding.EXCEL_SERIAL,
    "deal_first_seen_close_date": DateEncoding.EXCEL_SERIAL,
    "opp_first_commit_as_of": DateEncoding.EXCEL_SERIAL,
}

# Columns that hold money in this export (ARCHITECTURE 12.16). Declared by name
# because money is a semantic fact, not a numeric shape: `deal_amount_vs_rep_avg`
# is a ratio and `deal_amount_percentile_overall` is a percentile, and neither is
# money although both have "amount" in the name. A money column reports exact
# extremes and no float summary statistics.
MONETARY_COLUMNS: frozenset[str] = frozenset(
    {
        "new_amount",
        "terminal_amount",
        "rep_avg_deal_amount",
    }
)


def _c(
    name: str,
    category,
    availability: Availability,
    disposition: Disposition,
    note: str,
    *,
    recomputable: bool = False,
    requires_confirmation: bool = True,
    sentinels: tuple[float, ...] = (),
    lineage: FeatureLineage | None = None,
    leakage_check: str | None = None,
) -> ColumnClassification:
    quarantine = QUARANTINE_REASONS.get(name) if disposition is Disposition.QUARANTINE else None
    monetary = MonetaryStatus.MONETARY if name in MONETARY_COLUMNS else None
    return ColumnClassification(
        name=name,
        category=category,
        availability=availability,
        disposition=disposition,
        recomputable=recomputable,
        note=note,
        requires_confirmation=requires_confirmation,
        sentinels=sentinels,
        lineage=lineage,
        leakage_check=leakage_check,
        quarantine=quarantine,
        monetary=monetary,
    )


# Explicit reason for every quarantined column (ARCHITECTURE 5.15). The registry
# refuses to construct a quarantined column without one, so a new quarantine
# cannot be added silently.
_REFERENCE_POPULATION_REASON = QuarantineReason(
    code=QuarantineCode.UNSTATED_REFERENCE_POPULATION,
    detail=(
        "Computed against a reference population (a ranking pool, an average, or a "
        "share denominator) that is not documented. 'Overall' most likely spans "
        "all eight quarters, which would include deals that did not exist at this "
        "snapshot."
    ),
    resolution=(
        "Document the reference population and confirm it is bounded by the snapshot's as_of."
    ),
)
_WINDOW_REASON = QuarantineReason(
    code=QuarantineCode.UNSTATED_WINDOW,
    detail="An account-level count whose lookback window is not documented.",
    resolution="Document the window and confirm it ends at as_of.",
)
_COMPOSITE_REASON = QuarantineReason(
    code=QuarantineCode.UNDEFINED_COMPOSITE,
    detail="A composite score whose formula and inputs are not documented.",
    resolution="Document the formula and inputs, then classify each input.",
)

QUARANTINE_REASONS: dict[str, QuarantineReason] = {
    "train": QuarantineReason(
        code=QuarantineCode.NON_ANALYTIC_METADATA,
        detail=(
            "Machine-learning train and test split flag. Not a business attribute, "
            "and subsetting on it would silently change every number."
        ),
        resolution=("Not released for analysis. Retained only so the export can be reconciled."),
    ),
    "opp_commit_to_close_days": QuarantineReason(
        code=QuarantineCode.AMBIGUOUS_DEFINITION,
        detail=(
            "'Close' most likely means the terminal close, which would make this a "
            "cycle-time outcome label rather than a snapshot feature. The "
            "definition has not been confirmed."
        ),
        resolution=("Confirm whether it measures to the snapshot close_date or to terminal_date."),
    ),
    "activity_density": _COMPOSITE_REASON,
    "eoq_urgency_score": _COMPOSITE_REASON,
    "account_open_deal_count": _WINDOW_REASON,
    "account_historical_deal_count": _WINDOW_REASON,
    **{
        name: _REFERENCE_POPULATION_REASON
        for name in (
            "deal_amount_rank_pct",
            "deal_amount_percentile_overall",
            "deal_age_percentile_overall",
            "deal_is_outlier_amount",
            "deal_amount_vs_rep_avg",
            "deal_amount_vs_account_avg",
            "deal_amount_vs_stage_avg",
            "deal_age_vs_stage_avg",
            "deal_amount_share_of_rep_pipeline_overall",
            "deal_amount_share_of_rep_pipeline_cq",
            "deal_amount_share_of_rep_at_current_stage",
            "pipeline_density_at_stage",
        )
    },
    "deal_amount_vs_segment_avg": QuarantineReason(
        code=QuarantineCode.UNSTATED_REFERENCE_POPULATION,
        detail=(
            "Compared against a segment average, but no segment column was "
            "exported, so the grouping cannot be reproduced or checked."
        ),
        resolution="Export the segment column and document the averaging population.",
    ),
}


_AS_OF = Availability.AS_OF_FACT
_BACK = Availability.BACKWARD_DERIVED
_FUTURE = Availability.FUTURE_CONTAMINATED
_UNK = Availability.UNKNOWN

_DIRECT = Disposition.DIRECT
_RECOMP = Disposition.RECOMPUTE
_PROOF = Disposition.USE_WITH_PROOF
_QUAR = Disposition.QUARANTINE

_CLOSE_CONFIRMED = (
    "Confirmed: close_date is the expected close date recorded at this snapshot, "
    "and days_to_close is close_date minus as_of. Time-varying snapshot state, "
    "not terminal-derived."
)
_CUMULATIVE = (
    "Confirmed cumulative since opportunity creation and stamped on each "
    "snapshot, so it is backward-derived and safe. NOT recomputed: the export "
    "covers eight quarters, so an opportunity created before the window has "
    "history the system cannot see, and recomputation would silently undercount."
)
_CALENDAR_NOTE = (
    "Pure calendar arithmetic on as_of. Recomputed from the fiscal calendar and reconciled."
)
_CD_NOTE = (
    "Depends on the reconstructed per-snapshot close date; blocked until that "
    "reconstruction is verified."
)
_ACCOUNT_LEAK = (
    "Confirmed as as_of minus the account's earliest terminal_date for the given "
    "fate. Intended as a historical feature, but its derivation reads outcome "
    "fields, so the no-look-ahead property is a requirement rather than a "
    "guarantee and must be verified."
)
_MOVE_NOTE = (
    "Movement counter over the opportunity's own history. Recomputable from "
    "snapshot history; window unstated, so recomputation wins."
)

INDIVIDUAL: list[ColumnClassification] = [
    # Identity and grain
    _c(
        "opp_id",
        CATEGORY.IDENTITY,
        _AS_OF,
        _DIRECT,
        "Opportunity identifier. Half of the grain.",
        requires_confirmation=False,
    ),
    _c(
        "as_of",
        CATEGORY.IDENTITY,
        _AS_OF,
        _DIRECT,
        "Snapshot date. Half of the grain.",
        requires_confirmation=False,
    ),
    _c("account_id", CATEGORY.IDENTITY, _AS_OF, _DIRECT, "Account identifier."),
    _c(
        "OwnerID",
        CATEGORY.IDENTITY,
        _AS_OF,
        _DIRECT,
        "Owning rep. Mutable across snapshots, so dimension attribution applies.",
    ),
    _c("RenewalManager", CATEGORY.IDENTITY, _AS_OF, _DIRECT, "Renewal owner. Mutable."),
    # Snapshot state
    _c(
        "Stage",
        CATEGORY.SNAPSHOT_STATE,
        _AS_OF,
        _DIRECT,
        "Sales stage in this snapshot. NOT authoritative for open, won, or lost "
        "(ARCHITECTURE 5.13): the vocabulary runs 0-Qualification through "
        "6-Order Placed plus Closed Won, Closed Lost, not available and "
        "SFDCDELETED. Keyword inference reads 6-Order Placed as open and "
        "would count deleted records as pipeline. Use for stage transition, "
        "progression, regression and distribution analysis only.",
        requires_confirmation=False,
    ),
    _c(
        "ForecastCategory",
        CATEGORY.SNAPSHOT_STATE,
        _AS_OF,
        _DIRECT,
        "Forecast category in this snapshot.",
    ),
    _c(
        "new_amount",
        CATEGORY.SNAPSHOT_STATE,
        _AS_OF,
        _DIRECT,
        "Deal amount in this snapshot. Confirmed by the project owner as the canonical amount.",
        requires_confirmation=False,
    ),
    _c(
        "Probability",
        CATEGORY.SNAPSHOT_STATE,
        _AS_OF,
        _DIRECT,
        "Win probability in this snapshot. Enables the weighted measure.",
    ),
    _c("Type", CATEGORY.DEAL_FEATURE, _AS_OF, _DIRECT, "Opportunity type. A usable dimension."),
    _c(
        "pipe_type",
        CATEGORY.DEAL_FEATURE,
        _AS_OF,
        _DIRECT,
        "Pipeline type. A usable dimension once its vocabulary is confirmed.",
    ),
    # Quarter labels
    _c("as_of_qtr", CATEGORY.QUARTER, _AS_OF, _RECOMP, _CALENDAR_NOTE, recomputable=True),
    _c(
        "close_date_qtr",
        CATEGORY.QUARTER,
        _AS_OF,
        _RECOMP,
        "Quarter of this snapshot's own close date. " + _CLOSE_CONFIRMED + " Still "
        "reconciled against the fiscal calendar applied to the reconstructed date.",
        recomputable=True,
    ),
    _c(
        "as_of_qtr_plus_1",
        CATEGORY.QUARTER,
        _AS_OF,
        _RECOMP,
        "Label of the following quarter. Pure arithmetic on as_of, so it leaks nothing by itself.",
        recomputable=True,
    ),
    _c(
        "as_of_qtr_plus_2",
        CATEGORY.QUARTER,
        _AS_OF,
        _RECOMP,
        "Label of the second following quarter.",
        recomputable=True,
    ),
    _c(
        "as_of_qtr_plus_3",
        CATEGORY.QUARTER,
        _AS_OF,
        _RECOMP,
        "Label of the third following quarter.",
        recomputable=True,
    ),
    _c("qtr_number", CATEGORY.QUARTER, _AS_OF, _RECOMP, _CALENDAR_NOTE, recomputable=True),
    _c(
        "qtr_segment",
        CATEGORY.QUARTER,
        _AS_OF,
        _RECOMP,
        "Confirmed: which third of the quarter the snapshot falls in, bucketed "
        "from days_since_boq. A quarter-phase feature, NOT a customer segment. "
        "Recomputable from the fiscal calendar once the bucket boundaries are "
        "known.",
        recomputable=True,
        requires_confirmation=False,
    ),
    # Metadata
    _c(
        "train",
        CATEGORY.METADATA,
        _AS_OF,
        _QUAR,
        "Machine-learning train and test split flag. Never a dimension or filter: "
        "subsetting on it would silently change every number. Its presence means "
        "this export was engineered for model training, which sharpens the leakage "
        "concern rather than softening it.",
    ),
    # Account features
    _c(
        "account_ti_first_won",
        CATEGORY.ACCOUNT_FEATURE,
        _BACK,
        _PROOF,
        "Account's first win . " + _ACCOUNT_LEAK,
        sentinels=(-999999.0,),
        leakage_check=(
            "every non-sentinel value must be >= 0, because a negative value means "
            "as_of precedes the terminal event and the feature is reading the future"
        ),
    ),
    _c(
        "account_ti_first_loss",
        CATEGORY.ACCOUNT_FEATURE,
        _BACK,
        _PROOF,
        "Account's first loss. " + _ACCOUNT_LEAK,
        sentinels=(-999999.0,),
        leakage_check=(
            "every non-sentinel value must be >= 0, because a negative value means "
            "as_of precedes the terminal event and the feature is reading the future"
        ),
    ),
    _c(
        "prev_won",
        CATEGORY.ACCOUNT_FEATURE,
        _BACK,
        _PROOF,
        "Prior wins for the account. Historical feature available at as_of; the "
        "window must be shown to end at as_of.",
        lineage=_LINEAGE_UNCONFIRMED,
    ),
    _c(
        "prev_loss",
        CATEGORY.ACCOUNT_FEATURE,
        _BACK,
        _PROOF,
        "Prior losses for the account. Same condition and same check.",
        lineage=_LINEAGE_UNCONFIRMED,
    ),
    _c(
        "account_open_deal_count",
        CATEGORY.ACCOUNT_FEATURE,
        _UNK,
        _QUAR,
        "Open deals on the account. Scope unstated.",
    ),
    _c(
        "account_historical_deal_count",
        CATEGORY.ACCOUNT_FEATURE,
        _UNK,
        _QUAR,
        "Historical deal count. Window unstated.",
    ),
    _c(
        "deal_amount_vs_account_avg",
        CATEGORY.ACCOUNT_FEATURE,
        _UNK,
        _QUAR,
        "Ratio to an account average whose window is unstated.",
    ),
    # Rep-referencing features not caught by the rep_ prefix
    _c(
        "deal_amount_vs_rep_avg",
        CATEGORY.REP_FEATURE,
        _UNK,
        _QUAR,
        "Ratio to a rep average whose window is unstated.",
    ),
    _c(
        "deal_amount_share_of_rep_pipeline_overall",
        CATEGORY.REP_FEATURE,
        _UNK,
        _QUAR,
        "'Overall' most likely spans all eight quarters, which would include "
        "pipeline that did not exist at this snapshot.",
    ),
    _c(
        "deal_amount_share_of_rep_pipeline_cq",
        CATEGORY.REP_FEATURE,
        _UNK,
        _QUAR,
        "Current-quarter share. Safer in principle, still unstated.",
    ),
    _c(
        "deal_amount_share_of_rep_at_current_stage",
        CATEGORY.REP_FEATURE,
        _UNK,
        _QUAR,
        "Share within the rep's current-stage pipeline. Scope unstated.",
    ),
    # Deal-relative features
    _c(
        "deal_amount_rank_pct",
        CATEGORY.DEAL_FEATURE,
        _UNK,
        _QUAR,
        "Ranking scope unstated. A global rank spans the whole export.",
    ),
    _c(
        "deal_amount_percentile_overall",
        CATEGORY.DEAL_FEATURE,
        _UNK,
        _QUAR,
        "'Overall' most likely means across all eight quarters, making this future-contaminated.",
    ),
    _c(
        "deal_age_percentile_overall",
        CATEGORY.DEAL_FEATURE,
        _UNK,
        _QUAR,
        "Same concern as the amount percentile.",
    ),
    _c(
        "deal_is_outlier_amount",
        CATEGORY.DEAL_FEATURE,
        _UNK,
        _QUAR,
        "Outlier flag against an unstated reference population.",
    ),
    _c(
        "deal_size_bucket_num",
        CATEGORY.DEAL_FEATURE,
        _AS_OF,
        _PROOF,
        "Bucketed amount. An as-of fact if the boundaries are fixed, contaminated "
        "if they are quantiles over the whole export. A usable dimension once "
        "confirmed.",
    ),
    _c(
        "deal_amount_vs_segment_avg",
        CATEGORY.DEAL_FEATURE,
        _UNK,
        _QUAR,
        "References a segment column that was not exported, so the grouping cannot "
        "be reproduced or checked.",
    ),
    _c(
        "deal_amount_vs_stage_avg",
        CATEGORY.DEAL_FEATURE,
        _UNK,
        _QUAR,
        "Stage average scope unstated.",
    ),
    _c(
        "deal_age_vs_stage_avg", CATEGORY.DEAL_FEATURE, _UNK, _QUAR, "Stage average scope unstated."
    ),
    _c(
        "pipeline_density_at_stage",
        CATEGORY.DEAL_FEATURE,
        _UNK,
        _QUAR,
        "Density measure with an unstated reference population.",
    ),
    # Historical aggregates
    _c(
        "historical_win_rate_at_stage",
        CATEGORY.HISTORICAL_FEATURE,
        _BACK,
        _PROOF,
        "Preserved rather than recomputed, on the frozen-through-prior-quarter "
        "basis. Lineage unconfirmed. If the historical population includes this "
        "opportunity, the feature is self-referential and any model built on it "
        "is circular, so that remains the check to run.",
        lineage=_LINEAGE_UNCONFIRMED,
    ),
    _c(
        "quarter_close_rate_hist",
        CATEGORY.HISTORICAL_FEATURE,
        _BACK,
        _PROOF,
        "Historical quarter close rate. Preserved rather than recomputed; lineage unconfirmed.",
        lineage=_LINEAGE_UNCONFIRMED,
    ),
    # Temporal anchors
    _c(
        "deal_first_seen_close_date",
        CATEGORY.TEMPORAL,
        _BACK,
        _PROOF,
        "The close date at the opportunity's first observation. Backward-looking "
        "by construction, which makes it a clean baseline for slip computation. "
        "The most useful column in this neighbourhood.",
        recomputable=True,
    ),
    _c(
        "opp_first_commit_as_of",
        CATEGORY.TEMPORAL,
        _UNK,
        _RECOMP,
        "Snapshot at which the opportunity first reached Commit. Contaminated if "
        "it records a commit that postdates this snapshot.",
        recomputable=True,
    ),
    # Movement counters, all recomputable from snapshot history
    _c(
        "is_pushed_out_deal",
        CATEGORY.DERIVED_TEMPORAL,
        _UNK,
        _RECOMP,
        _MOVE_NOTE,
        recomputable=True,
    ),
    _c(
        "is_pulled_in_deal", CATEGORY.DERIVED_TEMPORAL, _UNK, _RECOMP, _MOVE_NOTE, recomputable=True
    ),
    _c(
        "is_qtr_pushed_out_deal",
        CATEGORY.DERIVED_TEMPORAL,
        _UNK,
        _RECOMP,
        "Quarter-crossing push. This is the bridge's slipped_out concept, and the "
        "bridge computes it independently.",
        recomputable=True,
    ),
    _c(
        "is_qtr_pulled_in_deal",
        CATEGORY.DERIVED_TEMPORAL,
        _UNK,
        _RECOMP,
        "Quarter-crossing pull-in. The bridge computes it independently.",
        recomputable=True,
    ),
    _c("qtrs_pulled_in", CATEGORY.DERIVED_TEMPORAL, _UNK, _RECOMP, _MOVE_NOTE, recomputable=True),
    _c(
        "days_since_pulled_into_qtr",
        CATEGORY.DERIVED_TEMPORAL,
        _BACK,
        _RECOMP,
        "Days since the pull-in. Backward-looking if the event precedes as_of.",
        recomputable=True,
    ),
    _c(
        "close_date_push_count",
        CATEGORY.DERIVED_TEMPORAL,
        _BACK,
        _PROOF,
        _CUMULATIVE,
    ),
    _c(
        "close_date_pull_count",
        CATEGORY.DERIVED_TEMPORAL,
        _BACK,
        _PROOF,
        _CUMULATIVE,
    ),
    _c(
        "stage_changes_count",
        CATEGORY.DERIVED_TEMPORAL,
        _BACK,
        _PROOF,
        _CUMULATIVE,
    ),
    _c(
        "stage_regression_count",
        CATEGORY.DERIVED_TEMPORAL,
        _BACK,
        _PROOF,
        _CUMULATIVE,
    ),
    _c(
        "amount_revision_count",
        CATEGORY.DERIVED_TEMPORAL,
        _BACK,
        _PROOF,
        _CUMULATIVE,
    ),
    _c(
        "forecast_category_changes_count",
        CATEGORY.DERIVED_TEMPORAL,
        _BACK,
        _PROOF,
        _CUMULATIVE,
    ),
    _c(
        "close_date_variance",
        CATEGORY.DERIVED_TEMPORAL,
        _BACK,
        _PROOF,
        _CUMULATIVE,
    ),
    _c(
        "close_date_net_movement",
        CATEGORY.DERIVED_TEMPORAL,
        _BACK,
        _PROOF,
        _CUMULATIVE,
    ),
    _c("cd_movement", CATEGORY.DERIVED_TEMPORAL, _UNK, _RECOMP, _MOVE_NOTE, recomputable=True),
    _c(
        "new_amount_change",
        CATEGORY.DERIVED_TEMPORAL,
        _UNK,
        _RECOMP,
        "Change in amount. The bridge computes amount movement independently.",
        recomputable=True,
    ),
    _c(
        "probability_trend",
        CATEGORY.DERIVED_TEMPORAL,
        _BACK,
        _PROOF,
        _CUMULATIVE,
    ),
    _c(
        "amount_trend",
        CATEGORY.DERIVED_TEMPORAL,
        _BACK,
        _PROOF,
        _CUMULATIVE,
    ),
    _c(
        "activity_density",
        CATEGORY.DERIVED_TEMPORAL,
        _UNK,
        _QUAR,
        "Composite activity measure with an undocumented formula and window.",
    ),
    _c(
        "days_in_current_stage",
        CATEGORY.DERIVED_TEMPORAL,
        _BACK,
        _RECOMP,
        "Dwell time in the current stage. Recomputable from stage history within "
        "the export window.",
        recomputable=True,
    ),
    _c(
        "days_stalled",
        CATEGORY.DERIVED_TEMPORAL,
        _BACK,
        _RECOMP,
        "Time without movement. Recomputable; the stall definition needs confirming.",
        recomputable=True,
    ),
    _c(
        "age",
        CATEGORY.DERIVED_TEMPORAL,
        _BACK,
        _PROOF,
        "Opportunity age at this snapshot. The reconstruction source for the "
        "canonical created_date, as as_of minus age.",
    ),
    _c(
        "opp_created_to_commit_days",
        CATEGORY.DERIVED_TEMPORAL,
        _UNK,
        _RECOMP,
        "Created to first commit. Contaminated if the commit postdates as_of.",
        recomputable=True,
    ),
    _c(
        "opp_commit_to_close_days",
        CATEGORY.DERIVED_TEMPORAL,
        _FUTURE,
        _QUAR,
        "'Close' here most likely means the terminal close, which makes this a "
        "cycle-time label rather than a snapshot feature.",
    ),
    # Quarter-phase features, pure calendar arithmetic
    _c(
        "days_since_boq",
        CATEGORY.DERIVED_TEMPORAL,
        _AS_OF,
        _RECOMP,
        _CALENDAR_NOTE,
        recomputable=True,
    ),
    _c(
        "days_to_eoq", CATEGORY.DERIVED_TEMPORAL, _AS_OF, _RECOMP, _CALENDAR_NOTE, recomputable=True
    ),
    _c(
        "quarter_progress_pct",
        CATEGORY.DERIVED_TEMPORAL,
        _AS_OF,
        _RECOMP,
        _CALENDAR_NOTE,
        recomputable=True,
    ),
    _c(
        "is_eoq_sprint",
        CATEGORY.DERIVED_TEMPORAL,
        _AS_OF,
        _RECOMP,
        "End-of-quarter flag. Calendar arithmetic; the threshold needs confirming.",
        recomputable=True,
    ),
    # Close-date dependants, all blocked on the reconstruction
    _c(
        "days_to_close",
        CATEGORY.SNAPSHOT_STATE,
        _AS_OF,
        _PROOF,
        "Reconstructs the canonical close_date as as_of + days_to_close. "
        + _CLOSE_CONFIRMED
        + " Three cross-checks remain available: the reconstructed quarter must "
        "equal close_date_qtr, CD_in_qtr must agree, and eoq_close_diff must "
        "equal the reconstructed date minus quarter end.",
        requires_confirmation=False,
    ),
    _c(
        "CD_in_qtr",
        CATEGORY.DERIVED_TEMPORAL,
        _AS_OF,
        _RECOMP,
        "Close date falls in the snapshot's quarter. A cross-check on the reconstruction.",
        recomputable=True,
    ),
    _c(
        "CD_in_past",
        CATEGORY.DERIVED_TEMPORAL,
        _AS_OF,
        _RECOMP,
        "Close date is in the past relative to as_of.",
        recomputable=True,
    ),
    _c(
        "eoq_close_diff",
        CATEGORY.DERIVED_TEMPORAL,
        _AS_OF,
        _RECOMP,
        "Close date minus quarter end. A second cross-check on the reconstruction.",
        recomputable=True,
    ),
    _c(
        "days_to_close_less10",
        CATEGORY.DERIVED_TEMPORAL,
        _AS_OF,
        _RECOMP,
        _CD_NOTE,
        recomputable=True,
    ),
    _c(
        "days_to_close_vs_remaining",
        CATEGORY.DERIVED_TEMPORAL,
        _AS_OF,
        _RECOMP,
        _CD_NOTE,
        recomputable=True,
    ),
    _c(
        "is_close_before_eoq", CATEGORY.DERIVED_TEMPORAL, _UNK, _RECOMP, _CD_NOTE, recomputable=True
    ),
    _c(
        "eoq_urgency_score",
        CATEGORY.DERIVED_TEMPORAL,
        _UNK,
        _QUAR,
        "Composite urgency score with an undocumented formula.",
    ),
]

COVERAGE: list[CanonicalCoverage] = [
    CanonicalCoverage(
        canonical=CanonicalColumn.AS_OF.value,
        status=CoverageStatus.MAPPED,
        source="as_of",
        verified=True,
        note="Grain column.",
    ),
    CanonicalCoverage(
        canonical=CanonicalColumn.OPP_ID.value,
        status=CoverageStatus.MAPPED,
        source="opp_id",
        verified=True,
        note="Grain column.",
    ),
    CanonicalCoverage(
        canonical=CanonicalColumn.AMOUNT.value,
        status=CoverageStatus.MAPPED,
        source="new_amount",
        verified=True,
        note="Confirmed by the project owner as the deal amount.",
    ),
    CanonicalCoverage(
        canonical=CanonicalColumn.STAGE.value,
        status=CoverageStatus.MAPPED,
        source="Stage",
        note="Present, but the stage vocabulary must be confirmed before "
        "is_closed and is_won are derived from it.",
    ),
    CanonicalCoverage(
        canonical=CanonicalColumn.CLOSE_DATE.value,
        status=CoverageStatus.RECONSTRUCTIBLE,
        source="days_to_close",
        derivation="as_of + days_to_close",
        note="BLOCKING. Required by the canonical schema and absent from the export, "
        "so milestone one's ingestion hard-fails on this file today. Three "
        "independent cross-checks exist: the reconstructed quarter must equal "
        "close_date_qtr, CD_in_qtr must agree, and eoq_close_diff must equal the "
        "reconstructed date minus quarter end.",
    ),
    CanonicalCoverage(
        canonical=CanonicalColumn.CREATED_DATE.value,
        status=CoverageStatus.RECONSTRUCTIBLE,
        source="age",
        derivation="as_of - age",
        note="Optional in the canonical schema. Needed for the created_date creation "
        "basis; the first_seen basis is available regardless.",
    ),
    CanonicalCoverage(
        canonical=CanonicalColumn.IS_CLOSED.value,
        status=CoverageStatus.DERIVABLE,
        source="Stage",
        derivation="stage keyword classification",
        note="Blocked on the stage vocabulary. If stages are open-only or "
        "numbered, the keyword inference silently returns all-open and "
        "every rate metric returns zero.",
    ),
    CanonicalCoverage(
        canonical=CanonicalColumn.IS_WON.value,
        status=CoverageStatus.DERIVABLE,
        source="Stage",
        derivation="stage keyword classification",
        note="Same dependency. terminal_fate is the only other won signal and "
        "is contaminated, so it cannot substitute prospectively.",
    ),
    CanonicalCoverage(
        canonical=CanonicalColumn.FORECAST_CATEGORY.value,
        status=CoverageStatus.MAPPED,
        source="ForecastCategory",
    ),
    CanonicalCoverage(
        canonical=CanonicalColumn.OWNER_ID.value, status=CoverageStatus.MAPPED, source="OwnerID"
    ),
    CanonicalCoverage(
        canonical=CanonicalColumn.ACCOUNT_ID.value,
        status=CoverageStatus.MAPPED,
        source="account_id",
    ),
    CanonicalCoverage(
        canonical=CanonicalColumn.PROBABILITY.value,
        status=CoverageStatus.MAPPED,
        source="Probability",
    ),
    CanonicalCoverage(
        canonical=CanonicalColumn.ARR.value,
        status=CoverageStatus.ABSENT,
        note="No ARR column. The measure resolver already falls back to amount, "
        "so ARR-denominated questions answer in new_amount and say so.",
    ),
    CanonicalCoverage(
        canonical=CanonicalColumn.SEGMENT.value,
        status=CoverageStatus.ABSENT,
        note="No customer segment column, so segment analysis is not answerable. "
        "qtr_segment is most likely quarter phase, not customer segment.",
    ),
    CanonicalCoverage(
        canonical=CanonicalColumn.REGION.value,
        status=CoverageStatus.ABSENT,
        note="No region column.",
    ),
    CanonicalCoverage(
        canonical=CanonicalColumn.INDUSTRY.value,
        status=CoverageStatus.ABSENT,
        note="No industry column.",
    ),
    CanonicalCoverage(
        canonical=CanonicalColumn.OPP_NAME.value,
        status=CoverageStatus.ABSENT,
        note="No opportunity name, so ranked lists identify deals by opp_id.",
    ),
]

ALL_COLUMNS: tuple[str, ...] = (
    "opp_id",
    "as_of",
    "terminal_date",
    "account_id",
    "terminal_fate",
    "terminal_quarter_eoq",
    "ForecastCategory",
    "OwnerID",
    "Stage",
    "Type",
    "RenewalManager",
    "new_amount",
    "Probability",
    "as_of_qtr",
    "close_date_qtr",
    "terminal_date_qtr",
    "as_of_qtr_plus_1",
    "as_of_qtr_plus_2",
    "as_of_qtr_plus_3",
    "pipe_type",
    "train",
    "account_ti_first_won",
    "prev_won",
    "account_ti_first_loss",
    "prev_loss",
    "terminal_amount",
    "rep_open_deal_count",
    "rep_avg_deal_amount",
    "deal_amount_vs_rep_avg",
    "deal_amount_rank_pct",
    "rep_deal_concentration",
    "rep_win_rate",
    "rep_loss_rate",
    "rep_deal_velocity",
    "rep_avg_cycle_time",
    "rep_recent_win_rate",
    "rep_large_deal_win_rate",
    "rep_stage_progression_rate",
    "rep_pipeline_coverage",
    "rep_tenure_proxy",
    "deal_first_seen_close_date",
    "is_pushed_out_deal",
    "is_pulled_in_deal",
    "is_qtr_pulled_in_deal",
    "is_qtr_pushed_out_deal",
    "qtrs_pulled_in",
    "days_since_pulled_into_qtr",
    "opp_first_commit_as_of",
    "opp_created_to_commit_days",
    "opp_commit_to_close_days",
    "rep_active_opp_count_cq",
    "deal_amount_share_of_rep_pipeline_overall",
    "deal_amount_share_of_rep_pipeline_cq",
    "deal_amount_share_of_rep_at_current_stage",
    "deal_size_bucket_num",
    "rep_pushout_rate",
    "rep_pullin_rate",
    "rep_pushout_win_rate",
    "rep_pullin_win_rate",
    "rep_avg_commit_to_close_days",
    "rep_avg_created_to_commit_days",
    "rep_win_rate_at_current_stage",
    "rep_avg_cycle_time_at_current_stage",
    "rep_win_rate_at_current_fc",
    "rep_win_rate_at_deal_size_bucket",
    "stage_changes_count",
    "close_date_push_count",
    "close_date_pull_count",
    "days_in_current_stage",
    "amount_revision_count",
    "stage_regression_count",
    "close_date_variance",
    "close_date_net_movement",
    "days_stalled",
    "forecast_category_changes_count",
    "probability_trend",
    "amount_trend",
    "activity_density",
    "deal_amount_vs_segment_avg",
    "deal_amount_vs_stage_avg",
    "deal_age_vs_stage_avg",
    "pipeline_density_at_stage",
    "account_open_deal_count",
    "historical_win_rate_at_stage",
    "deal_amount_vs_account_avg",
    "deal_amount_percentile_overall",
    "deal_age_percentile_overall",
    "deal_is_outlier_amount",
    "account_historical_deal_count",
    "days_since_boq",
    "days_to_eoq",
    "days_to_close",
    "CD_in_qtr",
    "CD_in_past",
    "qtr_number",
    "age",
    "eoq_close_diff",
    "cd_movement",
    "days_to_close_less10",
    "qtr_segment",
    "new_amount_change",
    "eoq_urgency_score",
    "is_eoq_sprint",
    "quarter_progress_pct",
    "days_to_close_vs_remaining",
    "is_close_before_eoq",
    "quarter_close_rate_hist",
    "AVPNotes_updated_days",
    "Channel_Notes_updated_days",
    "CloseDate_updated_days",
    "Decision_Criteria_updated_days",
    "ForecastCategory_updated_days",
    "Major_Pain_Point_updated_days",
    "ManagerNotes_updated_days",
    "NextStep_updated_days",
    "OwnerID_updated_days",
    "CSMNotes_updated_days",
    "POVNextSteps_updated_days",
    "Paperwork_Process_updated_days",
    "RVPNotes_updated_days",
    "RenewalManager_updated_days",
    "RetentionTeamNotes_updated_days",
    "SENotes_updated_days",
    "Stage_updated_days",
    "Type_updated_days",
    "Why_Do_Anything_updated_days",
    "Why_Now_updated_days",
    "Why_Us_updated_days",
    "new_amount_updated_days",
)


# Open questions recorded as data (ARCHITECTURE 5.15). None is assumed.
UNRESOLVED: list[UnresolvedItem] = [
    # The close-date reconstruction is built and checked, but three facts about
    # its inputs were never established. Recorded as data rather than prose so
    # `open_unresolved()` shows them beside the reconstruction they qualify
    # (ARCHITECTURE 12.19), instead of a reader seeing only a confirmed binding.
    UnresolvedItem(
        id="date_encoding_declarations",
        subject="Which date columns are really Excel serials",
        question=(
            "The owner confirmed both real exports store date fields as Excel serial "
            "numbers. Only as_of has been decoded against real rows. Are terminal_date, "
            "terminal_quarter_eoq, deal_first_seen_close_date and opp_first_commit_as_of "
            "serials too, and is the 1900 date system (base 1899-12-30) the one in use?"
        ),
        columns=(
            "as_of",
            "terminal_date",
            "terminal_quarter_eoq",
            "deal_first_seen_close_date",
            "opp_first_commit_as_of",
        ),
        blocks=(
            "Trusting a decoded date in any column other than as_of. A wrong "
            "declaration fails ingestion loudly, so the risk is a blocked ingest, "
            "not a wrong date; the 1904 system would be wrong silently by 1462 days."
        ),
    ),
    UnresolvedItem(
        id="days_to_close_edge_cases",
        subject="What days_to_close holds outside the ordinary case",
        question=(
            "Is the value ever fractional, what does it hold on deleted and terminal "
            "rows, and is it clamped at zero? Probed on a sibling export only (the "
            "target export was not reachable): an integer on every row, never null, "
            "negative on two thirds of rows so not clamped, and on Closed Won rows "
            "always negative (about 26 to 44 days), so on closed deals it looks like "
            "a realized close date rather than a forecast. That export has no "
            "deletion marker, so deleted rows are still unknown. None of this has "
            "been checked on the target export."
        ),
        columns=("days_to_close", "close_date"),
        blocks=(
            "Trusting the reconstructed close date on rows that are deleted or "
            "already closed. On closed rows it is a realized date, so slip and "
            "pipeline metrics must not read it as a forecast."
        ),
    ),
    UnresolvedItem(
        id="eoq_close_diff_convention",
        subject="Sign convention and rounding of eoq_close_diff",
        question=(
            "On the sibling export eoq_close_diff equals days_to_eoq minus "
            "days_to_close plus an offset of exactly 0, +1 or -1 on every row "
            "(about 50%, 38% and 12%), so the direction looks settled there but the "
            "value is off by a day on half the rows. Is that a time-of-day rounding "
            "effect (the export's as_of carries a time of day) or a real convention, "
            "and does the target export behave the same way?"
        ),
        columns=("eoq_close_diff", "days_to_eoq", "days_to_close", "close_date"),
        blocks=(
            "The close_date_matches_eoq_close_diff agreement test. It is exact, so "
            "an off-by-one on half the rows fails it and withholds "
            "expected_close_date. Whether to allow an explicit one-day tolerance is "
            "a decision for the project owner; none has been added."
        ),
    ),
    UnresolvedItem(
        id="quarter_boundary_day_convention",
        subject="How the stamped in-quarter flag treats a quarter's first and last day",
        question=(
            "On the sibling export CD_in_qtr agrees with the calendar on all but 237 "
            "of 54,999 rows, and every disagreement sits on a quarter's first day "
            "(78) or last day (159), never in the interior, always in the direction "
            "flag-says-out and calendar-says-in. Does the flag exclude boundary "
            "days, or is the reconstructed date a day off there? The two readings "
            "are indistinguishable from this data."
        ),
        columns=("CD_in_qtr", "CD_in_past", "close_date"),
        blocks=(
            "The close_date_matches_cd_in_qtr agreement test, which is exact and so "
            "fails on those rows. It also bounds how far the reconstruction can be "
            "trusted at a quarter edge, which is where pipeline and slip questions "
            "are most sensitive."
        ),
    ),
    UnresolvedItem(
        id="fiscal_year_start_month",
        subject="The tenant's fiscal year start month",
        question=(
            "Which month does the fiscal year begin in? The default is January "
            "and nothing in the export confirms it."
        ),
        columns=("as_of_qtr", "close_date_qtr", "close_date"),
        blocks=(
            "Both quarter-label agreement tests, which compare a computed fiscal "
            "quarter against a stamped one. A wrong start month fails them for a "
            "reason that has nothing to do with the close date."
        ),
    ),
    UnresolvedItem(
        id="target_export_unprobed",
        subject="The target export has never been read",
        question=(
            "Every real-data finding recorded here comes from the sibling tenant's "
            "export. The target export is in S3 and no AWS credentials were "
            "available through the standard chain, so the probe has not run on it "
            "(it stops with 'probe not run' rather than claiming anything). Does "
            "the target behave like the sibling on days_to_close, eoq_close_diff, "
            "CD_in_qtr, date encodings and status?"
        ),
        blocks=(
            "Any production claim about the target export. Run "
            "scripts/probe_production.py against it with credentials from the "
            "standard AWS chain, never as arguments."
        ),
    ),
    UnresolvedItem(
        id="sub_cent_amounts",
        subject="Amounts stored with more than two decimal places",
        question=(
            "On the sibling export 1,576 new_amount values carry more than two "
            "decimal places. Conformance reads the source as text and casts to "
            "DECIMAL(18,2), which rounds; on 7 values that lands a cent away from "
            "rounding the stored DOUBLE directly. Should sub-cent amounts round, "
            "truncate, or be refused, and does the target export carry them too?"
        ),
        blocks=(
            "Cent-exact reconciliation of monetary totals against another system. "
            "Totals are exact under the declared text-then-DECIMAL rule; the rule "
            "itself is not confirmed by the owner."
        ),
    ),
    UnresolvedItem(
        id="opp_commit_to_close_days_definition",
        subject="Definition of opp_commit_to_close_days",
        question=(
            "Does 'close' mean the snapshot's expected close_date, or the terminal "
            "close in terminal_date?"
        ),
        columns=("opp_commit_to_close_days",),
        blocks=(
            "Releasing the column from quarantine. If it measures to terminal_date "
            "it is a retrospective outcome, not a snapshot feature."
        ),
    ),
    UnresolvedItem(
        id="rep_aggregate_lineage",
        subject="Lineage and lookback of the rep aggregates",
        question=(
            "What is each rep_* feature's lookback window, and is each snapshot's "
            "value frozen as of that snapshot's own previous quarter or as of the "
            "end of the dataset? Frozen data is confirmed to run through the "
            "previous quarter, with the last ten quarters available; day-level "
            "data for the current quarter is supplied separately."
        ),
        columns=tuple(sorted(c for c in ALL_COLUMNS if c.startswith("rep_"))),
        blocks=(
            "Treating any rep aggregate as knowable at a historical snapshot. A "
            "freeze at dataset end would embed outcomes that postdate older "
            "snapshots. Lookbacks must not be assumed uniform across the family."
        ),
    ),
]


def _build_registry() -> ColumnRegistry:
    """Family patterns first, then individual assignments."""
    individual = {c.name: c for c in INDIVIDUAL}
    classified: list[ColumnClassification] = []
    for column in ALL_COLUMNS:
        if column in individual:
            classified.append(individual[column])
            continue
        for family in FAMILIES:
            if family.matches(column):
                base = family.classify(column)
                if column in MONETARY_COLUMNS:
                    base = base.model_copy(update={"monetary": MonetaryStatus.MONETARY})
                classified.append(base)
                break
        else:
            raise ValueError(f"column {column!r} matched no family and has no assignment")
    return ColumnRegistry(
        name="opportunity_snapshot_v1",
        monetary_columns=MONETARY_COLUMNS,
        date_encodings=DATE_ENCODINGS,
        columns=classified,
        families=FAMILIES,
        coverage=COVERAGE,
        unresolved=UNRESOLVED,
    )


OPPORTUNITY_SNAPSHOT_V1: ColumnRegistry = _build_registry()
