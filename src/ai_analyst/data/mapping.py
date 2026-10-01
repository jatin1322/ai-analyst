"""Column mapping: source headers onto the canonical schema.

ARCHITECTURE §7.2 places an LLM proposal and a human confirmation here. This
milestone implements the deterministic half: an alias table, normalization, and
a fuzzy fallback that flags itself as requiring confirmation. No LLM call.
"""

from __future__ import annotations

import difflib
import re

from ai_analyst.config import Settings, get_settings
from ai_analyst.contracts.columns import ColumnRegistry, CoverageStatus
from ai_analyst.contracts.errors import (
    AmbiguousMapping,
    FuzzyBinding,
    MappingError,
    MissingRequiredColumns,
    UnconfirmedRequiredMapping,
)
from ai_analyst.contracts.schema import (
    CANONICAL_COLUMNS,
    REQUIRED_COLUMNS,
    CanonicalColumn,
    ColumnMapping,
    MappingConfidence,
    MappingProposal,
)

ALIASES: dict[CanonicalColumn, set[str]] = {
    CanonicalColumn.AS_OF: {
        "as_of", "as_of_date", "asof", "asof_date", "snapshot_date", "snapshot",
        "snapshot_dt", "report_date", "reporting_date", "effective_date",
    },
    CanonicalColumn.OPP_ID: {
        "opp_id", "opportunity_id", "oppid", "opportunity", "opp", "deal_id",
        "opportunity_key", "opp_number",
    },
    CanonicalColumn.CLOSE_DATE: {
        "close_date", "expected_close_date", "closedate", "forecast_close_date",
        "expected_close", "projected_close_date", "close_dt",
    },
    CanonicalColumn.STAGE: {
        "stage", "sales_stage", "stage_name", "opportunity_stage", "deal_stage",
        "current_stage",
    },
    CanonicalColumn.AMOUNT: {
        "amount", "deal_amount", "opportunity_amount", "value", "deal_value",
        "total_amount", "bookings", "new_amount",
    },
    CanonicalColumn.CREATED_DATE: {
        "created_date", "created", "create_date", "created_at", "date_created",
        "opportunity_created_date",
    },
    CanonicalColumn.IS_CLOSED: {"is_closed", "closed", "isclosed", "closed_flag"},
    CanonicalColumn.IS_WON: {"is_won", "won", "iswon", "won_flag"},
    CanonicalColumn.FORECAST_CATEGORY: {
        "forecast_category", "forecast_cat", "forecast", "forecastcategory",
        "forecast_category_name",
    },
    CanonicalColumn.ARR: {
        "arr", "annual_recurring_revenue", "annual_recurring_rev", "recurring_revenue",
        "annualized_revenue",
    },
    CanonicalColumn.SEGMENT: {
        "segment", "customer_segment", "account_segment", "market_segment",
        "business_segment",
    },
    CanonicalColumn.REGION: {"region", "sales_region", "geo", "territory", "area"},
    CanonicalColumn.INDUSTRY: {"industry", "vertical", "account_industry", "sector"},
    CanonicalColumn.OWNER_ID: {
        "owner_id", "owner", "opportunity_owner", "sales_rep", "rep", "account_executive",
        "ae",
    },
    CanonicalColumn.ACCOUNT_ID: {
        "account_id", "account", "customer_id", "account_key", "account_name",
        "customer_name",
    },
    CanonicalColumn.OPP_NAME: {
        "opp_name", "opportunity_name", "deal_name", "name", "opportunity_title",
    },
    CanonicalColumn.PROBABILITY: {
        "probability", "win_probability", "prob", "probability_pct", "likelihood",
    },
}

_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def normalize(header: str) -> str:
    """Lowercase, collapse punctuation to underscores, strip edges."""
    return _NORMALIZE_RE.sub("_", header.strip().lower()).strip("_")


_ALIAS_INDEX: dict[str, CanonicalColumn] = {}
for _canonical, _alias_set in ALIASES.items():
    for _alias in _alias_set:
        _ALIAS_INDEX[normalize(_alias)] = _canonical


def propose_mapping(
    source_columns: list[str], settings: Settings | None = None
) -> MappingProposal:
    """Propose a canonical mapping for the given source headers.

    Exact canonical names win, then aliases, then a fuzzy fallback which is
    always flagged as requiring confirmation. A canonical column is claimed at
    most once, by the best-scoring candidate.
    """
    settings = settings or get_settings()
    claimed: dict[CanonicalColumn, ColumnMapping] = {}
    unmapped: list[str] = []
    fuzzy: list[ColumnMapping] = []

    def _offer(candidate: ColumnMapping) -> None:
        existing = claimed.get(candidate.canonical_column)
        if existing is None or candidate.score > existing.score:
            if existing is not None:
                unmapped.append(existing.source_column)
            claimed[candidate.canonical_column] = candidate
        else:
            unmapped.append(candidate.source_column)

    deferred_fuzzy: list[str] = []

    for source in source_columns:
        norm = normalize(source)
        canonical_names = {c.value for c in CanonicalColumn}
        if norm in canonical_names:
            _offer(
                ColumnMapping(
                    source_column=source,
                    canonical_column=CanonicalColumn(norm),
                    confidence=MappingConfidence.EXACT,
                    score=1.0,
                )
            )
        elif norm in _ALIAS_INDEX:
            _offer(
                ColumnMapping(
                    source_column=source,
                    canonical_column=_ALIAS_INDEX[norm],
                    confidence=MappingConfidence.ALIAS,
                    score=0.95,
                )
            )
        else:
            deferred_fuzzy.append(source)

    # Fuzzy matching runs last so it can only fill columns nothing claimed
    # exactly. This keeps a near-miss from displacing a real alias hit.
    for source in deferred_fuzzy:
        norm = normalize(source)
        best_target: CanonicalColumn | None = None
        best_score = 0.0
        for alias_norm, canonical in _ALIAS_INDEX.items():
            if canonical in claimed:
                continue
            score = difflib.SequenceMatcher(None, norm, alias_norm).ratio()
            if score > best_score:
                best_score, best_target = score, canonical
        if best_target is not None and best_score >= settings.fuzzy_match_threshold:
            # Recorded as a proposal, never applied. The source column stays
            # unmapped so it survives ingestion as a discovered column rather
            # than being consumed by a guess.
            fuzzy.append(
                ColumnMapping(
                    source_column=source,
                    canonical_column=best_target,
                    confidence=MappingConfidence.FUZZY,
                    score=round(best_score, 4),
                )
            )
        unmapped.append(source)

    # A required column with a near-miss candidate is reported as *unconfirmed*
    # rather than *missing*, because the two need different answers from a
    # human: confirm this binding, versus supply the column.
    guessed = {m.canonical_column for m in fuzzy}
    missing = [c for c in REQUIRED_COLUMNS if c not in claimed and c not in guessed]
    return MappingProposal(
        mappings=sorted(claimed.values(), key=lambda m: m.canonical_column.value),
        unmapped_source_columns=sorted(set(unmapped)),
        missing_required=missing,
        fuzzy_candidates=sorted(fuzzy, key=lambda m: m.canonical_column.value),
    )


def mapping_from_overrides(
    source_columns: list[str], overrides: dict[str, CanonicalColumn]
) -> MappingProposal:
    """Build a mapping entirely from user-supplied bindings.

    Used when a human corrects or replaces the proposal. Every binding is
    recorded at USER confidence and never requires confirmation.
    """
    available = set(source_columns)
    unknown = sorted(set(overrides) - available)
    if unknown:
        raise MappingError(
            AmbiguousMapping(
                message=f"override references columns not present in source: {unknown}",
                canonical_column=CanonicalColumn.OPP_ID,
                candidates=unknown,
            )
        )
    mappings = [
        ColumnMapping(
            source_column=source,
            canonical_column=canonical,
            confidence=MappingConfidence.USER,
            score=1.0,
        )
        for source, canonical in overrides.items()
    ]
    claimed = {m.canonical_column for m in mappings}
    return MappingProposal(
        mappings=sorted(mappings, key=lambda m: m.canonical_column.value),
        unmapped_source_columns=sorted(available - set(overrides)),
        missing_required=[c for c in REQUIRED_COLUMNS if c not in claimed],
    )


def mapping_from_registry(
    registry: ColumnRegistry, source_columns: list[str]
) -> MappingProposal:
    """Derive the mapping from an export registry's own coverage records.

    The registry already documents which source column satisfies each canonical
    field, so the mapping and the coverage cannot disagree. Nothing here is
    fuzzy: a canonical column is bound only where the registry declares a
    source that is present, or where a source header equals the canonical name
    exactly. That is what keeps a header like a quarter label from being taken
    for a date.
    """
    available = set(source_columns)
    claimed: dict[CanonicalColumn, ColumnMapping] = {}

    for record in registry.coverage:
        if record.status is not CoverageStatus.MAPPED or record.source not in available:
            continue
        canonical = CanonicalColumn(record.canonical)
        claimed[canonical] = ColumnMapping(
            source_column=record.source,
            canonical_column=canonical,
            confidence=MappingConfidence.USER,
            score=1.0,
        )

    for source in source_columns:
        norm = normalize(source)
        if norm in {c.value for c in CanonicalColumn}:
            canonical = CanonicalColumn(norm)
            if canonical not in claimed:
                claimed[canonical] = ColumnMapping(
                    source_column=source,
                    canonical_column=canonical,
                    confidence=MappingConfidence.EXACT,
                    score=1.0,
                )

    bound = {m.source_column for m in claimed.values()}
    return MappingProposal(
        mappings=sorted(claimed.values(), key=lambda m: m.canonical_column.value),
        unmapped_source_columns=sorted(available - bound),
        missing_required=[c for c in REQUIRED_COLUMNS if c not in claimed],
    )


def assert_mappable(
    proposal: MappingProposal,
    source_columns: list[str],
    *,
    hints: dict[CanonicalColumn, str] | None = None,
) -> None:
    """Fail loudly when a required canonical field is absent or only guessed.

    `hints` lets a caller explain why a missing column is not simply gone, for
    example that the registry documents how it could be reconstructed.
    """
    if proposal.missing_required:
        names = ", ".join(c.value for c in proposal.missing_required)
        extra = " ".join(
            hints[c] for c in proposal.missing_required if hints and c in hints
        )
        raise MappingError(
            MissingRequiredColumns(
                message=(
                    f"cannot ingest: required canonical columns are unmapped: {names}. "
                    "The system never guesses a primary key." + (f" {extra}" if extra else "")
                ),
                missing=list(proposal.missing_required),
                available_source_columns=sorted(source_columns),
            )
        )
    if proposal.unconfirmed_required:
        listing = ", ".join(
            f"{m.canonical_column.value} <- {m.source_column!r} (score {m.score})"
            for m in proposal.unconfirmed_required
        )
        raise MappingError(
            UnconfirmedRequiredMapping(
                message=(
                    "cannot ingest: required columns are bound only by a fuzzy header "
                    f"match that nobody has confirmed: {listing}. Supply mapping_overrides "
                    "or a column registry."
                ),
                bindings=[
                    FuzzyBinding(
                        source_column=m.source_column,
                        canonical_column=m.canonical_column,
                        score=m.score,
                    )
                    for m in proposal.unconfirmed_required
                ],
            )
        )


def describe_canonical_columns() -> str:
    """Human-readable canonical schema, for error messages and docs."""
    lines = []
    for name, spec in CANONICAL_COLUMNS.items():
        flag = "required" if spec.required else "optional"
        lines.append(f"{name.value:20} {spec.dtype.value:8} {flag:9} {spec.description}")
    return "\n".join(lines)
