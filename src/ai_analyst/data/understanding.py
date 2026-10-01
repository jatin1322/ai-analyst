"""The dataset understanding layer (ARCHITECTURE 12.4).

Runs after ingestion and profiling, reads only artifacts that already exist,
and produces the dataset's analyst context. It performs **no LLM call**: it
assembles evidence and applies the binding rules of 12.2.

Relationships between columns are deliberately not precomputed. Over 145
columns the pairwise space is mostly noise, and a stored correlation invites a
reader to find causation in an artifact nobody asked for. They are computed on
request instead.
"""

from __future__ import annotations

import duckdb
from pydantic import BaseModel, ConfigDict

from ai_analyst.config import Settings, get_settings
from ai_analyst.contracts.agreement import AgreementReport
from ai_analyst.contracts.binding import ConceptBindings
from ai_analyst.contracts.columns import Disposition, InformationClass
from ai_analyst.contracts.concepts import (
    CONCEPTS,
    AnalyticalOperation,
    ConceptRequirement,
)
from ai_analyst.contracts.context import (
    AnalystContext,
    ColumnDetail,
    ConceptCard,
    QualityWarning,
    TextFieldCard,
)
from ai_analyst.contracts.dataset import DatasetRegistry
from ai_analyst.contracts.profile import DatasetProfile, ProfileKind
from ai_analyst.contracts.schema import DatasetSchema
from ai_analyst.contracts.tenant import TenantProfile
from ai_analyst.data.binding import build_bindings
from ai_analyst.data.conform import quote_ident
from ai_analyst.data.grants import load_grants
from ai_analyst.data.reconstruct import evaluate_reconstruction_agreement
from ai_analyst.data.store import DuckDBStore

# Categorical columns above this many distinct values are not useful as a
# group-by and are left out of the dimension list.
MAX_DIMENSION_CARDINALITY = 50


def _fiscal_month(tenant: TenantProfile | None, settings: Settings) -> int:
    """The fiscal year start in force: the tenant's declaration, else the default."""
    if tenant is not None and tenant.fiscal_year_start_month is not None:
        return tenant.fiscal_year_start_month
    return settings.fiscal_year_start_month


def evaluate_agreement(
    conn: duckdb.DuckDBPyConnection,
    scan: str,
    schema: DatasetSchema,
    *,
    dataset_id: str,
    settings: Settings | None = None,
) -> AgreementReport:
    """Run every agreement test this dataset can support, from its own rows."""
    settings = settings or get_settings()
    return evaluate_reconstruction_agreement(
        conn,
        scan,
        dataset_id=dataset_id,
        reconstructed={d.column for d in schema.derived_columns if d.is_reconstructed},
        fiscal_year_start_month=settings.fiscal_year_start_month,
    )


def _dimensions(registry: DatasetRegistry, profile: DatasetProfile) -> tuple[str, ...]:
    """Categorical columns a group-by could legitimately use."""
    by_name = registry.by_name()
    out: list[str] = []
    for column in profile.columns:
        classified = by_name.get(column.name)
        if classified is None or classified.is_quarantined:
            continue
        if column.kind not in (ProfileKind.CATEGORICAL, ProfileKind.BOOLEAN):
            continue
        if column.distinct_count > MAX_DIMENSION_CARDINALITY or column.distinct_count <= 1:
            continue
        out.append(column.name)
    return tuple(out)


def _measures(registry: DatasetRegistry, profile: DatasetProfile) -> tuple[str, ...]:
    """Numeric columns that are not withheld."""
    by_name = registry.by_name()
    return tuple(
        c.name
        for c in profile.columns
        if c.kind is ProfileKind.NUMERIC
        and (col := by_name.get(c.name)) is not None
        and not col.is_quarantined
        and col.classification.disposition is not Disposition.QUARANTINE
    )


def _text_fields(
    registry: DatasetRegistry, profile: DatasetProfile
) -> tuple[TextFieldCard, ...]:
    by_name = registry.by_name()
    by_profile = {c.name: c for c in profile.columns}
    cards: list[TextFieldCard] = []
    for entry in profile.text_columns:
        column = by_name.get(entry.name)
        stats = by_profile.get(entry.name)
        cards.append(
            TextFieldCard(
                name=entry.name,
                classified=column is not None and column.classification.classified,
                knowable_at_snapshot=(
                    column.classification.knowable_at_snapshot if column else False
                ),
                null_rate=round(stats.null_rate, 3) if stats else 0.0,
                mean_length=stats.mean_length if stats else None,
                max_length=stats.max_length if stats else None,
            )
        )
    return tuple(cards)


def _column_index(
    registry: DatasetRegistry, max_columns: int
) -> tuple[dict[str, tuple[str, ...]], bool]:
    """Names grouped by information class, dropped when the dataset is too wide."""
    if len(registry.columns) > max_columns:
        return {}, True
    index: dict[str, list[str]] = {}
    for column in registry.columns:
        index.setdefault(column.information_class.value, []).append(column.name)
    return {k: tuple(v) for k, v in sorted(index.items())}, False


def _quality_warnings(
    profile: DatasetProfile, registry: DatasetRegistry, schema: DatasetSchema
) -> tuple[QualityWarning, ...]:
    warnings: list[QualityWarning] = []
    quality = getattr(profile, "quality", None)
    for flag, detail in (
        ("negative_amounts", "rows carry a negative amount"),
        ("close_before_created", "rows close before they were created"),
    ):
        count = getattr(quality, flag, 0) if quality else 0
        if count:
            warnings.append(QualityWarning(code=flag, detail=f"{count} {detail}"))
    vanished = getattr(quality, "vanished_without_terminal_state", 0) if quality else 0
    if vanished:
        warnings.append(
            QualityWarning(
                code="vanished_without_terminal_state",
                detail=(
                    f"{vanished} opportunities disappear from the snapshots without "
                    "closing. Every derived rate is suspect while this is large."
                ),
            )
        )
    if not schema.status_is_authoritative:
        warnings.append(
            QualityWarning(
                code="status_not_authoritative",
                detail=(
                    "Open, won and lost were inferred from stage keywords. A "
                    "vocabulary can encode an outcome in a label this cannot read."
                ),
            )
        )
    unclassified = registry.unclassified()
    if unclassified:
        warnings.append(
            QualityWarning(
                code="unclassified_columns",
                detail=(
                    f"{len(unclassified)} columns nothing could classify. They are "
                    "inspectable, usable as filters, and never usable as a measure."
                ),
            )
        )
    return tuple(warnings)


def build_context(
    schema: DatasetSchema,
    registry: DatasetRegistry,
    profile: DatasetProfile,
    bindings: ConceptBindings,
    agreement: AgreementReport | None = None,
    settings: Settings | None = None,
    tenant: TenantProfile | None = None,
) -> AnalystContext:
    """Project a dataset's artifacts into the tier-0 card."""
    settings = settings or get_settings()
    month = _fiscal_month(tenant, settings)
    snapshots = sorted(s.as_of for s in profile.snapshots)
    class_counts: dict[str, int] = {}
    for column in registry.columns:
        key = column.information_class.value
        class_counts[key] = class_counts.get(key, 0) + 1

    index, degraded = _column_index(registry, settings.context_card_max_indexed_columns)
    resolution = schema.status_resolution

    cards = tuple(
        ConceptCard(
            concept=b.concept,
            display_name=CONCEPTS[b.concept].display_name,
            definition=CONCEPTS[b.concept].definition,
            columns=b.columns,
            status=b.status,
            alternatives=b.alternatives,
            caveats=b.caveats,
            note=b.note,
        )
        for b in bindings.bindings
    )

    return AnalystContext(
        dataset_id=registry.dataset_id,
        row_count=profile.row_count,
        snapshot_count=len(profile.snapshots),
        first_snapshot=snapshots[0] if snapshots else None,
        last_snapshot=snapshots[-1] if snapshots else None,
        fiscal_year_start_month=month,
        fiscal_calendar_resolved=bool(tenant and tenant.calendar_resolved),
        reconstruction_verdicts=tuple(
            f"{v.concept.value}: {v.verdict.value}"
            + (f" ({'; '.join(v.reasons)})" if v.reasons else "")
            for v in bindings.reconstruction_verdicts
        ),
        concepts=cards,
        dimensions=_dimensions(registry, profile),
        measures=_measures(registry, profile),
        text_fields=_text_fields(registry, profile),
        column_count=len(registry.columns),
        class_counts=class_counts,
        quarantined_count=len(registry.quarantined()),
        column_index=index,
        index_degraded=degraded,
        status_strategy=resolution.strategy if resolution else None,
        status_is_authoritative=bool(resolution and resolution.is_authoritative),
        open_questions=tuple(u.id for u in registry.open_unresolved()),
        quality_warnings=_quality_warnings(profile, registry, schema),
        agreement=tuple(agreement.results) if agreement else (),
    )


def column_detail(
    name: str,
    registry: DatasetRegistry,
    profile: DatasetProfile,
    bindings: ConceptBindings | None = None,
) -> ColumnDetail:
    """Tier 1: everything known about one column."""
    column = registry.get(name)
    classification = column.classification
    stats = {c.name: c for c in profile.columns}.get(name)
    bound_to = None
    if bindings is not None:
        bound_to = next(
            (b.concept for b in bindings.bindings if name in b.columns), None
        )
    return ColumnDetail(
        name=name,
        origin=column.origin.value,
        source_name=column.source_name,
        dtype=column.dtype,
        category=classification.category.value,
        information_class=column.information_class.value,
        availability=classification.availability.value,
        disposition=classification.disposition.value,
        knowable_at_snapshot=classification.knowable_at_snapshot,
        quarantined=column.is_quarantined,
        quarantine_code=column.quarantine.code.value if column.quarantine else None,
        quarantine_detail=column.quarantine.detail if column.quarantine else None,
        monetary=classification.monetary_status,
        lineage_confirmed=(
            classification.lineage.confirmed if classification.lineage else None
        ),
        unresolved=tuple(u.id for u in registry.unresolved_for(name)),
        bound_to=bound_to,
        profile=stats,
    )


def list_columns(
    registry: DatasetRegistry,
    information_class: InformationClass | None = None,
    pattern: str | None = None,
) -> list[str]:
    """Names only, filtered. The path to a wide dataset's columns."""
    names = [
        c.name
        for c in registry.columns
        if information_class is None or c.information_class is information_class
    ]
    if pattern:
        needle = pattern.lower()
        names = [n for n in names if needle in n.lower()]
    return names


class DatasetUnderstanding(BaseModel):
    """Everything the understanding layer derived for one dataset.

    Assembled in one pass so the agreement tests run once, at understanding
    time, rather than being re-evaluated mid-conversation.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    dataset_id: str
    bindings: ConceptBindings
    agreement: AgreementReport
    context: AnalystContext

    def card(self) -> str:
        """The tier-0 card, as it enters the prompt."""
        return self.context.render()

    def check(self, operation: AnalyticalOperation) -> ConceptRequirement:
        """Whether one analytical operation is answerable on this dataset."""
        return self.bindings.check_operation(operation)

    def unanswerable(self) -> dict[AnalyticalOperation, str]:
        """Every operation this dataset cannot support, and why.

        The reason is phrased as the milestone asks, so a later layer reports
        "amount concept unavailable" rather than failing a query.
        """
        out: dict[AnalyticalOperation, str] = {}
        for operation in AnalyticalOperation:
            result = self.check(operation)
            if not result.satisfied:
                out[operation] = result.reason
        return out


def understand(
    dataset,
    tenant: TenantProfile | None = None,
    settings: Settings | None = None,
    store: DuckDBStore | None = None,
) -> DatasetUnderstanding:
    """Run the understanding layer over an ingested, profiled dataset.

    Reads only artifacts that already exist plus the dataset's own rows, and
    makes no LLM call.
    """
    settings = settings or get_settings()
    if dataset.profile is None:
        raise ValueError(f"dataset {dataset.dataset_id!r} must be profiled first")

    report = dataset.agreement
    month = _fiscal_month(tenant, settings)
    stale = report is not None and report.fiscal_year_start_month != month
    if report is None or stale:
        # Nothing was persisted, or it was evaluated under another fiscal calendar.
        store = store or DuckDBStore(settings)
        with store.connect() as conn:
            report = evaluate_agreement(
                conn,
                store.snapshots_scan(dataset.dataset_id),
                dataset.schema,
                dataset_id=dataset.dataset_id,
                settings=settings.model_copy(update={"fiscal_year_start_month": month}),
            )
    # The persisted ledger, validated against this dataset and tenant. Loading
    # fails loudly on a mismatched, stale, corrupted or contradicted ledger.
    ledger = (
        load_grants(
            dataset.dataset_id, tenant=tenant, registry=dataset.registry, settings=settings
        )
        if tenant is not None
        else None
    )
    bindings = build_bindings(
        dataset.schema,
        dataset.registry,
        dataset.profile,
        tenant=tenant,
        agreement=report,
        grants=ledger,
    )
    context = build_context(
        dataset.schema, dataset.registry, dataset.profile, bindings, report, settings, tenant
    )
    return DatasetUnderstanding(
        dataset_id=dataset.dataset_id,
        bindings=bindings,
        agreement=report,
        context=context,
    )


class SampleRows(BaseModel):
    """Tier 2: a bounded sample, reachable only by asking for it (12.5).

    Rows never enter the default context card. This is the explicit path, and
    it is narrow on purpose: a hard row cap, free-text columns excluded so no
    CRM narrative reaches a prompt through the side door, and every withheld
    column named so the caller knows the sample is partial.
    """

    model_config = ConfigDict(frozen=True)

    dataset_id: str
    columns: tuple[str, ...]
    rows: tuple[dict[str, str | None], ...]
    excluded_text_columns: tuple[str, ...] = ()
    excluded_quarantined_columns: tuple[str, ...] = ()
    row_limit: int = 0

    def render(self) -> str:
        header = " | ".join(self.columns)
        body = [
            " | ".join("" if r.get(c) is None else str(r.get(c)) for c in self.columns)
            for r in self.rows
        ]
        out = [header, "-" * len(header), *body]
        if self.excluded_text_columns:
            out.append(
                f"({len(self.excluded_text_columns)} narrative columns withheld: "
                "content is never sampled)"
            )
        if self.excluded_quarantined_columns:
            out.append(
                f"({len(self.excluded_quarantined_columns)} quarantined columns withheld)"
            )
        return "\n".join(out)


def sample_rows(
    dataset,
    limit: int = 5,
    settings: Settings | None = None,
    store: DuckDBStore | None = None,
) -> SampleRows:
    """Read a capped sample of one dataset's rows.

    Not an agent tool and not wired into any prompt: it is the data-layer path
    a tier-2 inspection tool would later call. A bounded sample is not the
    dataset, which is why it exists at all, but it is still the only thing here
    that returns real rows, so the caps are not optional.
    """
    settings = settings or get_settings()
    store = store or DuckDBStore(settings)
    capped = max(1, min(int(limit), settings.max_sample_rows))

    text = {c.name for c in dataset.registry.text_columns()}
    quarantined = {c.name for c in dataset.registry.quarantined()}
    selected = [
        c.name for c in dataset.registry.columns if c.name not in text and c.name not in quarantined
    ]

    projection = ", ".join(quote_ident(c) for c in selected)
    with store.connect() as conn:
        raw = conn.execute(
            f"SELECT {projection} FROM {store.snapshots_scan(dataset.dataset_id)} LIMIT {capped}"
        ).fetchall()

    return SampleRows(
        dataset_id=dataset.dataset_id,
        columns=tuple(selected),
        rows=tuple(
            {c: (None if v is None else str(v)) for c, v in zip(selected, row, strict=True)}
            for row in raw
        ),
        excluded_text_columns=tuple(sorted(text)),
        excluded_quarantined_columns=tuple(sorted(quarantined)),
        row_limit=capped,
    )
