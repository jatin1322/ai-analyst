"""Ingestion: source file to canonical partitioned Parquet.

Pipeline (ARCHITECTURE §7.2):
    read -> map -> conform -> grain assertion -> write -> persist schema

The grain assertion is a hard failure, not a warning. If (as_of, opp_id) is not
unique then every snapshot metric in the system is silently wrong.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import duckdb

from ai_analyst.config import Settings, get_settings
from ai_analyst.contracts.agreement import AgreementReport
from ai_analyst.contracts.columns import ColumnRegistry, CoverageStatus
from ai_analyst.contracts.dataset import DatasetRegistry
from ai_analyst.contracts.errors import (
    CaptureResolutionFailed,
    DateEncodingInvalid,
    DuplicateKey,
    GrainViolation,
    IngestionError,
    NullGrainKey,
    StatusConfigInvalid,
    UnreadableSource,
)
from ai_analyst.contracts.schema import (
    CANONICAL_COLUMNS,
    REQUIRED_COLUMNS,
    CanonicalColumn,
    ColumnMapping,
    ColumnSpec,
    DatasetSchema,
    DataType,
    DateEncoding,
    MappingConfidence,
    MappingProposal,
)
from ai_analyst.contracts.snapshot_policy import (
    CaptureResolution,
    IntradayPolicy,
    SnapshotGranularity,
    SnapshotPolicy,
)
from ai_analyst.contracts.source import SourceFormat, TableSource
from ai_analyst.contracts.status import OpportunityStatus, StatusMapping, StatusStrategy
from ai_analyst.contracts.tenant import FamilyInvariant
from ai_analyst.data.classify import build_dataset_registry
from ai_analyst.data.conform import (
    build_cast_failure_select,
    build_select,
    build_type_detection_select,
    detect_types,
    discovered_columns,
    quote_ident,
    quote_literal,
    typed_discovered_type,
)
from ai_analyst.data.dates import measure_date_conversions
from ai_analyst.data.invariants import evaluate_family_invariants, with_invariants
from ai_analyst.data.mapping import (
    assert_mappable,
    mapping_from_overrides,
    mapping_from_registry,
    propose_mapping,
)
from ai_analyst.data.reconstruct import (
    evaluate_reconstruction_agreement,
    plan_reconstructions,
)
from ai_analyst.data.sources import ProbeAccessError, prepare_source, scan_sql
from ai_analyst.data.store import DuckDBStore

CONFORMED_TABLE = "conformed"
# Hidden helper column: the raw capture timestamp, present only while a declared
# `latest_capture` policy is being applied. Dropped before anything is written.
CAPTURE_TS = "_capture_ts"


@dataclass(frozen=True)
class IngestionResult:
    dataset_id: str
    schema: DatasetSchema
    registry: DatasetRegistry
    row_count: int
    snapshot_count: int
    canonical_path: Path
    # Agreement tests run over reconstructed columns, persisted beside the
    # dataset. Empty when nothing was reconstructed.
    agreement: AgreementReport | None = None


def _local_single_file_format(path: Path) -> SourceFormat:
    """Classify a local file by extension only. `.txt` is treated as CSV."""
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv", ".txt"}:
        return SourceFormat.CSV
    if suffix == ".parquet":
        return SourceFormat.PARQUET
    raise IngestionError(
        UnreadableSource(
            message=f"unsupported source extension {suffix!r}; expected .csv or .parquet",
            path=str(path),
            reason="unsupported_extension",
        )
    )


def _coerce_source(source_path: str | Path | TableSource) -> TableSource:
    """Normalise `ingest`'s `source_path` argument into a `TableSource`.

    A bare path or string is treated as one local CSV or Parquet file, exactly
    as before this milestone: existence is checked up front (a clear
    `not_found` error beats a confusing DuckDB failure), and its format comes
    from its extension alone, never from sniffing content. A `TableSource`
    passed in directly (a partitioned Parquet directory, a Delta table, or an
    `s3://` location) is used as given; its existence is checked when the
    source is actually read.
    """
    if isinstance(source_path, TableSource):
        return source_path
    path = Path(source_path)
    if not path.exists():
        raise IngestionError(
            UnreadableSource(
                message=f"source file not found: {path}",
                path=str(path),
                reason="not_found",
            )
        )
    fmt = _local_single_file_format(path)
    return TableSource(format=fmt, uri=str(path.resolve()))


def _local_single_file(source: TableSource) -> Path | None:
    """The local file backing `source`, or None for a directory/remote source."""
    if source.format not in (SourceFormat.CSV, SourceFormat.PARQUET):
        return None
    if source.uri.startswith("s3://"):
        return None
    return Path(source.uri)


def read_source_columns(
    source: str | Path | TableSource, store: DuckDBStore | None = None
) -> list[str]:
    """Return the source's headers without ingesting it."""
    table_source = _coerce_source(source)
    store = store or DuckDBStore()
    with store.connect() as conn:
        try:
            prepare_source(conn, table_source)
            rel = conn.sql(f"SELECT * FROM {scan_sql(table_source)} LIMIT 0")
            return list(rel.columns)
        except (duckdb.Error, ProbeAccessError) as exc:
            raise IngestionError(
                UnreadableSource(
                    message=f"could not read {table_source.uri}: {exc}",
                    path=table_source.uri,
                    reason=str(exc),
                )
            ) from exc


def _assert_grain(conn: duckdb.DuckDBPyConnection, settings: Settings) -> None:
    """Hard-fail on null keys or duplicate (as_of, opp_id) pairs."""
    _assert_no_null_keys(conn)
    _assert_unique_keys(conn, settings)


def _assert_no_null_keys(conn: duckdb.DuckDBPyConnection) -> None:
    null_as_of, null_opp_id = conn.execute(
        f"SELECT COUNT(*) FILTER (WHERE as_of IS NULL), "
        f"COUNT(*) FILTER (WHERE opp_id IS NULL) FROM {CONFORMED_TABLE}"
    ).fetchone()
    if null_as_of or null_opp_id:
        raise IngestionError(
            NullGrainKey(
                message=(
                    f"grain columns contain nulls or unparseable values: "
                    f"{null_as_of} row(s) with null as_of, "
                    f"{null_opp_id} row(s) with null opp_id"
                ),
                null_as_of_rows=int(null_as_of),
                null_opp_id_rows=int(null_opp_id),
            )
        )


def _assert_unique_keys(conn: duckdb.DuckDBPyConnection, settings: Settings) -> None:
    duplicates = conn.execute(
        f"""
        SELECT as_of, opp_id, COUNT(*) AS n
        FROM {CONFORMED_TABLE}
        GROUP BY as_of, opp_id
        HAVING COUNT(*) > 1
        ORDER BY n DESC, as_of, opp_id
        """
    ).fetchall()
    if not duplicates:
        return

    offending_rows = sum(int(row[2]) for row in duplicates)
    samples = [
        DuplicateKey(as_of=row[0], opp_id=str(row[1]), row_count=int(row[2]))
        for row in duplicates[: settings.max_duplicate_samples]
    ]
    raise IngestionError(
        GrainViolation(
            message=(
                f"(as_of, opp_id) is not unique: {len(duplicates)} duplicated key(s) "
                f"across {offending_rows} rows. Every snapshot metric would be wrong, "
                "so ingestion stopped."
            ),
            duplicate_key_count=len(duplicates),
            offending_row_count=offending_rows,
            samples=samples,
            sample_limit=settings.max_duplicate_samples,
        )
    )


def _resolve_captures(
    conn: duckdb.DuckDBPyConnection,
    settings: Settings,
    capture_column: str,
) -> CaptureResolution:
    """Apply the declared `latest_capture` policy to the conformed table.

    Among rows sharing (as_of date, opp_id) the latest capture timestamp is
    kept. A group whose latest timestamp is shared by several rows, or that has
    a missing timestamp, cannot be resolved (a date-only source is the common
    case): that is a hard failure, never a silent pick. Groups whose kept and
    dropped rows differ in any conformed column are counted as conflicts.
    """
    payload = [
        r[0]
        for r in conn.execute(
            f"SELECT column_name FROM information_schema.columns "
            f"WHERE table_name = '{CONFORMED_TABLE}' AND column_name <> '{CAPTURE_TS}' "
            "ORDER BY ordinal_position"
        ).fetchall()
    ]
    row_struct = "ROW(" + ", ".join(quote_ident(c) for c in payload) + ")"
    ts = quote_ident(CAPTURE_TS)
    groups = conn.execute(
        f"""
        SELECT as_of, opp_id,
               COUNT(*) AS n,
               COUNT(*) FILTER (WHERE {ts} IS NULL) AS null_ts,
               COUNT(*) FILTER (WHERE {ts} = mx) AS at_latest,
               COUNT(DISTINCT {row_struct}) AS distinct_rows
        FROM (SELECT *, MAX({ts}) OVER (PARTITION BY as_of, opp_id) AS mx
              FROM {CONFORMED_TABLE})
        GROUP BY as_of, opp_id
        HAVING COUNT(*) > 1
        ORDER BY as_of, opp_id
        """
    ).fetchall()

    unresolvable = [g for g in groups if g[3] > 0 or g[4] != 1]
    if unresolvable:
        raise IngestionError(
            CaptureResolutionFailed(
                message=(
                    f"intraday_policy 'latest_capture' cannot resolve "
                    f"{len(unresolvable)} duplicated (as_of, opp_id) group(s): the "
                    f"capture timestamp in {capture_column!r} is missing or tied "
                    "within the group (a date-only source carries no time of day "
                    "to order captures by). Refusing to pick one arbitrarily."
                ),
                reason="capture_timestamp_unresolvable",
                unresolvable_groups=len(unresolvable),
                samples=[
                    DuplicateKey(as_of=g[0], opp_id=str(g[1]), row_count=int(g[2]))
                    for g in unresolvable[: settings.max_duplicate_samples]
                ],
            )
        )

    conn.execute(
        f"""
        DELETE FROM {CONFORMED_TABLE} AS c
        USING (SELECT as_of, opp_id, MAX({ts}) AS mx FROM {CONFORMED_TABLE}
               GROUP BY as_of, opp_id HAVING COUNT(*) > 1) AS g
        WHERE c.as_of = g.as_of AND c.opp_id = g.opp_id AND c.{ts} < g.mx
        """
    )
    conn.execute(f"ALTER TABLE {CONFORMED_TABLE} DROP COLUMN {ts}")
    return CaptureResolution(
        policy=IntradayPolicy.LATEST_CAPTURE,
        capture_column=capture_column,
        duplicate_groups=len(groups),
        rows_dropped=sum(int(g[2]) - 1 for g in groups),
        conflicting_groups=sum(1 for g in groups if g[5] > 1),
    )


def _capture_ts_sql(source: str, source_type: str | None) -> str:
    """The raw capture timestamp of `source`, before it is floored to a DATE."""
    if source_type is not None and source_type.upper() == "TIMESTAMP":
        return quote_ident(source)
    text = f"NULLIF(TRIM(CAST({quote_ident(source)} AS VARCHAR)), '')"
    return f"TRY_CAST({text} AS TIMESTAMP)"


def _describe_types(conn: duckdb.DuckDBPyConnection, read_expr: str) -> dict[str, str]:
    """Source column types as DuckDB reads them. Only asked of typed sources."""
    rows = conn.execute(f"DESCRIBE SELECT * FROM {read_expr}").fetchall()
    return {str(name): str(dtype) for name, dtype, *_ in rows}


def _mapping_for(
    source_columns: list[str],
    mapping_overrides: dict[str, CanonicalColumn] | None,
    column_registry: ColumnRegistry | None,
    settings: Settings,
) -> MappingProposal:
    if mapping_overrides:
        return mapping_from_overrides(source_columns, mapping_overrides)
    if column_registry is not None:
        return mapping_from_registry(column_registry, source_columns)
    return propose_mapping(source_columns, settings)


def _with_grain_columns(
    source: TableSource,
    mapping_overrides: dict[str, CanonicalColumn] | None,
    column_registry: ColumnRegistry | None,
    settings: Settings,
    store: DuckDBStore,
) -> TableSource:
    """A declared projection always keeps the grain columns.

    The grain's source headers are found by mapping the source's full header, so
    a projection that omits `as_of` or `opp_id` reads them anyway rather than
    failing the required-column check.
    """
    if not source.columns:
        return source
    full = read_source_columns(source.model_copy(update={"columns": None}), store)
    by_canonical = _mapping_for(full, mapping_overrides, column_registry, settings).by_canonical()
    columns = list(source.columns)
    for canonical in (CanonicalColumn.AS_OF, CanonicalColumn.OPP_ID):
        mapped = by_canonical.get(canonical)
        if mapped is not None and mapped.source_column not in columns:
            columns.append(mapped.source_column)
    return source.model_copy(update={"columns": tuple(columns)})


def _measure_cast_failures(
    conn: duckdb.DuckDBPyConnection,
    mapping: MappingProposal,
    read_expr: str,
    discovered_types: dict[str, DataType],
    date_encodings: dict[str, DateEncoding] | None = None,
    source_types: dict[str, str] | None = None,
) -> dict[str, int]:
    """Count source cells that held a value but did not survive their cast."""
    select = build_cast_failure_select(mapping, discovered_types, date_encodings, source_types)
    if select.endswith("_placeholder"):
        return {}
    row = conn.execute(f"{select} FROM {read_expr}").fetchone()
    columns = [d[0] for d in conn.description]
    return {name: int(count) for name, count in zip(columns, row, strict=True) if count}


def _bind_status(
    mapping: MappingProposal,
    source_columns: list[str],
    status_mapping: StatusMapping | None,
    settings: Settings,
) -> tuple[MappingProposal, StatusMapping | None]:
    """Apply the configured authoritative status source, or refuse to guess.

    The status column and its value mapping are configuration, never code. A
    configured column that is missing from the source, a column bound to status
    with no value mapping, and a mapping with nothing to map are all hard
    errors: each would otherwise fall back to stage inference in silence.
    """
    column = settings.status_column
    effective = status_mapping if status_mapping is not None else settings.status_mapping()

    if column and column not in source_columns:
        raise IngestionError(
            StatusConfigInvalid(
                message=(
                    f"the configured status column {column!r} is not in the source. "
                    "Refusing to fall back to stage inference silently."
                ),
                column=column,
                reason="column_not_in_source",
            )
        )

    if column:
        # Rebind: the configured column becomes status. Whatever else was bound
        # to it, or bound to status before, is released.
        released = [
            m.source_column
            for m in mapping.mappings
            if m.canonical_column is CanonicalColumn.STATUS and m.source_column != column
        ]
        claimed = [
            m
            for m in mapping.mappings
            if m.canonical_column is not CanonicalColumn.STATUS and m.source_column != column
        ]
        claimed.append(
            ColumnMapping(
                source_column=column,
                canonical_column=CanonicalColumn.STATUS,
                confidence=MappingConfidence.USER,
                score=1.0,
            )
        )
        bound = {m.canonical_column for m in claimed}
        mapping = MappingProposal(
            mappings=sorted(claimed, key=lambda m: m.canonical_column.value),
            unmapped_source_columns=sorted(
                {c for c in mapping.unmapped_source_columns if c != column} | set(released)
            ),
            missing_required=[c for c in REQUIRED_COLUMNS if c not in bound],
            # Rebuilding the proposal must not lose the near-misses it carried.
            fuzzy_candidates=mapping.fuzzy_candidates,
        )

    status_bound = CanonicalColumn.STATUS in mapping.by_canonical()
    if status_bound and effective is None:
        raise IngestionError(
            StatusConfigInvalid(
                message=(
                    "a source column is bound to status but no value mapping is "
                    "configured. Refusing to guess what its values mean."
                ),
                column=mapping.by_canonical()[CanonicalColumn.STATUS].source_column,
                reason="column_without_value_mapping",
            )
        )
    if effective is not None and not status_bound:
        raise IngestionError(
            StatusConfigInvalid(
                message="a status value mapping was supplied but no status column is bound.",
                reason="mapping_without_column",
            )
        )
    if effective is not None and not (
        effective.won_values or effective.lost_values or effective.open_values
    ):
        raise IngestionError(
            StatusConfigInvalid(
                message=(
                    "the status value mapping lists no won, lost, or open values, so "
                    "every row would resolve to unknown."
                ),
                column=column,
                reason="empty_value_mapping",
            )
        )
    return mapping, effective


def _check_stage_status_map(
    mapping: MappingProposal,
    status_mapping: StatusMapping | None,
    stage_status_map: dict[str, OpportunityStatus] | None,
) -> None:
    """A declared stage map needs a stage column and must be the only status source."""
    if stage_status_map is None:
        return

    def refuse(message: str, reason: str) -> IngestionError:
        return IngestionError(StatusConfigInvalid(message=message, reason=reason))

    if not stage_status_map:
        raise refuse("the declared stage-to-status map is empty.", "empty_stage_map")
    if OpportunityStatus.UNKNOWN in stage_status_map.values():
        raise refuse(
            "a declared stage-to-status map may not map a stage to 'unknown'.",
            "stage_map_unknown_status",
        )
    if status_mapping is not None or CanonicalColumn.STATUS in mapping.by_canonical():
        raise refuse(
            "both an authoritative status column and a stage-to-status map are "
            "declared. Refusing to pick one.",
            "two_status_sources",
        )
    if CanonicalColumn.STAGE not in mapping.by_canonical():
        raise refuse(
            "a stage-to-status map is declared but no stage column is bound.",
            "stage_map_without_stage",
        )


def _stage_map_coverage(
    conn: duckdb.DuckDBPyConnection,
    read_expr: str,
    column: str,
    stage_status_map: dict[str, OpportunityStatus],
    limit: int = 50,
) -> tuple[list[str], int]:
    """Stage values the declared map does not cover, and the rows they cover.

    Comparison is exact, matching the SQL that resolves status. NULL stages are
    unmapped rows but not a value.
    """
    ident = quote_ident(column)
    rows = conn.execute(
        f"SELECT CAST({ident} AS VARCHAR), COUNT(*) FROM {read_expr} GROUP BY 1"
    ).fetchall()
    unmapped = [(v, int(n)) for v, n in rows if v is None or v not in stage_status_map]
    values = sorted(v for v, _ in unmapped if v is not None)[:limit]
    return values, sum(n for _, n in unmapped)


def _unmapped_status_values(
    conn: duckdb.DuckDBPyConnection,
    read_expr: str,
    column: str,
    status_mapping: StatusMapping,
    limit: int = 50,
) -> list[str]:
    """Raw status values the mapping does not cover. They resolve to unknown."""
    ident = quote_ident(column)
    rows = conn.execute(
        f"SELECT DISTINCT TRIM(CAST({ident} AS VARCHAR)) FROM {read_expr} WHERE {ident} IS NOT NULL"
    ).fetchall()
    known = {v.strip().lower() for v in status_mapping.all_values}
    return sorted(v for (v,) in rows if v and v.strip().lower() not in known)[:limit]


def _reconstruction_hints(
    column_registry: ColumnRegistry | None,
) -> dict[CanonicalColumn, str]:
    """Explain a missing column that the registry says is reconstructible."""
    hints: dict[CanonicalColumn, str] = {}
    for record in column_registry.coverage if column_registry else []:
        if record.status is CoverageStatus.RECONSTRUCTIBLE:
            hints[CanonicalColumn(record.canonical)] = (
                f"The export registry documents that {record.canonical} is "
                f"reconstructed as `{record.derivation}`."
            )
    return hints


def _resolve_date_encodings(
    registry: ColumnRegistry | None,
    explicit: dict[str, DateEncoding] | None,
    mapping: MappingProposal,
    source_columns: list[str],
) -> dict[str, DateEncoding]:
    """Which source columns are declared to hold non-ISO dates (ARCHITECTURE 12.19).

    Declarations come from the export registry (only those whose column is in
    this source) and from the caller, who wins. Nothing is detected. A caller
    naming a column the source lacks, or a column bound to a canonical column
    that is not a date, is a misdeclaration and stops ingestion.
    """
    available = set(source_columns)
    declared = {
        c: e for c, e in (registry.date_encodings if registry else {}).items() if c in available
    }
    for column, encoding in (explicit or {}).items():
        if column not in available:
            raise IngestionError(
                DateEncodingInvalid(
                    message=(
                        f"a date encoding was declared for {column!r}, which is not in the source."
                    ),
                    column=column,
                    reason="column_not_in_source",
                )
            )
        declared[column] = encoding

    bound = {m.source_column: m.canonical_column for m in mapping.mappings}
    for column in declared:
        canonical = bound.get(column)
        if canonical is not None and CANONICAL_COLUMNS[canonical].dtype is not DataType.DATE:
            raise IngestionError(
                DateEncodingInvalid(
                    message=(
                        f"{column!r} is declared as dates but is bound to "
                        f"{canonical.value!r}, which is not a date column."
                    ),
                    column=column,
                    reason="bound_to_non_date_column",
                )
            )
    # ISO is the default path, so declaring it changes nothing (and lets a
    # caller cancel a registry declaration).
    return {c: e for c, e in declared.items() if e is not DateEncoding.ISO}


def _present_columns(
    mapping: MappingProposal, reconstructed: set[CanonicalColumn] | None = None
) -> list[ColumnSpec]:
    """Canonical columns that exist after conformance, including derived ones."""
    claimed = set(mapping.by_canonical()) | (reconstructed or set())
    present = []
    for canonical, spec in CANONICAL_COLUMNS.items():
        if canonical in claimed or spec.derivable:
            present.append(spec)
    return present


def ingest(
    source_path: str | Path | TableSource,
    dataset_id: str,
    *,
    mapping_overrides: dict[str, CanonicalColumn] | None = None,
    status_mapping: StatusMapping | None = None,
    column_registry: ColumnRegistry | None = None,
    date_encodings: dict[str, DateEncoding] | None = None,
    snapshot_policy: SnapshotPolicy | None = None,
    stage_status_map: dict[str, OpportunityStatus] | None = None,
    family_invariants: list[FamilyInvariant] | None = None,
    settings: Settings | None = None,
    store: DuckDBStore | None = None,
    copy_raw: bool = True,
) -> IngestionResult:
    """Ingest one source (a local file, or a `TableSource` lake location) into a
    dataset's canonical Parquet layout.

    A plain path or string is one local CSV or Parquet file, as before. A
    `TableSource` names a partitioned Parquet directory, a Delta table, or an
    `s3://` location; `copy_raw` is silently skipped for these, since there is
    no single local file to preserve a copy of.

    Mapping precedence is explicit overrides, then the export registry's own
    coverage records, then header inference. When a column registry is given,
    the dataset's registry is classified from it; otherwise every non-canonical
    column fails closed.
    """
    settings = settings or get_settings()
    store = store or DuckDBStore(settings)
    policy = snapshot_policy or SnapshotPolicy()
    if policy.snapshot_granularity is not SnapshotGranularity.DAY:
        raise IngestionError(
            CaptureResolutionFailed(
                message=(
                    "snapshot_granularity 'capture' is declared but not supported: "
                    "the conformed as_of is a DATE, so every capture cannot be its "
                    "own snapshot. Declare 'day' (with an intraday_policy) instead."
                ),
                reason="granularity_capture_unsupported",
            )
        )
    table_source = _coerce_source(source_path)
    local_path = _local_single_file(table_source)

    table_source = _with_grain_columns(
        table_source, mapping_overrides, column_registry, settings, store
    )
    source_columns = read_source_columns(table_source, store)
    mapping = _mapping_for(source_columns, mapping_overrides, column_registry, settings)

    mapping, status_mapping = _bind_status(mapping, source_columns, status_mapping, settings)
    _check_stage_status_map(mapping, status_mapping, stage_status_map)
    assert_mappable(mapping, source_columns, hints=_reconstruction_hints(column_registry))

    # A canonical column the export lacks may be rebuilt from other source
    # columns when the registry documents the derivation (ARCHITECTURE 12.15).
    encodings = _resolve_date_encodings(column_registry, date_encodings, mapping, source_columns)
    reconstructions = plan_reconstructions(column_registry, mapping, source_columns, encodings)

    read_expr = scan_sql(table_source)
    discovered = discovered_columns(mapping)
    canonical_dir = settings.canonical_dir(dataset_id)

    # Only columns that survive conformance are converted. A declared column that
    # collides with a canonical name is dropped like any other discovered column.
    bound_sources = {m.source_column: m.canonical_column.value for m in mapping.mappings}
    encoded_targets = {
        source: (bound_sources.get(source, source), encoding)
        for source, encoding in encodings.items()
        if source in bound_sources or source in discovered
    }
    encodings = {source: encoding for source, (_, encoding) in encoded_targets.items()}

    with store.connect() as conn:
        prepare_source(conn, table_source)
        # Fatal before anything is written: an invalid declared date is an error.
        date_conversions = measure_date_conversions(conn, read_expr, encoded_targets)

        # A typed source (Parquet, Delta) keeps its types: only what differs from
        # the canonical type is cast, and only text-shaped columns are detected.
        # A CSV is all text and takes the original path unchanged.
        source_types: dict[str, str] | None = None
        if table_source.format is not SourceFormat.CSV:
            source_types = _describe_types(conn, read_expr)

        discovered_types: dict[str, DataType] = {
            c: DataType.DATE for c in discovered if c in encodings
        }
        for c in discovered:
            if c not in encodings and source_types is not None:
                kept = typed_discovered_type(source_types.get(c, "VARCHAR"))
                if kept is not None:
                    discovered_types[c] = kept
        to_detect = [c for c in discovered if c not in discovered_types]
        if to_detect:
            counts = conn.execute(build_type_detection_select(to_detect, read_expr)).fetchone()
            discovered_types.update(detect_types(counts, to_detect))

        select_clause, derived, status_resolution = build_select(
            mapping,
            status_mapping,
            discovered_types,
            reconstructions,
            encodings,
            source_types,
            stage_status_map,
        )

        if status_resolution.strategy is StatusStrategy.DECLARED_STAGE_MAP:
            values, n_rows = _stage_map_coverage(
                conn, read_expr, status_resolution.column or "", stage_status_map or {}
            )
            status_resolution = status_resolution.model_copy(
                update={"unmapped_values": values, "unmapped_rows": n_rows}
            )
            derived = [
                d.model_copy(
                    update={"requires_confirmation": status_resolution.requires_confirmation}
                )
                if d.column
                in (CanonicalColumn.STATUS, CanonicalColumn.IS_CLOSED, CanonicalColumn.IS_WON)
                else d
                for d in derived
            ]

        if status_resolution.is_authoritative and status_mapping is not None:
            unmapped = _unmapped_status_values(
                conn, read_expr, status_resolution.column or "", status_mapping
            )
            status_resolution = status_resolution.model_copy(update={"unmapped_values": unmapped})
            derived = [
                d.model_copy(
                    update={"requires_confirmation": status_resolution.requires_confirmation}
                )
                if d.column
                in (CanonicalColumn.STATUS, CanonicalColumn.IS_CLOSED, CanonicalColumn.IS_WON)
                else d
                for d in derived
            ]

        capture_column: str | None = None
        if policy.intraday_policy is IntradayPolicy.LATEST_CAPTURE:
            capture_column = mapping.by_canonical()[CanonicalColumn.AS_OF].source_column
            if capture_column in encodings:
                raise IngestionError(
                    CaptureResolutionFailed(
                        message=(
                            f"intraday_policy 'latest_capture' cannot order captures by "
                            f"{capture_column!r}: it is a declared serial-date column."
                        ),
                        reason="capture_column_encoded",
                    )
                )
            # The raw timestamp, read before as_of is floored to a DATE.
            capture_sql = _capture_ts_sql(capture_column, (source_types or {}).get(capture_column))
            select_clause += f",\n       {capture_sql} AS {quote_ident(CAPTURE_TS)}"

        conn.execute(f"CREATE TABLE {CONFORMED_TABLE} AS SELECT {select_clause} FROM {read_expr}")
        cast_failures = _measure_cast_failures(
            conn, mapping, read_expr, discovered_types, encodings, source_types
        )
        capture_resolution: CaptureResolution | None = None
        if capture_column is not None:
            _assert_no_null_keys(conn)
            capture_resolution = _resolve_captures(conn, settings, capture_column)
        _assert_grain(conn, settings)
        # Corroborate anything that was rebuilt rather than read. Evaluated once,
        # here, against the conformed table, and persisted below.
        agreement = evaluate_reconstruction_agreement(
            conn,
            CONFORMED_TABLE,
            dataset_id=dataset_id,
            reconstructed=set(reconstructions),
            fiscal_year_start_month=settings.fiscal_year_start_month,
        )

        if family_invariants:
            agreement = with_invariants(
                agreement,
                evaluate_family_invariants(
                    conn, CONFORMED_TABLE, family_invariants, dataset_id=dataset_id
                ),
            )

        row_count = conn.execute(f"SELECT COUNT(*) FROM {CONFORMED_TABLE}").fetchone()[0]
        snapshot_count = conn.execute(
            f"SELECT COUNT(DISTINCT as_of) FROM {CONFORMED_TABLE}"
        ).fetchone()[0]

        if canonical_dir.exists():
            shutil.rmtree(canonical_dir)
        canonical_dir.mkdir(parents=True, exist_ok=True)
        target = quote_literal(str(canonical_dir.resolve()))
        conn.execute(
            f"COPY (SELECT * FROM {CONFORMED_TABLE}) TO {target} "
            "(FORMAT PARQUET, PARTITION_BY (as_of), "
            "OVERWRITE_OR_IGNORE 1, FILENAME_PATTERN 'part-{i}')"
        )

    schema = DatasetSchema(
        dataset_id=dataset_id,
        source_path=str(local_path.resolve()) if local_path is not None else table_source.uri,
        mapping=mapping,
        columns=_present_columns(mapping, set(reconstructions)),
        derived_columns=derived,
        date_conversions=date_conversions,
        capture_resolution=capture_resolution,
        cast_failures=cast_failures,
        status_resolution=status_resolution,
        discovered_columns=discovered,
        discovered_types=discovered_types,
    )
    registry = build_dataset_registry(schema, column_registry)

    if copy_raw and local_path is not None:
        raw_dir = settings.raw_dir(dataset_id)
        raw_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, raw_dir / local_path.name)

    settings.schema_path(dataset_id).parent.mkdir(parents=True, exist_ok=True)
    settings.schema_path(dataset_id).write_text(schema.model_dump_json(indent=2), encoding="utf-8")
    settings.classifications_path(dataset_id).write_text(
        registry.model_dump_json(indent=2), encoding="utf-8"
    )
    # A profile describes the data it was computed from. Re-ingesting replaces
    # that data, so any earlier profile is stale and must not outlive it.
    settings.profile_path(dataset_id).unlink(missing_ok=True)
    # The same holds for agreement results: they describe the rows just written.
    has_agreement = bool(reconstructions or family_invariants)
    if has_agreement:
        settings.agreement_path(dataset_id).write_text(
            agreement.model_dump_json(indent=2), encoding="utf-8"
        )
    else:
        settings.agreement_path(dataset_id).unlink(missing_ok=True)

    return IngestionResult(
        dataset_id=dataset_id,
        schema=schema,
        registry=registry,
        row_count=int(row_count),
        snapshot_count=int(snapshot_count),
        canonical_path=canonical_dir,
        agreement=agreement if has_agreement else None,
    )


def load_schema(dataset_id: str, settings: Settings | None = None) -> DatasetSchema:
    settings = settings or get_settings()
    path = settings.schema_path(dataset_id)
    if not path.exists():
        raise IngestionError(
            UnreadableSource(
                message=f"no schema persisted for dataset {dataset_id!r}",
                path=str(path),
                reason="not_found",
            )
        )
    return DatasetSchema.model_validate_json(path.read_text(encoding="utf-8"))


__all__ = [
    "CONFORMED_TABLE",
    "IngestionResult",
    "ingest",
    "load_schema",
    "quote_ident",
    "read_source_columns",
]
