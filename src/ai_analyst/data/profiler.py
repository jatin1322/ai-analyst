"""Dataset profiling (ARCHITECTURE §7.3).

Computed once, persisted, and later injected into the LLM context as a compact
card. All statistics are computed in DuckDB SQL, per the project rule that
analytical operations prefer SQL.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb

from ai_analyst.config import Settings, get_settings
from ai_analyst.contracts.columns import MonetaryStatus
from ai_analyst.contracts.dataset import DatasetRegistry
from ai_analyst.contracts.errors import ProfilingError, ProfilingFailed
from ai_analyst.contracts.profile import (
    ColumnProfile,
    DataQualityFlags,
    DatasetProfile,
    GrainCheck,
    LifecycleStats,
    SnapshotInfo,
    StageClassification,
    StageVocabulary,
    StatusProfile,
    TextColumnProfile,
)
from ai_analyst.contracts.schema import CanonicalColumn, DatasetSchema, DataType
from ai_analyst.data.column_profiler import profile_column
from ai_analyst.data.conform import classify_stage
from ai_analyst.data.store import DuckDBStore

_SCAN = "snap"


def _stage_vocabulary(
    conn: duckdb.DuckDBPyConnection, schema: DatasetSchema
) -> StageVocabulary:
    """Stage labels with a keyword-inferred closed/won reading.

    Always inferred and always flagged, even when an authoritative status
    column exists: this mapping reads stage labels and stage is not
    authoritative for status (ARCHITECTURE 5.13). Reporting it as confirmed
    would let a keyword guess pass for a finding.

    A tenant may export no stage at all (ARCHITECTURE 12.15), in which case the
    vocabulary is empty rather than an error.
    """
    if not schema.has(CanonicalColumn.STAGE):
        return StageVocabulary(stages=[], requires_confirmation=True)
    rows = conn.execute(
        f"""
        SELECT stage, COUNT(*) AS n
        FROM {_SCAN}
        WHERE stage IS NOT NULL
        GROUP BY stage
        ORDER BY n DESC, stage
        """
    ).fetchall()
    stages = []
    for label, count in rows:
        is_closed, is_won = classify_stage(label)
        stages.append(
            StageClassification(
                stage=str(label),
                row_count=int(count),
                is_closed=is_closed,
                is_won=is_won,
                inferred=True,
            )
        )
    return StageVocabulary(stages=stages, requires_confirmation=True)


def _lifecycle(
    conn: duckdb.DuckDBPyConnection,
    snapshot_count: int,
    sample_limit: int,
    max_as_of: date,
) -> LifecycleStats:
    """Per-opportunity lifecycle statistics.

    `vanished_without_terminal_state` counts opportunities whose last
    appearance is before the dataset's final snapshot and which were not in a
    terminal stage at that last appearance. This is the population behind the
    `other_removed` bridge term (ARCHITECTURE §5.2).
    """
    distinct_opps, min_s, max_s, median_s, present_all = conn.execute(
        f"""
        WITH per_opp AS (
            SELECT opp_id, COUNT(DISTINCT as_of) AS n FROM {_SCAN} GROUP BY opp_id
        )
        SELECT COUNT(*), MIN(n), MAX(n), MEDIAN(n),
               COUNT(*) FILTER (WHERE n = {int(snapshot_count)})
        FROM per_opp
        """
    ).fetchone()

    vanished_rows = conn.execute(
        f"""
        WITH last_seen AS (
            SELECT opp_id, MAX(as_of) AS last_as_of FROM {_SCAN} GROUP BY opp_id
        ),
        final_state AS (
            SELECT l.opp_id, l.last_as_of, s.is_closed
            FROM last_seen l
            JOIN {_SCAN} s ON s.opp_id = l.opp_id AND s.as_of = l.last_as_of
        )
        SELECT opp_id
        FROM final_state
        WHERE last_as_of < DATE '{max_as_of.isoformat()}'
          AND NOT is_closed
        ORDER BY opp_id
        """
    ).fetchall()
    vanished = [str(r[0]) for r in vanished_rows]

    return LifecycleStats(
        distinct_opportunities=int(distinct_opps or 0),
        snapshot_count=snapshot_count,
        min_snapshots_per_opportunity=int(min_s or 0),
        max_snapshots_per_opportunity=int(max_s or 0),
        median_snapshots_per_opportunity=float(median_s or 0.0),
        present_in_all_snapshots=int(present_all or 0),
        vanished_without_terminal_state=len(vanished),
        vanished_sample_opp_ids=vanished[:sample_limit],
    )


def _quality(conn: duckdb.DuckDBPyConnection, schema: DatasetSchema) -> DataQualityFlags:
    # Every check here is conditional on its columns existing. Since the
    # ingestion minimum became the grain alone (ARCHITECTURE 12.15), a dataset
    # may legitimately carry no amount and no close date, and a quality check
    # that assumed otherwise would crash on exactly the tenant it was meant to
    # describe.
    negative = 0
    if schema.has(CanonicalColumn.AMOUNT):
        negative = conn.execute(
            f"SELECT COUNT(*) FROM {_SCAN} WHERE amount < 0"
        ).fetchone()[0]

    close_before_created = 0
    if schema.has(CanonicalColumn.CREATED_DATE) and schema.has(CanonicalColumn.CLOSE_DATE):
        close_before_created = conn.execute(
            f"""
            SELECT COUNT(*) FROM {_SCAN}
            WHERE created_date IS NOT NULL
              AND close_date IS NOT NULL
              AND close_date < created_date
            """
        ).fetchone()[0]

    # Genuine cast failures are measured during ingestion, against the raw
    # text. A null here is usually a legitimately blank cell, and
    # ColumnProfile.null_count already reports those.
    return DataQualityFlags(
        negative_amount_rows=int(negative),
        close_date_before_created_date_rows=int(close_before_created),
        cast_failure_rows=dict(schema.cast_failures),
    )


def _status_profile(
    conn: duckdb.DuckDBPyConnection, schema: DatasetSchema
) -> StatusProfile | None:
    """Make the status strategy, and any missing authoritative source, visible."""
    resolution = schema.status_resolution
    if resolution is None:
        return None
    distribution: dict[str, int] = {}
    if schema.has(CanonicalColumn.STATUS):
        distribution = {
            str(status): int(n)
            for status, n in conn.execute(
                f"SELECT status, COUNT(*) FROM {_SCAN} GROUP BY status ORDER BY status"
            ).fetchall()
        }
    return StatusProfile(
        strategy=resolution.strategy,
        authoritative=resolution.is_authoritative,
        configured_column=resolution.column if resolution.is_authoritative else None,
        distribution=distribution,
        unmapped_values=list(resolution.unmapped_values),
        requires_confirmation=resolution.requires_confirmation,
        note=resolution.note,
    )


def profile_dataset(
    dataset_id: str,
    schema: DatasetSchema,
    *,
    registry: DatasetRegistry | None = None,
    settings: Settings | None = None,
    store: DuckDBStore | None = None,
    persist: bool = True,
    declared_monetary: frozenset[str] = frozenset(),
) -> DatasetProfile:
    """Profile every column of an ingested dataset and optionally persist profile.json.

    The registry is only read, to learn declared sentinels and declared text
    columns. The profile never writes into it and never reclassifies a column.

    `declared_monetary` names columns a tenant has declared to be money. Profiling
    runs before any concept binding exists, so a declaration has to arrive here to
    matter; it withholds the float summary statistics for those columns.
    """
    settings = settings or get_settings()
    store = store or DuckDBStore(settings)

    if not store.canonical_exists(dataset_id):
        raise ProfilingError(
            ProfilingFailed(
                message=f"dataset {dataset_id!r} has no canonical parquet to profile",
                reason="missing_canonical_data",
            )
        )

    with store.connect() as conn:
        conn.execute(
            f"CREATE VIEW {_SCAN} AS SELECT * FROM {store.snapshots_scan(dataset_id)}"
        )

        row_count = int(conn.execute(f"SELECT COUNT(*) FROM {_SCAN}").fetchone()[0])

        snapshot_rows = conn.execute(
            f"SELECT as_of, COUNT(*) FROM {_SCAN} GROUP BY as_of ORDER BY as_of"
        ).fetchall()
        snapshots = [SnapshotInfo(as_of=r[0], row_count=int(r[1])) for r in snapshot_rows]

        classified = registry.by_name() if registry else {}
        targets: list[tuple[str, DataType]] = [
            (spec.name.value, spec.dtype) for spec in schema.columns
        ] + [
            (name, schema.discovered_types.get(name, DataType.VARCHAR))
            for name in schema.discovered_columns
        ]
        columns: list[ColumnProfile] = []
        text_columns: list[TextColumnProfile] = []
        for name, dtype in targets:
            entry = classified.get(name)
            profile_row, catalogue = profile_column(
                conn,
                _SCAN,
                name,
                dtype,
                row_count,
                entry.classification if entry else None,
                settings,
                MonetaryStatus.MONETARY if name in declared_monetary else None,
            )
            columns.append(profile_row)
            if catalogue is not None:
                text_columns.append(catalogue)

        vocabulary = _stage_vocabulary(conn, schema)
        lifecycle = _lifecycle(
            conn,
            len(snapshots),
            settings.max_vanished_samples,
            max(s.as_of for s in snapshots),
        )

        distinct_keys = int(
            conn.execute(
                f"SELECT COUNT(*) FROM (SELECT DISTINCT as_of, opp_id FROM {_SCAN})"
            ).fetchone()[0]
        )
        grain = GrainCheck(
            passed=distinct_keys == row_count,
            total_rows=row_count,
            distinct_keys=distinct_keys,
            duplicate_key_count=max(0, row_count - distinct_keys),
        )

        quality = _quality(conn, schema)
        status = _status_profile(conn, schema)

    profile = DatasetProfile(
        dataset_id=dataset_id,
        row_count=row_count,
        snapshots=snapshots,
        columns=columns,
        stage_vocabulary=vocabulary,
        lifecycle=lifecycle,
        grain=grain,
        quality=quality,
        text_columns=text_columns,
        status=status,
    )

    if persist:
        path: Path = settings.profile_path(dataset_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")

    return profile


def load_profile(dataset_id: str, settings: Settings | None = None) -> DatasetProfile:
    settings = settings or get_settings()
    path = settings.profile_path(dataset_id)
    if not path.exists():
        raise ProfilingError(
            ProfilingFailed(
                message=f"no profile persisted for dataset {dataset_id!r}",
                reason="not_found",
            )
        )
    return DatasetProfile.model_validate_json(path.read_text(encoding="utf-8"))
