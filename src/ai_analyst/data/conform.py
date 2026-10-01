"""Type coercion and derived-flag inference.

Source data is read as text and cast explicitly here rather than relying on
DuckDB's sniffer, so that a cast failure is a countable data quality event
instead of a silent type surprise.
"""

from __future__ import annotations

from ai_analyst.contracts.schema import (
    CANONICAL_COLUMNS,
    EXCEL_EPOCH,
    EXCEL_MAX_SERIAL,
    EXCEL_MIN_SERIAL,
    CanonicalColumn,
    DataType,
    DateEncoding,
    DerivationRule,
    DerivedColumn,
    MappingProposal,
)
from ai_analyst.contracts.status import (
    INVALID_STAGE_VALUES,
    LOST_STAGE_KEYWORDS,
    WON_STAGE_KEYWORDS,
    OpportunityStatus,
    StatusMapping,
    StatusResolution,
    StatusStrategy,
    status_from_stage,
)

WON_KEYWORDS: tuple[str, ...] = WON_STAGE_KEYWORDS
LOST_KEYWORDS: tuple[str, ...] = LOST_STAGE_KEYWORDS


def quote_ident(name: str) -> str:
    """Quote a SQL identifier, escaping embedded double quotes."""
    return '"' + name.replace('"', '""') + '"'


def quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def classify_stage(stage: str | None) -> tuple[bool, bool]:
    """Infer (is_closed, is_won) from a stage label.

    Last resort only. `Stage` is not authoritative for open, won, and lost
    (ARCHITECTURE 5.13): a vocabulary can encode an outcome in a label this
    classifier cannot read, and invalid labels such as a deleted-record marker
    would otherwise be counted as open pipeline. Prefer an authoritative status
    column whenever one exists.
    """
    status = status_from_stage(stage)
    return (status.is_closed, status.is_won)


# A serial number as text: an optional sign, digits, an optional fraction, an
# optional exponent (a DOUBLE column can print in scientific notation).
SERIAL_PATTERN = r"^[+-]?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][+-]?[0-9]+)?$"


def _cleaned(source: str) -> str:
    """A source column as trimmed text, with blanks read as NULL."""
    return f"NULLIF(TRIM(CAST({quote_ident(source)} AS VARCHAR)), '')"


def date_sql(source: str, encoding: DateEncoding = DateEncoding.ISO) -> str:
    """The one place a source column becomes a DATE (ARCHITECTURE 12.19).

    Every path that produces a date calls this: the canonical cast, discovered
    date columns, and the close-date reconstruction. A conversion patched into
    only one of them would leave the others silently null on exactly the exports
    that need it.

    For `EXCEL_SERIAL` a value that is a number is converted as
    `1899-12-30 + FLOOR(serial)`; anything else is read as an ISO date. FLOOR,
    not CAST: DuckDB rounds on cast, so `45973.6` would land a day late. The time
    of day is discarded because the conformed type is DATE. A number outside
    the valid serial range yields NULL, which the caller counts as invalid; it
    is never guessed at.

    Nothing here is heuristic. Whether a column is a serial date is decided by
    declaration, so no numeric column is ever reinterpreted as dates by shape.
    """
    cleaned = _cleaned(source)
    if encoding is DateEncoding.ISO:
        return f"TRY_CAST({cleaned} AS DATE)"
    numeric = f"TRY_CAST({cleaned} AS DOUBLE)"
    in_range = f"FLOOR({numeric}) BETWEEN {EXCEL_MIN_SERIAL} AND {EXCEL_MAX_SERIAL}"
    return (
        f"CASE WHEN {cleaned} IS NULL THEN NULL "
        f"WHEN regexp_matches({cleaned}, {quote_literal(SERIAL_PATTERN)}) "
        f"THEN CASE WHEN {in_range} "
        f"THEN DATE {quote_literal(EXCEL_EPOCH)} + CAST(FLOOR({numeric}) AS INTEGER) END "
        f"ELSE TRY_CAST({cleaned} AS DATE) END"
    )


_INT_TYPES = frozenset({"TINYINT", "SMALLINT", "INTEGER", "BIGINT"})


def typed_cast_mode(source_type: str, dtype: DataType) -> str | None:
    """How a column of a typed source (Parquet, Delta) reaches `dtype`.

    * `"exact"`: the source type already is the canonical type; keep it as is.
    * `"direct"`: a plain `TRY_CAST` is lossless in intent (widening integers,
      timestamp truncated to its date, decimal to decimal).
    * `None`: go through the text path, exactly as a CSV would. This is the
      safe default, and it is deliberate for FLOAT/DOUBLE to DECIMAL: casting a
      binary double to DECIMAL(18,2) can land a cent away from the printed
      value (0.285 becomes 0.28), which the text path avoids (CLAUDE.md, money).

    A canonical VARCHAR column always takes the text path so blanks and padding
    are normalised the same way for every format.
    """
    t = source_type.upper()
    if dtype is DataType.VARCHAR:
        return None
    if t == dtype.duckdb_type:
        return "exact"
    if dtype is DataType.DATE and t.startswith("TIMESTAMP") and "TIME ZONE" not in t:
        return "direct"
    if dtype is DataType.BIGINT and t in _INT_TYPES:
        return "direct"
    if dtype is DataType.DOUBLE and (t in _INT_TYPES or t.startswith("DECIMAL(")):
        return "direct"
    if dtype is DataType.DECIMAL and (t in _INT_TYPES or t.startswith("DECIMAL(")):
        return "direct"
    return None


def typed_discovered_type(source_type: str) -> DataType | None:
    """The storage type a discovered column of a typed source keeps, or None to
    fall back to text-shape detection (VARCHAR, floats, anything exotic)."""
    t = source_type.upper()
    if t in _INT_TYPES:
        return DataType.BIGINT
    if t == "DOUBLE" or t.startswith("DECIMAL("):
        return DataType.DOUBLE
    if t == "BOOLEAN":
        return DataType.BOOLEAN
    if t == "DATE" or (t.startswith("TIMESTAMP") and "TIME ZONE" not in t):
        return DataType.DATE
    return None


def _cast_expression(
    source: str,
    dtype: DataType,
    encoding: DateEncoding | None = None,
    source_type: str | None = None,
) -> str:
    """Cast a source column to its canonical type, treating blanks as NULL.

    `source_type` is set only for a typed source; a CSV is all text.
    """
    if dtype is DataType.DATE and encoding is not None:
        return date_sql(source, encoding)
    if source_type is not None:
        mode = typed_cast_mode(source_type, dtype)
        if mode == "exact":
            return quote_ident(source)
        if mode == "direct":
            return f"TRY_CAST({quote_ident(source)} AS {dtype.duckdb_type})"
    cleaned = _cleaned(source)
    if dtype is DataType.VARCHAR:
        return cleaned
    return f"TRY_CAST({cleaned} AS {dtype.duckdb_type})"


def _status_from_column_sql(source: str, mapping: StatusMapping) -> str:
    """CASE expression mapping an authoritative status column onto our values."""
    value = f"LOWER(TRIM(CAST({quote_ident(source)} AS VARCHAR)))"

    def bucket(values: tuple[str, ...]) -> str:
        rendered = ", ".join(quote_literal(v.strip().lower()) for v in values)
        return f"{value} IN ({rendered})"

    branches = []
    if mapping.excluded_values:
        branches.append(f"WHEN {bucket(mapping.excluded_values)} THEN 'excluded'")
    if mapping.won_values:
        branches.append(f"WHEN {bucket(mapping.won_values)} THEN 'won'")
    if mapping.lost_values:
        branches.append(f"WHEN {bucket(mapping.lost_values)} THEN 'lost'")
    if mapping.open_values:
        branches.append(f"WHEN {bucket(mapping.open_values)} THEN 'open'")
    joined = "\n            ".join(branches)
    return (
        f"CASE\n            WHEN {value} IS NULL THEN 'unknown'\n"
        f"            {joined}\n            ELSE 'unknown' END"
    )


def _status_from_flags_sql(closed_source: str, won_source: str) -> str:
    closed = f"TRY_CAST(NULLIF(TRIM(CAST({quote_ident(closed_source)} AS VARCHAR)), '') AS BOOLEAN)"
    won = f"TRY_CAST(NULLIF(TRIM(CAST({quote_ident(won_source)} AS VARCHAR)), '') AS BOOLEAN)"
    return (
        f"CASE WHEN {won} THEN 'won' "
        f"WHEN {closed} THEN 'lost' "
        f"WHEN {closed} IS NULL THEN 'unknown' "
        f"ELSE 'open' END"
    )


def _status_from_stage_sql(stage_source: str) -> str:
    """Last-resort inference. Invalid stage labels resolve to 'excluded'."""
    stage = f"LOWER(TRIM(CAST({quote_ident(stage_source)} AS VARCHAR)))"
    invalid = ", ".join(quote_literal(v) for v in INVALID_STAGE_VALUES if v)
    lost = " OR ".join(f"{stage} LIKE {quote_literal(f'%{k}%')}" for k in LOST_KEYWORDS)
    won = " OR ".join(f"{stage} LIKE {quote_literal(f'%{k}%')}" for k in WON_KEYWORDS)
    return (
        f"CASE WHEN {stage} IS NULL THEN 'excluded' "
        f"WHEN {stage} IN ({invalid}) THEN 'excluded' "
        f"WHEN {lost} THEN 'lost' "
        f"WHEN {won} THEN 'won' "
        f"ELSE 'open' END"
    )


def _status_from_declared_stage_map_sql(
    stage_source: str, stage_map: dict[str, OpportunityStatus]
) -> str:
    """Exact-value lookup of a declared stage-to-status map (WP5).

    No trimming, no case folding, no keywords: a stage value is either declared
    or it is not. Anything undeclared (including NULL) is 'excluded', never open.
    """
    value = f"CAST({quote_ident(stage_source)} AS VARCHAR)"
    branches = "\n            ".join(
        f"WHEN {quote_literal(stage)} THEN {quote_literal(status.value)}"
        for stage, status in sorted(stage_map.items())
    )
    return f"CASE {value}\n            {branches}\n            ELSE 'excluded' END"


def resolve_status(
    mapping: MappingProposal,
    status_mapping: StatusMapping | None,
    stage_status_map: dict[str, OpportunityStatus] | None = None,
) -> tuple[str, StatusResolution]:
    """Pick a status strategy and build its SQL (ARCHITECTURE 5.13).

    The chain is: an authoritative status column, then directly mapped closed
    and won flags, then stage keywords. Only the first is authoritative, and the
    resolution records which was used so an answer can disclose it.
    """
    by_canonical = mapping.by_canonical()

    if CanonicalColumn.STATUS in by_canonical and status_mapping is not None:
        source = by_canonical[CanonicalColumn.STATUS].source_column
        return _status_from_column_sql(source, status_mapping), StatusResolution(
            strategy=StatusStrategy.AUTHORITATIVE_COLUMN,
            column=source,
            mapping=status_mapping,
            note="Resolved from the authoritative status column.",
        )

    if stage_status_map is not None and CanonicalColumn.STAGE in by_canonical:
        source = by_canonical[CanonicalColumn.STAGE].source_column
        return _status_from_declared_stage_map_sql(source, stage_status_map), StatusResolution(
            strategy=StatusStrategy.DECLARED_STAGE_MAP,
            column=source,
            stage_status_map=dict(stage_status_map),
            note=(
                "Resolved from a tenant-declared stage-to-status map (exact value "
                "lookup). Stage values not in the map are EXCLUDED, never open."
            ),
        )

    has_closed = CanonicalColumn.IS_CLOSED in by_canonical
    has_won = CanonicalColumn.IS_WON in by_canonical
    if has_closed and has_won:
        closed_source = by_canonical[CanonicalColumn.IS_CLOSED].source_column
        won_source = by_canonical[CanonicalColumn.IS_WON].source_column
        return _status_from_flags_sql(closed_source, won_source), StatusResolution(
            strategy=StatusStrategy.MAPPED_FLAGS,
            column=closed_source,
            note="Resolved from observed is_closed and is_won columns.",
        )

    if CanonicalColumn.STAGE in by_canonical:
        source = by_canonical[CanonicalColumn.STAGE].source_column
        return _status_from_stage_sql(source), StatusResolution(
            strategy=StatusStrategy.STAGE_KEYWORD,
            note=(
                "NOT AUTHORITATIVE. Inferred from stage label keywords because no "
                "status column was configured. A stage vocabulary can encode an "
                "outcome in a label this inference cannot read, so every rate "
                "derived from it requires confirmation."
            ),
        )

    return "'unknown'", StatusResolution(
        strategy=StatusStrategy.UNRESOLVED,
        note="No status column, no observed flags, and no stage column.",
    )


# ---------------------------------------------------------------------------
# Type detection for discovered columns
#
# Discovered columns arrive as text and must be typed before they can be
# profiled or aggregated. Detection is deliberately conservative, because a
# wrong guess silently changes values:
#
# * integrality and numeracy are tested with regular expressions, not casts,
#   because DuckDB rounds rather than fails: TRY_CAST('1.5' AS BIGINT) is 2;
# * leading zeros disqualify a value from being numeric, so '0012' stays text
#   instead of becoming 12;
# * booleans are only the literal tokens true and false, because '1' casts to
#   TRUE and would otherwise turn every 0/1 numeric column into a boolean.
# ---------------------------------------------------------------------------

INTEGER_PATTERN = r"^[+-]?(0|[1-9][0-9]{0,17})$"
NUMERIC_PATTERN = r"^[+-]?((0|[1-9][0-9]*)(\.[0-9]+)?|\.[0-9]+)([eE][+-]?[0-9]+)?$"
BOOLEAN_TOKENS: tuple[str, ...] = ("true", "false")


def decide_type(
    non_null: int, integer: int, numeric: int, boolean: int, date_like: int
) -> DataType:
    """Choose a storage type from how many non-null values fit each shape.

    A column takes a specific type only if every non-null value fits it.
    Numeric is tried before boolean and date, in that order.
    """
    if non_null == 0:
        return DataType.VARCHAR
    if integer == non_null:
        return DataType.BIGINT
    if numeric == non_null:
        return DataType.DOUBLE
    if boolean == non_null:
        return DataType.BOOLEAN
    if date_like == non_null:
        return DataType.DATE
    return DataType.VARCHAR


def build_type_detection_select(columns: list[str], source_expr: str) -> str:
    """One scan that counts, per column, how many values fit each shape.

    Detection deliberately cannot recognise Excel serial dates: `date_like` uses
    an ISO cast, so a serial stays BIGINT or DOUBLE. That is the requirement, not
    a gap. A numeric column is never reinterpreted as dates by its values;
    serial dates are converted only where a declaration says so. Do not extend
    this to guess them.
    """
    cleaned = ", ".join(
        f"NULLIF(TRIM(CAST({quote_ident(c)} AS VARCHAR)), '') AS c{i}"
        for i, c in enumerate(columns)
    )
    tokens = ", ".join(quote_literal(t) for t in BOOLEAN_TOKENS)
    aggregates = []
    for i in range(len(columns)):
        col = f"c{i}"
        aggregates.extend(
            [
                f"COUNT({col})",
                f"COUNT(*) FILTER (WHERE regexp_matches({col}, {quote_literal(INTEGER_PATTERN)}))",
                f"COUNT(*) FILTER (WHERE regexp_matches({col}, {quote_literal(NUMERIC_PATTERN)}))",
                f"COUNT(*) FILTER (WHERE LOWER({col}) IN ({tokens}))",
                f"COUNT(*) FILTER (WHERE TRY_CAST({col} AS DATE) IS NOT NULL)",
            ]
        )
    return (
        f"WITH cleaned AS (SELECT {cleaned} FROM {source_expr})\n"
        f"SELECT {', '.join(aggregates)} FROM cleaned"
    )


def detect_types(counts: tuple[int, ...], columns: list[str]) -> dict[str, DataType]:
    """Turn the flat aggregate row from `build_type_detection_select` into types."""
    detected: dict[str, DataType] = {}
    for i, column in enumerate(columns):
        non_null, integer, numeric, boolean, date_like = counts[i * 5 : i * 5 + 5]
        detected[column] = decide_type(
            int(non_null), int(integer), int(numeric), int(boolean), int(date_like)
        )
    return detected


def discovered_columns(mapping: MappingProposal) -> list[str]:
    """Unmapped source columns that can safely be carried through.

    A source column whose name collides with a canonical column is dropped,
    because the canonical meaning must win.
    """
    canonical = {c.value.lower() for c in CanonicalColumn}
    return [
        column
        for column in mapping.unmapped_source_columns
        if column.strip().lower() not in canonical
    ]


def build_select(
    mapping: MappingProposal,
    status_mapping: StatusMapping | None = None,
    discovered_types: dict[str, DataType] | None = None,
    reconstructions: dict[CanonicalColumn, object] | None = None,
    date_encodings: dict[str, DateEncoding] | None = None,
    source_types: dict[str, str] | None = None,
    stage_status_map: dict[str, OpportunityStatus] | None = None,
) -> tuple[str, list[DerivedColumn], StatusResolution]:
    """Build the SELECT list that conforms a raw text table.

    Returns the select clause, the provenance of any derived column, and how
    opportunity status was resolved.

    A canonical column the source lacks may be *reconstructed* from other source
    columns (ARCHITECTURE 12.15). The derivation is recorded on the returned
    `DerivedColumn` so nothing downstream can mistake a rebuilt value for one
    the export contained.
    """
    by_canonical = mapping.by_canonical()
    projections: list[str] = []
    derived: list[DerivedColumn] = []
    types_of = source_types or {}

    status_sql, resolution = resolve_status(mapping, status_mapping, stage_status_map)
    status_ident = quote_ident(CanonicalColumn.STATUS.value)

    for canonical, spec in CANONICAL_COLUMNS.items():
        if canonical is CanonicalColumn.STATUS:
            projections.append(f"{status_sql} AS {status_ident}")
            derived.append(
                DerivedColumn(
                    column=canonical,
                    rule=(
                        DerivationRule.STATUS_COLUMN
                        if resolution.is_authoritative
                        else DerivationRule.STAGE_KEYWORD
                    ),
                    note=resolution.note,
                    requires_confirmation=resolution.requires_confirmation,
                )
            )
            continue

        if canonical in (CanonicalColumn.IS_CLOSED, CanonicalColumn.IS_WON):
            # Always derived from status, so the two can never disagree.
            if canonical is CanonicalColumn.IS_CLOSED:
                expr = f"({status_sql}) IN ('won', 'lost')"
            else:
                expr = f"({status_sql}) = 'won'"
            projections.append(f"{expr} AS {quote_ident(canonical.value)}")
            derived.append(
                DerivedColumn(
                    column=canonical,
                    rule=(
                        DerivationRule.STATUS_COLUMN
                        if resolution.is_authoritative
                        else DerivationRule.STAGE_KEYWORD
                    ),
                    note=f"Derived from the canonical status column. {resolution.note}",
                    requires_confirmation=resolution.requires_confirmation,
                )
            )
            continue

        if canonical in by_canonical:
            source = by_canonical[canonical].source_column
            encoding = (date_encodings or {}).get(source)
            projections.append(
                f"{_cast_expression(source, spec.dtype, encoding, types_of.get(source))} "
                f"AS {quote_ident(canonical.value)}"
            )
            continue

        plan = (reconstructions or {}).get(canonical)
        if plan is not None:
            projections.append(
                f"{plan.expression_sql} AS {quote_ident(canonical.value)}"  # type: ignore[attr-defined]
            )
            derived.append(
                DerivedColumn(
                    column=canonical,
                    rule=DerivationRule.RECONSTRUCTED,
                    note=(
                        f"Rebuilt as `{plan.derivation}`; the source export carried no "  # type: ignore[attr-defined]
                        f"{canonical.value} column. {plan.note}"  # type: ignore[attr-defined]
                    ),
                    requires_confirmation=True,
                    expression=plan.derivation,  # type: ignore[attr-defined]
                    sources=plan.sources,  # type: ignore[attr-defined]
                    agreement_test_ids=plan.agreement_test_ids,  # type: ignore[attr-defined]
                    declared_by=plan.declared_by,  # type: ignore[attr-defined]
                )
            )

    types = discovered_types or {}
    for column in discovered_columns(mapping):
        dtype = types.get(column, DataType.VARCHAR)
        encoding = (date_encodings or {}).get(column)
        projections.append(
            f"{_cast_expression(column, dtype, encoding, types_of.get(column))} "
            f"AS {quote_ident(column)}"
        )

    return ",\n       ".join(projections), derived, resolution


def build_cast_failure_select(
    mapping: MappingProposal,
    discovered_types: dict[str, DataType] | None = None,
    date_encodings: dict[str, DateEncoding] | None = None,
    source_types: dict[str, str] | None = None,
) -> str:
    """Build a SELECT that counts genuine cast failures per column.

    A cast failure is a source cell that held text but did not survive the cast.
    A blank cell is a legitimate null, not a failure, so the two are counted
    apart. This must run against the raw text source; by the time a column is
    conformed the evidence is gone.
    """
    by_canonical = mapping.by_canonical()
    # A declared date column is validated by `data.dates`, where an invalid value
    # is fatal. Counting it here too would report every serial as a cast failure.
    declared = date_encodings or {}
    types_of = source_types or {}
    projections: list[str] = []

    def failure_count(source: str, dtype: DataType) -> str | None:
        """COUNT expression for one column, or None when it cannot fail."""
        mode = typed_cast_mode(types_of[source], dtype) if source in types_of else None
        if mode == "exact":
            return None
        if mode == "direct":
            if dtype is not DataType.DECIMAL:
                return None  # a widening cast: it cannot fail, so it costs no scan
            ident = quote_ident(source)
            return (
                f"COUNT(*) FILTER (WHERE {ident} IS NOT NULL "
                f"AND TRY_CAST({ident} AS {dtype.duckdb_type}) IS NULL)"
            )
        cleaned = f"NULLIF(TRIM(CAST({quote_ident(source)} AS VARCHAR)), '')"
        cast = f"TRY_CAST({cleaned} AS {dtype.duckdb_type})"
        return f"COUNT(*) FILTER (WHERE {cleaned} IS NOT NULL AND {cast} IS NULL)"

    for canonical, spec in CANONICAL_COLUMNS.items():
        if canonical not in by_canonical or spec.dtype is DataType.VARCHAR:
            continue
        source = by_canonical[canonical].source_column
        if source in declared:
            continue
        count = failure_count(source, spec.dtype)
        if count is not None:
            projections.append(f"{count} AS {quote_ident(canonical.value)}")
    for column, dtype in (discovered_types or {}).items():
        if dtype is DataType.VARCHAR or column in declared:
            continue
        count = failure_count(column, dtype)
        if count is not None:
            projections.append(f"{count} AS {quote_ident(column)}")
    if not projections:
        return "SELECT 1 AS _placeholder"
    return "SELECT " + ", ".join(projections)
