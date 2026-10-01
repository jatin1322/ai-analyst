"""Monetary profiling policy (ARCHITECTURE 12.16).

Money is decided semantically, from a classification or a concept binding, and
never from numeric shape. The consequence under test: a money column reports
exact extremes and no float mean, and so does a column whose monetary status
nobody has established.

The bug this closes was real. `terminal_amount` and `rep_avg_deal_amount` are
money stored as DOUBLE, and the profiler used to report a float mean, median
and standard deviation for both, contradicting the rule in CLAUDE.md.
"""

from __future__ import annotations

from ai_analyst.config import Settings
from ai_analyst.contracts.columns import MonetaryStatus
from ai_analyst.contracts.opportunity_snapshot_v1 import (
    MONETARY_COLUMNS,
    OPPORTUNITY_SNAPSHOT_V1,
)
from ai_analyst.contracts.profile import ProfileKind
from ai_analyst.contracts.schema import DataType
from ai_analyst.data.ingest import ingest
from ai_analyst.data.profiler import profile_dataset
from tests.fixtures.production_shape import write_production_csv


def _profile(tmp_path, settings: Settings):
    source = tmp_path / "export.csv"
    write_production_csv(source, include_close_date=False)
    result = ingest(source, "money", settings=settings, column_registry=OPPORTUNITY_SNAPSHOT_V1)
    profile = profile_dataset("money", result.schema, settings=settings, registry=result.registry)
    return result, {c.name: c for c in profile.columns}


def test_money_is_declared_semantically_not_by_numeric_shape():
    # Two decimal places make a ratio as easily as an amount. These names are
    # declared; nothing infers money from values.
    assert {"new_amount", "terminal_amount", "rep_avg_deal_amount"} == MONETARY_COLUMNS
    for name in MONETARY_COLUMNS:
        assert OPPORTUNITY_SNAPSHOT_V1.classify(name).is_monetary
    # Lookalikes that are emphatically not money.
    for name in (
        "deal_amount_vs_rep_avg",
        "deal_amount_percentile_overall",
        "deal_amount_rank_pct",
        "new_amount_change",
    ):
        assert not OPPORTUNITY_SNAPSHOT_V1.classify(name).is_monetary


def test_a_money_column_stored_as_double_gets_no_float_summary(
    tmp_path, settings: Settings
):
    _, columns = _profile(tmp_path, settings)
    for name in ("terminal_amount", "rep_avg_deal_amount"):
        column = columns[name]
        assert column.dtype is DataType.DOUBLE
        assert column.monetary is MonetaryStatus.MONETARY
        assert not column.has_float_summary, f"{name} leaked a float mean"
        assert column.mean is None and column.median is None and column.stddev is None
        # The safe descriptive metadata survives.
        assert column.min_value is not None and column.max_value is not None
        assert column.distinct_count > 0


def test_the_canonical_amount_column_is_monetary_and_unaveraged(
    tmp_path, settings: Settings
):
    _, columns = _profile(tmp_path, settings)
    amount = columns["amount"]
    assert amount.dtype is DataType.DECIMAL
    assert amount.monetary is MonetaryStatus.MONETARY
    assert not amount.has_float_summary
    # Exact strings, so no amount round-trips through a float even in the file.
    assert "." in (amount.min_value or "")


def test_a_column_whose_monetary_status_is_unknown_gets_no_float_summary(
    tmp_path, settings: Settings
):
    # An unclassified column is UNKNOWN, not NON_MONETARY. Nothing about it has
    # been established, so its values never reach a float summary statistic.
    source = tmp_path / "unknown.csv"
    source.write_text(
        "as_of,opp_id,mystery_value\n"
        "2025-01-01,O-1,1000.50\n"
        "2025-02-01,O-1,2000.75\n",
        encoding="utf-8",
    )
    result = ingest(source, "unknown", settings=settings)
    profile = profile_dataset(
        "unknown", result.schema, settings=settings, registry=result.registry
    )
    column = {c.name: c for c in profile.columns}["mystery_value"]
    assert column.kind is ProfileKind.NUMERIC
    assert column.monetary is MonetaryStatus.UNKNOWN
    assert not column.has_float_summary
    assert column.min_value is not None and column.max_value is not None


def test_a_classified_non_monetary_column_keeps_its_summary_statistics(
    tmp_path, settings: Settings
):
    # Suppressing statistics everywhere would make the profile useless. A
    # column someone classified is known not to be money.
    _, columns = _profile(tmp_path, settings)
    days = columns["account_ti_first_won"]
    assert days.monetary is MonetaryStatus.NON_MONETARY
    assert days.has_float_summary
    assert days.mean is not None
    # And its sentinel is still excluded from that mean.
    assert days.sentinel_count > 0


def test_monetary_status_fails_closed_for_an_unclassified_column():
    unclassified = OPPORTUNITY_SNAPSHOT_V1.classify("a_column_nobody_documented")
    assert not unclassified.classified
    assert unclassified.monetary_status is MonetaryStatus.UNKNOWN
    assert not MonetaryStatus.UNKNOWN.allows_float_summary
    assert not MonetaryStatus.MONETARY.allows_float_summary
    assert MonetaryStatus.NON_MONETARY.allows_float_summary


def test_profiling_precedes_binding_so_unknown_is_what_protects_a_money_column(
    tmp_path, settings: Settings
):
    # Ordering that matters: the profiler runs at ingestion, before any concept
    # binding exists, so `ConceptBindings.monetary_columns()` can never feed it.
    # What protects an undeclared money column is that UNKNOWN withholds the
    # float summary, not that the binding later identifies it.
    from ai_analyst.contracts.concepts import BusinessConcept
    from ai_analyst.contracts.tenant import TenantProfile
    from ai_analyst.data.binding import build_bindings

    source = tmp_path / "money.csv"
    source.write_text(
        "as_of,opp_id,booking_value\n"
        "2025-01-01,O-1,1000.50\n"
        "2025-02-01,O-1,2500.25\n",
        encoding="utf-8",
    )
    result = ingest(source, "declared", settings=settings)
    profile = profile_dataset(
        "declared", result.schema, settings=settings, registry=result.registry
    )
    column = {c.name: c for c in profile.columns}["booking_value"]
    assert column.monetary is MonetaryStatus.UNKNOWN
    assert not column.has_float_summary

    tenant = TenantProfile(
        tenant_id="t", concept_columns={BusinessConcept.AMOUNT: ("booking_value",)}
    )
    bindings = build_bindings(result.schema, result.registry, profile, tenant=tenant)
    assert "booking_value" in bindings.monetary_columns()
    # The binding arrives after the fact; no float statistic was ever computed.
    assert not column.has_float_summary


# -- monetary analytical measures (ARCHITECTURE 12.16) -----------------------------


def test_a_double_money_column_resolves_to_an_explicit_decimal_measure():
    from ai_analyst.data.money import monetary_measure_sql

    assert monetary_measure_sql("terminal_amount", DataType.DOUBLE) == (
        'CAST("terminal_amount" AS DECIMAL(18,2))'
    )
    # An integer amount is converted too, so every measure has the same type.
    assert "DECIMAL(18,2)" in monetary_measure_sql("cents", DataType.BIGINT)
    # A column that is already exact passes through untouched.
    assert monetary_measure_sql("amount", DataType.DECIMAL) == '"amount"'


def test_a_column_that_cannot_hold_money_is_refused_not_coerced():
    import pytest

    from ai_analyst.data.money import monetary_measure_sql

    for dtype in (DataType.VARCHAR, DataType.DATE, DataType.BOOLEAN):
        with pytest.raises(ValueError, match="cannot be resolved as a monetary measure"):
            monetary_measure_sql("x", dtype)


def test_the_decimal_conversion_removes_the_float_error_it_exists_to_remove():
    # Ten payments of 0.10 stored as DOUBLE sum to 0.9999999999999999. Converted
    # first, they sum to exactly 1.00. This is why the cast happens before the
    # aggregate and not after it.
    import duckdb

    from ai_analyst.data.money import monetary_measure_sql

    conn = duckdb.connect()
    conn.execute("CREATE TABLE t AS SELECT 0.1::DOUBLE AS amount FROM range(10)")
    floaty = conn.execute("SELECT SUM(amount) FROM t").fetchone()[0]
    exact = conn.execute(
        f"SELECT SUM({monetary_measure_sql('amount', DataType.DOUBLE)}) FROM t"
    ).fetchone()[0]
    assert floaty != 1.0
    assert str(exact) == "1.00"


def test_the_conversion_reports_what_it_would_round_or_cannot_hold():
    import duckdb

    from ai_analyst.data.money import measure_monetary_conversion

    conn = duckdb.connect()
    conn.execute(
        "CREATE TABLE t AS SELECT * FROM (VALUES (10.50), (10.505), (99.99), (1e20)) v(amount)"
    )
    result = measure_monetary_conversion(conn, "t", "amount", DataType.DOUBLE)
    assert result.checked_rows == 4
    assert result.altered_rows == 1  # 10.505 has a third decimal and is rounded
    assert result.unrepresentable_rows == 1  # 1e20 does not fit DECIMAL(18,2)
    assert not result.is_lossless


def test_an_already_exact_column_converts_losslessly():
    import duckdb

    from ai_analyst.data.money import measure_monetary_conversion

    conn = duckdb.connect()
    conn.execute("CREATE TABLE t AS SELECT * FROM (VALUES (10.50::DECIMAL(18,2))) v(amount)")
    result = measure_monetary_conversion(conn, "t", "amount", DataType.DECIMAL)
    assert result.is_lossless


# -- structural profiling versus monetary meaning ------------------------------------


def test_a_withheld_summary_says_why(tmp_path, settings: Settings):
    _, columns = _profile(tmp_path, settings)
    assert "monetary column" in columns["terminal_amount"].summary_withheld_reason
    assert columns["account_ti_first_won"].summary_withheld_reason is None  # has a mean


def test_an_unknown_column_says_it_is_unknown_rather_than_money(tmp_path, settings: Settings):
    source = tmp_path / "unknown.csv"
    source.write_text(
        "as_of,opp_id,mystery_value\n2025-01-01,O-1,10.5\n2025-02-01,O-1,20.5\n",
        encoding="utf-8",
    )
    result = ingest(source, "unk", settings=settings)
    profile = profile_dataset("unk", result.schema, settings=settings, registry=result.registry)
    column = {c.name: c for c in profile.columns}["mystery_value"]
    # Not called money: nobody has said so. Just not established as non-money.
    assert column.summary_withheld_reason == (
        "monetary status unknown: not established as non-monetary"
    )
    # The structural half of the profile is intact.
    assert column.distinct_count == 2 and column.null_count == 0
    assert column.min_value is not None and column.max_value is not None


def test_a_tenant_declaration_reaches_the_profiler_and_withholds_the_summary(
    tmp_path, settings: Settings
):
    # `rep_deal_velocity` is classified non-monetary by the export registry, so
    # by default it keeps its mean. A tenant declaring it as money outranks that,
    # and the declaration has to reach the profiler, which runs before any
    # concept binding exists.
    from ai_analyst.contracts.concepts import BusinessConcept
    from ai_analyst.contracts.tenant import TenantProfile
    from ai_analyst.data.dataset import register_dataset

    source = tmp_path / "export.csv"
    write_production_csv(source, include_close_date=False)

    plain = register_dataset(
        source, "plain", column_registry=OPPORTUNITY_SNAPSHOT_V1, settings=settings
    )
    assert {c.name: c for c in plain.profile.columns}["rep_deal_velocity"].has_float_summary

    tenant = TenantProfile(
        tenant_id="t", concept_columns={BusinessConcept.TERMINAL_AMOUNT: ("rep_deal_velocity",)}
    )
    declared = register_dataset(
        source,
        "declared_money",
        column_registry=OPPORTUNITY_SNAPSHOT_V1,
        tenant=tenant,
        settings=settings,
    )
    column = {c.name: c for c in declared.profile.columns}["rep_deal_velocity"]
    assert column.monetary is MonetaryStatus.MONETARY
    assert not column.has_float_summary
    assert column.min_value is not None  # structural profiling survives


def test_a_declaration_resolves_a_renamed_header_to_the_conformed_column(
    tmp_path, settings: Settings
):
    from ai_analyst.contracts.concepts import BusinessConcept
    from ai_analyst.contracts.tenant import TenantProfile
    from ai_analyst.data.money import declared_monetary_columns

    source = tmp_path / "arr.csv"
    source.write_text("as_of,opp_id,ARR\n2025-01-01,O-1,1000.50\n", encoding="utf-8")
    result = ingest(source, "arr", settings=settings)
    tenant = TenantProfile(tenant_id="t", concept_columns={BusinessConcept.AMOUNT: ("ARR",)})
    assert declared_monetary_columns(tenant, result.schema) == {"arr"}
    assert declared_monetary_columns(None, result.schema) == frozenset()


def test_a_tenant_declaration_can_only_add_money_never_remove_it(tmp_path, settings: Settings):
    from ai_analyst.contracts.tenant import TenantProfile
    from ai_analyst.data.dataset import register_dataset

    source = tmp_path / "export.csv"
    write_production_csv(source, include_close_date=False)
    dataset = register_dataset(
        source,
        "still_money",
        column_registry=OPPORTUNITY_SNAPSHOT_V1,
        tenant=TenantProfile(tenant_id="t"),  # declares nothing
        settings=settings,
    )
    column = {c.name: c for c in dataset.profile.columns}["terminal_amount"]
    assert column.monetary is MonetaryStatus.MONETARY
    assert not column.has_float_summary
