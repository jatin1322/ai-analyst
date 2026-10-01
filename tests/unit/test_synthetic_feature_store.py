"""Tests for the synthetic feature-store generator (WP7).

Nothing here touches the product's ingestion/semantic path — this generator
is a standalone demo-data tool. The tests exist to prove the generator's own
promises: determinism, grain integrity, planted ground truth, invariants
that must hold by construction, and that no real company name leaks through.
"""

from __future__ import annotations

import json

import duckdb
import pytest

from ai_analyst.synthetic.feature_store import (
    ALL_COLUMNS,
    RENAME_MAPS,
    GeneratedDataset,
    GeneratorConfig,
    contains_forbidden_name,
    generate,
    variant_columns,
    write_csv,
    write_hive_parquet,
    write_parquet,
)

DEFAULT_CONFIG_KWARGS = dict(seed=7, n_opportunities=100, n_quarters=2)


@pytest.fixture(scope="module")
def default_dataset() -> GeneratedDataset:
    return generate(GeneratorConfig(**DEFAULT_CONFIG_KWARGS))


@pytest.fixture(scope="module")
def default_csv(tmp_path_factory, default_dataset: GeneratedDataset):
    path = tmp_path_factory.mktemp("synth") / "snapshots.csv"
    write_csv(default_dataset, path, variant="base")
    return path


@pytest.fixture(scope="module")
def con(default_csv) -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect(":memory:")
    # A materialized TABLE, not a VIEW over read_csv: a view re-parses and
    # re-auto-detects the CSV on every query, which is most of this test
    # module's wall-clock time across ~20 queries against it.
    connection.execute(
        f"CREATE TABLE snapshots AS SELECT * FROM read_csv('{default_csv.as_posix()}', header=true)"
    )
    yield connection
    connection.close()


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_same_seed_same_config_gives_identical_csv_bytes(tmp_path):
    cfg = GeneratorConfig(**DEFAULT_CONFIG_KWARGS)
    ds1 = generate(cfg)
    ds2 = generate(cfg)
    p1 = tmp_path / "a.csv"
    p2 = tmp_path / "b.csv"
    write_csv(ds1, p1)
    write_csv(ds2, p2)
    assert p1.read_bytes() == p2.read_bytes()


def test_different_seed_gives_different_csv_bytes(tmp_path):
    cfg_a = GeneratorConfig(**{**DEFAULT_CONFIG_KWARGS, "seed": 1})
    cfg_b = GeneratorConfig(**{**DEFAULT_CONFIG_KWARGS, "seed": 2})
    pa = tmp_path / "a.csv"
    pb = tmp_path / "b.csv"
    write_csv(generate(cfg_a), pa)
    write_csv(generate(cfg_b), pb)
    assert pa.read_bytes() != pb.read_bytes()


def test_generation_is_fast(default_dataset: GeneratedDataset):
    # Generation itself already happened in the module-scoped fixture; this
    # just asserts the size lands in the range the task describes so a
    # regression that silently shrinks/grows the panel gets caught.
    assert 5_000 <= default_dataset.ground_truth.n_rows <= 60_000


# ---------------------------------------------------------------------------
# Grain
# ---------------------------------------------------------------------------


def test_opp_id_as_of_is_unique(con):
    total = con.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0]
    distinct = con.execute(
        "SELECT COUNT(*) FROM (SELECT DISTINCT opp_id, as_of FROM snapshots)"
    ).fetchone()[0]
    assert total == distinct
    assert total > 0


def test_rows_are_generated_already_sorted_by_opp_id_and_as_of(default_dataset):
    # generate() relies on opp_id's zero-padding, per-opportunity date
    # ordering, and DST-pair time ordering to skip an explicit global sort
    # (see the comment in generate()). Verify that reasoning actually holds.
    opp_id_idx = ALL_COLUMNS.index("opp_id")
    as_of_idx = ALL_COLUMNS.index("as_of")
    keys = [(r[opp_id_idx], r[as_of_idx]) for r in default_dataset.rows]
    assert keys == sorted(keys)


def test_duplicate_calendar_day_groups_match_ground_truth(default_dataset, con):
    dup_groups = con.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT opp_id, as_of_date, COUNT(*) c
            FROM snapshots GROUP BY 1, 2 HAVING c = 2
        )
        """
    ).fetchone()[0]
    assert dup_groups == default_dataset.ground_truth.duplicate_groups
    assert dup_groups > 0  # non-vacuous: the DST mechanic actually fired

    # no group has more than 2 rows on the same calendar day
    max_group = con.execute(
        """
        SELECT MAX(c) FROM (
            SELECT opp_id, as_of_date, COUNT(*) c FROM snapshots GROUP BY 1, 2
        )
        """
    ).fetchone()[0]
    assert max_group == 2


def test_conflicting_groups_match_ground_truth(default_dataset, con):
    conflicting = con.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT opp_id, as_of_date, COUNT(DISTINCT stage) n_stages, COUNT(*) c
            FROM snapshots GROUP BY 1, 2 HAVING c = 2
        ) t WHERE n_stages > 1
        """
    ).fetchone()[0]
    assert conflicting == default_dataset.ground_truth.conflicting_groups
    assert conflicting > 0
    assert conflicting <= default_dataset.ground_truth.duplicate_groups


def test_non_conflicting_duplicate_pairs_are_identical_except_as_of(default_dataset, con):
    # Every duplicate pair either differs only in `as_of` (and possibly
    # `stage`, if it's a planted conflict) and nothing else.
    other_cols = [c for c in ALL_COLUMNS if c not in ("as_of", "stage")]
    # COUNT(DISTINCT col) ignores NULLs, so two NULL values in a pair count
    # as agreeing (<=1 distinct non-null value), unlike a raw MIN=MAX check.
    select_list = ", ".join(f"COUNT(DISTINCT {c}) <= 1 AS ok_{i}" for i, c in enumerate(other_cols))
    row = con.execute(
        f"""
        WITH dup AS (
            SELECT opp_id, as_of_date FROM snapshots GROUP BY 1, 2 HAVING COUNT(*) = 2
        )
        SELECT {select_list}
        FROM snapshots s JOIN dup USING (opp_id, as_of_date)
        GROUP BY s.opp_id, s.as_of_date
        """
    ).fetchall()
    assert row, "expected at least one duplicate group"
    for record in row:
        assert all(record), "a duplicate pair differed in a column other than as_of/stage"


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("window", ["7d", "14d", "30d", "60d", "90d", "180d", "lifetime"])
def test_inbound_plus_outbound_equals_email_count(con, window):
    bad = con.execute(
        f"""
        SELECT COUNT(*) FROM snapshots
        WHERE inbound_count_{window} + outbound_count_{window} != email_count_{window}
        """
    ).fetchone()[0]
    assert bad == 0


@pytest.mark.parametrize(
    "family", ["email_count", "inbound_count", "outbound_count", "meeting_count"]
)
def test_windows_are_monotone(con, family):
    labels = ["7d", "14d", "30d", "60d", "90d", "180d", "lifetime"]
    conditions = " OR ".join(
        f"{family}_{labels[i]} > {family}_{labels[i+1]}" for i in range(len(labels) - 1)
    )
    bad = con.execute(f"SELECT COUNT(*) FROM snapshots WHERE {conditions}").fetchone()[0]
    assert bad == 0


def test_label_null_iff_mask_zero(con):
    bad_win = con.execute(
        "SELECT COUNT(*) FROM snapshots WHERE (win_label IS NULL) != (win_label_mask = 0)"
    ).fetchone()[0]
    bad_slip = con.execute(
        "SELECT COUNT(*) FROM snapshots WHERE (slip_label IS NULL) != (slip_label_mask = 0)"
    ).fetchone()[0]
    assert bad_win == 0
    assert bad_slip == 0
    # non-vacuous: both mask values are actually present
    counts = con.execute(
        "SELECT win_label_mask, COUNT(*) FROM snapshots GROUP BY 1"
    ).fetchall()
    assert {0, 1} == {c[0] for c in counts}


def test_slip_label_matches_a_later_observed_push(con):
    # slip_label is forward-looking: for a mask==1 row, label must be 1 iff
    # the SAME opportunity has some later row where close_date_updated_days
    # reset to 0 (a push). Creation day can never be "later" than itself,
    # and DST twin rows share the same as_of_date so neither can be a
    # "later" row relative to the other.
    bad = con.execute(
        """
        SELECT COUNT(*) FROM snapshots s
        WHERE s.slip_label_mask = 1
          AND s.slip_label != (
              EXISTS (
                  SELECT 1 FROM snapshots t
                  WHERE t.opp_id = s.opp_id
                    AND t.as_of_date > s.as_of_date
                    AND t.close_date_updated_days = 0
              )
          )::INTEGER
        """
    ).fetchone()[0]
    assert bad == 0
    # non-vacuous: both label values occur among mask==1 rows
    labels = con.execute(
        "SELECT slip_label, COUNT(*) FROM snapshots WHERE slip_label_mask = 1 GROUP BY 1"
    ).fetchall()
    assert {0, 1} == {v[0] for v in labels}


def test_closed_deals_stay_closed(con):
    violations = con.execute(
        """
        WITH closed_first AS (
            SELECT opp_id, MIN(as_of) AS c_as_of
            FROM snapshots WHERE stage IN ('Closed Won', 'Closed Lost')
            GROUP BY 1
        )
        SELECT COUNT(*)
        FROM snapshots s JOIN closed_first cf USING (opp_id)
        WHERE s.as_of > cf.c_as_of AND s.stage NOT IN ('Closed Won', 'Closed Lost')
        """
    ).fetchone()[0]
    assert violations == 0
    n_closed_opps = con.execute(
        "SELECT COUNT(DISTINCT opp_id) FROM snapshots WHERE stage IN ('Closed Won', 'Closed Lost')"
    ).fetchone()[0]
    assert n_closed_opps > 0  # non-vacuous


def test_rows_continue_appearing_after_close(con):
    # Otherwise "stays closed" would pass vacuously (closed opps vanishing
    # from the panel right at close).
    rows_after_close = con.execute(
        """
        SELECT COUNT(*) FROM snapshots
        WHERE outcome_date IS NOT NULL AND as_of_date > outcome_date
        """
    ).fetchone()[0]
    assert rows_after_close > 0


def test_outcome_fate_agrees_with_terminal_stage(con):
    # outcome_fate is BACKFILLED (it's a leakage column, see label_convention):
    # every row of an opportunity that closes in the panel carries the
    # eventual fate, not just its own closing row. So the check is per
    # opportunity (constant, and matching the opportunity's own terminal
    # stage), not per row.
    bad = con.execute(
        """
        WITH per_opp AS (
            SELECT opp_id,
                   COUNT(DISTINCT outcome_fate) AS distinct_fates,
                   MAX(CASE WHEN stage = 'Closed Won' THEN 1 ELSE 0 END) AS has_won,
                   MAX(CASE WHEN stage = 'Closed Lost' THEN 1 ELSE 0 END) AS has_lost,
                   MAX(outcome_fate) AS fate
            FROM snapshots GROUP BY 1
        )
        SELECT COUNT(*) FROM per_opp
        WHERE distinct_fates > 1
           OR (has_won = 1 AND fate != 'W')
           OR (has_lost = 1 AND fate != 'L')
           OR (has_won = 0 AND has_lost = 0 AND fate IS NOT NULL)
        """
    ).fetchone()[0]
    assert bad == 0


def test_outcome_fate_leaks_onto_pre_close_rows(con):
    # The whole point of this column: it must be visible on rows well
    # before the opportunity actually reaches a terminal stage.
    leaking_rows = con.execute(
        """
        SELECT COUNT(*) FROM snapshots
        WHERE stage NOT IN ('Closed Won', 'Closed Lost') AND outcome_fate IS NOT NULL
        """
    ).fetchone()[0]
    assert leaking_rows > 0


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ["alpha", "beta"])
def test_variant_renames_columns_and_preserves_row_values(default_dataset, tmp_path, variant):
    base_path = tmp_path / "base.csv"
    variant_path = tmp_path / f"{variant}.csv"
    write_csv(default_dataset, base_path, variant="base")
    write_csv(default_dataset, variant_path, variant=variant)

    header = variant_path.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header == variant_columns(variant)
    # at least the core rename set actually changed something
    mapping = RENAME_MAPS[variant]
    assert mapping  # non-empty
    for base_name, renamed in mapping.items():
        assert base_name in ALL_COLUMNS
        assert renamed in header
        assert base_name not in header or base_name == renamed

    base_lines = base_path.read_text(encoding="utf-8").splitlines()
    variant_lines = variant_path.read_text(encoding="utf-8").splitlines()
    assert len(base_lines) == len(variant_lines)
    # bodies (everything after the header row) are byte-identical: only
    # names changed, not values or row order.
    assert base_lines[1:] == variant_lines[1:]


def test_variants_dont_collide_with_each_other(default_dataset, tmp_path):
    alpha = variant_columns("alpha")
    beta = variant_columns("beta")
    assert alpha != beta


# ---------------------------------------------------------------------------
# Planted effects
# ---------------------------------------------------------------------------


PUSH_EFFECT_QUERY = """
    SELECT inbound_count_30d = 0 AS silent,
           AVG(CASE WHEN close_date_updated_days = 0 THEN 1.0 ELSE 0.0 END) AS push_rate,
           COUNT(*) AS n
    FROM snapshots
    WHERE as_of_date > created_date
      AND (win_label_mask = 0 OR as_of_date < outcome_date)
    GROUP BY 1
"""

WIN_EFFECT_QUERY = """
    SELECT meeting_count_30d >= 2 AS many_meetings,
           AVG(win_label) AS win_rate,
           COUNT(*) AS n
    FROM snapshots
    WHERE as_of_date = outcome_date
    GROUP BY 1
"""


def test_silent_inbound_email_predicts_close_date_push(con):
    rows = con.execute(PUSH_EFFECT_QUERY).fetchall()
    rates = {silent: (rate, n) for silent, rate, n in rows}
    assert False in rates and True in rates
    active_rate, active_n = rates[False]
    silent_rate, silent_n = rates[True]
    assert active_n > 50 and silent_n > 20, "planted-effect population too small to trust"
    assert silent_rate > active_rate * 3, (
        f"expected silent inbound to sharply raise push rate: "
        f"active={active_rate:.4f} (n={active_n}) silent={silent_rate:.4f} (n={silent_n})"
    )


def test_more_meetings_predicts_higher_win_rate(con):
    rows = con.execute(WIN_EFFECT_QUERY).fetchall()
    rates = {many: (rate, n) for many, rate, n in rows}
    assert False in rates and True in rates
    low_rate, low_n = rates[False]
    high_rate, high_n = rates[True]
    assert low_n > 5 and high_n > 5, "planted-effect population too small to trust"
    assert high_rate - low_rate > 0.15, (
        f"expected meeting-heavy close days to win more often: "
        f"few_meetings={low_rate:.3f} (n={low_n}) many_meetings={high_rate:.3f} (n={high_n})"
    )


def test_effects_are_not_seed_luck(tmp_path):
    # The two tests above run on the module's default seed only. This
    # re-checks both effects across several other seeds so a change that
    # makes the effect real only for one lucky seed gets caught.
    for seed in (1, 2, 3, 4, 5):
        cfg = GeneratorConfig(seed=seed, n_opportunities=100, n_quarters=2)
        ds = generate(cfg)
        path = tmp_path / f"seed_{seed}.csv"
        write_csv(ds, path)
        con = duckdb.connect(":memory:")
        try:
            csv_url = path.as_posix()
            con.execute(
                f"CREATE TABLE snapshots AS SELECT * FROM read_csv('{csv_url}', header=true)"
            )
            push_rows = dict(
                (silent, rate) for silent, rate, _ in con.execute(PUSH_EFFECT_QUERY).fetchall()
            )
            win_rows = dict(
                (many, rate) for many, rate, _ in con.execute(WIN_EFFECT_QUERY).fetchall()
            )
        finally:
            con.close()
        assert push_rows[True] > push_rows[False] * 2, f"push effect too weak for seed {seed}"
        assert win_rows[True] - win_rows[False] > 0.1, f"win effect too weak for seed {seed}"


def test_effect_assertions_would_fail_without_the_planted_effects(monkeypatch, tmp_path):
    # Mutation check (CLAUDE.md #24): if the hazards/effect are neutralised,
    # the same statistics used above must no longer show the effect. This
    # proves the tests above are actually discriminating, not just
    # confirming whatever the generator happens to produce.
    import ai_analyst.synthetic.feature_store as fs

    monkeypatch.setattr(fs, "SILENT_PUSH_HAZARD", fs.BASE_PUSH_HAZARD)
    monkeypatch.setattr(fs, "MEETING_WIN_EFFECT", 0.0)

    cfg = GeneratorConfig(seed=7, n_opportunities=300, n_quarters=2)
    ds = generate(cfg)
    path = tmp_path / "neutralized.csv"
    write_csv(ds, path)
    con = duckdb.connect(":memory:")
    try:
        csv_url = path.as_posix()
        con.execute(f"CREATE TABLE snapshots AS SELECT * FROM read_csv('{csv_url}', header=true)")
        push_rows = dict(
            (silent, rate) for silent, rate, _ in con.execute(PUSH_EFFECT_QUERY).fetchall()
        )
        win_rows = dict((many, rate) for many, rate, _ in con.execute(WIN_EFFECT_QUERY).fetchall())
    finally:
        con.close()
    assert not (push_rows[True] > push_rows[False] * 3)
    assert not (win_rows[True] - win_rows[False] > 0.15)


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def test_parquet_writer_round_trips_row_count(default_dataset, tmp_path):
    csv_path = tmp_path / "s.csv"
    write_csv(default_dataset, csv_path)
    out = write_parquet(default_dataset, csv_path, tmp_path / "s.parquet", "base")
    con = duckdb.connect(":memory:")
    try:
        n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out.as_posix()}')").fetchone()[0]
    finally:
        con.close()
    assert n == default_dataset.ground_truth.n_rows


def test_hive_parquet_writer_round_trips_row_count_and_partitions(default_dataset, tmp_path):
    csv_path = tmp_path / "s.csv"
    write_csv(default_dataset, csv_path)
    out_dir = write_hive_parquet(default_dataset, csv_path, tmp_path / "hive", "base")
    con = duckdb.connect(":memory:")
    try:
        glob = f"{out_dir.as_posix()}/**/*.parquet"
        n = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{glob}', hive_partitioning=true)"
        ).fetchone()[0]
        quarters = con.execute(
            f"SELECT COUNT(DISTINCT as_of_qtr) FROM read_parquet('{glob}', hive_partitioning=true)"
        ).fetchone()[0]
    finally:
        con.close()
    assert n == default_dataset.ground_truth.n_rows
    assert quarters >= 2


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_conflict_pairs_exceeding_duplicates_raises():
    with pytest.raises(ValueError):
        generate(
            GeneratorConfig(
                seed=1,
                n_opportunities=5,
                n_quarters=1,
                dst_dates=[],
                dst_duplicates_per_date=0,
                conflict_pairs=1,
            )
        )


def test_dst_date_outside_panel_window_rejected():
    from datetime import date

    with pytest.raises(ValueError):
        GeneratorConfig(
            start_date=date(2025, 1, 1),
            n_quarters=1,
            dst_dates=[date(2026, 1, 1)],
        )


# ---------------------------------------------------------------------------
# No real names anywhere
# ---------------------------------------------------------------------------


def test_no_forbidden_real_names_in_columns_or_ground_truth(default_dataset):
    haystacks: list[str] = list(ALL_COLUMNS)
    for variant in ("alpha", "beta"):
        haystacks.extend(variant_columns(variant))
    haystacks.append(json.dumps(default_dataset.ground_truth.model_dump(mode="json")))

    for text in haystacks:
        assert not contains_forbidden_name(text), f"forbidden name fragment found in {text!r}"


def test_no_forbidden_real_names_in_any_value(default_dataset):
    # Every DISTINCT stringified value across the whole dataset, not a
    # sample: the set of distinct values is small (ids, stages, categories,
    # etc. repeat constantly), so this is cheap and exhaustive rather than
    # a spot check that could miss a rare offending value.
    distinct_values: set[str] = set()
    for row in default_dataset.rows:
        for value in row:
            if value is not None:
                distinct_values.add(str(value))

    assert len(distinct_values) > 0
    for value in distinct_values:
        assert not contains_forbidden_name(value), (
            f"forbidden name fragment found in value {value!r}"
        )
