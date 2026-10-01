"""Application configuration."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from ai_analyst.config import Settings, get_settings, reset_settings


def test_defaults_match_the_architecture():
    s = Settings()
    assert s.fiscal_year_start_month == 1
    assert s.max_snapshot_drift_days == 10
    assert s.max_result_rows == 10_000
    assert s.query_timeout_seconds == 30


def test_path_helpers_are_rooted_at_data_root(tmp_path: Path):
    s = Settings(data_root=tmp_path)
    assert s.dataset_dir("d1") == tmp_path / "datasets" / "d1"
    assert s.canonical_dir("d1").name == "snapshots"
    assert s.schema_path("d1").name == "schema.json"
    assert s.profile_path("d1").name == "profile.json"
    assert s.raw_dir("d1").name == "raw"
    assert s.results_dir("d1").name == "results"


@pytest.mark.parametrize("month", [0, 13, -1])
def test_fiscal_month_is_bounded(month):
    with pytest.raises(ValidationError):
        Settings(fiscal_year_start_month=month)


def test_env_vars_use_the_prefix(monkeypatch):
    monkeypatch.setenv("AI_ANALYST_FISCAL_YEAR_START_MONTH", "2")
    reset_settings()
    assert get_settings().fiscal_year_start_month == 2
    reset_settings()


def test_get_settings_is_a_singleton():
    reset_settings()
    assert get_settings() is get_settings()
