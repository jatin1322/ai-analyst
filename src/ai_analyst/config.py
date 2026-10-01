"""Application configuration.

Every knob that changes an analytical answer lives here, not in a call site, so
that a run record can state the configuration it was produced under.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ai_analyst.contracts.status import StatusMapping


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AI_ANALYST_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Storage
    data_root: Path = Path("data")

    # Fiscal calendar. Used by the semantic layer in a later milestone.
    fiscal_year_start_month: int = Field(default=1, ge=1, le=12)

    # Snapshot selection tolerance (ARCHITECTURE §5.1)
    max_snapshot_drift_days: int = Field(default=10, ge=0)

    # Profiling
    top_k_values: int = Field(default=10, ge=1, le=100)
    # Text detection thresholds (ARCHITECTURE 5.11). A VARCHAR column is a text
    # candidate rather than a categorical dimension when it clears both.
    text_min_mean_length: float = Field(default=40.0, gt=0)
    text_min_distinct_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    # Top values are only listed for columns with at most this many distinct
    # values. Beyond it they are noise, and the profile has to stay compact
    # enough to be used as model context.
    profile_max_top_value_cardinality: int = Field(default=50, ge=1)
    max_duplicate_samples: int = Field(default=20, ge=1)

    # Analyst context card (ARCHITECTURE 12.5). Above this many columns the
    # tier-0 card drops its name index and the agent reaches names through the
    # listing tool, so a wide tenant cannot silently blow the prompt budget.
    context_card_max_indexed_columns: int = Field(default=250, ge=1)
    # The tier-0 budget itself, enforced by test rather than hoped for.
    context_card_token_budget: int = Field(default=2000, ge=100)
    # Hard cap on a tier-2 row sample. A bounded sample is not the dataset.
    max_sample_rows: int = Field(default=20, ge=1, le=100)
    max_vanished_samples: int = Field(default=20, ge=1)

    # Planner (ARCHITECTURE 13.6, 13.16). Every limit the planning loop obeys
    # lives here, never as a constant inside the loop. The model is
    # configuration: the planner adapter is the only code that names a provider.
    planner_model: str = "claude-opus-5"
    planner_max_output_tokens: int = Field(default=16000, ge=1024)
    # Every model call counts as a turn: inspections, repairs and output retries.
    planner_max_turns: int = Field(default=12, ge=1)
    # Inspection tool calls per question (13.16: at most 8).
    planner_max_tool_calls: int = Field(default=8, ge=0)
    # Gate-rejected submissions returned to the planner (13.16: at most 2).
    planner_max_repairs: int = Field(default=2, ge=0)
    # Malformed outputs returned to the planner before the turn fails closed.
    planner_max_output_retries: int = Field(default=1, ge=0)
    # The whole planning conversation, estimated conservatively.
    planner_max_context_tokens: int = Field(default=24_000, ge=1_000)
    # One tool result, before it is truncated for the conversation.
    planner_max_tool_result_tokens: int = Field(default=1_500, ge=100)

    # Mapping
    fuzzy_match_threshold: float = Field(default=0.82, ge=0.0, le=1.0)

    # Authoritative opportunity status (ARCHITECTURE 5.13). Left unset until
    # the real status column is supplied; ingestion then falls back to stage
    # keywords and records the result as non-authoritative.
    status_column: str | None = None
    status_won_values: tuple[str, ...] = ()
    status_lost_values: tuple[str, ...] = ()
    status_open_values: tuple[str, ...] = ()
    status_excluded_values: tuple[str, ...] = ()

    def status_mapping(self) -> StatusMapping | None:
        """The configured status value mapping, or None when unset."""
        if not self.status_column:
            return None
        return StatusMapping(
            won_values=self.status_won_values,
            lost_values=self.status_lost_values,
            open_values=self.status_open_values,
            excluded_values=self.status_excluded_values,
        )

    # DuckDB
    duckdb_memory_limit: str = "2GB"
    duckdb_threads: int = Field(default=4, ge=1)

    # Execution caps (ARCHITECTURE §7.4)
    query_timeout_seconds: int = Field(default=30, ge=1)
    max_result_rows: int = Field(default=10_000, ge=1)

    def dataset_dir(self, dataset_id: str) -> Path:
        return self.data_root / "datasets" / dataset_id

    def canonical_dir(self, dataset_id: str) -> Path:
        return self.dataset_dir(dataset_id) / "canonical" / "snapshots"

    def raw_dir(self, dataset_id: str) -> Path:
        return self.dataset_dir(dataset_id) / "raw"

    def schema_path(self, dataset_id: str) -> Path:
        return self.dataset_dir(dataset_id) / "schema.json"

    def classifications_path(self, dataset_id: str) -> Path:
        return self.dataset_dir(dataset_id) / "classifications.json"

    def agreement_path(self, dataset_id: str) -> Path:
        return self.dataset_dir(dataset_id) / "agreement.json"

    def profile_path(self, dataset_id: str) -> Path:
        return self.dataset_dir(dataset_id) / "profile.json"

    def results_dir(self, dataset_id: str) -> Path:
        return self.dataset_dir(dataset_id) / "results"


_settings: Settings | None = None


def get_settings() -> Settings:
    """Process-wide settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Drop the cached settings. Used by tests."""
    global _settings
    _settings = None
