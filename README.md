# AI Analyst

Natural-language analytics over multi-quarter opportunity snapshot data, where
the model reasons and deterministic tools compute.

Read [ARCHITECTURE.md](ARCHITECTURE.md) first. The central decision is
plan-as-data: the LLM emits a typed, validated `AnalysisPlan` and a
deterministic compiler turns it into DuckDB SQL, so no number the system reports
originates in the model.

## Status

Milestones 1 to 3 of the MVP: the data and contract layer, column classification,
and the dataset-scoped registry with complete profiling. Milestone 4 designed
the multi-tenant architecture (docs/DESIGN_LOG.md §12) and milestone 5 built its
foundation: the analytical ontology, concept bindings, agreement tests, the
dataset understanding layer and the tiered context card. Milestone 6 built the
deterministic semantic engine: a hand-written `AnalysisPlan` is validated,
compiled to DuckDB SQL and executed into a checked `ResultSet`, with no LLM
anywhere in the path. Milestone 7 built every deterministic piece the LLM layer
needs (docs/DESIGN_LOG.md §13). Later milestones added the LLM planner and
onboarding for any CRM snapshot dataset; the responder and orchestrator are not
started.

| Component | Status |
|---|---|
| Typed contracts (schema, profile, plan, result, errors) | Done |
| Configuration | Done |
| DuckDB connection management | Done |
| CSV and Parquet ingestion | Done |
| Column mapping onto the canonical schema | Done, deterministic half |
| `(as_of, opp_id)` uniqueness assertion | Done |
| Dataset profiling | Done |
| Tiny test fixture and unit tests | Done |
| Column classification for the 129-column production export | Done, unconfirmed against real rows |
| Dataset-scoped registry, persisted beside each dataset | Done |
| Type-aware profile of every canonical and discovered column | Done |
| Authoritative status configuration | Done, real column not yet supplied |
| Analytical concepts and dataset-specific bindings | Done |
| Agreement tests and close-date reconstruction | Done |
| Dataset understanding and tiered analyst context card | Done |
| Grain-only ingestion, monetary profiling policy | Done |
| Excel serial date support | Done, declared per column |
| Close-date reconstruction gated on required agreement tests | Done |
| Aggregate-only production probe (`scripts/probe_production.py`) | Done. Run on the sibling export; the target export is still unprobed (no AWS credentials in the standard chain) |
| Fiscal calendar, snapshot selection, cohort, trace, transitions | Done |
| Pipeline bridge with its balance invariant | Done, nine independently computed terms |
| Rate primitive, metric registry (12 metrics, incl. `cohort_fate`) | Done, metrics gated on concepts |
| Plan gate with structured rejections | Done |
| Deterministic compiler and execution | Done, refuses an unvalidated plan |
| Prospective / retrospective stance and knowledge cutoff | Done, prospective by default |
| Trust tiers A / B / C | Done, computed from inputs |
| Reconstruction verdicts (validity vs reconciliation) | Done (DESIGN_LOG §13.1) |
| Tenant usage grants, structured trust assessment | Done; grants persisted as a validated ledger (`grants.json`) |
| Tenant column classifications (generic grants) | Done, never confirm a concept binding |
| Prospective attribution guard (gate and compiler) | Done |
| Measure concepts and typed compiled comparisons | Done, exact DECIMAL arithmetic |
| Clarification materiality probe | Done, deterministic and bounded |
| Deterministic acceptance suite (question to provenance) | Done on fixtures; opt-in on a real export |
| Investigation plan, gate and compiler | Done, at most trust tier B |
| Session plan edits with carry-forward reports | Done |
| Renderer and provenance scanner | Done |
| Deterministic tool surface and planner context | Done, not wired to a model |
| Guarded SQL fallback | Designed, pending by design |
| LLM planner (typed outcomes, bounded tool loop, Anthropic adapter) | Done; real-model evaluation opt-in, not yet run |
| Planner evaluation harness (35 golden cases, failure taxonomy) | Done |
| Lake sources: CSV, Parquet, partitioned Parquet, Delta (local or `s3://`, read-only) | Done |
| Declared capture policy for intraday double captures | Done; conflicts recorded, not yet a trust factor |
| Typed fast-path ingestion with column projection | Done |
| Column-family inference and family declarations | Done; proposals never apply by themselves |
| Declared stage-to-status map and family invariants | Done; undeclared stages are excluded |
| Onboarding CLI: profile, propose, approve (`scripts/onboard.py`) | Done |
| Synthetic feature-store generator with renamed-header variants | Done |
| `AnalysisTrace` and `scripts/explain.py` | Done |
| Responder, end-to-end orchestrator, API | Not started |

## Setup

Requires Python 3.12 or newer.

```bash
python3.13 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest
```

## Data model

Opportunity snapshots are uniquely identified by `(as_of, opp_id)`. The same
opportunity appears in many snapshots, so snapshots are never treated as
independent opportunities. Ingestion asserts that uniqueness and stops if it
fails, because every snapshot metric would otherwise be silently wrong.

Source files may use any column headers. The mapping layer translates them onto
the canonical schema in `contracts/schema.py`, and ingestion fails loudly when a
required canonical field is absent.

## Usage

```python
from ai_analyst.config import Settings
from ai_analyst.data.ingest import ingest
from ai_analyst.data.profiler import profile_dataset

settings = Settings(data_root="data")
result = ingest("snapshots.csv", dataset_id="q1-fy25", settings=settings)
profile = profile_dataset(result.dataset_id, result.schema, settings=settings)

print(profile.row_count, len(profile.snapshots))
print(profile.stage_vocabulary.won_labels)
print(profile.lifecycle.vanished_without_terminal_state)
```

Ingestion writes Hive-partitioned Parquet under
`data/datasets/{dataset_id}/canonical/snapshots/as_of=YYYY-MM-DD/`, plus
`schema.json` and `profile.json`.

## Onboarding a new dataset (generalization demo)

Nothing about a new dataset is assumed. The tool proposes, a human confirms,
and only confirmed items reach the engine (ARCHITECTURE §3).

```bash
# 1. A synthetic feature store (generic names), and the same data with renamed
#    headers, as a second "tenant".
python scripts/generate_synthetic.py --out data/synthetic/base
python scripts/generate_synthetic.py --variant alpha --out data/synthetic/alpha

# 2. Profile and propose. The draft lists the proposed key, a capture policy
#    for same-day double captures, the amount and stage columns, a status for
#    every stage value, column families and the invariants that hold.
#    Nothing in it is confirmed.
python -m scripts.onboard propose data/synthetic/base/snapshots.csv \
    --out local_data/base.draft.json

# 3. Approving an unreviewed draft is refused, with every reason listed.
python -m scripts.onboard approve local_data/base.draft.json --dataset-id base

# 4. Edit the draft: set each load-bearing item to "confirmed" and each stage
#    value's status (open, won, lost, excluded). Then approve again.

# 5. Explain an answer end to end: plan, gate decisions, bindings and their
#    evidence, compiled SQL, trust factors and provenance.
python -m scripts.explain metric_opening_q2
```

A Delta table or partitioned Parquet directory, local or on S3, is declared
with `--format delta` or `--format parquet_dir`. S3 credentials come only from
the standard AWS chain (`AWS_PROFILE`, plus `AWS_REGION` for Delta).
Sources are only ever read. Drafts of real data list stage values, so keep
them under the gitignored `local_data/`.

`tests/unit/test_onboarding.py` checks the generalization claim: the renamed
variant, once declared, gives exactly the same answers as the original.

Design decisions for review are in [docs/DESIGN_DECISIONS.md](docs/DESIGN_DECISIONS.md).

## Layout

See the code map in ARCHITECTURE.md §11.

## Testing

```bash
.venv/bin/python -m pytest
```

For the fast loop, run the suite in parallel with `pytest-xdist` (part of
the `dev` extra: `pip install -e '.[dev]'`):

```bash
.venv/bin/python -m pytest -n auto
```

Use `-n0` for a single file or when debugging: worker startup makes small
runs slower, not faster. `pytest -m "not slow"` deselects tests marked
`@pytest.mark.slow` (currently the opt-in `AI_ANALYST_REAL_EXPORT` acceptance
tests, which are also skipped by default); no test in the default run takes
more than a couple of seconds, so that flag saves nothing today, and `-n auto`
is the loop worth reaching for.

The tiny fixture in `tests/fixtures/tiny/` is 40 hand-authored rows covering
slippage, pull-in, segment change, amount change, vanishing without a terminal
state, and mid-quarter creation. Its expected values are documented and derived
in `tests/fixtures/tiny/README.md`, and asserted as literals in the tests.

`tests/acceptance/` walks ten questions through the whole deterministic chain,
from plan to provenance scan, with hand-written plans and drafts standing in for
the planner and responder. Two parts are opt-in:

```bash
# The same cases on a real opportunity_snapshot_v1 export, checked against an
# independent raw-SQL reference. Ingest takes several minutes.
AI_ANALYST_REAL_EXPORT=/path/to/export.parquet \
AI_ANALYST_REAL_EXPORT_DATA_ROOT=/tmp/real_export \
  .venv/bin/python -m pytest tests/acceptance/test_real_export.py

# The target export probe. Credentials come only from the standard AWS chain.
AI_ANALYST_TARGET_EXPORT='s3://bucket/prefix/snapshot.parquet/' \
  .venv/bin/python -m pytest tests/acceptance/test_target_export.py
```

The planner is tested without a network: every model in the default suite is
scripted. The real-model evaluation is opt-in and needs the `llm` extra and
Anthropic credentials in the environment:

```bash
pip install -e '.[llm]'
python -m evals.planner.run_planner_eval --smoke   # one case: request shape
python -m evals.planner.run_planner_eval           # the 35-case golden set
```

It runs on synthetic fixtures only and writes a metadata-only report under
`evals/reports/` (git-ignored).

The deterministic semantic engine is the authoritative computational layer. LLM
work is built on top of it and is committed separately from the deterministic
baseline, so every change to how numbers are computed is reviewable on its own.
