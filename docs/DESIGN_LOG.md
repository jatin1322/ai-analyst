# AI Analyst: Design Log

**This is the full, chronological design record**, kept for reference. For a
readable overview, start with [ARCHITECTURE.md](../ARCHITECTURE.md). Section
numbers cited in code comments ("ARCHITECTURE 5.13", "§12.7") refer to this
file. Earlier sections describe proposals that later sections refine or
supersede.

---

## 0. Summary of the Central Decision

Most "chat with your data" systems are text-to-SQL: the LLM writes a SQL string,
the system runs it, the LLM summarizes the output. That design cannot be made
trustworthy, because there is no artifact between the model and the database
that can be checked. The SQL string is either right or wrong, and the only way
to find out is to run it and hope.

This system uses **plan-as-data** instead.

```
Question  →  LLM  →  AnalysisPlan (typed, validated)  →  Compiler  →  SQL  →  DuckDB  →  ResultSet
                          ↑                                                                 │
                          └──────────── rejected plans return structured errors ────────────┘
```

The LLM's job is to emit a **validated Pydantic `AnalysisPlan`**: which metric,
at which grain, over which snapshot selection rule, with which filters and
comparisons. A deterministic compiler turns that plan into DuckDB SQL. The LLM
never writes the arithmetic.

Why this is the load-bearing decision:

| Consequence | With plan-as-data | With free-form SQL |
|---|---|---|
| Validation | Schema validation before execution; plan is inspectable | Parse the string and hope |
| Snapshot semantics | Encoded once in the compiler, cannot be forgotten | Re-derived by the LLM every single query |
| Evaluation | Score plan correctness and numeric correctness independently | Only end-to-end pass/fail |
| Determinism | Same plan always produces the same SQL and the same number | Same question produces different SQL run to run |
| Auditability | Plan + compiled SQL + result hash is a complete provenance record | SQL string only |

Free-form SQL still exists, as a **gated escape hatch** (§7.4) for the long tail
of questions the semantic layer does not cover. It is not the primary path, and
its results carry a lower trust tier.

> **Sections 1 to 11 assume one canonical schema. §12 adapts the system to many
> tenants with different physical schemas** by separating the stable analytical
> ontology from each tenant's physical columns. It supersedes the parts of §5.5,
> §5.10, §6.3 and §7.4 that it names. The plan-as-data decision above, the
> temporal safety rule, and the fail-closed default are unchanged by it.

---

## 1. Requirements Analysis

### 1.1 What the data actually is

The dataset is a **slowly-changing dimension history of opportunities**, captured
as periodic full snapshots. Grain:

```
PRIMARY KEY (as_of, opp_id)
```

This has three consequences that dominate the design.

**Consequence A: `COUNT(*)` is almost always wrong.** A naive count over the raw
table counts opportunity-snapshots, not opportunities. Every metric must declare
its snapshot-selection rule explicitly. There is no sensible default that works
across questions.

**Consequence B: most interesting questions are temporal diffs, not aggregates.**
"Slipped", "created after quarter start", "pushed", "downgraded" are all
statements about a field changing value between two snapshots of the same
`opp_id`. They are self-joins on the snapshot history, not `GROUP BY`s.

**Consequence C: dimensions are mutable.** An opportunity's segment, owner, amount,
stage, and close date can all differ across snapshots. "Win rate by segment"
is ambiguous until you say *which snapshot's segment*. See §5.3.

### 1.2 Question taxonomy

The ten example questions are not ten unrelated features. They fall into six
patterns, and five of the six are parameterizations of a single object: the
**pipeline bridge** (§5.2).

| # | Example question | Pattern | Primitive |
|---|---|---|---|
| 1 | Pipeline at the beginning of each quarter | Point-in-time snapshot | `snapshot_at` |
| 2 | Pipeline created after quarter started | Bridge component | `bridge.created` |
| 3 | Which opportunities slipped from Q2 to Q3 | Temporal diff | `transition` |
| 4 | % of opening pipeline that eventually closed | Cohort trace | `cohort` + `trace` |
| 5 | Segments with highest win rate | Rate by dimension | `rate` |
| 6 | Win rate over 8 quarters | Rate time series | `rate` + grain |
| 7 | Which opportunities are most likely to slip | Predictive scoring | `slip_risk` (§6.5) |
| 8 | What caused coverage to decline | Decomposition + narration | `bridge` + narrate |
| 9 | Top 20 deals by ARR created this quarter | Ranked detail list | `cohort` + rank |
| 10 | Forecast accuracy across 8 quarters | Cohort trace vs. actual | `cohort` + `trace` |

Two of these are **not SQL questions** and must be handled honestly:

- **Q7 (slip likelihood)** is a prediction. It is implemented as a deterministic,
  fully explainable feature score (§6.5), not as an opaque model and never as an
  LLM guess. Every contributing factor is surfaced with its weight.
- **Q8 (causation)** is a causal claim the data cannot support. The system returns
  the bridge decomposition showing which components moved and by how much, and
  the LLM narrates *contributions*, explicitly not causes. Stating this limit is
  a feature, not a shortcoming.

### 1.3 Non-negotiable constraints

From the project engineering rules:

1. The LLM must never fabricate analytical results. Every number traces to an
   executed computation.
2. Snapshots are never treated as independent opportunities.
3. Reasoning, planning, tool execution, validation, and response generation are
   separate components.
4. Quantitative questions are answered by computation, never by LLM reasoning.
5. Stack is Python, DuckDB, Parquet, Pydantic, FastAPI, pytest. No additional
   infrastructure without written justification.

### 1.4 Explicit assumptions

**No dataset has been provided yet.** This document therefore defines a
*canonical schema contract* and a mapping step from the user's real column
headers onto it (§4.2). The assumed canonical fields are:

| Canonical field | Type | Required | Notes |
|---|---|---|---|
| `as_of` | DATE | yes | Snapshot date |
| `opp_id` | VARCHAR | yes | Stable opportunity identifier |
| `close_date` | DATE | yes | Expected or actual close |
| `stage` | VARCHAR | yes | Sales stage label |
| `amount` | DECIMAL(18,2) | yes | Deal value in reporting currency |
| `created_date` | DATE | no | Falls back to first-seen snapshot |
| `is_closed` | BOOLEAN | no | Derived from `stage` if absent |
| `is_won` | BOOLEAN | no | Derived from `stage` if absent |
| `forecast_category` | VARCHAR | no | Commit / Best Case / Pipeline / Omitted |
| `arr` | DECIMAL(18,2) | no | Falls back to `amount` |
| `segment`, `region`, `industry` | VARCHAR | no | Dimensions |
| `owner_id`, `account_id`, `opp_name` | VARCHAR | no | Dimensions and labels |
| `probability` | DECIMAL | no | Used only if present |

If a required field is missing after mapping, ingestion **fails loudly**. The
system never guesses a primary key.

---

## 2. Architecture Overview

```
┌────────────────────────────────────────────────────────────────────────────┐
│                              FastAPI Surface                               │
│   POST /datasets   POST /chat   GET /sessions/{id}   GET /runs/{id}        │
└──────────────────────────────────┬─────────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────────┐
│                            Orchestrator (agent loop)                       │
│         owns the turn, the tool loop, budgets, and the run record           │
└───┬─────────────┬──────────────┬──────────────┬──────────────┬─────────────┘
    │             │              │              │              │
┌───▼────┐  ┌─────▼─────┐  ┌─────▼──────┐  ┌────▼─────┐  ┌─────▼────────┐
│ Context│  │  Planner  │  │  Execution │  │Validator │  │  Responder   │
│Builder │  │  (LLM →   │  │   Engine   │  │          │  │ (LLM → text  │
│        │  │   Plan)   │  │            │  │          │  │  w/ refs)    │
└───┬────┘  └─────┬─────┘  └─────┬──────┘  └────┬─────┘  └─────┬────────┘
    │             │              │              │              │
    │        ┌────▼──────────────▼──────┐       │        ┌─────▼────────┐
    │        │     Semantic Layer       │       │        │   Renderer   │
    │        │  metrics · bridge ·      │       │        │ (substitutes │
    │        │  snapshot rules · compiler│      │        │  real values)│
    │        └────────────┬─────────────┘       │        └─────┬────────┘
    │                     │                     │              │
┌───▼─────────────────────▼─────────────────────▼──────────────▼───────────┐
│                          Data Layer (DuckDB + Parquet)                     │
│   raw snapshots · conformed view · profile stats · result registry         │
└────────────────────────────────────────────────────────────────────────────┘
```

Data flows one way. The LLM appears in exactly two places: **Planner** (question
to typed plan) and **Responder** (result set to narrative with reference tokens).
It touches numbers in neither.

### 2.1 Component responsibilities

| Component | Owns | Must never |
|---|---|---|
| **API surface** | HTTP contracts, upload handling, SSE streaming, session routing | Contain analytical logic |
| **Orchestrator** | The turn loop, tool dispatch, repair budget, run record assembly | Compute or interpret numbers |
| **Context Builder** | Schema card, profile card, schema-filtered metric catalog, session state injection | Include raw data rows |
| **Planner** | Question to `AnalysisPlan` via structured output; ambiguity detection | Emit SQL or arithmetic |
| **Plan Gate** | Semantic validation beyond types; structured rejections | Silently repair a bad plan |
| **Semantic Layer** | Snapshot rules, metric definitions, bridge, primitives, fiscal calendar | Call an LLM |
| **Compiler** | `AnalysisPlan` compiled to DuckDB SQL, deterministically | Accept unvalidated input |
| **Execution Engine** | Read-only execution, caps, result registration, `query_id` assignment | Execute unparsed SQL |
| **Validator** | Sanity invariants, provenance scan, trust tiering | Pass a failing result through |
| **Responder** | Narrative with reference tokens, caveat surfacing | Write a literal number |
| **Renderer** | Token substitution from the result registry | Invent a value for a missing token |
| **Data Layer** | Parquet layout, conformance, profiling, grain assertion | Infer a primary key |
| **Session State** | Resolved entities, carried filters, plan mutation for follow-ups | Store raw transcripts as the source of truth |

The two rules that make the rest hold: the Semantic Layer and everything below
it contain **no LLM calls**, and the Planner and Responder contain **no
arithmetic**. Every bug is therefore localized to one side of that line.

---

## 3. Technology Stack

### 3.1 Core (MVP)

| Layer | Choice | Justification |
|---|---|---|
| Language | Python 3.12+ | Mandated; `Literal`, `match`, typing maturity |
| Analytics engine | DuckDB | Mandated; embedded OLAP, no server, excellent window functions, reads Parquet natively, `read_only` connections |
| Storage | Parquet | Mandated; columnar, compressed, partitionable by `as_of` |
| Contracts | Pydantic v2 | Mandated; the `AnalysisPlan` gate depends on strict validation |
| API | FastAPI | Mandated; native Pydantic integration, SSE for streaming |
| Tests | pytest | Mandated; `pytest-asyncio`, `pytest-benchmark` for query budgets |
| LLM | Anthropic Claude, `claude-opus-5` | Strong structured-output and tool-use support |
| SQL safety | `sqlglot` | Parse-and-inspect DuckDB dialect ASTs before execution; the escape hatch is unsafe without it |
| Charts | Vega-Lite specs (JSON) | Declarative, serializable, renderable in any frontend; no server-side image pipeline |

### 3.2 Justified additions

Only two dependencies beyond the mandated set, both narrow:

- **`sqlglot`**: required to make the free-form SQL escape hatch safe. An
  allowlist over a parsed AST is sound; an allowlist over a regex is not.
- **`structlog`**: structured JSON run records. Could be hand-rolled, but
  observability is a first-class requirement here (§9.5).

### 3.3 Explicitly rejected for MVP

| Rejected | Why |
|---|---|
| Vector database | Metric selection over dozens of metrics is a prompt-with-catalog problem, not a retrieval problem. A catalog fits in context. |
| Redis / Celery | Single-process FastAPI with an in-process DuckDB handles this workload. Session state lives in SQLite or DuckDB. |
| LangChain / LlamaIndex / agent frameworks | The agent loop here is roughly 200 lines and needs precise control over validation gates. A framework would obscure exactly the part that matters. |
| dbt | The semantic layer is Python-native and dynamic. dbt models are static. |
| Pandas as the compute engine | DuckDB is faster and the rules mandate SQL-first. Pandas appears only in the narrow post-processing sandbox (§6.4). |

---

## 4. Repository Structure

```
ai-analyst/
├── ARCHITECTURE.md
├── README.md
├── CLAUDE.md
├── pyproject.toml
├── Makefile
│
├── src/ai_analyst/
│   ├── config.py                    # Settings (pydantic-settings)
│   │
│   ├── contracts/                   # ── Every typed boundary in the system ──
│   │   ├── schema.py                #    CanonicalColumn, DatasetSchema, ColumnMapping
│   │   ├── profile.py               #    DatasetProfile, ColumnProfile, GrainCheck
│   │   ├── plan.py                  #    AnalysisPlan and every sub-model  ★ core
│   │   ├── result.py                #    ResultSet, ResultCell, QueryId, TrustTier
│   │   ├── answer.py                #    DraftAnswer, RenderedAnswer, Citation
│   │   └── errors.py                #    PlanRejection, ExecutionError, ValidationFailure
│   │
│   ├── data/                        # ── Storage and access ──
│   │   ├── ingest.py                #    CSV/Parquet → canonical Parquet
│   │   ├── mapping.py               #    LLM-assisted column mapping + human confirm
│   │   ├── conform.py               #    Type coercion, derived flags, fiscal calendar
│   │   ├── profiler.py              #    Grain check, cardinality, nulls, ranges
│   │   ├── store.py                 #    DuckDB connection management, read-only pool
│   │   └── registry.py              #    ResultSet persistence, query_id → result
│   │
│   ├── semantic/                    # ── The domain core  ★ ──
│   │   ├── calendar.py              #    Fiscal quarters, period arithmetic
│   │   ├── snapshots.py             #    Snapshot selection rules (§5.1)
│   │   ├── metrics.py               #    Metric registry and definitions
│   │   ├── bridge.py                #    Pipeline bridge decomposition (§5.2)
│   │   ├── transitions.py           #    Slip, push, pull-in, stage movement
│   │   ├── cohort.py                #    Cohort fixing and terminal-state tracing
│   │   └── compiler.py              #    AnalysisPlan → DuckDB SQL  ★ core
│   │
│   ├── agent/                       # ── LLM-facing ──
│   │   ├── orchestrator.py          #    Turn loop, budgets, run record
│   │   ├── context.py               #    Schema card + profile card + metric catalog
│   │   ├── planner.py               #    Question → AnalysisPlan (structured output)
│   │   ├── responder.py             #    ResultSet → narrative with {{refs}}
│   │   ├── clarify.py               #    Ambiguity detection and question-asking
│   │   ├── prompts/                 #    Versioned prompt templates
│   │   └── tools/                   #    Tool definitions and handlers (§6)
│   │       ├── introspection.py     #      get_schema, profile_column, list_metrics
│   │       ├── analysis.py          #      run_plan  ★ primary
│   │       ├── escape.py            #      run_sql (gated), run_python (sandboxed)
│   │       ├── scoring.py           #      compute_slip_risk
│   │       └── charting.py          #      make_chart
│   │
│   ├── validation/                  # ── The trust boundary  ★ ──
│   │   ├── plan_gate.py             #    Semantic validation beyond Pydantic
│   │   ├── sql_guard.py             #    sqlglot AST allowlist
│   │   ├── sanity.py                #    Post-execution invariant checks (§8.3)
│   │   ├── provenance.py            #    Numeral traceability scan (§8.4)
│   │   └── rendering.py             #    Reference-token substitution
│   │
│   ├── session/
│   │   ├── state.py                 #    Conversation + resolved entities
│   │   └── resolver.py              #    "those deals", "that quarter" → concrete refs
│   │
│   └── api/
│       ├── app.py
│       ├── routes/                  #    datasets, chat, sessions, runs
│       └── sse.py                   #    Streaming progress events
│
├── tests/
│   ├── fixtures/
│   │   └── tiny/                    #    ~40 rows, hand-verifiable  ★ (§9.1)
│   ├── unit/                        #    Per-module
│   ├── semantic/                    #    Metric correctness on the tiny fixture
│   ├── golden/
│   │   ├── questions.yaml           #    Question → expected plan + expected numbers
│   │   └── reference/               #    Independent reference implementation  ★ (§9.2)
│   ├── adversarial/
│   │   └── unanswerable.yaml        #    Abstention probes (§9.3)
│   └── e2e/
│
├── evals/
│   ├── run_eval.py
│   └── reports/
│
└── scripts/
    ├── generate_synthetic.py        #    Synthetic snapshot data for dev
    └── explain_plan.py              #    Plan → SQL, no execution (debug aid)
```

Items marked ★ are where the system's value concentrates. A reviewer should read
`contracts/plan.py`, `semantic/compiler.py`, and `validation/provenance.py` first.

---

## 5. The Semantic Layer (Domain Core)

This section, not the agent loop, is where the system earns correctness. Every
ambiguity below has two defensible readings. Silently picking one is how wrong
numbers ship. Each is named, given a **default**, and exposed as a plan option.

### 5.1 Snapshot selection rules

Snapshot dates will not land on quarter boundaries. Define selection explicitly.

| Rule | Meaning | Default resolution |
|---|---|---|
| `AS_OF_EXACT` | The snapshot on a given date | Error if no snapshot exists that day |
| `PERIOD_OPEN` | "Beginning of quarter" | **Nearest snapshot on-or-before the quarter start date** |
| `PERIOD_CLOSE` | "End of quarter" | **Nearest snapshot on-or-before the quarter end date** |
| `LATEST` | Most recent snapshot overall | `MAX(as_of)` |
| `LATEST_IN_PERIOD` | Last snapshot within a period | `MAX(as_of)` where `as_of` in period |
| `ALL` | Every snapshot (time series) | Used for trend charts |

**Every `ResultSet` reports the actual `as_of` dates resolved**, never just the
requested rule. If the user asks for Q3 opening pipeline and the nearest snapshot
is six days before quarter start, the answer says so. A configurable
`max_snapshot_drift_days` (default: 10) triggers a warning above threshold.

### 5.2 The pipeline bridge

Five of the six question patterns are slices of one identity. Define it once.

```
opening_pipeline(P)                     -- open pipe at PERIOD_OPEN(P), close_date in P
  + created_in_period                   -- new opps landing in P
  + pulled_in                           -- close_date moved INTO P from a later period
  + amount_increased                    -- net expansion on surviving opps
  - amount_decreased                    -- net contraction on surviving opps
  - closed_won                          -- terminal: won
  - closed_lost                         -- terminal: lost
  - slipped_out                         -- close_date moved OUT of P to a later period
  - other_removed                       -- disqualified, deleted, or vanished from snapshots
  = ending_pipeline(P)                  -- open pipe at PERIOD_CLOSE(P), close_date in P
```

The compiler emits this decomposition as a single result set. The bridge must
**balance to zero within a tolerance of 0.01** or the sanity checker fails the
result (§8.3). This is the single most valuable invariant in the system: it
catches double-counting, missed transitions, and snapshot-selection bugs
automatically, on every query that touches it.

**Every term is computed independently, including `other_removed`.** This is
essential and easy to get wrong. If `other_removed` were derived as the residual
needed to make the identity close, the bridge would balance by construction and
the invariant would verify nothing. Instead `other_removed` is computed directly
as the set of opportunities present in the opening snapshot with a close date in
P, absent or no longer open in the closing snapshot, and carrying no terminal
won or lost state and no close-date move that would classify them elsewhere. The
identity is then a genuine check on nine separately computed quantities, and a
failure means a real bug. Section 7.3 surfaces the size of this term at profiling
time, because a large `other_removed` makes every derived rate suspect.

Question 2 is `bridge.created_in_period`. Question 8 is the whole bridge with
period-over-period deltas on each component. Question 4 is the opening-pipeline
cohort traced to terminal state.

### 5.3 The six ambiguities, with defaults

| # | Ambiguity | Readings | **Default** | Plan option |
|---|---|---|---|---|
| 1 | **Created in period** | (a) first appearance in any snapshot; (b) `created_date` falls in period | **(b) `created_date`**, falling back to (a) when `created_date` is absent | `creation_basis: "created_date" \| "first_seen"` |
| 2 | **Slippage** | Measured between which two snapshots, and does a close-date move within the same quarter count? | **`close_date` moves from period P to a strictly later period, measured `LATEST_IN_PERIOD(P)` vs. `LATEST_IN_PERIOD(P+1)`**; intra-period moves are "pushes", not slips | `slip_basis`, `from_snapshot`, `to_snapshot` |
| 3 | **Win rate denominator** | won / (won + lost), or won / all-opps-including-still-open | **won / (won + lost)**, keyed by the period in which the opp reached terminal state | `win_rate_basis: "closed_only" \| "all_cohort"` |
| 4 | **Win rate keying** | By `close_date` period, or by the `as_of` period in which the close was observed | **`close_date` period** | `rate_key: "close_date" \| "observed_at"` |
| 5 | **Dimension attribution** | Segment/owner can change across snapshots | **Attribute as-of the opening snapshot of the period under analysis** | `attribution: "period_open" \| "latest" \| "at_close"` |
| 6 | **Amount field** | `amount` vs. `arr` vs. weighted-by-probability | **`arr` when present, else `amount`; never probability-weighted unless asked** | `measure: "arr" \| "amount" \| "weighted"` |

Ambiguity 5 deserves emphasis. "Which segments have the highest win rate" gives a
different answer depending on whether you use the segment recorded at quarter
open or the segment recorded at close. Opportunities get re-segmented. The
default of period-open attribution is chosen because it makes cohorts stable:
the denominator population is fixed at the same moment the cohort is fixed.

### 5.4 Named primitives

Six composable operations. Every metric in the registry is built from these.

```python
snapshot_at(rule, period)          -> a single as_of date, plus drift metadata
cohort(snapshot, filters)          -> a frozen set of opp_ids, immutable thereafter
trace(cohort, through_snapshot)    -> terminal state per opp_id (won/lost/open/vanished)
transition(snap_a, snap_b, field)  -> per-opp_id before/after values where changed
bridge(period, filters)            -> the full decomposition of §5.2
rate(numerator, denominator, by)   -> a ratio with both components exposed
```

`cohort` is the primitive that makes question 4 and question 10 correct.
"What percentage of opening pipeline eventually closed" fixes an `opp_id` set at
the opening snapshot and then follows those specific opportunities forward. It
does **not** compare two independent aggregates. Forecast accuracy (question 10)
is the same shape: fix the commit cohort at quarter open, trace to actual.

### 5.5 Metric registry

```python
class MetricDefinition(BaseModel):
    name: str                              # "open_pipeline"
    display_name: str
    description: str                       # shown to the LLM in the catalog
    measure: Literal["sum", "count", "count_distinct", "ratio", "avg"]
    column: str | None
    default_snapshot_rule: SnapshotRule
    required_columns: list[str]            # gate: hide metric if schema lacks these
    ambiguity_notes: list[str]             # surfaced in answers when relevant
    compiler: Callable[[MetricContext], exp.Expression]
```

The registry is **schema-aware**. If the uploaded dataset has no
`forecast_category` column, forecast-accuracy metrics are removed from the
catalog the LLM sees, so it cannot plan a query that cannot be answered. This
converts a class of hallucination into a class of abstention.

> **Superseded in part by §12.** `required_columns` becomes required *concepts*
> (§12.1), resolved per tenant through a concept binding (§12.2). That is what
> lets one metric definition serve a tenant whose amount column is `ARR` and one
> whose amount column is `new_amount`. The gating behavior described here is
> unchanged; only what it gates on becomes a concept rather than a column name.
> Metrics are also no longer the only analytical path: see §12.6.

### 5.6 Fiscal calendar

Configurable fiscal year start month, default January. Quarter labels resolve
through `calendar.py`, never through string parsing in the compiler. "This
quarter" resolves against the dataset's `MAX(as_of)`, not against wall-clock
today, and the answer states which quarter that resolved to.

---

> **Sections 5.7 to 5.11 are framework; 5.12 to 5.15 are this dataset.** They define how real
> dataset columns will be classified and governed. They deliberately contain no
> column assignments, because the production column list has not yet been
> supplied when they were written. Sections 5.7 to 5.11 describe how columns are
> governed; §5.12 applies that framework to the production export. The canonical
> schema in §1.4 is unchanged and remains the cross-dataset contract, not a
> claim about any one export's columns.

### 5.7 Column taxonomy

A production snapshot export carries far more than the canonical minimum:
precomputed temporal features, rep and account aggregates, CRM free text, and
dataset-specific amount columns. Every discovered column is assigned exactly one
**category** and exactly one **availability class** (§5.8). The category says
what the column is; the availability class says when it may be read.

| Category | What makes a column belong | Typical availability | Usable as |
|---|---|---|---|
| `identity` | Identifies the opportunity, account, or snapshot. Part of or adjacent to the grain. | As-of fact | Keys, never measures |
| `snapshot_state` | The opportunity's recorded state in this snapshot: stage, amount, close date, forecast category, owner. Mutable across snapshots. | As-of fact | Measures, dimensions, filters |
| `outcome` | What eventually happened: final won or lost flags, realized close date, actual booking. | Future-contaminated unless the value is the state recorded in this snapshot | Retrospective analysis only |
| `temporal` | Raw dates: created, close, last activity, stage entry. | As-of fact | Period logic, filters |
| `quarter` | Period labels attached to a row: close quarter, created quarter, snapshot quarter. | As-of fact, but see the warning below | Grouping, after verification |
| `derived_temporal` | Counts and flags computed from a row's own date history: push counts, stage change counts, days in stage, slip flags. | Backward-derived **if** the window ends at this `as_of`; otherwise contaminated | Recompute by default (§5.9) |
| `historical_feature` | Aggregates over prior outcomes, often segment or stage scoped: historical win rate at stage, median cycle length. | Unknown until the window is documented; treat as contaminated | Quarantine until proven |
| `rep_feature` | Aggregates scoped to the owning rep: rep win rate, rep quota attainment, rep tenure. | Unknown until the window is documented; treat as contaminated | Quarantine until proven |
| `account_feature` | Aggregates scoped to the account: prior bookings, account age, existing ARR, logo tier. | Mixed. Static attributes are as-of facts, aggregates are unknown | Case by case |
| `deal_feature` | Per-opportunity descriptive attributes that are not state: product, channel, source, competitor. | As-of fact | Dimensions, filters |
| `text` | Free-form CRM narrative: manager notes, next step, qualification narratives. | As-of fact if written in-period, contaminated if backfilled | Catalogued only (§5.11) |
| `metadata` | Pipeline and export bookkeeping: load timestamps, source system, row hashes. | As-of fact | Excluded from analysis |

**The `quarter` warning.** A stamped quarter label is only an as-of fact if it
was computed from the values in that same snapshot. If the export stamps a
"close quarter" derived from the opportunity's *final* close date, the column is
future-contaminated and looks entirely innocent. Every `quarter` column is
reconciled against the fiscal calendar (§5.6) applied to the row's own date, and
a mismatch reclassifies it.

### 5.8 Availability classes and temporal safety

This is the section that prevents the most damaging class of wrong answer.
"What was the pipeline at the beginning of Q3" and "which deals were likely to
slip, as judged at Q3 open" must not read a column whose value depends on what
happened after Q3 opened. Leakage of that kind produces confident, plausible,
unfalsifiable numbers.

Every column carries one of four classes:

| Class | Meaning | Prospective analysis | Retrospective analysis |
|---|---|---|---|
| `AS_OF_FACT` | The value is a property of the row as recorded at this `as_of`. | Allowed | Allowed |
| `BACKWARD_DERIVED` | Computed only from snapshots at or before this `as_of`, with the window documented. | Allowed | Allowed |
| `FUTURE_CONTAMINATED` | The value depends on observations after this `as_of`. | **Rejected** | Allowed, and disclosed |
| `UNKNOWN` | Provenance not established. | **Rejected** | Allowed, and disclosed |

**`UNKNOWN` is treated exactly like `FUTURE_CONTAMINATED`.** The system fails
closed. A precomputed feature whose derivation window nobody has documented is
assumed to leak until someone proves otherwise. This default is the entire
safety property: it makes the silent-leak case loud.

**Enforcement is structural, not advisory.** The mechanism is the one already
used for schema-aware metrics in §5.5: the model cannot plan what it cannot see.

1. Each `AnalysisSpec` carries an **analysis stance**, proposed in Appendix C.
   `prospective` means "as judged at the snapshot", and is the default.
   `retrospective` means "using everything we now know".
2. Under a prospective stance, the column resolver exposes only `AS_OF_FACT` and
   `BACKWARD_DERIVED` columns. Contaminated and unknown columns are absent from
   the catalog, absent from the dimension list, and absent from the filter
   vocabulary.
3. The plan gate rejects any spec that names a column its stance does not
   permit, with a structured rejection naming the column, its class, and the
   reason the class was assigned.
4. The compiler refuses to emit SQL referencing a column outside the permitted
   set, as a second gate behind the first.
5. Any retrospective result is disclosed as such in the answer, so a reader
   never mistakes hindsight for foresight.

A further rule follows from the snapshot grain. Under a prospective stance at
snapshot `T`, the compiler must not read rows with `as_of > T` for any purpose,
including joins used only to attribute a dimension. Dimension attribution
(§5.3 ambiguity 5) is therefore constrained: `attribution: latest` is a
retrospective operation and is rejected under a prospective stance.

### 5.9 Precomputed features are evidence, not truth

The project rule that the LLM must never fabricate results has a quieter
counterpart: the *pipeline* must not launder an unverified number into an
answer. A precomputed feature column is someone else's computation, with
someone else's window and someone else's edge cases.

Every non-canonical feature column receives one of three dispositions:

| Disposition | Meaning |
|---|---|
| `RECOMPUTE` | The semantic layer computes the value from raw snapshot history. The precomputed column is used only for reconciliation, never in an answer. |
| `USE_WITH_PROOF` | The column may be read directly, but only after a reconciliation test passes on this dataset and its window is documented. |
| `QUARANTINE` | Not exposed to metrics at all until provenance is established. |

**Reconciliation harness.** For every column marked `RECOMPUTE`, the system
computes its own value across all snapshots and compares. The output is a match
rate plus a sample of disagreements. A mismatch is a finding, surfaced at
profiling time. **The recomputation always wins.** The precomputed column never
overrides it.

Applying this to the six columns named in the milestone request. These are the
only columns that have been stated, so they are the only ones classified here.

| Column | Category | Recomputable from snapshots | Disposition | Why |
|---|---|---|---|---|
| `is_pushed_out_deal` | `derived_temporal` | Yes, from `close_date` history | `RECOMPUTE` | A push is a close-date move to a later date between two snapshots. Fully determined by data the system already has. The precomputed flag's window and its treatment of intra-period moves are unstated, and §5.3 ambiguity 2 shows those choices change the answer. |
| `is_pulled_in_deal` | `derived_temporal` | Yes, from `close_date` history | `RECOMPUTE` | Same reasoning, inverted. Also needed independently as a bridge component. |
| `close_date_push_count` | `derived_temporal` | Yes | `RECOMPUTE` | **Ambiguous in a dangerous way.** If the count is cumulative to this `as_of` it is `BACKWARD_DERIVED` and safe. If it is the opportunity's lifetime total stamped onto every snapshot, it is `FUTURE_CONTAMINATED`, because an early snapshot would carry knowledge of pushes that had not happened yet. The column name cannot distinguish these. Requires confirmation. |
| `stage_changes_count` | `derived_temporal` | Yes | `RECOMPUTE` | Identical ambiguity and identical leak risk. |
| `rep_win_rate` | `rep_feature` | Partially | `QUARANTINE` | **The dangerous kind.** Window unstated. If computed over the whole dataset it embeds outcomes that postdate the snapshot, and a slip-risk or win-rate analysis built on it is circular: the feature already contains the answer. Even a correctly windowed version needs its lookback documented before use. |
| `historical_win_rate_at_stage` | `historical_feature` | Partially | `QUARANTINE` | Same. Additionally, if the historical population includes the opportunity being scored, the feature is self-referential. |

The first four are recomputable and will be recomputed. The last two are the
ones that would quietly invalidate every predictive and rate-based answer, and
they stay quarantined until their windowing is documented.

### 5.10 Optional and dynamic columns

The canonical schema in §1.4 is the required minimum, not an enumeration of what
a dataset may contain. Datasets carry dataset-specific fields, including
segment-scoped amount columns, and the system must not hardcode them.

- **Core versus discovered.** Canonical columns are fixed and required.
  Everything else is a *discovered column* carrying a classification record:
  category, availability class, disposition, and how each was assigned.
- **Column families.** Related columns are registered as a family by pattern
  rather than enumerated one by one. A family declares a shared category,
  availability class, and role, so a dataset with several segment-scoped amount
  columns needs no code change. Family membership is proposed by pattern and
  confirmed by a human, never silently assumed.
- **Roles, not names.** Metrics declare the *role* they need, such as a measure
  role or a segment role, and the registry binds roles to concrete columns for
  a given dataset. This is what lets one metric definition serve datasets whose
  amount column is named differently. **§12.1 and §12.2 develop this into the
  concept and binding model**, which adds the two things a role alone lacks:
  typed evidence for why a binding was made, and a status that distinguishes a
  confirmed binding from an inferred one.
- **Persistence.** The classification record lives in `schema.json` beside the
  mapping, is human-reviewable, and is versioned. A reclassification is a schema
  change with a visible diff, not a silent behavior change.
- **Unclassified is unusable.** A discovered column with no classification is
  `UNKNOWN`, and by §5.8 is therefore excluded from prospective analysis. Adding
  a column to an export cannot silently change an answer.

### 5.11 Text columns: catalogue only

CRM narrative fields are catalogued in this design and used in none of it. No
embeddings, no retrieval, no semantic search. That is a later milestone, and
this section exists so the later milestone inherits a classification rather than
a surprise.

- **Detection.** A `VARCHAR` column is a text candidate rather than a
  categorical dimension when its distinct ratio, mean length, and token count
  exceed thresholds. The thresholds are configuration; the decision is recorded
  and confirmable, because getting it wrong turns a useful dimension into an
  unusable blob or vice versa.
- **Catalogue contents.** Name, category, null rate, mean and maximum length,
  distinct ratio, and whether the column appears to be append-only across
  snapshots. No content is summarized or embedded.
- **Exclusions.** Text columns are never dimensions, never measures, and admit
  only null and non-null filters. A group-by on free text produces a meaningless
  high-cardinality result and is rejected by the plan gate.
- **Availability.** Narrative written during a period is an `AS_OF_FACT`. A
  field backfilled after the fact, such as a post-mortem loss reason attached
  retroactively to every snapshot, is `FUTURE_CONTAMINATED` and would leak the
  outcome. Append-only detection across snapshots is the evidence for this, and
  the result requires confirmation.

### 5.12 Findings from the production export

The production opportunity-snapshot export has 129 columns. `contracts/
opportunity_snapshot_v1.py` holds the full proposed classification. Every
assignment was made from column names alone, with no row of the data inspected,
so all but three carry `requires_confirmation`. Three findings change the plan.

#### Finding 1: there is no per-snapshot close date

> **Resolved in §5.14.** `days_to_close` is confirmed to be `close_date - as_of`
> and not terminal-derived, so the reconstruction below is sound and the bridge,
> transition and slip primitives are unblocked. The finding is kept because the
> reconstruction and its cross-checks remain part of the design.

`close_date` is required by the canonical schema (§1.4) and is not in the
export. Milestone one's ingestion therefore hard-fails on this file, which is
the grain assertion working as designed rather than a bug.

The concept exists in the data. `close_date_push_count`,
`close_date_net_movement`, `cd_movement`, `CD_in_qtr` and `eoq_close_diff` are
all meaningless unless a per-snapshot close date varies. It was simply not
exported as a raw column.

Reconstruction is `as_of + days_to_close`, and three independent cross-checks
exist inside the same export:

| Check | Assertion |
|---|---|
| Quarter agreement | the reconstructed date's fiscal quarter equals `close_date_qtr` |
| In-quarter flag | `CD_in_qtr` agrees with whether the reconstructed date falls in `as_of_qtr` |
| Quarter-end offset | `eoq_close_diff` equals the reconstructed date minus the quarter end |

A reconstruction satisfying all three is trustworthy, and it becomes a
conform-stage derivation with recorded provenance, exactly like the existing
stage-derived flags.

**The premise must be verified first.** If `days_to_close` is measured against
`terminal_date` rather than the snapshot's own close date, the reconstruction
yields the final close date at every snapshot. That is both total leakage and
silently fatal to slip analysis, because no opportunity would ever appear to
move and every slip metric would return zero while looking healthy.

The discriminating test needs no new data. Take an opportunity with
`close_date_push_count > 0` and check whether `as_of + days_to_close` varies
across its snapshots. Constant means terminal-derived and the column is
contaminated. Varying means per-snapshot and the reconstruction is sound.

**This single test gates whether the bridge, transition and slip primitives can
be written at all.**

#### Finding 2: the answerable question set is narrower than §1.2 assumed

There is no customer segment, region, or industry column. `qtr_segment` is most
likely quarter phase rather than customer segment, given how many quarter-phase
features this export carries. The available dimensions are `OwnerID`,
`account_id`, `Stage`, `ForecastCategory`, `Type`, `pipe_type`,
`RenewalManager`, and `deal_size_bucket_num` once its bucketing is confirmed.

Against the ten example questions in §1.2:

| Question | Status |
|---|---|
| Pipeline at the beginning of each quarter | Answerable, after the close-date reconstruction |
| Pipeline created after the quarter started | Answerable; `created_date` reconstructs from `age`, and the first-seen basis works regardless |
| Which opportunities slipped Q2 to Q3 | Answerable, after the close-date reconstruction |
| Percentage of opening pipeline that closed | Answerable, retrospective |
| Which segments have the highest win rate | **Not answerable.** No segment column. Rep, type, and deal-size-bucket are the available substitutes |
| Win rate over the last 8 quarters | Answerable, once the stage vocabulary is confirmed |
| Which opportunities are most likely to slip | Later milestone, and every candidate feature is quarantined |
| What caused pipeline coverage to decline | Bridge answerable; true coverage needs a quota or target, which is absent |
| Top 20 deals by ARR created this quarter | Answerable in `new_amount`; there is no ARR column and the answer says so |
| Forecast accuracy across 8 quarters | Answerable, retrospective, using `ForecastCategory` against outcome |

#### Finding 3: no free-text column was exported

The 22 `*_updated_days` columns are recency derivatives of CRM narrative fields.
The narrative fields themselves, such as manager notes and next steps, are not
in the export. §5.11's catalogue therefore has nothing to catalogue for this
dataset, and a future text milestone needs those raw fields from elsewhere.

The recency columns are not a consolation prize. They are backward-derived,
prospectively safe, and a genuine staleness signal: how long since anyone
touched the close date or the forecast category is real evidence about a deal.

#### Classification summary

| Axis | Count |
|---|---|
| Columns classified | 129 |
| Prospectively usable | 49 |
| Quarantined | 48 |
| Marked for recomputation | 43 |
| Awaiting confirmation | 126 |

By category: `derived_temporal` 58, `rep_feature` 27, `deal_feature` 11,
`account_feature` 7, `quarter` 7, `identity` 5, `outcome` 5, `snapshot_state` 4,
`historical_feature` 2, `temporal` 2, `metadata` 1, `text` 0.

Roughly two of every five columns are quarantined. That is the expected shape
for an export engineered for model training, and the `train` column confirms
that is what this is. Training feature pipelines routinely include
target-derived features and are careless about temporal windows, so the high
quarantine rate is the fail-closed rule doing its job, not excessive caution.

#### The leakage cluster

Beyond the five `terminal_*` outcome columns, these are the columns most likely
to leak, and each would produce a confident, plausible, unfalsifiable number:

- All 23 `rep_*` aggregates. Any rate or cycle-time variant requires closed
  outcomes, so an unwindowed version embeds the future. A slip or win-rate
  analysis built on `rep_win_rate` is circular: the feature contains the answer.
- Everything ending `_percentile_overall` or `_rank_pct`, where "overall" most
  likely spans all eight quarters.
- `historical_win_rate_at_stage` and `quarter_close_rate_hist`, both with
  unstated windows.
- `opp_commit_to_close_days`, where "close" most likely means terminal close,
  making it a cycle-time label rather than a snapshot feature.
- `close_date_push_count` and `stage_changes_count`, if they are lifetime totals
  stamped on every snapshot rather than counts cumulative to each snapshot.

One column in that neighbourhood is clean and useful.
`deal_first_seen_close_date` is by construction the earliest observation, which
makes it a sound baseline for slip computation.

#### Required confirmations

> **All seven were answered; see §5.14.** Kept as the record of what had to be
> asked before any of this could be built.

1. **Is `days_to_close` measured against the snapshot's own close date or
   against `terminal_date`?** Blocks the close-date reconstruction and therefore
   the bridge, transitions, and slip.
2. **What are the distinct values of `Stage`?** If the vocabulary is
   open-stages-only or numbered, keyword inference returns all-open and every
   rate metric silently returns zero. `terminal_fate` cannot substitute, because
   it is contaminated.
3. **Is `close_date_push_count` cumulative to each snapshot, or a lifetime
   total?** Decides whether a family of movement counters is backward-derived or
   contaminated. `stage_changes_count` has the same question.
4. **Is `qtr_segment` quarter phase or customer segment?** Decides whether any
   segment-like dimension exists at all.
5. **Are `account_ti_first_won` and `account_ti_first_loss` bounded by `as_of`?**
6. **What lookback window do the `rep_*` features use?** Decides whether 23
   columns leave quarantine.
7. **Do the CRM narrative fields exist upstream?** Decides whether a text
   milestone is possible.

### 5.13 Opportunity status is authoritative; stage is an attribute

Status and stage are different things, and conflating them silently corrupts
every rate in the system.

**Status** is the authoritative open, won, or lost state of an opportunity in a
snapshot. **Stage** is a snapshot attribute describing position in the sales
process, used for transition, progression, regression, dwell, and distribution
analysis. Status is never inferred from stage when an authoritative source
exists.

The production stage vocabulary shows why this separation is not academic:

```
0 - Qualification   3 - Proposal      6 - Order Placed   not available
1 - Discovery       4 - Negotiation   Closed Won         SFDCDELETED
2 - Solution Design 5 - Commit        Closed Lost
```

`6 - Order Placed` sounds final, but in this tenant it is still open pipeline.
Keyword inference happens to read it as open, by luck rather than knowledge: on
another tenant the same words could mean a win. It would also count `SFDCDELETED` and `not available` as open
pipeline, inflating every pipeline figure with deleted and unknown records.
Neither failure announces itself.

#### Resolution strategy chain

Status resolves through an ordered chain, and the strategy used is recorded on
the dataset schema so an answer can disclose it.

| Order | Strategy | Authoritative | When it applies |
|---|---|---|---|
| 1 | `authoritative_column` | Yes | A status column is configured and mapped |
| 2 | `mapped_flags` | No | Observed `is_closed` and `is_won` columns exist |
| 3 | `stage_keyword` | No | Only a stage column exists. Recorded as non-authoritative |
| 4 | `unresolved` | No | Nothing to resolve from. Everything is `unknown` |

Status is a five-valued concept, not a pair of booleans:

| Status | Closed | Won | Counts as pipeline |
|---|---|---|---|
| `open` | no | no | **yes** |
| `won` | yes | yes | no |
| `lost` | yes | no | no |
| `excluded` | no | no | no |
| `unknown` | no | no | no |

`excluded` is the value the boolean pair could not express. A deleted record is
not open pipeline, and without this value it would be counted as such.
`unknown` is never guessed into another value: an unmapped status value stays
unknown and is surfaced.

`is_closed` and `is_won` are always derived from `status`, so the three can
never disagree. The status source and its value mapping are configuration, not
code, because the authoritative column has not yet been supplied.

### 5.14 Confirmed dataset semantics

The findings in §5.12 have been confirmed by the project owner. What changed:

| Confirmation | Effect |
|---|---|
| `days_to_close` is `close_date - as_of`, not terminal-derived | **Unblocks the bridge, transitions, and slip.** The close-date family becomes as-of fact. `close_date` is time-varying snapshot state, reconstructed at conform time |
| Stage is not authoritative for status | §5.13. Stage-keyword inference is demoted to a recorded fallback |
| `close_date_push_count` and `stage_changes_count` are cumulative since creation | Backward-derived and safe. **Not recomputed**, see below |
| `qtr_segment` is quarter phase, bucketed from `days_since_boq` | Confirmed temporal, not a customer dimension |
| Account time-since features are historical, with a `-999999` sentinel | Sentinel handling plus a runnable leakage check, see below |
| Rep features are frozen through the previous quarter | Out of quarantine, preserved rather than recomputed, lineage still per-feature unconfirmed |
| CRM narrative fields exist upstream | Text cataloguing implemented; the fields are absent from the 129-column export |

Two consequences deserve their own statement.

**Cumulative counters must not be recomputed.** A counter cumulative since
opportunity creation cannot be reproduced from an export covering eight
quarters, because an opportunity created before the window has history the
system cannot see. Recomputation would silently undercount. This is the one
place where §5.9's "recompute by default" rule is deliberately inverted, and
the reason is a data-coverage limit rather than a trust judgement.

**A sentinel value is a correctness hazard, not a formatting detail.** The
`-999999` used when an account has no prior terminal event would, entering a
mean or a sum unmasked, produce a wildly wrong number with no error. Sentinels
are therefore declared on the column classification and must be masked before
any aggregate.

**The account time-since features get a runnable leakage check.** They are
defined as `as_of` minus the account's earliest `terminal_date` for a given
fate, so their derivation reads outcome fields. The no-look-ahead property is a
requirement rather than a guarantee, and it is directly checkable: every
non-sentinel value must be at least zero, because a negative value means the
terminal event postdates the snapshot. That assertion is recorded on the column
as `leakage_check`. It is prose today; no executor stands behind it yet.

#### Revised classification

| Axis | Before confirmations | After |
|---|---|---|
| Prospectively usable | 49 | 95 |
| Quarantined | 48 | 19 |
| Awaiting confirmation | 126 | 123 |

The nineteen still quarantined are the columns whose reference population is
genuinely unstated: the `_percentile_overall` and `_rank_pct` family, the
`_vs_*_avg` comparisons, the composite scores `activity_density` and
`eoq_urgency_score`, `opp_commit_to_close_days` where "close" probably means
terminal, and `train`, which is quarantined permanently because subsetting on a
train and test split would silently change every number.

#### Information classes

The project asked for the snapshot, historical, and retrospective distinction to
be explicit. It is exposed as `information_class`, **derived** from the category
rather than stored alongside it, so the two can never drift apart.

It answers *what a column is*. It deliberately does **not** answer *when a
column is safe to read*: availability answers that, and `usable_prospectively`
is the gate. Keeping the two apart means a column with a settled category but an
undocumented lookback still reports what it is, instead of disappearing into an
"unclassified" bucket that would hide how much is known about it.

| Information class | Columns | Prospective use |
|---|---|---|
| `identity` | 5 | Keys |
| `snapshot_state` | 16 | Yes |
| `temporal_context` | 18 | Yes |
| `historical_feature` | 84 | Yes, subject to availability |
| `retrospective_outcome` | 5 | **No** |
| `metadata` | 1 | No |
| `text` | 0 in this export | Catalogue only |
| `unclassified` | 0 | **No**, fails closed |

All 129 are classified. `unclassified` is reserved for a column that no registry
entry and no family pattern could place, which is the fail-closed path for a
column appearing in an export without anyone documenting it.

Availability cuts across this independently: of the 84 historical features, 19
are quarantined because their reference population or lookback is unstated. They
are still known to be historical features; what is unknown is the window.

#### Dynamic columns are preserved, not dropped

A non-canonical source column now survives ingestion as a *discovered column*,
carried through under its original name. Dropping it at conform time
would have made §5.10 classification and §5.11 cataloguing impossible, because
there would be nothing left to classify. A discovered column whose name collides
with a canonical one is dropped, since the canonical meaning must win.

Discovered columns are unclassified by default and therefore excluded from
prospective analysis. Adding a column to an export can never silently change an
answer.

They were first preserved as text, because their types were not known at
conform time. That is superseded: they are now typed once at ingestion (§5.15),
so a declared sentinel is compared against a real number and no downstream
layer has to cast.

**Wired in §5.15.** The registry is now created for each ingested dataset and
persisted beside it, and discovered columns are typed and fully profiled.

### 5.15 The dataset-scoped registry and complete profiling

`OPPORTUNITY_SNAPSHOT_V1` describes the *shape* of an export and remains a static
reference. What a running system consults is a **dataset registry**, created for
each ingested dataset, holding the columns that dataset actually contains.

```
Dataset
  |- schema           how the file mapped onto the canonical contract     schema.json
  |- registry         what every column means and when it may be read     classifications.json
  `- profile          what values actually occur                          profile.json
```

The three are persisted separately and joined only by column name. Loading
(`load_dataset`) reconstructs all three from disk and never re-runs the profiler.
A profile whose columns no longer match the registry is refused rather than
trusted, and re-ingesting deletes the previous profile, because a profile
describes the data it was computed from.

#### Classification and profile are different objects

| | Classification (registry) | Profile |
|---|---|---|
| Answers | What does this column mean, and when may it be read? | What values actually occur? |
| Source | Export registry, canonical contract, family patterns | Observation, in DuckDB SQL |
| Holds | Category, information class, availability, disposition, sentinels, lineage, quarantine reason | Null count, distinct count, extremes, summary statistics, top values, length statistics |
| May change because of data? | **Never** | Always |

Nothing observed writes into a classification field. The profiler receives the
registry read-only, to learn which sentinels are declared and which columns are
declared text, and it returns a separate object. A column whose values look like
prose but which no registry classifies is *catalogued* as text-like by the
profile (`basis: detected`) while its classification stays `unclassified`. A
column declared text stays text even when its values are short (`NextStep`).
Tests assert that profiling leaves the persisted classification byte-identical.

#### How a dataset registry is built

For each canonical column present, the registry takes the classification of the
**source column it was mapped from**, re-keyed to the canonical name (so
`new_amount` supplies the classification of `amount`). Canonical columns with no
such source, including the derived `status`, `is_closed`, and `is_won`, take the
canonical contract's own classification. For each discovered column the order is
the export registry, then its family patterns, then **fail closed**:
`unclassified`, quarantined with `unclassified_column`. A dataset with no export
registry therefore has no usable discovered column at all. A column name alone
never classifies anything.

Each entry also records origin (`canonical`, `derived`, `discovered`), source
header, storage type, and declared nullability. `nullable` is structure, not
observation: only the grain columns are non-nullable, because ingestion
hard-fails on a null grain key. Observed null counts are in the profile.

#### Mapping is driven by the registry, and cannot be satisfied by a guess

Running the real 129 headers through the header matcher exposed two defects. It
bound `close_date_qtr`, a quarter label, to the required close date at 0.83,
and it never found `new_amount`. Together they would have let the production
export ingest with an all-null close date, which is strictly worse than the hard
failure §5.12 documents.

- `new_amount` is now a confirmed alias for the amount.
- A fuzzy match on a **required** column makes the mapping incomplete, and
  ingestion refuses it with a typed error naming the guess. Fuzzy matches on
  optional columns are still only flagged.
- With an export registry, the mapping is derived from its own coverage
  records, so the two cannot disagree, and no fuzzy matching occurs.

The production export still cannot ingest, deliberately: it has no close-date
column, and reconstruction as `as_of + days_to_close` is documented but **not
implemented**. Ingestion stops and says so in the error, rather than guessing.

#### Discovered columns are typed, then profiled by type

Discovered columns arrive as text and are typed once, at ingestion, so nothing
downstream has to cast them. Detection is conservative because each shortcut
silently corrupts values:

| Trap | Handling |
|---|---|
| DuckDB rounds: `TRY_CAST('1.5' AS BIGINT)` is 2 | Integrality is tested by pattern, not by cast |
| `'1'` casts to TRUE, so 0/1 columns look boolean | Only the literal tokens `true` and `false` are boolean |
| `'0012'` cast to a number becomes 12 | Leading zeros disqualify a value, so it stays text |
| Integers beyond 18 digits overflow BIGINT | They fall back to DOUBLE |

Integers are stored as `BIGINT`, other numerics as `DOUBLE`, and a column takes a
type only if **every** non-null value fits it. Canonical money remains
`DECIMAL(18,2)`. **A discovered money-valued column is `DOUBLE`**, which departs
from the money rule; such a column must be cast before it is ever used as a
measure, and none currently is.

Profiles are type-appropriate and computed entirely in SQL:

| Kind | Recorded |
|---|---|
| Numeric (`BIGINT`, `DOUBLE`) | nulls, distinct, min, max, mean, median, standard deviation |
| Numeric (`DECIMAL`) | nulls, distinct, exact min and max as strings. **No mean**, because averaging money passes it through a float |
| Date | nulls, distinct, min, max |
| Boolean | nulls, distinct, counts |
| Categorical | nulls, distinct, min, max, top values |
| Text | nulls, distinct, mean and maximum length. **No content**: no min, max, or top values |

To stay compact enough for model context, top values are listed only for columns
with at most `profile_max_top_value_cardinality` distinct values (default 50);
beyond that the column is marked `high_cardinality` and lists none.

#### Sentinels are column metadata, never a value rule

`-999999` marks "no prior event" in `account_ti_first_won` and
`account_ti_first_loss`, and only there. The declaration lives on the column's
classification. For a declared column the profile reports the sentinel count
apart from the null count and the observed count, and every statistic is
computed over observed values only. In any other column `-999999` is an ordinary
value and stays in the statistics; a test holds `-999999` in an undeclared
column to prove nothing was globalised.

#### Quarantine states its reason

A quarantined column now carries a `QuarantineReason` (a code, the detail, and
what would release it), and the classification refuses to construct without one.

| Code | Columns |
|---|---|
| `unstated_reference_population` | 13: percentile and rank features, versus-average comparisons, rep pipeline shares, density, outlier flag |
| `undefined_composite` | 2: `activity_density`, `eoq_urgency_score` |
| `unstated_window` | 2: account open and historical deal counts |
| `ambiguous_definition` | 1: `opp_commit_to_close_days` |
| `non_analytic_metadata` | 1: `train` |
| `unclassified_column` | any column nothing could place |

No quarantine decision changed.

#### Unresolved items are data

Open questions are recorded as `UnresolvedItem` records so a later layer can
enumerate them. Nothing in them is assumed.

| Id | Question | Columns |
|---|---|---|
| `authoritative_status` | Which column is authoritative for open, won, lost, and how do its values map? Raised whenever none is configured | `status`, `is_closed`, `is_won` |
| `status_unmapped_values` | An authoritative column holds values outside the mapping | same |
| `opp_commit_to_close_days_definition` | Does "close" mean the snapshot close date or the terminal date? | 1 |
| `rep_aggregate_lineage` | Lookback per feature, and whether each snapshot's value is frozen as of its own previous quarter or as of the dataset's end | 23 |

**The rep aggregates deserve a flag.** Their classification is unchanged from the
previous milestone: `use_with_proof`, backward-derived, usable prospectively,
with lineage unconfirmed. They are not quarantined. The lineage item exists
because "frozen through the previous quarter" is only safe if each *snapshot's*
value was frozen as of that snapshot's own previous quarter; a freeze at the
dataset's end would embed outcomes after every older snapshot. The requirement
check attaches the item as a caveat so a consumer can withhold them, but the
registry itself does not.

#### Status configuration

Status resolution is configuration: an authoritative column and a value mapping
to `open`, `won`, `lost`, and `excluded`, with the fallback chain intact
(authoritative column, then observed flags, then stage keywords). The real
column name has not been supplied and is not guessed. The strategy is recorded
on the schema, restated in the profile with the resulting distribution, and
raised as an unresolved item whenever no authoritative source is configured.

Misconfiguration is a hard error, never a silent fallback: a configured column
missing from the source, a column bound to status with no value mapping, a value
mapping with no column, and a mapping that lists no won, lost, or open value.
Raw values outside the mapping are listed and resolve to `unknown` rather than
being guessed. The stage vocabulary in the profile is always reported as
keyword-inferred and unconfirmed, even under an authoritative status column.

#### The requirement check

The next milestone needs to know which metrics to hide without the registry
knowing anything about metrics. `check_requirements(names)` answers per column:

| State | Meaning |
|---|---|
| `available` | Present, knowable at the snapshot, no open question |
| `available_with_caveats` | Usable, but an unresolved item or unconfirmed lineage applies; the ids are returned |
| `quarantined` | Withheld; the reason is returned |
| `not_knowable_at_snapshot` | Present but contaminated or unknown: retrospective only |
| `missing` | Not in this dataset |

The result reports whether the set is satisfiable prospectively (nothing missing,
quarantined, or not knowable) and retrospectively (nothing missing or
quarantined).

#### Limitations recorded, not hidden

- No real dataset was available. Everything here is verified against a
  synthetic fixture carrying the 129 production column names; no classification
  has been checked against real rows.
- `close_date` reconstruction is not implemented, so the production export as
  described fails ingestion by design.
- `DatasetColumn.dtype` for a discovered money-valued column is `DOUBLE`.
- A parquet source column holding `NaN` or `inf` is not recognised as numeric
  and is stored as text.
- The `leakage_check` assertions on the account time-since features are still
  prose with no executor.
- `historical_win_rate_at_stage`, `quarter_close_rate_hist`, `prev_won`, and
  `prev_loss` also carry unconfirmed lineage but have no unresolved item; they
  appear only as `lineage_unconfirmed` caveats.

---

## 6. Agent and Tool Architecture

### 6.1 The loop

> **Superseded by §13.3.** The orchestrator state machine there replaces this
> loop; the principle, that only planning and response drafting call a model,
> is unchanged.

```
User turn
  │
  ├─ 1. Context assembly      schema card + profile card + metric catalog + session state
  │                           (cached prefix; static across the session)
  │
  ├─ 2. Intent triage         answerable · needs clarification · out of scope
  │                           out-of-scope and unanswerable exit here → abstention
  │
  ├─ 3. Planning              LLM emits AnalysisPlan via structured output
  │                           ↓
  ├─ 4. Plan gate             Pydantic validation + semantic validation
  │                           rejection → structured error back to planner (max 2 repairs)
  │                           ↓
  ├─ 5. Compilation           AnalysisPlan → DuckDB SQL (deterministic, no LLM)
  │                           ↓
  ├─ 6. Execution             read-only DuckDB, row cap, timeout → ResultSet + query_id
  │                           ↓
  ├─ 7. Sanity checks         invariants per result type (§8.3)
  │                           ↓
  ├─ 8. Response drafting     LLM writes narrative with {{query_id.row.col}} tokens
  │                           ↓
  ├─ 9. Rendering             deterministic substitution of real values
  │                           ↓
  ├─ 10. Provenance scan      every numeral must trace to a result cell (§8.4)
  │                           failure → one retry → degrade to table-only answer
  │                           ↓
  └─ 11. Emit                 answer + evidence table + chart + plan + SQL + run_id
```

Steps 5, 6, 7, 9, and 10 contain no LLM call. That is the point.

### 6.2 Why a hand-written loop and not a framework

The validation gates at steps 4, 7, and 10 are the product. A framework's
built-in loop would run the model, execute, and return, giving no natural place
to reject a plan and feed a structured repair error back. The loop is small
enough to own outright.

### 6.3 Tool surface

Tools are deliberately few and deliberately narrow. A large tool surface invites
the model to improvise.

| Tool | Purpose | Trust tier |
|---|---|---|
| `get_schema()` | Canonical columns, types, nullability, mapping applied | n/a |
| `profile_column(name)` | Distinct values, null rate, min/max, top-k | n/a |
| `list_metrics()` | Schema-filtered metric catalog | n/a |
| `describe_metric(name)` | Definition, defaults, ambiguity notes | n/a |
| `run_plan(plan)` | **Primary path.** Compile, validate, execute | **A, trusted** |
| `compute_slip_risk(...)` | Deterministic explainable score (§6.5) | **A, trusted** |
| `run_sql(sql)` | Gated escape hatch (§7.4) | **B, flagged** |
| `run_python(query_id, code)` | Post-processing sandbox (§6.4) | **B, flagged** |
| `make_chart(query_id, spec)` | Vega-Lite spec over an existing result | n/a |
| `request_clarification(...)` | Ask the user rather than assume | n/a |

Trust tier travels with the result into the answer. A tier-B result renders with
an explicit caveat in the response and is recorded as such in the run record.

> **Revised by §12.9.** The multi-tenant surface adds schema inspection and
> concept definition tools, replaces `run_sql` as the primary ad-hoc path with a
> typed `InvestigationPlan` (§12.8), and extends the two trust tiers to three
> (§12.7), where the tier is computed from the inputs rather than chosen.

`request_clarification` is a real tool, not a fallback. The model is instructed
to prefer asking over guessing when an ambiguity from §5.3 is load-bearing and
the question gives no signal. Abstention accuracy is a headline metric (§9.3).

### 6.4 The Python sandbox is narrow by design

`run_python` operates **only on already-materialized result frames**, never on
raw data and never with filesystem or network access. It exists for pivots,
reshaping, and derived percentages that are awkward in SQL. Restrictions:

- Input is a registered `query_id`, loaded as a read-only DataFrame
- Import allowlist: `pandas`, `numpy`, `math`, `statistics` only
- No `open`, no `__import__`, no `eval`, no `exec`, no attribute access to dunders
- Subprocess isolation with wall-clock and memory caps
- Output must be a DataFrame or scalar, which is re-registered as a new `query_id`

This keeps the escape hatch from becoming a way to bypass the semantic layer.

### 6.5 Slip risk scoring is deterministic and explainable

Question 7 asks for a prediction. The MVP answer is a transparent additive score
computed in SQL, not a trained model:

| Factor | Signal | Direction |
|---|---|---|
| Prior close-date pushes | Count of times `close_date` moved later across snapshots | increases risk |
| Days in current stage | Current `as_of` minus stage-entry snapshot | increases risk |
| Age vs. typical cycle | Days open vs. median won-deal cycle for that segment | increases risk |
| Stage vs. time remaining | Early stage with a close date inside 30 days | increases risk |
| Amount volatility | Number of `amount` changes across snapshots | increases risk |
| Forecast category downgrade | movement from Commit to Best Case to Pipeline | increases risk |

Each factor's contribution is returned alongside the score. The LLM narrates the
factors; it does not compute or adjust the score. Weights are configuration, not
model output, and are stated in the answer. A learned model is a future item
(§10.3), gated on having enough closed history to validate against.

### 6.6 Model configuration

> **Superseded by §13.16.** Model choice per call is configuration, with a
> faster model for the responder than for the planner.

```python
model            = "claude-opus-5"
thinking         = {"type": "adaptive"}        # planning benefits from reasoning depth
output_config    = {"effort": "high"}          # correctness over token economy
```

Structured output via `client.messages.parse(output_format=AnalysisPlan)` so the
plan arrives already validated against the Pydantic model. Planner and Responder
are separate calls with separate prompts. The static prefix (schema card, profile
card, metric catalog) sits behind a cache breakpoint; the volatile question goes
last. Mid-conversation operator instructions append as a `system` role message in
`messages[]` rather than editing the top-level system prompt, preserving the
cached prefix.

### 6.7 Conversation context

> **Superseded by §13.13.** Session state holds plans and typed `PlanEdit`s,
> and the carry-forward report is computed by diffing plans.

Session state is structured, not a raw transcript:

```python
class SessionState(BaseModel):
    dataset_id: str
    turns: list[Turn]                       # question, plan, query_ids, answer
    resolved_entities: dict[str, Entity]    # "that quarter" → FY25-Q3
    active_filters: list[Filter]            # carried forward until cleared
    last_result_ids: list[QueryId]          # for "chart that", "show me more"
```

Follow-ups like "now break that down by region" resolve by **mutating the prior
`AnalysisPlan`**, not by re-planning from scratch. This is a direct benefit of
plan-as-data: the plan is a diffable object. The resolver is explicit about what
it carried forward, and the answer says so.

---

## 7. Data Storage and Query Layer

### 7.1 Physical layout

```
data/
└── datasets/{dataset_id}/
    ├── raw/                      # exactly as uploaded, never mutated
    ├── canonical/
    │   └── snapshots/as_of=YYYY-MM-DD/part-0.parquet    # Hive-partitioned
    ├── profile.json              # DatasetProfile: observed values, every column
    ├── classifications.json      # DatasetRegistry: what each column means (5.15)
    ├── schema.json               # mapping + canonical schema + status resolution
    └── results/{query_id}.parquet                        # result registry
```

Partitioning by `as_of` lets nearly every query prune to one or two partitions.
Snapshot selection resolves to specific dates *before* the scan, so
`PERIOD_OPEN` reads exactly one partition.

### 7.2 Ingestion pipeline

```
Upload → Sniff (delimiter, encoding, types)
       → Column mapping (LLM proposes, user confirms)   ← human in the loop
       → Conform (types, fiscal calendar, derived flags)
       → Grain assertion: (as_of, opp_id) unique        ← HARD FAIL if violated
       → Write partitioned Parquet
       → Profile
       → Build schema-filtered metric catalog
```

The grain assertion is not a warning. If `(as_of, opp_id)` is not unique,
every snapshot metric in the system is silently wrong, so ingestion stops and
reports the offending duplicate keys.

### 7.3 Profiling

The profile is computed once and persisted. It is **not** injected into the LLM
context: the implemented profile is 59 KB of compact JSON, roughly 15k tokens,
for 145 columns. §12.5 defines the tiered card that is projected from it, and
the tool that serves one column's profile on request. The profile contains:

- Snapshot inventory: dates, row counts, gaps, cadence regularity
- Per-column: type, null rate, distinct count, min/max, top-k values
- Stage vocabulary and inferred closed/won mapping, flagged for confirmation
- Opportunity lifecycle stats: median snapshots per opp, appearance/disappearance
- Detected fiscal boundaries and snapshot drift from quarter starts
- Data quality flags: negative amounts, close dates before created dates,
  opportunities that vanish without a terminal state
- Cast failures, measured during ingestion rather than here. A cast failure is a
  source cell that held text but did not survive its cast, which can only be
  seen while the raw text is still available. A blank cell is a legitimate null
  and is reported as a null count, not as a failure.

The last item matters. Opportunities that disappear from snapshots without
closing are the `other_removed` term in the bridge. If that term is large, every
derived rate is suspect, and the profile surfaces it up front.

### 7.4 The SQL escape hatch and its guard

> **Extended by §12.8.** Under the multi-tenant design, free-form SQL is the
> *fallback* behind a typed `InvestigationPlan`, not the primary ad-hoc path,
> and the guard below gains a column allowlist derived from the analysis stance
> so that a prospective question cannot reference a contaminated column at all.

Free-form SQL is available but constrained:

1. **Parse first** with `sqlglot` in DuckDB dialect. Unparseable is rejected.
2. **AST allowlist**: `SELECT` and `WITH` only. Reject `ATTACH`, `COPY`,
   `INSTALL`, `LOAD`, `EXPORT`, `PRAGMA`, `CREATE`, `INSERT`, `UPDATE`,
   `DELETE`, `DROP`, and any function reading the filesystem
   (`read_csv`, `read_parquet`, `glob`).
3. **Table allowlist**: only the conformed view and registered result tables.
4. **Read-only connection**: DuckDB opened with `read_only=True` against a
   per-session copy. Defense in depth behind the allowlist.
5. **Caps**: `LIMIT` injected if absent (default 10,000 rows), wall-clock
   timeout (default 30s), memory cap.
6. **Trust tier B**: the result is flagged, and the answer discloses that a
   custom query was used.

Every escape-hatch use is logged as a **semantic layer gap**. Recurring gaps are
the backlog for new metrics. The escape hatch shrinking over time is a health
signal.

---

## 8. Preventing Hallucinated Results

Five independent layers. Each catches what the others miss.

### 8.1 Layer 1, structural: the LLM cannot express arithmetic

The plan schema has no free-text numeric field. The model selects from
enumerated metrics, dimensions, and rules. It cannot write `SUM(amount) * 1.1`
because the plan has nowhere to put it. Whole categories of fabrication become
unrepresentable rather than merely discouraged.

### 8.2 Layer 2: Plan gate

Beyond Pydantic type validation:

- Referenced columns exist in the canonical schema
- Referenced metrics exist in the **schema-filtered** catalog
- Filter values are checked against profiled distinct values, with
  near-miss suggestions on failure ("`Enterprise` not found; did you mean `ENT`?")
- Requested periods fall within the dataset's snapshot coverage
- Snapshot rule is compatible with the metric
- Dimension cardinality is sane for the requested output shape

Rejections return **structured errors**, not prose, and the planner gets at most
two repair attempts before the turn escalates to clarification.

### 8.3 Layer 3: Post-execution sanity checks

Invariants asserted on the result, by type:

| Result type | Invariant |
|---|---|
| Bridge | Components sum to ending minus opening, tolerance 0.01 |
| Any rate | Numerator ≤ denominator; denominator > 0 |
| Counts | Non-negative integers |
| Cohort trace | Terminal states partition the cohort exactly, no double-count |
| Time series | Expected number of periods present; gaps flagged, not silently dropped |
| Snapshot query | Resolved `as_of` within `max_snapshot_drift_days` of requested boundary |
| Any result | Non-empty, or the answer explicitly reports "no matching rows" |
| Slip risk | Score within declared bounds; factor contributions sum to score |

A failed invariant fails the turn. It does not produce a hedged answer.

### 8.4 Layer 4: Numbers never pass through the model as free text

This is the strongest layer and the one that distinguishes this design.

The Responder is instructed to emit **reference tokens**, not values:

```
Draft from LLM:
  "Opening pipeline for {{q1.r0.period}} was {{q1.r0.opening_pipeline}},
   which is {{q2.r0.pct_change}} lower than the prior quarter. The largest
   single contributor was slippage at {{q3.r0.slipped_out}}."

Rendered deterministically:
  "Opening pipeline for FY25-Q3 was $14,207,500, which is 12.4% lower than
   the prior quarter. The largest single contributor was slippage at
   $2,840,000."
```

Then the **provenance scanner** runs over the rendered text:

1. Extract every numeral, currency amount, and percentage
2. Each must match a value substituted from a registered `(query_id, row, column)`
3. Whitelist: years, quarter labels, ordinals ("top 20" when it matches the
   requested limit), and values the user themselves supplied in the question
4. Any unmatched numeral fails the answer

On failure: one regeneration with the failure surfaced to the model. On second
failure: degrade to a table-only answer with a note. **Never** emit an unverified
number. Unresolvable tokens are also a failure, so the model cannot invent a
reference that does not exist.

`provenance_coverage`, the percentage of emitted numerals traceable to a result
cell, is a headline metric and should sit at 100% in production.

### 8.5 Layer 5: Abstention as a first-class outcome

When the schema cannot support a question, the correct output is a refusal or a
clarifying question, not a best-effort number. Three abstention triggers:

- **Schema gap**: the metric needs a column the dataset lacks. The metric was
  already filtered out of the catalog (§5.5), so the model sees the absence.
- **Coverage gap**: the question asks about periods outside the snapshot range.
- **Load-bearing ambiguity**: an ambiguity from §5.3 would change the answer
  materially and the question gives no signal.

Abstention is measured (§9.3) and is not treated as a failure mode.

### 8.6 Full provenance record

Every turn persists a complete, replayable record:

```python
class RunRecord(BaseModel):
    run_id: str
    session_id: str
    question: str
    plan: AnalysisPlan | None
    plan_repairs: list[PlanRejection]
    compiled_sql: list[str]
    query_ids: list[QueryId]
    result_hashes: list[str]
    sanity_results: list[SanityCheck]
    draft_answer: str                    # with tokens, pre-substitution
    rendered_answer: str
    provenance_report: ProvenanceReport
    trust_tier: Literal["A", "B"]
    abstained: bool
    tokens_in: int
    tokens_out: int
    latency_ms: int
    model_version: str
    prompt_version: str
    semantic_layer_version: str
```

Any number in any answer can be traced back to the exact SQL that produced it
and re-executed to confirm. The UI exposes "show the SQL" and "show the plan" on
every answer.

---

## 9. Evaluation Strategy

Four tiers. Each catches a distinct failure class, and none substitutes for
another.

### 9.1 Tier 1: Semantic layer unit tests

A **tiny hand-built fixture**: roughly 40 rows, 8 opportunities, 6 snapshots,
2 quarters. Small enough that every expected answer is provable by reading the
CSV. Deliberately includes the hard cases:

- An opportunity that slips from Q1 to Q2 to Q3
- One that gets pulled in from a later quarter
- One whose segment changes mid-life (tests attribution, §5.3 #5)
- One whose amount changes across snapshots
- One that vanishes without a terminal state
- One created mid-quarter
- A duplicate-key violation in a separate negative fixture

These tests run without any LLM call. They are the regression suite for the
compiler, and they are fast enough to run on every commit.

### 9.2 Tier 2: End-to-end numeric exact-match

A golden set of roughly 60 to 100 questions with expected values.

**Critical discipline:** expected values come from an **independently written
reference implementation** in `tests/golden/reference/`, written directly in
Pandas from the metric definitions, not by running the agent and blessing its
output. Blessing agent output makes the eval agree with whatever the system
currently does, including its bugs.

Scored dimensions, separately:

| Dimension | What it catches |
|---|---|
| Plan correctness | Did it choose the right metric, grain, snapshot rule, filters? |
| Numeric exactness | Do the numbers match the reference, to the cent? |
| Answer completeness | Did it surface the caveats it should have? |

Plan correctness and numeric correctness are scored independently because they
fail independently. A right plan with a compiler bug and a wrong plan that
coincidentally matches are very different problems.

### 9.3 Tier 3: Adversarial and abstention probes

Questions the system **should refuse or clarify**:

- Metrics requiring absent columns ("what is win rate by industry" with no industry column)
- Periods outside snapshot coverage ("how did we do in FY22")
- Genuinely causal questions ("why did the Enterprise team miss")
- Load-bearing ambiguity with no signal ("how many deals slipped" with no period)
- Nonsense with plausible vocabulary ("what was the pipeline velocity coefficient")

**Abstention accuracy is a headline metric.** A system that answers everything is
worse than one that knows its limits, and this tier is the only place that shows
up. Both directions are scored: wrongly answering an unanswerable question, and
wrongly refusing an answerable one.

### 9.4 Tier 4: LLM-as-judge, narrative quality only

Judges clarity, appropriate hedging, caveat surfacing, and whether the chart
choice fits the data shape.

**A judge never scores numeric correctness.** That is tier 2's job, by exact
match against an independent reference. Using a judge for numbers reintroduces
exactly the fallibility the architecture exists to remove.

### 9.5 Headline metrics

| Metric | Target | Tier |
|---|---|---|
| Numeric exactness | 100% on golden set | 2 |
| Plan validity rate (first attempt) | > 90% | 2 |
| Execution success rate | > 98% | 2 |
| Provenance coverage | 100% | 4 (every run) |
| Bridge balance failures | 0 | 1, 3 |
| Abstention accuracy | > 90% | 3 |
| Escape-hatch rate | < 10% and falling | production |
| p50 / p95 latency | tracked per turn | production |
| Cost per question | tracked per turn | production |

Numeric exactness has a target of 100%, not 95%. An analyst tool that is wrong
five percent of the time is worse than no tool, because the user cannot tell
which five percent.

### 9.6 CI wiring

- Tiers 1 and 3 run on every commit. No LLM calls in tier 1, cheap calls in tier 3.
- Tier 2 runs on pull requests and nightly, with cost budgeted.
- Tier 4 runs nightly.
- Prompt, semantic layer, and model versions are recorded per eval run so
  regressions attribute to a change.

---

## 10. MVP versus Future

### 10.1 MVP: the defensible core

Everything needed to prove the central thesis, and nothing else.

| Area | Scope |
|---|---|
| Ingestion | CSV and Parquet upload, LLM-assisted column mapping with user confirmation, grain assertion, profiling |
| Semantic layer | Snapshot rules, fiscal calendar, pipeline bridge, the six primitives, roughly 15 metrics |
| Question patterns | All six patterns from §1.2, covering the ten example questions |
| Planning | `AnalysisPlan` with structured output, plan gate, bounded repair loop |
| Execution | Compiler, read-only DuckDB, result registry |
| Validation | All five anti-hallucination layers (§8) |
| Response | Reference-token rendering, evidence table, provenance scan |
| Charts | Vega-Lite specs for the four core shapes: time series, bar, waterfall (bridge), ranked table |
| Sessions | Structured state, plan mutation for follow-ups |
| Slip risk | Deterministic explainable score (§6.5) |
| API | FastAPI with SSE progress streaming |
| Evaluation | All four tiers wired into CI |
| Escape hatch | `run_sql` with the full guard, `run_python` sandbox |

Deliberately **excluded from MVP**: authentication, multi-tenancy, a polished
frontend beyond a minimal chat UI, multi-dataset joins, and scheduled reporting.

### 10.2 Near-term

- Narrative-first executive summaries over a whole quarter
- Anomaly detection on the bridge, flagging unusual component movements
- Metric lineage visualization
- Export to slides and spreadsheets
- Multi-currency with dated FX rates
- User-defined metrics persisted into the registry

### 10.3 Longer-term, with justification required

| Feature | Trigger for building it |
|---|---|
| Learned slip-risk model | Enough closed history to hold out a validation set, plus a measured lift over the deterministic score |
| Cross-dataset joins (bookings, activity, marketing) | A concrete question the snapshot data alone cannot answer |
| Semantic caching of plans | Measured repeat-question rate above a threshold |
| Vector search over metric catalog | Catalog exceeds what fits comfortably in a cached prefix |
| Background job queue | Query latencies that exceed a request timeout |
| Multi-tenant deployment | An actual second tenant |

Each of these adds infrastructure. Per the engineering rules, none is adopted
without a written justification tied to a measured need.

---

## 11. Key Risks

| Risk | Mitigation |
|---|---|
| Real schema diverges sharply from the canonical contract | Mapping layer plus hard-fail profiling; the canonical contract is explicitly an assumption (§1.4) |
| Snapshot cadence is irregular or has gaps | Profile surfaces cadence and gaps; snapshot rules report actual resolved dates and drift |
| Opportunities vanish without terminal state | Explicit `other_removed` bridge term; profile flags the rate up front |
| Stage vocabulary does not map cleanly to closed/won | Inferred mapping is surfaced for user confirmation at ingest, never silently assumed |
| Planner produces valid-but-wrong plans | Plan correctness scored separately in tier 2; clarification preferred over guessing |
| Provenance scanner false positives on legitimate numerals | Explicit whitelist for years, quarter labels, and user-supplied values; tuned against the golden set |
| Semantic layer becomes a bottleneck for new questions | Escape-hatch usage logged as gap signal, feeding the metric backlog |
| Cost per question grows unbounded | Cached static prefix, per-turn tool-call budget, tracked cost metric |

---

## 12. Heterogeneous Schemas and the Analytical Ontology

**Status: partly implemented.** The ontology, concept bindings, agreement
tests, the dataset understanding layer, the tiered context card, close-date
reconstruction, grain-only ingestion and the monetary policy are built and
tested (12.1 to 12.5, 12.10, 12.11, 12.15, 12.16). The analytical paths, the
trust model, the ad-hoc surface and the tool surface are **not** built
(12.6 to 12.9); §12.18 records exactly what is which. Sections 1 to 11
describe a system with one canonical schema. This section adapts that system to
many tenants with different physical schemas, without weakening a single
guarantee. Where it contradicts an earlier section, it supersedes it and says so
explicitly.

### 12.0 What changed and what did not

The system will be used across tenants. Tenants differ in column names, column
subsets, custom columns, CRM-derived features, text fields, segment-scoped
amount columns, and in how they represent the same business concept. A fixed
physical schema, and a fixed list of metrics bound to it, cannot serve them.

What changes: the schema is discovered and adapted at runtime, and the agent
reasons about which columns are relevant rather than choosing from a hardcoded
metric list.

What does not change, and must not:

- The LLM never produces a number. Every numeral traces to executed computation.
- The temporal safety rule (§5.8). A prospective question cannot read a column
  whose value depends on what happened later.
- Fail closed. What is not known is not used.
- Nothing observed rewrites what a column means (§5.15).

The distinction that makes all three survive the change: **the LLM decides what
to look at; deterministic code decides what a number is.** Discovery widens the
first and leaves the second untouched.

### 12.1 The stable analytical ontology

Today §1.4 defines 18 canonical columns and `conform.py` renames tenant headers
onto them. That conflates two different things: the *business concept* a metric
needs, and the *physical column* a tenant happens to store it in.

Separate them. A **concept** is a stable analytical idea with a written
definition. It is the vocabulary metrics and plans are written in, and it never
varies by tenant. A **physical column** is what a dataset actually contains.

| Concept | Cardinality | Class | Definition |
|---|---|---|---|
| `opportunity_id` | one | grain | Stable identifier for one opportunity across snapshots |
| `snapshot_date` | one | grain | The date at which this row's state was recorded |
| `amount` | one | core | Deal value in the reporting currency, as recorded in this snapshot |
| `expected_close_date` | one | core | The close date recorded *in this snapshot*, not the realized one |
| `stage` | one | core | Sales stage label, an attribute and never authoritative for status |
| `opportunity_status` | one | core | Authoritative open / won / lost / excluded |
| `forecast_category` | one | optional | Commit, Best Case, Pipeline, Omitted |
| `created_date` | one | optional | Opportunity creation date |
| `account` | one | optional | Account identifier or name |
| `owner` | one | optional | Owning rep |
| `probability` | one | optional | Win probability as recorded in this snapshot |
| `segment`, `region`, `industry` | one each | optional | Mutable descriptive dimensions |
| `terminal_outcome` | one | retrospective | What eventually happened |
| `terminal_date`, `terminal_amount` | one each | retrospective | Realized close date and value |
| `segment_amount` | many | optional | Tenant-specific amount columns that may partition `amount` |
| `narrative_text` | many | optional | Free-text CRM fields |
| `precomputed_feature` | many | optional | Derived features supplied by the tenant's pipeline |

`quarter` is deliberately **not** a concept bound to a column. It is computed by
the fiscal calendar (§5.6) from a date concept. A tenant's stamped quarter label
is a physical column that gets *reconciled against* the calendar (§5.7), never
trusted as the definition.

Two consequences worth stating. First, the concept table is short and the
physical schema is long: a 145-column dataset binds perhaps fifteen concepts and
leaves 130 columns as discovered features reachable only by inspection. Second,
`CanonicalColumn` does not disappear. It remains the physical naming contract of
the *conformed* table for the grain and whatever else binds, so the compiler
still emits stable SQL. The ontology sits above it.

### 12.2 Concept binding: evidence, status, and the rule that name similarity never confirms

A **binding** connects one concept to one or more physical columns in one
dataset, and records why.

```
concept  →  candidate physical columns  →  evidence  →  status
```

| Status | Meaning | Admissible where |
|---|---|---|
| `confirmed` | Declared by tenant config, the export registry, or a user | Semantic metrics (path A) and everywhere below |
| `inferred` | Evidence supports it; nobody has confirmed it | Investigation only (path B), always disclosed |
| `ambiguous` | Two or more candidates, none decisive | Nowhere. The agent asks |
| `unavailable` | Nothing binds | Nowhere. Metrics needing it are hidden |

Evidence is typed and carries its source, so a binding can be explained:

| Evidence kind | Example | Can it confirm? |
|---|---|---|
| `user_confirmation` | A person answered "yes, ARR is the amount" | Yes |
| `tenant_config` | `tenant.json` declares the binding | Yes |
| `export_registry` | A reviewed registry documents coverage (§5.15) | Yes |
| `documentation` | A supplied data dictionary defines the column | Yes, when reviewed |
| `exact_name` | Header equals the concept name | No |
| `alias` | Header is a known synonym | No |
| `value_pattern` | Values look like dates, or agree with another column | No |
| `type_shape` | Type and cardinality are consistent with the concept | No |

**Name similarity never confirms a binding, and no amount of inferential
evidence ever promotes to `confirmed`.** Only a declaration does. This is the
generalization of the rule the current mapper already enforces, and it was
earned: on the real headers the fuzzy matcher bound `close_date_qtr`, a quarter
label, to the required close date. Ten tenants make that failure ten times more
likely, not less.

Inferential evidence still has a job. It proposes bindings for a human to
confirm, and it is what lets the agent reason about a tenant column in the
investigation path while disclosing that the reading is unconfirmed.

### 12.3 The agreement test: evidence, never proof

Several questions in this design have the same shape: *does this column mean
what we think it means?* They are answerable by a deterministic check whose
failure is decisive and whose pass is merely evidence.

An **agreement test** is a named SQL predicate evaluated over every row, which
reports the disagreeing row count. Three properties make it useful:

- **Failure is decisive.** One disagreeing row disproves the reading.
- **A pass is evidence, not proof.** Agreement can be coincidental, and the
  export may not contain the case that would break it.
- **The count is reported**, so a near-miss is visible rather than rounded away.

Two uses, both of which would otherwise tempt someone to guess:

**Reconstructing a date.** If `expected_close_date` is reconstructed as
`snapshot_date + days_to_close`, the reconstruction is checked against every
independent representation the export carries: the fiscal quarter of the
reconstructed date must equal the stamped close quarter, and the end-of-quarter
difference column must equal the reconstructed date minus quarter end. See
§12.15.

**Tenant segment amounts.** Tenant C has `segment_amount_enterprise` and
`segment_amount_midmarket`. Whether these partition `amount` is testable: do
they sum to the amount, on every row, in every snapshot? A pass raises
confidence enough to *propose* unpivoting them into a segment dimension. It does
not confirm it, because a third segment may simply be absent from this export.
Only the tenant declaring the partition confirms it.

This is the mechanism that lets the system reason from value patterns without
ever silently equating two columns.

### 12.4 The dataset understanding layer

A component that runs after ingestion and profiling, reads only artifacts that
already exist, and produces the dataset's analyst context. It performs no LLM
call: it assembles evidence and applies the binding rules of §12.2.

Inputs: the dataset registry and profile (§5.15), tenant configuration, any
supplied documentation, and the export registry when one exists.

What it derives:

- **Concept bindings**, with evidence and status, including the ambiguous and
  unavailable ones. What is *missing* is as load-bearing as what is present.
- **Available dimensions**: the categorical columns with usable cardinality that
  a group-by could legitimately use.
- **Available measures**: numeric columns not quarantined, with their sentinel
  declarations attached.
- **The status resolution** (§5.13) and which strategy produced it.
- **Stage vocabulary** as observed, always marked unconfirmed.
- **Text fields**, discovered by profile shape, catalogued only.
- **Open questions**: unresolved items, unconfirmed lineage, ambiguous bindings.

Relationships between columns are deliberately *not* precomputed here. Over 145
columns the pairwise space is large, most of it is noise, and a stored
correlation invites the model to read causation into an artifact nobody asked
for. `inspect_relationships` computes one on request instead, in SQL, bounded
and stance-filtered.

Agreement tests are evaluated here, once, and their results stored with the
context so the agent never re-runs them mid-conversation.

### 12.5 The analyst context card

The profile is 59 KB of compact JSON, roughly 15k tokens, for 145 columns. It
must never be injected into a prompt. The card is a *projection* of it.

Three tiers:

| Tier | Contents | Where it lives |
|---|---|---|
| 0 | Concept bindings, grain, row and snapshot counts, calendar, status resolution, dimension and measure names, open questions, per-class column counts | The cached prompt prefix |
| 1 | One column's profile, values, and classification | Returned by a tool, on request |
| 2 | Rows | Never in the context at all |

Tier 0 lists *concepts*, not columns, which is what makes it small. It also
carries a **column index**: names only, grouped by information class, so the
agent knows what it may ask about. For 145 columns that index is roughly 800
tokens and the whole card lands near 2k.

**The degradation rule.** A wide tenant would silently break that budget. Above
a configured column count, tier 0 drops the full index and carries per-class
counts plus the bound concepts only; the agent then reaches names through a
listing tool that accepts a class filter and a name pattern. The card is
budgeted in tokens and the budget is enforced when it is built, not hoped for.

A sketch of tier 0 for the production export:

```
dataset: client_xyz   rows: 56   snapshots: 5 (2026-01-05 .. 2026-03-30)
grain: (snapshot_date, opportunity_id)   fiscal year starts: January

concepts
  opportunity_id       -> opp_id          confirmed (registry)
  snapshot_date        -> as_of           confirmed (registry)
  amount               -> new_amount      confirmed (owner)
  stage                -> Stage           confirmed (registry)
  expected_close_date  -> UNAVAILABLE     reconstructible from days_to_close, not implemented
  opportunity_status   -> UNAVAILABLE     no authoritative column configured
  segment              -> UNAVAILABLE     no candidate column
  terminal_outcome     -> terminal_fate   confirmed (registry, retrospective only)
  narrative_text       -> ManagerNotes, SENotes, NextStep, +9

columns: 145   identity 4 · snapshot_state 11 · temporal_context 18
                historical_feature 63 · retrospective_outcome 5 · text 12
                metadata 2 · quarantined 57

open questions: 3   authoritative_status · opp_commit_to_close_days_definition
                    rep_aggregate_lineage
```

Note what that card does: it tells the model, before it plans anything, that
segment analysis and status-based rates are not answerable on this dataset. An
unanswerable question is refused from the card, not from a failed query.

### 12.6 Two analytical paths

**Path A, semantic.** The question maps to a concept-bound metric. Question →
`AnalysisPlan` → compiler → DuckDB → validation → answer. Unchanged from §6.1.
Available only when every concept the metric needs is `confirmed` and every
column it reads passes `check_requirements` under the active stance.

**Path B, investigation.** The question is not covered by the metric registry.
"Is there a relationship between forecast-category changes and deal slippage?"
is a real analytical question and refusing it because it is not one of fifteen
metrics is the wrong answer.

The agent may: inspect the schema and candidate columns, determine whether the
necessary information exists, formulate an analysis, generate a bounded
computation, execute it, validate the result, and explain the methodology with
its assumptions and limitations stated.

The agent may not: define a metric the system will then present as canonical,
compute a number itself, or read a column the stance forbids.

The boundary between the paths is *availability of a definition*, not
difficulty. If the registry defines win rate, path A computes it. If the user
asks about a relationship nobody has defined, path B investigates it and the
answer says so. A path B result never silently becomes a metric; promoting one
is a reviewed change to the registry.

### 12.7 The trust model

Three tiers, extending §6.3's two:

| Tier | Meaning | Rendered as |
|---|---|---|
| A | Semantic deterministic: registry metric, confirmed bindings, compiled plan | The number, plainly |
| B | Validated ad-hoc: executed and checked, but the definition or a binding is not established | The number, with its assumptions and caveats stated |
| C | Insufficient evidence | No number. An explanation, a clarifying question, or an abstention |

**The tier is computed, not chosen.** It is the weakest of its inputs, assigned
deterministically after execution and never by the model:

- any bound concept `inferred` → at most B
- any column read with unconfirmed lineage → at most B
- path B computation → at most B
- any required concept `ambiguous` or `unavailable` → C, and nothing executes

A tier-B answer must disclose *why* it is tier B, naming the unconfirmed binding
or the undocumented feature. "Roughly 42% of the pipeline slipped" with a
footnote about which definition of slippage was used is a good answer. The same
sentence without the footnote is not.

### 12.8 Controlled ad-hoc analysis

Path B needs a computational surface. Unrestricted SQL is not it.

**The primary surface is a second plan type.** An `InvestigationPlan` is still
plan-as-data: select columns, filter, group, aggregate from a fixed function
set, and self-join on the grain across two snapshot selections. That last
capability is what makes it more than a reporting query. The forecast-category
and slippage question is exactly a `transition` on forecast category between two
snapshots, cross-tabulated against a close-date movement measure, and it needs
no free SQL at all.

**Guarded SQL is the fallback for what that provably cannot express**, and its
use is recorded as such in the run record so the boundary is measurable rather
than assumed. The guard:

- parsed with `sqlglot`; a single `SELECT`, no CTE writing, no DDL, no DML
- no `ATTACH`, `COPY`, `INSTALL`, `LOAD`, `PRAGMA`, `SET`, no `read_csv`,
  `read_parquet`, or `glob`: no filesystem reachability at all
- table allowlist: the dataset's conformed view, nothing else
- **column allowlist derived from the active stance**, so under a prospective
  question the contaminated columns are not merely discouraged, they are
  unreferenceable
- read-only connection, enforced row limit, statement timeout, memory cap
- the result is registered as a `query_id` with its SQL and row hash, and every
  numeral in the answer must trace back to a cell of it (§8.4)

The stance-derived column allowlist is the load-bearing line. It means the
leakage guarantee holds on the ad-hoc path by construction, not by the model's
good behavior.

### 12.9 The tool surface

> **Superseded by §13.5**, which consolidates this table into eleven
> stance-bounded tools and makes the run tools terminal actions of planning.

Small on purpose. A wide surface invites improvisation.

| Tool | Purpose | Tier |
|---|---|---|
| `get_analyst_context()` | The tier-0 card | n/a |
| `get_business_definition(concept)` | What a concept means, how it is bound here, caveats | n/a |
| `list_columns(filter)` | Names by information class or pattern, for wide datasets | n/a |
| `inspect_column(name)` | Tier-1 profile plus classification | n/a |
| `inspect_values(name)` | Representative values, cardinality-capped | n/a |
| `inspect_relationships(a, b)` | Correlation or contingency, in SQL, stance-filtered | B |
| `identify_candidate_columns(concept)` | Candidates with evidence, for an unbound concept | n/a |
| `run_analysis_plan(plan)` | Path A | A |
| `run_investigation_plan(plan)` | Path B, primary | B |
| `run_safe_sql(sql)` | Path B, fallback, guarded | B |
| `inspect_sample_rows(n)` | A capped sample, stance-filtered, text columns excluded | n/a |
| `request_clarification(...)` | Ask instead of assume | n/a |

`get_business_definition` is the anti-hallucination tool for *semantics*, the
counterpart to what the plan gate does for arithmetic. The model asks what a
concept means rather than assuming, and the answer is written by a human once,
not invented per conversation.

`compare_periods` is deliberately absent: it is `run_analysis_plan` over two
snapshot selections, and a separate tool would be a second way to express the
same thing, which is how two implementations of one semantic drift apart.

**On sample rows.** The direction says not to put raw datasets into the LLM
context, and a bounded sample is not the dataset. Five rows is often what makes
a column's shape legible when a profile cannot. `inspect_sample_rows` is
therefore included, under the same constraints as everything else: a hard row
cap, the stance-derived column allowlist, and free-text columns excluded so no
CRM narrative reaches the context through the side door. `inspect_values`
returns distributions rather than records, and remains the preferred tool.
Datasets are never loaded into context; samples are.

### 12.10 Multi-tenant column differences

| Case | Handling |
|---|---|
| Missing column | The concept is `unavailable`. Metrics needing it are hidden from the catalog, so they cannot be planned |
| Renamed column | Bound by config, registry, or user confirmation. Never by name similarity alone |
| Extra columns | Discovered, classified by family or left unclassified, profiled, inspectable |
| Duplicate representations | `ambiguous`. The agent asks. Never resolved by picking the higher-scoring name |
| Tenant-specific columns | Discovered features. Usable in path B with disclosure, per §12.11, or under the unclassified rule below when nothing places them |
| Custom metrics | A tenant may declare a metric in config; it enters the registry as tenant-scoped, tier A only if its definition is written |
| Segment-scoped amounts | A `many`-cardinality concept. Unpivoting requires a declaration; the sum agreement test is evidence for proposing it (§12.3) |
| Different status vocabularies | Configured value mapping (§5.13), unchanged |
| Different stage vocabularies | Observed, reported, never authoritative (§5.13) |

The single rule behind the table: **two columns are equated only by a
declaration, never by resemblance.**

### 12.11 Precomputed features

Five classes, as the direction requires. The first four are expressible in the
existing contract; the fifth already is.

| Class | Current encoding | Admissible in path A | Admissible in path B |
|---|---|---|---|
| Directly observable | `as_of_fact` + `direct` | Yes | Yes |
| Reproducible from snapshots | `recompute` | Yes, recomputed; the supplied column reconciles | Yes |
| Precomputed, lineage documented | `use_with_proof`, `lineage.confirmed` | Yes | Yes |
| Precomputed, lineage unknown | `use_with_proof`, lineage unconfirmed | **Only when the reference population and window are not load-bearing for correctness** | Yes, with the lineage status disclosed |
| Outcome dependent | `future_contaminated` | Retrospective only | Retrospective only |

**Unclassified columns are a sixth case, and the table above does not cover
them.** A column no registry or family places has no classification at all, so
it also has no sentinel declaration. The stance-derived allowlist protects
against leakage, but nothing protects against `-999999` entering a mean and
producing a wildly wrong number with no error. The rule:

> An unclassified column is **inspectable always**, admissible as a **filter or
> group-by** in path B with disclosure, and **never admissible as a measure**
> while it remains unclassified. Classifying it is the path to using it as a
> measure.

This keeps a newly added tenant column visible and useful without letting an
aggregate run over values nobody has characterized.

The fourth row is where the 23 rep aggregates sit, and it is the disposition
already implemented: available as dataset fields, carrying the caveat, with
`rep_aggregate_lineage` recorded as an open question. They are not silently
upgraded to trusted metrics, and they are not hidden either.

"Load-bearing for correctness" is the operative test and it is decided per
metric, not per column. A win-rate metric built on a precomputed win rate of
unknown window is circular and is refused. Reporting the value of
`rep_win_rate` because the user asked for that field is not circular, and is
answered with the lineage stated.

The user may always ask for a named field directly. That request carries its own
authority; the answer discloses what is unknown about it.

### 12.12 Status and stage

Unchanged from §5.13 and restated because it survives the direction change.
Authoritative status is configuration. The fallback chain remains authoritative
column, then observed flags, then stage inference, with stage inference
non-authoritative and flagged. The production status column is still unresolved
and is not guessed.

Multi-tenancy makes this more important: stage vocabularies differ per tenant,
and a keyword rule tuned to one tenant's labels is actively wrong on another's.
`6 - Order Placed` reads as open to a keyword matcher.

### 12.13 Text columns

Discovered by profile shape rather than by a fixed list, since the fields differ
per tenant. Catalogued and profiled with length statistics only, no content.
Still no embeddings, no retrieval, no vector search. §5.11 stands unchanged;
what changes is that the field list is discovered rather than declared.

### 12.14 Required contract changes

New, none of them built in this milestone:

| Contract | Purpose |
|---|---|
| `BusinessConcept` | The stable ontology of §12.1 |
| `ConceptBinding`, `BindingEvidence`, `BindingStatus` | §12.2 |
| `AgreementTest`, `AgreementResult` | §12.3 |
| `TenantProfile` | Per-tenant declarations: bindings, status config, calendar, documentation pointers |
| `AnalystContext` | The tier-0 card, with a token budget enforced at build time |
| `InvestigationPlan` | Path B's plan type |
| `TrustTier` | A, B, C, computed from inputs |

Changed:

- `AnalysisPlan` gains an analysis stance (Appendix C.1), which now also drives
  the ad-hoc column allowlist.
- `ResultSet` carries a trust tier and the disclosures behind it.
- `MetricDefinition.required_columns` becomes required *concepts*, which is what
  lets one definition serve tenants with different headers.

Unchanged, deliberately: `ColumnClassification`, `DatasetRegistry`,
`ColumnProfile`, and the separation between classification and profile. The
registry already answers "what does this column mean and when may it be read"
per dataset, which is exactly what the binding layer needs underneath it.

### 12.15 The close-date blocker

**Why it is absent.** The export was built for model training, where a horizon
in days is the useful form. It carries `days_to_close` rather than a date, plus
`close_date_qtr`, `CD_in_qtr`, `eoq_close_diff`, `close_date_push_count`,
`cd_movement`, and `deal_first_seen_close_date`. The concept plainly exists in
the export; only the column is missing.

**Where the information is.** `days_to_close`, confirmed by the project owner as
the snapshot's own expected close date minus `as_of`, not derived from
`terminal_date`. So `expected_close_date = as_of + days_to_close`.

**Is the reconstruction deterministic?** Arithmetically yes, *conditionally on
facts not yet in evidence*. It is a day count added to a date. It stops being
deterministic if the column is fractional, sentinel-bearing, clamped, or
computed against a different date for closed rows.

**Evidence still needed**, and not to be assumed:

1. Is `days_to_close` an integer day count, or a rounded or fractional value?
2. What does it hold on closed, deleted, and null-close-date rows: the realized
   horizon, zero, null, or a sentinel?
3. Can it be negative, for a close date already past at `as_of`, or is it
   clamped at zero?
4. Does the fiscal quarter of `as_of + days_to_close` equal `close_date_qtr` on
   every row?
5. What exactly is `CD_in_qtr`, and does it agree?
6. What is `eoq_close_diff`'s sign convention, and does it equal the
   reconstructed date minus quarter end?
7. Is `deal_first_seen_close_date` a real date? If so it is a direct check of
   the reconstruction at each opportunity's first snapshot.
8. Which month does the tenant's fiscal year start? Questions 4 and 6 cannot be
   evaluated without it.

**The proposed implementation.** A reconstruction is a declared derivation with
mandatory agreement tests (§12.3), not a silent cast. At ingestion the
derivation is computed, each agreement test is evaluated, and the disagreeing
row counts are recorded in the schema. If a test fails, the binding is not made:
`expected_close_date` stays `unavailable` with the failure recorded, and every
metric needing it is hidden. A wrong date is far worse than a missing one.

**Two changes are needed and they are not the same change.** Reconstruction
makes this export answerable for close-date questions. Separately, requiring
close date, stage, and amount for *ingestion at all* is wrong for multi-tenancy:
the true ingestion minimum is the grain, and everything else should be enforced
at metric resolution by `check_requirements`, which already exists and already
returns exactly that answer. Demoting the requirement without implementing
reconstruction would remove the forcing function that gets the eight questions
above answered, so the demotion should land with or after reconstruction, never
instead of it.

### 12.16 Money across tenants

**The adopted policy**, per the direction: preserve the source representation at
rest, and cast explicitly to `DECIMAL(18,2)` at semantic measure resolution
before any monetary aggregation. Binary floating-point arithmetic never
silently defines a monetary result. This is a deliberate boundary conversion,
declared in the measure, visible in the compiled SQL, and not a heuristic.
Broad decimal type detection is explicitly not adopted: it would also catch
two-decimal ratios and would be guessing about meaning from shape.

**An open decision this exposes.** `CLAUDE.md` currently says money never passes
through a float "not in profiling, not in serialization, not in a test." The
canonical `amount` column honors that: it is `DECIMAL` and the profiler computes
no mean for it. Discovered money columns do not. `terminal_amount` and
`rep_avg_deal_amount` are typed `DOUBLE` and the profile currently reports a
float mean, median, and standard deviation for them.

Measure resolution is not where that happens, so the §13 policy does not cover
it. Three options, and this needs a decision rather than a quiet rewording:

1. Accept it. Profile statistics are descriptive metadata, never an answer, and
   they are already separated from classification. Document the exception.
2. Suppress summary statistics for columns a tenant declares monetary, matching
   the `DECIMAL` treatment. Requires the tenant to declare them.
3. Detect fixed-scale decimals at ingestion. Rejected above for guessing.

Option 1 is the smallest change and is defensible, because no answer is ever
computed from the profile. It is recorded here as an exception rather than
folded into the rule, so that the rule keeps meaning what it says.

**Resolved.** The exception is closed rather than accepted. Structural numeric
profiling and monetary analytical measures are now separate things:

* **Profiling** always reports counts and exact extremes. It withholds the float
  mean, median and deviation for a column declared monetary and for a column whose
  monetary status is *unknown* (nothing classifies it), and says why through
  `summary_withheld_reason`. Only a column someone classified as non-monetary keeps
  them. Nothing infers money from numeric shape.
* **A tenant's declaration reaches the profiler.** Profiling runs before any
  concept binding exists, so a declaration has to be passed in: a tenant naming a
  column as `amount` or `terminal_amount` withholds its float summary, resolved
  from the tenant's own header to the conformed column. A declaration can only add
  money, never remove it.
* **A measure is converted explicitly.** `monetary_measure_sql` casts a `DOUBLE` or
  integer column to `DECIMAL(18,2)` in visible SQL before any aggregate; a `DECIMAL`
  passes through; any other type is refused. `measure_monetary_conversion` reports
  the rows the cast would round or cannot hold, so a lossy conversion is disclosed.
  Source representation is preserved at rest.

### 12.17 What the current implementation needs before the semantic milestone

Reviewed against the 340 tests. **No existing test contradicts this design, and
none needs to change now.** The pieces this direction depends on are already
built and already tested: per-dataset classification separate from observation,
fail-closed treatment of unclassified columns, `check_requirements` as a
metric-agnostic availability answer, per-column sentinels, configurable status,
and a mapper that refuses to satisfy a required column with a guess.

Ordered work for the next milestone:

1. Concepts and bindings, over the existing registry. New contracts only.
2. The understanding layer and the analyst context card, with its token budget
   enforced by test at a realistic width, not a lax one.
3. Close-date reconstruction with agreement tests, once the §12.15 evidence
   arrives.
4. Requirement demotion to grain-only, with reconstruction. Six tests assert
   today's behavior and would change, none of them accidentally:
   `test_missing_required_column_hard_fails` (`test_ingest.py`);
   `test_missing_required_columns_are_listed` and
   `test_assert_mappable_fails_loudly_with_a_typed_error` (`test_mapping.py`);
   `test_registry_mapping_reports_close_date_missing_when_the_export_lacks_it`,
   `test_the_production_export_fails_loudly_and_explains_reconstruction` and
   `test_without_a_registry_the_same_file_is_refused_for_the_fuzzy_guess`
   (`test_mapping_safety.py`). The last two are the ones that matter: they pin
   the two distinct refusals, a documented reconstruction and a fuzzy guess, and
   demotion must keep the second refusal exactly as it is.
5. Metric definitions declaring concepts rather than columns.
6. `InvestigationPlan` and the stance-derived column allowlist.
7. Guarded SQL last, so the plan surface is built first rather than skipped.

One caveat that outlasts this milestone: every classification in this repository
was assigned from column names and the project owner's written confirmations.
Not one row of real data has been inspected, and the fixture is synthetic.
Across tenants that gap widens, which is the reason confirmation is a status in
§12.2 rather than a formality.

---

### 12.18 Implementation status

| Area | Section | State |
|---|---|---|
| Analytical concepts | 12.1 | `contracts/concepts.py`. 15 concepts, none naming a physical column |
| Concept bindings and evidence | 12.2 | `contracts/binding.py`, `data/binding.py` |
| Tenant declarations | 12.14 | `contracts/tenant.py` |
| Agreement tests | 12.3 | `contracts/agreement.py`, `data/agreement.py` |
| Dataset understanding | 12.4 | `data/understanding.py` |
| Tiered context card | 12.5 | `contracts/context.py`, budget enforced by test |
| Close-date reconstruction | 12.15 | `data/reconstruct.py`, wired through conform |
| Grain-only ingestion | 12.15 | `REQUIRED_COLUMNS` is the grain |
| Monetary policy | 12.16 | `MonetaryStatus`, honoured by the profiler |
| Tier-2 bounded row sample | 12.5 | `data/understanding.py:sample_rows`, not an agent tool |
| Excel serial dates | 12.19 | `conform.date_sql`, `data/dates.py`, declared per source column |
| Reconstruction gating | 12.19 | Required agreement tests must pass; persisted as `agreement.json` |
| Production probe | 12.19 | `data/probe.py`, `scripts/probe_production.py`, aggregates only |
| Monetary measure conversion | 12.16 | `data/money.py`; tenant declarations reach the profiler |
| Fiscal calendar and periods | 5.6 | `semantic/calendar.py`; start month declared, never inferred |
| Snapshot selection, all six rules | 5.1 | `semantic/snapshots.py`, with drift metadata on every result |
| Concept resolution under a stance | 12.1, 5.8 | `semantic/resolver.py`, the only concept-to-column boundary |
| Cohort and trace, six states | 5.4 | `semantic/cohort.py`; resolved from status, not a terminal column |
| Transitions | 5.4 | `semantic/transitions.py`; recomputed, never read from a counter |
| Pipeline bridge and its invariant | 5.2 | `semantic/bridge.py`; nine independent terms |
| Rate primitive | 5.4 | `semantic/rate.py`; both components always exposed |
| Metric registry, 11 metrics | 5.5, 12.1 | `semantic/metrics.py`; gated on concepts, not columns |
| Plan gate | 8.2 | `semantic/gate.py`, `contracts/rejection.py`; structured rejections |
| Deterministic compiler | 2.1, 8.1 | `semantic/compiler.py`; refuses an unvalidated plan |
| Execution and sanity checks | 8.3 | `semantic/execute.py`; bridge and rate invariants enforced |
| Analysis stance and knowledge cutoff | C.1, C.2 | `AnalysisSpec.stance`, `knowledge_cutoff`; prospective by default |
| Trust model A/B/C | 12.7 | Computed from inputs in `ConceptResolver` and `execute.py` |
| Two analytical paths | 12.6 | Path A built; path B **not built** |
| Investigation plan, guarded SQL | 12.8 | **Not built** |
| Tool surface | 12.9 | **Not built** |
| Provenance scanner | 8.4 | **Not built**; `ResultSet` carries the metadata it will need |
| Reconstruction validity versus reconciliation | 13.1 | `AgreementRole`, `ReconstructionVerdict`, `AgreementReport.assess`; supersedes 12.19's all-required gating |
| Purpose-scoped usage grants from tenant declarations | 13.2 | `UsageGrant`, `declaration_conflict`, resolver purposes |
| Structured trust | 13.10 | `TrustFactor`, `TrustAssessment`, `FACTOR_CEILINGS`, `semantic/trust.py` |
| Investigation plan, gate, compiler | 13.8 | `contracts/investigation.py`, `semantic/investigation.py` |
| Session plan edits and carry-forward | 13.13 | `contracts/session.py`, `session/edits.py`, `session/state.py` |
| Renderer and provenance scanner | 13.14, 13.15 | `contracts/answer.py`, `validation/rendering.py`, `validation/provenance.py` |
| Deterministic tool surface | 13.5 | `contracts/tools.py`, `agent/tools/surface.py` |
| Planner context builder | 13.4 | `agent/context.py`, budget enforced by test |
| Guarded SQL | 13.9 | **Pending by design** |
| Planner, responder, orchestrator, API | 13.3, 13.6, 13.14, 13.19 | **Not built** |

Three things were learned by building it.

**A fuzzy match now never conforms, for any column.** Demoting the close date
from required to optional would otherwise have *weakened* the anti-guessing
rule: `close_date_qtr` would have bound silently as an optional column instead
of being refused as a required one. A fuzzy candidate is now a proposal in
`MappingProposal.fuzzy_candidates`, surfaced as an inferred binding, and never
applied. The source column survives as a discovered column rather than being
consumed by the guess.

**The grain is confirmed by assertion, not by name.** `opportunity_id` and
`snapshot_date` carry `EvidenceKind.GRAIN_ASSERTION`, which confirms because
ingestion hard-fails unless `(as_of, opp_id)` is unique across every row. That
is a proof about behaviour rather than a claim about a header, which is why it
is the one verified confirming kind. Every other concept still needs a
declaration.

**Grain-only ingestion exposed two latent assumptions**, both fixed: the
profiler's quality checks read `amount` and `close_date` unconditionally, and
the stage vocabulary read `stage`. A tenant exporting none of the three now
profiles cleanly instead of raising a binder error.

### 12.19 Production-data hardening

> **Gating revised by §13.1, and implemented.** Ancillary-field tests
> (`eoq_close_diff`, `CD_in_qtr`, `close_date_qtr`) are reconciliation checks
> unless their semantics are declared, and the sibling export's close date is now
> available with warnings rather than withheld. The findings below are unchanged.

This section records what was built to make real exports ingestible and what
probing one of them showed. Two real exports exist. **Only one was reachable**:
the sibling tenant's parquet is local and was probed; the target export lives in
S3, no AWS credentials were available to the session, and it has **not** been
probed. Everything measured below is from the sibling, and the target export's
behaviour is unverified on every point.

#### Excel serial dates

Both exports store date fields as Excel serial numbers. Conversion is by
**declaration, never detection**: `ColumnRegistry.date_encodings` names the source
columns, keyed by source header, and a caller may add or cancel declarations. A
numeric column is never a date because its values look like one, and type
detection deliberately cannot find serials.

| Rule | Behaviour |
|---|---|
| Epoch | `1899-12-30`, a named constant. Not `1900-01-01`: Excel counts a nonexistent 1900-02-29 as serial 60, so from serial 61 on the base plus the serial is the true date |
| Valid range | Serial 61 (1900-03-01) to 2,958,465 (9999-12-31). Serial 60 and below are ambiguous and refused |
| Fractions | `FLOOR`, never `CAST`: DuckDB rounds on cast, so `45973.6` would land a day late. Time of day is discarded because the conformed type is `DATE`, and the count of truncated rows is recorded |
| ISO text | Accepted in a serial column, so a mixed column converts. An 8-digit integer such as `20250101` is out of range and refused, not guessed |
| Invalid value | A hard error naming the column and the offending values, raised before anything is written. It is not counted as a cast failure and not turned into a null |
| Provenance | `DatasetSchema.date_conversions`: serial, ISO, null and truncated row counts and the date range, per column |

The conversion is one SQL expression used at every site that produces a date.
Patching only the canonical cast would have left the close-date reconstruction
reading a serial `as_of` as NULL, so every rebuilt date would be NULL and the
concept would have been withheld for a reason unrelated to the data.

Five columns are declared in the export registry. Only `as_of` has been decoded
against real rows; the rest come from the owner's confirmation and are recorded
as `date_encoding_declarations`. A wrong declaration fails ingestion loudly, so
the risk is a blocked ingest. The 1904 date system would be wrong silently by
1462 days, but decoding `as_of` under it gives a date in 2029, so the 1900 system
is the only one consistent with the data.

#### Reconstruction is gated on required tests

`expected_close_date` is `as_of + days_to_close`, unchanged. It is available only
if every **required** agreement test ran and passed. A failure withholds it, and
so does a test that could not run: skipped is not passed, and an empty report is
not a pass. The old rule withheld only on failure, so a rebuilt column with every
corroborating column absent stayed confirmed with zero evidence.

All four close-date tests are required. The reason for a withhold is on the
binding, naming each unmet test. The report is persisted as `agreement.json`
beside the dataset, so a failure survives a reload; it records the fiscal calendar
it assumed and is re-evaluated if that changes.

`days_to_close` must be a whole number. A fractional value yields a NULL close
date, not a rounded one, because DuckDB would round `2.6` to 3 and move the date a
day with no signal. That does not change the derivation; it refuses to guess what
a fractional horizon means.

#### The probe

`data/probe.py` (script: `scripts/probe_production.py`) reads an export and
returns **aggregates only**: counts, quantiles, and category labels for
low-cardinality status candidates, flags and quarter-label shapes. Identifiers are
cardinality and null count, never values; agreement samples are switched off; it
writes nothing. A test asserts that no identifier from the data appears in the
rendered report or its JSON. S3 credentials resolve through the standard AWS
chain and are never an argument.

Status fields it finds are **candidates only**. No name match becomes a mapping.

#### What the sibling export showed

| Question | Finding |
|---|---|
| Shape of `days_to_close` | An integer on every row, never null, never fractional. Negative on two thirds of rows, so it is not clamped |
| Closed deals | On `Closed Won` rows it is always negative, roughly 26 to 44 days: on a closed deal it reads as a realized close date, not a forecast |
| Deleted rows | Unknown: this export has no deletion marker |
| Do reconstructed dates move? | Yes, but for only 129 of 18,333 opportunities across three snapshots. A movement check on pushed deals could not run: `close_date_push_count` is absent |
| `CD_in_qtr` vs the calendar | Agrees on 54,762 of 54,999 rows. All 237 disagreements sit on a quarter's first day (78) or last day (159), none in the interior, all in one direction |
| `eoq_close_diff` | Equals `days_to_eoq - days_to_close` plus an offset of exactly 0, +1 or -1 on every row (27,481 / 20,742 / 6,776) |
| Flags | `CD_in_qtr` and `CD_in_past` are 0 and 100, not 0 and 1 |
| Quarter labels, fiscal start | Not evaluable: no stamped label column, and the fiscal year start is still the configured default |

Two of these matter together. The direction of `eoq_close_diff` is settled, and
both it and `CD_in_qtr` are wrong by a day at the edges. The sibling's `as_of`
carries an 11:00 time of day, which is consistent with a day-level rounding effect
in the stamped columns, but that is a hypothesis: nothing here distinguishes it
from a real convention, or from the reconstruction being a day off on those rows.

**The exact agreement tests therefore fail on this export, and they stay exact.**
No tolerance was added to make them pass. Whether to allow an explicit one-day
tolerance, and on which tests, is a decision for the project owner. Until then the
close date is withheld on any export that behaves like this one.

#### Still open

`days_to_close_edge_cases`, `eoq_close_diff_convention`,
`quarter_boundary_day_convention`, `fiscal_year_start_month` and
`date_encoding_declarations` are `UnresolvedItem` records attached to the columns
they qualify, so they travel as caveats on the binding and appear in the card. All
of them are unverified on the target export.

One structural caveat about the movement test: `close_date_push_count` is
cumulative since creation, so a deal pushed *before* the window can legitimately
show a still close date. The test asserts every pushed opportunity moved, which is
the specification as written, and may fail on real data for that reason. The
probe reports how the counter's increases line up with date movement so the choice
can be made on evidence.

---

### 12.20 Semantic engine v1

The deterministic analytical engine: a hand-written `AnalysisPlan` in, a
validated `ResultSet` out, with no LLM anywhere in the path.

#### The shape

```
AnalysisPlan
  -> gate.validate_plan      concepts, columns, stance, cutoff, snapshot coverage
  -> compiler.compile_plan   concept -> column, DECIMAL boundary, as_of ceiling
  -> execute.run_plan        run, check invariants, compute the trust tier
  = ResultSet
```

Each arrow refuses what the previous one did not establish. The compiler takes
a `GateOutcome`, not a plan, so an unvalidated plan cannot be compiled by
mistake rather than merely by convention.

#### What the stance actually does

`AnalysisStance` is the field §5.8 enforcement hangs on, and enforcement is
structural in two independent places:

* `ConceptResolver.permitted_columns` omits a future-contaminated column under
  a prospective stance, so it is unreachable rather than discouraged;
* every compiled query carries an `as_of` ceiling, so the leakage guarantee is
  in the emitted SQL rather than in the planner's good behaviour.

Retrospective analysis *may* read an outcome column, and doing so is disclosed
on the result and caps the trust tier at B. Quarantined and unclassified
columns are refused under both stances: hindsight is not a licence to read a
column nobody could define.

#### Money never passes through a float

DuckDB's `/` returns DOUBLE for every operand type, DECIMAL included, so
`SUM(amount) / COUNT(*)` silently turns an exact total into a float. Every
division in the engine goes through `sql.exact_divide`, which scales, uses
integer division, and scales back, staying DECIMAL end to end. `AVG` is never
emitted. A monetary average truncates at cent scale, which is deterministic and
documented rather than platform-dependent.

#### The bridge stays exhaustive without becoming a residual

Every opportunity in the union of the two pipeline sets is a survivor, an
entrant, or a leaver, and each position is classified by its own predicate.
`other_removed` is the leavers no other predicate claimed, computed directly
rather than as the arithmetic remainder, so the identity remains a genuine
check on nine separately measured quantities. `pulled_in` and `other_removed`
each report their composition, because both names promise more than their
predicate delivers.

#### Two ontology changes this milestone forced

**`created_date` became a concept.** Ambiguity 1 of §5.3 defaults "created in
period" to a creation date, and the ontology had no concept for one. Without it
the default was unreachable and the bridge could only ever use first-seen.

**`terminal_outcome` no longer gates `COHORT_TRACE`.** A trace resolves each
member's fate from the status concept across the snapshots it already has, and
§5.12 holds that a terminal column is the reconciliation target for a trace and
never an input to one. Requiring it made the primitive unavailable on exactly
the datasets it exists to serve, including the sibling export. Forecast
accuracy genuinely needs the realized outcome and keeps the requirement.

#### A declaration confirms a binding; it does not lift a quarantine

Building scenario 15 of the golden tests turned this up. A tenant declaring
`enterprise_amount` for the amount concept produces a **confirmed** binding, and
the plan is still refused, because a discovered column no registry places is
unclassified and therefore quarantined. The two gates answer different
questions: a declaration says what a column *means*, and a classification says
when its values are *knowable*. Only the second can release a quarantine.

That is the correct fail-closed direction and it is also a real limitation: a
tenant cannot use a custom measure column, such as the segment-scoped amount of
12.10, until someone classifies it. Releasing a quarantine from a tenant
declaration would mean treating "this column is our amount" as also asserting
"and its values were knowable at their own snapshot", which is a different and
unevidenced claim. Whether a tenant profile should be able to carry that second
assertion explicitly is an open question, not something to infer from the first.

#### What is still assumed

`pipeline_coverage` is measured against realized bookings, not against a quota,
because no quota or target concept exists in the ontology. A tenant wanting
coverage against quota has to supply a target first. This is a stated
limitation on the metric, not a silent substitution.

## 13. The LLM Layer

> **Status: deterministic scaffolding and the planner implemented.** §13.21
> and §13.22 record the deterministic layer; §13.23 records the planner runtime
> and its evaluation. The responder and the end-to-end orchestrator remain
> design only, and guarded SQL (§13.9) is deliberately pending. It builds on
> the deterministic engine of §12.20 and supersedes §6.1, §6.3, §6.6, §6.7 and
> §12.9 where they differ. It changes one existing rule, reconstruction gating
> (§13.1), and one existing behaviour, tenant declarations against quarantine
> (§13.2). Both are reviewed first because the LLM layer inherits their answers.

The LLM layer has one job: turn a question into a *typed plan* and a set of
*typed results* into a readable answer, without ever originating a number,
choosing a trust tier, or reading information the analysis stance forbids.
Everything else is deterministic code that already exists or is specified here.

```
                ┌──────────────────── LLM ────────────────────┐
question ──► Context ──► Planner ──(tools: inspect_*)──► PlannerDecision
                                                           │
       ┌────────────────────┬─────────────────────┬────────┴──────────┐
       ▼                    ▼                     ▼                   ▼
  AnalysisPlan      InvestigationPlan     ClarificationRequest   Unanswerable
       │                    │                     │                   │
   Plan gate       Investigation gate             │                   │
       │                    │                     │                   │
   Compiler        Investigation compiler         │                   │
       └────────┬───────────┘                     │                   │
            DuckDB (read-only)                    │                   │
                │                                 │                   │
         Sanity checks ──► TrustAssessment        │                   │
                │                                 │                   │
                └──────────► Responder (LLM) ◄────┴───────────────────┘
                                  │ AnswerDraft with {{tokens}}
                             Renderer ──► Provenance scan ──► Answer
```

Only two boxes call a model: the **Planner** and the **Responder**. The line
from §2 still localizes every bug: neither LLM call contains arithmetic, and
nothing between them contains an LLM.

---

### 13.1 Close-date gating review (prerequisite)

**The problem.** §12.19 made reconstruction availability depend on every
agreement test passing, and all four close-date tests are `required`. Two of
them check the reconstructed date against *ancillary derived fields*,
`eoq_close_diff` and `CD_in_qtr`, whose own definitions have never been
established. On the sibling export both fail in a structured way:

| Field | Disagreeing rows | Shape |
|---|---|---|
| `eoq_close_diff` | 27,518 of 54,999 | offset from `days_to_eoq − days_to_close`: +0 on 27,481, +1 on 20,742, −1 on 6,776 |
| `CD_in_qtr` | 237 of 54,999 | all "flag says out, calendar says in"; 78 on a quarter's first day, 159 on its last, 0 interior |

Neither pattern says the reconstructed date is wrong. Both say the ancillary
field follows a convention nobody has written down: a day-count that is
inclusive on one end, a quarter membership that excludes a boundary day, or a
fiscal calendar that is itself unresolved for this tenant. Requiring an
undocumented field to agree makes the *reconstruction's* validity hostage to
the *field's* semantics. That inverts the evidence: the thing we understand is
being judged by the thing we do not.

**The redesign.** Agreement tests get a role, and the role decides what a
failure means. A role is assigned by declaration, never inferred from how the
disagreement looks.

| Role | What it checks | On pass | On failure | On skip |
|---|---|---|---|---|
| `VALIDITY` | The derivation itself, or a field whose semantics are **declared** | Corroborates | **Withhold**: contradictory evidence | Unverified: caveat, tier ≤ B |
| `RECONCILIATION` | Agreement with an ancillary field whose semantics are **not** established | Corroborates | **Warn**: reconciliation warning, tier ≤ B, `UnresolvedItem` | Nothing |

Three outcomes for a reconstructed concept replace the current two:

```python
class ReconstructionVerdict(StrEnum):
    VALID          # declared derivation, every VALIDITY test passed
    VALID_WITH_WARNINGS
                   # declared derivation, no VALIDITY failure, but a VALIDITY
                   # test was unverifiable or a RECONCILIATION test failed
    CONTRADICTED   # a VALIDITY test failed: unavailable
    UNDECLARED     # no declaration of the derivation: unavailable
```

The rules, in order:

1. **A derivation must be declared.** `as_of + days_to_close` is available only
   because a declaration (the export registry, confirmed by the project owner)
   says `days_to_close` is the snapshot's own horizon. No declaration, no
   reconstruction. This preserves §12.19's refusal of vacuous truth: an
   unchecked reconstruction is not available merely because nothing contradicts
   it; it is available because someone accountable stated it.
2. **Structural checks are VALIDITY and always run.** `days_to_close` is a whole
   number; `as_of` decoded under its declared encoding; the result is a real
   calendar date. These are already enforced at ingestion (§12.19).
3. **The discriminating test is VALIDITY.** The movement test is the only check
   that separates "the snapshot's own close date" from "the terminal date". Its
   failure is contradiction. Its scope is already correct after §12.20: only
   adjacent snapshot pairs where the cumulative `close_date_push_count` actually
   increased inside the window. With no such pair, or no counter column, it is
   **unverified**, never passed and never failed.
4. **Ancillary-field tests default to RECONCILIATION.** `eoq_close_diff`,
   `CD_in_qtr` and `close_date_qtr` start as reconciliation checks. A tenant
   promotes one to VALIDITY by declaring its semantics, for example
   `eoq_close_diff = days_to_eoq − days_to_close, both inclusive`, as an
   `AgreementDeclaration` in the tenant profile. After promotion a disagreement
   is contradiction.
5. **Calendar-dependent tests inherit calendar uncertainty.** `CD_in_qtr` and
   `close_date_qtr` compare against fiscal quarters. While the tenant's fiscal
   year start is unresolved, those tests cannot be promoted to VALIDITY at all,
   because a disagreement could be the calendar and not the date.
6. **No tolerance.** A reconciliation test reports its exact disagreement count
   and its offset histogram as diagnostics. Nothing converts "+1 on 38% of rows"
   into a pass. The histogram is evidence for the owner deciding what the
   convention is, which is what turns it into a declaration.

**Effect on the sibling export.** `expected_close_date` moves from *withheld* to
*available with warnings*: the derivation is declared, structural checks pass,
the movement test is unverified (no counter column), and two reconciliation
tests fail with recorded diagnostics. Every result using the concept is capped
at tier B and names three reasons: movement unverified, `eoq_close_diff`
convention unresolved, `CD_in_qtr` convention unresolved. That is honest in
both directions: the date is usable, and nobody is told it was corroborated.

**What stays exactly as it is.** A failed movement test still withholds. A
declared-semantics field that disagrees still withholds. Fuzzy bindings still
never conform. The persisted `agreement.json` records the role and verdict of
every test.

---

### 13.2 Tenant declarations versus quarantine (prerequisite)

§12.20 found that a tenant declaring `enterprise_amount` for the amount concept
produces a confirmed binding while the plan is still refused, because the
column is discovered, unclassified, and quarantined. That is fail-closed, and it
also makes a tenant's custom measure permanently unusable, which §12.10 says
must be supported.

The fix separates three questions that the current code answers with one flag:

| Layer | Question | Established by | Scope |
|---|---|---|---|
| **Concept confirmation** | Does this column mean concept X? | A declaration (§12.2) | The binding |
| **Physical classification** | What is this column, and when are its values knowable? | Export registry, family pattern, or explicit classification | The column, for any use |
| **Usability grant** | May this column be read *for this purpose, under this stance*? | Derived from the two above | One purpose, one stance |

**The key observation.** A concept definition already carries timing. `amount`
is defined as "deal value as recorded in this snapshot"; `terminal_outcome` is
defined as retrospective. So a valid declaration that `enterprise_amount` *is*
the amount concept is also, necessarily, a statement that its values are
snapshot-state and knowable at their own `as_of`. The declaration is not just
naming a column; it is asserting the concept's contract about that column.

**The rule.** An explicit, valid tenant declaration issues a **purpose-scoped
usability grant**:

```python
class UsageGrant(BaseModel):
    column: str
    concept: BusinessConcept
    purposes: frozenset[ConceptRole]      # from the concept definition: MEASURE for amount
    availability: Availability            # inherited from the concept, not the column
    source: BindingEvidence               # the declaration that issued it
```

- Through the grant, the column is readable **only as that concept**: for
  `amount`, as the measure in amount-based metrics, through the monetary DECIMAL
  boundary. It is not a free dimension, filter, feature or `inspect_values`
  target. Generic reads still see an unclassified, quarantined column.
- The grant's availability is the concept's. A retrospective concept yields a
  grant usable only under a retrospective stance.
- The grant is recorded in the compilation metadata and adds a trust factor:
  "`amount` read from `enterprise_amount` under tenant declaration; the column
  is otherwise unclassified". The result can still be tier A if every other
  input is A: a declared binding on a declared column is the strongest evidence
  the system accepts.

**A declaration is valid only if all of these hold**, checked when the tenant
profile is loaded, not at query time:

1. The concept exists and the column exists in this dataset.
2. Cardinality is respected.
3. The storage type can hold the concept's semantic type: money needs a
   DECIMAL, DOUBLE or integer column; a date concept needs a DATE.
4. **No classification conflict.** If a registry has classified the column and
   that classification disagrees with the concept's timing, for example a
   column classified `future_contaminated` declared as the snapshot-state
   `amount`, the declaration is **rejected** and the conflict surfaced. A
   declaration can release an *unknown*; it cannot overrule a *known*.
5. The declaration names its source (who, when), which is persisted.

**What does not change.** An inferred binding issues no grant. A fuzzy candidate
issues no grant. An unclassified column reached by name, without a concept, stays
quarantined under every stance. Hindsight still does not license reading a column
nobody defined. Only the narrow path "an accountable person said this column is
concept X, and X's contract says when it is knowable" is opened.

A tenant who wants the column usable *generically*, as a dimension or feature,
declares a classification for it (`TenantProfile.column_classifications`). That
is a separate, explicit statement and is validated the same way.

---

### 13.3 Orchestrator

A hand-written state machine, as §6.2 argued; no agent framework. The
orchestrator owns every transition, every budget, every retry, and every call
into deterministic code. The model is invoked at exactly two states.

```
TurnState:
  RESOLVE_CONTEXT     session + dataset → PlannerContext              (no LLM)
  PLAN                planner loop: inspect_* tools → PlannerDecision (LLM)
  GATE                AnalysisPlan / InvestigationPlan validation     (no LLM)
    └─ rejected → PLAN with structured rejections (≤ 2 repairs)
  EXECUTE             compile, run, sanity-check                      (no LLM)
    └─ empty/ill-posed → optional PLAN round 2 (≤ 1, same budget)
  ASSESS              TrustAssessment per result, then per answer     (no LLM)
  RESPOND             responder → AnswerDraft                         (LLM)
  RENDER              token substitution, formatting, disclosures     (no LLM)
  VERIFY              provenance scan (≤ 1 regeneration)              (no LLM)
  EMIT                Answer + RunRecord
```

**Why the planner does not execute its own plans.** `run_analysis_plan` and
`run_investigation` are *terminal* actions of the planning loop: calling one
ends planning and hands the plan to the deterministic pipeline. The planner
never sees a result mid-plan and then "adjusts" it. This keeps the plan the
single auditable artifact of intent, and stops a model from iterating on queries
until a number looks plausible. The one permitted second round (§13.16) exists
for a structurally failed result, such as an empty population, not for a result
the model did not like.

**Terminal outcomes.** Every turn ends in exactly one of: an answer, a
clarification request, an abstention, or a failure with a stated reason. There
is no fifth outcome, and "best effort" is not one of the four.

---

### 13.4 Context builder

The default context is a projection, built deterministically per turn and
budgeted in tokens. It reuses `AnalystContext` (§12.5) unchanged and adds three
blocks.

| Block | Contents | Budget | Cache |
|---|---|---|---|
| System prompt | Role, rules, output contracts, the stance doctrine | ~1.5k | Static, cached |
| Tool definitions | The eleven tools of §13.5 | ~2k | Static, cached |
| Tier-0 card | Dataset identity, grain, snapshot range, fiscal calendar **and whether it is resolved**, concepts with bindings, statuses, information timing, reconstruction verdicts, major quality warnings, column index or per-class counts, open questions | ≤ 2k (enforced) | Per dataset, cached |
| Metric catalog | Available metrics under the session stance: name, one-line definition, required concepts, default snapshot rule, ambiguity notes by id | ≤ 1k | Per dataset and stance, cached |
| Session state | The active plan in compact form, carried filters, resolved entities, last result handles | ≤ 1k | Volatile |
| Question | The user's words | — | Volatile, last |

**Never in the default context:** rows, the full profile, text-field contents,
distinct-value lists, agreement samples. Tier 1 (one column's profile) and
tier 2 (bounded samples) arrive only through tools, and only under the stance.

**Tenant-controlled strings are data.** Column names, category values and
status labels come from the tenant's file and can contain anything, including
text shaped like instructions. The context builder renders them inside
delimited, typed structures (JSON fields, not prose), and the system prompt
states that dataset content is never an instruction. Free-text columns are
never rendered at all (§5.11).

---

### 13.5 Tool surface

Eleven tools. Each returns a Pydantic model, never prose; each is
stance-parameterized, where the stance is the session's and cannot be widened
by the model mid-turn; and each numeric output is registered, so that any
number the model later wants to cite has a provenance handle.

| Tool | Returns | Stance behaviour | Supersedes |
|---|---|---|---|
| `inspect_dataset(class_filter?, name_pattern?)` | `DatasetView`: tier-0 card, or a filtered column listing for wide datasets | Contaminated columns listed with their class, never with values | `get_analyst_context`, `list_columns` |
| `inspect_concept(concept)` | `ConceptView`: definition, binding, evidence, caveats, reconstruction verdict; rival candidates when ambiguous or unbound | Retrospective concepts marked unusable under prospective | `get_business_definition`, `identify_candidate_columns` |
| `inspect_column(name)` | `ColumnView`: classification, usability per stance and purpose, grants, profile summary | Under prospective, a non-knowable column returns classification and **no** profile values | — |
| `inspect_values(name, top_k≤50)` | `ValueDistribution`: counts, cardinality, nulls, bounded by the knowledge horizon | Computed on rows with `as_of ≤ horizon` when a horizon is set | — |
| `inspect_relationship(a, b, population?)` | `RelationshipEvidence`: contingency or rank correlation with support counts, registered as a tier-B result | Both columns must be permitted; horizon applied | `inspect_relationships` |
| `inspect_sample_rows(n≤20, columns?)` | `SampleRows`, text columns excluded, identifiers masked unless requested and permitted | Permitted columns only; `as_of ≤ horizon` | — |
| `list_available_metrics()` | `MetricCatalog` under the session stance, with each unavailable metric and its missing concept | — | `list_metrics`, `describe_metric` |
| `run_analysis_plan(plan)` | *Terminal.* Submits an `AnalysisPlan` | Gate enforces | `run_plan` |
| `run_investigation(plan)` | *Terminal.* Submits an `InvestigationPlan` | Investigation gate enforces | `run_investigation_plan` |
| `run_guarded_sql(sql, justification)` | *Terminal, disabled by default.* §13.9 | View-enforced | `run_sql`, `run_safe_sql` |
| `request_clarification(request)` | *Terminal.* A `ClarificationRequest` | — | — |

Abstention is a fourth terminal action, `declare_unanswerable(reason)`, emitted
as a planner decision rather than a tool (§13.6).

**Two details that carry most of the safety.**

- **Horizon-bounded profiles.** The persisted profile describes every snapshot,
  including ones after a prospective question's horizon. A stage distribution
  over all snapshots already contains how many deals eventually closed won. So
  under a prospective stance with a horizon, `inspect_values` recomputes on
  `as_of ≤ horizon` instead of reading the stored profile. Without that, the
  model could learn the future through a profiling tool and write it into prose
  that never touches a query.
- **Registered evidence.** `inspect_values` and `inspect_relationship` register
  their numbers as results with query ids. The responder may cite them by token.
  A numeral the model copies from a tool output *without* a token fails
  provenance, like any other untraced number.

Tool definitions use strict schemas. Inputs are validated against the Pydantic
model before the tool runs, and a validation failure is returned to the model as
a structured tool error, not executed.

---

### 13.6 Planner

One structured-output model call per planning step, inside a bounded tool-use
loop. The loop ends when the model takes a terminal action. The final output is
a `PlannerDecision`, a discriminated union validated by Pydantic:

```python
class PlannerDecision(BaseModel):
    decision: SemanticDecision | InvestigationDecision | ClarifyDecision | UnanswerableDecision

class PlanFraming(BaseModel):                # required on every answerable decision
    stance: AnalysisStance                    # REQUIRED: no default at this boundary
    stance_basis: StanceBasis                 # enum: why this stance (see 13.11)
    knowledge_horizon: HorizonSpec | None     # a snapshot rule, never a free date guess
    required_concepts: list[BusinessConcept]
    required_features: list[FeatureRef]
    dimensions: list[str]
    comparisons: list[ComparisonKind]
    ambiguities_resolved: list[AmbiguityResolution]   # which §5.3 default, and why
    carried_forward: PlanEdit | None          # follow-ups: an edit, not a new plan (13.13)

class SemanticDecision(BaseModel):
    kind: Literal["semantic"]
    framing: PlanFraming
    plan: AnalysisPlan

class InvestigationDecision(BaseModel):
    kind: Literal["investigation"]
    framing: PlanFraming
    plan: InvestigationPlan
    why_not_semantic: NotSemanticReason       # enum: no metric defines this; needs a
                                              # derived feature; needs an association

class ClarifyDecision(BaseModel):
    kind: Literal["clarify"]
    request: ClarificationRequest

class UnanswerableDecision(BaseModel):
    kind: Literal["unanswerable"]
    reasons: list[UnanswerableReason]         # coded, each naming a concept or period
```

**What the planner decides**, and what checks it deterministically:

| Decision | Checked by |
|---|---|
| Answerable at all | Unavailable concepts are visible in the card; the gate re-checks |
| Path, semantic or investigation | Semantic is **required** when a registry metric covers the question: an investigation plan whose population and operation are expressible as a registry metric is rejected (`SEMANTIC_PATH_AVAILABLE`), so the weaker path is never chosen to route around a gate |
| Stance | Required field; the gate enforces allowlists; the §13.11 lint cross-checks |
| Knowledge horizon | Expressed as a snapshot rule, resolved deterministically |
| Required concepts, features, dimensions | The gate verifies that `required_concepts` ⊇ each metric's required concepts. A mismatch is a planner misunderstanding and a repair error, not a silent correction |
| Comparisons | Only `ComparisonKind`; the engine computes deltas |
| Clarification needed | §13.12 policy |

**Absolute constraints**, enforced by the output schema rather than the prompt:

- No field anywhere in `PlannerDecision` accepts an expression, formula, SQL
  string, or free numeric value derived by the model. Numbers the model may
  write are **user-supplied literals** (a filter threshold, a limit, a year),
  each tagged `source="user"` and whitelisted by provenance only if it appears
  in the user's text.
- No SQL on the semantic path. `run_guarded_sql` is a separate tool, off by
  default, and only reachable after an investigation plan was refused with
  `INEXPRESSIBLE` (§13.9).
- The schemas are non-recursive. Structured outputs reject recursive schemas,
  and that limit is welcome: an expression tree is exactly what the plan must
  not contain.

> **As implemented (§13.23).** The decision is a `PlannerOutcome`
> (`FINAL_PLAN`, `TOOL_REQUEST`, `CLARIFICATION_REQUEST`, `REJECTED`) rather
> than a `PlannerDecision` with a `PlanFraming`. The framing fields were not
> built: stance, horizon, concepts and dimensions are already fields of the
> plan itself, and the gate checks them there. A follow-up is `FINAL_PLAN` with
> path `edit`, carrying a `PlanEdit`. `UnanswerableDecision` is `REJECTED` with a
> model-chosen reason.

**Repair.** A gate rejection is returned to the planner as the
`PlanValidation` object: codes, fields, remedies. Two repairs per turn. A third
rejection ends planning with a clarification built from the outstanding
rejections, or an abstention if they are unanswerable (`CONCEPT_UNAVAILABLE`,
`COLUMN_QUARANTINED`).

---

### 13.7 AnalysisPlan changes

The contract in `contracts/plan.py` is close to final after §12.20. Four changes,
each needed by this layer and nothing speculative:

| Field | Change | Why |
|---|---|---|
| `AnalysisSpec.measure` | Replace the `Measure` enum (`arr`, `amount`, `weighted`) with `measure_concept`. **Landed (§13.22)** as an optional concept name; `measure_column` was removed rather than kept, so no plan field accepts a physical column as a measure | `arr` and `weighted` name columns, not concepts; §12.1 says metrics speak concepts. `weighted` returns when a probability concept exists |
| `AnalysisSpec.comparison` | Keep, and **compile it**: period-over-period deltas and percent change as engine-computed columns | The responder may cite a change only if the engine produced it (§13.14) |
| `Filter.values` | Add `value_source: Literal["user", "profile", "session"]` | Provenance must know which literals the user supplied |
| `AnalysisPlan.plan_id`, `parent_plan_id` | New | Follow-up edits and the carry-forward report reference plans by id |

`stance` and `knowledge_cutoff` (Appendix C.1, C.2) already landed. `features`
(C.3) already landed and is consumed by `RANKED_LIST`. `feature_source` (C.5)
stays unimplemented until the reconciliation harness exists. The compiler still
accepts only a `GateOutcome`.

---

### 13.8 InvestigationPlan

A second plan type for questions the metric registry does not define. It is
plan-as-data in the same sense as `AnalysisPlan`: a closed vocabulary of
populations, variables and operations, each compiled by deterministic code over
the same resolver, the same stance allowlist, and the same monetary boundary.
It is not a program. It cannot loop, branch, define a function, or compute an
expression the vocabulary does not name.

```python
class InvestigationPlan(BaseModel):
    plan_id: str
    question_restatement: str
    hypotheses: list[Hypothesis]              # what is being asked, typed
    stance: AnalysisStance                    # required, no default here
    knowledge_cutoff: date | None
    population: Population                    # who is analysed
    unit: AnalysisUnit = AnalysisUnit.OPPORTUNITY
    variables: list[Variable]                 # what is measured per unit, ≤ 6
    grouping: list[Grouping]                  # how units are split, ≤ 2
    operation: StatisticalOperation           # what is computed
    comparison: InvestigationComparison | None
    evidence: EvidenceRequirements            # what makes an answer admissible
    limit: int | None

class Hypothesis(BaseModel):
    statement: str                            # the user's question, restated
    kind: HypothesisKind                      # ASSOCIATION · DIFFERENCE · DISTRIBUTION
                                              # · COMPOSITION · TREND
    outcome: VariableId | None                # the variable being explained
    exposures: list[VariableId]               # what it might depend on

class Population(BaseModel):
    period: Period
    cohort_snapshot: SnapshotSelection        # membership frozen here (5.4)
    window: SnapshotWindow                    # which snapshots are observed after it
    filters: list[Filter]
    status_filter: list[OpportunityStatus] = [OPEN]

class Variable(BaseModel):
    id: VariableId
    source: ConceptRef | ColumnRef | DerivedFeatureRef
    role: Literal["outcome", "exposure", "covariate"]

class DerivedFeatureRef(BaseModel):
    """A per-opportunity feature computed by a semantic primitive over the window.
    The closed list is the whole point: these are recomputed from snapshot
    history (5.9), never read from a precomputed counter."""
    feature: DerivedFeature
    concept: BusinessConcept | None           # the attribute it is computed over
    params: DerivedFeatureParams              # enumerated, e.g. period basis

class DerivedFeature(StrEnum):
    CHANGE_COUNT          # times the attribute changed across the window
    CHANGED               # whether it changed at all
    DIRECTION             # first-to-last: up, down, unchanged (ordinal attributes)
    SLIPPED               # close date moved into a later period (5.3 ambiguity 2)
    PUSH_COUNT            # close date moved later, any amount
    PULLED_IN
    DAYS_IN_STATE         # snapshots spent in the current stage or category
    FINAL_STATE           # trace outcome: retrospective only
    VALUE_AT              # the attribute at a named snapshot rule

class StatisticalOperation(StrEnum):
    COUNT · SUM · DISTRIBUTION · CROSSTAB · RATE_BY_GROUP
    DIFFERENCE_IN_RATES · RANK_CORRELATION · TREND

class EvidenceRequirements(BaseModel):
    min_group_support: int = 10               # a group below this is reported, not rated
    required_concepts: list[BusinessConcept]
    disclose_association_not_causation: bool = True   # always true for ASSOCIATION
```

**Worked example.** "Is forecast-category volatility associated with deal
slippage?"

```yaml
stance: retrospective          # slippage is observed after the cohort was fixed
population:
  period: FY2025-Q2
  cohort_snapshot: period_open  # the opening pipeline, frozen
  window: period_open .. period_close
  status_filter: [open]
variables:
  - id: volatility
    source: {feature: CHANGE_COUNT, concept: forecast_category}
    role: exposure
  - id: slipped
    source: {feature: SLIPPED, concept: expected_close_date, params: {basis: period_move}}
    role: outcome
grouping:
  - {variable: volatility, bins: explicit, edges: [0, 1, 2]}   # 0, 1, 2+
operation: RATE_BY_GROUP        # slip rate within each volatility group
comparison: {kind: vs_reference_group, reference: "0"}
evidence: {min_group_support: 10, required_concepts: [forecast_category, expected_close_date, opportunity_status]}
```

This compiles into cohort, transition and rate primitives that already exist.
The result is a table with each group's numerator, denominator, rate, and
support, plus a difference in rates against the reference group, every value
engine-computed. The answer says "associated with" and never "causes". A group
under the support floor is shown with its counts and no rate.

**Statistics stay inside DuckDB.** Counts, rates, differences and rank
correlation are native SQL. Significance testing is deliberately absent in v1:
adding scipy is new infrastructure, and a p-value over a few hundred deals
invites more confidence than it deserves. The answer reports effect size with
support, which a reader can judge.

**Investigation gate.** It runs the same checks as the plan gate (concepts,
stance allowlist, grants, horizon, snapshot coverage), plus four of its own:
`FINAL_STATE` and any outcome-derived feature are refused under a prospective
stance; every variable resolves; the unit of analysis is the opportunity unless
the plan says otherwise, in which case a warning states that snapshot rows are
not independent; and `SEMANTIC_PATH_AVAILABLE` rejects a plan that is really a
registry metric.

**Trust.** An investigation result is at most tier B, always. Its definition is
ad hoc even when every input is confirmed.

---

### 13.9 Safe ad-hoc computation: the guarded SQL fallback

The typed investigation plan is the ad-hoc path. Guarded SQL exists for the
residue it provably cannot express, and its use is measured so the residue is
known rather than assumed. Design only; not implemented.

**Reachability.** Off by default per deployment. When enabled, reachable only
after the investigation gate refused a plan with `INEXPRESSIBLE`, and the call
must carry a `justification` naming the missing capability (an enum plus a
sentence). Every use is recorded, and a recurring justification is a backlog
item for the investigation vocabulary.

**The relation is built, not filtered.** For each turn the engine creates a
temporary view, `turn_view`, containing:

- only the columns permitted under the stance, including grant-scoped columns
  exposed under their concept name (`amount`), never under the raw header;
- only rows with `as_of ≤ horizon`;
- monetary columns already cast to `DECIMAL(18,2)`.

The leakage guarantee is therefore the view's contents, not the guard's
parsing. A forbidden column is not refused by the guard; it does not exist.

**AST validation** with `sqlglot` (the justified addition of §3.2), over the
parsed tree, never over text:

| Rule | Enforcement |
|---|---|
| Exactly one statement, a `SELECT` (CTEs allowed, read-only) | Root node type |
| Relations | Only `turn_view` and CTE names defined in the statement |
| Prohibited statements and functions | `ATTACH`, `COPY`, `EXPORT`, `INSTALL`, `LOAD`, `PRAGMA`, `SET`, `CALL`, `CREATE`, any DML or DDL; `read_*`, `glob`, `query`, `query_table`, `getenv`, `current_setting`, any table function |
| Columns | Every column reference resolves to `turn_view` or a CTE output |
| Money | No `/` and no `AVG` whose operand references a monetary column; `safe_div(a, b, scale)`, a registered macro over `exact_divide`, is provided instead |
| Output | Must end in a `LIMIT` ≤ the row cap; the guard appends one if absent |

**Execution.** A separate read-only connection that can see only the temporary
view; statement timeout (`query_timeout_seconds`); memory cap; row cap
(`max_result_rows`); `SET` locked after the connection is configured.

**Provenance.** The SQL text, its normalized AST hash, the view definition, the
justification, and a hash of the result are recorded, and the result is
registered as a query id. Trust is B at best, with `guarded_sql` as a named
trust factor.

---

### 13.10 Trust propagation

The tier is computed, never chosen, and nothing in any LLM output schema has a
field for it. §12.20 implemented the rule for single results as a list of
reason strings. The LLM layer needs it structured, because the responder must
disclose *why*, and the evaluation must check the tier.

```python
class TrustFactor(BaseModel):
    kind: TrustFactorKind          # enum, below
    tier: TrustTier                # the ceiling this factor imposes
    subject: str                   # concept, column, test id, or plan field
    reason: str                    # deterministic sentence, templated

class TrustAssessment(BaseModel):
    tier: TrustTier                # = weakest(f.tier for f in factors), or A if none
    factors: list[TrustFactor]
```

| Factor | Ceiling |
|---|---|
| Path: registry metric, compiled plan | A |
| Path: investigation plan | B |
| Path: guarded SQL | B |
| Binding confirmed by declaration, or grain assertion | A |
| Binding inferred (dimension or filter only; load-bearing is refused) | B |
| Usage grant on an otherwise unclassified column | A, disclosed |
| Feature lineage unconfirmed | B |
| Reconstruction `VALID` | A |
| Reconstruction `VALID_WITH_WARNINGS` | B |
| Status not authoritative (stage-keyword fallback) | B |
| Fiscal calendar unresolved, when the result depends on period boundaries | B |
| Snapshot drift beyond tolerance | B |
| Retrospective read of a contaminated or unknown column | B |
| A §5.3 ambiguity resolved by documented default and disclosed | A |
| An ambiguity the planner flagged as unresolved | C: clarify instead |
| Required concept ambiguous, unavailable, or `CONTRADICTED` | C: nothing executes |
| Sanity invariant failed | C: the turn fails |

**Propagation.** A result's tier is the weakest of its factors. An answer's tier
is the weakest among the results it cites. A tier-C input never reaches the
responder as a number; it reaches it as the reason for abstaining. The responder
receives the `TrustAssessment` and must render every B factor's reason. The
renderer appends any it omitted (§13.14), so a disclosure cannot be dropped by
the model.

"No unresolved assumptions" in tier A means no *undisclosed* assumption. A
documented default, stated in the answer, is a disclosed assumption, and it
keeps tier A.

---

### 13.11 Temporal safety

Stance is the most consequential field the planner fills. The design makes the
choice explicit, then makes a wrong choice unable to leak.

**Explicit choice.** `PlanFraming.stance` has no default at the LLM boundary;
the planner must emit one, with a `StanceBasis`:

```python
class StanceBasis(StrEnum):
    AS_KNOWN_AT          # "what was knowable at", "as of", "at quarter open"
    EVENTUAL_OUTCOME     # "eventually", "ended up", "how did X perform"
    CURRENT_STATE        # latest snapshot only: prospective with horizon = latest
    DEFAULT_PROSPECTIVE  # no signal; prospective is the safe default
```

The session stance starts prospective and changes only by an explicit plan
decision, which the answer discloses ("using outcomes observed after
2025-04-01").

**Enforcement is layered**, and each layer works even if the others are wrong:

1. **Context.** Under prospective, the catalog and the tools do not expose
   contaminated columns' values, and profiles are horizon-bounded (§13.5). The
   model cannot reason from what it cannot see.
2. **Gate.** Contaminated, unknown and retrospective-concept reads are refused
   with temporal-safety codes (implemented, §12.20).
3. **Compiler.** Every prospective query carries an `as_of` ceiling at the
   analysis snapshot and at the knowledge cutoff (implemented).
4. **Attribution.** Under prospective, dimension attribution must be
   `period_open` or earlier. `latest` and `at_close` attribution read a later
   snapshot's segment or owner, which is retrospective attribution; the gate
   rejects them (`STANCE_VIOLATION`). *This check is new.*
5. **Stance lint.** A deterministic cross-check: if a prospective plan's
   population window extends past its horizon, or if it includes `FINAL_STATE`,
   `COHORT_TRACE`, or a terminal concept, the plan is rejected with a repair
   message naming the conflict. The lint never *changes* the stance; it forces
   the planner to choose again.

**Worked pair.**

| Question | Stance | Horizon | Readable |
|---|---|---|---|
| "What was knowable at Q3 opening?" | prospective | Q3 `PERIOD_OPEN` snapshot | as-of and backward-derived columns at or before that snapshot |
| "How did the Q3 opening cohort eventually perform?" | retrospective | none, or an explicit cutoff | the cohort frozen at Q3 open, traced forward; terminal columns permitted and disclosed |

---

### 13.12 Clarification

Clarification is a terminal decision with a typed request, not a question in
prose.

```python
class ClarificationRequest(BaseModel):
    reason: ClarificationReason       # MISSING_PERIOD · AMBIGUOUS_DEFINITION
                                      # · CONCEPT_UNAVAILABLE · AMBIGUOUS_BINDING
                                      # · STANCE_UNCLEAR · COVERAGE_GAP
    ambiguity_id: str | None          # the §5.3 ambiguity, when it is one
    question: str                     # to the user, no numbers
    options: list[ClarificationOption]   # 2 to 4; each is a typed PlanEdit
    default_option: int | None        # applied if the user says "whatever's standard"
    available_alternatives: list[str] # deterministic, from the card, never invented
```

Each option is a **plan edit**, so an answer to the clarification applies
deterministically and never re-enters free-form planning.

**The policy.** Each ambiguity in the §5.3 registry carries a policy:

| Ambiguity | Policy |
|---|---|
| Period missing, multi-quarter dataset | **Ask**; offer the latest complete quarter as the default option |
| Win-rate denominator | Default `closed_only`, disclosed |
| Win-rate keying | Default `close_date`, disclosed |
| Created-in-period basis | Default `created_date` if bound, else **ask** before using first-seen |
| Dimension attribution | Default `period_open`, disclosed |
| Slip basis | Default `period_move`, disclosed |
| Stance, when the question gives signals both ways | **Ask** |

**Materiality probe (optional, deterministic).** For a defaulted ambiguity, the
orchestrator may run both readings, which are two compiled plans and cheap on
this data. If they differ by more than a configured threshold, the default is
withdrawn and the user is asked, with both values available as registered
results. This turns "does the ambiguity change the result materially" from a
model judgement into a measurement.

**Unavailable concepts are never substituted.** "Which segments had the
highest win rate?" on a dataset where `customer_segment` is unavailable:

1. The card shows the concept as unavailable, so the planner decides
   `unanswerable` or `clarify` with `CONCEPT_UNAVAILABLE`.
2. `available_alternatives` is filled **deterministically** from the bound
   dimension concepts: owner, stage, forecast category, and so on. The model
   may present them; it may not add to them, and it may not describe any as
   equivalent to segment.
3. If `inspect_concept` shows an **unbound candidate** column (say
   `Segment__c`, inferred), the request becomes `AMBIGUOUS_BINDING`: "a column
   that may hold customer segment exists but has not been confirmed. Confirm
   it?" A yes is a user confirmation, recorded as declarative evidence with the
   user and timestamp, scoped to the session unless the tenant makes it
   permanent.

---

### 13.13 Follow-ups and session behaviour

The session holds plans, not a transcript. A follow-up edits the active plan.

```python
class SessionState(BaseModel):
    session_id: str
    dataset_id: str
    stance: AnalysisStance                    # session default, prospective
    plans: dict[str, AnalysisPlan | InvestigationPlan]
    active_plan_id: str | None
    entities: dict[str, ResolvedEntity]       # "that quarter" → FY2025-Q3
    result_handles: dict[str, QueryId]        # q1, q2 ... for this session
    turns: list[TurnSummary]                  # question, decision kind, plan id, tier

class PlanEdit(BaseModel):
    base_plan_id: str
    operations: list[EditOperation]           # typed, ordered

EditOperation = (AddDimension | RemoveDimension | SetFilter | RemoveFilter
                 | ChangePeriod | ChangeMetric | ChangeSnapshotRule
                 | ChangeStance | SetLimit | SetOrder | Replace)
```

`apply_edit(base, edit)` is deterministic. Its output is a new plan with a new
id and `parent_plan_id`, which goes through the gate again in full: a filter
carried forward is re-validated against the new plan, not trusted because it
passed last turn.

**The carry-forward report** is produced by diffing parent and child, not by the
model, and it is shown in the answer:

```
Q: "What was Q3 opening pipeline?"
   plan p1: opening_pipeline · FY2025-Q3 · period_open · prospective

Q: "Break that down by owner."
   edit on p1: AddDimension(owner)
   plan p2: carried metric, period, snapshot rule, stance · added owner
            (attributed as of the opening snapshot)

Q: "Only show deals above $100k."
   edit on p2: SetFilter(amount ≥ 100000, value_source=user)
   plan p3: carried metric, period, snapshot rule, stance, owner · added filter
            amount ≥ 100,000 read at the opening snapshot
```

Two rules keep this honest:

- **The filter's snapshot is stated.** "Deals above $100k" is itself ambiguous,
  because amount changes. The filter reads the snapshot the plan reads, and the
  carry-forward report says which.
- **An edit that changes meaning is not silent.** `ChangeStance`, a changed
  metric, or a period change that crosses the dataset's coverage is surfaced as
  a change, not as a carry-forward.

A question with no referent in the session is planned fresh. A referent the
resolver cannot pin to one plan ("that" after two different plans) is an
`AMBIGUOUS_REFERENCE` clarification, not a guess.

---

### 13.14 Responder

A separate structured-output call. It sees results; it never produces a number.

**Input**, assembled deterministically:

- the question, and the carry-forward report if this is a follow-up;
- the executed plan or plans, in compact form;
- each result as a `ResultView`: column names, semantic types, up to 50 rows
  with their **token handles**, row counts, and whether the table was truncated;
- assumptions by id, warnings, and the `TrustAssessment`;
- for comparisons, the engine-computed direction columns (`delta`,
  `pct_change`, `direction`), so "rose" and "fell" can be grounded.

**Output:**

```python
class AnswerDraft(BaseModel):
    headline: str                       # one or two sentences, tokens only
    explanation: str | None             # short; tokens only
    breakdown: ResultTableRef | None    # a result id: the renderer draws the table
    assumptions: list[str]              # assumption ids to show, from the input
    caveats: list[str]                  # trust factor ids to show, from the input
    comparisons: list[ComparisonClaim]  # typed claims backing any comparative prose
    chart: ChartSpec | None             # over a result id and its columns
    follow_up_suggestions: list[str]    # optional, no numbers
```

**Reference tokens** use per-turn aliases: `{{q1.r0.opening_pipeline}}`, where
`q1` maps to a registered query id. Metadata is addressable too:
`{{q1.meta.resolved_as_of}}`, `{{q1.meta.drift_days}}`. The renderer formats
each value by semantic type: money as currency, ratios as percentages, dates in
ISO. The model never formats a number, so it never rounds one.

**Tables are not retyped.** A breakdown is a `ResultTableRef`; the renderer
draws the table from the result. A model copying twenty rows is a model with
twenty chances to transpose a digit.

**Comparative claims are typed.** "Pipeline fell" is a claim about two numbers.
The draft carries it as `ComparisonClaim(kind=less_than, left=token,
right=token)`, which the renderer verifies before emitting. A comparative word in
prose without a supporting claim is flagged by a vocabulary check (fell, rose,
higher, lower, more, fewer, exceeded) and fails the draft. This closes the gap
the numeral scanner cannot see: a correct number with the wrong direction.

**Mandatory disclosures are appended, not requested.** Retrospective stance,
every tier-B factor, drift beyond tolerance, an unresolved fiscal calendar, and
"association, not causation" for investigations are appended by the renderer if
the draft omits them. The model is asked to include them in context, but the
guarantee does not depend on it.

---

### 13.15 Provenance integration

The provenance scanner of §8.4 runs on the rendered answer, unchanged in
principle, with these specifics:

1. Extract every numeral, currency amount, percentage and date.
2. Each must match a value substituted from a token, formatted by the renderer.
3. The whitelist: user-supplied literals from the question (`$100k` matches
   `100000` after normalization), period labels, years, the plan's `limit`,
   ordinals, and resolved snapshot dates registered as result metadata.
4. An unresolvable token fails the draft. So does a token whose column is not in
   its result.
5. Comparison claims are verified (§13.14).

On failure: one regeneration, with the `ProvenanceReport` given to the
responder as structured feedback. On a second failure: a table-only answer,
built by the renderer from the result sets with no model prose, and a note that
the narrative was withheld. An unverified number is never emitted.

`contracts/answer.py`, still unimplemented, gains `Answer`, `AnswerDraft`,
`ProvenanceReport`, and `RenderedAnswer`. `RunRecord` (§8.6) gains
`planner_decision`, `plan_edit`, `trust_assessment`, `investigation_plan`,
`guarded_sql`, `tool_calls` (name, input hash, output hash, latency), and
`clarification`.

---

### 13.16 Cost and latency controls

| Control | Setting |
|---|---|
| Model per call | Planner: the most capable model, with adaptive thinking; planning quality is where errors are expensive. Responder: a faster, cheaper model; it composes, it does not reason over numbers. Both are configuration, not constants |
| Prompt caching | Stable prefix in order: system prompt, tool definitions, tier-0 card, metric catalog. One breakpoint after the catalog; session state and question after it. The card changes only when the dataset or tenant profile changes. Structured-output schemas are fixed per release, so the one-time schema compilation cost is paid rarely |
| Planning budget | ≤ 8 inspection calls per turn; ≤ 2 gate repairs; ≤ 2 execution rounds |
| Output budget | Explicit `max_tokens` per call; a `max_tokens` stop on a terminal action is a retry with a higher cap, never an executed partial plan |
| Result budget | ≤ 50 rows per result to the responder; larger tables go by reference and are rendered, not narrated |
| Follow-ups | A `PlanEdit` is small; follow-ups skip most inspection because the active plan already names its concepts |
| Result cache | Keyed by (dataset version, tenant-profile version, normalized plan hash). Deterministic compilation makes the key sound |
| Materiality probe | Only for ambiguities marked probe-able, and only when both readings compile |
| Turn deadline | A wall-clock budget per turn; on expiry, emit what is verified, or a timeout abstention |

Targets are set by evaluation, not guessed here. Measure p50 and p95 turn
latency, tokens in and out, and cache-read ratio per turn from the run record.

---

### 13.17 Failure and retry behaviour

Every failure has a class, a bounded retry, and a terminal behaviour. A failure
the orchestrator cannot classify fails the turn.

| Failure | Retry | Terminal behaviour |
|---|---|---|
| API transient (429, 5xx, overload) | SDK retries with backoff; bounded | Abstain: "the analysis service is unavailable" |
| `stop_reason: refusal` | None | Abstain; the refusal is logged |
| `stop_reason: max_tokens` on a terminal action | Once, with a higher cap | Abstain |
| Output fails Pydantic validation | Once, validation errors returned as a tool error | Clarify or abstain |
| Tool input fails validation | Returned to the model; counts against the inspection budget | — |
| Gate rejection | ≤ 2 repairs with structured rejections | Clarification built from the rejections, or abstention |
| Compiler error | **None**: a compiler error is an engine bug | Fail the turn, record, alert |
| Execution error or timeout | Once for a timeout, with a narrower period if the plan allows | Fail the turn with the reason |
| Sanity invariant failure | **None**: an invariant failure is a bug | Fail the turn; never a hedged number |
| Empty result | One planning round with the empty-result fact | Answer "no matching rows", stated as such |
| Provenance failure | One regeneration | Table-only answer |
| Turn deadline | — | Emit verified parts, or a timeout abstention |

The LLM is never asked to repair a deterministic component's failure. A
compiler error, an invariant failure, or a wrong sanity result is code to fix,
and routing it back to the model would teach the model to work around a bug.

---

### 13.18 Evaluation strategy

§9's four tiers stand. The LLM layer adds planner-level and multi-tenant tiers,
and the principle does not change: **numeric correctness is never scored by a
model.**

| Suite | What it measures | Scored by |
|---|---|---|
| Planner golden set | Question → expected `PlannerDecision`, compared **structurally**: path, metric, stance, period, dimensions, filters, ambiguity resolutions. Not text | Deterministic diff |
| Stance suite | Prospective and retrospective phrasings of the same underlying question. **Prospective leakage rate must be zero** | Deterministic: any contaminated read under prospective fails |
| Clarification suite | Questions that must ask, and questions that must not. Precision and recall | Deterministic decision-kind match |
| Abstention suite | Unavailable concepts; out-of-coverage periods. No substitute dimension may be presented as equivalent | Deterministic, plus an alternatives-list check |
| Follow-up suite | Multi-turn scripts; the resulting plan after each edit must equal the expected plan | Deterministic plan equality |
| End-to-end numeric | Rendered answer numbers equal hand-computed fixture values (§9.2) | Exact match |
| Provenance | `provenance_coverage` = 100%; comparison claims verified | Deterministic |
| Trust | Tier and factor set equal the expected assessment | Deterministic |
| **Schema permutation** | The same questions over the tiny fixture with renamed headers, a missing concept, an extra custom column, Excel serial dates, and a declared custom amount column. Answers must match the canonical run, or abstain for the right reason | Deterministic |
| Adversarial | Column names and category values carrying injected instructions; questions baiting outcome leakage; requests to "just estimate" | Deterministic outcome checks |
| Narrative quality | Concision, clarity, disclosure wording | LLM judge, **narrative only** |

**Replay.** Every eval run records model outputs. CI replays recorded outputs
through the deterministic pipeline, so a change to the gate, compiler or
renderer is tested without a model call and without flakiness. Live-model evals
run on a schedule and gate model or prompt changes.

---

### 13.19 API boundary

FastAPI (the declared stack), designed here and built after the runtime. One
turn is one request; the response is the `Answer` contract.

| Endpoint | Purpose |
|---|---|
| `POST /datasets` | Upload a parquet or CSV; returns a job id. Ingest, profile, understand |
| `GET /datasets/{id}/context` | The tier-0 card, as JSON |
| `GET /datasets/{id}/concepts` | Bindings, verdicts, grants, open questions |
| `POST /datasets/{id}/declarations` | Tenant declarations: bindings, classifications, agreement semantics, fiscal calendar. Validated per §13.2; audited |
| `POST /sessions` | Start a session on a dataset |
| `POST /sessions/{id}/turns` | Ask a question, or answer a clarification with an option id. Returns `Answer` |
| `GET /runs/{run_id}` | The full run record |
| `GET /runs/{run_id}/plan` and `/sql` | "Show analysis plan" and "Show SQL" |

```python
class Answer(BaseModel):
    run_id: str
    kind: Literal["answer", "clarification", "abstention", "failure"]
    text: str | None                     # rendered, provenance-verified
    tables: list[RenderedTable]
    chart: ChartSpec | None
    trust: TrustAssessment
    assumptions: list[str]
    caveats: list[str]
    carried_forward: CarryForwardReport | None
    clarification: ClarificationRequest | None
    evidence: list[QueryId]              # every result the answer cites
    plan_ref: str | None                 # link to the plan
    sql_ref: str | None                  # link to the compiled SQL
```

Streaming progress (planning, executing, rendering) is a later addition; the
state machine's named states are what it would stream.

---

### 13.20 Summary

#### Architecture

Two LLM calls, a planner and a responder, around a deterministic core that
already exists. The planner inspects the tenant's actual dataset through eleven
structured, stance-bounded tools and submits a typed plan. The deterministic
gate, compiler and executor produce a checked result with a computed trust tier.
The responder writes prose in reference tokens, and a deterministic renderer and
provenance scanner decide what is emitted. Two prerequisite changes come first:
reconstruction gating distinguishes validity from reconciliation (§13.1), and an
explicit tenant declaration issues a purpose-scoped usability grant (§13.2).

#### Contracts that need to change

| Contract | Change |
|---|---|
| `contracts/agreement.py` | `AgreementRole` (`VALIDITY`, `RECONCILIATION`) on tests and results; `ReconstructionVerdict`; offset histogram as a diagnostic field |
| `contracts/tenant.py` | `AgreementDeclaration` (declared semantics of an ancillary field); `column_classifications`; declaration source and timestamp |
| `contracts/binding.py` | `UsageGrant`; `ConceptBindings.grants` |
| `contracts/plan.py` | `measure_concept` replaces `Measure`; `Filter.value_source`; `plan_id` and `parent_plan_id`; the comparison field compiled |
| `contracts/rejection.py` | New codes: `SEMANTIC_PATH_AVAILABLE`, `INEXPRESSIBLE`, `DECLARATION_CONFLICT`, and `STANCE_VIOLATION` applied to attribution |
| `contracts/result.py` | `TrustFactor` and `TrustAssessment` replace `trust_reasons: list[str]`; `ResultView` with token handles; metadata cells addressable |
| New `contracts/investigation.py` | `InvestigationPlan` and its vocabulary |
| New `contracts/planner.py` | `PlannerDecision`, `PlanFraming`, `StanceBasis`, `ClarificationRequest`, `PlanEdit` |
| New `contracts/session.py` | `SessionState`, `CarryForwardReport` |
| New `contracts/answer.py` | `AnswerDraft`, `ComparisonClaim`, `Answer`, `ProvenanceReport`, `RunRecord` fields |
| `semantic/resolver.py` | Honours usage grants; attribution check under prospective |
| `data/reconstruct.py`, `data/binding.py` | Role-aware gating and the four-state verdict |

#### Contracts that can remain unchanged

`contracts/concepts.py` (ontology), `contracts/columns.py` (classification,
availability, disposition, quarantine), `contracts/dataset.py` (registry),
`contracts/schema.py`, `contracts/profile.py`, `contracts/status.py`,
`contracts/context.py` (`AnalystContext`: the tier-0 card is reused as is),
`contracts/errors.py`; the semantic engine's calendar, snapshots, cohort, trace,
transitions, bridge, rate, metric registry, compiler, executor and their
invariants; the `ResultSet` addressing scheme `(query_id, row, column)`.

#### Implementation order

Each step is testable without the next, and the LLM enters at step 7.

1. **Reconstruction verdicts** (§13.1): agreement roles, the four-state verdict,
   role-aware gating, and a rewrite of the reconstruction-gating tests. Re-probe
   the sibling export and confirm `expected_close_date` becomes available with
   warnings.
2. **Usage grants** (§13.2): declaration validation, grants in the resolver.
   The current golden test pinning "a declaration does not lift a quarantine" is
   rewritten to assert the grant's scope instead.
3. **Structured trust** (§13.10): `TrustFactor`, `TrustAssessment`, the
   attribution check, and the full factor table, all against existing results.
4. **Investigation plan and gate** (§13.8), compiled from existing primitives,
   with hand-computed goldens on the tiny and bridge fixtures.
5. **Session, plan edits, carry-forward report** (§13.13), with no model
   involved: edits are constructed in tests.
6. **Renderer, provenance scanner, comparison claims** (§13.14, §13.15),
   exercised with hand-written drafts.
7. **Tools and context builder** (§13.4, §13.5), deterministic and tested.
8. **Planner**, with the evaluation suites of §13.18 built alongside it, then
   the **responder**, then the **orchestrator** state machine.
9. **Guarded SQL** (§13.9), only if the investigation vocabulary proves
   insufficient, which the justification log will show.
10. **API** (§13.19).

#### Major risks

| Risk | Mitigation |
|---|---|
| **Stance misclassification** — a retrospective question answered prospectively is merely narrow; the reverse leaks | Explicit required stance, layered enforcement, horizon-bounded tools, a zero-tolerance leakage eval |
| **Leakage through inspection tools** rather than queries | Horizon-bounded profiles; contaminated columns show classification without values under prospective |
| **Investigation vocabulary too narrow**, pushing questions to guarded SQL or refusal | Justification log on every fallback; the vocabulary grows by reviewed change |
| **Correct numbers, wrong words** — direction, magnitude adjectives, causal language | Typed comparison claims; comparative-word check; mandatory "association" disclosure |
| **Planner routes to investigation to escape the gate** | `SEMANTIC_PATH_AVAILABLE` rejection; investigation results capped at B |
| **Over-clarification** makes the product tedious | Per-ambiguity policy; documented defaults with disclosure; materiality probe instead of reflexive asking |
| **Tenant strings as prompt injection** | Tenant content rendered as typed data, never as prose; adversarial eval suite |
| **Relaxed gating read as weakened gating** (§13.1) | Declaration still required; contradiction still withholds; reconciliation failures cap trust at B and are always disclosed |
| **Grants over-reach** (§13.2) | Purpose-scoped to the concept; conflicts with a known classification reject the declaration; inferred bindings never grant |
| **Cost and latency of agentic planning** | Cached prefix, inspection budget, cheap responder, plan-edit follow-ups, result cache |
| **Fiscal calendar and target export unverified** | Unresolved calendar is a B factor on period-dependent results; the target export still needs probing before any production claim |



### 13.21 Implementation status of the deterministic scaffolding

Everything in §13 that contains no model call is built and tested. The planner,
responder and orchestrator are not, and guarded SQL is deliberately pending: the
typed investigation plan is the ad-hoc path, and no justification log exists yet
to show what it cannot express.

**What landed, and where it departs from the design text above.**

* **Reconstruction verdicts (§13.1).** As designed, plus one test the design
  implied but did not name: `close_date_is_structurally_valid`, a `VALIDITY`
  test that every row with a `days_to_close` value rebuilt to a real date. A
  fractional or absurd horizon becomes NULL at conformance, and that is now a
  contradiction rather than a silent gap. The declaring source is recorded on
  the derived column (`declared_by`). A missing agreement report still
  withholds, because a verdict needs at least the structural check to have run.
  Re-probing the sibling export gives exactly the predicted outcome,
  `VALID_WITH_WARNINGS`, with three caveats: movement unverified, and the
  `CD_in_qtr` and `eoq_close_diff` conventions unresolved. The
  `eoq_close_diff` result carries its offset histogram as a diagnostic.
* **Usage grants (§13.2).** Purposes follow the concept's role through
  `ROLE_PURPOSES`; a measure also admits a filter, because "deals above $100k"
  is a read of the measure. A plan names a concept in a dimension or filter by
  its value or an alias (`amount`, `segment`), and only that route can use a
  grant. Every result that relied on one records it in its compilation
  metadata. *Superseded by §13.22:* grants are now issued once and persisted
  as a ledger (`grants.json`), and `TenantProfile.column_classifications` is
  implemented.
* **Trust (§13.10).** The tier is a derived property of a `TrustAssessment`,
  and every construction path forbids a supplied tier. A plan's own
  `unresolved_ambiguities` produce a tier-C factor, and a tier-C assessment
  raises `AbstentionRequired` before anything executes.
* **Investigation (§13.8).** The unit is the opportunity only. `DAYS_IN_STATE`
  and `DIRECTION` are not in the vocabulary, because nothing compiles them yet.
  The window is two snapshot selections around a frozen cohort, and both
  default to the cohort snapshot, which is the prospective case. Semantic
  equivalence is conservative: any derived feature makes a plan
  non-equivalent, and without one a count, a distribution, an open-amount sum
  or a won rate is a registry metric and is refused. The rank correlation is
  the one statistic computed through a float, rounded to six places; it is not
  money.
* **Session edits (§13.13).** The `Replace` operation was not built: a
  replacement is a new plan, not an edit. A session round-trips to JSON with its
  plans, edits and carry-forward reports.
* **Renderer and scanner (§13.14, §13.15).** A rendered answer is a list of
  segments labelled model, result or system, which is how the scanner knows a
  numeral in a table is traced while the same numeral in prose must be
  justified. Dates in prose must be resolved snapshot dates. An owner id or
  similar code written into prose by hand fails the scan; it must be a token.
* **Tools (§13.5).** `inspect_relationship` returns a contingency table only; a
  numeric correlation belongs to an investigation. The run tools refuse a plan
  that widens the session's stance and tighten every cutoff to the session's
  horizon.

### 13.22 Closing the deterministic gaps

The last deterministic milestone before the planner. Nothing here calls a
model. Each item lists what was built and the guard its mutation test proves is
load-bearing.

**Persisted usage grants (§13.2).** Grants are issued once, at registration,
by `data/grants.issue_grants`, a pure function of the dataset registry and the
tenant profile, and saved as a `GrantLedger` in `grants.json` beside the
dataset. Saving is byte-for-byte deterministic. The runtime *loads* the ledger
(`understand()` calls `load_grants`) and never rebuilds grants from the
profile: a missing ledger means no grants, and a trimmed ledger stays trimmed.

* Identity. Each grant's `grant_id` is a hash of its content (dataset, tenant,
  kind, column, concept, purposes, availability), so an edited grant fails to
  load.
* Binding. The ledger carries `dataset_id`, `tenant_id` and a fingerprint of
  the tenant's declarations. A mismatch is `DATASET_MISMATCH`,
  `TENANT_MISMATCH` or `STALE`, and the load fails.
* Re-validation. Every grant is re-checked against the dataset as it is now:
  a missing or retyped column is `SCHEMA_DRIFT`, and a declaration contradicted
  by a later registry classification is `CONTRADICTED`. Structural damage is
  `CORRUPTED`.
* Scope. The purpose rule is re-applied on load. A grant widened to a purpose
  its concept does not license is refused even when its id is forged to match.
* Declarations the ledger refused are kept in `rejected`, with reasons, for
  audit.

**Tenant column classifications (§13.2).** `TenantProfile.column_classifications`
is a list of `ColumnDeclaration`s, each reusing `ColumnClassification` rather
than a second ontology, with a status (`confirmed` or `inferred`) and a source.

* A confirmed, valid declaration issues a *generic* grant. It covers dimension,
  filter and feature reads, and measure reads only for a numeric column.
* An inferred one issues nothing.
* A declaration may release an *unknown* column only. It can never overrule a
  registry classification.
* A quarantine or text classification, or money declared on a non-numeric
  column, is rejected with its reason.
* A generic grant never binds a concept and never makes a column a measure
  concept.
* Its declared availability is enforced like any other: an unsafe or unknown
  timing is unreadable prospectively, and readable retrospectively with a
  RETROSPECTIVE_READ factor.
* A monetary declaration puts reads through the DECIMAL(18,2) boundary. This
  also fixed the investigation path, which summed a monetary column variable
  as DOUBLE.

**The attribution guard (§13.11 #4).** Every dimension and feature read is
recorded as an `AttributionRead`: the field, the snapshot it is read from, the
horizon, and a relation (BACKWARD, CONTEMPORANEOUS, LATER or
RETROSPECTIVE_TERMINAL). Under a prospective stance, a LATER or terminal read
is a STANCE_VIOLATION that names the field, the attempted snapshot and the
allowed horizon (`min(analysis snapshot, knowledge cutoff)`).

The compiler recomputes the horizon from the spec and refuses again, so a
tampered `ValidatedSpec` cannot compile a later read. The compiler now
implements attribution itself: a dimension is read at the attribution snapshot,
where before it was silently read at the analysis snapshot. A plan-path
rewrite-leakage test sits beside the investigation one, which is unchanged.

**Measure concepts and typed comparisons (§13.7).** `measure_concept` is an
optional ontology concept name, and the resolver alone makes it a column. Three
rules hold:

* An unknown name is UNKNOWN_MEASURE_CONCEPT. That includes a physical column
  name, even one with a monetary generic grant.
* The compiler refuses such a name as well, rather than falling back to the
  metric's default.
* There is no field that accepts a physical measure.

Comparisons compile to `CompiledComparison`s: two typed operands, an operator
(DIFFERENCE or RELATIVE_CHANGE), a result kind, and one expression. Operands of
different units are refused. Relative change goes through `exact_divide`, which
now lifts both operands by their scale (`operand_scale`), so a DECIMAL
denominator is never rounded to an integer.

* A baseline is validated as a spec of its own, and must itself be knowable
  prospectively.
* A group missing on one side is zero for a sum or a count, and its relative
  change is undefined (NULL).
* A rate or a bridge term cannot be compared period over period
  (INCOMPATIBLE_COMPARISON).

**Clarification materiality probe (§13.12).** `semantic/materiality.py`. A
probe is two to four readings, each a typed `PlanEdit` of the base plan. Each
ambiguity kind admits only its own operations:

* snapshot: a snapshot rule
* period: a period
* filter: a filter
* option: a documented §5.3 option, via the new `SetAnalysisOption` edit

The probe runs every reading through the full gate and executor and returns
MATERIAL, NOT_MATERIAL or INCONCLUSIVE. It has no field naming a chosen
reading.

It returns INCONCLUSIVE in these cases:

* a reading changes the question (metric, stance, dimension, measure);
* the ambiguity is a concept binding, which would mean computing on an
  unconfirmed binding;
* a reading fails the gate or is tier C;
* a result exceeds 50 rows.

Readings run under the base plan's stance and horizon, and the tool adapter
tightens both to the session scope.

**Snapshots actually read.** A result now reports every snapshot it read. For
a bridge, a trace or a transition, that is the period's window. It also
includes the snapshot each attribution reads from, and a comparison's baseline.
Investigations report their window edges. Before, multi-snapshot results
reported only the analysis snapshot.

**Smaller gaps.**

* `cohort_fate` is a registry metric for the COHORT_TRACE pattern, which
  compiled but was unreachable because no metric declared it. It is
  retrospective only.
* 'Created in period' with the default creation-date basis is rejected as
  CONCEPT_UNAVAILABLE on a dataset without that concept, with a remedy to
  choose `first_seen` explicitly. The bridge compiler refuses the same case, so
  it no longer falls back silently even if the gate is bypassed.
* The materiality probe is bounded in readings and rows, not in time:
  `query_timeout_seconds` is configured but the executor does not enforce it
  yet.

**Acceptance.** `tests/acceptance/` walks question → plan → gate → compiled SQL
→ DuckDB → ResultSet → trust → render → provenance scan for ten cases. The plans
and drafts are hand-written stand-ins for the planner and responder. Expected
values are hand-computed, and expected tiers are derived from the expected
factors through `FACTOR_CEILINGS`.

* It runs by default on the tiny, moves and custom fixtures.
* An opt-in run (`AI_ANALYST_REAL_EXPORT`) repeats the cases on the sibling
  tenant's export, against an independent raw-SQL reference.
* That run makes two test-only declarations: the stage-derived status column,
  and one text dimension. Neither resolves a production convention, and every
  result keeps its STATUS_NOT_AUTHORITATIVE and close-date factors.
* The sibling is nearly static across its three daily snapshots. Only open
  deals whose close date is the snapshot day move, one day at a time, so the
  bridge movement terms there are genuinely zero.

**Target export.** Still unprobed. The standard AWS credential chain has no
credentials in this environment, so the probe stops with "probe not run" and
claims nothing. This is recorded as the open question `target_export_unprobed`.

**Still unresolved, deliberately.** None of these is resolved:

* the `eoq_close_diff` convention (no ±1 tolerance was added);
* the `CD_in_qtr` boundary-day convention;
* the fiscal year start;
* the authoritative production status column;
* the unverified date declarations;
* the rounding of sub-cent amounts (`sub_cent_amounts`, below).

Each remains an open question, and those that bear on a result are trust
factors.

**A finding from the real-data run.** The sibling export stores 1,576
`new_amount` values with more than two decimal places. Conformance reads source
data as text and casts it to DECIMAL(18,2). On 7 values that lands a cent away
from casting the stored DOUBLE directly, which moved a cohort total by $0.07.
The engine follows the declared text-then-DECIMAL rule, and the independent
reference now states and follows the same rule. Whether sub-cent amounts should
round, truncate or be refused is the owner's decision, and is recorded as
`sub_cent_amounts`.

**Ready for the planner.** The planner's output contracts (`AnalysisPlan`,
`InvestigationPlan`, `PlanEdit`, `MaterialityProbe`) are closed and typed. Every
guard it could route around is enforced below it, twice where it matters:

* gate and compiler for attribution and measures;
* ledger load and resolver for grants.

The acceptance harness is the evaluation spine: replacing its hand-written plan
with the planner's is the only change the next milestone needs to measure plan
quality against hand-computed answers.


### 13.23 The planner runtime and its evaluation

The first model in the system. The planner proposes a typed plan; the
deterministic system decides whether it is valid and computes the answer.
Nothing downstream of the gate changed.

**Shape.**

```
question ─► PlanningLoop (deterministic, agent/planner/loop.py)
              │  LLMPlanner.plan / observe ─► PlannerModel session ─► provider
              │        ▲                            (anthropic_adapter.py)
              ▼        │ Observation (escaped tool result or error)
         PlannerOutcome: TOOL_REQUEST ─► tool args validated ─► surface tool
                         FINAL_PLAN   ─► boundary checks ─► scope + gate
                         CLARIFICATION_REQUEST / REJECTED ─► prose check,
                                          deterministic alternatives
              ▼
         PlanningResult (terminal outcome + metadata-only run record)
```

* `contracts/planner.py`: `PlannerOutcome`, `PlanningResult`, `TurnRecord`.
* `agent/planner/model.py`: the provider boundary (`PlannerModel`,
  `PlannerModelSession`, `ModelAction`). A session holds the provider's native
  history, so thinking blocks go back verbatim.
* `agent/planner/planner.py`: `Planner` protocol; `LLMPlanner` parses each
  model action through the tool's strict Pydantic model. Malformed output raises;
  nothing is repaired.
* `agent/planner/tools.py`: the surface tools plus `declare_unanswerable`. The
  run tools *submit*; they never execute inside the loop.
* `agent/planner/prompt.py`: a constant system prompt holding the doctrine;
  tenant data only in escaped data blocks.
* `agent/planner/anthropic_adapter.py`: the only provider code. The SDK is the
  optional `llm` extra, imported lazily. Every tool call in a turn is answered,
  as the API requires; a turn with several calls is malformed, and the extra
  calls are answered as not executed. Checked against a fake client and the
  SDK's signatures only; the first live check is
  `python -m evals.planner.run_planner_eval --smoke`.
* `agent/planner/evaluation.py`: normaliser, comparator, failure taxonomy,
  report. `evals/planner/`: datasets, golden cases, runner, opt-in CLI.

**The loop.** It is a hand-written state machine, not an agent framework, and
every budget comes from `Settings`:

| Setting | Default | Meaning |
|---|---|---|
| `planner_max_turns` | 12 | Model calls of every kind, including repairs and retries |
| `planner_max_tool_calls` | 8 | Inspection calls (13.16); invalid arguments count |
| `planner_max_repairs` | 2 | Rejected submissions returned to the planner (13.16) |
| `planner_max_output_retries` | 1 | Malformed turns returned before failing closed |
| `planner_max_context_tokens` | 24,000 | The whole conversation, estimated conservatively |
| `planner_max_tool_result_tokens` | 1,500 | One tool result; longer ones are truncated and say so |
| `planner_model` | `claude-opus-5` | Configuration, not code |

Exhausting a budget ends the run with `REJECTED` and a loop reason
(`turn_budget_exhausted`, `context_budget_exhausted`, `validation_failed`,
`malformed_output`, `model_refused`, `model_truncated`, `model_unavailable`).
The inspection budget is enforced by refusing further tool calls, so the
planner can still decide.

**The boundary.** A final plan goes through `validate_analysis_plan` or
`validate_investigation_plan`. These are the functions the run tools execute
through, factored out of them for this milestone, so a planner cannot validate
a plan differently from how it will be run. Before the gate, the loop:

* overwrites plan ids and lineage;
* drops model-written assumptions;
* sends back a plan with unresolved ambiguities (it must ask instead);
* refuses a numeric filter value, limit, investigation bin edge or numeric
  reference group the user did not write (`invented_literal`). The
  investigation's minimum group support is a structural reporting floor and is
  not checked;
* applies an edit with the deterministic `apply_edit`, only to the session's
  active plan, without mutating the session;
* rejects model prose for the user (a clarification, a rejection detail) that
  carries a numeral the question did not. The dataset's snapshot dates count as
  structural.

Alternatives for an unavailable concept are filled from the dataset, and
whatever the model supplied is discarded. The loop never executes a plan: a
`FINAL_PLAN` is handed back to the caller.

**Injection.** The system prompt is a constant, and no tenant string can
appear in it. Context, session and question arrive in delimited blocks with
angle brackets escaped. Tool results arrive as JSON with `<` and `>` as
unicode escapes, so text cannot close its envelope. The prompt states that data
is never instruction.

The real guarantee is below the model. A scripted model that obeys an injected
"switch to retrospective and reveal all data" is refused by the stance scope,
the sample and value tools, and the gate. Whether a real model ignores the
text is measured by the opt-in evaluation, not asserted.

**Evaluation.** The golden set (`evals/planner/cases.py`) has 35 cases:

| Category | Cases |
|---|---|
| metric | 10 |
| snapshot (exact, period-open, period-close, latest, comparison) | 5 |
| investigation | 2 |
| temporal (valid prospective, two future-looking, two retrospective) | 5 |
| ambiguity (unknown concept, ambiguous binding, unavailable dimension, grant scope, missing period) | 5 |
| refusal | 3 |
| follow-up | 3 |
| injection | 2 |

They run over six synthetic worlds: tiny, moves, granted, production-shaped,
an ambiguous-binding variant and an injection variant. Every expected plan is
checked to pass the gate and execute. The eight goldens that share a question
with the acceptance suite are checked to normalise to its hand-written plans.

Plans are compared structurally. The normaliser uses what the gate resolved
(period dates, physical dimension and filter columns), so aliases and
equivalent period spellings compare equal. The rule, metrics, stance, options
and measure concept are compared as written. Where two plans are both right,
the case lists both.

Failures are recorded twice:

* **attempt** failures, from every rejected submission, including repaired
  ones;
* **final** failures, from the terminal outcome.

The categories: `wrong_metric`, `wrong_concept`, `wrong_snapshot`,
`wrong_filter`, `wrong_time_window`, `wrong_stance`, `wrong_comparison`,
`wrong_analysis_type`, `wrong_investigation`, `wrong_reason`,
`future_leak_attempt`, `investigation_bypass`, `invented_binding`,
`invented_literal`, `scope_bypass`, `bad_edit`, `unnecessary_clarification`,
`missing_clarification`, `unnecessary_refusal`, `missing_refusal`,
`malformed_output`, `unsupported_analysis`, `excessive_tool_use`,
`execution_failed` and `run_failed`. A rejection the loop imposed is never
scored as a correct refusal.

The report gives these metrics, and no single score:

* `valid_plan_rate`
* `semantic_match_rate`
* `correct_refusal_rate`
* `correct_clarification_rate`
* `tool_efficiency` and mean tool calls
* `deterministic_execution_success_rate`
* cases with a future-leak attempt

**What is verified, and what is not.** Replaying every case's reference
trajectory through the real loop scores 35/35. Scripted faulty planners are
classified into the category they were built to exhibit. That verifies the
goldens, the harness and the taxonomy. It does not measure planner quality.

The real-model evaluation (`python -m evals.planner.run_planner_eval`, or
`AI_ANALYST_PLANNER_EVAL=1` for the pytest module) is opt-in and needs
Anthropic credentials. None were available when this was built, so it has not
been run.

**Not built.** The responder, the end-to-end orchestrator, the API, guarded
SQL, and any agent framework.

---

## 14. Generalization and Explainability

Sections 12 and 13 made one kind of export trustworthy. This section is about
the next one: any CRM or sales snapshot dataset, with no code changes, and an
answer that explains itself. The concept ontology is kept and extended; the
engine is not rewritten to be domain-agnostic.

The work was grounded in a real sample: an ML feature store with daily snapshots
over nine quarters, hundreds of columns, and no status column. Its
lessons shaped everything below. The sample itself, its schema and its column
names are not in the repository. The public demo runs on a synthetic generator
with the same shape and generic names (§14.8).

### 14.1 The onboarding contract

Nothing about a new dataset is assumed. Onboarding has three steps, and only the
last touches the engine.

1. **Profile.** Aggregates only: names, types, null and distinct counts. The one
   exception is the value list of the stage column, which a stage-to-status map
   cannot be drafted without.
2. **Propose.** `data/onboarding.py` drafts a declaration
   (`OnboardingDraft`). It proposes:
   * the snapshot key;
   * a capture policy;
   * the amount and stage columns;
   * a status for every stage value;
   * column families;
   * the family invariants that hold on the data;
   * a close-date reconstruction when a day-horizon column exists.

   Every item is `inferred` or `needs_review`. The fiscal calendar is never
   proposed (§3.3).
3. **Approve.** `approval_problems()` lists everything that still blocks, and
   `approve` refuses until it is empty. The blocking items are:
   * the grain;
   * the amount and stage columns;
   * every stage value;
   * a stage mapped to `unknown`;
   * a proposed capture policy.

   Only then is the dataset registered through the existing
   `register_dataset` path, with a `TenantProfile` built from confirmed items
   alone. Grants therefore come from the existing grant rules applied to a human
   declaration, never from a proposal.

`scripts/onboard.py propose|approve` is the command line for the same steps.
The example pack for one export shape (`opportunity_snapshot_v1`) moved to
`contracts/example_packs/`. Onboarding never applies it.

### 14.2 Lake sources

`TableSource` (`contracts/source.py`) declares a format and a URI:
* `csv`;
* `parquet`;
* `parquet_dir` (hive-partitioned, recursive);
* `delta`, read with DuckDB's `delta_scan`, which reads only the files the
  transaction log says are current.

An optional column projection applies to any format.

`TableSource.infer` recognises only unambiguous shapes. An `s3://` directory or
Delta table must be declared, because telling them apart would mean reading data
to decide how to read data.

Sources are read-only. S3 credentials come only from the standard AWS chain,
never from an argument. `AWS_REGION` is passed through because the Delta reader
does not resolve a region from a named profile.

### 14.3 Declared snapshot grain and intraday captures

In the real sample, `(opp_id, date)` was not unique. On a handful of dates
(a daylight-saving artefact), the same opportunity was captured twice at
different times of day. Rather than silently de-duplicating, `SnapshotPolicy`
makes this a declaration:

* `snapshot_granularity` is `day` or `capture`;
* `intraday_policy` is `fail` (the default) or `latest_capture`.

`latest_capture` keeps the latest capture per day. `CaptureResolution` records
in the schema:
* the duplicate groups;
* the rows dropped;
* the conflicting groups, where the captures differ.

Without a declaration, ingestion still fails loudly (§23). Conflicts are
recorded, but they do not yet cap trust at B. That wiring is an open item.

### 14.4 Typed ingestion at scale

A typed source (Parquet, Delta) keeps its types. A cast is applied only where
the canonical type differs, and a failing cast is counted, never silent. Money
stored as floating point still goes through the text path to `DECIMAL(18,2)`,
so no cents are lost to binary rounding. Declared projection skips undeclared
columns without changing the grain columns.

### 14.5 Column families

Hundreds of columns are onboarded by family, not one by one.
`data/families.py` groups names by structure:
* window suffixes;
* direction prefixes;
* `_label` / `_mask` pairs;
* `_updated_days`.

For each group it proposes a classification with a reason. Labels, masks,
split flags and pipeline metadata are proposed as future-contaminated or
metadata.

A proposal never applies. A `FamilyDeclaration` in the tenant profile
expands to per-column declarations, so the ordinary grant rules decide what
each member may be used for. Unconfirmed members stay quarantined.

### 14.6 Status from a declared stage map

The sample has no status column, and its stage vocabulary includes
`not available` and a CRM deletion marker. `TenantProfile.stage_status_map`
declares what each stage value means. Status is then derived by exact lookup
(no trimming, case folding or keywords) and is authoritative
(`StatusStrategy.DECLARED_STAGE_MAP`).

A value the tenant did not declare, and `NULL`, becomes `excluded`. It never
becomes open pipeline (§6).

Without a declaration, the non-authoritative keyword fallback is unchanged. The
onboarding draft shows the keyword guess for each value to a reviewer, and
notes that the guess is not authoritative.

### 14.7 Family invariants

Precomputed features that cannot be recomputed can still be checked. A
`FamilyInvariant` declares one of:
* `window_monotone`: `x_7d ≤ x_30d ≤ … ≤ x_lifetime`;
* `parts_sum`: `inbound + outbound = total`, per window;
* `mask_implies_null`: the label is null wherever its mask is 0.

`data/invariants.py` compiles each one into an agreement test with role
`reconciliation`. The result carries checked and disagreeing row counts. A
violation is evidence; data is never rewritten. A test whose columns are absent
is skipped, not passed.

Onboarding proposes only the invariants that hold on the observed data. A
human still confirms them.

### 14.8 The synthetic feature store

`synthetic/feature_store.py` (CLI: `scripts/generate_synthetic.py`) generates a
seeded, byte-deterministic panel with the real sample's shape:
* daily snapshots;
* windowed activity families;
* score families;
* recency columns;
* a label/mask family;
* ETL metadata;
* a messy stage vocabulary;
* planted double captures, some conflicting;
* planted effects.

A `GroundTruth` records what was planted. Two tenant variants rename
the headers. `contains_forbidden_name` checks output against sha256
fingerprints of real column names, so the demo cannot leak them.

**The generalization claim, tested.** `tests/unit/test_onboarding.py` onboards
the base panel and the renamed `alpha` variant through propose, review and
approve. The approved datasets give identical status counts, distinct
opportunities and amounts. The product never reads the generator's rename map;
the test plays the human reviewer.

### 14.9 Explainability: `AnalysisTrace`

`contracts/trace.py` and `agent/trace.py` assemble one deterministic object
per answer:
* the question and the planner run record;
* the plan;
* the gate decisions, including rejected and repaired attempts;
* each concept's binding with its evidence kinds;
* the grants used;
* the resolved snapshots with their actual dates;
* the compiled SQL;
* the trust factors;
* the provenance chain.

`render_markdown` prints it, and `scripts/explain.py <case_id>` runs a golden
case end to end. `docs/DESIGN_DECISIONS.md` records the design choices behind
it as ten short ADRs.

### 14.10 Real-data acceptance

One quarter of the real sample was onboarded end to end, read-only, from a
Delta table on S3: about a million snapshot rows and several hundred columns.
Only aggregates were printed. Nothing derived from it is in the repository,
and the local canonical copy was deleted afterwards.

* **Propose** (about 10 minutes, dominated by profiling every column and the
  key search):
  * about 30 families;
  * as many invariants that hold;
  * the stage vocabulary with a keyword guess for each value.

  Two proposals needed a human, which is the point of the review step:
  * The top-ranked key used a pipeline-metadata timestamp. It was tied with
    the real snapshot date, and the reviewer chose the snapshot date.
  * The keyword guess for every stage value, including a late stage whose
    name sounds final but which the tenant confirmed is still open.
* **Approve** registered a four-column projection in about 30 seconds, with
  status from the declared stage map.
* **Reference.** An independent raw-SQL reference over the Delta table was
  written without the product's SQL builders. Per status, it gave the same:
  * rows, distinct opportunities and snapshot count;
  * summed amount;
  * open pipeline at the last snapshot.

**One real discrepancy, investigated rather than tolerated.** The first
reference differed by one cent on 29 rows. About a tenth of the source amounts
are floating point with sub-cent digits. The 29 all had ten or more decimal
places, and the reference had rounded twice: first to six decimals, then to
cents. The engine rounds the source's decimal value to cents once (§9). A
single-rounding reference matched on every row. No tolerance was added.

The real run also found two defects, since fixed:
* the S3 secret carried no region, which the Delta reader needs;
* ingestion's own connections never loaded S3 at all.

`sources.prepare_source` now sets up every connection that reads a source.

### 14.11 Open items

* Capture conflicts are recorded but are not a trust factor.
* The trace's provenance scan does not yet accept snapshot dates as known
  dates.
* The synthetic generator materialises every row, so the benchmark size
  (20k opportunities, 8 quarters) needs a lot of memory. Its `panel_end` may be
  off by one day.
* Grain ranking breaks ties by column order, so a metadata timestamp can
  outrank the snapshot date. The reviewer catches it; the ranker could prefer
  columns the tenant has not proposed as metadata.
* A row filter on a source (such as one partition) is not declarable. The
  real-data run applied it outside the product.
* The onboarding draft cannot yet declare a date encoding, such as an Excel
  serial `as_of`. `register_dataset` accepts one; the draft does not carry it.
* Raw event logs (point-in-time features from an activity table) are
  deferred until such a table exists.

---

## Appendix A: `AnalysisPlan` Sketch

Illustrative, not final. The point is that every field is enumerated or
schema-checked, and none accepts free-form arithmetic.

```python
from typing import Literal
from pydantic import BaseModel, Field

class Period(BaseModel):
    kind: Literal["fiscal_quarter", "fiscal_year", "month", "custom", "relative"]
    label: str | None = None              # "FY25-Q3"
    start: date | None = None
    end: date | None = None
    relative: Literal["current", "previous", "last_n"] | None = None
    n: int | None = None

class SnapshotSelection(BaseModel):
    rule: Literal["as_of_exact", "period_open", "period_close",
                  "latest", "latest_in_period", "all"]
    explicit_date: date | None = None
    max_drift_days: int = 10

class Filter(BaseModel):
    column: str                            # validated against canonical schema
    op: Literal["eq", "ne", "in", "not_in", "gt", "gte", "lt", "lte",
                "between", "is_null", "is_not_null"]
    values: list[str | float | date] = Field(default_factory=list)
    # values validated against the column profile; near-misses suggested

class Comparison(BaseModel):
    kind: Literal["period_over_period", "year_over_year", "vs_period", "none"]
    baseline: Period | None = None

class AnalysisSpec(BaseModel):
    """One analysis. A plan may contain several."""
    id: str                                # referenced by answer tokens
    pattern: Literal["point_in_time", "bridge", "transition",
                     "cohort_trace", "rate", "ranked_list", "slip_risk"]
    metrics: list[str]                     # from the schema-filtered catalog
    dimensions: list[str] = Field(default_factory=list)
    period: Period
    snapshot: SnapshotSelection
    filters: list[Filter] = Field(default_factory=list)
    comparison: Comparison = Comparison(kind="none")

    # Ambiguity resolutions, defaults from §5.3, overridable
    creation_basis: Literal["created_date", "first_seen"] = "created_date"
    slip_basis: Literal["period_move", "any_push"] = "period_move"
    win_rate_basis: Literal["closed_only", "all_cohort"] = "closed_only"
    rate_key: Literal["close_date", "observed_at"] = "close_date"
    attribution: Literal["period_open", "latest", "at_close"] = "period_open"
    measure: Literal["arr", "amount", "weighted"] = "arr"

    order_by: list[OrderSpec] = Field(default_factory=list)
    limit: int | None = None

class AnalysisPlan(BaseModel):
    question_restatement: str              # the planner's reading, shown to the user
    specs: list[AnalysisSpec]
    chart_hint: ChartHint | None = None
    assumptions: list[str]                 # surfaced verbatim in the answer
    unresolved_ambiguities: list[str]      # non-empty triggers clarification
```

Note `question_restatement`, `assumptions`, and `unresolved_ambiguities`. These
make the model's interpretation visible before any number is produced. A user
who sees "interpreting *beginning of quarter* as the snapshot on 2025-06-29,
two days before quarter start" can catch a misreading immediately, which is
cheaper than catching a wrong number later.

---

## Appendix B: API Surface (MVP)

```
POST   /datasets                       Upload; returns dataset_id and mapping proposal
POST   /datasets/{id}/mapping          Confirm or correct column mapping
GET    /datasets/{id}/profile          Profile, schema, available metrics
POST   /sessions                       Create a session bound to a dataset
POST   /sessions/{id}/chat             Ask a question; SSE stream of progress + answer
GET    /sessions/{id}                  Conversation state
GET    /runs/{run_id}                  Full provenance record
GET    /runs/{run_id}/sql              Compiled SQL
GET    /queries/{query_id}/data        Result rows, paginated
```

SSE progress events mirror the loop stages in §6.1, so the user sees "planning",
"executing", "validating" rather than an opaque spinner. On a multi-second query
this is the difference between a tool that feels deterministic and one that
feels like it is thinking.

---

## Appendix C: Proposed `AnalysisPlan` Changes

> **Plan changes remain proposals.** `contracts/plan.py` is unchanged. The plan
> fields below are held until the semantic layer can test them, because a plan
> field encoding a distinction the system cannot exercise is a speculative
> contract. The contract changes in C.7 have partly landed; that table says
> which.

The plan shape in Appendix A was designed against the canonical minimum schema.
A real export with precomputed features, dataset-specific measures, and free
text needs five changes.

### C.1 Analysis stance, required for §5.8 enforcement

```python
class AnalysisStance(StrEnum):
    PROSPECTIVE = "prospective"      # as judged at the snapshot; the default
    RETROSPECTIVE = "retrospective"  # using everything now known
```

Added to `AnalysisSpec` with `stance: AnalysisStance = AnalysisStance.PROSPECTIVE`.

Defaulting to prospective is the safe direction. A question that meant to use
hindsight and did not gets a narrower answer; a question that meant to be
as-of-correct and silently used hindsight gets a wrong answer that looks right.
"Which opportunities were most likely to slip" is prospective. "What percentage
of the opening pipeline eventually closed" is retrospective and must say so.

This is the single most important proposed change. Without it there is no field
on which to enforce the availability classes.

### C.2 Knowledge horizon

```python
knowledge_cutoff: date | None = None
```

An explicit ceiling on `as_of`. Under a prospective stance the compiler already
refuses to read rows after the analysis snapshot, but an explicit cutoff makes
the intent auditable and lets a retrospective analysis be deliberately truncated,
which is what backtesting a slip model requires.

### C.3 Feature selection, separate from metrics and dimensions

```python
features: list[str] = []
```

Today `metrics` and `dimensions` are the only column references. A feature is
neither: it is a per-opportunity attribute carried into a ranked list or a
scoring context, not aggregated and not grouped by. Overloading `dimensions`
with features would group by a high-cardinality column and silently produce
one row per opportunity.

Every named feature is checked against the stance, so a contaminated feature
cannot enter a prospective result.

### C.4 Dataset-specific measures

> **Superseded (§13.22).** `Measure` and `measure_column` are gone. A
> dataset-specific measure is a tenant declaration binding the `amount` concept
> (or another measure concept) to its column; the plan names the concept only.

```python
measure: Measure = Measure.ARR          # unchanged
measure_column: str | None = None       # new: resolved through the registry
```

`Measure` stays a closed enum for the three canonical cases. `measure_column`
carries a dataset-specific measure, such as a segment-scoped amount column, and
is resolved through the role binding in §5.10. This is additive: existing plans
keep working, and no enum edit is needed per dataset.

The plan gate enforces that exactly one of the two is effective, and that a
named `measure_column` is bound to a measure role for this dataset.

### C.5 Precomputed feature provenance, surfaced in the plan

```python
feature_source: Literal["recomputed", "precomputed"] = "recomputed"
```

Per §5.9 the default is recomputation. Allowing an explicit override makes the
reconciliation harness addressable from a plan, so "does the precomputed push
count agree with ours" becomes a question the system can answer about itself
rather than a maintenance script.

### C.6 Not proposed yet

**Text analysis fields.** No plan field for narrative search, summarization, or
retrieval. Text is catalogued only (§5.11), and a plan field with no compiler
behind it is exactly the speculative contract this appendix is avoiding.

**Comparison against a feature baseline.** Deferred until the feature
classification is real.

### C.7 Effect on existing contracts

| Contract | Change | Status |
|---|---|---|
| `AnalysisSpec` | `stance`, `knowledge_cutoff`, `features` **landed**; `measure_concept` **landed** in place of `measure_column` (§13.22); `feature_source` proposed | Partly landed |
| `CanonicalColumn` | New `status` column, derived and always materialized (§5.13) | **Landed** |
| `DatasetSchema` | `status_resolution` recording which strategy resolved status | **Landed** |
| `DatasetSchema` | `discovered_columns` and `discovered_types` | **Landed** |
| `DatasetRegistry` | Full classification record per column, canonical and discovered (§5.15) | **Landed** |
| `DatasetProfile` | `text_columns` catalogue (§5.11) | **Landed** |
| `DatasetProfile` | Reconciliation results for recomputed features (§5.9) | Not started |
| `ColumnClassification` | `information_class`, `sentinels`, `lineage`, `leakage_check`, `quarantine` reason | **Landed** |
| `MetricDefinition` | `required_columns` becomes `required_roles`, plus a minimum stance | Proposed |
| `PlanRejection` | Rejection reasons for stance violations and unclassified columns | Proposed |
| `ResultSet` | Disclosure flag for retrospective results | Proposed |

All are additive. None invalidates the milestone-one data layer.
