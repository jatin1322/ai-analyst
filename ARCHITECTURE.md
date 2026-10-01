# AI Analyst: Architecture

An analyst that answers natural-language questions about sales pipeline data,
such as "what was Q3 opening pipeline?", "why did pipeline slip?" or "which
deals went quiet before they pushed out?", over any CRM snapshot dataset. The
answers are trustworthy enough to act on.

This document explains how the system works today. The full design record,
with every decision's history, is in [docs/DESIGN_LOG.md](docs/DESIGN_LOG.md).
Section numbers cited in code comments refer to that file. Short design
decision records are in [docs/DESIGN_DECISIONS.md](docs/DESIGN_DECISIONS.md).

---

## 1. The central idea: plan-as-data

Most "chat with your data" systems are text-to-SQL: the model writes SQL, the
system runs it, and nothing in between can be checked. Here the model never
writes SQL or arithmetic. It emits a **typed plan**, and deterministic code does
everything else.

```
Question ─▶ LLM planner ─▶ typed plan ─▶ gate ─▶ compiler ─▶ DuckDB ─▶ checked result
                               ▲            │
                               └─ structured rejection (bounded repair)
```

| | Plan-as-data | Free-form SQL |
|---|---|---|
| Validation | Before execution, against the dataset | Run it and hope |
| Snapshot semantics | Encoded once in the compiler | Re-derived by the model every time |
| Determinism | Same plan, same SQL, same number | Varies run to run |
| Audit | Plan, SQL and result form a full record | A SQL string |

**The one boundary that matters.** The LLM appears only in the planner (and,
later, the responder). The data layer, semantic engine, gate, compiler, trust
and rendering contain no LLM calls. The model decides *what* to compute; code
decides *how*, and is the only source of numbers.

---

## 2. End to end

```
              ┌──────────────────── LLM ─────────────────────┐
 question ──▶ │ planner: inspects the dataset through tools,  │
              │ submits a plan, a clarification, or a refusal │
              └───────────────────────┬───────────────────────┘
                                      │ AnalysisPlan / InvestigationPlan
 ─────────────────────────── deterministic from here ───────────────────────────
                                      ▼
   gate ──▶ compiler ──▶ DuckDB ──▶ ResultSet ──▶ trust tier ──▶ renderer ──▶ answer
    │                                  │                            │
    rejects with a typed reason        every cell addressable       numbers substituted
                                       by (query, row, column)      from results only
```

Two paths lead to a result:

* **Semantic path**, for known questions. A metric from the registry (opening
  pipeline, win rate, slipped pipeline and so on) runs through the
  `AnalysisPlan`. Results can reach trust tier A.
* **Investigation path**, for novel questions. An `InvestigationPlan`, drawn
  from a closed vocabulary, runs through its own gate and compiler. Results are
  at most tier B. If a semantic metric already answers the question, the gate
  rejects the investigation so it cannot be used to bypass the stronger path.

---

## 3. Understanding a new dataset

The same engine must work for any tenant's export. That rests on one
separation.

### Concepts are stable; columns are not

Metrics are written against **business concepts** (`amount`, `stage`,
`opportunity_status`, `expected_close_date`, `owner` and so on), never against
physical columns. A **binding** maps a concept to a tenant's column. One tenant's
`amount` is `deal_value`, another's is `booking_value`.

A binding has a status: `confirmed`, `inferred`, `ambiguous` or `unavailable`.
**A name match never confirms a binding.** Neither does an alias, a data type or
a correlation. Only a tenant declaration does. An inferred binding can be used
only where the resulting trust cap is acceptable. An ambiguous one is never
silently resolved.

### Usage grants

Confirming what a column *means* is different from permitting what it may be
*used for*. A declaration issues an explicit, scoped, persisted **usage grant**,
for example "this column may be used as the amount measure". Grants are written
to a ledger and audited. Inferred bindings never create grants.

### Onboarding: profile, propose, approve

```
python -m scripts.onboard propose <source> --out draft.json   # nothing confirmed
# a human reviews and confirms items in the draft
python -m scripts.onboard approve draft.json --dataset-id X   # refused until confirmed
```

The proposer profiles the source (aggregates only) and drafts:

| Proposal | What a human confirms |
|---|---|
| Snapshot key, e.g. `(opp_id, as_of)` | The key columns |
| Capture policy, when one opportunity is captured twice a day | `latest_capture`, or keep the default hard failure |
| Amount and stage columns | The columns |
| A status for every stage value | Each value: open, won, lost or excluded |
| Column families (hundreds of columns grouped by name structure) | A classification per family |
| Invariants that hold on the data | Which to keep as ongoing checks |

Approval is refused, with every blocking reason listed, until the load-bearing
items are confirmed. Only then is the dataset registered and grants issued.
**Proposals never apply themselves.**

### What onboarding handles

* **Sources.** CSV, Parquet, partitioned Parquet and Delta tables, local or on
  S3. Always read-only; S3 credentials come only from the standard AWS chain.
* **Column families.** Window suffixes (`_7d … _lifetime`), direction prefixes
  (inbound, outbound, all), label and mask pairs, recency columns. One
  declaration covers a whole family. Unconfirmed columns stay quarantined.
* **Status from stage.** Many exports have no status column. A declared
  **stage→status map** is authoritative, using exact values only. A stage nobody
  declared, including `NULL` and junk values such as "not available", is
  `excluded`. It never counts as open pipeline.
* **Intraday captures.** A duplicate `(opportunity, day)` fails ingestion unless
  the tenant declares `latest_capture`. The resolution is recorded: groups
  resolved, rows dropped, and groups whose captures disagreed.
* **Family invariants.** Precomputed features that cannot be recomputed can
  still be checked. The declarable invariants:
  * window monotonicity: `x_7d ≤ x_30d ≤ x_lifetime`;
  * parts summing to a whole: `inbound + outbound = total`;
  * a mask implying a null label.

  Violations are reported with row counts and never repaired.

---

## 4. The semantic engine

The correctness core. Deterministic primitives, each tested against
hand-computed values.

**Snapshot grain.** A row is one opportunity at one snapshot, `(as_of, opp_id)`.
Counting rows counts observations, not opportunities, so every metric states
its snapshot semantics.

**Snapshot selection.** The rules are `AS_OF_EXACT`, `PERIOD_OPEN`,
`PERIOD_CLOSE`, `LATEST`, `LATEST_IN_PERIOD` and `ALL`. Every result carries
the *actual* snapshot date used. "Quarter open" resolved to a snapshot six
days early says so.

**Fiscal calendar.** Configured explicitly, never inferred from labels. A
stamped quarter column is reconciled against the calendar, not trusted.

**Metrics** (12): `deal_count`, `opening_pipeline`, `ending_pipeline`,
`created_pipeline`, `slipped_pipeline`, `pulled_in_pipeline`, `won_pipeline`,
`lost_pipeline`, `win_rate`, `average_deal_size`, `pipeline_coverage`,
`cohort_fate`. Each declares the concepts it needs and is unavailable when
they are not bound.

**The pipeline bridge** is the system's strongest self-check:

```
opening + created + pulled_in + increased − decreased
        − won − lost − slipped − other_removed  =  ending
```

All nine terms are computed independently. In particular, `other_removed` is
never derived as the residual, which would make the identity balance by
construction. A bridge that does not balance is a bug, and the tolerance is
never widened to hide one.

**Money.** Monetary measures are cast to `DECIMAL(18,2)` from the source's
decimal text, rounded once. Division uses an exact-divide helper. Money is
never averaged as floating point.

**Dates.** Conversions (such as Excel serial dates) are declared per column,
never inferred from numeric shape. An invalid declared date fails ingestion.

---

## 5. The investigation path

For questions no metric covers, such as "do deals whose forecast category
changed close at a different rate?". An `InvestigationPlan` is data, not code.
It combines a closed vocabulary:

* **population and time window**, bounded by the stance;
* **variables**: concepts, or derived features such as `value_at_start`,
  `value_at_end`, `change_count` and `changed`, with roles outcome, exposure or
  covariate;
* **binning** at explicit edges;
* **operation**: `count`, `sum`, `distribution`, `crosstab`, `rate_by_group`,
  `difference_in_rates`, `rank_correlation` or `trend`;
* **comparison** against a reference group;
* **evidence requirements**, such as minimum group sizes.

No formulas, no `eval`, no free-text expressions. If a question needs arbitrary
code, the contract is extended by a reviewed change. Results are at most tier
B, and are described as associations, not causes.

---

## 6. Temporal safety

The most dangerous failure in pipeline analytics is quietly using the future.

* **Stance.** Every plan is `prospective` ("what was knowable then", the
  default) or `retrospective` ("what we know now").
* **Availability classes.** Every column is `AS_OF_FACT`, `BACKWARD_DERIVED`,
  `FUTURE_CONTAMINATED` or `UNKNOWN`. Prospective analysis may use only the
  first two. `UNKNOWN` fails closed.
* **Knowledge horizon.** A prospective analysis at snapshot `T` reads no row
  after `T`, including in joins, attribution, profiling and helper queries.
* **ML labels and outcomes.** Labels, masks, train/test flags and terminal
  outcome columns are proposed as future-contaminated, so they cannot be used
  to "predict" the thing they encode.

These rules are enforced three times: when building the planner's context, at
the gate, and at compilation. None of them relies on the prompt.

---

## 7. Trust

Trust is computed, never chosen by the model. It is the weakest of its inputs.

| Tier | Meaning | Answer |
|---|---|---|
| **A** | Registry metric, confirmed bindings, no open evidence issues | The number |
| **B** | Investigation, inferred binding, unconfirmed lineage, reconciliation warning, retrospective data | The number, with the reason it is B |
| **C** | Missing concept, unresolved ambiguity, failed validation | No number: a clarification or an explanation |

Each result carries a `TrustAssessment`: the factors that lowered it, with
evidence. A tier-B answer must say why it is B.

---

## 8. The LLM planner

* **Outcomes.** Every run ends in exactly one typed outcome:
  * `FINAL_PLAN` (only after the gate has passed the plan);
  * `CLARIFICATION_REQUEST`, as typed plan edits drawn only from concepts the
    dataset has;
  * `REJECTED`, with a typed reason.
* **Tools.** Twelve narrow, structured tools:
  * inspect the dataset, a concept, a column, values, a relationship or
    sample rows;
  * list metrics;
  * probe materiality;
  * run an analysis or an investigation;
  * request clarification;
  * declare a question unanswerable.

  Tool output respects stance, horizon, grants and row limits. Raw data never
  enters the model context.
* **Loop.** A hand-written state machine, with no agent framework. Turns, tool
  calls and repairs are all bounded by settings. A rejected plan returns a
  structured reason for repair.
* **Follow-ups** ("break that down by owner") are typed edits to the prior
  plan, with a carry-forward report of what was kept and what changed.
* **Model.** Behind a provider-independent interface. The Anthropic adapter is
  an optional dependency.
* **Evaluation.** 35 golden cases across six synthetic worlds, scored by
  structural comparison into a failure taxonomy: wrong metric, future-leak
  attempt, invented binding, unnecessary clarification and others. There is no
  single score.

---

## 9. Explainability and provenance

**Every number is traceable:**

```
answer ─▶ reference token {{q1.r0.opening_pipeline}} ─▶ result cell ─▶ compiled SQL ─▶ dataset
```

The responder (not yet built) will write prose with reference tokens, never
literal numbers. The renderer substitutes values from results. A provenance
scanner flags any number in the answer that is not a result, a user-supplied
value, or a structural value such as a year.

**`AnalysisTrace`** explains one answer end to end:
* the question and the plan;
* the gate decisions, including rejected and repaired attempts;
* each concept's binding and its evidence;
* the grants used;
* the actual snapshots selected;
* the compiled SQL;
* the trust factors;
* the provenance chain.

```
python -m scripts.explain metric_opening_q2
```

---

## 10. Data quality rules

Ingestion fails loudly on:
* duplicate grain keys;
* null keys;
* invalid declared dates;
* contradictory declarations.

Failed casts are counted, never silent. The states missing, null, invalid,
unavailable, unknown and contradicted stay distinct. Data-quality errors are
never downgraded to warnings to make ingestion succeed.

---

## 11. Code map

| Package | Responsibility |
|---|---|
| `contracts/` | Typed Pydantic contracts: plans, results, concepts, bindings, trust, tenant declarations |
| `data/` | Sources, ingestion, conformance, classification, families, bindings, grants, agreement tests, onboarding |
| `semantic/` | Calendar, snapshots, cohort, transitions, bridge, metrics, gate, compiler, execution, investigation, trust |
| `session/` | Conversation state and typed plan edits |
| `validation/` | Rendering and the provenance scanner |
| `agent/` | Planner context, tool surface, planner loop and model adapter, analysis trace |
| `synthetic/` | Seeded feature-store generator with renamed-header tenant variants |
| `scripts/` | `onboard`, `explain`, `generate_synthetic`, `probe_production`, `bench_ingest` |
| `evals/planner/` | Golden cases, synthetic worlds, the evaluation runner |

**Stack:**
* core: Python, DuckDB, Parquet and Pydantic;
* optional: `anthropic`, for the planner adapter only;
* not used: LangChain, vector databases and orchestration infrastructure.

---

## 12. Status

| Built | Not built yet |
|---|---|
| Data layer, onboarding, lake sources | Responder (LLM prose with reference tokens) |
| Semantic engine, 12 metrics, bridge | End-to-end orchestrator |
| Investigation path | API and UI |
| Trust, rendering, provenance, `AnalysisTrace` | Guarded SQL fallback (designed, off by design) |
| LLM planner and its evaluation harness | Real-model planner evaluation run |

**Verified on real data.** One quarter of a real export (about a million
snapshot rows, several hundred columns) was onboarded read-only. It matched an
independent raw-SQL reference on every figure checked: rows, distinct
opportunities, snapshot count and summed amount per status, and open
pipeline at the last snapshot.

**Open items.**
* Conflicting double captures do not yet lower trust.
* A source row filter (such as one partition) cannot be declared.
* The onboarding draft cannot yet declare a date encoding.
* The ranking of proposed keys breaks ties by column order.
