# AI Analyst — Engineering Rules

## Project Goal

Build a production-quality AI Analyst capable of answering natural-language questions over multi-quarter opportunity snapshot data across heterogeneous tenants.

The system must be able to inspect an uploaded dataset, understand the concepts and information actually available, answer known analytical questions through deterministic semantics, and investigate novel questions through a constrained typed analysis path.

The system must remain trustworthy even when:

* tenant schemas differ,
* columns are missing,
* custom columns exist,
* precomputed features have uncertain lineage,
* quarter definitions differ,
* and some data represents future outcomes.

---

# 1. Core Principles

## 1.1 Numbers come from computation, not model reasoning

The LLM must never fabricate an analytical result.

Every quantitative claim in an answer must originate from:

1. deterministic semantic computation,
2. deterministic investigation computation,
3. executed guarded SQL when explicitly permitted,
4. or explicitly supplied dataset metadata.

The LLM may reason about:

* what should be investigated,
* which concepts are relevant,
* which tools to call,
* which analysis to perform,
* how to explain the result.

The LLM must never be the source of a computed number.

---

## 1.2 Plan-as-data, not text-to-SQL

The primary analytical interface is a typed plan.

Known analytical questions use:

```
User question
    ↓
LLM planner
    ↓
AnalysisPlan
    ↓
Plan Gate
    ↓
deterministic compiler
    ↓
DuckDB
    ↓
ResultSet
```

Novel questions use:

```
User question
    ↓
LLM planner
    ↓
InvestigationPlan
    ↓
Investigation Gate
    ↓
deterministic investigation compiler
    ↓
DuckDB
    ↓
ResultSet
```

The model must not write arithmetic inside a plan.

Do not introduce free-text arithmetic expressions into `AnalysisPlan` or `InvestigationPlan`.

If a plan needs arbitrary code to become executable, stop and redesign the contract.

---

## 1.3 Deterministic core / LLM boundary

The following contain NO LLM calls:

* data layer
* semantic layer
* concept resolution
* snapshot selection
* cohort construction
* transition logic
* bridge calculations
* metric calculations
* investigation compilation
* plan validation
* SQL compilation
* result validation
* trust calculation
* rendering
* provenance validation

The LLM is restricted to:

* planner
* responder

The orchestration/state machine may call the LLM, but it must not perform business calculations itself.

The responder must not calculate arithmetic.

---

# 2. Multi-Tenant Architecture

## 2.1 Concepts are stable; columns are not

Metrics and analytical logic must reference business concepts, never tenant-specific physical columns.

Example:

```
metric
  ↓
amount concept
  ↓
tenant binding
  ↓
new_amount
```

Another tenant may use:

```
amount
  ↓
ARR
```

Never hardcode one tenant's physical name into semantic metrics or the compiler.

---

## 2.2 Bindings require evidence

A `ConceptBinding` maps a stable concept to one or more tenant-specific physical columns.

Binding states include:

* confirmed
* inferred
* ambiguous
* unavailable

Column-name similarity alone can NEVER confirm a binding.

These are not sufficient for confirmation by themselves:

* exact name match
* alias
* fuzzy similarity
* data type
* value pattern
* statistical correlation

A human/tenant declaration or explicitly approved confirming evidence is required.

Inference remains inference.

Ambiguity remains ambiguity.

Never silently select one of multiple plausible columns.

---

## 2.3 Usage grants

Concept confirmation, physical classification, and analytical usability are separate concepts.

A tenant declaration may explicitly grant:

```
enterprise_amount
    → amount concept
    → usable as amount measure
```

This does NOT automatically make the physical column usable for unrelated purposes.

A grant must be:

* explicit,
* typed,
* scoped,
* persisted,
* auditable.

An inferred or fuzzy binding must never create a usage grant.

---

## 2.4 Agreement tests

Agreement tests are evidence, not automatic proof.

An agreement test must report:

* assertion
* role
* checked row count
* disagreeing row count
* sample disagreement information where permitted
* result status

Roles include:

* validity
* reconciliation

A validity failure can invalidate a derived concept.

A reconciliation failure generally becomes a warning unless the tenant has explicitly declared that field as part of the validity definition.

Never change a test from validity to reconciliation merely because it is failing.

---

# 3. Snapshot Data

## 3.1 Grain

The fundamental snapshot grain is:

```
(as_of, opp_id)
```

An opportunity may appear in many snapshots.

Never treat each snapshot row as an independent opportunity.

`COUNT(*)` over the raw snapshot table generally counts snapshot observations, not opportunities.

Any opportunity-level metric must define its snapshot/cohort semantics explicitly.

---

## 3.2 Snapshot semantics

Use the deterministic snapshot selector.

Supported rules include:

* `AS_OF_EXACT`
* `PERIOD_OPEN`
* `PERIOD_CLOSE`
* `LATEST`
* `LATEST_IN_PERIOD`
* `ALL`

Every result must preserve the actual snapshot date selected.

Do not report only a semantic label such as "quarter open" when the actual snapshot was several days before the quarter boundary.

Snapshot drift must remain visible.

---

## 3.3 Quarter semantics

Fiscal calendar configuration is explicit.

Do not infer the fiscal year start from:

* strings,
* model guesses,
* tenant naming conventions.

Quarter labels must be validated against the configured calendar before being treated as trustworthy.

A stamped quarter field may itself be future-contaminated if it was calculated using final outcome information.

---

# 4. Temporal Safety

## 4.1 Analysis stance

Every analysis has a stance:

```
prospective
retrospective
```

Prospective is the default.

### Prospective

"what was knowable at the time?"

Only information available at or before the knowledge horizon may be used.

### Retrospective

"what do we know now about what happened?"

Future/terminal information may be used when explicitly permitted.

The stance must be explicit in the plan.

---

## 4.2 Availability classes

Every governed column has an availability classification:

* `AS_OF_FACT`
* `BACKWARD_DERIVED`
* `FUTURE_CONTAMINATED`
* `UNKNOWN`

`UNKNOWN` is treated as unsafe for prospective analysis.

For prospective analysis:

* `AS_OF_FACT` is allowed
* `BACKWARD_DERIVED` is allowed
* `FUTURE_CONTAMINATED` is forbidden
* `UNKNOWN` is forbidden

The same restriction must be enforced:

1. before planning,
2. at the plan/investigation gate,
3. at compilation.

Do not rely on prompt instructions alone.

---

## 4.3 Knowledge horizon

Prospective analysis must not read rows after its knowledge horizon.

If the analysis is about snapshot `T`, rows with:

```
as_of > T
```

must not be used, even for:

* joins,
* attribution,
* profiling,
* dimension selection,
* helper calculations.

Never leak future information through an auxiliary query.

---

# 5. Precomputed Features

Precomputed features are evidence, not automatically trusted truth.

Classify them as:

* `RECOMPUTE`
* `USE_WITH_PROOF`
* `QUARANTINE`

When a value can be safely recomputed from raw snapshot history, the semantic engine should compute it itself.

If a precomputed value has an undocumented historical/lookback window, treat it as unsafe for prospective automatic metrics.

Examples of high-risk features include:

* rep win rates
* historical stage win rates
* overall percentiles
* versus-average features
* composite scores
* features derived from future outcomes

Do not use an outcome-derived feature to "predict" the same outcome without establishing its temporal boundary.

That is leakage and/or circularity.

---

# 6. Status and Stage

`status` and `stage` are separate concepts.

## Status

Status determines whether an opportunity is:

* open
* won
* lost
* excluded
* unknown

The authoritative status source is configurable per tenant.

Never assume the production status field by column-name similarity.

## Stage

Stage describes sales-process position.

Stage may be used for:

* distribution
* stage transitions
* progression
* regression
* dwell time
* stage-based analysis

Do not infer status from stage when an authoritative status source exists.

Stage keyword inference is a fallback only and is explicitly non-authoritative.

`SFDCDELETED` and `not available` must never silently become open pipeline.

---

# 7. Close-Date Reconstruction

The production export may not contain a physical close-date column.

Where explicitly declared:

```
expected_close_date = as_of + days_to_close
```

This is a declared derivation, not a guessed mapping.

The reconstructed concept must have an explicit reconstruction verdict:

* `VALID`
* `VALID_WITH_WARNINGS`
* `CONTRADICTED`
* `UNDECLARED`

Ancillary fields such as:

* `eoq_close_diff`
* `CD_in_qtr`
* `close_date_qtr`

do not become validity requirements unless their semantics are explicitly declared.

A reconciliation mismatch should generally become a warning and may cap trust at B.

Do NOT introduce blanket ±1 tolerances merely to make a test pass.

Any tolerance must be:

* explicitly defined,
* justified,
* tested,
* documented,
* tenant-aware where appropriate.

The cumulative `close_date_push_count` movement check must only be evaluated where the counter actually increases within the observed export window.

If there are no qualifying within-window changes, mark the evidence unverified/skipped rather than failing a valid dataset.

---

# 8. Dates

Date conversion is declaration-driven.

Excel serial dates must be declared by source header.

Never infer "this integer is an Excel date" from numeric shape.

Use the project's explicit Excel date conversion logic and preserve conversion provenance.

DuckDB casts that round values must not be used where truncation/flooring is the intended semantic.

Invalid declared date values fail ingestion.

---

# 9. Money

Money must not silently rely on binary floating-point arithmetic.

Policy:

* preserve source representation at rest,
* identify monetary semantics through classification/binding,
* explicitly cast monetary measures to `DECIMAL(18,2)` at measure resolution,
* use exact monetary arithmetic,
* never let `/` silently introduce DOUBLE,
* never use `AVG` over a monetary column if it causes float conversion.

Use the project's `sql.exact_divide` helper for exact division.

Do not infer that a numeric field is monetary merely because it has two decimal places.

Unconfirmed numeric fields should not expose floating-point statistics as monetary facts.

---

# 10. Semantic Engine

The semantic engine is the correctness core.

It implements deterministic primitives such as:

* `snapshot_at`
* `cohort`
* `trace`
* `transition`
* `bridge`
* `rate`

A metric must declare business concepts, not physical columns.

The only place a concept becomes a physical column is the tenant-aware resolver/binding layer.

Never put tenant physical names directly into:

* `metrics.py`
* `bridge.py`
* `compiler.py`
* analytical logic

---

## 10.1 Compiler boundary

The compiler accepts only validated gate output.

Do NOT create a compiler bypass.

Do NOT add a convenience path that accepts an unchecked `AnalysisPlan`.

Validation of:

* concepts,
* bindings,
* stance,
* columns,
* filters,
* periods

must happen before compilation.

---

## 10.2 Bridge

The pipeline bridge is a real invariant.

Every component must be independently computed.

Never derive `other_removed` as:

```
ending - everything_else
```

because that causes the check to balance by construction.

A bridge that fails its invariant is a defect unless the failure is explicitly explained by a documented semantic/data limitation.

Do not widen tolerances to make bridge failures disappear.

---

# 11. Investigation Path

Novel questions are allowed.

Do not reject a question simply because it is not in the metric registry.

The investigation path uses a typed `InvestigationPlan`.

It is a closed but composable vocabulary.

The plan should combine things such as:

* population
* time window
* variables
* derived features
* grouping
* statistical operation
* comparison
* evidence requirements
* analysis stance

The investigation plan is data, not arbitrary code.

Do not allow:

* arbitrary Python
* arbitrary formulas
* loops
* `eval`
* unrestricted expressions

---

## 11.1 Semantic path takes precedence

If an existing semantic metric can answer the question correctly, a novel investigation must not be used to bypass it.

The Investigation Gate should reject such a plan with a structured indication that the semantic path exists.

This preserves:

* stronger validation,
* stronger provenance,
* deterministic semantics.

---

## 11.2 Trust

Investigation results are at most trust tier B.

Trust is computed by deterministic code.

The model cannot:

* select the tier,
* override the tier,
* provide a trusted/untrusted flag.

A result involving:

* inferred evidence,
* unresolved lineage,
* reconciliation warnings,
* ad-hoc investigation,
* retrospective data

must have the appropriate trust consequence applied automatically.

---

# 12. LLM Planner

The planner converts natural language into a structured decision.

It may determine:

* intent,
* semantic vs investigation path,
* stance,
* knowledge horizon,
* concepts,
* features,
* dimensions,
* periods,
* comparisons,
* need for clarification.

The planner must NOT:

* compute results,
* write arithmetic,
* invent columns,
* invent concepts,
* invent metric definitions,
* choose trust tier,
* bypass gates.

A planner output that fails validation should produce a structured repair or clarification path.

Do not allow unbounded planner retries.

---

# 13. LLM Tools

Tools must return structured evidence.

The agent should be able to inspect the dataset through narrowly scoped tools such as:

* `inspect_dataset`
* `inspect_concept`
* `inspect_column`
* `inspect_values`
* `inspect_relationship`
* `inspect_sample_rows`
* `list_available_metrics`
* `run_analysis_plan`
* `run_investigation`
* `request_clarification`

Tool outputs must respect:

* analysis stance,
* knowledge horizon,
* usage grants,
* concept availability,
* row limits,
* allowed columns.

Under a prospective stance, inspection tools must not reveal future information through profiling or samples.

Never pass the raw dataset into the model context.

---

# 14. Context

The default analyst context is a compact projection of the dataset.

Tier 0 should contain:

* available concepts,
* concise definitions,
* physical bindings,
* binding status,
* high-level profile information,
* metric availability,
* important warnings,
* temporal safety metadata.

Do not place into Tier 0:

* full datasets,
* full profiles,
* large row samples,
* raw CRM text.

Tier 1 may expose detailed metadata for explicitly requested concepts/columns.

Tier 2 may expose bounded sample rows only through an explicit tool.

Context budgets must be enforced by tests.

---

# 15. Follow-Up Questions

Conversation state is structured.

Follow-ups should operate on prior typed plans where possible.

Example:

```
"What was Q3 opening pipeline?"
    ↓
AnalysisPlan A

"Break that down by owner."
    ↓
typed plan edit

"Only deals above $100k."
    ↓
typed filter edit
```

Do not reconstruct the entire analytical meaning from raw conversational text when a prior structured plan exists.

Persist:

* plan id,
* parent plan id,
* edits,
* carried-forward elements,
* changed elements.

Conflicting edits must be rejected explicitly.

---

# 16. Clarification

Clarification is a first-class outcome.

Ask rather than guess when an ambiguity materially changes the answer.

Examples:

"What was the win rate?"

may require clarification about:

* period,
* denominator,
* cohort.

"Which segments had the highest win rate?"

when the tenant has no customer-segment concept should not silently substitute:

* owner,
* type,
* deal-size bucket.

Alternatives may be suggested only from concepts actually available in the dataset.

Clarification options should be represented as typed plan edits.

---

# 17. Responder and Rendering

The responder is a separate LLM call.

Its job is to produce an explanation from verified results.

The responder must not write literal analytical values when reference tokens can be used.

Example:

```
{{q1.r0.opening_pipeline}}
```

The renderer deterministically substitutes the actual value.

The renderer owns:

* numerical substitution,
* tables,
* structured comparisons,
* warnings,
* required disclosures.

The model does not get to modify a rendered number.

---

# 18. Provenance

Every numerical claim must be traceable.

The final answer must distinguish:

* result-derived numbers,
* user-supplied values,
* structural values such as years/ordinals,
* unsupported literals.

Any unsupported analytical number is a validation failure.

Provenance should connect:

```
answer
  ↓
reference token
  ↓
query id
  ↓
result cell
  ↓
compiled computation
  ↓
dataset
```

The final answer should be reproducible from this chain.

---

# 19. Trust and Disclosures

Trust is data.

Never pass a manually selected trust value from the LLM.

The system computes trust from:

* binding status,
* lineage status,
* stance,
* reconciliation warnings,
* analytical path,
* unresolved assumptions,
* evidence quality.

Tier A:

* deterministic semantic analysis,
* sufficiently confirmed bindings,
* no unresolved load-bearing evidence issues.

Tier B:

* validated investigation,
* inferred bindings where permitted,
* unconfirmed lineage,
* reconciliation warnings,
* other explicitly governed weaker evidence.

Tier C:

* insufficient evidence,
* unavailable concept,
* failed validation,
* unresolved load-bearing ambiguity.

Tier C must not emit an unsupported numerical answer.

---

# 20. Guarded SQL

Guarded free-form SQL is an escape hatch, not the primary analytical mechanism.

It is OFF by default until its milestone is implemented.

When implemented, it must:

* parse SQL with `sqlglot`,
* allow read-only operations only,
* use an allowlisted schema,
* expose only stance-permitted rows/columns,
* prohibit filesystem access,
* enforce query limits,
* enforce timeouts,
* record provenance,
* carry trust tier B,
* never bypass concept/stance safety.

Prefer InvestigationPlan over guarded SQL whenever the typed vocabulary can represent the analysis.

---

# 21. Text Data

Text columns differ across tenants.

Detect and catalogue them dynamically.

Examples may include:

* `ManagerNotes`
* `SENotes`
* `NextStep`
* `Why_Now`
* `Why_Us`

Do not assume these fields exist for every tenant.

For the structured analytics milestone:

* catalogue text fields,
* profile metadata,
* expose availability,
* do not place raw text into default context.

Embeddings, vector search, and RAG are separate future capabilities unless explicitly implemented.

Do not introduce a vector database merely because text fields exist.

---

# 22. Unknown / Unclassified Columns

An unknown column must fail closed.

A previously unseen physical column may be:

* inspected,
* profiled,
* proposed for classification,
* used in explicitly governed exploratory paths where the architecture permits,

but it must never silently become a trusted measure.

A column name alone does not establish meaning.

Do not let adding a new export column silently alter an existing trusted analysis.

---

# 23. Data Quality

Ingestion must fail loudly on:

* duplicate grain keys,
* invalid declared dates,
* invalid required transformations,
* contradictory hard constraints.

Do not convert data-quality errors into warnings merely to improve ingestion success.

The system should distinguish:

* missing,
* null,
* invalid,
* unavailable,
* unknown,
* contradicted.

Do not collapse these states.

---

# 24. Testing

The tiny fixture is the primary regression instrument.

Expected semantic values must be:

* hand-computed,
* or generated by an independent reference implementation.

Never bless current agent output as ground truth.

Test:

* deterministic semantics,
* plan validation,
* leakage controls,
* tenant bindings,
* usage grants,
* agreement tests,
* trust computation,
* provenance,
* investigation gating,
* follow-up edits,
* clarification behavior.

Use adversarial tests for:

* inferred bindings being treated as confirmed,
* future columns leaking into prospective analysis,
* trust tier being influenced by the model,
* raw SQL bypassing gates,
* unsupported numbers appearing in answers,
* semantic metrics being bypassed through investigation,
* stale/future profiles leaking information.

Critical safeguards should have mutation tests where practical.

---

# 25. Dependencies

Preferred stack:

* Python
* DuckDB
* Parquet
* Pydantic
* FastAPI
* pytest

Justified additions:

* `sqlglot` for guarded SQL
* `structlog` for structured run records
* `anthropic` (optional `llm` extra) for the planner's model adapter only

Do not introduce:

* LangChain
* LlamaIndex
* vector databases
* Redis
* Celery
* dbt
* unnecessary orchestration infrastructure

Add infrastructure only when a concrete requirement justifies it.

---

# 26. Engineering Workflow

Before implementing a substantial feature:

1. Read `ARCHITECTURE.md`.
2. Read the relevant contracts and existing implementation.
3. Identify the exact boundary being changed.
4. Check for existing tests that encode the intended behavior.
5. Implement the smallest coherent change.
6. Add deterministic tests.
7. Run the full test suite.
8. Run lint/type checks used by the project.
9. Update documentation.
10. Stop at the requested milestone.

Do not rewrite working components unnecessarily.

Do not implement future milestones opportunistically.

Do not introduce abstractions merely because they might be useful later.

---

# 27. Definition of Done

A feature is not done merely because the happy-path test passes.

For analytical features, completion requires:

* typed contract,
* validation,
* deterministic execution,
* relevant negative tests,
* provenance implications understood,
* temporal safety checked,
* tenant variability considered,
* documentation updated,
* full test suite passing.

For LLM features additionally require:

* structured output,
* bounded retries,
* deterministic downstream validation,
* explicit failure/clarification behavior,
* no model-selected trust,
* no model-supplied analytical numbers.

---

# 28. Current Build Principle

The project should evolve in this order:

```
dataset understanding
    ↓
concept/binding resolution
    ↓
deterministic semantic engine
    ↓
typed investigation engine
    ↓
deterministic trust / provenance / rendering
    ↓
LLM planner
    ↓
LLM responder
    ↓
orchestrator
    ↓
guarded SQL if justified
    ↓
API / UI
```

Do not collapse these boundaries.

The objective is not to make the LLM do everything.

The objective is to give the LLM enough intelligence and tools to investigate the right question while making the underlying analytical result deterministic, auditable, tenant-aware, and temporally safe.
