# Design decisions

Ten ADRs for the decisions that shape this system. Each is sourced from
`ARCHITECTURE.md` and `CLAUDE.md`; where something is planned but not yet
built, that is stated rather than implied.

---

## 1. Plan-as-data, not text-to-SQL

**Context.** The primary interface a model has to the data is a natural-
language question. The obvious implementation is the model writing SQL.

**Decision.** The model emits a typed `AnalysisPlan` or `InvestigationPlan`
(Pydantic, closed vocabulary, no free-text arithmetic). A deterministic gate
validates it and a deterministic compiler turns it into SQL (CLAUDE.md §1.2,
DESIGN_LOG §13.6-13.8). If a question needs arbitrary code to be
expressible, the contract is redesigned, not bypassed with `eval` or a raw
expression field.

**Consequences.** Every question the system can answer is enumerable from the
plan schema. A new analytical shape requires a schema change, not a prompt
change, which is slower but keeps every plan auditable and its space of
possible mistakes small.

---

## 2. Deterministic core / LLM boundary

**Context.** An LLM in the computation path can silently fabricate a number
that reads as a fact.

**Decision.** The LLM is restricted to two roles: planner (question ->
typed plan) and responder (results -> prose with reference tokens). Data
layer, semantic layer, plan validation, compilation, execution, trust and
rendering contain no LLM call (CLAUDE.md §1.3). The orchestrator may call the
LLM but never performs a business calculation itself.

**Consequences.** Every quantitative claim traces to deterministic code, which
is unit-testable and reviewable independently of model behavior. The cost is
that the model cannot shortcut a missing capability by "just computing it" --
a gap must be closed in the deterministic layer.

---

## 3. Concepts bind by declaration, never by resemblance

**Context.** Tenant schemas differ, and a column named `amount` on one tenant
may be `enterprise_amount`, `deal_value`, or absent on another. Name matching
is the tempting shortcut and the most common way multi-tenant systems get
quietly wrong data.

**Decision.** A `ConceptBinding` is `CONFIRMED` only from a declarative
evidence kind: user confirmation, tenant config, export registry,
documentation, or a grain assertion. Exact name, alias, fuzzy match, type
shape, value pattern and even a passing agreement test can never confirm a
binding by themselves (CLAUDE.md §2.2, DESIGN_LOG §12.2). This is enforced
in the type: `ConceptBinding` raises if `CONFIRMED` lacks confirming evidence.

**Consequences.** Ambiguity is preserved rather than silently resolved -- an
`AMBIGUOUS` binding stays ambiguous until a declaration exists, even under
model or user pressure. This is a fail-closed design: a dataset with weaker
evidence answers fewer questions rather than wrong ones.

---

## 4. Usage grants are scoped by purpose, not by column

**Context.** Confirming a column's meaning (a concept binding) and permitting
its use for one purpose (measure, dimension, filter) are different claims. A
tenant that declares `enterprise_amount` as the amount measure has not thereby
licensed grouping by it.

**Decision.** A `UsageGrant` is typed, scoped to a `frozenset[ColumnPurpose]`
derived from the concept's role, persisted as a content-addressed ledger
(`grant_id` hashes the grant's own fields), and re-validated against the
dataset on every load (CLAUDE.md §2.3, DESIGN_LOG §13.2, §13.22). An
inferred or fuzzy binding never issues a grant.

**Consequences.** "Break down pipeline by amount" is rejected even when amount
is confirmed as a measure, because no purpose licenses grouping by it -- the
system must say why, not guess a substitute.

---

## 5. Point-in-time horizon and the attribution guard

**Context.** Snapshot data makes it easy to answer "what happened" while
believing you answered "what was knowable at the time" -- reading a later
snapshot's stage or owner for an earlier analysis is hindsight disguised as
history.

**Decision.** Every analysis has an explicit stance (prospective by default)
and a knowledge horizon. Every governed column has an availability class
(`AS_OF_FACT`, `BACKWARD_DERIVED`, `FUTURE_CONTAMINATED`, `UNKNOWN`), and
`UNKNOWN` is unsafe by default (CLAUDE.md §4). Enforcement is layered --
context, gate, compiler, and a dedicated attribution guard that classifies
every dimension/feature read as `BACKWARD`, `CONTEMPORANEOUS`, `LATER` or
`RETROSPECTIVE_TERMINAL` and refuses a `LATER` read under a prospective stance
(DESIGN_LOG §13.11 #4, §13.22) -- so no single layer's bug is a leak.

**Consequences.** The same rejection fires whether the leak would have entered
through a dimension, a join, a sample, or a profile. The cost is a real one:
a prospective-only tenant answers strictly fewer questions than one that
declares a retrospective stance explicitly.

---

## 6. Precomputed features are evidence, recompute or quarantine

**Context.** CRM exports carry precomputed fields (rep win rates, percentiles,
composite scores) whose lookback window is often undocumented. Using one to
"predict" the outcome it was partly derived from is leakage; using one with an
unknown window under a prospective stance is an unstated assumption.

**Decision.** A precomputed feature is classified `RECOMPUTE`, `USE_WITH_PROOF`
or `QUARANTINE` (CLAUDE.md §5). When the semantic engine can recompute a value
from raw snapshot history, it does, rather than trusting the precomputed
column. An undocumented lookback window makes a feature unsafe for automatic
prospective use.

**Consequences.** Some tenant-supplied "insights" are unusable until their
lineage is documented -- the system reports them as unavailable rather than
using them and hoping the window happens to be safe.

---

## 7. Trust tier is computed, never chosen

**Context.** A tier the model could select is a tier the model could inflate,
which defeats the entire purpose of having one.

**Decision.** `TrustTier` is a derived property of `TrustAssessment.factors`
via `FACTOR_CEILINGS`, a fixed mapping from factor kind to ceiling
(DESIGN_LOG §13.10). `TrustAssessment` uses `extra="forbid"` and has no tier
field or constructor argument -- there is no code path, including a model's
structured output, through which a tier can be supplied directly. A tier-C
assessment raises `AbstentionRequired` before execution (CLAUDE.md §19).

**Consequences.** Every "why B and not A" is answerable by listing factors,
never by trusting a label. Adding a new way to lower trust means adding an
enum member and a ceiling entry, not touching call sites that already assess
results.

---

## 8. Money as DECIMAL, division as an explicit exact operation

**Context.** Binary floating point does not represent most decimal currency
values exactly; summing or averaging monetary columns as `DOUBLE` accumulates
error that is invisible until it is a headline number.

**Decision.** A column is monetary only through concept classification/binding
(never by "has two decimal places"), and a monetary measure is explicitly cast
to `DECIMAL(18,2)` at measure resolution. Division goes through
`sql.exact_divide`, never a bare `/`, so a DECIMAL numerator or denominator is
never silently widened to `DOUBLE` (CLAUDE.md §9).

**Consequences.** Monetary arithmetic is exact and reproducible across
DuckDB versions. Sub-cent rounding behavior from casting text-sourced values
is still an open question (`sub_cent_amounts`, DESIGN_LOG §13.22) --
recorded rather than silently resolved with a tolerance.

---

## 9. Reference tokens and a provenance scanner, not model-formatted numbers

**Context.** A responder that writes "$1,234,567" itself can transpose a
digit, round incorrectly, or state a number the computation never produced.

**Decision.** The responder writes prose with reference tokens
(`{{q1.r0.opening_pipeline}}`), never a literal value. A deterministic
renderer substitutes and formats every value by its `ValueKind`; comparative
words ("rose", "fell") require a typed `ComparisonClaim` the renderer verifies
against the raw values. A provenance scanner then classifies every numeral in
the rendered prose as `RESULT`, `SYSTEM`, `USER`, `STRUCTURAL` or
`UNVERIFIED`, and an `UNVERIFIED` numeral fails the answer with no tolerance
(CLAUDE.md §17-18, DESIGN_LOG §13.14-13.15).

**Consequences.** A model cannot get a correct number with the wrong direction
past the renderer, and cannot get an invented number past the scanner. A
second failure produces a table-only answer with no model prose, rather than
an unverified claim reaching a reader (§13.15).

---

## 10. Planner evaluation by structural comparison, with a failure taxonomy

**Context.** An LLM planner's output varies run to run even for the same
question; evaluation by exact string match is both too strict (rejects a
semantically identical plan with different formatting) and too weak (misses a
plan that differs in a load-bearing field).

**Decision.** The golden set (`evals/planner/cases.py`, 35 cases) pairs each
question with every acceptable outcome kind and, for a final plan, every
acceptable typed plan. A normaliser and comparator (`agent/planner/
evaluation.py`) check structural equivalence against the expected plan(s)
rather than textual identity, and classify a mismatch through a fixed failure
taxonomy. Every model in the default test suite is `ScriptedModel`
(network-free); the real-model run is opt-in (DESIGN_LOG §13.23,
`README.md`).

**Consequences.** The suite catches a planner that reaches the right outcome
through the wrong tool-call trajectory as a pass (only tool use *beyond*
budget is scored) while still catching a plan that differs in a field that
changes the answer. This is implemented; the real-model evaluation itself has
not yet been run in this environment (README: "real-model evaluation opt-in,
not yet run").
