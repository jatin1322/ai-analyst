# Scripts

Developer utilities.

| Script | Purpose |
|---|---|
| `probe_production.py` | Diagnose a real export's close-date behaviour. Aggregates only |
| `generate_synthetic.py` | Placeholder for synthetic snapshot data |
| `explain.py` | Run a golden case's plan on a synthetic world and print its `AnalysisTrace` as Markdown |

## `probe_production.py`

```
python scripts/probe_production.py /path/to/export.parquet
python scripts/probe_production.py 's3://bucket/prefix/snapshot.parquet/'
python scripts/probe_production.py export.parquet --as-of-encoding iso --json
```

It prints counts, quantiles and category labels for low-cardinality status
candidates. It never prints a row or an identifier value, never writes anything,
and lists status fields as candidates only. Credentials are never arguments: for S3
DuckDB uses the standard AWS credential chain, so set `AWS_PROFILE` or run
`aws configure` first.

## `explain.py`

```
python -m scripts.explain metric_opening_q2
python -m scripts.explain metric_opening_q2 --world tiny
```

Builds a synthetic world from `evals.planner.datasets` (default `tiny`), runs a
named golden case's expected plan from `evals.planner.cases.CASES_BY_ID`
through `ai_analyst.agent.trace.trace_plan`, and prints the resulting
`AnalysisTrace` as Markdown. No model and no network: the plan is the case's
own hand-written golden plan, not a planner's output.
