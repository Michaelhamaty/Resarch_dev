# Phase 4 runbook — adaptive orchestration and run integration

Phase 4 wires the Phase 2 single-pass scaffold and the Phase 3
structural verifier into the MVP's one-shot adaptive pipeline:

```
low-budget full-page parse
   -> deterministic verifier
   -> (if REPARSE) one-shot high-budget full-page reparse
   -> final selected artifact + run log
```

There is no second reparse. There is no learned routing. There is no
crop repair. Contract 1 ("one page, one policy path") is enforced by
construction: the adapter is called at most twice per page.

## What it produces

For each run:

```
outputs/runs/{run_id}/
├── first_pass/
│   ├── raw/{page_id}.md        # always — low-budget parse
│   └── pages/{page_id}.json    # always — first-pass sidecar
├── reparse/
│   ├── raw/{page_id}.md        # only if verifier says REPARSE
│   └── pages/{page_id}.json    # only if REPARSE
├── final/
│   ├── raw/{page_id}.md        # always — copy of chosen pass
│   └── pages/{page_id}.json    # always — chosen-pass sidecar + verifier meta
└── run.log.jsonl               # one JSONL line per page
```

### Final sidecar fields (on top of Phase 2's shape)

- `final_output_source`: `"first_pass"` or `"reparse"`
- `reparse_triggered`: bool
- `verifier_decision`: `"PASS"` or `"REPARSE"`
- `verifier_failure_codes`: list of the Phase 3 stable strings
- `predicted_table_count`: int

### Log fields (per page)

`run_id`, `split`, `page_id`, `model_name`, `prompt_id`, `budget_low`,
`budget_high`, `reparse_triggered`, `verifier_decision`,
`verifier_failure_codes`, `predicted_table_count`,
`first_pass_output_tokens`, `reparse_output_tokens`,
`first_pass_runtime_ms`, `verifier_runtime_ms`, `reparse_runtime_ms`,
`total_runtime_ms`, `first_pass_raw_path`, `reparse_raw_path`,
`final_raw_path`, `final_output_source`, `status`.

Reparse-only fields are `null` when no reparse fired.

## Key invariants

- Contract 1: adapter runs at most twice per page (first pass + optional
  reparse). The orchestration test asserts this with a wrapped adapter.
- Contract 5: first-pass raw output is always preserved on disk; the
  `final/` copy is a separate file, not a symlink.
- `run.log.jsonl` is truncated at the start of each run so line count
  always equals pages processed.
- First-pass and reparse passes share the same adapter instance, same
  model, same prompt, and (for reparse) only `budget` differs.

## How to run

```bash
uv run python scripts/fixtures/generate_placeholder_images.py  # once
uv run python scripts/main_runs/run_adaptive.py \
    --config configs/runs/smoke_adaptive.yaml
```

Expected output:

```
run_id          : smoke_2b_adaptive_v1
model           : internvl2-2b (adapter=stub)
budget_low      : low (max_tiles=4)
budget_high     : high (max_tiles=12)
prompt          : table_parse_v1 v1
split           : calibration_split
pages_processed : 5
reparse_count   : 0
output_dir      : outputs/runs/smoke_2b_adaptive_v1
log_path        : outputs/runs/smoke_2b_adaptive_v1/run.log.jsonl
```

The stub adapter's output is structurally clean, so the smoke PASS
branch fires for every page. To exercise the REPARSE branch in tests,
unit tests monkeypatch `verify_page_tables` — see
`tests/runner/test_adaptive.py`.

## Config reference

A Phase 4 run config (`configs/runs/*.yaml`) has six sections; only
`budget` differs from the Phase 2 shape:

```yaml
run_id: ...
inputs: { split_name, manifest_path, records_path, image_root }
model:  { config_path, name }
budget: { config_path, low_name, high_name }     # paired budgets
prompt: { config_path }
output: { dir }
```

Swap low/high pairing by editing only this YAML. No code changes needed.

## Module map

```
src/adaptive_inference/
├── config/
│   └── adaptive_runs.py          # AdaptiveRunConfig + loader (paired budgets)
├── policy/
│   └── escalation.py             # should_reparse(verifier) -> bool
└── runner/
    ├── adaptive.py               # the orchestrator (thin composition layer)
    ├── adaptive_writer.py        # first_pass/, reparse/, final/ writers
    └── adaptive_logger.py        # extended Phase-4 JSONL schema
```

Phase 2's `single_pass.py`, `output_writer.py`, and `runtime_logger.py`
are untouched — the single-pass path still exists, unchanged, for
baseline comparisons.

## Where real InternVL2 plugs in

Same answer as Phase 2: register a new `adapter_kind` in
`inference/factory.py`. The adaptive orchestrator is agnostic to the
adapter implementation — it only uses the `InferenceAdapter.run`
interface and the paired `Budget` objects.
