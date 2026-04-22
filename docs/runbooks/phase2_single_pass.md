# Phase 2 runbook — single-pass inference scaffold

Phase 2 runs each page listed in a manifest through one model at one budget
with one fixed prompt, and writes structured artifacts plus a runtime log.
There is no verifier, no reparse, and no calibration yet — those land in
Phases 3–5.

## What it produces

For each run:

```
outputs/runs/{run_id}/
├── raw/{page_id}.md      # model's raw page markdown (HTML tables inside)
├── pages/{page_id}.json  # per-page metadata sidecar (evaluator-facing)
└── run.log.jsonl         # one JSON object per page (runtime log)
```

Each sidecar carries: `page_id`, `model_name`, `budget_name`, `prompt_id`,
`output_token_count`, `runtime_ms`, `started_at`, `finished_at`,
`raw_output_path`, `status`.

Each log line adds `run_id` and `split` on top of the sidecar fields.

## Key invariants

- Phase 2 uses a **stub adapter** (`adapter_kind: stub`) that emits a
  deterministic HTML-table page. Re-running the same `run_id` produces
  byte-identical `raw/*.md` files.
- Raw artifacts are never mixed with metadata (Contract 5).
- Prompt text is pinned by id + version in
  `configs/prompts/table_parse_v1.yaml` (Contract 3).
- `run.log.jsonl` is reset on each run so line count always equals pages
  processed.

## How to run

First-time setup (generates 20 tiny placeholder PNGs the fixture records
point at):

```bash
uv sync --extra dev
uv run python scripts/fixtures/generate_placeholder_images.py
```

Then run the smoke config:

```bash
uv run python scripts/main_runs/run_single_pass.py \
    --config configs/runs/smoke_single_pass.yaml
```

Expected output:

```
run_id          : smoke_2b_low_v1
model           : internvl2-2b (adapter=stub)
budget          : low (max_tiles=4)
prompt          : table_parse_v1 v1
split           : calibration_split
pages_processed : 5
output_dir      : outputs/runs/smoke_2b_low_v1
log_path        : outputs/runs/smoke_2b_low_v1/run.log.jsonl
```

## Config reference

A run config (`configs/runs/*.yaml`) has five sections:

- `run_id`: unique name; also the output directory suffix.
- `inputs`: `split_name`, `manifest_path`, `records_path`, `image_root`.
- `model`: `config_path` + `name` key into a model registry YAML.
- `budget`: `config_path` + `name` key into a budget YAML.
- `prompt`: `config_path` pointing at a versioned prompt YAML.
- `output.dir`: where the run's artifacts go.

Swap model, budget, or prompt by editing only this YAML; no code changes.

## Where real InternVL2 plugs in

The adapter boundary is
[`src/adaptive_inference/inference/adapter.py`](../../src/adaptive_inference/inference/adapter.py).
To add a real model:

1. Implement a new class under `src/adaptive_inference/inference/` that
   subclasses `InferenceAdapter` and honors the `Budget.max_tiles` value
   (e.g. by picking an image-tiling strategy for InternVL2).
2. Register a new `adapter_kind` branch in
   [`factory.py`](../../src/adaptive_inference/inference/factory.py).
3. Flip `adapter_kind: stub` → the new kind in
   `configs/models/internvl2.yaml`.

No other module changes. Logs, writer, and runner stay identical.

## Module map

```
src/adaptive_inference/
├── config/     prompts.py, budgets.py, models.py, runs.py
├── inference/  types.py, adapter.py, stub.py, factory.py
└── runner/     pages.py, output_writer.py, runtime_logger.py, single_pass.py
```
