# adaptive-inference

Research MVP for **matched-budget adaptive inference** on complex English table pages.

Each page gets a low-budget InternVL2-2B parse, a deterministic HTML structural
verifier, and at most one higher-budget reparse when the first output looks
structurally broken — all while holding the average per-page compute equal to
fixed-cost baselines.

> ⚠️ **Research MVP — adapters are stubs.** This repository ships a
> deterministic `StubInferenceAdapter` for `internvl2-2b` and
> `internvl2-8b`. All numbers in `outputs/` are pipeline-validation
> artifacts, not research results. Real model integration is deferred
> behind a single-file boundary
> (`src/adaptive_inference/inference/factory.py`). See
> [`docs/PROJECT_STATE.md`](docs/PROJECT_STATE.md) for the full honest
> snapshot.

Start here: [`docs/specs/adaptive_inference_build_brief.md`](docs/specs/adaptive_inference_build_brief.md).
Project rules: [`CLAUDE.md`](CLAUDE.md).

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11.

```bash
uv sync --extra dev
```

## Tests

```bash
uv run pytest
uv run ruff check .
```

## Status

| Phase | Component | Status |
|---|---|---|
| 1 | Frozen split + subset manifests | implemented |
| 2 | Single-pass inference scaffold (stub adapter) | implemented |
| 3 | Deterministic structural verifier | implemented |
| 4 | Adaptive orchestration (one-shot reparse) | implemented |
| 5 | Calibration sweep + frozen budgets | implemented |
| 6 | Main-runs orchestrator + manifest | implemented |
| 7 | Analysis package + integration audit | implemented |
| — | Real InternVL2 adapter | **stubbed** |
| — | TEDS / edit-distance scorer | **deferred** |
| — | OmniDocBench live snapshot | **deferred** |

## End-to-end run on the shipped fixtures

```bash
# Phase 1 — freeze the universe
uv run python scripts/subset_extraction/build_phase1_manifests.py \
    --config configs/dataset/phase1.yaml

# Make sure placeholder page images exist
uv run python scripts/fixtures/generate_placeholder_images.py

# Phase 5 — calibrate and freeze budgets
uv run python scripts/calibration/run_calibration.py \
    --config configs/calibration/phase5.yaml

# Phase 6 — main runs (stub 8B requires the explicit gate)
uv run python scripts/main_runs/run_phase6.py \
    --config configs/experiment/phase6.yaml --allow-stubbed-8b

# Phase 7 — analysis + integration audit
uv run python scripts/analysis/run_phase7.py \
    --config configs/analysis/phase7.yaml
```

## Phase 1 — freeze the universe

Locks the immutable page-ID manifests every later milestone consumes.
See [`docs/runbooks/phase1_freeze_universe.md`](docs/runbooks/phase1_freeze_universe.md).

```bash
uv run python scripts/subset_extraction/build_phase1_manifests.py \
    --config configs/dataset/phase1.yaml
```

Writes four manifests (`eval_universe`, `hard_subset`,
`calibration_split`, `held_out_eval_split`) under `data/splits/`.

## Phase 2 — single-pass inference smoke

Runs each page through one model + budget + prompt and writes structured
artifacts. Uses the deterministic stub adapter so the smoke path is
CPU-only. See [`docs/runbooks/phase2_single_pass.md`](docs/runbooks/phase2_single_pass.md).

```bash
uv run python scripts/main_runs/run_single_pass.py \
    --config configs/runs/smoke_single_pass.yaml
```

Artifacts land under `outputs/runs/{run_id}/` as `raw/*.md`,
`pages/*.json`, and `run.log.jsonl`.

## Phase 3 — deterministic structural verifier

Pure function inspecting first-pass raw page output and returning
`PASS` or `REPARSE` based on five structural checks (presence,
parsability, span expansion, rectangular consistency, degenerate-table).
See [`docs/runbooks/phase3_verifier.md`](docs/runbooks/phase3_verifier.md).

```python
from adaptive_inference.verifier.structural import verify_page_tables
result = verify_page_tables(raw_page_markdown)
```

## Phase 4 — adaptive orchestration

Composes Phase 2 + Phase 3 into the MVP's one-shot pipeline:
low-budget parse → verifier → optional high-budget reparse → final
artifact. See [`docs/runbooks/phase4_adaptive.md`](docs/runbooks/phase4_adaptive.md).

```bash
uv run python scripts/main_runs/run_adaptive.py \
    --config configs/runs/smoke_adaptive.yaml
```

## Phase 5 — budget calibration

Sweeps candidate budgets on the **calibration split only** and freezes
`B_low`, `B_high`, `B_fix_2B`, `B_fix_8B` into a stable artifact
consumed read-only by Phase 6. See
[`docs/runbooks/phase5_calibration.md`](docs/runbooks/phase5_calibration.md).

```bash
uv run python scripts/calibration/run_calibration.py \
    --config configs/calibration/phase5.yaml
```

Writes `configs/calibration/frozen_budgets.json` (the Phase 6 contract)
and `outputs/calibration/sweep_summaries.jsonl` (per-sweep-point audit).

## Phase 6 — main experimental runs

Executes the five required systems on the **held-out evaluation split**
under one harness, using the frozen budgets. See
[`docs/runbooks/phase6_main_runs.md`](docs/runbooks/phase6_main_runs.md).

```bash
uv run python scripts/main_runs/run_phase6.py \
    --config configs/experiment/phase6.yaml --allow-stubbed-8b
```

Writes `outputs/runs/phase6/manifest.json` plus per-system output trees.

## Phase 7 — analysis & integration audit

Read-only consumer of Phase 6 outputs. Produces results table, cost
summary, reparse summary, qualitative examples, and a programmatic
integration audit (11 checks). See
[`docs/runbooks/phase7_analysis.md`](docs/runbooks/phase7_analysis.md).

```bash
uv run python scripts/analysis/run_phase7.py \
    --config configs/analysis/phase7.yaml
```

Phase 7 explicitly does **not** compute table-level accuracy on stub
adapters. The audit reports `accuracy_status: not_applicable_stub_adapters`
as an explicit warn until a real adapter and a real scorer are wired in.
