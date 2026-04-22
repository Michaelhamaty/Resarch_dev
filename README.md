# adaptive-inference

Research MVP for **matched-budget adaptive inference** on complex English table pages.

Each page gets a low-budget InternVL2-2B parse, a deterministic HTML structural
verifier, and at most one higher-budget reparse when the first output looks
structurally broken — all while holding the average per-page compute equal to
fixed-cost baselines.

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
```

## Phase 1 — freeze the universe

Phase 1 locks down the immutable page-ID manifests used by every later
milestone. See [`docs/runbooks/phase1_freeze_universe.md`](docs/runbooks/phase1_freeze_universe.md).

```bash
uv run python scripts/subset_extraction/build_phase1_manifests.py \
    --config configs/dataset/phase1.yaml
```

This writes four manifests (`eval_universe`, `hard_subset`,
`calibration_split`, `held_out_eval_split`) under `data/splits/`.

## Phase 2 — single-pass inference smoke

Phase 2 runs each page in a manifest through one model + budget + prompt and
writes structured artifacts. Uses a deterministic stub adapter so the smoke
path is CPU-only. See
[`docs/runbooks/phase2_single_pass.md`](docs/runbooks/phase2_single_pass.md).

```bash
uv run python scripts/fixtures/generate_placeholder_images.py
uv run python scripts/main_runs/run_single_pass.py \
    --config configs/runs/smoke_single_pass.yaml
```

Artifacts land under `outputs/runs/{run_id}/` as `raw/*.md`, `pages/*.json`,
and `run.log.jsonl`.

## Phase 3 — deterministic structural verifier

Phase 3 adds the pure-function verifier that inspects first-pass raw
page output and returns `PASS` or `REPARSE` based on five structural
checks (presence, parsability, span expansion, rectangular consistency,
degenerate-table). See
[`docs/runbooks/phase3_verifier.md`](docs/runbooks/phase3_verifier.md).

```python
from adaptive_inference.verifier.structural import verify_page_tables
result = verify_page_tables(raw_page_markdown)
```

No orchestration, logging, or reparse logic yet — those wire in during
Phase 4.

## Status

Phase 1 (dataset freezing), Phase 2 (single-pass inference scaffold,
stub adapter), and Phase 3 (HTML extraction + deterministic structural
verifier) complete. Phase 4 (adaptive routing / one-shot reparse) not
started.
