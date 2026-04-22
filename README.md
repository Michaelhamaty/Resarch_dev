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

## Status

Phase 1 (dataset freezing) complete. Phase 2 (single-page inference scaffold)
not started.
