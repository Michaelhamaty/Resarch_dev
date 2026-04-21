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

## Status

Milestone 1 (repo scaffold). No runtime code yet.
