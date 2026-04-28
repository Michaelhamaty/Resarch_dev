# Project State — Adaptive Inference MVP

This is the one-page honest snapshot of what is and is not built. For
the full design rationale, see
[`docs/specs/adaptive_inference_build_brief.md`](specs/adaptive_inference_build_brief.md).

## Implemented (Phase 1 → Phase 7)

| Phase | Component | Status |
|------|------|------|
| 1 | Frozen split + subset manifests under `data/splits/` | ✅ implemented |
| 2 | Single-pass inference scaffold (loader, adapter, writer, JSONL log) | ✅ implemented |
| 3 | Deterministic structural verifier with five failure codes | ✅ implemented |
| 4 | Adaptive orchestrator (verifier-gated, one-shot reparse) + extended log | ✅ implemented |
| 5 | Calibration sweep + frozen artifact at `configs/calibration/frozen_budgets.json` | ✅ implemented |
| 6 | Phase 6 main-runs orchestrator + manifest at `outputs/runs/phase6/manifest.json` | ✅ implemented |
| 7 | Analysis package (`results / cost / reparse / qualitative / audit`) + CLI | ✅ implemented |

End-to-end test coverage: 224 tests, all green. `ruff check .` clean.

## Stubbed (intentional, gated, honest)

These are placeholders the project ships **deliberately** so the
research scaffolding can be exercised on commodity hardware. They are
documented in code, in the frozen artifact, and in the Phase 7 audit.

- **`StubInferenceAdapter`** — `src/adaptive_inference/inference/stub.py`.
  Returns a deterministic canned HTML table per page, ignoring image
  contents and `max_tiles`. Both `internvl2-2b` and `internvl2-8b`
  registry entries point at this adapter via
  `configs/models/internvl2.yaml`. The factory at
  `src/adaptive_inference/inference/factory.py` only knows how to
  build the stub.
  - **Consequence**: the calibration reparse rate is 0.0, all costs
    collapse to `B_low.max_tiles`, the random baseline never escalates
    on stubs, and Phase 7 emits
    `accuracy_status: not_applicable_stub_adapters`.

- **Hard-table rule placeholder** — `src/adaptive_inference/dataset/subsets.py`
  evaluates a structural rule (`min_row_count`, `min_col_count`,
  `require_any_of`) against fixture metadata. Real OmniDocBench
  difficulty signals are **not yet wired**; the rule is a stand-in
  until the snapshot lands.

- **Stubbed 8B baseline gate** — Phase 6 refuses to run
  `fixed_8b_matched` when `B_fix_8B.adapter_kind == "stub"` unless
  `--allow-stubbed-8b` is passed. This guards the headline
  upper-bound number from being silently faked.

## Deferred (defined as out of scope for the MVP)

- **TEDS / edit-distance accuracy scorer.** No table-level accuracy is
  computed in Phase 7. The audit warns explicitly. Adding a real
  scorer requires either OmniDocBench's evaluation code or an
  in-house implementation; both are outside the MVP and need real
  model output to be meaningful.
- **Real OmniDocBench evaluator integration.** The dataset module
  pins fixture metadata, not the live OmniDocBench snapshot.
- **Hard-subset extraction from OmniDocBench metadata.** See above.

## Future work (per spec, post-MVP)

- Learned routing / hybrid routing / confidence-based routing
- OCR-assisted routing
- Crop-level repair
- Multi-step escalation policies (low → medium → high)
- Multilingual / rotated / general document expansion
- Formal policy analysis (precision / recall / expected gain)

## Required for a real-results run

1. Implement a real InternVL2 adapter and register it in
   `configs/models/internvl2.yaml`.
2. Re-run Phase 5 (`scripts/calibration/run_calibration.py`) to
   refreeze `frozen_budgets.json` on real outputs.
3. Re-run Phase 6 with all systems (no `--allow-stubbed-8b`).
4. Wire a real TEDS / edit-distance scorer and add an accuracy step
   to Phase 7.

## How to reproduce the current MVP end-to-end

```bash
uv sync --extra dev

# Phase 1: freeze the universe
uv run python scripts/subset_extraction/build_phase1_manifests.py \
    --config configs/dataset/phase1.yaml

# Generate placeholder PNGs (Phase 2 page loader needs real image files)
uv run python scripts/fixtures/generate_placeholder_images.py

# Phase 5: calibrate and freeze budgets
uv run python scripts/calibration/run_calibration.py \
    --config configs/calibration/phase5.yaml

# Phase 6: main runs (--allow-stubbed-8b because adapters are stubs)
uv run python scripts/main_runs/run_phase6.py \
    --config configs/experiment/phase6.yaml --allow-stubbed-8b

# Phase 7: analysis + audit
uv run python scripts/analysis/run_phase7.py \
    --config configs/analysis/phase7.yaml

# Tests
uv run pytest -q
uv run ruff check .
```

## Known limitations of the current artifacts

- All numbers in the current `outputs/analysis/phase6_main_v1/` tree
  are **provisional pipeline-validation data**, not research results.
- The random baseline is byte-identical to `fixed_2b_low` on stubs
  (probability is 0.0). The audit warns. Real adapters will produce
  variance.
- Verifier failure-code histograms are empty across all systems
  because the stub never emits malformed HTML. This is expected.
