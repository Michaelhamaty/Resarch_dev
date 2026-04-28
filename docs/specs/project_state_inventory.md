# Project State Inventory

This is the file-by-file map of the `adaptive-inference` repository.
For a one-page honest snapshot of what is implemented vs. stubbed vs.
deferred, see [`docs/PROJECT_STATE.md`](../PROJECT_STATE.md).

## Phase status

The MVP has shipped Phases 1–7:

- **Phase 1** — dataset freezing
- **Phase 2** — single-pass inference scaffold (stub adapter)
- **Phase 3** — deterministic structural verifier
- **Phase 4** — adaptive orchestration with one-shot reparse
- **Phase 5** — calibration sweep + frozen artifact
- **Phase 6** — main-runs orchestrator + manifest
- **Phase 7** — analysis package + integration audit

Stubbed (intentional): real InternVL2 adapters, hard-table rule.
Deferred: TEDS / edit-distance scorer, real OmniDocBench snapshot.
See `docs/PROJECT_STATE.md` for details.

Tests: 224 pytest cases, all green. `ruff check .` clean.

## Technology

- **Language**: Python 3.11 (`.python-version`).
- **Env / packaging**: `uv`, `hatchling`.
- **Runtime deps**: `PyYAML` (config), `Pillow` (image loader),
  `lxml` (verifier HTML parsing).
- **Test / lint**: `pytest`, `ruff`.
- **Data formats**: YAML for configs, JSON for manifests + sidecars,
  JSONL for run logs and analysis outputs.

## Top-level files

- `.gitignore`, `.python-version`, `pyproject.toml`, `uv.lock`,
  `CLAUDE.md`, `README.md`.

## `configs/`

- `configs/dataset/phase1.yaml` — Phase 1 freeze config.
- `configs/prompts/table_parse_v1.yaml` — versioned prompt template.
- `configs/budgets/phase2.yaml` — named budgets for Phase 2 smokes.
- `configs/models/internvl2.yaml` — model registry. Both 2B and 8B
  point at `adapter_kind: stub` until a real adapter lands.
- `configs/runs/smoke_single_pass.yaml` — Phase 2 smoke.
- `configs/runs/smoke_adaptive.yaml` — Phase 4 smoke.
- `configs/calibration/phase5.yaml` — Phase 5 sweep grids and matched-cost target.
- `configs/calibration/frozen_budgets.json` — **Phase 6 contract**;
  written by Phase 5, read read-only by Phase 6 and Phase 7.
- `configs/experiment/phase6.yaml` — Phase 6 inputs (frozen artifact
  path, held-out manifest, output root, random seeds).
- `configs/analysis/phase7.yaml` — Phase 7 inputs (Phase 6 manifest,
  frozen artifact, splits, output root).

## `data/`

- `data/fixtures/sample_pages.json` — synthetic page metadata.
- `data/fixtures/images/page_*.png` — placeholder PNGs.
- `data/splits/eval_universe.json`,
  `data/splits/hard_subset.json`,
  `data/splits/calibration_split.json`,
  `data/splits/held_out_eval_split.json` — Phase 1 frozen manifests.

## `src/adaptive_inference/`

### `dataset/` (Phase 1)

- `freeze.py` — orchestrator: load Phase 1 config, filter, split, write
  all four manifests.
- `manifests.py` — manifest schema + deterministic JSON writer.
- `pinning.py` — `DatasetPinning` header.
- `records.py` — `PageRecord` loader with validation.
- `splits.py` — seeded deterministic splitter.
- `subsets.py` — English-table filter + placeholder hard-table rule.

### `inference/` (Phase 2)

- `adapter.py` — abstract `InferenceAdapter`.
- `factory.py` — `build_adapter` dispatcher (stub only).
- `stub.py` — deterministic `StubInferenceAdapter`.
- `types.py` — `InferenceResult` dataclass.

### `config/` (Phase 2 / Phase 4)

- `prompts.py`, `budgets.py`, `models.py`, `runs.py` — Phase 2 loaders.
- `adaptive_runs.py` — Phase 4 paired-budget loader.

### `runner/` (Phase 2 / Phase 4 / Phase 6)

- `pages.py` — manifest + records + image-root → `LoadedPage` list.
- `output_writer.py` — Phase 2 `raw/*.md` + `pages/*.json` writer.
- `runtime_logger.py` — Phase 2 JSONL logger.
- `single_pass.py` — Phase 2 orchestrator.
- `adaptive.py` — Phase 4 verifier-gated orchestrator.
- `adaptive_writer.py` — `first_pass/`, `reparse/`, `final/` artifact writer.
- `adaptive_logger.py` — extended JSONL logger.
- `adaptive_random.py` — Phase 6 random-baseline runner (kept separate
  from `adaptive.py` so the verifier-gated path stays untouched).

### `verifier/` (Phase 3)

- `codes.py` — public failure-code constants (write contract).
- `types.py` — `TableSummary`, `VerifierResult`.
- `spans.py` — HTML parsing + rowspan/colspan expansion + rectangular
  consistency.
- `structural.py` — `verify_page_tables(raw_text) -> VerifierResult`.

### `parsing/`

- `html_tables.py` — regex-based outermost `<table>` block extractor.

### `policy/` (Phase 4 / Phase 6)

- `escalation.py` — verifier-decision policy.
- `random_escalation.py` — seeded random policy used by Phase 6's random baseline.

### `calibration/` (Phase 5)

- `cost.py` — `adaptive_cost_tiles`, `fixed_cost_tiles` (single source of truth).
- `summary.py` — `SweepPointSummary` + log summarizers.
- `sweep.py` — drive Phase 2/4 runners across candidate grids.
- `select.py` — pick adaptive pair + matched fixed baselines.
- `artifact.py` — `FrozenBudgets` (read/write of `frozen_budgets.json`).
- `config.py` — `CalibrationConfig` loader.

### `experiment/` (Phase 6)

- `runner.py` — Phase 6 orchestrator. Reads frozen artifact read-only,
  dispatches to runners, writes manifest.
- `systems.py` — canonical `SystemSpec` constants. Hardcoded (not YAML)
  to prevent drift.
- `manifest.py` — manifest schema + `sha256_of_file`, `iso_now`,
  `git_head_or_none` helpers.
- `frozen_inputs.py` — bridge from `FrozenBudget` to `Budget` /
  `ModelConfig` / random reparse rate.
- `config.py` — Phase 6 YAML loader.

### `analysis/` (Phase 7)

- `loaders.py` — Phase 6 manifest + run-log JSONL + split readers.
- `results.py` — `SystemResult` summary, reusing
  `calibration.summary.summarize_*_log`.
- `cost.py` — `CostSummary`, reusing `calibration.cost`.
- `reparse.py` — reparse-rate slices + verifier failure-code histograms.
- `qualitative.py` — per-page join across all systems.
- `audit.py` — 11 integration audit checks.
- `runner.py` — `run_phase7` orchestrator.
- `config.py` — `Phase7Config` YAML loader.

## `scripts/`

- `subset_extraction/build_phase1_manifests.py` — Phase 1 CLI.
- `fixtures/generate_placeholder_images.py` — fixture image generator.
- `main_runs/run_single_pass.py` — Phase 2 smoke CLI.
- `main_runs/run_adaptive.py` — Phase 4 smoke CLI.
- `main_runs/run_phase6.py` — Phase 6 main-runs CLI.
- `calibration/run_calibration.py` — Phase 5 CLI.
- `analysis/run_phase7.py` — Phase 7 CLI.

## `tests/`

Test packages mirror source layout:

- `tests/dataset/` — Phase 1 freeze, manifest, splits, subsets.
- `tests/config/` — prompt / budget / model / run / adaptive_run loaders.
- `tests/inference/` — stub adapter + types.
- `tests/runner/` — single_pass, adaptive, adaptive_random, output_writer,
  runtime_logger, page loader.
- `tests/parsing/` — HTML extractor.
- `tests/verifier/` — structural verifier, spans, **codes-public guard**.
- `tests/policy/` — verifier escalation, random escalation.
- `tests/calibration/` — cost, summary, sweep smoke, select, artifact.
- `tests/experiment/` — config, manifest, frozen-inputs, systems,
  runner smoke, stub-8B gate, **frozen-artifact read-only guard**.
- `tests/analysis/` — loaders, results, cost, audit (happy path +
  tampering), end-to-end Phase 7 smoke.
- `tests/smoke/` — package-importable, single-pass smoke, adaptive smoke.

## `outputs/` (gitignored)

Research outputs created at runtime, not committed:

- `outputs/runs/<run_id>/` — Phase 2 / Phase 4 smokes.
- `outputs/calibration/` — Phase 5 sweep work.
- `outputs/runs/phase6/` — Phase 6 manifests + per-system trees.
- `outputs/analysis/<run_set_id>/` — Phase 7 analysis artifacts.

## `docs/`

- `docs/specs/adaptive_inference_build_brief.md` — primary research /
  implementation brief.
- `docs/specs/project_state_inventory.md` — this file.
- `docs/PROJECT_STATE.md` — one-page honest state snapshot.
- `docs/runbooks/phase1_freeze_universe.md`
- `docs/runbooks/phase2_single_pass.md`
- `docs/runbooks/phase3_verifier.md`
- `docs/runbooks/phase4_adaptive.md`
- `docs/runbooks/phase5_calibration.md`
- `docs/runbooks/phase6_main_runs.md`
- `docs/runbooks/phase7_analysis.md`

## Notes

- This inventory reflects the workspace at the end of Phase 7. Update
  it whenever a new module / config / runbook lands or a stubbed
  component is replaced with a real implementation.
- The single source of truth for **what is real vs stubbed vs
  deferred** is `docs/PROJECT_STATE.md`. Keep it in sync.
