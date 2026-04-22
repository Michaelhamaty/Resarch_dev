# Project State Inventory

This document captures the current state of the `adaptive-inference` repository, including:
- what has been implemented so far,
- the technology used,
- and a description of every file currently present in the project workspace snapshot.

## What Has Been Done So Far

The project has completed **Phase 1 (dataset freezing)**, **Phase 2
(single-pass inference scaffold, stub adapter)**, **Phase 3
(deterministic structural verifier)**, and **Phase 4 (adaptive
orchestration and run integration)** for the research MVP:

- The repository scaffold and packaging setup are in place.
- A fixture-based dataset pipeline has been implemented under `src/adaptive_inference/dataset/`.
- Deterministic filtering and splitting logic exists for:
  - the English table-page evaluation universe,
  - a placeholder hard-table subset,
  - calibration and held-out evaluation splits.
- Manifest writing is deterministic and includes dataset pinning metadata.
- A CLI script runs the full Phase 1 freeze process from config.
- Fixture manifests have already been generated under `data/splits/`.
- **Phase 2**: YAML-driven config layer for prompts / budgets / models / runs
  under `src/adaptive_inference/config/`.
- **Phase 2**: `InferenceAdapter` interface plus a deterministic
  `StubInferenceAdapter` (real InternVL2 integration deferred; plug-in point
  is `inference/factory.py`).
- **Phase 2**: Runner module that loads pages via Pillow, calls the adapter,
  writes per-page raw markdown + JSON sidecar, and appends JSONL runtime logs.
- **Phase 2**: Smoke CLI (`scripts/main_runs/run_single_pass.py`) plus a
  placeholder image generator (`scripts/fixtures/generate_placeholder_images.py`).
- Unit and smoke tests cover dataset logic, config loaders, the stub adapter,
  output writer, runtime logger, orchestrator wiring, and an end-to-end smoke
  run.

Status from `README.md`:
- Phase 1 complete.
- Phase 2 complete (stub adapter; real InternVL2 integration pending).
- Phase 3 complete (deterministic structural verifier).
- Phase 4 complete (adaptive orchestration, one-shot reparse, extended
  run log with verifier metadata, `first_pass/` + `reparse/` + `final/`
  artifact layout, smoke adaptive CLI).
- Phase 5 (budget calibration) not started.

## Technology Used

- **Language/runtime**: Python 3.11 (`.python-version`, `pyproject.toml`).
- **Environment/package management**: `uv`.
- **Build backend**: `hatchling`.
- **Core runtime dependencies**: `PyYAML` (config loading), `Pillow` (page image loading).
- **Testing**: `pytest`.
- **Linting**: `ruff`.
- **Data/config formats**:
  - YAML for configs,
  - JSON for fixtures and frozen manifests.
- **Architecture pattern currently implemented**:
  - deterministic, config-driven offline data-freezing pipeline.

## File-by-File Description

### Root Files

- `.gitignore` — Ignore rules for Python artifacts, virtual environments, tool caches, research outputs, and editor/OS files.
- `.python-version` — Pins local Python version to `3.11`.
- `CLAUDE.md` — Project-level implementation constraints, MVP scope boundaries, and milestone order.
- `README.md` — Project overview, setup/test commands, Phase 1 run command, and current milestone status.
- `pyproject.toml` — Project metadata, dependencies, build system configuration, pytest config, and ruff config.
- `uv.lock` — Resolved lockfile for reproducible dependency versions in the `uv` workflow.

### Configuration

- `configs/dataset/phase1.yaml` — Phase 1 freeze config: pinning metadata, input records path, hard-subset rule, split seed/sizes, and output directory.
- `configs/prompts/table_parse_v1.yaml` — Versioned prompt template used by every Phase 2+ inference call.
- `configs/budgets/phase2.yaml` — Named compute budgets (`low`, `high`) with placeholder `max_tiles` values awaiting Phase 5 calibration.
- `configs/models/internvl2.yaml` — Model registry for `internvl2-2b` and `internvl2-8b`; Phase 2 wires both to the stub adapter.
- `configs/runs/smoke_single_pass.yaml` — Run config pointing the smoke CLI at the 2B stub adapter over the calibration split at low budget.

### Data Fixtures and Manifests

- `data/fixtures/sample_pages.json` — Synthetic/fixture page metadata used to exercise Phase 1 filtering, hard-subset selection, and deterministic split logic.
- `data/fixtures/images/page_*.png` — Tiny 8×8 placeholder PNGs (one per fixture `page_id`) that Phase 2's page loader actually opens. Regenerated via `scripts/fixtures/generate_placeholder_images.py`.
- `data/splits/eval_universe.json` — Frozen manifest of all English table pages from the fixture dataset.
- `data/splits/hard_subset.json` — Frozen manifest of pages matching the placeholder hard-table rule.
- `data/splits/calibration_split.json` — Frozen manifest of pages allocated to calibration (seeded deterministic split).
- `data/splits/held_out_eval_split.json` — Frozen manifest of pages allocated to held-out evaluation (complement of calibration in the eval universe for current config).

### Documentation

- `docs/specs/adaptive_inference_build_brief.md` — Primary research/implementation brief describing MVP scope, architecture, contracts, risks, and phased roadmap.
- `docs/specs/project_state_inventory.md` — This repository inventory document.
- `docs/runbooks/phase1_freeze_universe.md` — Operational runbook for Phase 1: outputs, invariants, run steps, and config semantics.
- `docs/runbooks/phase2_single_pass.md` — Operational runbook for Phase 2: smoke pipeline, produced artifacts, invariants, run steps, and how real InternVL2 integration plugs into the adapter boundary.

### Scripts

- `scripts/subset_extraction/build_phase1_manifests.py` — CLI entrypoint that loads a Phase 1 config, runs freezing, and prints a summary of generated artifacts.
- `scripts/fixtures/generate_placeholder_images.py` — Writes tiny 8×8 placeholder PNGs for every page referenced by `sample_pages.json` so the Phase 2 page loader has real image files to open.
- `scripts/main_runs/run_single_pass.py` — Phase 2 smoke CLI: loads a run config and executes the single-pass pipeline end-to-end.

### Source Code

- `src/adaptive_inference/dataset/__init__.py` — Public exports for Phase 1 dataset-freezing utilities and dataclasses.
- `src/adaptive_inference/dataset/freeze.py` — Main orchestration logic: load config, parse pinning/rules, filter records, split IDs, and write all four manifests.
- `src/adaptive_inference/dataset/manifests.py` — Manifest schema/types plus deterministic JSON serialization/writer.
- `src/adaptive_inference/dataset/pinning.py` — `DatasetPinning` dataclass and validation for pinning metadata embedded in manifests.
- `src/adaptive_inference/dataset/records.py` — `PageRecord` dataclass and JSON loader with validation and duplicate-ID rejection.
- `src/adaptive_inference/dataset/splits.py` — Deterministic seed-based page-ID splitting into calibration/held-out/unused partitions.
- `src/adaptive_inference/dataset/subsets.py` — English-table filtering and placeholder hard-table rule evaluation.
- `src/adaptive_inference/config/__init__.py` — Public exports for prompt, budget, model, and run-config loaders.
- `src/adaptive_inference/config/prompts.py` — `PromptTemplate` dataclass and versioned YAML loader.
- `src/adaptive_inference/config/budgets.py` — `Budget` dataclass and named-budget YAML loader.
- `src/adaptive_inference/config/models.py` — `ModelConfig` dataclass and model-registry YAML loader.
- `src/adaptive_inference/config/runs.py` — `RunConfig` dataclass that transitively resolves a run YAML into model + budget + prompt.
- `src/adaptive_inference/inference/__init__.py` — Public exports for the adapter interface, stub adapter, factory, and shared types.
- `src/adaptive_inference/inference/types.py` — `InferenceResult` dataclass returned by every adapter.
- `src/adaptive_inference/inference/adapter.py` — Abstract `InferenceAdapter` base class defining the page-level model boundary.
- `src/adaptive_inference/inference/stub.py` — `StubInferenceAdapter`: deterministic canned HTML-table output; no model loaded.
- `src/adaptive_inference/inference/factory.py` — `build_adapter` dispatch from `ModelConfig.adapter_kind` to a concrete adapter.
- `src/adaptive_inference/runner/__init__.py` — Public exports for the runner package (page loader, output writer, runtime logger, orchestrator).
- `src/adaptive_inference/runner/pages.py` — Manifest + records + image-root → `list[LoadedPage]` loader using Pillow.
- `src/adaptive_inference/runner/output_writer.py` — Writes `raw/{page_id}.md` and `pages/{page_id}.json` per page.
- `src/adaptive_inference/runner/runtime_logger.py` — Append-only JSONL runtime logger with a `reset_log` helper for idempotent re-runs.
- `src/adaptive_inference/runner/single_pass.py` — Orchestrator: loads pages, builds an adapter, runs each page, writes artifacts and log entries, returns a summary.
- `src/adaptive_inference/config/adaptive_runs.py` — Phase 4 `AdaptiveRunConfig` dataclass + YAML loader (one model + prompt + paired low/high budgets).
- `src/adaptive_inference/policy/__init__.py` — Policy package exports (`should_reparse`).
- `src/adaptive_inference/policy/escalation.py` — One-shot escalation policy: `should_reparse(verifier_result) -> bool`.
- `src/adaptive_inference/runner/adaptive.py` — Phase 4 adaptive orchestrator: low-budget parse → verifier → optional one-shot high-budget reparse → final artifacts + log.
- `src/adaptive_inference/runner/adaptive_writer.py` — Subfolder-aware artifact writer for `first_pass/`, `reparse/`, `final/`.
- `src/adaptive_inference/runner/adaptive_logger.py` — Phase 4 JSONL logger with extended per-page schema (verifier decision, budgets, split runtimes, artifact paths).
- `configs/runs/smoke_adaptive.yaml` — Phase 4 smoke adaptive run config (stub adapter, paired low/high budgets, calibration split).
- `scripts/main_runs/run_adaptive.py` — Phase 4 smoke CLI.
- `docs/runbooks/phase4_adaptive.md` — Phase 4 runbook.
- `tests/config/test_adaptive_runs.py` — `AdaptiveRunConfig` loader tests.
- `tests/policy/__init__.py`, `tests/policy/test_escalation.py` — Policy tests.
- `tests/runner/test_adaptive.py` — Orchestrator tests (PASS branch, REPARSE branch, log schema, idempotent re-run, empty manifest, Contract-1 at-most-one-reparse).
- `tests/smoke/test_adaptive_smoke.py` — End-to-end Phase 4 smoke test.

### Tests

- `tests/conftest.py` — Top-level fixtures exposing `repo_root`, `configs_dir`, and `data_dir` to every test.
- `tests/smoke/test_package_importable.py` — Smoke import test for key package namespaces expected by later milestones.
- `tests/smoke/test_single_pass_smoke.py` — End-to-end Phase 2 smoke: runs the real smoke config against fixture images and asserts five raw markdown files, five sidecars, and five log lines with the expected model/budget/prompt fields.
- `tests/dataset/__init__.py` — Marks dataset tests as a package (currently empty).
- `tests/dataset/conftest.py` — Shared test fixtures for repo-root-relative paths (sample pages and Phase 1 config).
- `tests/dataset/test_freeze.py` — End-to-end Phase 1 freeze tests: outputs, set invariants, idempotency/byte stability, payload shape, and config validation.
- `tests/dataset/test_manifests.py` — Manifest writer tests for byte stability, pinning header presence, sorting, newline behavior, and directory creation.
- `tests/dataset/test_records.py` — Record loader tests for successful fixture load and error handling (duplicates, missing fields, wrong JSON shape).
- `tests/dataset/test_splits.py` — Splitter tests for determinism, seed sensitivity, disjointness, sizing, sorting, and invalid input handling.
- `tests/dataset/test_subsets.py` — Subset/rule tests for English filtering, hard-rule trigger behavior, fixture hard-subset expectations, and rule validation.
- `tests/config/__init__.py` — Marks config tests as a package (empty).
- `tests/config/test_prompts.py` — Prompt loader tests: successful load, required-field validation, non-mapping rejection, frozen dataclass, type coercion.
- `tests/config/test_budgets.py` — Budget loader tests: named lookup, missing-name error, non-positive rejection, missing-section rejection.
- `tests/config/test_models.py` — Model registry tests: registry load, single lookup, missing-name error, required-field validation.
- `tests/config/test_runs.py` — Run config tests: successful load of the smoke run, missing-top-level-key error.
- `tests/inference/__init__.py` — Marks inference tests as a package (empty).
- `tests/inference/test_types.py` — `InferenceResult` shape and frozen-ness.
- `tests/inference/test_stub.py` — Stub adapter tests: field shape, HTML table present, determinism, page-id variance, factory dispatch, unknown-kind rejection.
- `tests/runner/__init__.py` — Marks runner tests as a package (empty).
- `tests/runner/test_pages.py` — Page loader tests: manifest id roundtrip, record + image load, missing image error, manifest-references-unknown-id error.
- `tests/runner/test_output_writer.py` — Output writer tests: raw + sidecar written, sidecar field set, deterministic sidecar JSON, per-page isolation.
- `tests/runner/test_runtime_logger.py` — Runtime logger tests: one line per call, exact field set, `reset_log` clears the file, reset is a no-op when missing.
- `tests/runner/test_single_pass.py` — Orchestrator tests: all artifacts produced, log line count, idempotent re-run, empty-manifest still materializes the log file.

### Generated Cache/Bytecode Files Currently Present

These files are tooling artifacts and not part of the core source design:

- `.ruff_cache/.gitignore` — Internal marker file for ruff cache directory behavior.
- `.ruff_cache/CACHEDIR.TAG` — Standard cache directory tag file.
- `.ruff_cache/0.15.11/10691315320799242638` — Ruff cache entry for lint analysis results.
- `.ruff_cache/0.15.11/16316253643953720980` — Ruff cache entry for lint analysis results.
- `.ruff_cache/0.15.11/4678298413470410296` — Ruff cache entry for lint analysis results.
- `tests/dataset/__pycache__/__init__.cpython-311.pyc` — Compiled bytecode for test package init.
- `tests/dataset/__pycache__/conftest.cpython-311-pytest-9.0.3.pyc` — Compiled bytecode for shared pytest fixtures.
- `tests/dataset/__pycache__/test_freeze.cpython-311-pytest-9.0.3.pyc` — Compiled bytecode for freeze tests.
- `tests/dataset/__pycache__/test_manifests.cpython-311-pytest-9.0.3.pyc` — Compiled bytecode for manifest tests.
- `tests/dataset/__pycache__/test_records.cpython-311-pytest-9.0.3.pyc` — Compiled bytecode for record tests.
- `tests/dataset/__pycache__/test_splits.cpython-311-pytest-9.0.3.pyc` — Compiled bytecode for split tests.
- `tests/dataset/__pycache__/test_subsets.cpython-311-pytest-9.0.3.pyc` — Compiled bytecode for subset tests.

## Notes

- This inventory reflects the current workspace snapshot and generated artifacts visible at the time of writing.
- As new milestones (inference/verifier/policy/evaluation) land, this file should be updated to remain a reliable project map.
