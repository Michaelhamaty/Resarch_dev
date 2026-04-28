# Phase 7 — Final Analysis & Integration Audit

## What this phase does

Phase 7 ingests the Phase 6 manifest and per-system run logs and
produces:

1. A **results table** with per-system runtime / token / reparse
   summaries.
2. A **cost summary** (measured vs. target tiles, deltas, tolerance).
3. A **reparse summary** with verifier failure-code histograms.
4. **Qualitative examples** — a per-page join across all systems.
5. An **integration audit** report (11 checks) that fails loudly when
   any phase boundary breaks.
6. A top-level **analysis manifest** pinning phase-6 SHAs, frozen
   artifact SHA, and git HEAD.

Phase 7 is **read-only** with respect to every Phase 1–6 artifact. It
loads no model. It writes only under `<output.root>/<run_set_id>/`.

## What this phase does NOT do

- **No accuracy scoring.** The MVP repo runs entirely on stub adapters.
  Any TEDS or edit-distance number computed on stub output would be
  fictional. The audit reports
  `accuracy_status: not_applicable_stub_adapters` as an explicit warn.
- **No re-calibration.** Frozen budgets are read; never written.
- **No re-running of Phase 6.** If `outputs/runs/phase6/manifest.json`
  is stale, re-run Phase 6 first (see
  [`phase6_main_runs.md`](phase6_main_runs.md)).

## Prerequisites

1. Phase 6 has produced a complete manifest:
   `outputs/runs/phase6/manifest.json` with one entry per required
   system (and one per random seed). Re-run Phase 6 in full if this
   file is partial.
2. The frozen calibration artifact exists at
   `configs/calibration/frozen_budgets.json` and has not been edited
   since Phase 6 ran (the audit verifies this).
3. The held-out and calibration split manifests exist under
   `data/splits/`.

## How to run

```bash
uv run python scripts/analysis/run_phase7.py \
    --config configs/analysis/phase7.yaml
```

Exit code:

- `0`: every audit check returned `ok` or `warn`.
- `1`: at least one audit check returned `fail`.

Warns are surfaced in stdout but do **not** trip a non-zero exit. The
two warns Phase 7 always emits on this MVP are
`accuracy_status` (stubs in use) and
`random_baseline_seeds_distinct` (degenerate stub probability).

## Outputs

```
outputs/analysis/<run_set_id>/
├── analysis_manifest.json   # provenance: SHAs, git HEAD, generated_at
├── results_table.json       # per-system runtime / tokens / cost / reparse
├── cost_summary.json        # measured vs target tiles, deltas
├── reparse_summary.json     # reparse rate + verifier failure-code histograms
├── qualitative_examples.jsonl  # per-page join across all systems
└── audit_report.json        # 11 integration audit checks
```

`<run_set_id>` is taken from the Phase 6 manifest header (currently
`phase6_main_v1`).

## The 11 audit checks

| name | meaning |
|---|---|
| `phase1_splits_disjoint` | calibration ∩ held_out is empty |
| `phase6_manifest_sha_matches_held_out` | manifest header SHA equals current held-out file SHA |
| `phase6_manifest_sha_matches_frozen` | manifest header SHA equals current frozen-budgets SHA |
| `phase6_entries_complete` | all 5 expected families + 1 entry per random seed are present |
| `phase6_entries_match_disk` | every ok entry has a `run.log.jsonl` and matching `pages_processed` |
| `phase6_log_pages_match_held_out` | log page_ids ⊆ held-out split (no calibration leakage) |
| `frozen_artifact_unchanged_by_phase6` | frozen file has not been edited since Phase 6 wrote its manifest |
| `verifier_decision_codes_known` | all observed failure codes are constants in `verifier.codes` |
| `prompt_id_pinned` | every log entry has the same `prompt_id` as the manifest header |
| `random_baseline_seeds_distinct` | ≥2 random seeds produce distinct decision sequences (warn on stubs) |
| `accuracy_status` | warn on stub adapters; ok only when real adapters are wired |

A `fail` on any check means a real contract violation; investigate
before drawing any conclusion from the analysis outputs.

## Module map

```
src/adaptive_inference/analysis/
├── config.py        # Phase7Config + load_phase7_config
├── loaders.py       # Phase 6 manifest + run.log.jsonl + split readers
├── results.py       # SystemResult + summarize_system (reuses calibration.summary)
├── cost.py          # CostSummary (reuses calibration.cost helpers)
├── reparse.py       # ReparseSummary + verifier failure-code histograms
├── qualitative.py   # PageRow + iter_jsonl_rows (no raw output dumps)
├── audit.py         # AuditReport + 11 checks
└── runner.py        # run_phase7 orchestrator
scripts/analysis/run_phase7.py
configs/analysis/phase7.yaml
```

## When real adapters land

1. Flip `configs/models/internvl2.yaml` from `adapter_kind: stub` to
   the real adapter.
2. Re-freeze Phase 5 (writes a new `frozen_budgets.json`).
3. Re-run Phase 6 in full.
4. Re-run Phase 7. The `accuracy_status` audit check will switch to
   `ok` and a separate scorer must be wired in to compute TEDS /
   edit-distance — that scorer is **deferred work** (see
   `docs/PROJECT_STATE.md`).
