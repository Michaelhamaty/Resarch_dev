# Phase 1 runbook — freeze the experiment universe

Phase 1 locks down the exact set of page IDs the rest of the project will
operate on. It is pure metadata work: no model, no verifier, no calibration.

## What it produces

Four JSON manifests under the configured output directory (default
`data/splits/`):

| File | Meaning |
| --- | --- |
| `eval_universe.json` | All English table pages (the MVP universe). |
| `hard_subset.json` | Pages matching the placeholder "hard-table" rule. |
| `calibration_split.json` | Pages chosen for budget calibration. |
| `held_out_eval_split.json` | Pages held out for final evaluation. |

Each manifest embeds a pinning header (`dataset_name`, `snapshot_version`,
`eval_code_commit`, `source`, `notes`) so stray files remain self-describing.
`page_ids` are sorted alphabetically for stable diffs.

## Key invariants

- `calibration_split` ∩ `held_out_eval_split` = ∅ (enforced by the splitter).
- `calibration_split` ∪ `held_out_eval_split` ⊆ `eval_universe`.
- `hard_subset` ⊆ `eval_universe`; it is a **parallel dimension**, not a
  split. A page may be in both the hard subset and either split.
- Splits are deterministic under a fixed seed — re-running the freeze twice
  produces byte-identical manifests.

## How to run

```bash
uv sync --extra dev
uv run python scripts/subset_extraction/build_phase1_manifests.py \
    --config configs/dataset/phase1.yaml
```

Re-freezing is idempotent: the script overwrites the four JSON files and any
two consecutive runs will be byte-identical. If you need a new split, change
the seed or split sizes in `configs/dataset/phase1.yaml` — don't edit the
manifests by hand.

## Config reference

`configs/dataset/phase1.yaml` sections:

- `pinning`: identity of the frozen dataset (dataset name, snapshot version,
  evaluation code commit placeholder, source path, notes).
- `input.records_path`: path to a JSON list of page-record dicts.
- `subsets.hard_table_rule`: placeholder hard-table rule. A page is "hard"
  if any threshold is met or any flag in `require_any_of` is true. Supported
  flags: `has_merged_cells`, `has_nested_headers`.
- `split.seed`, `split.calibration_size`, `split.eval_size`: deterministic
  split parameters. `eval_size: null` means "take the remainder of the eval
  universe."
- `output.manifests_dir`: where the four JSON manifests are written.

## When real OmniDocBench lands

The rest of the pipeline consumes `list[PageRecord]` from
`adaptive_inference.dataset.records`. To plug in real data, add a new
loader that emits the same dataclass (e.g. an `omnidocbench_loader.py`
alongside `records.py`) and point `input.records_path` at it — or extend
`load_page_records` with a second file-format branch. The manifest writer
and the splitter do not need to change.

## Why this matters

The brief's Contract 6 requires calibration to be frozen *before* held-out
evaluation, and Risk #6 is calibration leakage. Phase 1 is the mechanism
that makes both guarantees concrete: once the manifests are written to
`data/splits/` and committed, downstream runs reference them by file and
cannot accidentally mix sets.
