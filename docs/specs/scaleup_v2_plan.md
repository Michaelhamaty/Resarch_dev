# Scale-Up v2 Execution Plan

**Final save location on approval:** `docs/specs/scaleup_v2_plan.md`
**Working branch:** `scaleup/v2` off `main`
**Author handoff:** This plan is to be reviewed by the user's professor before any execution begins.

---

## Context

The MVP pipeline (Phases 1–7) runs end-to-end on real OmniDocBench data, but the headline result is unpublishable as-is: n=20, in-sample (calibration split == eval split), 8B baseline is a stub, single dataset, no error bars, and the only verifier signal that fires is `NO_TABLE_FOUND`. The pipeline plumbing is ~85% done; the science is ~15% done. This plan turns the pipeline into something defensible at an IEEE student venue (CogMI / RISC) by adding the things every reviewer would demand: a real 8B upper bound, a held-out split disjoint from calibration, a second dataset (FinTabNet) for cross-dataset transfer, bootstrap CIs + paired tests + TEDS, and stratification by table complexity. It deliberately does **not** try to fix the verifier (that becomes future work) — the paper's framing is the rigorous negative result from §8 of the prior brainstorm. Three working days of execution, then paper writing downstream.

## One-paragraph summary

Over three working days on a single GCE L4-24GB VM, port the existing pipeline from local stubs to real-adapter runs on two datasets (OmniDocBench + FinTabNet, n=200 each, 50 calibration / 150 held-out, stratified by row count). Wire a real InternVL2-8B adapter (real 2B adapter already exists). Add resumable JSONL logging with a manifest and `--resume` mode. Run a (dataset × model × budget × policy) sweep overnight on day 2. On day 3, extend the analysis layer with bootstrap 95% CIs, paired Wilcoxon, TEDS, and difficulty stratification; generate paper-ready tables and figures; rewrite the Results / Methodology / Limitations sections of `paper/research_paper.tex`. Related-work search is explicitly out of the 3-day intensive and allocated half a day on day 4+.



## Stage-by-stage execution

Each stage names its deliverables, files touched, and the verification gate that must pass before the next stage starts. **Do not advance a stage with a failing gate.**

### Stage 0 — Preflight (≤2 hours, local)

**Goal:** Make sure the v2 branch boots cleanly and the existing MVP still passes its tests before any divergence.

- Create branch `scaleup/v2` off `main`. Do not delete or rewrite `outputs/runs/phase6_omnidocbench/` — it is the canonical MVP result for the paper's "MVP baseline" sentence.
- Run `uv sync --extra dev` and `uv run pytest -q` locally. Confirm 284 tests still green.
- Confirm GCP project ID, billing enabled, GPU quota for `nvidia-l4` in your chosen region (likely `us-central1`). Default quota is 0 GPUs; you must request an increase **before** day 1 or stage 1 blocks. Quota requests can take hours to a day.
- New directories to create later (do not pre-create empty ones):
  - `scripts/scaleup/` — orchestration scripts for this effort
  - `configs/experiment/scaleup_v2.yaml` — top-level sweep config
  - `configs/dataset/fintabnet.yaml` — FinTabNet loader config
  - `outputs/scaleup_v2/` — all new run artifacts land here, mirrored under per-system subdirs

**Gate G0:** branch exists, all tests green locally, L4 GPU quota approved in target region. If quota is not approved, request it now and continue with stage 0 tasks; do not start stage 1 until granted.

---

### Stage 1 — GCP infrastructure (Day 1 morning, ~3 hours)

**Goal:** A reproducible, resumable VM with the repo cloned, dependencies installed, and a tmux/SSH workflow that survives disconnects.

**For someone new to GCP, the literal sequence is:**

1. Install `gcloud` CLI locally; `gcloud auth login`; `gcloud config set project <your-project>`.
2. Create VM with a **Deep Learning VM image** (CUDA + PyTorch preinstalled, saves hours):
   - Machine type: `g2-standard-8` (8 vCPU, 32 GB RAM, includes 1× L4 24 GB)
   - Image family: `pytorch-latest-gpu` from project `deeplearning-platform-release`
   - Boot disk: 200 GB SSD (model checkpoints + datasets)
   - Network tags: allow SSH only; no public ports
3. SSH in via `gcloud compute ssh`. Accept the NVIDIA driver install prompt on first boot.
4. `git clone` the repo (HTTPS + a PAT, or SSH key uploaded via `gcloud compute os-login`).
5. Install `uv`; `uv sync --extra dev`. Install `tmux` if not present.
6. Verify GPU: `nvidia-smi` shows L4 24 GB, CUDA visible to PyTorch (`python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`).
7. Establish a **tmux-only workflow** for any run longer than ~5 minutes. Document the three commands you need (`tmux new -s run`, detach `Ctrl+B d`, reattach `tmux attach -t run`) in your runbook. This is the single biggest source of "I lost 4 hours of work because my laptop slept" pain on GCP.
8. **Stop the VM** when not running (you pay for disk only while stopped, ~$0.04/GB-month).

**Files created:** `docs/runbooks/gcp_vm_setup.md` (a short runbook so professor can reproduce — VM image name, machine type, gcloud commands, tmux cheatsheet).

**Gate G1:** `nvidia-smi` shows L4; `uv run pytest -q` passes on the VM; tmux verified by detach/reattach cycle. **Cost so far: ~$2.**

---

### Stage 2 — Real 8B adapter smoke test (Day 1 morning, hard ½-day cap)

**Goal:** Confirm InternVL2-8B loads, runs one page, and produces sane HTML on the L4 — before committing any further time.

**Files touched:**

- `configs/models/internvl2_real.yaml` — flip `internvl2-8b` from `adapter_kind: stub` to `adapter_kind: internvl2`. Add `dtype: float16`, `device: cuda:0` if not present.
- `configs/calibration/frozen_budgets.json` — change `B_fix_8B.adapter_kind` from `"stub"` to `"internvl2"`. Do **not** re-freeze budgets yet — that's stage 5/6.
- No changes to `src/adaptive_inference/inference/factory.py` or `src/adaptive_inference/inference/internvl2.py` are needed; the dispatch already supports `"internvl2"`. The 2B adapter file already implements the loading pattern we need to reuse for 8B.
- New script: `scripts/scaleup/smoke_8b_one_page.py` — load 8B, run one OmniDocBench page through `InternVL2Adapter`, print `raw_text`, `output_token_count`, `runtime_ms`, peak VRAM (`torch.cuda.max_memory_allocated`).

**Checks the smoke script must perform and print:**
1. `nvidia-smi` peak VRAM after generation is ≤ 22 GB (leaves 2 GB headroom on L4 24 GB).
2. Tokenizer is the same family as 2B (same `<image>` placeholder handling); no shape mismatch.
3. Greedy decoding produces deterministic output across two consecutive runs of the same page (byte-identical).
4. `raw_text` contains at least one `<table>` tag on a known-table page.

**Hard gate G2 (½ day cap):** All four checks pass. If any fails by end of Day 1 morning (~4 hr in), **stop trying to fix 8B**. Fallback path: leave 8B stubbed, drop "5-system comparison with real 8B" from the headline claim, demote 8B to a "future work / requires larger GPU" sentence in Limitations, and proceed with the rest of the plan using only 2B systems. The plan still produces a defensible paper; it just loses the gap-closure framing.

---

### Stage 3 — Resumable JSONL + manifest + --resume mode (Day 1 afternoon, ~3 hours)

**Goal:** Eliminate the "OOM at hour 4 loses 4 hours of work" failure mode. The current `src/adaptive_inference/runner/adaptive_logger.py` opens with `"a"` mode, does not fsync, and `reset_adaptive_log()` truncates on re-run. This is unacceptable for a 10+ hour sweep.

**Files modified:**

- `src/adaptive_inference/runner/adaptive_logger.py` — add `flush() + os.fsync(fileno())` after every `append_adaptive_page_log()` write. Remove or guard `reset_adaptive_log()` behind an explicit `--reset` CLI flag.
- New module: `src/adaptive_inference/runner/resume.py` — exposes:
  - `write_manifest(run_dir, manifest: dict)` — writes `manifest.json` with `run_id`, `dataset_id`, `system_id`, deterministic `page_ids` list, git SHA, config SHA, started_at.
  - `read_completed_page_ids(run_dir) -> set[str]` — reads `run.log.jsonl`, returns set of `page_id`s with a terminal `status`.
  - `pending_pages(manifest_page_ids, completed_page_ids) -> list[str]` — preserves manifest order.
- `scripts/main_runs/run_phase6.py` and the new `scripts/scaleup/run_sweep.py` — accept `--resume` flag; on resume, skip pages already in `run.log.jsonl`, append to the same file rather than truncate.
- Unit test: `tests/runner/test_resume.py` — simulate a partial run (5 of 10 pages written), invoke with `--resume`, assert the remaining 5 run and final log has 10 lines in original order.

**Gate G3:** unit test passes; manual test runs the existing OmniDocBench 20-page set, kills it midway with `Ctrl+C`, resumes, ends with 20 deterministic lines matching a single uninterrupted run byte-for-byte (modulo timestamps).

---

### Stage 4 — FinTabNet dataset loader (Day 1 late afternoon → early evening, ~4 hours)

**Goal:** A FinTabNet loader that produces `PageRecord`s of the same shape as the existing OmniDocBench loader, with PNG images pre-rendered from source PDFs.

**Files created:**

- `configs/dataset/fintabnet.yaml` — mirror the structure of `configs/dataset/phase1_omnidocbench.yaml`. Names the source PDF/JSON paths, output PNG dir, DPI, English-only filter (FinTabNet is English by construction; filter is a no-op but kept for symmetry), row/col extraction rules.
- `src/adaptive_inference/dataset/fintabnet.py` — mirror `src/adaptive_inference/dataset/omnidocbench.py`. Exposes `SelectedPage` extraction with table HTML, row/col counts, span detection. Returns `PageRecord`s with the same field shape so downstream code (`freeze.py`, `manifests.py`, runner) is dataset-agnostic.
- `scripts/data/build_fintabnet_fixture.py` — one-time prep:
  1. Download FinTabNet from IBM's distribution (Zenodo / HuggingFace mirror — verify URL on day 1).
  2. For each table entry, render the source PDF page to a PNG at fixed DPI (recommend 144 DPI, matches the OmniDocBench fixture). Use `pdf2image` (requires Poppler) or `PyMuPDF` (`fitz`, lighter).
  3. Normalize the GT HTML to match the existing scorer's expected shape (the existing `src/adaptive_inference/analysis/cell_f1.py` consumes a specific HTML normalization; the FinTabNet loader must produce GT in the same shape — verify on a 3-page sample first).
  4. Write a `data/fintabnet/manifest.jsonl` describing every page with `page_id`, `image_path`, `gt_html_path`, `row_count`, `col_count`, `has_merged_cells`.
- New script: `scripts/scaleup/smoke_fintabnet_5pages.py` — loads 5 pages via the loader, runs them through the real 2B adapter, scores against GT with the existing scorer, prints cell-F1 per page. Sanity check: numbers in the same ballpark as OmniDocBench at the same budget.

**Gate G4:** smoke script on 5 FinTabNet pages produces non-zero cell-F1 (≥ 0.05), no parser exceptions, GT HTML loads without normalization warnings.

**Risk callout:** FinTabNet's licensing and exact distribution channel needs verification on day 1 morning. If the distribution is unavailable or the GT normalization gap is large, the fallback is to scope back to OmniDocBench-only with deeper stratification (see Risk #1 below).

---

### Stage 5 — Stratified n=200 splits (Day 2 morning, ~1.5 hours)

**Goal:** Frozen, stratified, disjoint calibration (50) and held-out (150) splits for each dataset.

**Files created:**

- `scripts/scaleup/build_scaleup_splits.py` — for each dataset:
  1. Load all candidate English-table pages via the dataset module.
  2. Bucket by row-count: `simple` (<6 rows), `complex` (6–15 rows), `very_complex` (≥16 rows OR has merged cells).
  3. Stratified random sample, seeded with `0xSCALEUPv2`, to produce 200 pages with the same per-bucket proportions as the population (or a floor of N per bucket if any bucket is sparse).
  4. Disjoint split into 50 calibration and 150 held-out per dataset, also stratified.
  5. Write frozen manifests: `data/splits/scaleup_v2/{omnidocbench,fintabnet}/calibration.json` and `.../held_out.json`. Each is a JSON list of `page_id`s plus a header with seed, SHA, sampling rule.
- Reuse `src/adaptive_inference/dataset/freeze.py` and `src/adaptive_inference/dataset/manifests.py` patterns.

**Gate G5:** the four manifest files exist; their page_id sets are disjoint per dataset (`calib ∩ held_out == ∅`); bucket proportions printed and reasonable.

---

### Stage 6 — Re-calibrate and re-freeze budgets on real data (Day 2 morning–early afternoon, ~3 hours GPU)

**Goal:** Replace the existing `configs/calibration/frozen_budgets.json` values (chosen on stub data) with budgets chosen on real 2B and real 8B outputs on the **calibration split only** (not held-out).

**Files touched:**

- New: `configs/calibration/scaleup_v2.yaml` — calibration sweep config pointing at the new calibration splits and real adapters.
- Script reused: `scripts/calibration/run_calibration.py` (already exists).
- New: `configs/calibration/frozen_budgets_v2.json` — written by the calibration run. Do **not** overwrite the old `frozen_budgets.json`; keep it for the MVP record.

**Sweep:** tile budgets `{2, 4, 6, 8, 10, 12, 16}` on both datasets, both models. ~7 budgets × 2 models × 2 datasets × 50 pages = 1400 page-inferences. At ~20s avg, that's ~8 GPU-hours. Run in tmux.

**Gate G6:** for each (dataset, model) the cost-vs-cell-F1 curve is monotone non-decreasing (within noise) and produces a defensible `B_low`, `B_high`, and matched-cost budgets `B_fix_2B`, `B_fix_8B`. If the curves are flat or inverted, *stop* — something is wrong with the adapter or scoring and the main sweep would be wasted GPU. Investigate before stage 7.

---

### Stage 7 — Main sweep (Day 2 afternoon → overnight, ~12 GPU-hours)

**Goal:** One sweep, fully resumable, that produces every number the paper needs.

**Loop axes (in this order, outermost to innermost):**

```
for dataset in [omnidocbench, fintabnet]:                 # 2
  for model in [internvl2-2b, internvl2-8b]:              # 2
    for policy in [fixed_low, fixed_matched, adaptive]:    # 3 (model-relevant only)
      for seed in seeds_for_policy(policy):               # adaptive=1, fixed=1, random=3
         run on held_out_150
```

Explicit system list per dataset (matching the MVP's 7-system table):

1. `fixed_2b_low` — 2B at `B_low` tiles
2. `fixed_2b_matched` — 2B at matched cost
3. `fixed_8b_matched` — 8B at matched cost (the real upper bound this plan unlocks)
4. `adaptive_2b` — 2B with deterministic verifier escalation to `B_high`
5. `random_2b_seed0`, `random_2b_seed1`, `random_2b_seed2` — random-escalation control

Total: 7 systems × 2 datasets × 150 held-out pages = **2,100 page-inferences**. Adaptive and random also touch first-pass + reparse, so wall-clock is closer to ~3,000 inferences. At ~15–25 s/page on L4, expect **8–14 GPU-hours**. Start before dinner, check on it after dinner, expect completion by morning.

**Difficulty is post-hoc, not a loop axis.** The runner does not know about difficulty buckets; the analyzer slices on `row_count` / `has_merged_cells` from the manifest.

**Files created:**

- `configs/experiment/scaleup_v2.yaml` — top-level sweep config (datasets, models, policies, seeds, output root).
- `scripts/scaleup/run_sweep.py` — outer loop driver. For each (dataset, system) it: writes a per-system manifest, invokes the existing runner (`run_phase6` codepaths) with `--resume`, fsyncs after each page.
- All artifacts under `outputs/scaleup_v2/{dataset}/{system_id}/` mirroring the existing layout (`first_pass/`, `reparse/`, `final/`, `run.log.jsonl`, `manifest.json`).

**Gate G7 (end of Day 2, the user-mandated stance gate):**

Run a *fast preliminary analysis* on whatever completed by end of day 2 (likely all of dataset 1, partial dataset 2) — just per-system mean cell-F1, no CIs yet. Then explicitly decide:

- **Stance A (default):** rigorous negative result. Proceed to stage 8 with current scope.
- **Hybrid:** if real-data results suggest a simple content-quality signal (e.g., cell-count consistency across rows) would clearly close the gap on visible cases, flip to hybrid and add a small subsection to the paper. **Do not add new code on day 3 unless the signal is obvious from inspection** — if you're squinting at the numbers, the answer is "stay with A."

Document the decision in `outputs/scaleup_v2/STANCE_DECISION.md` with the preliminary numbers that justified it. Sweep continues to finish overnight regardless.

---

### Stage 8 — Statistics layer (Day 3 morning, ~3 hours)

**Goal:** Add the four non-negotiable statistical artifacts. No existing code provides any of these (verified — `src/adaptive_inference/analysis/cell_f1.py` only computes per-page cell-F1 and text similarity; nothing for CIs, paired tests, or TEDS).

**Files created:**

- `src/adaptive_inference/analysis/bootstrap.py` — `bootstrap_ci(per_page_values, n_resamples=1000, ci=0.95) -> (mean, lo, hi)`. Page-level resampling with a fixed seed for reproducibility.
- `src/adaptive_inference/analysis/paired_tests.py` — `paired_wilcoxon(per_page_a, per_page_b) -> (statistic, p_value, n_pairs)`. Uses `scipy.stats.wilcoxon`. Paired on `page_id`, both sides must score the same page set.
- `src/adaptive_inference/analysis/teds.py` — TEDS implementation. Two options:
  1. Vendor the reference implementation from the PubTabNet repo (Zhong et al., 2019) — recommended, well-tested.
  2. Use `apted` + a custom HTML → tree converter.
  Pick (1); cite the source in the file header.
- `src/adaptive_inference/analysis/stratify.py` — `stratify_by_difficulty(scores, manifest) -> {bucket: [scores]}` keyed on the row-count buckets from stage 5.
- Tests: `tests/analysis/test_bootstrap.py`, `tests/analysis/test_paired_tests.py`, `tests/analysis/test_teds.py` (synthetic tables with known TEDS expectations).

**Gate G8:** all four modules unit-tested; `pytest -q` clean. Bootstrap CI on a synthetic constant array returns CI width ≈ 0; Wilcoxon on identical arrays returns p=1.0; TEDS on identical HTML returns 1.0.

---

### Stage 9 — Results aggregation, tables, figures (Day 3 afternoon, ~3 hours)

**Goal:** A single artifact tree that contains every number and figure the paper's Results section needs.

**Files created:**

- `scripts/scaleup/run_phase7_v2.py` — extends the existing `scripts/analysis/run_phase7.py` codepath:
  - Walks every `outputs/scaleup_v2/{dataset}/{system_id}/run.log.jsonl`.
  - Scores against GT (existing scorer + new TEDS).
  - Produces a per-(dataset, system, bucket) table with mean, bootstrap-CI, n.
  - Produces paired Wilcoxon p-values for adaptive vs fixed_2b_matched and adaptive vs random (averaged across seeds).
  - Writes `outputs/scaleup_v2/analysis/results_v2.json` (machine-readable) and `results_v2.md` (paper-paste).
- `scripts/scaleup/make_figures.py` — produces the three figures the paper needs:
  1. Cost-vs-accuracy scatter, one panel per dataset, error bars from bootstrap CI.
  2. **The diagnostic scatter** (the most important new figure): per page, x = first-pass cell-F1, y = reparse cell-F1 lift if 10-tile baseline is used as reparse, colored by verifier decision (PASS vs REPARSE). This is the figure that *visually proves* the negative result and motivates the future-work direction.
  3. Stratified bar chart: cell-F1 by difficulty bucket, grouped by system, with CIs.
- Output: `outputs/scaleup_v2/analysis/figures/*.{png,pdf}` (PDF for LaTeX inclusion).

**Gate G9:** `results_v2.md` opens cleanly; all three figures render; no NaN cells; per-system n matches manifest size; nothing labelled "stub" in the 8B column.

---

### Stage 10 — Paper update pass (Day 3 evening, ~3 hours)

**Goal:** Replace placeholder numbers in `paper/research_paper.tex` and update the affected sections. Paper writing proper (related work, polish) is downstream.

**Sections to revise (line ranges from the current draft):**

- **Abstract (lines 22–46):** update headline to reflect cross-dataset, real-8B, held-out matched-cost finding. Tone: rigorous negative result with localized cause.
- **System Architecture → dataset section (lines 218–302):** add FinTabNet description; update "MVP" framing to "scale-up" framing.
- **Methodology → calibration (lines 341–375):** describe the held-out split protocol, the stratified sampling, the n=200/dataset count.
- **Methodology → evaluation metric (lines 420–435):** add TEDS alongside cell-F1; cite Zhong et al.
- **Results (lines 438–566):** complete rewrite. New tables (per-dataset, per-bucket, with CIs and p-values). Insert the three figures.
- **Discussion → Limitations (lines 627–671):** rewrite. The current "in-sample, n=20, 8B stubbed" limitations no longer apply; new limitations are "two datasets, not population-wide" and "verifier richness untested" (the future-work hook).

**Files touched:** `paper/research_paper.tex`, `docs/paper_measurables.md` (append new row per the user's `MEMORY.md` rule about logging real-adapter runs).

**Gate G10:** paper compiles (`pdflatex`); every results number traces back to a cell in `results_v2.json`; no remaining `TODO` or `forthcoming` markers in Results.

---

### Stage 11 — Related work + polish (Day 4+, out of 3-day intensive)

**Goal:** Half-day allocation, post-runs. Not part of the 3-day intensive.

**Citation buckets to populate:**

- **Table benchmarks:** Zhong et al. 2019 (PubTabNet + TEDS), Zheng et al. 2021 (GTE/FinTabNet), Ouyang et al. 2024 (OmniDocBench), Smock et al. 2022 (PubTables-1M).
- **VLMs for documents:** Chen et al. 2024 (InternVL2 / InternVL-1.5), Liu et al. 2023 (LLaVA), Hu et al. 2024 (mPLUG-DocOwl).
- **Adaptive / cascade inference:** Chen et al. 2023 (FrugalGPT), Schuster et al. 2022 (CALM), Teerapittayanon et al. 2016 (BranchyNet), Leviathan et al. 2023 (speculative decoding).
- **Verifier-based generation:** Cobbe et al. 2021 (verifiers for math), Lightman et al. 2023 ("Let's verify step by step").

Tool stack: Semantic Scholar API + citation graph walks. Half day total.

---

## Decision gates summary

| Gate | When | Pass criterion | Action on fail |
|---|---|---|---|
| G0 | End of preflight | Branch + green tests + L4 quota approved | Wait for quota; do not start stage 1 |
| G1 | End of stage 1 | `nvidia-smi` shows L4, tests pass on VM | Debug VM; do not start 8B work |
| **G2** | **End of Day 1 morning (½-day cap)** | **8B loads, runs sane HTML, fits VRAM, deterministic** | **Drop 8B from headline; demote to Limitations; proceed** |
| G3 | End of stage 3 | Resume test produces byte-identical log | Fix before any long run |
| G4 | End of stage 4 | 5 FinTabNet pages score nonzero cell-F1 | See Risk #1 |
| G5 | End of stage 5 | Disjoint stratified manifests written | Re-seed and retry |
| G6 | End of stage 6 | Calibration curves monotone | Investigate adapter; do not start sweep |
| **G7** | **End of Day 2 (stance gate)** | **Stance A vs Hybrid decision documented in `STANCE_DECISION.md`** | **Default to A** |
| G8 | Stage 8 | All four stats modules unit-tested | Fix before aggregation |
| G9 | Stage 9 | Results JSON + 3 figures generated | Fix before paper update |
| G10 | Stage 10 | Paper compiles, every number traceable | Don't ship until clean |

---

## Risks and fallbacks (top 5)

| # | Risk | Probability | Cheapest fallback |
|---|---|---|---|
| 1 | **FinTabNet data unavailable / GT normalization gap too large** | Med | Scope back to OmniDocBench-only with deeper stratification (n=400, 4 buckets). Lose cross-dataset transfer claim, keep everything else. Add to Limitations. |
| 2 | **8B adapter fails on L4 (VRAM, tokenizer, model card)** | Med | Per G2: leave 8B stubbed, drop gap-closure framing, ship a "matched-cost adaptive 2B vs matched-cost fixed 2B" paper. Note in Limitations that 8B requires ≥40 GB GPU and is future work. |
| 3 | **GCP quota delay (L4 quota not granted in time)** | Med | Request increase Day 0; if blocked, fall back to T4 16 GB which fits 2B but not 8B at fp16 → triggers Risk #2 fallback. Or request A100 spot as a stretch (~$1.10/hr). |
| 4 | **Main sweep doesn't finish overnight (>14 GPU-hours)** | Low-Med | Resume logic (stage 3) makes this trivial: cut held-out to 100 per dataset, restart with `--resume`, rerun completes by lunch on day 3. Document n=100 in paper. |
| 5 | **TEDS implementation bugs / disagreement with cell-F1** | Low-Med | Vendor the PubTabNet reference implementation rather than reimplementing. If still flaky, ship cell-F1 + parse-error rate only and note TEDS as future work — defensible because cell-F1 is more conservative anyway. |

---

## Cost envelope (single L4-24GB on-demand, ~$0.80/hr all-in)

| Stage | GPU-hours | Wall-clock | Est. cost |
|---|---:|---:|---:|
| Stage 1 (setup, idle) | 1 | 3 hr | $1 |
| Stage 2 (8B smoke) | 0.5 | 0.5 hr | $0.40 |
| Stage 4 (FinTabNet smoke) | 0.5 | 4 hr | $0.40 |
| Stage 6 (calibration, both datasets, both models) | 8 | 8 hr | $6.40 |
| Stage 7 (main sweep) | 14 | overnight | $11 |
| Stage 8–10 (CPU-only mostly) | 1 | 9 hr | $0.80 |
| **Subtotal** | **25** | | **~$20** |
| **+30% rerun/debug buffer** | 8 | | $6 |
| VM idle / stopped time | n/a | | ~$5 |
| **Total estimate** | | | **~$30** |
| **Budget envelope** | | | **$400** |

You have ~13× headroom. If a single stage blows up and needs to be entirely re-run, the budget still survives. The cost risk is essentially zero; the *time* risk is real.

---

## Notebooks vs scripts

- **Scripts (everything that runs on the VM, all long jobs):** every stage 0–8 deliverable. Reason: notebooks have hidden kernel state and die on SSH disconnect. Use `tmux + python script.py` exclusively for any run >5 minutes.
- **Notebooks (local, day 3+):** exploratory analysis of `results_v2.json`, prototyping figures, sanity-checking bootstrap distributions. Final figures get re-generated by `scripts/scaleup/make_figures.py` for reproducibility — never paste a notebook figure straight into the paper.
- **Colab specifically:** do not use it as a runtime. It is acceptable as a viewer/SSH frontend to the GCE VM if you prefer Colab's UI, but every actual run command must originate from `tmux` on the VM.

---

## What this plan deliberately excludes

- **No verifier richness work.** Adding content-quality signals (cell-count consistency, row-length entropy, etc.) is the natural next paper. Doing it in this 3-day window would compromise stats / figures / paper quality.
- **No PubTabNet.** Cropped tables, breaks the full-page contract, would force a separate cost model.
- **No third dataset** (SciTSR, ICDAR cTDaR, TableBank).
- **No multilingual.** Hard veto.
- **No rotated table handling.**
- **No new repo.** Branch `scaleup/v2` off `main`.
- **No multi-step escalation** (low → mid → high).
- **No crop repair.**
- **No new policy module rewrites.** Existing `src/adaptive_inference/policy/` stays untouched.
- **No learned routing.**
- **No second VM, no spot, no A100.** Single on-demand L4.
- **No paper writing in the 3-day window.** Sections get *updated* on day 3 evening; full related-work / discussion polish is downstream.

---

## End-to-end verification plan

After stage 10, the deliverable is verified by:

1. `git status` clean on `scaleup/v2`; commits annotated per stage.
2. `uv run pytest -q` green (~290+ tests, new stats + resume tests added).
3. `outputs/scaleup_v2/analysis/results_v2.md` opens; every cell has a mean, bootstrap-CI, and an n.
4. `outputs/scaleup_v2/analysis/figures/` contains 3 PDFs.
5. `paper/research_paper.tex` compiles; `results_v2.json` numbers match the LaTeX tables (spot-check 5 cells).
6. `docs/paper_measurables.md` has a new dated row per the user's `MEMORY.md` rule.
7. `outputs/runs/phase6_omnidocbench/` is untouched (MVP record preserved).
8. `outputs/scaleup_v2/STANCE_DECISION.md` documents the day-2 gate decision with the preliminary numbers that justified it.
9. VM stopped, billing dashboard confirms ≤$50 spent (well under $400 envelope).
10. Branch is ready for PR review by the professor before merge to `main`.
