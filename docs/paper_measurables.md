
## 2026-05-17 — Phase 2 + Phase 4 on real OmniDocBench (20-page calib)
After pinning `transformers==4.44.2`, dropping notebook patches B–E, and
adding the markdown-pipe-table → HTML scorer normalizer.

| run | macro_cell_f1 | macro_text_sim | parse_errors / 20 |
|---|---:|---:|---:|
| Phase 2 single-pass low (4 tiles)         | 0.0298 | 0.1298 | 11 |
| Phase 4 adaptive (4 → 12 on reparse)      | 0.0644 | 0.2939 |  4 |

Adaptive lift: 2.16× cell-F1, 7/11 parse-error pages rescued.
Not yet matched-cost (adaptive ~2.65× baseline tile budget) — Phase 5/6 next.

## 2026-05-17 — Phase 6 + Phase 7 on real OmniDocBench (20-page calib, in-sample)

Real five-system matched-cost run on the 20-page OmniDocBench English-table
calibration split. **No held-out split exists yet — all numbers are in-sample
for Phase 5's budget pick.** 8B row is stub output.

| system | macro_cell_f1 | macro_text_sim | parse_errors / 20 | cost (tiles/page) |
|---|---:|---:|---:|---:|
| adaptive_2b           | 0.1272 | 0.2836 |  6 |  9.5 |
| fixed_2b_low          | 0.0298 | 0.1298 | 11 |  4.0 |
| fixed_2b_matched      | 0.1861 | 0.3329 |  9 | 10.0 |
| random_2b_seed0       | 0.0461 | 0.1635 | 10 |  7.0 |
| random_2b_seed1       | 0.1038 | 0.2354 | 10 | 11.5 |
| random_2b_seed2       | 0.1609 | 0.2780 | 11 |  8.5 |
| fixed_8b_matched (stub) | 0.0000 | 0.0764 |  0 | 10.0 |

Matched-cost finding: adaptive_2b underperforms fixed_2b_matched by 0.0589
absolute cell-F1 at near-matched cost (9.5 vs 10 tiles/page). The deterministic
verifier (NO_TABLE_FOUND only, 11/20 pages) rescues 5 parse-error pages but
misses the 9 verifier-passing pages that would still gain quality at 10 tiles.

Phase 7 audit: 9 ok, 2 warn (splits_identical MVP shortcut, accuracy_status
stub-adapter gate), 0 fail. Artifacts under
`outputs/analysis/phase6_omnidocbench_v1/`.

## 2026-05-29 — Stage 6 re-validation under the surgical loop-stop (50-pp calib splits)

The InternVL2 adapter now carries the exact-periodicity runaway loop-stop
(commit `6e2f89e`): it only **truncates** degenerate exact-repeat token tails
and never alters token choices. Re-ran the *frozen* budget points on each
50-page calibration split to confirm the loop-stop is cost-neutral (so the
frozen Stage-6 budgets remain valid). Single-point grids written to separate
`*_validate` artifacts; committed budgets not overwritten by the run.

| dataset | budgets (B_low/B_high/B_fix) | reparse | adaptive cost_tiles (committed → re-measured) | fixed within_tol |
|---|---|---:|---:|:--:|
| OmniDocBench | 2 / 12 / 11 | 0.72 (36/50) | 10.64 → **10.64** (identical) | true / true |
| FinTabNet    | 6 / 16 / 10 | 0.26→**0.28** (13→14/50) | 10.16 → **10.48** | true / true |

**OmniDoc:** byte-for-byte cost-neutral — same 36 pages reparse, cost unchanged.
**FinTab:** one extra borderline page reparses under the loop-stop (deterministic,
not noise), nudging adaptive cost 10.16 → 10.48 (3.1%, within the 0.10 matched
tolerance). Budgets (tile counts) are **unchanged** — single-point grid, no
re-selection possible.

**Action — re-froze FinTab `measured_cost_tiles` 10.16 → 10.48** in
`configs/calibration/frozen_budgets_v2_fintabnet.json` (also the two fixed
`target_cost_tiles`). Rationale: Stage 7's random-escalation probability is
derived as `p = (measured_cost_tiles − B_low) / B_high`
(`experiment/frozen_inputs.py:derive_calibration_reparse_rate`). Using the
loop-stop-measured 10.48 sets random `p = 0.28`, so adaptive / random /
fixed_matched all sit at 10.48 tiles/page — preserving exact matched-cost
fairness under the adapter Stage 7 actually runs. OmniDoc needed no change.

**Loop-stop behavioural note (accepted tradeoff):** the stopper reliably catches
*low-budget* single-token loops (B_low pages drop from the ~64s/2048-token cap to
seconds) but does **not** catch *matched/high-budget* runaways, whose tails are
not exactly periodic within the 100-token window — fixed_2b@matched still caps
~30% of pages at 2048 tok (~64s), and 8B@matched runaways cost ~157s/page. Net:
calibration is now affordable and the low-budget worst case is bounded, but the
full Stage 7 sweep remains in the ~15–19 GPU-hr envelope (8B dominates).
Validation pace (mean total/page): OmniDoc adaptive 57.8s, FinTab adaptive 22.1s.

## 2026-05-30 — Stage 7 main sweep (real adapters, held-out splits) + Stage 9 results

First full held-out evaluation. All 14 system-runs (7 systems × 2 datasets) ran
on real InternVL2 2B/8B adapters over the held-out splits (OmniDoc n=90,
FinTab n=150); **all status=ok**. Scored CPU-only by Stage 9
(`run_phase7_v2.py`): cell-F1 + TEDS with 95% bootstrap CIs and paired Wilcoxon.
Artifacts: `outputs/scaleup_v2/analysis/results_v2.{json,md}`, per-page
`diagnostic_{dataset}.jsonl`; figures in `paper/figures/`. git `cae3f7e`.

**OmniDocBench (n=90)**

| system | cost_tiles | macro_cell_f1 | macro_text_sim (TEDS) | parse_err/90 |
|---|---:|---:|---:|---:|
| fixed_2b_low        |  2.00 | 0.0387 | 0.1055 | 70 |
| fixed_2b_matched    | 11.00 | 0.1199 | 0.2683 | 57 |
| adaptive_2b         | 11.33 | 0.1208 | 0.2740 | 45 |
| random_2b_seed0     | 11.20 | 0.1230 | 0.2491 | 55 |
| random_2b_seed1     | 11.47 | 0.1196 | 0.2454 | 51 |
| random_2b_seed2     | 10.13 | 0.1000 | 0.2208 | 61 |
| fixed_8b_matched    | 11.00 | 0.1995 | 0.3963 | 38 |

**FinTabNet (n=150)**

| system | cost_tiles | macro_cell_f1 | macro_text_sim (TEDS) | parse_err/150 |
|---|---:|---:|---:|---:|
| fixed_2b_low        |  6.00 | 0.1298 | 0.4827 | 41 |
| fixed_2b_matched    | 10.00 | 0.1197 | 0.4646 | 37 |
| adaptive_2b         | 10.37 | 0.1334 | 0.4915 | 30 |
| random_2b_seed0     |  9.95 | 0.1157 | 0.4624 | 41 |
| random_2b_seed1     |  9.84 | 0.1190 | 0.4758 | 44 |
| random_2b_seed2     | 11.55 | 0.1242 | 0.4715 | 42 |
| fixed_8b_matched    | 10.00 | 0.1335 | 0.5267 | 46 |

**Paired Wilcoxon (cell-F1):**

| comparison | OmniDoc Δ (a−b) / p | FinTab Δ (a−b) / p |
|---|---|---|
| adaptive vs fixed-matched          | +0.0009 / 0.729 (null) | **+0.0137 / 0.0479 (sig.)** |
| adaptive vs random (pooled seeds)  | +0.0066 / 0.269 (null) | **+0.0138 / 0.000176 (sig.)** |

**G7 fairness invariant — PASSES.** Within each dataset adaptive ≈ random ≈
fixed_matched on cost_tiles (FinTab 10.37 / ~10.4 / 10.0; OmniDoc 11.33 / ~11.0 /
11.0), so cell-F1 differences reflect compute *allocation*, not *amount*. The
matched-budget comparison is valid.

**Finding.** On **FinTabNet the verifier-triggered adaptive policy works**: it
beats both the matched-cost baseline (p=0.048) and random escalation (p=0.0002)
at equal cost, and 2B-adaptive (0.1334) essentially **matches fixed_8b**
(0.1335) at ⅕ the model size. On **OmniDocBench the effect is null** — adaptive
≈ fixed_matched ≈ random; here only the bigger 8B model (0.1995) clearly helps.
Honest read: the "spend compute where the verifier flags trouble" thesis holds
on the lower-reparse-rate FinTab regime but not on the high-reparse (0.72)
OmniDoc regime, where nearly everything reparses and the allocation signal
washes out.

**Provenance note.** Per-system manifest snapshots were never stashed during the
sweep loop, so `manifest.json` initially listed only the last system. Both
dataset manifests were reconstructed from the authoritative `run.log.jsonl`
files via the new `scripts/scaleup/rebuild_manifest.py` (no GPU re-run; reused
header + 8B entry verbatim, reconstructed entries tagged with a provenance note).
Two latent post-processing bugs surfaced and were fixed with regression tests:
TEDS crash on HTML comments (`analysis/teds.py`), and a doubled-path resolution
bug in the adaptive first/final diagnostic (`analysis/run_scoring.py`).
