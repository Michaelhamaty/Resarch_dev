
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
