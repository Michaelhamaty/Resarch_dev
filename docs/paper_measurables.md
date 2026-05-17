
## 2026-05-17 — Phase 2 + Phase 4 on real OmniDocBench (20-page calib)
After pinning `transformers==4.44.2`, dropping notebook patches B–E, and
adding the markdown-pipe-table → HTML scorer normalizer.

| run | macro_cell_f1 | macro_text_sim | parse_errors / 20 |
|---|---:|---:|---:|
| Phase 2 single-pass low (4 tiles)         | 0.0298 | 0.1298 | 11 |
| Phase 4 adaptive (4 → 12 on reparse)      | 0.0644 | 0.2939 |  4 |

Adaptive lift: 2.16× cell-F1, 7/11 parse-error pages rescued.
Not yet matched-cost (adaptive ~2.65× baseline tile budget) — Phase 5/6 next.
