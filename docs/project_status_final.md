# Project Status — Final (research MVP complete)

**Date:** 2026-05-17
**Branch:** main
**Phase 6 manifest:** `outputs/runs/phase6_omnidocbench/manifest.json`
**Phase 7 artifacts:** `outputs/analysis/phase6_omnidocbench_v1/`
**Audit:** 9 ok, 2 warn, 0 fail (exit 0).

---

## Headline finding

**At matched cost (~10 tiles/page) on n=20 OmniDocBench English-table pages,
the deterministic adaptive system did not beat the fixed-2B matched baseline.**

| system | macro_cell_f1 | cost (tiles/pg) |
|---|---:|---:|
| `adaptive_2b` | 0.1272 | 9.5 |
| `fixed_2b_matched` | **0.1861** | 10.0 |

Fixed-2B-matched wins by **+0.0589 absolute cell-F1** (≈ +46% relative). Always
running at 10 tiles beats selectively running at 10 tiles under the current
verifier.

## Five-system table

| system | macro_cell_f1 | macro_text_sim | parse_errors / 20 | cost | reparse_rate |
|---|---:|---:|---:|---:|---:|
| adaptive_2b           | 0.1272 | 0.2836 |  6 |  9.5 | 0.55 |
| fixed_2b_low          | 0.0298 | 0.1298 | 11 |  4.0 | n/a  |
| fixed_2b_matched      | **0.1861** | **0.3329** |  9 | 10.0 | n/a  |
| random_2b_seed0       | 0.0461 | 0.1635 | 10 |  7.0 | 0.30 |
| random_2b_seed1       | 0.1038 | 0.2354 | 10 | 11.5 | 0.75 |
| random_2b_seed2       | 0.1609 | 0.2780 | 11 |  8.5 | 0.45 |
| fixed_8b_matched      | 0.0000 | 0.0764 |  0 | 10.0 | n/a (stub) |

## Honest interpretation

**Why adaptive loses at matched cost.** The deterministic verifier fires on
exactly one failure code: `NO_TABLE_FOUND` (11/20 pages). Adaptive correctly
escalates those 11 pages and rescues 5 of them (parse errors drop 11→6, a
clean structural win). But the other 9 pages — which the verifier passes at
4 tiles — still gain quality when given 10 tiles. Adaptive never sees that
signal, so its reparse budget is spent on the *structurally* broken subset
and misses the *quality-lift* subset. The fixed-matched baseline runs all 20
pages at 10 tiles and captures both wins.

**Random control.** Mean random cell-F1 ≈ 0.104 @ ~9.0 tiles vs. adaptive
0.127 @ 9.5. Adaptive's ~0.023 lift over the random mean is well within the
3-seed spread (0.046 → 0.161). With n=20, we **cannot claim** the
verifier-driven escalation beats random allocation on cell-F1.

**Random control on parse errors.** Random produces 10–11 parse errors per
seed; adaptive produces 6. The verifier signal **does** outperform random
allocation on structural validity — just not on overall accuracy.

**The 8B row is stub output** (mean_output_tokens=9, runtime ≈ 4 µs). It is
a placeholder, not a real upper bound. The audit emits
`accuracy_status: not_applicable_stub_adapters` accordingly.

## Methodological caveats (load-bearing)

1. **In-sample evaluation.** The 20-page calibration split *is* the evaluation
   split. There is no real-data held-out split. All matched-cost numbers are
   in-sample for Phase 5's budget pick. The Phase 7 audit downgrades the
   disjoint-splits check to a warn under
   `splits_are_identical_acknowledged: true`; this is the single biggest
   weakness of the MVP.
2. **n=20 is tiny.** Seed variance dwarfs the adaptive/fixed delta on the
   random control; small-sample noise dominates any conclusion.
3. **8B is stub.** No real 8B adapter is wired. The 8B matched-cost row is not
   a real upper bound.
4. **English-only, full-page parsing only, one-shot reparse only,
   deterministic verifier only.** Hard scope rails kept the MVP narrow.
5. **Greedy decoding.** All numbers are deterministic given inputs;
   re-running with the same inputs produces byte-identical outputs.

## What the next research step would be

1. **Build a real held-out split** from OmniDocBench (e.g., 100+ English-table
   pages disjoint from the calibration split). All current numbers should be
   re-evaluated on it before any paper claim.
2. **Larger n.** 20 pages cannot resolve the adaptive vs. random delta on
   accuracy. 200–500 pages is a reasonable target.
3. **Richer verifier.** The current verifier only catches structural failures
   (`NO_TABLE_FOUND` is the only code observed in 20 pages). A content-quality
   signal — even a simple heuristic on cell count, row consistency, or token
   entropy in the first-pass output — would let adaptive escalate the
   verifier-passing-but-quality-poor pages. The Phase 7 results show this is where matched-cost lift would have to come from.
4. **Real 8B adapter.** Required for any genuine upper-bound claim.
5. **Bigger `B_high`.** Frozen budget was 10 tiles. A higher ceiling on
   reparse (matched by cost budget) is worth exploring.

## What "done" means here

The research MVP is complete:
- Repo runs end-to-end on real data.
- All 284 tests passing.
- Real Phase 2, 4, 5, 6, 7 all executed on real OmniDocBench data.
- Phase 7 audit passes with explicit, surfaced caveats.
- The honest matched-cost finding is documented above.

The paper write-up itself is downstream of this engagement and out of scope.
The headline finding to defend: **the deterministic structural verifier is
not sufficient to make matched-cost adaptive beat a fixed baseline of equal
cost.** The verifier rescues structural failures but misses quality lifts,
and the fixed baseline captures both. A richer verifier is the natural next
research direction.
