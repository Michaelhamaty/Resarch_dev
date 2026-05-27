# Scale-Up v2 — Split Sizing Decisions

**Status:** Locked 2026-05-27 (Phase A4).
**Audience:** Reviewer / professor reproducing the scale-up v2 results.
**Companion to:** [`docs/specs/scaleup_v2_plan.md`](../specs/scaleup_v2_plan.md), [`docs/runbooks/fintabnet_provenance.md`](fintabnet_provenance.md).

---

## Final per-dataset n

| Dataset | Population (filtered) | n_total | n_calibration | n_held_out |
|---|---:|---:|---:|---:|
| FinTabNet (`apoidea/fintabnet-html`, en/validation) | 250 (capped from ~7,200) | 200 | 50 | 150 |
| OmniDocBench (`opendatalab/OmniDocBench`) | 146 | 140 | 50 | 90 |

**Asymmetric.** Calibration size matches on both sides (n=50) so that
matched-cost budget freezing in Stage 6 stays comparable; the held-out
side diverges.

---

## Why asymmetric

The plan calls for symmetric `n=200 / 50-calib / 150-held-out` per
dataset. After Phase A3, the realized populations are:

- **FinTabNet:** the apoidea/fintabnet-html `en/validation` shards
  contain ~7,200 candidate rows. We cap at 250 candidates via the
  fixture builder; the plan's n=200 fits comfortably.
- **OmniDocBench:** the full English-table page population, after the
  `min_non_empty_cells=4` quality filter, is exactly **146** pages. No
  `--limit` increase can produce more — the dataset's English-table
  subset is bounded.

Two viable responses were considered:

1. **Drop the quality filter** to chase n=200 on OmniDocBench. Rejected
   because the filter exists to exclude `<table>`-as-layout-primitive
   pages (slides, worksheets) that are not tabular-data benchmarks.
   Admitting them would dilute the OmniDocBench numbers with
   non-tabular content and break filter symmetry with FinTabNet.
2. **Scope OmniDocBench held-out down to fit the filtered
   population.** This preserves the science-quality bar, costs only the
   plan's symmetric-n property, and is still 4.5× the MVP's n=20.

Option 2 was selected. The 50/90 split was chosen over 50/96 (the
maximum at the 146 ceiling) for round-number cleanliness; the 6-page
delta is statistically inconsequential at this scale.

---

## Statistical implications

- **Bootstrap CIs on OmniDocBench held-out:** n=90 yields ~95% CI half-
  widths of roughly `1.96 × σ / sqrt(90) ≈ 0.21 σ`, vs. `0.16 σ` at
  n=150. Power is reduced by ~28% on absolute-effect detection, but the
  paper's headline contrasts (adaptive vs fixed_matched, adaptive vs
  random) only need `p < 0.05` paired-Wilcoxon, which n=90 supports
  fine for moderate effects.
- **Stratification:** the OmniDocBench held-out still admits stratified
  reporting by row-count bucket; the buckets just hold fewer pages.
- **Calibration parity:** both datasets use n=50 calibration pages, so
  matched-cost budget freezing in Stage 6 sees equal-power signal on
  each side.

---

## Plan-compliance note

The scale-up v2 plan ([`docs/specs/scaleup_v2_plan.md`](../specs/scaleup_v2_plan.md))
specifies n=200 per dataset. This document records the single
deviation: OmniDocBench at n=140 due to dataset-bounded population.
All other plan constraints (no PubTabNet, no third dataset, no
multilingual, no learned routing, etc.) remain in force.

The deviation will be acknowledged in the paper's Limitations section
alongside the existing "two datasets, not population-wide" caveat.

---

## How the splits were built

```bash
# FinTabNet (default n_total=200)
uv run python scripts/scaleup/build_scaleup_splits.py \
    --dataset fintabnet \
    --records-path data/fintabnet/records.json

# OmniDocBench (asymmetric n_total=140)
uv run python scripts/scaleup/build_scaleup_splits.py \
    --dataset omnidocbench \
    --records-path data/omnidocbench/records.json \
    --n-total 140 \
    --n-calibration 50
```

Seed phrase: `SCALEUPv2` (default). Determinism: re-running the same
commands against unchanged `records.json` files produces byte-identical
manifests.
