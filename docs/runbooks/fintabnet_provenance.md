# FinTabNet Provenance — Scale-Up v2

**Status:** Source locked 2026-05-27. Download not yet executed.
**Audience:** Reviewer / professor reproducing the scale-up v2 results.

---

## Canonical source

| Field | Value |
|---|---|
| HuggingFace repo | [`apoidea/fintabnet-html`](https://huggingface.co/datasets/apoidea/fintabnet-html) |
| Repo type | dataset |
| Subset used | English (FinTabNet is English by construction) |
| Splits pulled | `test` (FinTabNet's own held-out split) |
| Cap | 250 candidate pages, then stratified-sampled to 200 by `scripts/scaleup/build_scaleup_splits.py` |
| Original paper | Zheng, Tang, Han, Lakshmanan, Garcia-Olano, Mitra, Jandaghi, Riedel (2021). *Global Table Extractor (GTE): A Framework for Joint Table Identification and Cell Structure Recognition Using Visual Context.* WACV 2021. |

---

## License

**CDLA-Permissive 1.0** (Community Data License Agreement — Permissive, v1.0), inherited from the original IBM FinTabNet release.

The apoidea mirror ships an empty README. License inheritance argument: CDLA-Permissive 1.0 §2.4 forbids any recipient of CDLA-P 1.0 data from imposing additional restrictions on downstream recipients. A derivative HuggingFace repo that re-distributes FinTabNet content cannot therefore be more restrictive than the upstream license — so research use, redistribution with attribution, and derivative-work creation are all permitted.

**Citation requirement:** The paper's dataset section must cite both Zheng et al. 2021 and the CDLA-Permissive 1.0 license. The apoidea repo is the *mirror*, not the *source* — the citation is to the original work.

---

## Why this mirror (not `bsmock25/FinTabNet.c`)

The earlier default `bsmock25/FinTabNet.c` ships the TATR-cleaned variant used for table-detection benchmarks; its annotations are oriented toward bounding-box structure detection. Scale-up v2 needs the HTML-annotated form (PubTabNet-style structure tokens + cell content) that matches the existing loader at [`src/adaptive_inference/dataset/fintabnet.py`](../../src/adaptive_inference/dataset/fintabnet.py).

The apoidea mirror is the HTML-annotation form. Both inherit the same upstream license.

---

## Verified repo shape (2026-05-27 discovery)

The apoidea mirror ships **parquet shards** (not JSONL) with the layout:

```
en/train-00000-of-00018.parquet  …  en/train-00017-of-00018.parquet
en/validation-00000-of-00003.parquet  …  en/validation-00002-of-00003.parquet
sc/…  tc/…   (simplified/traditional Chinese — not used by scale-up v2)
```

**No `test` split exists.** Scale-up v2 uses `en/validation-*.parquet` (~7,200 rows total across 3 shards) as the held-out source.

Each parquet row has only two columns:

| Column | Type | Notes |
|---|---|---|
| `image` | struct `{bytes: binary, path: string}` | PNG payload embedded directly in the parquet row |
| `html_table` | string | Rendered, pretty-printed HTML table — **not** PubTabNet structure tokens |

This is a different shape from the PubTabNet-token form. Scale-up v2 therefore ingests via the **HTML-shape path** at `src/adaptive_inference/dataset/fintabnet_html.py` (`HtmlRow` + `select_html_pages`) and the `--parquet-glob` mode of the fixture builder. The legacy token-shape loader in `fintabnet.py` is untouched and remains available for other mirrors.

`page_id` is content-addressed: `fintabnet_<sha1(html_table)[:12]>` — stable across re-downloads and shard reorderings.

---

## Build command

```bash
uv run python scripts/data/build_fintabnet_fixture.py \
    --parquet-glob 'en/validation-*.parquet' \
    --splits validation \
    --limit 250 \
    --min-non-empty-cells 4 \
    --out-dir data/fintabnet
```

Writes:
- `data/fintabnet/images/<page_id>.png`
- `data/fintabnet/records.json`
- `data/fintabnet/ground_truth.json`
- `data/fintabnet/manifest.jsonl`

`data/fintabnet/` is gitignored — never committed.

---

## Why we download locally, not on the VM

Bandwidth is free here; VM bandwidth costs $0.80/hr of compute time we'd otherwise be using for GPU work. The download + fixture build is pure CPU + network. After it completes locally, we `rsync` `data/fintabnet/` up to the VM during Phase B.
