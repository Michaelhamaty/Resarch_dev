# Phase 3 runbook — deterministic structural verifier

Phase 3 adds the decision layer that turns the single-pass inference
scaffold into an adaptive pipeline. The verifier inspects raw page
output and returns `PASS` or `REPARSE` based on a fixed set of
structural checks against HTML `<table>` blocks. It does not call a
model, does not read from disk, does not maintain state across pages,
and does not implement the reparse itself — that orchestration lands in
Phase 4.

## What it does

`verify_page_tables(raw_text: str) -> VerifierResult` runs these checks,
in order, over every `<table>` block found on the page:

1. **Presence** — if no `<table>` block is found, return `REPARSE` with
   `NO_TABLE_FOUND`. The caller is responsible for only invoking the
   verifier on pages that are expected to contain tables.
2. **Parsability** — each extracted block is parsed via
   `lxml.html.fromstring` in recovery mode. If parsing fails, the table
   is tagged `HTML_PARSE_ERROR`.
3. **Span expansion** — `rowspan`/`colspan` attributes are normalized
   (defaulting to 1) and cells are placed left-to-right, top-to-bottom
   in a sparse grid. Non-positive or non-integer spans, or a span
   extent that collides with an already-occupied position, yield
   `SPAN_EXPANSION_FAILED`.
4. **Rectangular consistency** — after expansion, every row must have
   the same width and every position in that rectangle must be
   occupied. Otherwise `RECTANGULAR_INCONSISTENCY`.
5. **Degenerate check** — a table with zero rows, or whose cells are
   all empty/whitespace, yields `DEGENERATE_TABLE`. Per the locked
   spec, header-only and single-cell tables are **not** treated as
   degenerate in the MVP.

If any table on the page fails any check, the page-level decision is
`REPARSE`. Otherwise `PASS`.

## What it does not do

- No semantic correctness judgment. The verifier cannot tell "mostly
  right" from "mostly wrong" — it only sees shape.
- No OCR, no image access, no crop repair.
- No learned routing, no self-critique agents, no multi-step repair
  loops.
- No disk I/O or logging. Phase 4's runner composes it with the logger
  and the reparse orchestrator.

## Why precision-first

A false `REPARSE` costs extra compute. A false `PASS` misses an
improvement opportunity but preserves the low-budget average cost. For
the MVP we deliberately miss some semantically wrong but structurally
valid outputs — the research claim is about whether spending compute
on flagged pages helps, not about catching every parse error.

Consequences of this stance, visible in the test suite:

- 1×1 tables and header-only tables pass (they are structurally valid).
- All-empty tables and zero-row tables fail (they are structurally
  useless regardless of semantics).

## Failure codes

| Code | Meaning |
| --- | --- |
| `NO_TABLE_FOUND` | Page output contains zero `<table>` blocks. |
| `HTML_PARSE_ERROR` | A `<table>` block could not be parsed. |
| `SPAN_EXPANSION_FAILED` | Invalid rowspan/colspan or cell overlap. |
| `RECTANGULAR_INCONSISTENCY` | Expanded grid is ragged or has gaps. |
| `DEGENERATE_TABLE` | Zero rows, or every cell empty/whitespace. |

These strings are part of the persistent output contract and will be
written into run logs in Phase 4; do not rename them casually.

## Module map

```
src/adaptive_inference/
├── parsing/html_tables.py        # extract <table>…</table> substrings
└── verifier/
    ├── codes.py                  # stable failure-code constants
    ├── types.py                  # VerifierResult, TableSummary
    ├── spans.py                  # rowspan/colspan -> rectangular grid
    └── structural.py             # verify_page_tables entry point
```

## How to use it

```python
from pathlib import Path
from adaptive_inference.verifier.structural import verify_page_tables

raw = Path("outputs/runs/smoke_2b_low_v1/raw/fixture_page_0001.md").read_text()
result = verify_page_tables(raw)
print(result.decision, result.failure_codes)
```

## Known limitations

- Nested tables are **not** recursed into. Top-level extraction picks
  the outermost `<table>...</table>` substring; the regex is
  non-greedy, so a nested closing tag may end the match early. If the
  evaluation universe ever includes nested tables, this layer needs
  revisiting.
- The extractor is purely syntactic — if a model emits an unclosed
  `<table>` (no matching `</table>`), the block is not extracted and
  the page falls through to `NO_TABLE_FOUND`, which is the correct
  precision-first behavior.
