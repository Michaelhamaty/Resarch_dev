"""Scale-up v2 Stage 2: real InternVL2-8B smoke test on a single page.

Plan reference: `docs/specs/scaleup_v2_plan.md` Stage 2. This script is
the half-day-capped gate (G2) that confirms 8B loads and produces sane
output on the L4 24 GB VM before any further GPU spend.

Four checks (per plan):

1. **VRAM**: ``torch.cuda.max_memory_allocated()`` after generation is
   ≤ ``--vram-budget-gb`` (default 22 GB; leaves 2 GB headroom on L4).
2. **Tokenizer parity**: 8B and 2B tokenizers map ``<image>`` to the
   same token id (same family; no preprocessing-time shape mismatch).
3. **Determinism**: two consecutive ``adapter.run()`` calls with the
   same inputs and ``do_sample=False`` produce byte-identical
   ``raw_text``.
4. **Sanity**: the output ``raw_text`` contains at least one
   ``<table>`` tag (it should — the input page has a real table).

Default inputs:
    --records-path   data/omnidocbench/records.json
    --split-path     data/splits/scaleup_v2/omnidocbench/calibration.json
    --model-name     internvl2-8b
    --models-config  configs/models/internvl2_real.yaml
    --prompt-path    configs/prompts/table_parse_v1.yaml
    --max-tiles      10
    --vram-budget-gb 22.0

Exit code 0 iff all four checks pass. The script writes a structured
JSONL line to ``outputs/scaleup_v2/smoke/smoke_8b_one_page.jsonl`` so
the result is persisted across SSH disconnects and reproducible runs
diff cleanly.

Run on the VM inside tmux::

    tmux new -s smoke
    uv run python scripts/scaleup/smoke_8b_one_page.py
    # Ctrl+B d to detach
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from PIL import Image

from adaptive_inference.config.budgets import Budget
from adaptive_inference.config.models import load_model_configs
from adaptive_inference.config.prompts import load_prompt_template
from adaptive_inference.dataset.records import PageRecord, load_page_records


_TABLE_TAG_RE = re.compile(r"<table\b", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Pure-logic helpers (unit-tested without GPU)                                #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SmokeChecks:
    """Outcome of the four Stage-2 gates."""

    vram_within_budget: bool
    peak_vram_gb: float
    vram_budget_gb: float
    tokenizers_match: bool
    image_token_id_2b: int | None
    image_token_id_8b: int | None
    deterministic_across_runs: bool
    output_contains_table: bool

    def all_passed(self) -> bool:
        return (
            self.vram_within_budget
            and self.tokenizers_match
            and self.deterministic_across_runs
            and self.output_contains_table
        )

    def failures(self) -> list[str]:
        out: list[str] = []
        if not self.vram_within_budget:
            out.append(
                f"VRAM: peak={self.peak_vram_gb:.2f} GB > budget={self.vram_budget_gb:.2f} GB"
            )
        if not self.tokenizers_match:
            out.append(
                f"tokenizer parity: 2B image_token_id={self.image_token_id_2b!r} "
                f"!= 8B image_token_id={self.image_token_id_8b!r}"
            )
        if not self.deterministic_across_runs:
            out.append("determinism: two consecutive runs produced different raw_text")
        if not self.output_contains_table:
            out.append("sanity: raw_text contained no <table> tag")
        return out


def pick_smoke_page(
    records: Iterable[PageRecord],
    *,
    allowed_page_ids: Iterable[str] | None = None,
    explicit_page_id: str | None = None,
) -> PageRecord:
    """Deterministically choose one ``PageRecord`` for the smoke test.

    Selection rules:
      * If ``explicit_page_id`` is given, return that record or raise.
      * Else if ``allowed_page_ids`` is non-empty, restrict to that set.
      * Sort the remaining candidates by ``page_id`` and return the first
        one whose ``contains_table`` is True (so the ``<table>`` sanity
        check has something to find).
      * Raise ``ValueError`` if no candidate qualifies.
    """

    by_id = {r.page_id: r for r in records}
    if explicit_page_id is not None:
        if explicit_page_id not in by_id:
            raise ValueError(
                f"page_id {explicit_page_id!r} not in records "
                f"({len(by_id)} loaded)."
            )
        return by_id[explicit_page_id]

    if allowed_page_ids is not None:
        allowed = set(allowed_page_ids)
        candidates = [r for r in by_id.values() if r.page_id in allowed]
    else:
        candidates = list(by_id.values())

    candidates.sort(key=lambda r: r.page_id)
    for rec in candidates:
        if rec.contains_table:
            return rec
    raise ValueError(
        "No candidate page in records has contains_table=True; cannot "
        "run the <table> sanity check."
    )


def output_contains_table(raw_text: str) -> bool:
    """True if ``raw_text`` has at least one ``<table>`` opener."""

    return bool(_TABLE_TAG_RE.search(raw_text))


def read_calibration_page_ids(split_path: Path) -> list[str]:
    """Read the ``page_ids`` list from a scale-up v2 split manifest."""

    payload = json.loads(split_path.read_text(encoding="utf-8"))
    page_ids = payload.get("page_ids")
    if not isinstance(page_ids, list):
        raise ValueError(f"Split manifest at {split_path} missing page_ids list")
    return [str(p) for p in page_ids]


def format_summary(checks: SmokeChecks) -> str:
    """One-screen text summary of all four checks for human eyeballs."""

    lines = []
    lines.append("=" * 60)
    lines.append("Stage 2 (G2) — InternVL2-8B smoke checks")
    lines.append("=" * 60)
    lines.append(_check_line(
        "1. VRAM ≤ budget",
        checks.vram_within_budget,
        f"peak={checks.peak_vram_gb:.2f} GB, budget={checks.vram_budget_gb:.2f} GB",
    ))
    lines.append(_check_line(
        "2. Tokenizer parity (8B == 2B family)",
        checks.tokenizers_match,
        f"<image> token id: 2B={checks.image_token_id_2b!r} 8B={checks.image_token_id_8b!r}",
    ))
    lines.append(_check_line(
        "3. Deterministic greedy decoding",
        checks.deterministic_across_runs,
        "two runs byte-identical" if checks.deterministic_across_runs else "MISMATCH",
    ))
    lines.append(_check_line(
        "4. Output contains <table>",
        checks.output_contains_table,
        "<table> present" if checks.output_contains_table else "no <table> tag found",
    ))
    lines.append("-" * 60)
    if checks.all_passed():
        lines.append("RESULT: ALL PASS — gate G2 satisfied. Proceed to Stage 6.")
    else:
        lines.append("RESULT: FAIL — invoke plan Stage 2 fallback (stub 8B, drop")
        lines.append("        gap-closure framing; demote 8B to Limitations).")
        lines.append("Failures:")
        for f in checks.failures():
            lines.append(f"  - {f}")
    return "\n".join(lines)


def _check_line(label: str, passed: bool, detail: str) -> str:
    mark = "PASS" if passed else "FAIL"
    return f"  [{mark}] {label:42s} | {detail}"


# --------------------------------------------------------------------------- #
# Main (GPU section)                                                          #
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records-path", type=Path,
        default=Path("data/omnidocbench/records.json"),
    )
    parser.add_argument(
        "--split-path", type=Path,
        default=Path("data/splits/scaleup_v2/omnidocbench/calibration.json"),
        help=(
            "Restrict page selection to this split's page_ids. Set to "
            "an empty string to skip restriction and pick from all "
            "records."
        ),
    )
    parser.add_argument(
        "--page-id", default=None,
        help="Override the auto-selected page id.",
    )
    parser.add_argument(
        "--model-name", default="internvl2-8b",
        help="Model registry key whose adapter we smoke-test.",
    )
    parser.add_argument(
        "--reference-model-name", default="internvl2-2b",
        help="Second model whose tokenizer we compare against for check #2.",
    )
    parser.add_argument(
        "--models-config", type=Path,
        default=Path("configs/models/internvl2_real.yaml"),
    )
    parser.add_argument(
        "--prompt-path", type=Path,
        default=Path("configs/prompts/table_parse_v1.yaml"),
    )
    parser.add_argument("--max-tiles", type=int, default=10)
    parser.add_argument("--vram-budget-gb", type=float, default=22.0)
    parser.add_argument(
        "--out-jsonl", type=Path,
        default=Path("outputs/scaleup_v2/smoke/smoke_8b_one_page.jsonl"),
    )
    args = parser.parse_args(argv)

    # --- 1. Resolve which page to run -------------------------------------- #

    records = load_page_records(args.records_path)
    allowed_ids: list[str] | None = None
    if str(args.split_path):  # truthy → restrict
        try:
            allowed_ids = read_calibration_page_ids(args.split_path)
        except FileNotFoundError:
            print(
                f"WARN: split manifest {args.split_path} not found; "
                "falling back to all records.",
                file=sys.stderr,
            )
            allowed_ids = None
    page = pick_smoke_page(
        records, allowed_page_ids=allowed_ids, explicit_page_id=args.page_id
    )
    image_path = _resolve_image_path(args.records_path, page.image_path)
    print(f"[smoke] page_id={page.page_id} image={image_path}")
    image = Image.open(image_path)

    # --- 2. Build target adapter (8B) -------------------------------------- #

    prompt = load_prompt_template(args.prompt_path)
    models = load_model_configs(args.models_config)
    if args.model_name not in models:
        print(
            f"ERROR: model {args.model_name!r} not in {args.models_config} "
            f"(available={sorted(models)})",
            file=sys.stderr,
        )
        return 2
    target_cfg = models[args.model_name]
    if target_cfg.adapter_kind != "internvl2":
        print(
            f"ERROR: model {args.model_name!r} has adapter_kind="
            f"{target_cfg.adapter_kind!r}; this smoke requires the real "
            "internvl2 adapter (flip configs/models/internvl2_real.yaml).",
            file=sys.stderr,
        )
        return 2

    print(f"[smoke] loading {args.model_name} from {target_cfg.model_id} ...")
    t_load = time.perf_counter()
    from adaptive_inference.inference.factory import build_adapter
    target_adapter = build_adapter(target_cfg)
    load_seconds = time.perf_counter() - t_load
    print(f"[smoke] load took {load_seconds:.1f}s")

    # --- 3. Two consecutive runs for determinism + VRAM measurement -------- #

    import torch  # available because adapter loaded

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    budget = Budget(name=f"smoke_max_tiles_{args.max_tiles}", max_tiles=args.max_tiles)
    print(f"[smoke] run #1 budget={budget.name} ...")
    result1 = target_adapter.run(page.page_id, image, budget, prompt)
    print(f"[smoke] run #1: {result1.runtime_ms/1000.0:.1f}s, "
          f"out_tokens={result1.output_token_count}")

    print("[smoke] run #2 (determinism probe) ...")
    result2 = target_adapter.run(page.page_id, image, budget, prompt)
    print(f"[smoke] run #2: {result2.runtime_ms/1000.0:.1f}s, "
          f"out_tokens={result2.output_token_count}")

    peak_vram_gb = 0.0
    if torch.cuda.is_available():
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024.0 ** 3)
    print(f"[smoke] peak VRAM: {peak_vram_gb:.2f} GB")

    deterministic = result1.raw_text == result2.raw_text
    has_table = output_contains_table(result1.raw_text)

    # --- 4. Tokenizer parity check (load 2B tokenizer only) ---------------- #

    image_token_id_8b, image_token_id_2b = _tokenizer_image_token_ids(
        target_model_id=target_cfg.model_id,
        reference_model_name=args.reference_model_name,
        models=models,
    )
    tokenizers_match = (
        image_token_id_8b is not None
        and image_token_id_2b is not None
        and image_token_id_8b == image_token_id_2b
    )

    checks = SmokeChecks(
        vram_within_budget=peak_vram_gb <= args.vram_budget_gb,
        peak_vram_gb=peak_vram_gb,
        vram_budget_gb=args.vram_budget_gb,
        tokenizers_match=tokenizers_match,
        image_token_id_2b=image_token_id_2b,
        image_token_id_8b=image_token_id_8b,
        deterministic_across_runs=deterministic,
        output_contains_table=has_table,
    )

    # --- 5. Persist + print -------------------------------------------------- #

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "page_id": page.page_id,
        "model_name": args.model_name,
        "model_id": target_cfg.model_id,
        "max_tiles": args.max_tiles,
        "load_seconds": load_seconds,
        "result1": {
            "raw_text_len": len(result1.raw_text),
            "output_token_count": result1.output_token_count,
            "runtime_ms": result1.runtime_ms,
        },
        "result2": {
            "raw_text_len": len(result2.raw_text),
            "output_token_count": result2.output_token_count,
            "runtime_ms": result2.runtime_ms,
        },
        "checks": asdict(checks),
        "raw_text_head": result1.raw_text[:500],
    }
    with args.out_jsonl.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
        fh.flush()
    print(f"[smoke] appended record to {args.out_jsonl}")

    print()
    print(format_summary(checks))
    return 0 if checks.all_passed() else 1


# --------------------------------------------------------------------------- #
# I/O helpers                                                                  #
# --------------------------------------------------------------------------- #


def _resolve_image_path(records_path: Path, image_path_field: str) -> Path:
    """Records image_path is stored relative to the data dir parent."""

    candidate = Path(image_path_field)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    # The records.json typically lives at data/<dataset>/records.json
    # and image_path is "<dataset>/images/<page_id>.png".
    repo_root = records_path.resolve().parent.parent.parent
    rooted = repo_root / image_path_field
    if rooted.exists():
        return rooted
    # Fallback: same dir as records.json
    sibling = records_path.parent / candidate.name
    return sibling


def _tokenizer_image_token_ids(
    *,
    target_model_id: str,
    reference_model_name: str,
    models: dict,
) -> tuple[int | None, int | None]:
    """Return ``(reference_token_id, target_token_id)`` for ``<image>``.

    Loads two tokenizers (CPU-only, ~100MB system RAM total). Returns
    ``None`` for either side if loading fails so the check fails loudly
    rather than crashing the whole smoke.
    """

    from transformers import AutoTokenizer  # lazy

    def _safe_load_image_id(model_id: str) -> int | None:
        try:
            tok = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=True, use_fast=False
            )
            ids = tok.encode("<image>", add_special_tokens=False)
            # InternVL2 maps the literal "<image>" placeholder to a
            # single token id. If it doesn't, the family has changed.
            return int(ids[0]) if len(ids) == 1 else None
        except Exception as exc:
            print(f"[smoke] tokenizer load failed for {model_id}: {exc}",
                  file=sys.stderr)
            return None

    if reference_model_name not in models:
        print(
            f"[smoke] reference model {reference_model_name!r} not in registry; "
            "skipping tokenizer parity check.",
            file=sys.stderr,
        )
        return _safe_load_image_id(target_model_id), None

    ref_id = models[reference_model_name].model_id
    print(f"[smoke] checking tokenizer parity: {ref_id} vs {target_model_id}")
    return _safe_load_image_id(target_model_id), _safe_load_image_id(ref_id)


if __name__ == "__main__":
    raise SystemExit(main())
