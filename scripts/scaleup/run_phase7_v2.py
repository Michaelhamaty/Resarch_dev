"""Scale-up v2 Stage 9 results aggregation.

Plan reference: docs/specs/scaleup_v2_plan.md Stage 9. Walks every
``outputs/scaleup_v2/<dataset>/<system_id>/run.log.jsonl`` produced by
``scripts/scaleup/run_sweep.py``, scores each system against the
dataset's ground truth (cell-F1 via the canonical scorer, plus TEDS),
and emits the paper's machine-readable + paste-ready results:

    outputs/scaleup_v2/analysis/results_v2.json   (every number)
    outputs/scaleup_v2/analysis/results_v2.md     (paper-paste tables)
    outputs/scaleup_v2/analysis/diagnostic_<dataset>.jsonl
        (per-page first-pass vs final cell-F1 for the adaptive system,
         consumed by make_figures.py's diagnostic scatter)

Read-only with respect to every sweep artifact and the frozen budgets.
No model is loaded. Reuses, rather than re-implements:
  * cell-F1 scoring         -> analysis.run_scoring.score_run
  * confidence intervals    -> analysis.bootstrap.bootstrap_ci
  * paired significance     -> analysis.paired_tests.paired_wilcoxon
  * difficulty buckets      -> analysis.stratify.stratify_by_difficulty
  * TEDS                    -> analysis.teds.teds_score
  * config parsing          -> scripts.scaleup.run_sweep.load_sweep_config

Usage on the laptop (CPU only, after the sweep finishes + scp back)::

    uv run python scripts/scaleup/run_phase7_v2.py \\
        --config configs/experiment/scaleup_v2.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

# Make the sibling run_sweep importable when invoked as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_sweep import DatasetSpec, SweepConfig, load_sweep_config  # noqa: E402

from adaptive_inference.analysis.bootstrap import BootstrapResult, bootstrap_ci
from adaptive_inference.analysis.loaders import (
    load_loaded_systems,
    load_phase6_manifest,
    load_split_page_ids,
)
from adaptive_inference.analysis.paired_tests import align_pages, paired_wilcoxon
from adaptive_inference.analysis.run_scoring import score_run
from adaptive_inference.analysis.results import summarize_system
from adaptive_inference.analysis.stratify import stratify_by_difficulty
from adaptive_inference.analysis.teds import teds_score
from adaptive_inference.dataset.records import load_page_records


SCHEMA_VERSION = 1
BUCKETS = ("simple", "complex", "very_complex")


# --------------------------------------------------------------------------- #
# Pure-logic helpers (unit-tested without disk / GPU)                         #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Comparison:
    """One paired test the paper reports."""

    a: str  # system_id whose advantage we are testing
    b: str  # baseline system_id (or pseudo-system "random_2b_pooled")
    label: str


def derive_comparisons(system_ids: list[str]) -> list[Comparison]:
    """Return the paired comparisons to run, given the systems present.

    The paper's two headline contrasts:
      * adaptive_2b vs fixed_2b_matched   (does adaptive beat matched cost?)
      * adaptive_2b vs random_2b_pooled   (does the verifier beat chance?)

    A comparison is emitted only when both sides are present, so a
    partially-complete sweep still aggregates without raising.
    """

    present = set(system_ids)
    out: list[Comparison] = []
    if "adaptive_2b" in present and "fixed_2b_matched" in present:
        out.append(
            Comparison("adaptive_2b", "fixed_2b_matched", "adaptive vs fixed-matched")
        )
    random_seeds = sorted(s for s in present if s.startswith("random_2b_seed"))
    if "adaptive_2b" in present and random_seeds:
        out.append(
            Comparison("adaptive_2b", "random_2b_pooled", "adaptive vs random (pooled seeds)")
        )
    return out


def pool_random_seeds(
    per_system_cell_f1: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Average per-page cell-F1 across all ``random_2b_seed*`` systems.

    Returns ``{page_id: mean_cell_f1}`` over the seeds that scored each
    page. Pages no seed scored are absent. Empty if there are no random
    systems.
    """

    seed_maps = [
        m for sid, m in per_system_cell_f1.items() if sid.startswith("random_2b_seed")
    ]
    if not seed_maps:
        return {}
    all_pages: set[str] = set()
    for m in seed_maps:
        all_pages.update(m)
    pooled: dict[str, float] = {}
    for pid in all_pages:
        vals = [m[pid] for m in seed_maps if pid in m]
        if vals:
            pooled[pid] = sum(vals) / len(vals)
    return pooled


def bucketed_cis(
    per_page: dict[str, float],
    metadata: dict[str, dict],
    *,
    seed: int = 0,
) -> dict[str, dict]:
    """Bootstrap CI per difficulty bucket. Empty buckets are omitted."""

    by_bucket = stratify_by_difficulty(per_page, metadata)
    out: dict[str, dict] = {}
    for bucket, pairs in by_bucket.items():
        if not pairs:
            continue
        values = [v for _pid, v in pairs]
        out[bucket] = _ci_to_dict(bootstrap_ci(values, seed=seed))
    return out


def _ci_to_dict(r: BootstrapResult) -> dict:
    return {
        "mean": round(r.mean, 6),
        "lo": round(r.lo, 6),
        "hi": round(r.hi, 6),
        "n": r.n,
        "ci": r.ci,
    }


# --------------------------------------------------------------------------- #
# I/O orchestration                                                           #
# --------------------------------------------------------------------------- #


def ground_truth_path_for(dataset: DatasetSpec) -> Path:
    """GT lives beside records.json as ``ground_truth.json`` (flat map)."""

    return dataset.records_path.parent / "ground_truth.json"


def _resolve_pages_dir(run_dir: Path) -> Path | None:
    """Mirror run_scoring._resolve_pages_dir but return None if absent."""

    for cand in (run_dir / "pages", run_dir / "final" / "pages"):
        if cand.is_dir():
            return cand
    return None


def teds_per_page(run_dir: Path, ground_truth: dict[str, str]) -> dict[str, float]:
    """Per-page TEDS for a system run dir, keyed by page_id.

    Walks the same sidecars score_run uses. Pages without gold are
    skipped (no denominator). Read-only.
    """

    pages_dir = _resolve_pages_dir(run_dir)
    if pages_dir is None:
        return {}
    out: dict[str, float] = {}
    for sidecar_path in sorted(pages_dir.glob("*.json")):
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        page_id = str(sidecar["page_id"])
        gold = ground_truth.get(page_id)
        if gold is None:
            continue
        pred = (run_dir / sidecar["raw_output_path"]).read_text(encoding="utf-8")
        out[page_id] = teds_score(pred, gold)
    return out


def cell_f1_per_page(run_dir: Path, gt_path: Path) -> tuple[dict[str, float], dict]:
    """Canonical per-page cell-F1 via score_run, plus macro summary.

    Returns ``({page_id: cell_f1}, macro_summary_dict)``. Only pages with
    gold contribute to the map (gold-missing pages are excluded so they
    do not depress the bootstrap mean with structural zeros).
    """

    result = score_run(run_dir, gt_path)
    per_page = {
        p.page_id: p.cell_f1 for p in result.page_scores if not p.gold_missing
    }
    macro = {
        "macro_cell_f1": round(result.macro_cell_f1, 6),
        "macro_text_similarity": round(result.macro_text_similarity, 6),
        "pages_total": result.pages_total,
        "pages_with_gold": result.pages_with_gold,
        "pages_with_parse_error": result.pages_with_parse_error,
    }
    return per_page, macro


def aggregate_dataset(
    dataset: DatasetSpec,
    sweep_cfg: SweepConfig,
    *,
    repo_root: Path,
) -> tuple[dict, list[dict]]:
    """Aggregate one dataset's sweep into (results_dict, diagnostic_rows)."""

    out_root = sweep_cfg.output_root / dataset.name
    manifest_path = out_root / "manifest.json"
    if not manifest_path.exists():
        return (
            {"status": "no_manifest", "manifest_path": str(manifest_path)},
            [],
        )

    manifest = load_phase6_manifest(manifest_path)
    systems = load_loaded_systems(manifest, repo_root=repo_root)
    held_out_ids = load_split_page_ids(dataset.split_manifest_path)
    gt_path = ground_truth_path_for(dataset)
    ground_truth = json.loads(gt_path.read_text(encoding="utf-8"))
    records = load_page_records(dataset.records_path)
    metadata = {
        r.page_id: {
            "row_count": getattr(r, "row_count", 0),
            "has_merged_cells": getattr(r, "has_merged_cells", False),
        }
        for r in records
    }

    per_system_cell_f1: dict[str, dict[str, float]] = {}
    system_blocks: dict[str, dict] = {}

    for s in systems:
        entry = s.entry
        if entry.status != "ok" or entry.output_dir is None:
            system_blocks[entry.system_id] = {
                "status": entry.status,
                "runner": entry.runner,
            }
            continue
        run_dir = repo_root / entry.output_dir
        cell_map, macro = cell_f1_per_page(run_dir, gt_path)
        teds_map = teds_per_page(run_dir, ground_truth)
        per_system_cell_f1[entry.system_id] = cell_map

        # Canonical measured cost / reparse rate from the same summarizer
        # Phase 5/7 used — never invent a parallel cost accounting.
        summary = summarize_system(s)

        block: dict = {
            "status": entry.status,
            "runner": entry.runner,
            "family": entry.family,
            "n_scored": len(cell_map),
            "macro": macro,
            "cost_tiles": round(summary.cost_tiles, 4),
            "reparse_rate": summary.reparse_rate,
            "reparse_count": entry.reparse_count,
            "seed": entry.seed,
        }
        if cell_map:
            block["cell_f1"] = _ci_to_dict(bootstrap_ci(list(cell_map.values())))
            block["cell_f1_by_bucket"] = bucketed_cis(cell_map, metadata)
        if teds_map:
            block["teds"] = _ci_to_dict(bootstrap_ci(list(teds_map.values())))
        system_blocks[entry.system_id] = block

    # Paired comparisons.
    pooled_random = pool_random_seeds(per_system_cell_f1)
    comparisons: list[dict] = []
    for comp in derive_comparisons(list(per_system_cell_f1)):
        map_a = per_system_cell_f1.get(comp.a, {})
        map_b = pooled_random if comp.b == "random_2b_pooled" else per_system_cell_f1.get(comp.b, {})
        if not map_a or not map_b:
            continue
        page_ids, a_vals, b_vals = align_pages(map_a, map_b)
        if not page_ids:
            continue
        res = paired_wilcoxon(a_vals, b_vals)
        mean_a = sum(a_vals) / len(a_vals)
        mean_b = sum(b_vals) / len(b_vals)
        comparisons.append({
            "a": comp.a,
            "b": comp.b,
            "label": comp.label,
            "metric": "cell_f1",
            "n_pairs": res.n_pairs,
            "n_dropped_ties": res.n_dropped_ties,
            "statistic": round(res.statistic, 6),
            "p_value": round(res.p_value, 8),
            "mean_a": round(mean_a, 6),
            "mean_b": round(mean_b, 6),
            "delta_a_minus_b": round(mean_a - mean_b, 6),
        })

    # Diagnostic rows: adaptive first-pass vs final cell-F1 per page.
    diagnostic_rows = _adaptive_diagnostic_rows(
        systems, repo_root=repo_root, gt_path=gt_path, dataset=dataset.name
    )

    dataset_result = {
        "status": "ok",
        "n_held_out": len(held_out_ids),
        "ground_truth_path": str(gt_path),
        "systems": system_blocks,
        "comparisons": comparisons,
    }
    return dataset_result, diagnostic_rows


def _adaptive_diagnostic_rows(
    systems, *, repo_root: Path, gt_path: Path, dataset: str
) -> list[dict]:
    """Per-page first-pass vs final cell-F1 for adaptive_2b.

    Powers the plan's diagnostic scatter (Stage 9 figure 2). Returns []
    if the adaptive system or its first_pass/ + final/ dirs are absent.
    """

    adaptive = next(
        (s for s in systems if s.entry.system_id == "adaptive_2b"
         and s.entry.status == "ok" and s.entry.output_dir is not None),
        None,
    )
    if adaptive is None:
        return []
    run_dir = repo_root / adaptive.entry.output_dir
    first_dir = run_dir / "first_pass"
    final_dir = run_dir / "final"
    if not (first_dir / "pages").is_dir() or not (final_dir / "pages").is_dir():
        return []

    first = score_run(first_dir, gt_path)
    final = score_run(final_dir, gt_path)
    first_by_id = {p.page_id: p for p in first.page_scores}
    rows: list[dict] = []
    for fp in final.page_scores:
        if fp.gold_missing:
            continue
        fpre = first_by_id.get(fp.page_id)
        if fpre is None:
            continue
        reparsed = abs(fp.cell_f1 - fpre.cell_f1) > 1e-9 or fp.text_similarity != fpre.text_similarity
        rows.append({
            "dataset": dataset,
            "page_id": fp.page_id,
            "first_pass_cell_f1": round(fpre.cell_f1, 6),
            "final_cell_f1": round(fp.cell_f1, 6),
            "lift": round(fp.cell_f1 - fpre.cell_f1, 6),
            "reparsed": reparsed,
        })
    return rows


# --------------------------------------------------------------------------- #
# Markdown rendering                                                          #
# --------------------------------------------------------------------------- #


def render_markdown(results: dict) -> str:
    lines: list[str] = ["# Scale-up v2 results (`results_v2.md`)", ""]
    lines.append(f"_Generated: {results['generated_at']}  ·  git: {results.get('git_head')}_")
    lines.append("")
    for dataset, dres in results["datasets"].items():
        lines.append(f"## {dataset}")
        if dres.get("status") != "ok":
            lines.append(f"_status: {dres.get('status')}_\n")
            continue
        lines.append(f"_held-out n={dres['n_held_out']}_\n")
        lines.append("| system | n | cell-F1 [95% CI] | TEDS [95% CI] |")
        lines.append("|---|---:|---|---|")
        for sid, blk in dres["systems"].items():
            if "cell_f1" not in blk:
                lines.append(f"| {sid} | — | _{blk.get('status')}_ | — |")
                continue
            cf = blk["cell_f1"]
            cell = f"{cf['mean']:.4f} [{cf['lo']:.4f}, {cf['hi']:.4f}]"
            teds = (
                f"{blk['teds']['mean']:.4f} [{blk['teds']['lo']:.4f}, {blk['teds']['hi']:.4f}]"
                if "teds" in blk else "—"
            )
            lines.append(f"| {sid} | {blk['n_scored']} | {cell} | {teds} |")
        lines.append("")
        if dres["comparisons"]:
            lines.append("**Paired Wilcoxon (cell-F1):**")
            lines.append("")
            lines.append("| comparison | mean Δ (a−b) | p-value | n pairs |")
            lines.append("|---|---:|---:|---:|")
            for c in dres["comparisons"]:
                lines.append(
                    f"| {c['label']} | {c['delta_a_minus_b']:+.4f} | "
                    f"{c['p_value']:.4g} | {c['n_pairs']} |"
                )
            lines.append("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="configs/experiment/scaleup_v2.yaml")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Optional: aggregate only these datasets.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Default: <sweep output root>/analysis")
    args = parser.parse_args(argv)

    repo_root = Path.cwd().resolve()
    sweep_cfg = load_sweep_config(args.config)
    out_dir = args.output_dir or (sweep_cfg.output_root / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    from adaptive_inference.experiment.manifest import git_head_or_none, iso_now

    dataset_filter = set(args.datasets) if args.datasets else None
    datasets_out: dict[str, dict] = {}
    for dataset in sweep_cfg.datasets:
        if dataset_filter is not None and dataset.name not in dataset_filter:
            continue
        print(f"[phase7v2] aggregating {dataset.name} ...")
        dres, diag_rows = aggregate_dataset(dataset, sweep_cfg, repo_root=repo_root)
        datasets_out[dataset.name] = dres
        if diag_rows:
            diag_path = out_dir / f"diagnostic_{dataset.name}.jsonl"
            with diag_path.open("w", encoding="utf-8") as fh:
                for row in diag_rows:
                    fh.write(json.dumps(row, sort_keys=True) + "\n")
            print(f"[phase7v2]   wrote {diag_path} ({len(diag_rows)} pages)")

    results = {
        "schema_version": SCHEMA_VERSION,
        "run_set_id": sweep_cfg.run_set_id,
        "generated_at": iso_now(),
        "git_head": git_head_or_none(repo_root),
        "sweep_config_path": str(args.config),
        "datasets": datasets_out,
    }

    json_path = out_dir / "results_v2.json"
    json_path.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    md_path = out_dir / "results_v2.md"
    md_path.write_text(render_markdown(results), encoding="utf-8")

    print(f"[phase7v2] wrote {json_path}")
    print(f"[phase7v2] wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
