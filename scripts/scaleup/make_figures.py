"""Scale-up v2 Stage 9 paper figures.

Plan reference: docs/specs/scaleup_v2_plan.md Stage 9. Consumes the
aggregator output (``outputs/scaleup_v2/analysis/results_v2.json`` and
``diagnostic_<dataset>.jsonl``) and renders the three figures the paper
needs, as both PDF (LaTeX inclusion) and PNG (quick viewing):

  1. cost_accuracy.pdf   — cost-vs-cell-F1 scatter, one panel per
                           dataset, bootstrap-CI error bars.
  2. diagnostic_scatter.pdf — per page, first-pass cell-F1 (x) vs final
                           cell-F1 (y) for the adaptive system, colored
                           by whether the verifier triggered a reparse.
                           This is the figure that visually motivates the
                           negative result. Skipped if no diagnostic
                           JSONL is present.
  3. stratified_bars.pdf — cell-F1 by difficulty bucket, grouped by
                           system, with CI whiskers.

Pure consumer of JSON; no model, no GPU, no sweep re-run. Figures land in
``paper/figures/`` by default so the .tex \\includegraphics paths resolve.

Usage (laptop, after run_phase7_v2.py)::

    uv run python scripts/scaleup/make_figures.py \\
        --results outputs/scaleup_v2/analysis/results_v2.json \\
        --out-dir paper/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless; never needs a display
import matplotlib.pyplot as plt  # noqa: E402


# Deterministic system ordering + colors so the paper figures are stable.
SYSTEM_ORDER = (
    "fixed_2b_low",
    "fixed_2b_matched",
    "fixed_8b_matched",
    "adaptive_2b",
    "random_2b_seed0",
    "random_2b_seed1",
    "random_2b_seed2",
)
BUCKET_ORDER = ("simple", "complex", "very_complex")


def _ordered_systems(systems: dict) -> list[str]:
    known = [s for s in SYSTEM_ORDER if s in systems]
    extra = sorted(s for s in systems if s not in SYSTEM_ORDER)
    return known + extra


def _save(fig, out_dir: Path, stem: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=200)
        paths.append(p)
    plt.close(fig)
    return paths


# --------------------------------------------------------------------------- #
# Figure 1 — cost vs accuracy                                                 #
# --------------------------------------------------------------------------- #


def make_cost_accuracy(results: dict, out_dir: Path) -> list[Path]:
    datasets = {
        name: d for name, d in results["datasets"].items() if d.get("status") == "ok"
    }
    if not datasets:
        print("[figures] cost_accuracy: no ok datasets; skipping")
        return []

    fig, axes = plt.subplots(
        1, len(datasets), figsize=(6 * len(datasets), 4.5), squeeze=False
    )
    for ax, (name, dres) in zip(axes[0], datasets.items()):
        for sid in _ordered_systems(dres["systems"]):
            blk = dres["systems"][sid]
            if "cell_f1" not in blk or blk.get("cost_tiles") is None:
                continue
            cf = blk["cell_f1"]
            x = blk["cost_tiles"]
            y = cf["mean"]
            yerr = [[y - cf["lo"]], [cf["hi"] - y]]
            ax.errorbar(
                x, y, yerr=yerr, fmt="o", capsize=3, markersize=7, label=sid
            )
        ax.set_title(name)
        ax.set_xlabel("cost (tiles / page)")
        ax.set_ylabel("cell-F1")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle("Cost vs. accuracy (95% bootstrap CI)")
    return _save(fig, out_dir, "cost_accuracy")


# --------------------------------------------------------------------------- #
# Figure 2 — diagnostic scatter (first-pass vs final cell-F1)                 #
# --------------------------------------------------------------------------- #


def make_diagnostic_scatter(
    diagnostics: dict[str, list[dict]], out_dir: Path
) -> list[Path]:
    datasets = {k: v for k, v in diagnostics.items() if v}
    if not datasets:
        print("[figures] diagnostic_scatter: no diagnostic rows; skipping")
        return []

    fig, axes = plt.subplots(
        1, len(datasets), figsize=(5.5 * len(datasets), 5), squeeze=False
    )
    for ax, (name, rows) in zip(axes[0], datasets.items()):
        for reparsed, color, lbl in (
            (False, "tab:blue", "verifier PASS"),
            (True, "tab:red", "verifier REPARSE"),
        ):
            xs = [r["first_pass_cell_f1"] for r in rows if bool(r["reparsed"]) is reparsed]
            ys = [r["final_cell_f1"] for r in rows if bool(r["reparsed"]) is reparsed]
            if xs:
                ax.scatter(xs, ys, c=color, alpha=0.6, s=28, label=lbl)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)  # y=x reference
        ax.set_title(name)
        ax.set_xlabel("first-pass cell-F1")
        ax.set_ylabel("final cell-F1")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("Adaptive first-pass vs. final cell-F1 (points above y=x gained from reparse)")
    return _save(fig, out_dir, "diagnostic_scatter")


# --------------------------------------------------------------------------- #
# Figure 3 — stratified bars                                                  #
# --------------------------------------------------------------------------- #


def make_stratified_bars(results: dict, out_dir: Path) -> list[Path]:
    datasets = {
        name: d for name, d in results["datasets"].items() if d.get("status") == "ok"
    }
    if not datasets:
        print("[figures] stratified_bars: no ok datasets; skipping")
        return []

    fig, axes = plt.subplots(
        1, len(datasets), figsize=(6.5 * len(datasets), 4.5), squeeze=False
    )
    for ax, (name, dres) in zip(axes[0], datasets.items()):
        systems = _ordered_systems(dres["systems"])
        systems = [s for s in systems if "cell_f1_by_bucket" in dres["systems"][s]]
        n_sys = len(systems)
        if n_sys == 0:
            ax.set_title(f"{name} (no bucketed data)")
            continue
        bar_w = 0.8 / n_sys
        for i, sid in enumerate(systems):
            buckets = dres["systems"][sid]["cell_f1_by_bucket"]
            xs, ys, lo, hi = [], [], [], []
            for b_idx, bucket in enumerate(BUCKET_ORDER):
                if bucket not in buckets:
                    continue
                cf = buckets[bucket]
                xs.append(b_idx + i * bar_w)
                ys.append(cf["mean"])
                lo.append(cf["mean"] - cf["lo"])
                hi.append(cf["hi"] - cf["mean"])
            if xs:
                ax.bar(xs, ys, width=bar_w, label=sid, yerr=[lo, hi], capsize=2)
        ax.set_title(name)
        ax.set_xticks([i + 0.4 - bar_w / 2 for i in range(len(BUCKET_ORDER))])
        ax.set_xticklabels(BUCKET_ORDER)
        ax.set_ylabel("cell-F1")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle("Cell-F1 by difficulty bucket (95% bootstrap CI)")
    return _save(fig, out_dir, "stratified_bars")


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def load_diagnostics(results_path: Path, dataset_names: list[str]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for name in dataset_names:
        p = results_path.parent / f"diagnostic_{name}.jsonl"
        if not p.exists():
            continue
        rows = [
            json.loads(line)
            for line in p.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        out[name] = rows
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path,
        default=Path("outputs/scaleup_v2/analysis/results_v2.json"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("paper/figures"))
    args = parser.parse_args(argv)

    results = json.loads(args.results.read_text(encoding="utf-8"))
    diagnostics = load_diagnostics(args.results, list(results["datasets"]))

    written: list[Path] = []
    written += make_cost_accuracy(results, args.out_dir)
    written += make_diagnostic_scatter(diagnostics, args.out_dir)
    written += make_stratified_bars(results, args.out_dir)

    print(f"[figures] wrote {len(written)} files to {args.out_dir}:")
    for p in written:
        print(f"  - {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
