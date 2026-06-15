"""Generate the data figures for the CogMI 2026 manuscript.

Reads only committed analysis artifacts:
  * outputs/scaleup_v2/analysis/results_v2.json           (main sweep)
  * outputs/scaleup_v2_overmatch/analysis/results_v2.json (fixed_2b @ 11)
  * outputs/scaleup_v2/analysis/diagnostic_<ds>.jsonl     (per-page first/final)
  * outputs/scaleup_v2/analysis/reparse_flags_<ds>.json   (true trigger flags
        extracted from the per-system run.log.jsonl files)

Outputs (paper/figures/):
  * budget_response.pdf  — cell-F1 vs uniform tile budget per dataset, with
        the adaptive, random-seed, and 8B points overlaid. Sized for a
        full-width (figure*) IEEE placement.
  * adaptive_scatter.pdf — per-page first-pass vs final cell-F1, colored by
        whether the verifier actually triggered a reparse. Sized for a
        single-column placement (two stacked panels).

    uv run python paper/figures/make_paper_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "pdf.fonttype": 42,
})

C_FIXED = "#4878a8"
C_ADAPT = "#c44e52"
C_RAND = "#999999"
C_8B = "#55a868"

DS_LABEL = {"omnidocbench": "OmniDocBench (n=90)", "fintabnet": "FinTabNet (n=150)"}


def _load_results() -> tuple[dict, dict]:
    main = json.loads((ROOT / "outputs/scaleup_v2/analysis/results_v2.json").read_text())
    over = json.loads(
        (ROOT / "outputs/scaleup_v2_overmatch/analysis/results_v2.json").read_text()
    )
    return main, over


def _sys(block: dict, sid: str) -> dict:
    return block["systems"][sid]


def make_budget_response() -> None:
    main, over = _load_results()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.45))

    for ax, ds in zip(axes, ("omnidocbench", "fintabnet")):
        D = main["datasets"][ds]
        fixed_ids = ["fixed_2b_low", "fixed_2b_matched"]
        pts = []
        for sid in fixed_ids:
            b = _sys(D, sid)
            pts.append((b["cost_tiles"], b["cell_f1"]["mean"],
                        b["cell_f1"]["lo"], b["cell_f1"]["hi"]))
        if ds == "fintabnet":
            b = _sys(over["datasets"][ds], "fixed_2b_matched")
            pts.append((b["cost_tiles"], b["cell_f1"]["mean"],
                        b["cell_f1"]["lo"], b["cell_f1"]["hi"]))
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        yerr = [[p[1] - p[2] for p in pts], [p[3] - p[1] for p in pts]]
        ax.errorbar(xs, ys, yerr=yerr, marker="o", ms=4, lw=1.2, capsize=2.5,
                    color=C_FIXED, label="fixed 2B (uniform budget)", zorder=3)

        a = _sys(D, "adaptive_2b")
        ax.errorbar([a["cost_tiles"]], [a["cell_f1"]["mean"]],
                    yerr=[[a["cell_f1"]["mean"] - a["cell_f1"]["lo"]],
                          [a["cell_f1"]["hi"] - a["cell_f1"]["mean"]]],
                    marker="*", ms=11, color=C_ADAPT, capsize=2.5, lw=1.0,
                    label="adaptive 2B (verifier-gated)", zorder=5)

        for i in range(3):
            r = _sys(D, f"random_2b_seed{i}")
            ax.plot([r["cost_tiles"]], [r["cell_f1"]["mean"]], marker="^", ms=4.5,
                    color=C_RAND, ls="none",
                    label="random 2B (3 seeds)" if i == 0 else None, zorder=4)

        e = _sys(D, "fixed_8b_matched")
        ax.plot([e["cost_tiles"]], [e["cell_f1"]["mean"]], marker="D", ms=5,
                color=C_8B, ls="none", label="fixed 8B", zorder=4)

        ax.set_title(DS_LABEL[ds])
        ax.set_xlabel("mean visual tile budget / page")
        ax.grid(alpha=0.25, lw=0.5)

    axes[0].set_ylabel("macro cell-F1")
    axes[0].legend(loc="upper left", frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"budget_response.{ext}", bbox_inches="tight", dpi=220)
    plt.close(fig)
    print("wrote budget_response.pdf/.png")


def make_adaptive_scatter() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 4.7))
    for ax, ds in zip(axes, ("omnidocbench", "fintabnet")):
        rows = [json.loads(l) for l in
                (ROOT / f"outputs/scaleup_v2/analysis/diagnostic_{ds}.jsonl").open()]
        flags = json.loads(
            (ROOT / f"outputs/scaleup_v2/analysis/reparse_flags_{ds}.json").read_text()
        )
        for trig, color, lbl in ((False, "#b8b8b8", "verifier PASS (kept first pass)"),
                                 (True, C_ADAPT, "verifier REPARSE")):
            xs = [r["first_pass_cell_f1"] for r in rows if flags[r["page_id"]] is trig]
            ys = [r["final_cell_f1"] for r in rows if flags[r["page_id"]] is trig]
            ax.scatter(xs, ys, s=14, c=color, alpha=0.75, label=lbl, lw=0)
        ax.plot([0, 1], [0, 1], color="#444444", lw=0.8, ls="--")
        ax.set_xlim(-0.03, 1.0)
        ax.set_ylim(-0.03, 1.0)
        ax.set_title(DS_LABEL[ds])
        ax.set_ylabel("final cell-F1")
        ax.grid(alpha=0.25, lw=0.5)
    axes[1].set_xlabel("first-pass cell-F1 (at $B_{low}$)")
    axes[0].legend(loc="lower right", frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"adaptive_scatter.{ext}", bbox_inches="tight", dpi=220)
    plt.close(fig)
    print("wrote adaptive_scatter.pdf/.png")


if __name__ == "__main__":
    make_budget_response()
    make_adaptive_scatter()
