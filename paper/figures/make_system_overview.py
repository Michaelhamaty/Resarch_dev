"""Generate paper/figures/system_overview.{pdf,png}.

Architecture block diagram for the verifier-gated adaptive inference
pipeline: a page is parsed once at B_low, a deterministic structural
verifier inspects the HTML, and only FAIL pages are reparsed once at
B_high (the at-most-one-reparse contract). Pure matplotlib; no data
inputs. Lives under paper/ (untracked) since it is paper artwork, not
part of the tracked analysis pipeline.

    uv run python paper/figures/make_system_overview.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent

# Palette (colour-blind safe, muted).
C_INPUT = "#e8e8e8"
C_MODEL = "#cfe3f5"
C_VERIF = "#fde9c9"
C_DEC = "#f6d6d6"
C_OUT = "#d6efd6"
EDGE = "#333333"


def _box(ax, x, y, w, h, text, fc, *, fontsize=10, style="round,pad=0.02,rounding_size=0.06"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle=style,
            linewidth=1.3, edgecolor=EDGE, facecolor=fc, zorder=2,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, zorder=3, wrap=True)


def _arrow(ax, p0, p1, *, label=None, color=EDGE, rad=0.0, lx=0, ly=0):
    ax.add_patch(
        FancyArrowPatch(
            p0, p1, arrowstyle="-|>", mutation_scale=14,
            linewidth=1.4, color=color,
            connectionstyle=f"arc3,rad={rad}", zorder=1,
        )
    )
    if label:
        mx, my = (p0[0] + p1[0]) / 2 + lx, (p0[1] + p1[1]) / 2 + ly
        ax.text(mx, my, label, ha="center", va="center", fontsize=9,
                color=color, zorder=3,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))


def main() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 3.5))
    ax.set_xlim(0, 21)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Row centres.
    yc = 3.4   # main spine
    yt = 5.6   # PASS (top) path

    _box(ax, 0.3, yc - 0.9, 2.6, 1.8, "Page\nimage", C_INPUT, fontsize=10)
    _box(ax, 3.6, yc - 0.9, 3.3, 1.8,
         r"First pass" + "\n" + r"parse @ $B_{\mathrm{low}}$", C_MODEL)
    _box(ax, 7.7, yc - 1.05, 3.6, 2.1,
         "Deterministic\nstructural verifier\n(parse / table / span)", C_VERIF,
         fontsize=9.5)
    _box(ax, 12.2, yc - 0.95, 2.2, 1.9, "PASS?", C_DEC, fontsize=11)
    _box(ax, 15.2, yc - 0.95, 3.4, 1.9,
         r"Reparse @ $B_{\mathrm{high}}$" + "\n(at most once)", C_MODEL,
         fontsize=9.5)
    _box(ax, 15.4, yt - 0.6, 3.0, 1.3, "Output HTML", C_OUT)

    # Spine arrows.
    _arrow(ax, (2.9, yc), (3.6, yc))
    _arrow(ax, (6.9, yc), (7.7, yc))
    _arrow(ax, (11.3, yc), (12.2, yc))

    # FAIL -> reparse (right along spine).
    _arrow(ax, (14.4, yc), (15.2, yc), label="FAIL", ly=0.45)
    # reparse -> output (up).
    _arrow(ax, (16.9, yc + 0.95), (16.9, yt - 0.6), rad=0.0)

    # PASS -> output (up from decision, then right).
    _arrow(ax, (13.3, yc + 0.95), (13.3, yt), label="PASS", lx=-0.05, ly=0.0, rad=0.0)
    _arrow(ax, (13.3, yt), (15.4, yt + 0.05))

    ax.text(10.5, 0.55,
            r"One forward pass on every page; a second pass only on verifier-flagged pages.",
            ha="center", va="center", fontsize=9, style="italic", color="#555555")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"system_overview.{ext}", bbox_inches="tight", dpi=200)
    print(f"wrote system_overview.pdf / .png to {OUT}")


if __name__ == "__main__":
    main()
