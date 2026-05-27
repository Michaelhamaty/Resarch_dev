"""Paired Wilcoxon signed-rank test for system comparisons.

For the headline claim — "adaptive beats fixed-2B at matched cost" — we
need a paired test on the *same set of pages*. Wilcoxon is the
distribution-free choice and the right default for cell-F1 scores,
which are bounded and non-normal.

The interface returns the conventional (statistic, p_value, n_pairs)
triple plus a small helper to align two ``{page_id: score}`` mappings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from scipy.stats import wilcoxon


@dataclass(frozen=True)
class PairedWilcoxonResult:
    statistic: float
    p_value: float
    n_pairs: int
    n_dropped_ties: int


def paired_wilcoxon(
    per_page_a: Sequence[float],
    per_page_b: Sequence[float],
) -> PairedWilcoxonResult:
    """Run a two-sided Wilcoxon signed-rank test on paired scores.

    Pairs with zero difference are dropped (Wilcoxon's standard handling).
    Identical inputs return p=1.0. Length mismatch raises ``ValueError``.
    """

    if len(per_page_a) != len(per_page_b):
        raise ValueError(
            f"length mismatch: a={len(per_page_a)} b={len(per_page_b)}"
        )
    n = len(per_page_a)
    if n == 0:
        raise ValueError("paired_wilcoxon needs at least one pair")

    diffs = [float(x) - float(y) for x, y in zip(per_page_a, per_page_b)]
    nonzero = [d for d in diffs if d != 0.0]
    n_dropped = n - len(nonzero)

    if not nonzero:
        # Every pair tied — there is nothing to test; report p=1.0.
        return PairedWilcoxonResult(
            statistic=0.0, p_value=1.0, n_pairs=n, n_dropped_ties=n_dropped,
        )

    res = wilcoxon(nonzero, zero_method="wilcox", correction=False, alternative="two-sided")
    return PairedWilcoxonResult(
        statistic=float(res.statistic),
        p_value=float(res.pvalue),
        n_pairs=n,
        n_dropped_ties=n_dropped,
    )


def align_pages(
    scores_a: Mapping[str, float],
    scores_b: Mapping[str, float],
) -> tuple[list[str], list[float], list[float]]:
    """Return (page_ids, values_a, values_b) in sorted, intersected order."""

    common = sorted(set(scores_a) & set(scores_b))
    a = [float(scores_a[pid]) for pid in common]
    b = [float(scores_b[pid]) for pid in common]
    return common, a, b
