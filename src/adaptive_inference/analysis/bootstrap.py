"""Page-level bootstrap confidence intervals.

The scale-up v2 paper's tables need a 95% CI on every reported mean.
We resample page-level scores with replacement and report the
percentile interval. Resampling at the *page* level (not the cell
level) is the right unit because pages are the independent unit of
sampling; cells within a page are correlated.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class BootstrapResult:
    mean: float
    lo: float
    hi: float
    n: int
    n_resamples: int
    ci: float


def bootstrap_ci(
    per_page_values: Sequence[float],
    *,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Return mean and percentile CI bootstrapped across pages.

    A constant-array input produces a zero-width CI by construction.
    Empty input raises ``ValueError`` — callers should never silently
    report a CI on no data.
    """

    values = [float(v) for v in per_page_values]
    n = len(values)
    if n == 0:
        raise ValueError("bootstrap_ci needs at least one value")
    if not 0.0 < ci < 1.0:
        raise ValueError(f"ci must be in (0,1), got {ci}")
    if n_resamples <= 0:
        raise ValueError(f"n_resamples must be positive, got {n_resamples}")

    mean = sum(values) / n
    if n == 1:
        # CI is undefined with a single point; report the value with zero width.
        return BootstrapResult(mean=mean, lo=mean, hi=mean, n=1,
                               n_resamples=n_resamples, ci=ci)

    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_resamples):
        s = 0.0
        for _ in range(n):
            s += values[rng.randint(0, n - 1)]
        means.append(s / n)
    means.sort()
    alpha = (1.0 - ci) / 2.0
    lo_idx = int(alpha * n_resamples)
    hi_idx = int((1.0 - alpha) * n_resamples) - 1
    hi_idx = max(hi_idx, 0)
    lo_idx = min(lo_idx, n_resamples - 1)
    return BootstrapResult(
        mean=mean,
        lo=means[lo_idx],
        hi=means[hi_idx],
        n=n,
        n_resamples=n_resamples,
        ci=ci,
    )
