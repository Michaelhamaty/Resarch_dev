"""Random-escalation policy for the Phase 6 random baseline.

Sibling to :func:`should_reparse` in ``escalation.py``. The verifier is
still evaluated upstream (and its decision is logged) so Phase 7 can
compare where random and verifier-based policies diverge, but the
reparse boolean here is a purely random draw seeded by
``(seed, page_id)``.

Why ``(seed, page_id)`` and not a stream RNG:

- Determinism independent of iteration order: the decision for a page
  is stable even if the manifest is reordered, so re-running a subset
  reproduces the same per-page decisions.
- Independence across pages: no drift of the random stream based on
  page position.

This policy is intentionally separate from the verifier-based policy so
the Phase 4 adaptive path stays untouched.
"""

from __future__ import annotations

import random


def should_reparse_random(*, page_id: str, seed: int, probability: float) -> bool:
    """Return True iff a reparse should fire for this page under random policy.

    ``probability`` is expected to be in ``[0.0, 1.0]``. The seed is
    combined with ``page_id`` so each page gets its own independent draw
    that is reproducible across runs. Edge cases are handled explicitly:
    probability ``<= 0`` never reparses; probability ``>= 1`` always does.
    """

    if probability <= 0.0:
        return False
    if probability >= 1.0:
        return True
    rng = random.Random(f"{seed}:{page_id}")
    return rng.random() < probability
