"""Per-page cost accounting for calibration.

The MVP uses ``max_tiles`` as a proxy for compute — it is deterministic
across stub adapters and maps directly onto the real InternVL2 compute
knob we will swap in later. Runtime is still logged (see
``SweepPointSummary``) for audit, but matched-budget selection uses
tiles.

Cost formulas:

- Adaptive per page:  ``B_low + reparse_triggered * B_high``
  (the reparse runs *in addition to* the first pass, so both are paid.)
- Fixed per page:     ``B.max_tiles`` (constant per page).

These are pure functions with no I/O. ``summary.py`` is the module that
reads log files and calls them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping


def fixed_cost_tiles(max_tiles: int) -> float:
    """Return the per-page tile cost for a fixed-budget run."""

    if max_tiles <= 0:
        raise ValueError(f"max_tiles must be positive, got {max_tiles}")
    return float(max_tiles)


def adaptive_cost_tiles(
    log_records: Iterable[Mapping[str, object]],
    low_max_tiles: int,
    high_max_tiles: int,
) -> float:
    """Mean per-page tile cost over a Phase 4 adaptive ``run.log.jsonl``.

    ``log_records`` is the iterable of parsed JSON lines. Every record
    must carry a boolean ``reparse_triggered`` field (present since
    Phase 4). Empty input returns 0.0 so empty-manifest smoke paths
    don't crash selection.
    """

    if low_max_tiles <= 0 or high_max_tiles <= 0:
        raise ValueError(
            f"budgets must be positive; got low={low_max_tiles}, high={high_max_tiles}"
        )

    total = 0.0
    count = 0
    for rec in log_records:
        reparsed = bool(rec.get("reparse_triggered", False))
        total += float(low_max_tiles) + (
            float(high_max_tiles) if reparsed else 0.0
        )
        count += 1
    return total / count if count else 0.0
