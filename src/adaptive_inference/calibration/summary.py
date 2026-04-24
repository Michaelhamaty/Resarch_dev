"""Distill a run.log.jsonl into a single SweepPointSummary.

One summary object per sweep point. Feeds both the selection logic and
the per-sweep-point audit JSONL.

Statistics recorded (per the Phase 5 brief):

- ``mean_runtime_ms`` / ``p95_runtime_ms`` — total runtime per page.
- ``mean_output_tokens`` — final chosen pass for adaptive, the sole
  pass for single-pass.
- ``mean_tile_budget`` — same formula as ``cost_tiles`` (tile cost
  *is* the mean tile budget under the Phase 5 cost convention).
- ``reparse_rate`` — only meaningful for adaptive; ``None`` for fixed.
- ``cost_tiles`` — matched-budget selection key.
- ``sample_size`` — number of log lines consumed.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .cost import adaptive_cost_tiles, fixed_cost_tiles


@dataclass(frozen=True)
class SweepPointSummary:
    sample_size: int
    mean_runtime_ms: float
    p95_runtime_ms: float
    mean_output_tokens: float
    mean_tile_budget: float
    reparse_rate: float | None
    cost_tiles: float


def _read_jsonl(path: str | Path) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8")
    if not text.strip():
        return []
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _p95(values: Iterable[float]) -> float:
    """Linear-interpolated 95th percentile. Returns 0.0 on empty input."""

    sorted_vals = sorted(float(v) for v in values)
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    # Linear interpolation between order statistics, matching numpy's
    # default "linear" method so downstream readers don't get surprised.
    pos = 0.95 * (len(sorted_vals) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_vals[int(pos)]
    frac = pos - lo
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


def summarize_adaptive_log(
    path: str | Path,
    low_max_tiles: int,
    high_max_tiles: int,
) -> SweepPointSummary:
    """Summarize a Phase 4 ``run.log.jsonl``.

    Output tokens come from ``reparse_output_tokens`` when reparse
    fired, otherwise ``first_pass_output_tokens`` — i.e. the tokens of
    the finally-kept pass, matching what a downstream evaluator would
    have scored.
    """

    records = _read_jsonl(path)
    runtimes = [float(r["total_runtime_ms"]) for r in records]
    tokens: list[float] = []
    reparse_flags: list[bool] = []
    for r in records:
        reparsed = bool(r.get("reparse_triggered", False))
        reparse_flags.append(reparsed)
        if reparsed and r.get("reparse_output_tokens") is not None:
            tokens.append(float(r["reparse_output_tokens"]))
        else:
            tokens.append(float(r["first_pass_output_tokens"]))

    cost = adaptive_cost_tiles(records, low_max_tiles, high_max_tiles)
    reparse_rate = (
        sum(1 for flag in reparse_flags if flag) / len(reparse_flags)
        if reparse_flags
        else 0.0
    )

    return SweepPointSummary(
        sample_size=len(records),
        mean_runtime_ms=_mean(runtimes),
        p95_runtime_ms=_p95(runtimes),
        mean_output_tokens=_mean(tokens),
        mean_tile_budget=cost,
        reparse_rate=reparse_rate,
        cost_tiles=cost,
    )


def summarize_single_pass_log(
    path: str | Path,
    max_tiles: int,
) -> SweepPointSummary:
    """Summarize a Phase 2 ``run.log.jsonl`` (single-pass, fixed budget)."""

    records = _read_jsonl(path)
    runtimes = [float(r["runtime_ms"]) for r in records]
    tokens = [float(r["output_token_count"]) for r in records]
    cost = fixed_cost_tiles(max_tiles)
    return SweepPointSummary(
        sample_size=len(records),
        mean_runtime_ms=_mean(runtimes),
        p95_runtime_ms=_p95(runtimes),
        mean_output_tokens=_mean(tokens),
        mean_tile_budget=cost,
        reparse_rate=None,
        cost_tiles=cost,
    )
