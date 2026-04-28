"""Per-system result summaries built from Phase 6 logs.

Wraps the Phase 5 ``calibration.summary`` helpers so the runtime / token
statistics use the same code path Phase 5 used to pick budgets — Phase 7
must not invent a parallel summarizer.

Adaptive (and adaptive_random) systems get extra fields that single-pass
systems do not have: reparse rate, verifier failure-code histogram,
predicted-table-count distribution. Those are ``None`` for fixed
baselines, never zero, so a downstream reader can tell "no reparse
occurred" apart from "this system has no reparse concept".
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Mapping

from ..calibration.summary import (
    SweepPointSummary,
    summarize_adaptive_log,
    summarize_single_pass_log,
)
from .loaders import LoadedSystem


ADAPTIVE_RUNNERS = ("adaptive", "adaptive_random")
SINGLE_PASS_RUNNER = "single_pass"


@dataclass(frozen=True)
class SystemResult:
    """One row of the Phase 7 results table."""

    system_id: str
    family: str
    runner: str
    status: str
    pages_processed: int
    sample_size: int
    mean_runtime_ms: float
    p95_runtime_ms: float
    mean_output_tokens: float
    cost_tiles: float
    reparse_rate: float | None
    reparse_count: int | None
    verifier_failure_codes: Mapping[str, int] | None
    predicted_table_count_hist: Mapping[int, int] | None
    seed: int | None
    random_probability: float | None
    note: str | None


def summarize_system(system: LoadedSystem) -> SystemResult:
    """Build a ``SystemResult`` from a loaded manifest entry + records."""

    entry = system.entry
    if entry.status != "ok":
        return _placeholder_result(system)

    if entry.runner in ADAPTIVE_RUNNERS:
        return _summarize_adaptive(system)
    if entry.runner == SINGLE_PASS_RUNNER:
        return _summarize_single_pass(system)
    raise ValueError(
        f"{entry.system_id}: unknown runner {entry.runner!r} "
        f"(expected one of {SINGLE_PASS_RUNNER!r}, {ADAPTIVE_RUNNERS!r})"
    )


def _summarize_adaptive(system: LoadedSystem) -> SystemResult:
    entry = system.entry
    if entry.budget_low_max_tiles is None or entry.budget_high_max_tiles is None:
        raise ValueError(
            f"{entry.system_id}: adaptive entry is missing low/high tile budgets"
        )

    sweep: SweepPointSummary = summarize_adaptive_log(
        system.log_path,
        low_max_tiles=entry.budget_low_max_tiles,
        high_max_tiles=entry.budget_high_max_tiles,
    )

    failure_codes: Counter[str] = Counter()
    table_counts: Counter[int] = Counter()
    reparse_count = 0
    for rec in system.records:
        codes = rec.get("verifier_failure_codes") or []
        if isinstance(codes, list):
            for code in codes:
                failure_codes[str(code)] += 1
        if rec.get("reparse_triggered"):
            reparse_count += 1
        ptc = rec.get("predicted_table_count")
        if isinstance(ptc, int):
            table_counts[ptc] += 1

    return SystemResult(
        system_id=entry.system_id,
        family=entry.family,
        runner=entry.runner,
        status=entry.status,
        pages_processed=entry.pages_processed or len(system.records),
        sample_size=sweep.sample_size,
        mean_runtime_ms=sweep.mean_runtime_ms,
        p95_runtime_ms=sweep.p95_runtime_ms,
        mean_output_tokens=sweep.mean_output_tokens,
        cost_tiles=sweep.cost_tiles,
        reparse_rate=sweep.reparse_rate,
        reparse_count=reparse_count,
        verifier_failure_codes=dict(failure_codes),
        predicted_table_count_hist={int(k): int(v) for k, v in table_counts.items()},
        seed=entry.seed,
        random_probability=entry.random_probability,
        note=entry.notes[0] if entry.notes else None,
    )


def _summarize_single_pass(system: LoadedSystem) -> SystemResult:
    entry = system.entry
    if entry.budget_max_tiles is None:
        raise ValueError(
            f"{entry.system_id}: single_pass entry is missing budget_max_tiles"
        )
    sweep = summarize_single_pass_log(system.log_path, max_tiles=entry.budget_max_tiles)
    return SystemResult(
        system_id=entry.system_id,
        family=entry.family,
        runner=entry.runner,
        status=entry.status,
        pages_processed=entry.pages_processed or len(system.records),
        sample_size=sweep.sample_size,
        mean_runtime_ms=sweep.mean_runtime_ms,
        p95_runtime_ms=sweep.p95_runtime_ms,
        mean_output_tokens=sweep.mean_output_tokens,
        cost_tiles=sweep.cost_tiles,
        reparse_rate=None,
        reparse_count=None,
        verifier_failure_codes=None,
        predicted_table_count_hist=None,
        seed=entry.seed,
        random_probability=entry.random_probability,
        note=entry.notes[0] if entry.notes else None,
    )


def _placeholder_result(system: LoadedSystem) -> SystemResult:
    """Skipped/failed systems still appear in the table, just empty."""

    entry = system.entry
    return SystemResult(
        system_id=entry.system_id,
        family=entry.family,
        runner=entry.runner,
        status=entry.status,
        pages_processed=0,
        sample_size=0,
        mean_runtime_ms=0.0,
        p95_runtime_ms=0.0,
        mean_output_tokens=0.0,
        cost_tiles=0.0,
        reparse_rate=None,
        reparse_count=None,
        verifier_failure_codes=None,
        predicted_table_count_hist=None,
        seed=entry.seed,
        random_probability=entry.random_probability,
        note=entry.notes[0] if entry.notes else None,
    )
