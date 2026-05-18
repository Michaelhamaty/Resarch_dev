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
from .loaders import LoadedSystem, ScoringSummary


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
    macro_cell_f1: float | None = None
    macro_text_similarity: float | None = None
    pages_with_parse_error: int | None = None
    pages_with_gold: int | None = None


def summarize_system(
    system: LoadedSystem, *, scoring: ScoringSummary | None = None
) -> SystemResult:
    """Build a ``SystemResult`` from a loaded manifest entry + records.

    When ``scoring`` is provided (loaded from a per-system page_scores.json),
    its macro accuracy numbers are merged into the result. When absent,
    the accuracy fields remain ``None`` so the stub-only Phase 7 path
    keeps working unchanged.
    """

    entry = system.entry
    if entry.status != "ok":
        return _attach_scoring(_placeholder_result(system), scoring)

    if entry.runner in ADAPTIVE_RUNNERS:
        return _attach_scoring(_summarize_adaptive(system), scoring)
    if entry.runner == SINGLE_PASS_RUNNER:
        return _attach_scoring(_summarize_single_pass(system), scoring)
    raise ValueError(
        f"{entry.system_id}: unknown runner {entry.runner!r} "
        f"(expected one of {SINGLE_PASS_RUNNER!r}, {ADAPTIVE_RUNNERS!r})"
    )


def _attach_scoring(
    result: SystemResult, scoring: ScoringSummary | None
) -> SystemResult:
    if scoring is None:
        return result
    return SystemResult(
        system_id=result.system_id,
        family=result.family,
        runner=result.runner,
        status=result.status,
        pages_processed=result.pages_processed,
        sample_size=result.sample_size,
        mean_runtime_ms=result.mean_runtime_ms,
        p95_runtime_ms=result.p95_runtime_ms,
        mean_output_tokens=result.mean_output_tokens,
        cost_tiles=result.cost_tiles,
        reparse_rate=result.reparse_rate,
        reparse_count=result.reparse_count,
        verifier_failure_codes=result.verifier_failure_codes,
        predicted_table_count_hist=result.predicted_table_count_hist,
        seed=result.seed,
        random_probability=result.random_probability,
        note=result.note,
        macro_cell_f1=scoring.macro_cell_f1,
        macro_text_similarity=scoring.macro_text_similarity,
        pages_with_parse_error=scoring.pages_with_parse_error,
        pages_with_gold=scoring.pages_with_gold,
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
