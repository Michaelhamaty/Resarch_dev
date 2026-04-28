"""Reparse-rate and verifier failure-code analysis.

Adaptive systems only. Single-pass systems get an explicit ``None``
where reparse data would otherwise live, so a downstream reader cannot
mistake "no reparses occurred" for "this baseline has no reparse
concept".
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Mapping

from .loaders import LoadedSystem
from .results import ADAPTIVE_RUNNERS


@dataclass(frozen=True)
class ReparseSlice:
    """Per-system reparse / verifier histogram."""

    system_id: str
    family: str
    runner: str
    sample_size: int
    reparse_count: int
    reparse_rate: float
    verifier_decision_counts: Mapping[str, int]
    verifier_failure_codes: Mapping[str, int]
    seed: int | None
    random_probability: float | None


@dataclass(frozen=True)
class ReparseSummary:
    slices: tuple[ReparseSlice, ...]
    note: str


def build_reparse_summary(systems: tuple[LoadedSystem, ...]) -> ReparseSummary:
    """Build a reparse summary across all adaptive systems."""

    slices: list[ReparseSlice] = []
    for sys in systems:
        if sys.entry.runner not in ADAPTIVE_RUNNERS:
            continue
        slices.append(_one(sys))
    note = (
        "reparse_rate is observed on the held-out split. Single-pass baselines "
        "are intentionally absent because they have no reparse decision."
    )
    return ReparseSummary(slices=tuple(slices), note=note)


def _one(sys: LoadedSystem) -> ReparseSlice:
    entry = sys.entry
    decisions: Counter[str] = Counter()
    codes: Counter[str] = Counter()
    reparse_count = 0
    sample_size = 0
    for rec in sys.records:
        sample_size += 1
        decision = rec.get("verifier_decision")
        if isinstance(decision, str):
            decisions[decision] += 1
        rec_codes = rec.get("verifier_failure_codes") or []
        if isinstance(rec_codes, list):
            for c in rec_codes:
                codes[str(c)] += 1
        if rec.get("reparse_triggered"):
            reparse_count += 1

    rate = reparse_count / sample_size if sample_size else 0.0
    return ReparseSlice(
        system_id=entry.system_id,
        family=entry.family,
        runner=entry.runner,
        sample_size=sample_size,
        reparse_count=reparse_count,
        reparse_rate=rate,
        verifier_decision_counts=dict(decisions),
        verifier_failure_codes=dict(codes),
        seed=entry.seed,
        random_probability=entry.random_probability,
    )


def to_dict(summary: ReparseSummary) -> Mapping[str, object]:
    return {
        "note": summary.note,
        "slices": [
            {
                "system_id": s.system_id,
                "family": s.family,
                "runner": s.runner,
                "sample_size": s.sample_size,
                "reparse_count": s.reparse_count,
                "reparse_rate": s.reparse_rate,
                "verifier_decision_counts": dict(s.verifier_decision_counts),
                "verifier_failure_codes": dict(s.verifier_failure_codes),
                "seed": s.seed,
                "random_probability": s.random_probability,
            }
            for s in summary.slices
        ],
    }
