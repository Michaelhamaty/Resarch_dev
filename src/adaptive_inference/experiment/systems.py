"""Phase 6 system definitions: which runs we execute and how to label them.

Pure data. The five required systems plus seed expansion for the
random-escalation baseline. Keeping these as small dataclasses — not
loaded from YAML — protects the experiment surface from accidental
drift. Adding or removing a system is a source-code change reviewed in
git, not a silent YAML edit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


# Canonical system identifiers. These strings also double as the
# subdirectory names under ``outputs/runs/phase6/`` and the row keys in
# the run manifest, so changes to them are breaking for Phase 7.
ADAPTIVE_2B = "adaptive_2b"
FIXED_2B_LOW = "fixed_2b_low"
FIXED_2B_MATCHED = "fixed_2b_matched"
RANDOM_2B = "random_2b"  # expands per seed
FIXED_8B_MATCHED = "fixed_8b_matched"


RunnerKind = Literal["single_pass", "adaptive", "adaptive_random"]


@dataclass(frozen=True)
class SystemSpec:
    """Static description of one system run in Phase 6.

    ``seed`` is only set for the random-escalation variant; for every
    other system it stays ``None``.
    """

    system_id: str       # e.g. "adaptive_2b" or "random_2b_seed0"
    family: str          # e.g. "adaptive_2b", "random_2b" — shared across seeds
    runner: RunnerKind
    seed: int | None = None


def required_family_ids() -> tuple[str, ...]:
    """The canonical five-family list Phase 6 promises to execute."""

    return (
        ADAPTIVE_2B,
        FIXED_2B_LOW,
        FIXED_2B_MATCHED,
        RANDOM_2B,
        FIXED_8B_MATCHED,
    )


def expand_random_seeds(seeds: tuple[int, ...]) -> tuple[SystemSpec, ...]:
    """Return one ``SystemSpec`` per seed for the random-escalation family."""

    if not seeds:
        raise ValueError("random_2b requires at least one seed")
    return tuple(
        SystemSpec(
            system_id=f"{RANDOM_2B}_seed{seed}",
            family=RANDOM_2B,
            runner="adaptive_random",
            seed=seed,
        )
        for seed in seeds
    )


def build_all_system_specs(seeds: tuple[int, ...]) -> tuple[SystemSpec, ...]:
    """Build the full ordered list of Phase 6 system specs.

    Order matters: ``adaptive_2b`` first (cheap to read and the most
    referenced run), then the simple fixed baselines, then the random
    seeds, then ``fixed_8b_matched`` (often gated or slow).
    """

    specs: list[SystemSpec] = [
        SystemSpec(system_id=ADAPTIVE_2B, family=ADAPTIVE_2B, runner="adaptive"),
        SystemSpec(system_id=FIXED_2B_LOW, family=FIXED_2B_LOW, runner="single_pass"),
        SystemSpec(
            system_id=FIXED_2B_MATCHED, family=FIXED_2B_MATCHED, runner="single_pass"
        ),
    ]
    specs.extend(expand_random_seeds(seeds))
    specs.append(
        SystemSpec(
            system_id=FIXED_8B_MATCHED, family=FIXED_8B_MATCHED, runner="single_pass"
        )
    )
    return tuple(specs)


def filter_specs(
    specs: tuple[SystemSpec, ...], families: tuple[str, ...] | None
) -> tuple[SystemSpec, ...]:
    """Keep only specs whose family (or system_id) appears in ``families``.

    ``None`` means keep everything. If ``families`` is provided, every
    entry must match at least one spec or we raise — silent no-ops on a
    typo would be painful.
    """

    if families is None:
        return specs
    wanted = set(families)
    kept = tuple(s for s in specs if s.family in wanted or s.system_id in wanted)
    seen_families = {s.family for s in kept} | {s.system_id for s in kept}
    unknown = sorted(w for w in wanted if w not in seen_families)
    if unknown:
        raise ValueError(
            f"Unknown --systems entries: {unknown}. "
            f"Available families: {sorted({s.family for s in specs})}."
        )
    return kept
