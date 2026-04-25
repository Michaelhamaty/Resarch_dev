"""Phase 6 experiment orchestrator: run every system, write the manifest.

This module is deliberately thin. It:

1. Reads the Phase 5 frozen artifact + held-out split (both treated as
   read-only truth sources).
2. For each ``SystemSpec``, constructs the right config (Phase 2's
   ``RunConfig``, Phase 4's ``AdaptiveRunConfig``, or Phase 6's
   ``AdaptiveRandomRunConfig``) by pulling budgets straight out of the
   frozen artifact — not from a YAML budget file — so there is no path
   through which Phase 6 could reintroduce drift against calibration.
3. Calls the existing runner and collects a ``ManifestEntry`` per system.
4. Writes one ``manifest.json`` at the end.

Failure of one system does not abort the others: failed entries land in
the manifest with ``status="failed"`` and an ``error`` message. The CLI
driver turns "any failure" into a non-zero exit.

The 8B stub gate is enforced here: if ``B_fix_8B.adapter_kind == "stub"``
and ``allow_stubbed_8b`` is False, the ``fixed_8b_matched`` system is
skipped (``status="skipped_stub_8b"``). This matches the Phase 6
promise of not silently faking the headline 8B number.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path

from ..calibration.artifact import FrozenBudget, FrozenBudgets, load_frozen_budgets
from ..config.adaptive_runs import AdaptiveRunConfig
from ..config.prompts import PromptTemplate, load_prompt_template
from ..config.runs import RunConfig
from ..runner.adaptive import run_adaptive
from ..runner.adaptive_random import (
    AdaptiveRandomRunConfig,
    run_adaptive_random,
)
from ..runner.single_pass import run_single_pass
from .frozen_inputs import (
    CalibrationReparseRate,
    derive_calibration_reparse_rate,
    frozen_to_budget,
    model_cfg_for,
)
from .manifest import (
    ManifestEntry,
    ManifestHeader,
    git_head_or_none,
    iso_now,
    sha256_of_file,
    write_manifest,
)
from .systems import (
    ADAPTIVE_2B,
    FIXED_2B_LOW,
    FIXED_2B_MATCHED,
    FIXED_8B_MATCHED,
    RANDOM_2B,
    SystemSpec,
    build_all_system_specs,
    filter_specs,
)


@dataclass(frozen=True)
class Phase6Config:
    """Top-level config for a Phase 6 execution, loaded from YAML."""

    run_set_id: str
    frozen_budgets_path: Path
    split_name: str
    held_out_manifest_path: Path
    records_path: Path
    image_root: Path
    model_config_path: Path
    prompt_config_path: Path
    output_root: Path
    random_seeds: tuple[int, ...]


@dataclass(frozen=True)
class Phase6Result:
    manifest_path: Path
    entries: tuple[ManifestEntry, ...]
    any_failed: bool


def run_phase6(
    cfg: Phase6Config,
    *,
    allow_stubbed_8b: bool = False,
    systems_filter: tuple[str, ...] | None = None,
    repo_root: Path | None = None,
) -> Phase6Result:
    """Execute every requested system and write the run manifest."""

    frozen = load_frozen_budgets(cfg.frozen_budgets_path)
    prompt = load_prompt_template(cfg.prompt_config_path)
    reparse_rate = derive_calibration_reparse_rate(frozen)

    specs = build_all_system_specs(cfg.random_seeds)
    specs = filter_specs(specs, systems_filter)

    output_root = cfg.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    entries: list[ManifestEntry] = []
    for spec in specs:
        entry = _run_one_system(
            spec=spec,
            cfg=cfg,
            frozen=frozen,
            prompt=prompt,
            reparse_rate=reparse_rate,
            allow_stubbed_8b=allow_stubbed_8b,
        )
        entries.append(entry)

    header = ManifestHeader(
        run_set_id=cfg.run_set_id,
        generated_at=iso_now(),
        held_out_manifest_path=str(cfg.held_out_manifest_path),
        held_out_manifest_sha256=sha256_of_file(cfg.held_out_manifest_path),
        frozen_budgets_path=str(cfg.frozen_budgets_path),
        frozen_budgets_sha256=sha256_of_file(cfg.frozen_budgets_path),
        prompt_id=prompt.id,
        prompt_version=prompt.version,
        git_head=git_head_or_none(repo_root if repo_root is not None else Path.cwd()),
        calibration_reparse_rate=reparse_rate.probability,
        calibration_reparse_rate_degenerate=reparse_rate.degenerate,
        random_seeds=list(cfg.random_seeds),
    )
    manifest_path = write_manifest(
        output_root=output_root, header=header, entries=entries
    )

    any_failed = any(e.status == "failed" for e in entries)
    return Phase6Result(
        manifest_path=manifest_path,
        entries=tuple(entries),
        any_failed=any_failed,
    )


def _run_one_system(
    *,
    spec: SystemSpec,
    cfg: Phase6Config,
    frozen: FrozenBudgets,
    prompt: PromptTemplate,
    reparse_rate: CalibrationReparseRate,
    allow_stubbed_8b: bool,
) -> ManifestEntry:
    """Dispatch to the right runner for ``spec`` and wrap the outcome."""

    system_out_dir = cfg.output_root / spec.system_id
    started_at = iso_now()

    try:
        if spec.family == ADAPTIVE_2B:
            return _run_adaptive_2b(
                spec, cfg, frozen, prompt, system_out_dir, started_at
            )
        if spec.family == FIXED_2B_LOW:
            return _run_fixed_single_pass(
                spec, cfg, frozen.b_low, frozen, prompt, system_out_dir, started_at
            )
        if spec.family == FIXED_2B_MATCHED:
            return _run_fixed_single_pass(
                spec, cfg, frozen.b_fix_2b, frozen, prompt, system_out_dir, started_at
            )
        if spec.family == RANDOM_2B:
            return _run_random_2b(
                spec, cfg, frozen, prompt, reparse_rate, system_out_dir, started_at
            )
        if spec.family == FIXED_8B_MATCHED:
            return _run_fixed_8b(
                spec,
                cfg,
                frozen,
                prompt,
                system_out_dir,
                started_at,
                allow_stubbed_8b=allow_stubbed_8b,
            )
        raise ValueError(f"Unknown system family: {spec.family!r}")
    except Exception as exc:  # noqa: BLE001 — we want one bad system to not abort the others
        return ManifestEntry(
            system_id=spec.system_id,
            family=spec.family,
            runner=spec.runner,
            status="failed",
            output_dir=str(system_out_dir),
            started_at=started_at,
            finished_at=iso_now(),
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            seed=spec.seed,
        )


def _run_adaptive_2b(
    spec: SystemSpec,
    cfg: Phase6Config,
    frozen: FrozenBudgets,
    prompt: PromptTemplate,
    out_dir: Path,
    started_at: str,
) -> ManifestEntry:
    adaptive_cfg = AdaptiveRunConfig(
        run_id=spec.system_id,
        split_name=cfg.split_name,
        manifest_path=cfg.held_out_manifest_path,
        records_path=cfg.records_path,
        image_root=cfg.image_root,
        model_cfg=model_cfg_for(frozen.b_low, str(cfg.model_config_path)),
        budget_low=frozen_to_budget(frozen.b_low),
        budget_high=frozen_to_budget(frozen.b_high),
        prompt=prompt,
        output_dir=out_dir,
    )
    summary = run_adaptive(adaptive_cfg)
    return ManifestEntry(
        system_id=spec.system_id,
        family=spec.family,
        runner=spec.runner,
        status="ok",
        output_dir=str(summary.output_dir),
        pages_processed=summary.pages_processed,
        reparse_count=summary.reparse_count,
        model_name=adaptive_cfg.model_cfg.name,
        adapter_kind=adaptive_cfg.model_cfg.adapter_kind,
        budget_low_name=adaptive_cfg.budget_low.name,
        budget_low_max_tiles=adaptive_cfg.budget_low.max_tiles,
        budget_high_name=adaptive_cfg.budget_high.name,
        budget_high_max_tiles=adaptive_cfg.budget_high.max_tiles,
        started_at=started_at,
        finished_at=iso_now(),
    )


def _run_fixed_single_pass(
    spec: SystemSpec,
    cfg: Phase6Config,
    fb: FrozenBudget,
    frozen: FrozenBudgets,
    prompt: PromptTemplate,
    out_dir: Path,
    started_at: str,
) -> ManifestEntry:
    run_cfg = RunConfig(
        run_id=spec.system_id,
        split_name=cfg.split_name,
        manifest_path=cfg.held_out_manifest_path,
        records_path=cfg.records_path,
        image_root=cfg.image_root,
        model_cfg=model_cfg_for(fb, str(cfg.model_config_path)),
        budget=frozen_to_budget(fb),
        prompt=prompt,
        output_dir=out_dir,
    )
    summary = run_single_pass(run_cfg)
    return ManifestEntry(
        system_id=spec.system_id,
        family=spec.family,
        runner=spec.runner,
        status="ok",
        output_dir=str(summary.output_dir),
        pages_processed=summary.pages_processed,
        model_name=run_cfg.model_cfg.name,
        adapter_kind=run_cfg.model_cfg.adapter_kind,
        budget_name=run_cfg.budget.name,
        budget_max_tiles=run_cfg.budget.max_tiles,
        started_at=started_at,
        finished_at=iso_now(),
    )


def _run_random_2b(
    spec: SystemSpec,
    cfg: Phase6Config,
    frozen: FrozenBudgets,
    prompt: PromptTemplate,
    reparse_rate: CalibrationReparseRate,
    out_dir: Path,
    started_at: str,
) -> ManifestEntry:
    assert spec.seed is not None, "random_2b spec must carry a seed"
    random_cfg = AdaptiveRandomRunConfig(
        run_id=spec.system_id,
        split_name=cfg.split_name,
        manifest_path=cfg.held_out_manifest_path,
        records_path=cfg.records_path,
        image_root=cfg.image_root,
        model_cfg=model_cfg_for(frozen.b_low, str(cfg.model_config_path)),
        budget_low=frozen_to_budget(frozen.b_low),
        budget_high=frozen_to_budget(frozen.b_high),
        prompt=prompt,
        output_dir=out_dir,
        seed=spec.seed,
        probability=reparse_rate.probability,
    )
    summary = run_adaptive_random(random_cfg)
    notes: list[str] = []
    if reparse_rate.degenerate:
        notes.append(
            "calibration-measured reparse rate is 0.0 — random policy will never "
            "escalate. This matches Phase 5 data as frozen."
        )
    return ManifestEntry(
        system_id=spec.system_id,
        family=spec.family,
        runner=spec.runner,
        status="ok",
        output_dir=str(summary.output_dir),
        pages_processed=summary.pages_processed,
        reparse_count=summary.reparse_count,
        model_name=random_cfg.model_cfg.name,
        adapter_kind=random_cfg.model_cfg.adapter_kind,
        budget_low_name=random_cfg.budget_low.name,
        budget_low_max_tiles=random_cfg.budget_low.max_tiles,
        budget_high_name=random_cfg.budget_high.name,
        budget_high_max_tiles=random_cfg.budget_high.max_tiles,
        seed=spec.seed,
        random_probability=reparse_rate.probability,
        random_probability_source=reparse_rate.source,
        started_at=started_at,
        finished_at=iso_now(),
        notes=notes,
    )


def _run_fixed_8b(
    spec: SystemSpec,
    cfg: Phase6Config,
    frozen: FrozenBudgets,
    prompt: PromptTemplate,
    out_dir: Path,
    started_at: str,
    *,
    allow_stubbed_8b: bool,
) -> ManifestEntry:
    fb = frozen.b_fix_8b
    if fb.adapter_kind == "stub" and not allow_stubbed_8b:
        return ManifestEntry(
            system_id=spec.system_id,
            family=spec.family,
            runner=spec.runner,
            status="skipped_stub_8b",
            output_dir=None,
            model_name=fb.model_name,
            adapter_kind=fb.adapter_kind,
            budget_name=fb.name,
            budget_max_tiles=fb.max_tiles,
            started_at=started_at,
            finished_at=iso_now(),
            reason=(
                "B_fix_8B.adapter_kind == 'stub' and --allow-stubbed-8b was not "
                "passed. The 8B matched-cost system is the headline upper-bound "
                "baseline and Phase 6 refuses to fake it silently. Pass the "
                "flag explicitly to run the stubbed 8B for pipeline validation."
            ),
        )
    return _run_fixed_single_pass(
        spec, cfg, fb, frozen, prompt, out_dir, started_at
    )
