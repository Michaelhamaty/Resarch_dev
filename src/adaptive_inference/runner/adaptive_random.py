"""Adaptive orchestrator variant that uses a seeded random escalation policy.

This is the Phase 6 random-baseline runner. It is a near-copy of
``run_adaptive`` with one behavioral change: the reparse boolean comes
from :func:`should_reparse_random` instead of :func:`should_reparse`.

The verifier is still invoked on the first-pass output (so Phase 7 can
compare random vs verifier decisions head-to-head on the same pages),
but its decision is ignored for routing. Artifact layout, log schema,
and Contract 1 ("at most one reparse per page") are all identical to
``run_adaptive``.

Two extra fields land in each log record: ``random_seed`` and
``random_probability`` — so a reader can reconstruct exactly which
random draw produced each decision.

Kept as its own module on purpose: the Phase 4 adaptive runner stays
untouched, and the two policies are impossible to confuse at the call
site.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from ..config.budgets import Budget
from ..config.models import ModelConfig
from ..config.prompts import PromptTemplate
from ..inference.factory import build_adapter
from ..policy.random_escalation import should_reparse_random
from ..verifier.structural import verify_page_tables
from .adaptive_logger import (
    AdaptivePageLog,
    LOG_FILENAME,
    append_adaptive_page_log,
    reset_adaptive_log,
)
from .adaptive_writer import write_final_artifacts, write_pass_artifacts
from .pages import load_pages_for_manifest


FIRST_PASS = "first_pass"
REPARSE = "reparse"


@dataclass(frozen=True)
class AdaptiveRandomRunConfig:
    """Config for the random-escalation adaptive variant.

    Mirrors ``AdaptiveRunConfig`` plus ``seed`` and ``probability``.
    No YAML loader is provided because the Phase 6 orchestrator builds
    these programmatically from the frozen calibration artifact.
    """

    run_id: str
    split_name: str
    manifest_path: Path
    records_path: Path
    image_root: Path
    model_cfg: ModelConfig
    budget_low: Budget
    budget_high: Budget
    prompt: PromptTemplate
    output_dir: Path
    seed: int
    probability: float


@dataclass(frozen=True)
class AdaptiveRandomRunSummary:
    run_id: str
    output_dir: Path
    pages_processed: int
    reparse_count: int
    log_path: Path
    seed: int
    probability: float


def run_adaptive_random(cfg: AdaptiveRandomRunConfig) -> AdaptiveRandomRunSummary:
    """Execute every page through the random-escalation adaptive pipeline."""

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    reset_adaptive_log(cfg.output_dir)

    adapter = build_adapter(cfg.model_cfg)
    loaded_pages = load_pages_for_manifest(
        manifest_path=cfg.manifest_path,
        records_path=cfg.records_path,
        image_root=cfg.image_root,
    )

    reparse_count = 0
    log_path: Path | None = None

    for loaded in loaded_pages:
        t_page_start = time.perf_counter()

        first_pass_result = adapter.run(
            page_id=loaded.record.page_id,
            image=loaded.image,
            budget=cfg.budget_low,
            prompt=cfg.prompt,
        )
        first_pass_written = write_pass_artifacts(
            first_pass_result, cfg.output_dir, FIRST_PASS
        )

        # Verifier still runs so its decision is recorded for Phase 7 analysis,
        # but the reparse boolean comes from the random policy.
        t_ver_start = time.perf_counter()
        verifier_result = verify_page_tables(first_pass_result.raw_text)
        verifier_runtime_ms = (time.perf_counter() - t_ver_start) * 1000.0

        reparse_decision = should_reparse_random(
            page_id=loaded.record.page_id,
            seed=cfg.seed,
            probability=cfg.probability,
        )

        reparse_result = None
        reparse_runtime_ms: float | None = None
        reparse_raw_rel: str | None = None
        if reparse_decision:
            t_rep_start = time.perf_counter()
            reparse_result = adapter.run(
                page_id=loaded.record.page_id,
                image=loaded.image,
                budget=cfg.budget_high,
                prompt=cfg.prompt,
            )
            reparse_runtime_ms = (time.perf_counter() - t_rep_start) * 1000.0
            reparse_written = write_pass_artifacts(
                reparse_result, cfg.output_dir, REPARSE
            )
            reparse_raw_rel = str(
                reparse_written.raw_path.relative_to(cfg.output_dir)
            )
            reparse_count += 1

        if reparse_result is not None:
            chosen_result = reparse_result
            final_output_source = REPARSE
            reparse_triggered = True
        else:
            chosen_result = first_pass_result
            final_output_source = FIRST_PASS
            reparse_triggered = False

        final_written = write_final_artifacts(
            chosen_result,
            verifier_result,
            cfg.output_dir,
            final_output_source=final_output_source,
            reparse_triggered=reparse_triggered,
        )

        total_runtime_ms = (time.perf_counter() - t_page_start) * 1000.0

        log_path = append_adaptive_page_log(
            AdaptivePageLog(
                run_id=cfg.run_id,
                split=cfg.split_name,
                model_name=cfg.model_cfg.name,
                prompt_id=cfg.prompt.id,
                budget_low_name=cfg.budget_low.name,
                budget_high_name=cfg.budget_high.name,
                first_pass=first_pass_result,
                verifier=verifier_result,
                verifier_runtime_ms=verifier_runtime_ms,
                reparse=reparse_result,
                reparse_runtime_ms=reparse_runtime_ms,
                first_pass_raw_rel=str(
                    first_pass_written.raw_path.relative_to(cfg.output_dir)
                ),
                reparse_raw_rel=reparse_raw_rel,
                final_raw_rel=str(
                    final_written.raw_path.relative_to(cfg.output_dir)
                ),
                final_output_source=final_output_source,
                reparse_triggered=reparse_triggered,
                total_runtime_ms=total_runtime_ms,
            ),
            cfg.output_dir,
        )

        # Augment the just-written last line with random-policy metadata.
        # The base logger only knows about the adaptive schema; extending
        # it would broaden the Phase 4 contract, so we attach the Phase 6
        # fields here instead.
        _augment_last_log_line(
            log_path,
            extras={
                "random_seed": cfg.seed,
                "random_probability": cfg.probability,
                "policy": "random_escalation",
            },
        )

    if log_path is None:
        log_path = cfg.output_dir / LOG_FILENAME
        log_path.touch()

    return AdaptiveRandomRunSummary(
        run_id=cfg.run_id,
        output_dir=cfg.output_dir,
        pages_processed=len(loaded_pages),
        reparse_count=reparse_count,
        log_path=log_path,
        seed=cfg.seed,
        probability=cfg.probability,
    )


def _augment_last_log_line(log_path: Path, *, extras: dict) -> None:
    """Re-serialize the final JSONL line with extra keys merged in.

    Keeps the phase-4 schema stable in the shared logger while letting
    the random-escalation runner attach its extra reproducibility fields
    without duplicating the whole logger.
    """

    text = log_path.read_text(encoding="utf-8")
    if not text:
        return
    lines = text.splitlines()
    last = json.loads(lines[-1])
    last.update(extras)
    lines[-1] = json.dumps(last, sort_keys=True)
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
