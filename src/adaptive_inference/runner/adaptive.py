"""Adaptive orchestrator: low-budget parse -> verifier -> optional reparse.

This is the Phase 4 entry point. It is a thin composition layer on top
of existing Phase 1–3 pieces:

- ``load_pages_for_manifest`` gives us ``(PageRecord, Image)`` pairs,
- ``build_adapter`` returns the model adapter,
- ``verify_page_tables`` is the deterministic structural decision,
- ``should_reparse`` collapses the verifier decision into a boolean,
- ``write_pass_artifacts`` / ``write_final_artifacts`` handle the
  subfolder layout,
- ``append_adaptive_page_log`` writes one JSONL record per page.

Contract 1 ("one page, one policy path") is enforced by construction:
the adapter is called at most twice per page (first pass + optional
reparse); there is no third pass, ever.

Artifact layout under ``cfg.output_dir``:

```
first_pass/raw/{pid}.md      always
first_pass/pages/{pid}.json  always
reparse/raw/{pid}.md         only if REPARSE
reparse/pages/{pid}.json     only if REPARSE
final/raw/{pid}.md           always (copy of chosen pass)
final/pages/{pid}.json       always (with verifier metadata)
run.log.jsonl                one line per page
```
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from ..config.adaptive_runs import AdaptiveRunConfig
from ..inference.factory import build_adapter
from ..policy.escalation import should_reparse
from ..verifier.structural import verify_page_tables
from .adaptive_logger import (
    AdaptivePageLog,
    append_adaptive_page_log,
    reset_adaptive_log,
)
from .adaptive_writer import write_final_artifacts, write_pass_artifacts
from .pages import load_pages_for_manifest


FIRST_PASS = "first_pass"
REPARSE = "reparse"


@dataclass(frozen=True)
class AdaptiveRunSummary:
    run_id: str
    output_dir: Path
    pages_processed: int
    reparse_count: int
    log_path: Path


def run_adaptive(cfg: AdaptiveRunConfig) -> AdaptiveRunSummary:
    """Run every page in ``cfg.manifest_path`` through the adaptive pipeline."""

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    # Idempotent re-run: clear the log so line count == pages processed.
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

        # --- low-budget first pass --------------------------------------
        first_pass_result = adapter.run(
            page_id=loaded.record.page_id,
            image=loaded.image,
            budget=cfg.budget_low,
            prompt=cfg.prompt,
        )
        first_pass_written = write_pass_artifacts(
            first_pass_result, cfg.output_dir, FIRST_PASS
        )

        # --- verifier ---------------------------------------------------
        t_ver_start = time.perf_counter()
        verifier_result = verify_page_tables(first_pass_result.raw_text)
        verifier_runtime_ms = (time.perf_counter() - t_ver_start) * 1000.0

        # --- optional one-shot reparse ----------------------------------
        reparse_result = None
        reparse_runtime_ms: float | None = None
        reparse_raw_rel: str | None = None
        if should_reparse(verifier_result):
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

        # --- final selected artifact ------------------------------------
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

        # --- log --------------------------------------------------------
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

    # Materialize an empty log file even when the manifest is empty so
    # downstream tools can assume it exists.
    if log_path is None:
        log_path = cfg.output_dir / "run.log.jsonl"
        log_path.touch()

    return AdaptiveRunSummary(
        run_id=cfg.run_id,
        output_dir=cfg.output_dir,
        pages_processed=len(loaded_pages),
        reparse_count=reparse_count,
        log_path=log_path,
    )
