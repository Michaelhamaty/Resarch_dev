"""Single-pass orchestrator: run one manifest through one adapter end-to-end.

This is the only module that knows how the pieces compose in Phase 2:
config in → pages loaded → adapter called → raw + sidecar + log written
→ summary returned. No verifier, no reparse, no calibration — those are
added to this orchestrator (or a sibling orchestrator) in later phases.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config.runs import RunConfig
from ..inference.factory import build_adapter
from .output_writer import write_page_result
from .pages import load_pages_for_manifest
from .runtime_logger import append_page_log, reset_log


@dataclass(frozen=True)
class SinglePassSummary:
    run_id: str
    output_dir: Path
    pages_processed: int
    raw_paths: tuple[Path, ...]
    sidecar_paths: tuple[Path, ...]
    log_path: Path


def run_single_pass(cfg: RunConfig) -> SinglePassSummary:
    """Execute the Phase 2 smoke path over every page in ``cfg.manifest_path``."""

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    # Re-running a run_id is idempotent: clear the log so we never double-append.
    reset_log(cfg.output_dir)

    adapter = build_adapter(cfg.model_cfg)
    loaded_pages = load_pages_for_manifest(
        manifest_path=cfg.manifest_path,
        records_path=cfg.records_path,
        image_root=cfg.image_root,
    )

    raw_paths: list[Path] = []
    sidecar_paths: list[Path] = []
    log_path: Path | None = None

    for loaded in loaded_pages:
        result = adapter.run(
            page_id=loaded.record.page_id,
            image=loaded.image,
            budget=cfg.budget,
            prompt=cfg.prompt,
        )
        written = write_page_result(result, cfg.output_dir)
        log_path = append_page_log(
            result,
            run_id=cfg.run_id,
            split=cfg.split_name,
            raw_output_path=written.raw_path.relative_to(cfg.output_dir),
            output_dir=cfg.output_dir,
        )
        raw_paths.append(written.raw_path)
        sidecar_paths.append(written.sidecar_path)

    # With zero pages the log file never gets created; materialize an empty
    # one so downstream tools can always assume it exists.
    if log_path is None:
        log_path = cfg.output_dir / "run.log.jsonl"
        log_path.touch()

    return SinglePassSummary(
        run_id=cfg.run_id,
        output_dir=cfg.output_dir,
        pages_processed=len(loaded_pages),
        raw_paths=tuple(raw_paths),
        sidecar_paths=tuple(sidecar_paths),
        log_path=log_path,
    )
