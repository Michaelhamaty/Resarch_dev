"""Append-only JSONL logger for adaptive (Phase 4) per-page records.

Reuses the Phase 2 filename (``run.log.jsonl``) with an extended schema
— see docs/runbooks/phase4_adaptive.md. New fields (``reparse_triggered``,
``verifier_*``, split runtime/token fields, artifact paths) sit alongside
the base identity fields. Any downstream reader must tolerate unknown
keys, per Phase 2's contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..inference.types import InferenceResult
from ..verifier.types import VerifierResult


LOG_FILENAME = "run.log.jsonl"


@dataclass(frozen=True)
class AdaptivePageLog:
    """Container for the Phase 4 per-page log fields.

    Collected by the orchestrator and handed to ``append_adaptive_page_log``
    so the log schema lives in exactly one place.
    """

    run_id: str
    split: str
    model_name: str
    prompt_id: str
    budget_low_name: str
    budget_high_name: str
    first_pass: InferenceResult
    verifier: VerifierResult
    verifier_runtime_ms: float
    reparse: InferenceResult | None
    reparse_runtime_ms: float | None
    first_pass_raw_rel: str
    reparse_raw_rel: str | None
    final_raw_rel: str
    final_output_source: str
    reparse_triggered: bool
    total_runtime_ms: float
    status: str = "ok"


def append_adaptive_page_log(entry: AdaptivePageLog, output_dir: str | Path) -> Path:
    """Append one JSON line describing the full adaptive transaction for a page."""

    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    log_path = base / LOG_FILENAME

    record = {
        "run_id": entry.run_id,
        "split": entry.split,
        "page_id": entry.first_pass.page_id,
        "model_name": entry.model_name,
        "prompt_id": entry.prompt_id,
        "budget_low": entry.budget_low_name,
        "budget_high": entry.budget_high_name,
        "reparse_triggered": entry.reparse_triggered,
        "verifier_decision": entry.verifier.decision,
        "verifier_failure_codes": list(entry.verifier.failure_codes),
        "predicted_table_count": entry.verifier.predicted_table_count,
        "first_pass_output_tokens": entry.first_pass.output_token_count,
        "reparse_output_tokens": (
            entry.reparse.output_token_count if entry.reparse is not None else None
        ),
        "first_pass_runtime_ms": entry.first_pass.runtime_ms,
        "verifier_runtime_ms": entry.verifier_runtime_ms,
        "reparse_runtime_ms": entry.reparse_runtime_ms,
        "total_runtime_ms": entry.total_runtime_ms,
        "first_pass_raw_path": entry.first_pass_raw_rel,
        "reparse_raw_path": entry.reparse_raw_rel,
        "final_raw_path": entry.final_raw_rel,
        "final_output_source": entry.final_output_source,
        "status": entry.status,
    }
    line = json.dumps(record, sort_keys=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    return log_path


def reset_adaptive_log(output_dir: str | Path) -> None:
    """Delete the adaptive log if present. Used before re-running a run_id."""

    log_path = Path(output_dir) / LOG_FILENAME
    if log_path.exists():
        log_path.unlink()
