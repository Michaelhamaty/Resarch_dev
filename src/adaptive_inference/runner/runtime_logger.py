"""Append-only JSONL runtime logger for single-pass runs.

One JSON object per inference call per page lands in
``{output_dir}/run.log.jsonl``. The schema is intentionally Phase-2-scoped
— no verifier/reparse/total-runtime fields yet. Later phases *extend* the
schema by adding fields; existing readers must tolerate unknown keys.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..inference.types import InferenceResult


LOG_FILENAME = "run.log.jsonl"


def append_page_log(
    result: InferenceResult,
    *,
    run_id: str,
    split: str,
    raw_output_path: str | Path,
    output_dir: str | Path,
    status: str = "ok",
) -> Path:
    """Append one JSON line describing this page's inference call."""

    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    log_path = base / LOG_FILENAME

    record = {
        "run_id": run_id,
        "page_id": result.page_id,
        "split": split,
        "model_name": result.model_name,
        "budget_name": result.budget_name,
        "prompt_id": result.prompt_id,
        "runtime_ms": result.runtime_ms,
        "output_token_count": result.output_token_count,
        "raw_output_path": str(raw_output_path),
        "started_at": result.started_at,
        "finished_at": result.finished_at,
        "status": status,
    }
    line = json.dumps(record, sort_keys=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    return log_path


def reset_log(output_dir: str | Path) -> None:
    """Delete the log file if it exists. Used before re-running a run_id."""

    log_path = Path(output_dir) / LOG_FILENAME
    if log_path.exists():
        log_path.unlink()
