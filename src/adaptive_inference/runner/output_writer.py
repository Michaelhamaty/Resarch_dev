"""Per-page artifact writer: raw markdown + JSON sidecar.

Contract 5 ("Raw artifacts are preserved") requires the model's raw
output to land on disk untouched. We split the per-page payload into:

- ``raw/{page_id}.md`` — the model's raw text, byte-for-byte.
- ``pages/{page_id}.json`` — a sidecar with all metadata the evaluator
  adapter (a later phase) needs, including a relative pointer to the
  raw file.

Keeping these split lets downstream tools read metadata without pulling
raw text into memory, and lets us diff raw outputs across runs easily.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..inference.types import InferenceResult


@dataclass(frozen=True)
class WrittenPage:
    raw_path: Path
    sidecar_path: Path


def write_page_result(
    result: InferenceResult,
    output_dir: str | Path,
) -> WrittenPage:
    """Write ``raw/{page_id}.md`` and ``pages/{page_id}.json`` under ``output_dir``."""

    base = Path(output_dir)
    raw_dir = base / "raw"
    sidecar_dir = base / "pages"
    raw_dir.mkdir(parents=True, exist_ok=True)
    sidecar_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / f"{result.page_id}.md"
    sidecar_path = sidecar_dir / f"{result.page_id}.json"

    raw_path.write_text(result.raw_text, encoding="utf-8")

    sidecar = {
        "page_id": result.page_id,
        "model_name": result.model_name,
        "budget_name": result.budget_name,
        "prompt_id": result.prompt_id,
        "output_token_count": result.output_token_count,
        "runtime_ms": result.runtime_ms,
        "started_at": result.started_at,
        "finished_at": result.finished_at,
        "raw_output_path": str(raw_path.relative_to(base)),
        "status": "ok",
    }
    payload = json.dumps(sidecar, sort_keys=True, indent=2)
    sidecar_path.write_text(payload + "\n", encoding="utf-8")

    return WrittenPage(raw_path=raw_path, sidecar_path=sidecar_path)
