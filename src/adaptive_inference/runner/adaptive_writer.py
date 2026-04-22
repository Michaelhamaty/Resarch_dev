"""Subfolder-aware artifact writer for the adaptive pipeline.

Phase 4 needs three distinct artifact families per run:

- ``first_pass/``  — always written; the low-budget parse and its sidecar.
- ``reparse/``     — written only when the verifier fires REPARSE.
- ``final/``       — always written; a copy of the chosen pass's raw .md
  plus a sidecar carrying verifier metadata (decision, failure codes,
  which pass was selected).

Keeping first-pass raw output on disk satisfies Contract 5 (raw
artifacts are preserved). The ``final/`` copy is a separate file (not a
symlink) so downstream tools always see a self-contained final artifact.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..inference.types import InferenceResult
from ..verifier.types import VerifierResult


@dataclass(frozen=True)
class WrittenPassArtifacts:
    raw_path: Path
    sidecar_path: Path


@dataclass(frozen=True)
class WrittenFinalArtifacts:
    raw_path: Path
    sidecar_path: Path


def write_pass_artifacts(
    result: InferenceResult,
    output_dir: str | Path,
    subfolder: str,
) -> WrittenPassArtifacts:
    """Write ``{subfolder}/raw/{page_id}.md`` + ``{subfolder}/pages/{page_id}.json``.

    Used for both the first pass (``subfolder="first_pass"``) and the
    optional reparse (``subfolder="reparse"``). The sidecar carries just
    the inference-level metadata — verifier metadata only appears on the
    final-selected sidecar.
    """

    base = Path(output_dir) / subfolder
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
        "raw_output_path": str(raw_path.relative_to(Path(output_dir))),
        "pass_kind": subfolder,
        "status": "ok",
    }
    payload = json.dumps(sidecar, sort_keys=True, indent=2)
    sidecar_path.write_text(payload + "\n", encoding="utf-8")

    return WrittenPassArtifacts(raw_path=raw_path, sidecar_path=sidecar_path)


def write_final_artifacts(
    chosen: InferenceResult,
    verifier_result: VerifierResult,
    output_dir: str | Path,
    *,
    final_output_source: str,
    reparse_triggered: bool,
) -> WrittenFinalArtifacts:
    """Write ``final/raw/{pid}.md`` and ``final/pages/{pid}.json``.

    The raw .md is a byte-for-byte copy of the chosen pass's raw output.
    The sidecar extends the Phase 2 sidecar shape with verifier metadata
    so analysis code can slice the final directory directly.
    """

    base = Path(output_dir) / "final"
    raw_dir = base / "raw"
    sidecar_dir = base / "pages"
    raw_dir.mkdir(parents=True, exist_ok=True)
    sidecar_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / f"{chosen.page_id}.md"
    sidecar_path = sidecar_dir / f"{chosen.page_id}.json"

    raw_path.write_text(chosen.raw_text, encoding="utf-8")

    sidecar = {
        "page_id": chosen.page_id,
        "model_name": chosen.model_name,
        "budget_name": chosen.budget_name,
        "prompt_id": chosen.prompt_id,
        "output_token_count": chosen.output_token_count,
        "runtime_ms": chosen.runtime_ms,
        "started_at": chosen.started_at,
        "finished_at": chosen.finished_at,
        "raw_output_path": str(raw_path.relative_to(Path(output_dir))),
        "final_output_source": final_output_source,
        "reparse_triggered": reparse_triggered,
        "verifier_decision": verifier_result.decision,
        "verifier_failure_codes": list(verifier_result.failure_codes),
        "predicted_table_count": verifier_result.predicted_table_count,
        "status": "ok",
    }
    payload = json.dumps(sidecar, sort_keys=True, indent=2)
    sidecar_path.write_text(payload + "\n", encoding="utf-8")

    return WrittenFinalArtifacts(raw_path=raw_path, sidecar_path=sidecar_path)
