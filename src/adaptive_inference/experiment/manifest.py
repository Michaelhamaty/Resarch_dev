"""Phase 6 run manifest: the JSON index Phase 7 consumes.

One file per Phase 6 execution, written at
``<output_root>/manifest.json``. It records, for every system attempted:

- system_id, family, runner, status
- output_dir (relative to the manifest's own dir so the manifest is
  portable if the whole tree is moved)
- pages_processed, reparse_count (where applicable)
- budget names + tile counts, model name, adapter_kind
- seed and probability (random escalation only)
- started_at / finished_at (ISO-8601 UTC)
- error message (failed) or reason (skipped)

Plus top-level pinning fields so Phase 7 can confirm it is reading the
right universe and budgets: the held-out manifest path + sha256, the
frozen-budgets path + sha256, and git HEAD if available.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


MANIFEST_FILENAME = "manifest.json"
SCHEMA_VERSION = 1


@dataclass
class ManifestEntry:
    """One row in the manifest — either a successful, skipped, or failed run."""

    system_id: str
    family: str
    runner: str
    status: str  # "ok" | "failed" | "skipped_stub_8b"
    output_dir: str | None = None
    pages_processed: int | None = None
    reparse_count: int | None = None
    model_name: str | None = None
    adapter_kind: str | None = None
    budget_low_name: str | None = None
    budget_low_max_tiles: int | None = None
    budget_high_name: str | None = None
    budget_high_max_tiles: int | None = None
    budget_name: str | None = None
    budget_max_tiles: int | None = None
    seed: int | None = None
    random_probability: float | None = None
    random_probability_source: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    reason: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class ManifestHeader:
    """Top-level provenance block written once per manifest."""

    run_set_id: str
    generated_at: str
    held_out_manifest_path: str
    held_out_manifest_sha256: str
    frozen_budgets_path: str
    frozen_budgets_sha256: str
    prompt_id: str
    prompt_version: int
    git_head: str | None
    calibration_reparse_rate: float
    calibration_reparse_rate_degenerate: bool
    random_seeds: list[int]


def write_manifest(
    *,
    output_root: Path,
    header: ManifestHeader,
    entries: list[ManifestEntry],
) -> Path:
    """Write the full manifest JSON to ``output_root / manifest.json``."""

    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / MANIFEST_FILENAME
    payload = {
        "schema_version": SCHEMA_VERSION,
        "header": asdict(header),
        "entries": [asdict(e) for e in entries],
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_of_file(path: str | Path) -> str:
    """Return the hex SHA-256 of a file — used to pin manifests and artifacts."""

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head_or_none(cwd: str | Path) -> str | None:
    """Best-effort ``git rev-parse HEAD``; returns ``None`` if unavailable."""

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 — best-effort provenance only
        return None
