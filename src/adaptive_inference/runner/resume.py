"""Manifest + resume utilities for long sweeps (scaleup/v2, Stage 3).

A "run" produces two on-disk artifacts that together support safe
mid-run interruption:

1. ``manifest.json`` — written at the start of the run. Lists the
   deterministic, ordered ``page_ids`` that the run intends to process,
   plus run identity metadata (run_id, started_at, config digest, git
   SHA when available). The manifest is immutable for a given run:
   re-invoking ``write_manifest`` with the same content is a no-op;
   re-invoking with different content raises so we never silently
   change what a partially-completed run is supposed to do.

2. ``run.log.jsonl`` — written one fsync'd line per page by
   ``adaptive_logger.append_adaptive_page_log``. The presence of a
   page_id in this file means the page has fully completed (artifacts
   written, log line fsync'd).

The two pieces compose: ``pending_pages(manifest_pages, completed)``
returns the manifest-ordered subset that has not yet completed, so the
orchestrator can resume from where it stopped.

Durability properties this module assumes:

- The logger fsyncs after each line. A torn final line is possible only
  if the OS crashes mid-write; we tolerate it by skipping unparseable
  trailing JSON lines.
- The manifest is written-then-renamed atomically, so a partially
  written manifest cannot be read as valid.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path

from .adaptive_logger import LOG_FILENAME


MANIFEST_FILENAME = "manifest.json"


def _utc_now_isoformat() -> str:
    """Return current UTC time as an ISO-8601 string (seconds precision)."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _try_git_sha(repo_root: Path | None = None) -> str | None:
    """Best-effort current commit SHA. Returns None if git is unavailable."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root) if repo_root else None,
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        return result.stdout.strip() or None
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None


def config_digest(payload: object) -> str:
    """Stable short digest of a JSON-serializable config payload.

    Used to detect config drift between an initial run and a --resume
    invocation. Anything JSON-serializable works; ordering of mappings
    is normalized via ``sort_keys`` so equivalent dicts hash equal.
    """

    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def build_manifest(
    *,
    run_id: str,
    system_id: str,
    dataset_id: str,
    page_ids: Sequence[str],
    config_payload: object | None = None,
    repo_root: Path | None = None,
) -> dict:
    """Assemble a manifest dict ready to hand to ``write_manifest``.

    ``page_ids`` is preserved as given (the caller controls ordering;
    we do not re-sort, because ordering may be experimentally meaningful).
    """

    return {
        "schema_version": 1,
        "run_id": run_id,
        "system_id": system_id,
        "dataset_id": dataset_id,
        "page_ids": list(page_ids),
        "page_count": len(page_ids),
        "started_at": _utc_now_isoformat(),
        "config_digest": (
            config_digest(config_payload) if config_payload is not None else None
        ),
        "git_sha": _try_git_sha(repo_root),
    }


def write_manifest(run_dir: str | Path, manifest: dict) -> Path:
    """Write the manifest atomically.

    If the manifest already exists with identical *page_ids* and
    *config_digest*, this is a no-op (caller is safely resuming).
    If it exists with different content, raise — the caller must
    explicitly start a new run dir rather than silently change what a
    half-completed run is supposed to do.
    """

    base = Path(run_dir)
    base.mkdir(parents=True, exist_ok=True)
    target = base / MANIFEST_FILENAME

    if target.exists():
        try:
            existing = json.loads(target.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Existing manifest at {target} is unreadable: {exc}. "
                "Delete the run dir or pick a new one."
            ) from exc

        # Treat page_ids + config_digest as the identity. started_at
        # and git_sha are allowed to differ (re-runs after restart).
        if (
            existing.get("page_ids") == manifest.get("page_ids")
            and existing.get("config_digest") == manifest.get("config_digest")
            and existing.get("dataset_id") == manifest.get("dataset_id")
            and existing.get("system_id") == manifest.get("system_id")
        ):
            return target
        raise RuntimeError(
            f"Manifest at {target} disagrees with new manifest "
            "(page_ids, config_digest, dataset_id, or system_id differ). "
            "Refusing to overwrite a partially-completed run. Either resume "
            "with the same config or use a fresh output dir."
        )

    # Atomic write: write to .tmp, fsync, rename.
    tmp_path = target.with_suffix(target.suffix + ".tmp")
    payload = json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    with tmp_path.open("w", encoding="utf-8") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, target)
    return target


def read_manifest(run_dir: str | Path) -> dict | None:
    """Read the manifest if it exists, else None."""

    path = Path(run_dir) / MANIFEST_FILENAME
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_completed_page_ids(run_dir: str | Path) -> set[str]:
    """Return the set of page_ids that have a fully-written log line.

    Tolerant of a torn trailing line (skipped silently): with fsync
    after each line, the only way to see a malformed line is an OS-level
    crash between write() and fsync(), in which case the safe thing is
    to re-run that page on resume rather than crash the resume itself.
    """

    log_path = Path(run_dir) / LOG_FILENAME
    if not log_path.exists():
        return set()

    completed: set[str] = set()
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # Torn trailing line; ignore and let the page re-run.
                continue
            page_id = record.get("page_id")
            if isinstance(page_id, str) and page_id:
                completed.add(page_id)
    return completed


def pending_pages(
    manifest_page_ids: Iterable[str],
    completed_page_ids: set[str],
) -> list[str]:
    """Return manifest-ordered page_ids not yet in the completed set."""

    return [pid for pid in manifest_page_ids if pid not in completed_page_ids]
