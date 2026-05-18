"""Phase 7 input loaders.

Read-only helpers that ingest the Phase 6 manifest, per-system run logs,
the frozen calibration artifact, and the Phase 1 split manifests.

Two log schemas are supported because Phase 6 mixes runners:

- ``single_pass`` runners (fixed_2b_low, fixed_2b_matched, fixed_8b_matched)
  emit Phase 2 records: page_id, runtime_ms, output_token_count, etc.
- ``adaptive`` and ``adaptive_random`` runners emit Phase 4 records:
  page_id, total_runtime_ms, reparse_triggered, verifier_*, etc.

A small ``LoadedSystem`` container packages the manifest entry plus the
parsed log records so downstream modules can iterate without re-reading
disk. No mutation of inputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


LOG_FILENAME = "run.log.jsonl"


@dataclass(frozen=True)
class Phase6ManifestHeader:
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
    random_seeds: tuple[int, ...]


@dataclass(frozen=True)
class Phase6ManifestEntry:
    """Subset of Phase 6 manifest fields Phase 7 cares about.

    The runner field is what tells downstream code which log schema to
    expect. ``output_dir`` is stored relative to the repo root, exactly
    as Phase 6 wrote it.
    """

    system_id: str
    family: str
    runner: str
    status: str
    output_dir: str | None
    pages_processed: int | None
    reparse_count: int | None
    model_name: str | None
    adapter_kind: str | None
    budget_low_name: str | None
    budget_low_max_tiles: int | None
    budget_high_name: str | None
    budget_high_max_tiles: int | None
    budget_name: str | None
    budget_max_tiles: int | None
    seed: int | None
    random_probability: float | None
    notes: tuple[str, ...]
    raw: Mapping[str, object]


@dataclass(frozen=True)
class Phase6Manifest:
    schema_version: int
    header: Phase6ManifestHeader
    entries: tuple[Phase6ManifestEntry, ...]


@dataclass(frozen=True)
class LoadedSystem:
    """Per-system bundle: manifest entry + parsed JSONL records + log path."""

    entry: Phase6ManifestEntry
    log_path: Path
    records: tuple[Mapping[str, object], ...]


def load_phase6_manifest(path: str | Path) -> Phase6Manifest:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1:
        raise ValueError(
            f"Phase 6 manifest at {path} has schema_version="
            f"{raw.get('schema_version')!r}, expected 1"
        )

    header_raw = raw["header"]
    header = Phase6ManifestHeader(
        run_set_id=str(header_raw["run_set_id"]),
        generated_at=str(header_raw["generated_at"]),
        held_out_manifest_path=str(header_raw["held_out_manifest_path"]),
        held_out_manifest_sha256=str(header_raw["held_out_manifest_sha256"]),
        frozen_budgets_path=str(header_raw["frozen_budgets_path"]),
        frozen_budgets_sha256=str(header_raw["frozen_budgets_sha256"]),
        prompt_id=str(header_raw["prompt_id"]),
        prompt_version=int(header_raw["prompt_version"]),
        git_head=(
            str(header_raw["git_head"]) if header_raw.get("git_head") else None
        ),
        calibration_reparse_rate=float(header_raw["calibration_reparse_rate"]),
        calibration_reparse_rate_degenerate=bool(
            header_raw["calibration_reparse_rate_degenerate"]
        ),
        random_seeds=tuple(int(s) for s in header_raw.get("random_seeds", [])),
    )

    entries = tuple(_parse_entry(e) for e in raw.get("entries", []))
    return Phase6Manifest(
        schema_version=int(raw["schema_version"]),
        header=header,
        entries=entries,
    )


def _parse_entry(raw: Mapping[str, object]) -> Phase6ManifestEntry:
    notes = tuple(str(n) for n in (raw.get("notes") or []))
    return Phase6ManifestEntry(
        system_id=str(raw["system_id"]),
        family=str(raw["family"]),
        runner=str(raw["runner"]),
        status=str(raw["status"]),
        output_dir=str(raw["output_dir"]) if raw.get("output_dir") else None,
        pages_processed=_opt_int(raw.get("pages_processed")),
        reparse_count=_opt_int(raw.get("reparse_count")),
        model_name=_opt_str(raw.get("model_name")),
        adapter_kind=_opt_str(raw.get("adapter_kind")),
        budget_low_name=_opt_str(raw.get("budget_low_name")),
        budget_low_max_tiles=_opt_int(raw.get("budget_low_max_tiles")),
        budget_high_name=_opt_str(raw.get("budget_high_name")),
        budget_high_max_tiles=_opt_int(raw.get("budget_high_max_tiles")),
        budget_name=_opt_str(raw.get("budget_name")),
        budget_max_tiles=_opt_int(raw.get("budget_max_tiles")),
        seed=_opt_int(raw.get("seed")),
        random_probability=_opt_float(raw.get("random_probability")),
        notes=notes,
        raw=raw,
    )


def _opt_int(v: object) -> int | None:
    return int(v) if v is not None else None


def _opt_float(v: object) -> float | None:
    return float(v) if v is not None else None


def _opt_str(v: object) -> str | None:
    return str(v) if v is not None else None


def load_jsonl(path: str | Path) -> tuple[Mapping[str, object], ...]:
    """Parse every non-blank line in a JSONL file. Empty file -> empty tuple."""

    text = Path(path).read_text(encoding="utf-8")
    records: list[Mapping[str, object]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if not isinstance(rec, dict):
            raise ValueError(
                f"{path}: every JSONL line must be a JSON object, got {type(rec).__name__}"
            )
        records.append(rec)
    return tuple(records)


def load_loaded_systems(
    manifest: Phase6Manifest, *, repo_root: Path
) -> tuple[LoadedSystem, ...]:
    """Pair every "ok" manifest entry with its parsed run.log.jsonl.

    Skipped or failed entries are returned with an empty record tuple so
    the audit layer can still report on them.
    """

    out: list[LoadedSystem] = []
    for entry in manifest.entries:
        if entry.output_dir is None:
            log_path = repo_root / "_missing_" / LOG_FILENAME
            out.append(LoadedSystem(entry=entry, log_path=log_path, records=()))
            continue
        log_path = repo_root / entry.output_dir / LOG_FILENAME
        records = load_jsonl(log_path) if log_path.exists() else ()
        out.append(LoadedSystem(entry=entry, log_path=log_path, records=records))
    return tuple(out)


@dataclass(frozen=True)
class ScoringSummary:
    """Macro accuracy numbers loaded from a per-system page_scores.json.

    Source: ``scripts/analysis/score_run.py`` writes one of these per
    system to ``<scoring_summaries_root>/score_phase6_<system_id>/page_scores.json``.
    Phase 7 surfaces them in the results table when the file is present;
    when it is absent the per-system fields stay ``None``.
    """

    system_id: str
    page_scores_path: Path
    macro_cell_f1: float
    macro_text_similarity: float
    pages_total: int
    pages_with_gold: int
    pages_with_parse_error: int


SCORING_DIR_PREFIX = "score_phase6_"
PAGE_SCORES_FILENAME = "page_scores.json"


def load_scoring_summary(
    system_id: str, scoring_summaries_root: str | Path
) -> ScoringSummary | None:
    """Return the per-system scoring summary, or ``None`` if absent.

    Raises ``ValueError`` if the file exists but its schema is wrong —
    silent skips would mask a real upstream change.
    """

    path = (
        Path(scoring_summaries_root)
        / f"{SCORING_DIR_PREFIX}{system_id}"
        / PAGE_SCORES_FILENAME
    )
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "summary" not in raw:
        raise ValueError(
            f"{path}: expected {{'summary': {{...}}}} payload, got {type(raw).__name__}"
        )
    summary = raw["summary"]
    if not isinstance(summary, dict):
        raise ValueError(f"{path}: 'summary' must be a JSON object")
    try:
        return ScoringSummary(
            system_id=system_id,
            page_scores_path=path,
            macro_cell_f1=float(summary["macro_cell_f1"]),
            macro_text_similarity=float(summary["macro_text_similarity"]),
            pages_total=int(summary["pages_total"]),
            pages_with_gold=int(summary["pages_with_gold"]),
            pages_with_parse_error=int(summary["pages_with_parse_error"]),
        )
    except KeyError as exc:
        raise ValueError(f"{path}: scoring summary missing key {exc.args[0]!r}") from exc


def load_split_page_ids(path: str | Path) -> tuple[str, ...]:
    """Extract the ordered ``page_ids`` array from a Phase 1 split manifest."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    page_ids = raw.get("page_ids")
    if not isinstance(page_ids, list):
        raise ValueError(f"{path}: expected 'page_ids' list, got {type(page_ids).__name__}")
    return tuple(str(pid) for pid in page_ids)
