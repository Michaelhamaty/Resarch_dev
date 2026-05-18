"""Phase 7 orchestrator: produce all analysis artifacts under outputs/analysis/.

Reads the Phase 6 manifest (and its referenced files), builds per-system
results, cost summary, reparse summary, qualitative joins, and an
integration audit, then writes a small set of JSON / JSONL files plus a
top-level ``analysis_manifest.json`` that pins provenance.

This module is intentionally read-only with respect to every Phase 1–6
artifact. The frozen calibration file is opened once and never written
to. No model is loaded.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ..calibration.artifact import load_frozen_budgets
from ..experiment.manifest import git_head_or_none, iso_now, sha256_of_file
from .audit import AuditReport, run_audit
from .audit import to_dict as audit_to_dict
from .config import Phase7Config
from .cost import build_cost_summary
from .cost import to_dict as cost_to_dict
from .loaders import (
    load_loaded_systems,
    load_phase6_manifest,
    load_scoring_summary,
    load_split_page_ids,
)
from .qualitative import build_page_rows, iter_jsonl_rows
from .reparse import build_reparse_summary
from .reparse import to_dict as reparse_to_dict
from .results import summarize_system


ANALYSIS_MANIFEST = "analysis_manifest.json"
RESULTS_TABLE = "results_table.json"
COST_SUMMARY = "cost_summary.json"
REPARSE_SUMMARY = "reparse_summary.json"
QUALITATIVE = "qualitative_examples.jsonl"
AUDIT_REPORT = "audit_report.json"


@dataclass(frozen=True)
class Phase7Result:
    output_dir: Path
    audit: AuditReport
    artifact_paths: tuple[Path, ...]


def run_phase7(cfg: Phase7Config, *, repo_root: Path | None = None) -> Phase7Result:
    """Execute the Phase 7 analysis pipeline end-to-end."""

    root = (repo_root or Path.cwd()).resolve()

    manifest = load_phase6_manifest(cfg.phase6_manifest_path)
    frozen = load_frozen_budgets(cfg.frozen_budgets_path)
    systems = load_loaded_systems(manifest, repo_root=root)
    held_out_ids = load_split_page_ids(cfg.held_out_split_path)

    # Per-system results. When a scoring_summaries_root is configured,
    # merge in per-system macro accuracy numbers from the standalone
    # scorer's page_scores.json. Missing files are silently skipped so
    # partial scoring (e.g. some systems still pending) remains valid.
    scoring_root = cfg.scoring_summaries_root
    results = tuple(
        summarize_system(
            s,
            scoring=(
                load_scoring_summary(s.entry.system_id, scoring_root)
                if scoring_root is not None
                else None
            ),
        )
        for s in systems
    )
    cost = build_cost_summary(systems, frozen)
    reparse = build_reparse_summary(systems)
    pages = build_page_rows(systems, held_out_ids)

    audit = run_audit(
        manifest=manifest,
        systems=systems,
        frozen=frozen,
        repo_root=root,
        calibration_split_path=cfg.calibration_split_path,
        held_out_split_path=cfg.held_out_split_path,
        frozen_budgets_path=cfg.frozen_budgets_path,
        splits_are_identical_acknowledged=cfg.splits_are_identical_acknowledged,
    )

    out_dir = cfg.output_root / manifest.header.run_set_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Results table
    results_path = out_dir / RESULTS_TABLE
    _write_json(
        results_path,
        {
            "run_set_id": manifest.header.run_set_id,
            "accuracy_status": _accuracy_status(frozen),
            "systems": [_system_result_to_dict(r) for r in results],
        },
    )

    # Cost summary
    cost_path = out_dir / COST_SUMMARY
    _write_json(cost_path, cost_to_dict(cost))

    # Reparse summary
    reparse_path = out_dir / REPARSE_SUMMARY
    _write_json(reparse_path, reparse_to_dict(reparse))

    # Qualitative JSONL
    qualitative_path = out_dir / QUALITATIVE
    with qualitative_path.open("w", encoding="utf-8") as f:
        for row in iter_jsonl_rows(pages):
            f.write(json.dumps(row, sort_keys=True) + "\n")

    # Audit report
    audit_path = out_dir / AUDIT_REPORT
    _write_json(audit_path, audit_to_dict(audit))

    # Top-level manifest pinning provenance
    analysis_manifest_path = out_dir / ANALYSIS_MANIFEST
    _write_json(
        analysis_manifest_path,
        {
            "schema_version": 1,
            "phase7_run_id": manifest.header.run_set_id,
            "generated_at": iso_now(),
            "phase6_manifest_path": str(cfg.phase6_manifest_path),
            "phase6_manifest_sha256": sha256_of_file(cfg.phase6_manifest_path),
            "frozen_budgets_path": str(cfg.frozen_budgets_path),
            "frozen_budgets_sha256": sha256_of_file(cfg.frozen_budgets_path),
            "held_out_split_path": str(cfg.held_out_split_path),
            "calibration_split_path": str(cfg.calibration_split_path),
            "git_head": git_head_or_none(root),
            "accuracy_status": _accuracy_status(frozen),
            "audit_any_failed": audit.any_failed,
            "audit_any_warned": audit.any_warned,
            "artifacts": [
                RESULTS_TABLE,
                COST_SUMMARY,
                REPARSE_SUMMARY,
                QUALITATIVE,
                AUDIT_REPORT,
            ],
        },
    )

    return Phase7Result(
        output_dir=out_dir,
        audit=audit,
        artifact_paths=(
            results_path,
            cost_path,
            reparse_path,
            qualitative_path,
            audit_path,
            analysis_manifest_path,
        ),
    )


def _accuracy_status(frozen) -> str:
    """Constant for stub adapters; documented in audit and README."""

    if any(
        b.adapter_kind == "stub"
        for b in (frozen.b_low, frozen.b_high, frozen.b_fix_2b, frozen.b_fix_8b)
    ):
        return "not_applicable_stub_adapters"
    return "scorer_not_bundled"


def _system_result_to_dict(r) -> dict:
    d = asdict(r)
    # asdict turns Mapping fields back into dicts already, but Counter
    # doesn't survive — we already passed plain dicts in results.py.
    return d


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
