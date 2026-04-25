"""Unit tests for the Phase 6 run-manifest writer."""

from __future__ import annotations

import json
from pathlib import Path

from adaptive_inference.experiment.manifest import (
    MANIFEST_FILENAME,
    SCHEMA_VERSION,
    ManifestEntry,
    ManifestHeader,
    sha256_of_file,
    write_manifest,
)


def _header() -> ManifestHeader:
    return ManifestHeader(
        run_set_id="phase6_test",
        generated_at="2026-04-23T00:00:00+00:00",
        held_out_manifest_path="data/splits/held_out_eval_split.json",
        held_out_manifest_sha256="deadbeef",
        frozen_budgets_path="configs/calibration/frozen_budgets.json",
        frozen_budgets_sha256="cafebabe",
        prompt_id="table_parse_v1",
        prompt_version=1,
        git_head=None,
        calibration_reparse_rate=0.0,
        calibration_reparse_rate_degenerate=True,
        random_seeds=[0, 1, 2],
    )


def test_manifest_written_to_output_root(tmp_path: Path) -> None:
    entries = [
        ManifestEntry(
            system_id="adaptive_2b",
            family="adaptive_2b",
            runner="adaptive",
            status="ok",
            output_dir=str(tmp_path / "adaptive_2b"),
            pages_processed=9,
            reparse_count=0,
        ),
    ]
    path = write_manifest(output_root=tmp_path, header=_header(), entries=entries)
    assert path.name == MANIFEST_FILENAME

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == SCHEMA_VERSION
    assert data["header"]["run_set_id"] == "phase6_test"
    assert len(data["entries"]) == 1
    assert data["entries"][0]["system_id"] == "adaptive_2b"
    assert data["entries"][0]["status"] == "ok"


def test_failed_entry_records_error(tmp_path: Path) -> None:
    entries = [
        ManifestEntry(
            system_id="fixed_8b_matched",
            family="fixed_8b_matched",
            runner="single_pass",
            status="failed",
            error="RuntimeError: kaboom",
        ),
    ]
    path = write_manifest(output_root=tmp_path, header=_header(), entries=entries)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["entries"][0]["status"] == "failed"
    assert "kaboom" in data["entries"][0]["error"]


def test_skipped_8b_entry_records_reason(tmp_path: Path) -> None:
    entries = [
        ManifestEntry(
            system_id="fixed_8b_matched",
            family="fixed_8b_matched",
            runner="single_pass",
            status="skipped_stub_8b",
            reason="stubbed 8B, flag not passed",
        ),
    ]
    path = write_manifest(output_root=tmp_path, header=_header(), entries=entries)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["entries"][0]["status"] == "skipped_stub_8b"
    assert "flag not passed" in data["entries"][0]["reason"]


def test_sha256_of_file_is_stable(tmp_path: Path) -> None:
    f = tmp_path / "x.txt"
    f.write_text("hello", encoding="utf-8")
    assert sha256_of_file(f) == sha256_of_file(f)
    assert len(sha256_of_file(f)) == 64
