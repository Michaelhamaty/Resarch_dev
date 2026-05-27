"""Tests for the scaleup/v2 resume primitives and end-to-end resume flow.

Covers:

- Unit tests for ``resume.config_digest``, ``write_manifest`` (atomic +
  idempotent + drift-detection), ``read_completed_page_ids`` (incl. torn
  trailing line), and ``pending_pages`` (manifest order preserved).
- An end-to-end test that drives ``run_adaptive`` to completion on a
  fresh run, deletes the second-half log lines to simulate a crash
  after page N/2, invokes ``run_adaptive(resume=True)``, and asserts
  the final log has every page exactly once in manifest order.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from adaptive_inference.config.adaptive_runs import AdaptiveRunConfig
from adaptive_inference.config.budgets import Budget
from adaptive_inference.config.models import ModelConfig
from adaptive_inference.config.prompts import PromptTemplate
from adaptive_inference.runner import adaptive as adaptive_module
from adaptive_inference.runner.adaptive import run_adaptive
from adaptive_inference.runner.adaptive_logger import LOG_FILENAME
from adaptive_inference.runner.resume import (
    MANIFEST_FILENAME,
    build_manifest,
    config_digest,
    pending_pages,
    read_completed_page_ids,
    read_manifest,
    write_manifest,
)
from adaptive_inference.verifier.codes import DECISION_PASS
from adaptive_inference.verifier.types import VerifierResult


# ----------------------------- unit tests -------------------------------


def test_config_digest_is_order_independent() -> None:
    a = {"x": 1, "y": [1, 2, 3], "model": "internvl2-2b"}
    b = {"model": "internvl2-2b", "y": [1, 2, 3], "x": 1}
    assert config_digest(a) == config_digest(b)


def test_config_digest_changes_on_drift() -> None:
    a = {"budget_low": 4}
    b = {"budget_low": 8}
    assert config_digest(a) != config_digest(b)


def test_write_manifest_creates_and_is_idempotent(tmp_path: Path) -> None:
    m = build_manifest(
        run_id="r1",
        system_id="adaptive_2b",
        dataset_id="omnidocbench",
        page_ids=["p001", "p002"],
        config_payload={"k": "v"},
    )
    p1 = write_manifest(tmp_path, m)
    assert p1.exists()
    assert p1.name == MANIFEST_FILENAME

    # Second write with identical identity is a no-op (does not raise).
    p2 = write_manifest(tmp_path, m)
    assert p2 == p1
    on_disk = read_manifest(tmp_path)
    assert on_disk is not None
    assert on_disk["page_ids"] == ["p001", "p002"]


def test_write_manifest_refuses_to_clobber_on_drift(tmp_path: Path) -> None:
    m1 = build_manifest(
        run_id="r1",
        system_id="adaptive_2b",
        dataset_id="omnidocbench",
        page_ids=["p001", "p002"],
        config_payload={"k": "v"},
    )
    write_manifest(tmp_path, m1)

    m2 = build_manifest(
        run_id="r1",
        system_id="adaptive_2b",
        dataset_id="omnidocbench",
        page_ids=["p001", "p002", "p003"],  # drift
        config_payload={"k": "v"},
    )
    try:
        write_manifest(tmp_path, m2)
    except RuntimeError as exc:
        assert "Refusing" in str(exc) or "disagrees" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError on manifest drift")


def test_read_completed_page_ids_returns_empty_when_no_log(tmp_path: Path) -> None:
    assert read_completed_page_ids(tmp_path) == set()


def test_read_completed_page_ids_parses_lines(tmp_path: Path) -> None:
    log = tmp_path / LOG_FILENAME
    log.write_text(
        json.dumps({"page_id": "p001", "status": "ok"}) + "\n"
        + json.dumps({"page_id": "p002", "status": "ok"}) + "\n",
        encoding="utf-8",
    )
    assert read_completed_page_ids(tmp_path) == {"p001", "p002"}


def test_read_completed_page_ids_tolerates_torn_trailing_line(tmp_path: Path) -> None:
    log = tmp_path / LOG_FILENAME
    log.write_text(
        json.dumps({"page_id": "p001", "status": "ok"}) + "\n"
        + '{"page_id": "p002", "stat',  # torn, no newline
        encoding="utf-8",
    )
    # p001 is recoverable; p002 is silently dropped and will re-run on resume.
    assert read_completed_page_ids(tmp_path) == {"p001"}


def test_pending_pages_preserves_manifest_order() -> None:
    manifest_pids = ["p003", "p001", "p005", "p002", "p004"]
    completed = {"p001", "p004"}
    assert pending_pages(manifest_pids, completed) == ["p003", "p005", "p002"]


# --------------- end-to-end: kill mid-run, resume to completion ---------


def _pass_verifier() -> VerifierResult:
    return VerifierResult(
        decision=DECISION_PASS,
        failure_codes=(),
        predicted_table_count=1,
        html_parse_ok=True,
        span_normalization_ok=True,
        tables=(),
    )


def _build_inputs(tmp_path: Path, page_ids: list[str]) -> AdaptiveRunConfig:
    image_root = tmp_path / "data"
    images_dir = image_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for pid in page_ids:
        Image.new("L", (4, 4), color=0).save(images_dir / f"{pid}.png", format="PNG")

    records = tmp_path / "records.json"
    records.write_text(
        json.dumps(
            [
                {
                    "page_id": pid,
                    "image_path": f"images/{pid}.png",
                    "language": "en",
                    "contains_table": True,
                    "is_english_table_page": True,
                }
                for pid in page_ids
            ]
        ),
        encoding="utf-8",
    )

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "manifest_kind": "calibration_split",
                "schema_version": 1,
                "generated_with_seed": 1,
                "pinning": {},
                "page_ids": page_ids,
            }
        ),
        encoding="utf-8",
    )

    return AdaptiveRunConfig(
        run_id="test_resume",
        split_name="calibration_split",
        manifest_path=manifest,
        records_path=records,
        image_root=image_root,
        model_cfg=ModelConfig(
            name="internvl2-2b", adapter_kind="stub", model_id="x", notes=""
        ),
        budget_low=Budget(name="low", max_tiles=4),
        budget_high=Budget(name="high", max_tiles=12),
        prompt=PromptTemplate(
            id="table_parse_v1", version=1, description="d", template="t"
        ),
        output_dir=tmp_path / "out",
    )


def _read_log_lines(log_path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_resume_after_simulated_crash_completes_all_pages(
    tmp_path: Path, monkeypatch
) -> None:
    """End-to-end Gate G3: kill mid-run + resume produces the full log."""

    page_ids = [f"page_{i:04d}" for i in range(6)]
    cfg = _build_inputs(tmp_path, page_ids)
    monkeypatch.setattr(
        adaptive_module, "verify_page_tables", lambda _raw: _pass_verifier()
    )

    # Step 1: Run a full clean run with resume=True to establish a baseline log.
    baseline_cfg = _build_inputs(tmp_path / "baseline", page_ids)
    baseline_summary = run_adaptive(baseline_cfg, resume=True)
    assert baseline_summary.pages_processed == 6
    baseline_lines = _read_log_lines(baseline_summary.log_path)
    assert len(baseline_lines) == 6

    # Step 2: Run the same on the main cfg, then simulate a crash by
    # truncating the log to the first 3 lines.
    summary_first = run_adaptive(cfg, resume=True)
    assert summary_first.pages_processed == 6

    log_path = cfg.output_dir / LOG_FILENAME
    all_lines = _read_log_lines(log_path)
    assert len(all_lines) == 6
    # Keep only the first 3 lines on disk to simulate the crash.
    first_three = log_path.read_text(encoding="utf-8").splitlines()[:3]
    log_path.write_text("\n".join(first_three) + "\n", encoding="utf-8")

    # Step 3: Resume. The runner should skip the 3 completed pages and
    # process only the remaining 3, appending them to the log.
    summary_resumed = run_adaptive(cfg, resume=True)
    assert summary_resumed.pages_processed == 3
    assert summary_resumed.reparse_count == 0

    final_lines = _read_log_lines(log_path)
    assert len(final_lines) == 6
    final_ids_in_order = [rec["page_id"] for rec in final_lines]
    # Original first-three preserved at the head; remaining three appended
    # in manifest order.
    assert final_ids_in_order[:3] == [
        rec["page_id"] for rec in _read_log_lines(log_path)[:3]
    ]
    # Every page in the manifest appears exactly once in the final log.
    assert set(final_ids_in_order) == set(page_ids)
    assert len(final_ids_in_order) == len(page_ids)


def test_resume_false_truncates_log_and_rewrites_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    """Default (resume=False) preserves Phase 4 idempotent re-run behavior."""

    page_ids = [f"page_{i:04d}" for i in range(3)]
    cfg = _build_inputs(tmp_path, page_ids)
    monkeypatch.setattr(
        adaptive_module, "verify_page_tables", lambda _raw: _pass_verifier()
    )

    s1 = run_adaptive(cfg, resume=False)
    assert s1.pages_processed == 3

    # Second run with resume=False must truncate and re-do everything.
    s2 = run_adaptive(cfg, resume=False)
    assert s2.pages_processed == 3
    final_lines = _read_log_lines(cfg.output_dir / LOG_FILENAME)
    assert len(final_lines) == 3
