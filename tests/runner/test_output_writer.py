"""Tests for the per-page raw markdown + JSON sidecar writer."""

from __future__ import annotations

import json
from pathlib import Path

from adaptive_inference.inference.types import InferenceResult
from adaptive_inference.runner.output_writer import write_page_result


def _make_result(page_id: str = "page_0001") -> InferenceResult:
    return InferenceResult(
        page_id=page_id,
        model_name="internvl2-2b",
        budget_name="low",
        prompt_id="table_parse_v1",
        raw_text="# page\n<table></table>\n",
        output_token_count=5,
        runtime_ms=1.23,
        started_at="2026-04-21T00:00:00+00:00",
        finished_at="2026-04-21T00:00:01+00:00",
    )


def test_writes_raw_and_sidecar(tmp_path: Path) -> None:
    result = _make_result()
    written = write_page_result(result, tmp_path)
    assert written.raw_path.exists()
    assert written.sidecar_path.exists()
    assert written.raw_path.read_text(encoding="utf-8") == result.raw_text


def test_sidecar_payload_fields(tmp_path: Path) -> None:
    result = _make_result()
    written = write_page_result(result, tmp_path)
    payload = json.loads(written.sidecar_path.read_text(encoding="utf-8"))
    assert payload["page_id"] == "page_0001"
    assert payload["model_name"] == "internvl2-2b"
    assert payload["budget_name"] == "low"
    assert payload["prompt_id"] == "table_parse_v1"
    assert payload["output_token_count"] == 5
    assert payload["runtime_ms"] == 1.23
    assert payload["raw_output_path"] == "raw/page_0001.md"
    assert payload["status"] == "ok"
    assert payload["started_at"].startswith("2026-")


def test_sidecar_json_is_deterministic(tmp_path: Path) -> None:
    result = _make_result()
    written = write_page_result(result, tmp_path)
    first = written.sidecar_path.read_bytes()
    # Overwrite with the same result and expect byte-identical bytes.
    written2 = write_page_result(result, tmp_path)
    assert first == written2.sidecar_path.read_bytes()


def test_separate_pages_go_to_separate_files(tmp_path: Path) -> None:
    write_page_result(_make_result("page_0001"), tmp_path)
    write_page_result(_make_result("page_0002"), tmp_path)
    raws = sorted(p.name for p in (tmp_path / "raw").iterdir())
    sides = sorted(p.name for p in (tmp_path / "pages").iterdir())
    assert raws == ["page_0001.md", "page_0002.md"]
    assert sides == ["page_0001.json", "page_0002.json"]
