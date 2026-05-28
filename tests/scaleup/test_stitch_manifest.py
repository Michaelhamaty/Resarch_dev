"""Unit tests for the Stage 7 manifest stitcher.

The per-system sweep loop (crash-safety guardrail G1) leaves a manifest
listing only the last system; ``stitch_manifest.py`` merges the stashed
per-system manifests back into one complete manifest that Stage 9 can
read. These cover the pure merge logic plus an end-to-end main() pass
over synthetic stash files (no GPU, no real sweep).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "stitch_manifest",
    Path(__file__).resolve().parents[2] / "scripts" / "scaleup" / "stitch_manifest.py",
)
sm = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = sm
_SPEC.loader.exec_module(sm)  # type: ignore[union-attr]


def _manifest(system_id: str, *, generated_at: str, status: str = "ok") -> dict:
    """A minimal single-entry manifest payload as run_phase6 would write."""
    return {
        "schema_version": 1,
        "header": {
            "run_set_id": "scaleup_v2__fintabnet",
            "generated_at": generated_at,
            "held_out_manifest_path": "data/fintabnet/held_out.json",
            "held_out_manifest_sha256": "deadbeef",
            "frozen_budgets_path": "configs/calibration/frozen_budgets_v2_fintabnet.json",
            "frozen_budgets_sha256": "cafef00d",
            "prompt_id": "table_html_v1",
            "prompt_version": 1,
            "git_head": "abc123",
            "calibration_reparse_rate": 0.26,
            "calibration_reparse_rate_degenerate": False,
            "random_seeds": [0, 1, 2],
        },
        "entries": [{
            "system_id": system_id,
            "family": system_id,
            "runner": "single_pass",
            "status": status,
            "output_dir": f"outputs/scaleup_v2/fintabnet/{system_id}",
            "pages_processed": 100,
            "reparse_count": None,
            "model_name": "internvl2-2b",
            "adapter_kind": "real",
            "budget_low_name": None,
            "budget_low_max_tiles": None,
            "budget_high_name": None,
            "budget_high_max_tiles": None,
            "budget_name": "matched",
            "budget_max_tiles": 10,
            "seed": None,
            "random_probability": None,
            "random_probability_source": None,
            "started_at": "2026-05-28T00:00:00+00:00",
            "finished_at": "2026-05-28T00:30:00+00:00",
            "error": None,
            "reason": None,
            "notes": [],
        }],
    }


def test_merge_combines_entries_in_canonical_order():
    merged = sm.merge_manifests([
        _manifest("adaptive_2b", generated_at="2026-05-28T02:00:00+00:00"),
        _manifest("fixed_2b_low", generated_at="2026-05-28T00:00:00+00:00"),
        _manifest("fixed_8b_matched", generated_at="2026-05-28T03:00:00+00:00"),
    ])
    ids = [e["system_id"] for e in merged["entries"]]
    # SYSTEM_ORDER: fixed_2b_low < fixed_8b_matched < adaptive_2b
    assert ids == ["fixed_2b_low", "fixed_8b_matched", "adaptive_2b"]
    assert merged["schema_version"] == 1


def test_merge_header_from_latest_generated_at():
    merged = sm.merge_manifests([
        _manifest("fixed_2b_low", generated_at="2026-05-28T00:00:00+00:00"),
        _manifest("adaptive_2b", generated_at="2026-05-28T09:00:00+00:00"),
    ])
    assert merged["header"]["generated_at"] == "2026-05-28T09:00:00+00:00"


def test_merge_dedups_system_id_last_wins():
    merged = sm.merge_manifests([
        _manifest("adaptive_2b", generated_at="2026-05-28T00:00:00+00:00", status="failed"),
        _manifest("adaptive_2b", generated_at="2026-05-28T01:00:00+00:00", status="ok"),
    ])
    assert len(merged["entries"]) == 1
    assert merged["entries"][0]["status"] == "ok"


def test_merge_rejects_bad_schema():
    bad = _manifest("adaptive_2b", generated_at="2026-05-28T00:00:00+00:00")
    bad["schema_version"] = 2
    with pytest.raises(ValueError, match="schema_version"):
        sm.merge_manifests([bad])


def test_merge_empty_raises():
    with pytest.raises(ValueError, match="no manifests"):
        sm.merge_manifests([])


def test_load_stashed_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no stashed manifests"):
        sm.load_stashed_manifests(tmp_path / "nope")


def test_main_writes_complete_manifest(tmp_path):
    # Minimal sweep config so load_sweep_config resolves output_root.
    out_root = tmp_path / "outputs" / "scaleup_v2"
    config_path = tmp_path / "scaleup_v2.yaml"
    config_path.write_text(
        json.dumps({
            "run_set_id": "scaleup_v2",
            "datasets": {
                "fintabnet": {
                    "split_manifest_path": "x/held_out.json",
                    "records_path": "x/records.json",
                    "image_root": "x",
                    "frozen_budgets_path": "x/frozen.json",
                }
            },
            "model": {"config_path": "configs/model.yaml"},
            "prompt": {"config_path": "configs/prompt.yaml"},
            "output": {"root": str(out_root)},
            "random_escalation": {"seeds": [0, 1, 2]},
        }),
        encoding="utf-8",
    )

    stash_dir = out_root / "fintabnet" / sm.STASH_DIRNAME
    stash_dir.mkdir(parents=True)
    for i, sid in enumerate(("fixed_2b_low", "adaptive_2b", "fixed_8b_matched")):
        (stash_dir / f"{sid}.json").write_text(
            json.dumps(_manifest(sid, generated_at=f"2026-05-28T0{i}:00:00+00:00")),
            encoding="utf-8",
        )

    rc = sm.main(["--config", str(config_path), "--dataset", "fintabnet"])
    assert rc == 0

    written = json.loads(
        (out_root / "fintabnet" / "manifest.json").read_text(encoding="utf-8")
    )
    assert written["schema_version"] == 1
    assert {e["system_id"] for e in written["entries"]} == {
        "fixed_2b_low", "adaptive_2b", "fixed_8b_matched"
    }
    # write_manifest round-trips through the dataclasses cleanly.
    assert all("notes" in e for e in written["entries"])
