"""Unit tests for the Stage 7 manifest *rebuild* recovery tool.

When the per-system sweep loop never stashes per-system manifest
snapshots, ``stitch_manifest.py`` has nothing to merge and the surviving
``<dataset>/manifest.json`` lists only the last system. ``rebuild_manifest.py``
reconstructs the full entries list from the intact ``run.log.jsonl`` files,
reusing the same frozen-budget / system-spec helpers ``run_phase6`` uses.

These cover the per-runner reconstruction logic plus an end-to-end
``rebuild()`` pass over a synthetic output tree (no GPU, no real sweep).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "rebuild_manifest",
    _REPO_ROOT / "scripts" / "scaleup" / "rebuild_manifest.py",
)
rb = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = rb
_SPEC.loader.exec_module(rb)  # type: ignore[union-attr]

from run_sweep import DatasetSpec  # noqa: E402  (added to path by rebuild_manifest)

DATASET = "fintabnet"


def _frozen_budgets_json() -> dict:
    """Mirror configs/calibration/frozen_budgets_v2_fintabnet.json shape."""
    return {
        "schema_version": 1,
        "run_id": "test",
        "generated_at": "2026-05-29T00:00:00+00:00",
        "pinning": {
            "calibration_config_path": "x.yaml",
            "calibration_split_manifest": "x.json",
            "calibration_split_sha256": "deadbeef",
            "cost_unit": "max_tiles",
            "matched_cost_tolerance": 0.1,
        },
        "budgets": {
            "B_low": {"name": "B_low", "max_tiles": 6,
                      "model_name": "internvl2-2b", "adapter_kind": "internvl2"},
            "B_high": {"name": "B_high", "max_tiles": 16,
                       "model_name": "internvl2-2b", "adapter_kind": "internvl2"},
            "B_fix_2B": {"name": "B_fix_2B", "max_tiles": 10,
                         "model_name": "internvl2-2b", "adapter_kind": "internvl2"},
            "B_fix_8B": {"name": "B_fix_8B", "max_tiles": 10,
                         "model_name": "internvl2-8b", "adapter_kind": "internvl2"},
        },
        "selection": {
            "adaptive": {"low_max_tiles": 6, "high_max_tiles": 16,
                         "measured_cost_tiles": 10.48, "target_cost_tiles": 10.0},
            "fixed_2b": {"max_tiles": 10, "measured_cost_tiles": 10.0,
                         "target_cost_tiles": 10.48, "within_tolerance": True},
            "fixed_8b": {"max_tiles": 10, "measured_cost_tiles": 10.0,
                         "target_cost_tiles": 10.48, "within_tolerance": True},
        },
    }


def _header() -> dict:
    return {
        "run_set_id": "scaleup_v2_main_sweep__fintabnet",
        "generated_at": "2026-05-29T23:49:54+00:00",
        "held_out_manifest_path": "data/splits/scaleup_v2/fintabnet/held_out.json",
        "held_out_manifest_sha256": "b61e8662",
        "frozen_budgets_path": "configs/calibration/frozen_budgets_v2_fintabnet.json",
        "frozen_budgets_sha256": "55d66c29",
        "prompt_id": "table_parse_v2",
        "prompt_version": 2,
        "git_head": "cae3f7ef",
        "calibration_reparse_rate": 0.28,
        "calibration_reparse_rate_degenerate": False,
        "random_seeds": [0, 1, 2],
    }


def _surviving_8b_entry() -> dict:
    return {
        "system_id": "fixed_8b_matched", "family": "fixed_8b_matched",
        "runner": "single_pass", "status": "ok",
        "output_dir": "outputs/scaleup_v2/fintabnet/fixed_8b_matched",
        "pages_processed": 150, "reparse_count": None,
        "model_name": "internvl2-8b", "adapter_kind": "internvl2",
        "budget_low_name": None, "budget_low_max_tiles": None,
        "budget_high_name": None, "budget_high_max_tiles": None,
        "budget_name": "B_fix_8B", "budget_max_tiles": 10,
        "seed": None, "random_probability": None,
        "random_probability_source": None,
        "started_at": "2026-05-29T22:07:25+00:00",
        "finished_at": "2026-05-29T23:49:54+00:00",
        "error": None, "reason": None, "notes": [],
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )


def _setup_tree(root: Path, *, n_pages: int = 5) -> SimpleNamespace:
    """Build a synthetic output tree + frozen budgets and return a sweep_cfg."""
    output_root = root / "outputs" / "scaleup_v2"
    ds_root = output_root / DATASET

    # single-pass logs (have started_at/finished_at)
    for sysid, bname in [("fixed_2b_low", "B_low"), ("fixed_2b_matched", "B_fix_2B")]:
        _write_jsonl(
            ds_root / sysid / "run.log.jsonl",
            [{"budget_name": bname, "model_name": "internvl2-2b", "status": "ok",
              "page_id": f"p{i}",
              "started_at": f"2026-05-29T1{i}:00:00+00:00",
              "finished_at": f"2026-05-29T1{i}:00:30+00:00"} for i in range(n_pages)],
        )
    # adaptive log (no top-level timestamps; reparse_triggered per page)
    _write_jsonl(
        ds_root / "adaptive_2b" / "run.log.jsonl",
        [{"page_id": f"p{i}", "status": "ok", "reparse_triggered": i % 2 == 0,
          "model_name": "internvl2-2b"} for i in range(n_pages)],
    )
    # random logs (reparse_triggered + seed)
    for seed in (0, 1, 2):
        _write_jsonl(
            ds_root / f"random_2b_seed{seed}" / "run.log.jsonl",
            [{"page_id": f"p{i}", "status": "ok",
              "reparse_triggered": i == 0, "random_seed": seed,
              "random_probability": 0.28, "model_name": "internvl2-2b"}
             for i in range(n_pages)],
        )
    # surviving partial manifest: header + only the 8B entry
    (ds_root).mkdir(parents=True, exist_ok=True)
    (ds_root / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "header": _header(),
                    "entries": [_surviving_8b_entry()]}),
        encoding="utf-8",
    )
    # frozen budgets file
    fb_path = root / "frozen.json"
    fb_path.write_text(json.dumps(_frozen_budgets_json()), encoding="utf-8")

    dataset_spec = DatasetSpec(
        name=DATASET,
        split_manifest_path=root / "held_out.json",
        records_path=root / "records.json",
        image_root=root,
        frozen_budgets_path=fb_path,
    )
    return SimpleNamespace(
        datasets=(dataset_spec,), output_root=output_root, random_seeds=(0, 1, 2)
    )


def test_rebuild_produces_all_seven_systems(tmp_path: Path) -> None:
    cfg = _setup_tree(tmp_path)
    header, entries = rb.rebuild(sweep_cfg=cfg, dataset=DATASET, repo_root=tmp_path)

    by_id = {e.system_id: e for e in entries}
    assert set(by_id) == {
        "adaptive_2b", "fixed_2b_low", "fixed_2b_matched",
        "random_2b_seed0", "random_2b_seed1", "random_2b_seed2",
        "fixed_8b_matched",
    }
    assert all(e.status == "ok" for e in entries)
    # the 6 reconstructed systems reflect the synthetic 5-page logs; the
    # reused 8B entry keeps its authoritative original page count (150).
    assert all(
        e.pages_processed == 5
        for e in entries
        if e.system_id != "fixed_8b_matched"
    )
    assert by_id["fixed_8b_matched"].pages_processed == 150
    # header reused verbatim
    assert header.git_head == "cae3f7ef"
    assert header.calibration_reparse_rate == 0.28


def test_fixed_single_pass_budget_and_times(tmp_path: Path) -> None:
    cfg = _setup_tree(tmp_path)
    _, entries = rb.rebuild(sweep_cfg=cfg, dataset=DATASET, repo_root=tmp_path)
    low = next(e for e in entries if e.system_id == "fixed_2b_low")
    assert low.runner == "single_pass"
    assert low.budget_name == "B_low"
    assert low.budget_max_tiles == 6
    assert low.adapter_kind == "internvl2"
    assert low.started_at == "2026-05-29T10:00:00+00:00"
    assert low.finished_at == "2026-05-29T14:00:30+00:00"
    assert rb.RECONSTRUCTION_NOTE in low.notes


def test_adaptive_reparse_count(tmp_path: Path) -> None:
    cfg = _setup_tree(tmp_path)
    _, entries = rb.rebuild(sweep_cfg=cfg, dataset=DATASET, repo_root=tmp_path)
    adaptive = next(e for e in entries if e.system_id == "adaptive_2b")
    # pages 0,2,4 triggered reparse -> 3
    assert adaptive.reparse_count == 3
    assert adaptive.budget_low_name == "B_low"
    assert adaptive.budget_low_max_tiles == 6
    assert adaptive.budget_high_name == "B_high"
    assert adaptive.budget_high_max_tiles == 16


def test_random_carries_seed_and_probability(tmp_path: Path) -> None:
    cfg = _setup_tree(tmp_path)
    _, entries = rb.rebuild(sweep_cfg=cfg, dataset=DATASET, repo_root=tmp_path)
    r1 = next(e for e in entries if e.system_id == "random_2b_seed1")
    assert r1.runner == "adaptive_random"
    assert r1.seed == 1
    assert abs(r1.random_probability - 0.28) < 1e-9
    assert r1.random_probability_source == "phase5_frozen_artifact.adaptive_selection"
    assert r1.reparse_count == 1  # only page 0


def test_surviving_8b_entry_reused_verbatim(tmp_path: Path) -> None:
    cfg = _setup_tree(tmp_path)
    _, entries = rb.rebuild(sweep_cfg=cfg, dataset=DATASET, repo_root=tmp_path)
    eb = next(e for e in entries if e.system_id == "fixed_8b_matched")
    # reused, not reconstructed: no reconstruction note, original timestamps kept
    assert rb.RECONSTRUCTION_NOTE not in eb.notes
    assert eb.started_at == "2026-05-29T22:07:25+00:00"
    assert eb.model_name == "internvl2-8b"
