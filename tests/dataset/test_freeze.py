from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from adaptive_inference.dataset import MANIFEST_FILENAMES, freeze_phase1_universe


def _write_config(
    config_dir: Path,
    *,
    records_path: Path,
    manifests_dir: Path,
    calibration_size: int = 5,
    eval_size: int | None = None,
    seed: int = 1337,
) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "pinning": {
            "dataset_name": "omnidocbench",
            "snapshot_version": "v1-test",
            "eval_code_commit": "TBD",
            "source": str(records_path),
            "notes": "freeze test",
        },
        "input": {"records_path": str(records_path)},
        "subsets": {
            "hard_table_rule": {
                "min_row_count": 20,
                "min_col_count": 6,
                "require_any_of": ["has_merged_cells", "has_nested_headers"],
            }
        },
        "split": {
            "seed": seed,
            "calibration_size": calibration_size,
            "eval_size": eval_size,
        },
        "output": {"manifests_dir": str(manifests_dir)},
    }
    path = config_dir / "phase1.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def test_end_to_end_over_fixture(
    tmp_path: Path, sample_pages_path: Path
) -> None:
    manifests_dir = tmp_path / "splits"
    config_path = _write_config(
        tmp_path / "cfg", records_path=sample_pages_path, manifests_dir=manifests_dir
    )
    result = freeze_phase1_universe(config_path)

    for kind, name in MANIFEST_FILENAMES.items():
        assert (manifests_dir / name).exists(), f"missing manifest: {kind}"
        assert result.written_paths[kind] == manifests_dir / name

    cal = set(result.split.calibration)
    held_out = set(result.split.held_out_eval)
    eval_universe = set(result.eval_universe_ids)
    hard_subset = set(result.hard_subset_ids)

    assert cal.isdisjoint(held_out), "calibration and held-out must not overlap"
    assert cal <= eval_universe
    assert held_out <= eval_universe
    assert hard_subset <= eval_universe
    assert len(eval_universe) == 14  # page_0001..page_0014
    assert hard_subset == {"page_0001", "page_0002", "page_0003", "page_0004"}
    assert len(cal) == 5
    assert len(held_out) == 9
    assert result.split.seed == 1337


def test_rerun_is_byte_identical(
    tmp_path: Path, sample_pages_path: Path
) -> None:
    manifests_a = tmp_path / "run_a"
    manifests_b = tmp_path / "run_b"
    config_a = _write_config(
        tmp_path / "a", records_path=sample_pages_path, manifests_dir=manifests_a
    )
    config_b = _write_config(
        tmp_path / "b", records_path=sample_pages_path, manifests_dir=manifests_b
    )

    freeze_phase1_universe(config_a)
    freeze_phase1_universe(config_b)

    for name in MANIFEST_FILENAMES.values():
        assert (manifests_a / name).read_bytes() == (manifests_b / name).read_bytes()


def test_manifest_payload_shape(
    tmp_path: Path, sample_pages_path: Path
) -> None:
    manifests_dir = tmp_path / "splits"
    config_path = _write_config(
        tmp_path / "cfg", records_path=sample_pages_path, manifests_dir=manifests_dir
    )
    freeze_phase1_universe(config_path)

    eval_manifest = json.loads(
        (manifests_dir / MANIFEST_FILENAMES["eval_universe"]).read_text(encoding="utf-8")
    )
    cal_manifest = json.loads(
        (manifests_dir / MANIFEST_FILENAMES["calibration_split"]).read_text(encoding="utf-8")
    )

    assert eval_manifest["manifest_kind"] == "eval_universe"
    assert eval_manifest["generated_with_seed"] is None
    assert eval_manifest["pinning"]["dataset_name"] == "omnidocbench"
    assert eval_manifest["page_ids"] == sorted(eval_manifest["page_ids"])

    assert cal_manifest["manifest_kind"] == "calibration_split"
    assert cal_manifest["generated_with_seed"] == 1337
    assert cal_manifest["page_ids"] == sorted(cal_manifest["page_ids"])


def test_rejects_missing_config_section(
    tmp_path: Path, sample_pages_path: Path
) -> None:
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text(
        yaml.safe_dump({"pinning": {}, "input": {}, "subsets": {}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing required section"):
        freeze_phase1_universe(bad_config)
