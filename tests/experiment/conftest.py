"""Shared fixtures for Phase 6 experiment tests.

Each test gets a self-contained tmp tree with:

- a two-page manifest (``manifest.json``)
- a matching ``records.json``
- tiny PNG images for each page
- a frozen-budgets JSON consistent with the Phase 5 schema
- a model registry YAML
- a prompt YAML

The runner smoke test then points a ``Phase6Config`` at this tree.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from PIL import Image


@dataclass(frozen=True)
class Phase6Fixture:
    root: Path
    manifest_path: Path
    records_path: Path
    image_root: Path
    frozen_budgets_path: Path
    model_config_path: Path
    prompt_config_path: Path
    output_root: Path


def _write_frozen_budgets(
    path: Path,
    *,
    low: int = 6,
    high: int = 8,
    fix2: int = 6,
    fix8: int = 6,
    measured: float = 6.0,
    fix8_adapter: str = "stub",
) -> None:
    payload = {
        "schema_version": 1,
        "run_id": "phase5_test",
        "generated_at": "2026-04-23T00:00:00+00:00",
        "pinning": {
            "calibration_split_manifest": "data/splits/calibration_split.json",
            "calibration_split_sha256": "abc",
            "calibration_config_path": "configs/calibration/phase5.yaml",
            "cost_unit": "max_tiles",
            "matched_cost_tolerance": 0.1,
        },
        "budgets": {
            "B_low": {
                "name": "B_low",
                "max_tiles": low,
                "model_name": "internvl2-2b",
                "adapter_kind": "stub",
            },
            "B_high": {
                "name": "B_high",
                "max_tiles": high,
                "model_name": "internvl2-2b",
                "adapter_kind": "stub",
            },
            "B_fix_2B": {
                "name": "B_fix_2B",
                "max_tiles": fix2,
                "model_name": "internvl2-2b",
                "adapter_kind": "stub",
            },
            "B_fix_8B": {
                "name": "B_fix_8B",
                "max_tiles": fix8,
                "model_name": "internvl2-8b",
                "adapter_kind": fix8_adapter,
            },
        },
        "selection": {
            "adaptive": {
                "target_cost_tiles": measured,
                "measured_cost_tiles": measured,
                "low_max_tiles": low,
                "high_max_tiles": high,
            },
            "fixed_2b": {
                "target_cost_tiles": measured,
                "measured_cost_tiles": measured,
                "max_tiles": fix2,
                "within_tolerance": True,
            },
            "fixed_8b": {
                "target_cost_tiles": measured,
                "measured_cost_tiles": measured,
                "max_tiles": fix8,
                "within_tolerance": True,
            },
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _make_fixture(
    tmp_path: Path,
    *,
    page_ids: list[str],
    measured: float = 6.0,
    fix8_adapter: str = "stub",
) -> Phase6Fixture:
    image_root = tmp_path / "data"
    images_dir = image_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for pid in page_ids:
        Image.new("L", (4, 4), color=0).save(images_dir / f"{pid}.png", format="PNG")

    records_path = tmp_path / "records.json"
    records_path.write_text(
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

    manifest_path = tmp_path / "held_out.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "manifest_kind": "held_out_eval_split",
                "page_ids": page_ids,
                "pinning": {},
            }
        ),
        encoding="utf-8",
    )

    frozen_path = tmp_path / "frozen_budgets.json"
    _write_frozen_budgets(
        frozen_path, measured=measured, fix8_adapter=fix8_adapter
    )

    model_yaml = tmp_path / "models.yaml"
    model_yaml.write_text(
        "models:\n"
        "  internvl2-2b:\n"
        "    adapter_kind: stub\n"
        "    model_id: OpenGVLab/InternVL2-2B\n"
        "    notes: test\n"
        "  internvl2-8b:\n"
        f"    adapter_kind: {fix8_adapter}\n"
        "    model_id: OpenGVLab/InternVL2-8B\n"
        "    notes: test\n",
        encoding="utf-8",
    )

    prompt_yaml = tmp_path / "prompt.yaml"
    prompt_yaml.write_text(
        "id: table_parse_v1\n"
        "version: 1\n"
        "description: test\n"
        "template: test\n",
        encoding="utf-8",
    )

    return Phase6Fixture(
        root=tmp_path,
        manifest_path=manifest_path,
        records_path=records_path,
        image_root=image_root,
        frozen_budgets_path=frozen_path,
        model_config_path=model_yaml,
        prompt_config_path=prompt_yaml,
        output_root=tmp_path / "outputs",
    )


@pytest.fixture
def phase6_fixture(tmp_path: Path) -> Phase6Fixture:
    return _make_fixture(tmp_path, page_ids=["page_0001", "page_0002"])


@pytest.fixture
def phase6_fixture_reparse_rate_half(tmp_path: Path) -> Phase6Fixture:
    # p = (10 - 6) / 8 = 0.5, so random baseline actually escalates.
    return _make_fixture(
        tmp_path, page_ids=["page_0001", "page_0002", "page_0003"], measured=10.0
    )
