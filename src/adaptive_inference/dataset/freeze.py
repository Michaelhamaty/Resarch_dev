"""Phase 1 orchestrator: config in, four manifests out.

Reads a YAML config that names the pinning metadata, the fixture path, the
hard-table rule, and the split sizes; produces the four frozen manifests
under the configured output directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .manifests import Manifest, write_manifest
from .pinning import DatasetPinning
from .records import load_page_records
from .splits import SplitResult, split_page_ids
from .subsets import HardTableRule, filter_english_table_pages, filter_hard_table_pages


MANIFEST_FILENAMES = {
    "eval_universe": "eval_universe.json",
    "hard_subset": "hard_subset.json",
    "calibration_split": "calibration_split.json",
    "held_out_eval_split": "held_out_eval_split.json",
}


@dataclass(frozen=True)
class Phase1Result:
    pinning: DatasetPinning
    manifests_dir: Path
    eval_universe_ids: tuple[str, ...]
    hard_subset_ids: tuple[str, ...]
    split: SplitResult
    written_paths: dict[str, Path]


def freeze_phase1_universe(config_path: str | Path) -> Phase1Result:
    config = _load_config(config_path)
    pinning = DatasetPinning.from_mapping(config["pinning"])
    records_path = Path(config["input"]["records_path"])
    hard_rule = HardTableRule.from_mapping(config["subsets"]["hard_table_rule"])

    split_cfg = config["split"]
    seed = int(split_cfg["seed"])
    calibration_size = int(split_cfg["calibration_size"])
    eval_size_raw = split_cfg.get("eval_size")
    eval_size = None if eval_size_raw is None else int(eval_size_raw)

    manifests_dir = Path(config["output"]["manifests_dir"])

    records = load_page_records(records_path)
    eval_universe = filter_english_table_pages(records)
    eval_universe_ids = tuple(sorted(r.page_id for r in eval_universe))

    hard_subset = filter_hard_table_pages(eval_universe, hard_rule)
    hard_subset_ids = tuple(sorted(r.page_id for r in hard_subset))

    split = split_page_ids(
        list(eval_universe_ids),
        calibration_size=calibration_size,
        eval_size=eval_size,
        seed=seed,
    )

    written: dict[str, Path] = {}
    written["eval_universe"] = write_manifest(
        Manifest(
            manifest_kind="eval_universe",
            pinning=pinning,
            page_ids=eval_universe_ids,
        ),
        manifests_dir / MANIFEST_FILENAMES["eval_universe"],
    )
    written["hard_subset"] = write_manifest(
        Manifest(
            manifest_kind="hard_subset",
            pinning=pinning,
            page_ids=hard_subset_ids,
        ),
        manifests_dir / MANIFEST_FILENAMES["hard_subset"],
    )
    written["calibration_split"] = write_manifest(
        Manifest(
            manifest_kind="calibration_split",
            pinning=pinning,
            page_ids=split.calibration,
            generated_with_seed=seed,
        ),
        manifests_dir / MANIFEST_FILENAMES["calibration_split"],
    )
    written["held_out_eval_split"] = write_manifest(
        Manifest(
            manifest_kind="held_out_eval_split",
            pinning=pinning,
            page_ids=split.held_out_eval,
            generated_with_seed=seed,
        ),
        manifests_dir / MANIFEST_FILENAMES["held_out_eval_split"],
    )

    return Phase1Result(
        pinning=pinning,
        manifests_dir=manifests_dir,
        eval_universe_ids=eval_universe_ids,
        hard_subset_ids=hard_subset_ids,
        split=split,
        written_paths=written,
    )


def _load_config(config_path: str | Path) -> dict[str, Any]:
    text = Path(config_path).read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {config_path} must be a YAML mapping")
    for key in ("pinning", "input", "subsets", "split", "output"):
        if key not in data:
            raise ValueError(f"Config at {config_path} missing required section: {key}")
    return data
