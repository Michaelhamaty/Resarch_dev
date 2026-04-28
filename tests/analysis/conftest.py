"""Shared fixtures for Phase 7 analysis tests.

The analysis package is purely an ingestor of Phase 1-6 artifacts.
Building a synthetic on-disk Phase 6 tree (manifest + per-system run
logs + matching frozen-budgets and split files) keeps the tests fast
and isolated from any real run output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from adaptive_inference.experiment.manifest import sha256_of_file


@dataclass(frozen=True)
class Phase7Fixture:
    """A self-contained Phase 6 tree on disk.

    Six systems: adaptive_2b, fixed_2b_low, fixed_2b_matched,
    random_2b_seed0, random_2b_seed1, fixed_8b_matched. Two pages on
    the held-out split. Stub adapters everywhere.
    """

    root: Path
    phase6_dir: Path
    phase6_manifest_path: Path
    frozen_budgets_path: Path
    held_out_split_path: Path
    calibration_split_path: Path
    output_root: Path
    held_out_page_ids: tuple[str, ...]


HELD_OUT = ("page_a", "page_b")
CAL_PAGES = ("page_c",)


def _frozen_payload(low: int = 6, high: int = 8) -> dict:
    return {
        "schema_version": 1,
        "run_id": "phase5_test",
        "generated_at": "2026-04-01T00:00:00+00:00",
        "pinning": {
            "calibration_split_manifest": "splits/calibration.json",
            "calibration_split_sha256": "fake",
            "calibration_config_path": "configs/calibration/phase5.yaml",
            "cost_unit": "max_tiles",
            "matched_cost_tolerance": 0.1,
        },
        "budgets": {
            "B_low": {"name": "B_low", "max_tiles": low, "model_name": "internvl2-2b", "adapter_kind": "stub"},
            "B_high": {"name": "B_high", "max_tiles": high, "model_name": "internvl2-2b", "adapter_kind": "stub"},
            "B_fix_2B": {"name": "B_fix_2B", "max_tiles": low, "model_name": "internvl2-2b", "adapter_kind": "stub"},
            "B_fix_8B": {"name": "B_fix_8B", "max_tiles": low, "model_name": "internvl2-8b", "adapter_kind": "stub"},
        },
        "selection": {
            "adaptive": {
                "target_cost_tiles": float(low),
                "measured_cost_tiles": float(low),
                "low_max_tiles": low,
                "high_max_tiles": high,
            },
            "fixed_2b": {
                "target_cost_tiles": float(low),
                "measured_cost_tiles": float(low),
                "max_tiles": low,
                "within_tolerance": True,
            },
            "fixed_8b": {
                "target_cost_tiles": float(low),
                "measured_cost_tiles": float(low),
                "max_tiles": low,
                "within_tolerance": True,
            },
        },
    }


def _write_split(path: Path, kind: str, page_ids: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "manifest_kind": kind,
                "page_ids": list(page_ids),
                "pinning": {},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _adaptive_log(
    *,
    run_id: str,
    page_id: str,
    reparse_triggered: bool,
    failure_codes: list[str],
    seed: int | None = None,
    probability: float | None = None,
) -> dict:
    rec = {
        "run_id": run_id,
        "split": "held_out_eval_split",
        "page_id": page_id,
        "model_name": "internvl2-2b",
        "prompt_id": "table_parse_v1",
        "budget_low": "B_low",
        "budget_high": "B_high",
        "reparse_triggered": reparse_triggered,
        "verifier_decision": "REPARSE" if failure_codes else "PASS",
        "verifier_failure_codes": failure_codes,
        "predicted_table_count": 1,
        "first_pass_output_tokens": 9,
        "reparse_output_tokens": 11 if reparse_triggered else None,
        "first_pass_runtime_ms": 0.5,
        "verifier_runtime_ms": 0.1,
        "reparse_runtime_ms": 0.7 if reparse_triggered else None,
        "total_runtime_ms": 1.5 if reparse_triggered else 0.7,
        "first_pass_raw_path": f"first_pass/raw/{page_id}.md",
        "reparse_raw_path": f"reparse/raw/{page_id}.md" if reparse_triggered else None,
        "final_raw_path": f"final/raw/{page_id}.md",
        "final_output_source": "reparse" if reparse_triggered else "first_pass",
        "status": "ok",
    }
    if seed is not None:
        rec["random_seed"] = seed
        rec["random_probability"] = probability
        rec["policy"] = "random_escalation"
    return rec


def _single_pass_log(*, run_id: str, page_id: str, budget_name: str) -> dict:
    return {
        "run_id": run_id,
        "page_id": page_id,
        "split": "held_out_eval_split",
        "model_name": "internvl2-2b",
        "budget_name": budget_name,
        "prompt_id": "table_parse_v1",
        "runtime_ms": 0.4,
        "output_token_count": 9,
        "raw_output_path": f"raw/{page_id}.md",
        "started_at": "2026-04-01T00:00:00+00:00",
        "finished_at": "2026-04-01T00:00:01+00:00",
        "status": "ok",
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n",
        encoding="utf-8",
    )


def _build_systems(phase6_dir: Path, repo_root: Path) -> list[dict]:
    """Write run.log.jsonl for every system; return list of manifest entries.

    ``output_dir`` in each entry is relative to ``repo_root`` so Phase 7's
    audit can resolve it as ``repo_root / output_dir``.
    """

    entries: list[dict] = []

    # adaptive_2b — 2 pages, 0 reparses
    adaptive_dir = phase6_dir / "adaptive_2b"
    _write_jsonl(
        adaptive_dir / "run.log.jsonl",
        [
            _adaptive_log(run_id="adaptive_2b", page_id=p, reparse_triggered=False, failure_codes=[])
            for p in HELD_OUT
        ],
    )
    entries.append({
        "system_id": "adaptive_2b",
        "family": "adaptive_2b",
        "runner": "adaptive",
        "status": "ok",
        "output_dir": str(adaptive_dir.relative_to(repo_root)),
        "pages_processed": 2,
        "reparse_count": 0,
        "model_name": "internvl2-2b",
        "adapter_kind": "stub",
        "budget_low_name": "B_low",
        "budget_low_max_tiles": 6,
        "budget_high_name": "B_high",
        "budget_high_max_tiles": 8,
        "budget_name": None,
        "budget_max_tiles": None,
        "seed": None,
        "random_probability": None,
        "random_probability_source": None,
        "started_at": "2026-04-01T00:00:00+00:00",
        "finished_at": "2026-04-01T00:00:01+00:00",
        "error": None,
        "reason": None,
        "notes": [],
    })

    # fixed_2b_low
    f1_dir = phase6_dir / "fixed_2b_low"
    _write_jsonl(
        f1_dir / "run.log.jsonl",
        [_single_pass_log(run_id="fixed_2b_low", page_id=p, budget_name="B_low") for p in HELD_OUT],
    )
    entries.append(_fixed_entry("fixed_2b_low", f1_dir, repo_root, "B_low", 6))

    # fixed_2b_matched
    f2_dir = phase6_dir / "fixed_2b_matched"
    _write_jsonl(
        f2_dir / "run.log.jsonl",
        [_single_pass_log(run_id="fixed_2b_matched", page_id=p, budget_name="B_fix_2B") for p in HELD_OUT],
    )
    entries.append(_fixed_entry("fixed_2b_matched", f2_dir, repo_root, "B_fix_2B", 6))

    # random_2b seeds 0 and 1 — both 0 reparses (degenerate)
    for seed in (0, 1):
        rd = phase6_dir / f"random_2b_seed{seed}"
        _write_jsonl(
            rd / "run.log.jsonl",
            [
                _adaptive_log(
                    run_id=f"random_2b_seed{seed}",
                    page_id=p,
                    reparse_triggered=False,
                    failure_codes=[],
                    seed=seed,
                    probability=0.0,
                )
                for p in HELD_OUT
            ],
        )
        entries.append({
            "system_id": f"random_2b_seed{seed}",
            "family": "random_2b",
            "runner": "adaptive_random",
            "status": "ok",
            "output_dir": str(rd.relative_to(repo_root)),
            "pages_processed": 2,
            "reparse_count": 0,
            "model_name": "internvl2-2b",
            "adapter_kind": "stub",
            "budget_low_name": "B_low",
            "budget_low_max_tiles": 6,
            "budget_high_name": "B_high",
            "budget_high_max_tiles": 8,
            "budget_name": None,
            "budget_max_tiles": None,
            "seed": seed,
            "random_probability": 0.0,
            "random_probability_source": "phase5_frozen_artifact.adaptive_selection",
            "started_at": "2026-04-01T00:00:00+00:00",
            "finished_at": "2026-04-01T00:00:01+00:00",
            "error": None,
            "reason": None,
            "notes": ["calibration-measured reparse rate is 0.0 — random policy will never escalate. This matches Phase 5 data as frozen."],
        })

    # fixed_8b_matched
    f8_dir = phase6_dir / "fixed_8b_matched"
    _write_jsonl(
        f8_dir / "run.log.jsonl",
        [_single_pass_log(run_id="fixed_8b_matched", page_id=p, budget_name="B_fix_8B") for p in HELD_OUT],
    )
    entries.append(_fixed_entry("fixed_8b_matched", f8_dir, repo_root, "B_fix_8B", 6, model="internvl2-8b"))

    return entries


def _fixed_entry(
    system_id: str,
    sys_dir: Path,
    repo_root: Path,
    budget_name: str,
    max_tiles: int,
    model: str = "internvl2-2b",
) -> dict:
    return {
        "system_id": system_id,
        "family": system_id,
        "runner": "single_pass",
        "status": "ok",
        "output_dir": str(sys_dir.relative_to(repo_root)),
        "pages_processed": 2,
        "reparse_count": None,
        "model_name": model,
        "adapter_kind": "stub",
        "budget_low_name": None,
        "budget_low_max_tiles": None,
        "budget_high_name": None,
        "budget_high_max_tiles": None,
        "budget_name": budget_name,
        "budget_max_tiles": max_tiles,
        "seed": None,
        "random_probability": None,
        "random_probability_source": None,
        "started_at": "2026-04-01T00:00:00+00:00",
        "finished_at": "2026-04-01T00:00:01+00:00",
        "error": None,
        "reason": None,
        "notes": [],
    }


@pytest.fixture
def phase7_fixture(tmp_path: Path) -> Phase7Fixture:
    """Build a complete on-disk Phase 6 tree under tmp_path."""

    root = tmp_path
    splits_dir = root / "data" / "splits"
    held_out_path = splits_dir / "held_out_eval_split.json"
    cal_path = splits_dir / "calibration_split.json"
    _write_split(held_out_path, "held_out_eval_split", HELD_OUT)
    _write_split(cal_path, "calibration_split", CAL_PAGES)

    frozen_path = root / "configs" / "calibration" / "frozen_budgets.json"
    frozen_path.parent.mkdir(parents=True, exist_ok=True)
    frozen_path.write_text(
        json.dumps(_frozen_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    phase6_dir = root / "outputs" / "runs" / "phase6"
    entries = _build_systems(phase6_dir, repo_root=root)

    manifest = {
        "schema_version": 1,
        "header": {
            "run_set_id": "phase7_test_run",
            "generated_at": "2026-04-01T00:00:00+00:00",
            "held_out_manifest_path": str(held_out_path.relative_to(root)),
            "held_out_manifest_sha256": sha256_of_file(held_out_path),
            "frozen_budgets_path": str(frozen_path.relative_to(root)),
            "frozen_budgets_sha256": sha256_of_file(frozen_path),
            "prompt_id": "table_parse_v1",
            "prompt_version": 1,
            "git_head": "deadbeef",
            "calibration_reparse_rate": 0.0,
            "calibration_reparse_rate_degenerate": True,
            "random_seeds": [0, 1],
        },
        "entries": entries,
    }
    manifest_path = phase6_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return Phase7Fixture(
        root=root,
        phase6_dir=phase6_dir,
        phase6_manifest_path=manifest_path,
        frozen_budgets_path=frozen_path,
        held_out_split_path=held_out_path,
        calibration_split_path=cal_path,
        output_root=root / "outputs" / "analysis",
        held_out_page_ids=HELD_OUT,
    )
