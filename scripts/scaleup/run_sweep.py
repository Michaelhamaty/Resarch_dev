"""Scale-up v2 Stage 7 main sweep driver.

Plan reference: docs/specs/scaleup_v2_plan.md Stage 7. This script is
the thin outer loop the plan calls for: it iterates the ``datasets:``
block of ``configs/experiment/scaleup_v2.yaml``, synthesizes a per-
dataset ``Phase6Config`` in memory, and invokes the existing
``run_phase6`` orchestrator on each. No code under
``src/adaptive_inference/experiment/`` is touched — this is composition
only, exactly as the plan specifies.

Per-dataset outputs land at::

    outputs/scaleup_v2/<dataset>/<system_id>/{first_pass,reparse,final,
                                              run.log.jsonl,
                                              manifest.json}

Failure of one dataset does not abort the next; failed datasets surface
in this driver's stdout summary and are reported as failed in the
top-level scale-up manifest at ``outputs/scaleup_v2/sweep_manifest.json``.

Resumability: if a dataset's run is killed mid-system, re-running the
script with the same config skips datasets whose
``<dataset>/manifest.json`` already exists (the underlying
``run_phase6`` writes that file only at the end of a successful pass).
Pass ``--force`` to re-run completed datasets.

Usage on the VM inside tmux::

    tmux new -s sweep
    uv run python scripts/scaleup/run_sweep.py \\
        --config configs/experiment/scaleup_v2.yaml
    # Ctrl+B d to detach
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

from adaptive_inference.experiment.runner import Phase6Config, run_phase6


# --------------------------------------------------------------------------- #
# Pure-logic helpers (unit-tested without GPU)                                #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DatasetSpec:
    """One entry from the scale-up v2 experiment config's datasets: block."""

    name: str
    split_manifest_path: Path
    records_path: Path
    image_root: Path
    frozen_budgets_path: Path


@dataclass(frozen=True)
class SweepConfig:
    """Top-level scale-up v2 experiment config."""

    run_set_id: str
    datasets: tuple[DatasetSpec, ...]
    model_config_path: Path
    prompt_config_path: Path
    output_root: Path
    random_seeds: tuple[int, ...]


def load_sweep_config(path: Path) -> SweepConfig:
    """Parse configs/experiment/scaleup_v2.yaml into a SweepConfig."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Sweep config at {path} must be a YAML mapping")

    for key in ("run_set_id", "datasets", "model", "prompt", "output"):
        if key not in raw:
            raise ValueError(f"Sweep config at {path} missing required key: {key}")

    datasets_section = raw["datasets"]
    if not isinstance(datasets_section, dict) or not datasets_section:
        raise ValueError(
            f"Sweep config at {path} 'datasets' must be a non-empty mapping"
        )

    datasets: list[DatasetSpec] = []
    for name, body in datasets_section.items():
        if not isinstance(body, dict):
            raise ValueError(f"datasets.{name} must be a mapping")
        for k in ("split_manifest_path", "records_path", "image_root",
                  "frozen_budgets_path"):
            if k not in body:
                raise ValueError(f"datasets.{name} missing key: {k}")
        datasets.append(DatasetSpec(
            name=str(name),
            split_manifest_path=Path(body["split_manifest_path"]),
            records_path=Path(body["records_path"]),
            image_root=Path(body["image_root"]),
            frozen_budgets_path=Path(body["frozen_budgets_path"]),
        ))

    random_section = raw.get("random_escalation") or {}
    seeds_raw = random_section.get("seeds", [0, 1, 2]) if isinstance(random_section, dict) else [0, 1, 2]
    if not isinstance(seeds_raw, list) or not all(isinstance(s, int) for s in seeds_raw):
        raise ValueError("random_escalation.seeds must be a list of ints")

    return SweepConfig(
        run_set_id=str(raw["run_set_id"]),
        datasets=tuple(datasets),
        model_config_path=Path(raw["model"]["config_path"]),
        prompt_config_path=Path(raw["prompt"]["config_path"]),
        output_root=Path(raw["output"]["root"]),
        random_seeds=tuple(seeds_raw),
    )


def phase6_config_for(
    sweep_cfg: SweepConfig, dataset: DatasetSpec
) -> Phase6Config:
    """Synthesize a Phase6Config for one dataset entry.

    Pure: takes only declarative inputs, returns a dataclass. Does not
    read frozen budgets or split files; ``run_phase6`` does that work
    when it consumes the config.
    """

    return Phase6Config(
        run_set_id=f"{sweep_cfg.run_set_id}__{dataset.name}",
        frozen_budgets_path=dataset.frozen_budgets_path,
        split_name=f"{dataset.name}_held_out",
        held_out_manifest_path=dataset.split_manifest_path,
        records_path=dataset.records_path,
        image_root=dataset.image_root,
        model_config_path=sweep_cfg.model_config_path,
        prompt_config_path=sweep_cfg.prompt_config_path,
        output_root=sweep_cfg.output_root / dataset.name,
        random_seeds=sweep_cfg.random_seeds,
    )


def dataset_already_completed(p6_cfg: Phase6Config) -> bool:
    """True iff the per-dataset run_phase6 manifest exists.

    ``run_phase6`` writes the manifest only on a non-aborted pass, so
    presence is a good 'already done' signal for the outer loop's
    resume behavior. Per-system partial work inside that manifest is
    not re-skippable here — the inner runner would have to expose
    --resume for that.
    """

    return (p6_cfg.output_root / "manifest.json").exists()


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, required=True,
        help="Path to configs/experiment/scaleup_v2.yaml.",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="Optional: run only the named dataset(s).",
    )
    parser.add_argument(
        "--systems", nargs="+", default=None,
        help=(
            "Optional: pass through to run_phase6's --systems filter "
            "(e.g. adaptive_2b random_2b)."
        ),
    )
    parser.add_argument(
        "--allow-stubbed-8b", action="store_true",
        help="Pass through to run_phase6. Should NOT be needed in scale-up v2.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run datasets even if their per-dataset manifest already exists.",
    )
    args = parser.parse_args(argv)

    sweep_cfg = load_sweep_config(args.config)
    print(f"[sweep] run_set_id={sweep_cfg.run_set_id}")
    print(f"[sweep] {len(sweep_cfg.datasets)} dataset(s) configured: "
          f"{[d.name for d in sweep_cfg.datasets]}")
    print(f"[sweep] output_root={sweep_cfg.output_root}")
    print(f"[sweep] random_seeds={sweep_cfg.random_seeds}")

    dataset_filter = set(args.datasets) if args.datasets else None
    systems_filter = tuple(args.systems) if args.systems else None

    outcomes: list[dict] = []
    any_failed = False

    for dataset in sweep_cfg.datasets:
        if dataset_filter is not None and dataset.name not in dataset_filter:
            print(f"[sweep] skipping {dataset.name} (not in --datasets filter)")
            continue
        p6_cfg = phase6_config_for(sweep_cfg, dataset)

        if not args.force and dataset_already_completed(p6_cfg):
            print(
                f"[sweep] {dataset.name}: already complete "
                f"({p6_cfg.output_root / 'manifest.json'} exists). "
                "Pass --force to re-run."
            )
            outcomes.append({
                "dataset": dataset.name,
                "status": "skipped_already_complete",
                "manifest_path": str(p6_cfg.output_root / "manifest.json"),
            })
            continue

        print()
        print("=" * 72)
        print(f"[sweep] starting dataset={dataset.name}")
        print(f"  held_out={dataset.split_manifest_path}")
        print(f"  records ={dataset.records_path}")
        print(f"  frozen  ={dataset.frozen_budgets_path}")
        print(f"  out_dir ={p6_cfg.output_root}")
        print("=" * 72)

        started_at = datetime.now(timezone.utc).isoformat()
        try:
            result = run_phase6(
                p6_cfg,
                allow_stubbed_8b=args.allow_stubbed_8b,
                systems_filter=systems_filter,
            )
        except Exception:
            print(
                f"[sweep] {dataset.name}: run_phase6 raised; continuing to "
                "next dataset",
                file=sys.stderr,
            )
            traceback.print_exc()
            outcomes.append({
                "dataset": dataset.name,
                "status": "errored",
                "started_at": started_at,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "error": "see traceback above",
            })
            any_failed = True
            continue

        finished_at = datetime.now(timezone.utc).isoformat()
        outcomes.append({
            "dataset": dataset.name,
            "status": "failed" if result.any_failed else "ok",
            "started_at": started_at,
            "finished_at": finished_at,
            "manifest_path": str(result.manifest_path),
            "n_systems": len(result.entries),
            "system_statuses": {
                e.system_id: e.status for e in result.entries
            },
        })
        if result.any_failed:
            any_failed = True

        print()
        print(f"[sweep] {dataset.name}: status="
              f"{'failed' if result.any_failed else 'ok'} "
              f"({len(result.entries)} systems)")
        for entry in result.entries:
            line = f"  - {entry.system_id:<24} status={entry.status}"
            if entry.pages_processed is not None:
                line += f" pages={entry.pages_processed}"
            print(line)

    sweep_manifest_path = sweep_cfg.output_root / "sweep_manifest.json"
    sweep_cfg.output_root.mkdir(parents=True, exist_ok=True)
    sweep_manifest_path.write_text(
        json.dumps({
            "schema_version": 1,
            "run_set_id": sweep_cfg.run_set_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "sweep_config_path": str(args.config),
            "outcomes": outcomes,
            "any_failed": any_failed,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print()
    print(f"[sweep] wrote {sweep_manifest_path}")
    print(f"[sweep] overall: {'FAILED' if any_failed else 'OK'}")
    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
