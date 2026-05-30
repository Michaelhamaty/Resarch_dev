"""Scale-up v2 Stage 7 recovery: rebuild a complete dataset manifest from run logs.

Companion to ``stitch_manifest.py``. ``stitch_manifest`` merges per-system
manifest *snapshots* that the per-system sweep loop is supposed to stash into
``<dataset>/.manifests/``. If that stash was never populated (the loop omitted
the copy step), the stash is empty and stitching is impossible — meanwhile
``run_phase6`` has rewritten ``<dataset>/manifest.json`` on every invocation, so
that file lists only the *last* system run. The Stage 9 aggregator discovers
systems *from* the manifest, so it would see just one system.

The run logs themselves are complete and authoritative: every system's
``<dataset>/<system_id>/run.log.jsonl`` holds one record per page. This tool
reconstructs the full ``entries`` list from those logs, reusing the exact
helpers ``run_phase6`` uses so the result cannot drift:

  * ``build_all_system_specs``           -> which systems / families / seeds
  * ``load_frozen_budgets`` (+ budgets)  -> budget names, tiles, model, adapter
  * ``derive_calibration_reparse_rate``  -> random-escalation probability/source

It writes via the canonical ``write_manifest`` so the on-disk schema is
byte-for-byte what Stage 9 expects. Two integrity choices:

  * The surviving header is reused verbatim (its sha256s / git_head / reparse
    rate are correct — written by the last real ``run_phase6`` invocation).
  * Any entry already present in the surviving manifest (e.g. the last system)
    is reused verbatim rather than reconstructed.
  * Every *reconstructed* entry carries a ``notes`` line recording that it was
    rebuilt post-hoc, so the manifest never hides its provenance.

No model is loaded; no inference re-runs. Usage (on the VM)::

    uv run python scripts/scaleup/rebuild_manifest.py \\
        --config configs/experiment/scaleup_v2.yaml \\
        --dataset fintabnet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow ``import adaptive_inference`` when run as a script.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from adaptive_inference.calibration.artifact import (  # noqa: E402
    FrozenBudget,
    FrozenBudgets,
    load_frozen_budgets,
)
from adaptive_inference.experiment.frozen_inputs import (  # noqa: E402
    CalibrationReparseRate,
    derive_calibration_reparse_rate,
)
from adaptive_inference.experiment.manifest import (  # noqa: E402
    ManifestEntry,
    ManifestHeader,
    write_manifest,
)
from adaptive_inference.experiment.systems import (  # noqa: E402
    SystemSpec,
    build_all_system_specs,
)

# run_sweep lives alongside this script; reuse its config loader so the
# output_root / dataset layout stays a single source of truth.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_sweep import load_sweep_config  # noqa: E402

RECONSTRUCTION_NOTE = (
    "entry reconstructed post-hoc from run.log.jsonl by rebuild_manifest.py; "
    "original per-system manifest snapshot was not stashed"
)


def read_run_log(path: Path) -> list[dict]:
    """Parse a ``run.log.jsonl`` into a list of per-page record dicts."""

    records: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _budget_by_name(frozen: FrozenBudgets) -> dict[str, FrozenBudget]:
    return {
        fb.name: fb
        for fb in (frozen.b_low, frozen.b_high, frozen.b_fix_2b, frozen.b_fix_8b)
    }


def _status_of(records: list[dict]) -> str:
    """``ok`` only if every page record is ``ok``; otherwise ``failed``."""

    if not records:
        return "failed"
    return "ok" if all(r.get("status") == "ok" for r in records) else "failed"


def _single_pass_times(records: list[dict]) -> tuple[str | None, str | None]:
    """Earliest ``started_at`` / latest ``finished_at`` across page records.

    Only single-pass logs carry these top-level timestamps; adaptive and
    random logs do not, so callers pass ``None`` there.
    """

    starts = [r["started_at"] for r in records if r.get("started_at")]
    ends = [r["finished_at"] for r in records if r.get("finished_at")]
    return (min(starts) if starts else None, max(ends) if ends else None)


def reconstruct_entry(
    spec: SystemSpec,
    *,
    records: list[dict],
    frozen: FrozenBudgets,
    reparse_rate: CalibrationReparseRate,
    output_dir: str,
) -> ManifestEntry:
    """Rebuild one ``ManifestEntry`` from a system's run-log records.

    Mirrors ``run_phase6``'s family-specific entry construction exactly,
    deriving every field from the run log + frozen artifact + system spec.
    """

    by_name = _budget_by_name(frozen)
    status = _status_of(records)
    pages = len(records)
    notes = [RECONSTRUCTION_NOTE]

    if spec.runner == "single_pass":
        # budget_name is identical on every page; take it from the first record.
        budget_name = records[0]["budget_name"] if records else None
        fb = by_name.get(budget_name) if budget_name else None
        started_at, finished_at = _single_pass_times(records)
        return ManifestEntry(
            system_id=spec.system_id,
            family=spec.family,
            runner=spec.runner,
            status=status,
            output_dir=output_dir,
            pages_processed=pages,
            model_name=records[0]["model_name"] if records else None,
            adapter_kind=fb.adapter_kind if fb else None,
            budget_name=budget_name,
            budget_max_tiles=fb.max_tiles if fb else None,
            started_at=started_at,
            finished_at=finished_at,
            notes=notes,
        )

    # adaptive / adaptive_random share the low/high budget pair and a
    # per-page ``reparse_triggered`` flag.
    reparse_count = sum(1 for r in records if r.get("reparse_triggered"))
    entry = ManifestEntry(
        system_id=spec.system_id,
        family=spec.family,
        runner=spec.runner,
        status=status,
        output_dir=output_dir,
        pages_processed=pages,
        reparse_count=reparse_count,
        model_name=frozen.b_low.model_name,
        adapter_kind=frozen.b_low.adapter_kind,
        budget_low_name=frozen.b_low.name,
        budget_low_max_tiles=frozen.b_low.max_tiles,
        budget_high_name=frozen.b_high.name,
        budget_high_max_tiles=frozen.b_high.max_tiles,
        started_at=None,
        finished_at=None,
        notes=notes,
    )
    if spec.runner == "adaptive_random":
        entry.seed = spec.seed
        entry.random_probability = reparse_rate.probability
        entry.random_probability_source = reparse_rate.source
        if reparse_rate.degenerate:
            entry.notes.append(
                "calibration-measured reparse rate is 0.0 — random policy will "
                "never escalate."
            )
    return entry


def rebuild(
    *,
    sweep_cfg,
    dataset: str,
    repo_root: Path = _REPO_ROOT,
) -> tuple[ManifestHeader, list[ManifestEntry]]:
    """Reconstruct (header, entries) for ``dataset`` from its run logs."""

    dataset_spec = next(
        (d for d in sweep_cfg.datasets if d.name == dataset), None
    )
    if dataset_spec is None:
        raise ValueError(
            f"dataset {dataset!r} not in config; have "
            f"{[d.name for d in sweep_cfg.datasets]}"
        )

    dataset_root = sweep_cfg.output_root / dataset
    existing_path = dataset_root / "manifest.json"
    if not existing_path.exists():
        raise FileNotFoundError(
            f"no surviving manifest at {existing_path} to source the header from"
        )
    existing = json.loads(existing_path.read_text(encoding="utf-8"))
    header = ManifestHeader(**existing["header"])
    existing_entries = {e["system_id"]: e for e in existing.get("entries", [])}

    frozen = load_frozen_budgets(dataset_spec.frozen_budgets_path)
    reparse_rate = derive_calibration_reparse_rate(frozen)
    specs = build_all_system_specs(tuple(header.random_seeds))

    entries: list[ManifestEntry] = []
    for spec in specs:
        if spec.system_id in existing_entries:
            # Authoritative — written by a real run_phase6 invocation. Keep it.
            entries.append(ManifestEntry(**existing_entries[spec.system_id]))
            continue
        run_log = dataset_root / spec.system_id / "run.log.jsonl"
        if not run_log.exists():
            raise FileNotFoundError(
                f"missing run log for {spec.system_id}: {run_log}. Cannot "
                f"reconstruct its entry."
            )
        records = read_run_log(run_log)
        output_dir = str(sweep_cfg.output_root / dataset / spec.system_id)
        entries.append(
            reconstruct_entry(
                spec,
                records=records,
                frozen=frozen,
                reparse_rate=reparse_rate,
                output_dir=output_dir,
            )
        )
    return header, entries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment/scaleup_v2.yaml"),
        help="Path to the sweep config (for output_root / dataset layout).",
    )
    parser.add_argument(
        "--dataset", required=True, help="Dataset whose manifest to rebuild."
    )
    args = parser.parse_args(argv)

    sweep_cfg = load_sweep_config(args.config)
    header, entries = rebuild(sweep_cfg=sweep_cfg, dataset=args.dataset)
    out_path = write_manifest(
        output_root=sweep_cfg.output_root / args.dataset,
        header=header,
        entries=entries,
    )
    print(f"[rebuild] wrote {len(entries)} entries to {out_path}")
    print(f"[rebuild] systems: {[e.system_id for e in entries]}")
    statuses = {e.system_id: e.status for e in entries}
    non_ok = {s: st for s, st in statuses.items() if st != "ok"}
    if non_ok:
        print(f"[rebuild] WARNING non-ok systems: {non_ok}")
        return 1
    print("[rebuild] all systems status=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
