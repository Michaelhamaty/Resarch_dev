"""Scale-up v2 Stage 7 helper: stitch per-system manifests into one.

Plan reference: docs/specs/scaleup_v2_plan.md Stage 7, crash-safety
guardrail G6. Background:

The Stage 7 main sweep is driven **one system per invocation** so a crash
costs at most one system, not a whole dataset (the inner runner does not
expose page-level resume to ``run_phase6``). But ``run_phase6`` rewrites
``<dataset>/manifest.json`` on every invocation, so after a per-system
loop that file lists only the *last* system. The Stage 9 aggregator
(``run_phase7_v2.py``) requires a **complete** manifest and discovers
systems *from it*, not by scanning dirs — so it would mis-read a
per-system run.

Rather than re-derive entry fields (budgets, model names, reparse counts)
and risk drift from what ``run_phase6`` actually wrote, the per-system
loop **stashes** each invocation's authoritative ``manifest.json`` into
``<dataset>/.manifests/<system_id>.json``. This script merges those
stashed single-entry manifests back into one complete
``<dataset>/manifest.json``, reusing the canonical
``experiment.manifest.write_manifest`` writer so the output schema is
byte-for-byte what Stage 9 expects.

Usage (on the VM, after the per-system loop)::

    uv run python scripts/scaleup/stitch_manifest.py \\
        --config configs/experiment/scaleup_v2.yaml \\
        --dataset omnidocbench
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

from adaptive_inference.experiment.manifest import (  # noqa: E402
    ManifestEntry,
    ManifestHeader,
    write_manifest,
)

# run_sweep lives alongside this script; reuse its config loader so the
# output_root / dataset layout stays a single source of truth.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_sweep import load_sweep_config  # noqa: E402

STASH_DIRNAME = ".manifests"


# Deterministic ordering so the stitched manifest is stable regardless of
# the order stash files happen to be globbed in.
SYSTEM_ORDER = (
    "fixed_2b_low",
    "fixed_2b_matched",
    "fixed_8b_matched",
    "adaptive_2b",
    "random_2b_seed0",
    "random_2b_seed1",
    "random_2b_seed2",
)


def _system_sort_key(system_id: str) -> tuple[int, str]:
    try:
        return (SYSTEM_ORDER.index(system_id), "")
    except ValueError:
        return (len(SYSTEM_ORDER), system_id)


def merge_manifests(manifests: list[dict]) -> dict:
    """Merge per-system manifest dicts into one complete manifest dict.

    Pure: takes a list of parsed manifest payloads (each typically a
    single-entry manifest written by ``run_phase6`` for one system) and
    returns a combined payload. The header is taken from the manifest
    with the latest ``generated_at`` (the per-dataset header fields —
    frozen-budget sha, prompt, held-out sha — are identical across
    invocations, so only ``generated_at`` realistically differs).

    Entries are deduplicated by ``system_id`` (last writer wins) and
    sorted into the canonical system order so the output is stable.
    """

    if not manifests:
        raise ValueError("no manifests to merge")

    for m in manifests:
        if m.get("schema_version") != 1:
            raise ValueError(
                f"unexpected schema_version {m.get('schema_version')!r}; expected 1"
            )

    newest = max(manifests, key=lambda m: m["header"]["generated_at"])
    header = newest["header"]

    by_system: dict[str, dict] = {}
    for m in manifests:
        for entry in m.get("entries", []):
            by_system[entry["system_id"]] = entry

    entries = sorted(by_system.values(), key=lambda e: _system_sort_key(e["system_id"]))
    return {"schema_version": 1, "header": header, "entries": entries}


def load_stashed_manifests(stash_dir: Path) -> list[dict]:
    """Parse every ``*.json`` in the stash dir into manifest payloads."""

    paths = sorted(stash_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(
            f"no stashed manifests in {stash_dir}. Did the per-system loop "
            f"copy each manifest.json into {STASH_DIRNAME}/?"
        )
    return [json.loads(p.read_text(encoding="utf-8")) for p in paths]


def write_stitched(merged: dict, output_root: Path) -> Path:
    """Reconstruct dataclasses from ``merged`` and write via write_manifest.

    Routing through ``write_manifest`` (rather than dumping ``merged``
    directly) guarantees the on-disk schema matches exactly what
    ``run_phase6`` produces — including field ordering and the trailing
    newline — so Stage 9's loader cannot drift from this writer.
    """

    header = ManifestHeader(**merged["header"])
    entries = [ManifestEntry(**e) for e in merged["entries"]]
    return write_manifest(output_root=output_root, header=header, entries=entries)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/experiment/scaleup_v2.yaml"),
        help="Path to the sweep config (for output_root / dataset layout).",
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Dataset name whose per-system manifests to stitch.",
    )
    parser.add_argument(
        "--stash-dir", type=Path, default=None,
        help=f"Override the stash dir (default <output_root>/<dataset>/{STASH_DIRNAME}).",
    )
    args = parser.parse_args(argv)

    sweep_cfg = load_sweep_config(args.config)
    dataset_root = sweep_cfg.output_root / args.dataset
    stash_dir = args.stash_dir or (dataset_root / STASH_DIRNAME)

    manifests = load_stashed_manifests(stash_dir)
    merged = merge_manifests(manifests)
    out_path = write_stitched(merged, dataset_root)

    print(f"[stitch] merged {len(manifests)} stashed manifest(s) into {out_path}")
    print(f"[stitch] systems: {[e['system_id'] for e in merged['entries']]}")
    statuses = {e["system_id"]: e["status"] for e in merged["entries"]}
    non_ok = {s: st for s, st in statuses.items() if st != "ok"}
    if non_ok:
        print(f"[stitch] WARNING non-ok systems: {non_ok}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
