"""Phase 7 integration audit — programmatic consistency checks.

Verifies that the chain Phase 1 → Phase 5 → Phase 6 → Phase 7 is
internally consistent. Each check returns one of:

- ``ok``   — invariant holds
- ``warn`` — informational (e.g. degenerate stub state)
- ``fail`` — invariant violated; Phase 7 CLI exits non-zero

Checks are intentionally conservative: a SHA mismatch fails, a missing
optional file warns, and accuracy gating warns rather than fails because
stubbed adapters are an explicit project state, not a defect.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ..calibration.artifact import FrozenBudgets
from ..experiment.manifest import sha256_of_file
from ..experiment.systems import (
    ADAPTIVE_2B,
    FIXED_2B_LOW,
    FIXED_2B_MATCHED,
    FIXED_8B_MATCHED,
    RANDOM_2B,
)
from ..verifier import codes as verifier_codes
from .loaders import LoadedSystem, Phase6Manifest, load_split_page_ids
from .results import ADAPTIVE_RUNNERS


STATUS_OK = "ok"
STATUS_WARN = "warn"
STATUS_FAIL = "fail"


@dataclass(frozen=True)
class AuditCheck:
    name: str
    status: str
    detail: str


@dataclass(frozen=True)
class AuditReport:
    checks: tuple[AuditCheck, ...]

    @property
    def any_failed(self) -> bool:
        return any(c.status == STATUS_FAIL for c in self.checks)

    @property
    def any_warned(self) -> bool:
        return any(c.status == STATUS_WARN for c in self.checks)


def run_audit(
    *,
    manifest: Phase6Manifest,
    systems: tuple[LoadedSystem, ...],
    frozen: FrozenBudgets,
    repo_root: Path,
    calibration_split_path: Path,
    held_out_split_path: Path,
    frozen_budgets_path: Path,
    splits_are_identical_acknowledged: bool = False,
) -> AuditReport:
    """Run every Phase 7 audit check and return the aggregated report.

    ``splits_are_identical_acknowledged`` is the MVP shortcut: when set,
    the disjoint check downgrades to a ``warn`` rather than failing. This
    exists because no real held-out split has been built for OmniDocBench
    yet; the limitation is surfaced in ``project_status_final.md``.
    """

    checks: list[AuditCheck] = [
        _check_splits_disjoint(
            calibration_split_path,
            held_out_split_path,
            splits_are_identical_acknowledged=splits_are_identical_acknowledged,
        ),
        _check_held_out_sha(manifest, repo_root, held_out_split_path),
        _check_frozen_sha(manifest, repo_root, frozen_budgets_path),
        _check_entries_complete(manifest),
        _check_entries_match_disk(systems, repo_root),
        _check_log_pages_subset_of_held_out(systems, held_out_split_path),
        _check_frozen_unchanged_by_phase6(manifest, frozen_budgets_path),
        _check_verifier_codes_known(systems),
        _check_prompt_id_pinned(manifest, systems),
        _check_random_seeds_distinct(systems),
        _check_accuracy_status(frozen),
    ]
    return AuditReport(checks=tuple(checks))


def _check_splits_disjoint(
    cal: Path, held: Path, *, splits_are_identical_acknowledged: bool = False
) -> AuditCheck:
    name = "phase1_splits_disjoint"
    try:
        cal_ids = set(load_split_page_ids(cal))
        held_ids = set(load_split_page_ids(held))
    except FileNotFoundError as exc:
        return AuditCheck(name, STATUS_FAIL, f"split file missing: {exc}")
    overlap = sorted(cal_ids & held_ids)
    if overlap:
        if splits_are_identical_acknowledged and cal_ids == held_ids:
            return AuditCheck(
                name,
                STATUS_WARN,
                (
                    "calibration_split == held_out_split (MVP shortcut, "
                    "acknowledged in config). All Phase 6 numbers are "
                    "in-sample for Phase 5's budget pick. A real held-out "
                    "real-data split is the next research step."
                ),
            )
        return AuditCheck(
            name, STATUS_FAIL, f"calibration ∩ held_out is non-empty: {overlap}"
        )
    return AuditCheck(
        name,
        STATUS_OK,
        f"calibration ({len(cal_ids)} pages) and held_out ({len(held_ids)} pages) are disjoint",
    )


def _check_held_out_sha(
    manifest: Phase6Manifest, repo_root: Path, held_out_path: Path
) -> AuditCheck:
    name = "phase6_manifest_sha_matches_held_out"
    expected = manifest.header.held_out_manifest_sha256
    actual = sha256_of_file(held_out_path)
    if expected != actual:
        return AuditCheck(
            name,
            STATUS_FAIL,
            f"held-out SHA mismatch — manifest header says {expected} but disk is {actual}",
        )
    return AuditCheck(name, STATUS_OK, f"held-out SHA matches: {actual[:12]}…")


def _check_frozen_sha(
    manifest: Phase6Manifest, repo_root: Path, frozen_path: Path
) -> AuditCheck:
    name = "phase6_manifest_sha_matches_frozen"
    expected = manifest.header.frozen_budgets_sha256
    actual = sha256_of_file(frozen_path)
    if expected != actual:
        return AuditCheck(
            name,
            STATUS_FAIL,
            f"frozen budgets SHA mismatch — manifest header says {expected} but disk is {actual}",
        )
    return AuditCheck(name, STATUS_OK, f"frozen budgets SHA matches: {actual[:12]}…")


def _check_entries_complete(manifest: Phase6Manifest) -> AuditCheck:
    """Every required family must appear, plus one entry per random seed."""

    name = "phase6_entries_complete"
    families_seen = {e.family for e in manifest.entries}
    required = {ADAPTIVE_2B, FIXED_2B_LOW, FIXED_2B_MATCHED, RANDOM_2B, FIXED_8B_MATCHED}
    missing = sorted(required - families_seen)
    if missing:
        return AuditCheck(
            name, STATUS_FAIL, f"manifest missing required families: {missing}"
        )

    seeds_expected = set(manifest.header.random_seeds)
    seeds_seen = {e.seed for e in manifest.entries if e.family == RANDOM_2B}
    missing_seeds = sorted(seeds_expected - seeds_seen)
    if missing_seeds:
        return AuditCheck(
            name, STATUS_FAIL, f"random_2b missing seeds: {missing_seeds}"
        )
    return AuditCheck(
        name,
        STATUS_OK,
        f"all 5 families present with random seeds {sorted(seeds_expected)}",
    )


def _check_entries_match_disk(
    systems: tuple[LoadedSystem, ...], repo_root: Path
) -> AuditCheck:
    name = "phase6_entries_match_disk"
    problems: list[str] = []
    for sys in systems:
        if sys.entry.status != "ok":
            continue
        if sys.entry.output_dir is None:
            problems.append(f"{sys.entry.system_id}: ok status but output_dir=None")
            continue
        if not sys.log_path.exists():
            problems.append(f"{sys.entry.system_id}: missing {sys.log_path}")
            continue
        if (
            sys.entry.pages_processed is not None
            and sys.entry.pages_processed != len(sys.records)
        ):
            problems.append(
                f"{sys.entry.system_id}: manifest pages_processed="
                f"{sys.entry.pages_processed} != log line count={len(sys.records)}"
            )
    if problems:
        return AuditCheck(name, STATUS_FAIL, "; ".join(problems))
    return AuditCheck(
        name, STATUS_OK, "every ok entry has a log file and matching pages_processed"
    )


def _check_log_pages_subset_of_held_out(
    systems: tuple[LoadedSystem, ...], held_out_path: Path
) -> AuditCheck:
    name = "phase6_log_pages_match_held_out"
    held_out = set(load_split_page_ids(held_out_path))
    for sys in systems:
        if sys.entry.status != "ok":
            continue
        log_pages = {str(r["page_id"]) for r in sys.records if "page_id" in r}
        leaks = sorted(log_pages - held_out)
        if leaks:
            return AuditCheck(
                name,
                STATUS_FAIL,
                f"{sys.entry.system_id}: log contains page_ids not in held_out: {leaks}",
            )
    return AuditCheck(
        name, STATUS_OK, "every system's log page_ids are a subset of held_out_eval_split"
    )


def _check_frozen_unchanged_by_phase6(
    manifest: Phase6Manifest, frozen_path: Path
) -> AuditCheck:
    """Read-only contract: SHA Phase 6 recorded equals current SHA on disk."""

    name = "frozen_artifact_unchanged_by_phase6"
    expected = manifest.header.frozen_budgets_sha256
    actual = sha256_of_file(frozen_path)
    if expected != actual:
        return AuditCheck(
            name,
            STATUS_FAIL,
            f"frozen budgets file has been modified since Phase 6 wrote its manifest "
            f"(was {expected}, now {actual}); the read-only contract is violated.",
        )
    return AuditCheck(
        name, STATUS_OK, "frozen artifact unchanged since Phase 6 manifest was written"
    )


def _check_verifier_codes_known(systems: tuple[LoadedSystem, ...]) -> AuditCheck:
    name = "verifier_decision_codes_known"
    known = {
        verifier_codes.NO_TABLE_FOUND,
        verifier_codes.HTML_PARSE_ERROR,
        verifier_codes.SPAN_EXPANSION_FAILED,
        verifier_codes.RECTANGULAR_INCONSISTENCY,
        verifier_codes.DEGENERATE_TABLE,
    }
    seen: set[str] = set()
    for sys in systems:
        if sys.entry.runner not in ADAPTIVE_RUNNERS:
            continue
        for rec in sys.records:
            for c in rec.get("verifier_failure_codes") or []:
                seen.add(str(c))
    unknown = sorted(seen - known)
    if unknown:
        return AuditCheck(
            name,
            STATUS_FAIL,
            f"verifier emitted unknown failure codes (drift from verifier.codes): {unknown}",
        )
    return AuditCheck(
        name,
        STATUS_OK,
        f"all observed failure codes are members of verifier.codes ({sorted(seen) or 'none observed'})",
    )


def _check_prompt_id_pinned(
    manifest: Phase6Manifest, systems: tuple[LoadedSystem, ...]
) -> AuditCheck:
    name = "prompt_id_pinned"
    expected = manifest.header.prompt_id
    drift: list[str] = []
    for sys in systems:
        if sys.entry.status != "ok":
            continue
        for rec in sys.records:
            pid = rec.get("prompt_id")
            if pid != expected:
                drift.append(
                    f"{sys.entry.system_id}/{rec.get('page_id')}: prompt_id={pid!r}"
                )
                break
    if drift:
        return AuditCheck(
            name,
            STATUS_FAIL,
            f"log prompt_id != manifest.header.prompt_id={expected!r}: {drift}",
        )
    return AuditCheck(name, STATUS_OK, f"every log entry pins prompt_id={expected!r}")


def _check_random_seeds_distinct(
    systems: tuple[LoadedSystem, ...],
) -> AuditCheck:
    """Informational warn if all random seeds make identical decisions.

    With a degenerate stub probability of 0.0 every seed will inevitably
    produce identical reparse decisions (always False). That is the
    correct behavior of the random policy, but a downstream paper claim
    needs distinct draws — so we surface it as a warn, not a fail.
    """

    name = "random_baseline_seeds_distinct"
    random_systems = [
        s
        for s in systems
        if s.entry.family == RANDOM_2B and s.entry.status == "ok"
    ]
    if len(random_systems) < 2:
        return AuditCheck(
            name, STATUS_WARN, f"only {len(random_systems)} random seed(s) ran; cannot test variance"
        )

    decisions_by_seed = {
        s.entry.seed: tuple(bool(r.get("reparse_triggered")) for r in s.records)
        for s in random_systems
    }
    distinct = len({tuple(v) for v in decisions_by_seed.values()})
    if distinct == 1:
        return AuditCheck(
            name,
            STATUS_WARN,
            (
                "all random seeds produced byte-identical reparse decisions — "
                "expected when the calibration-derived probability is 0.0 "
                "(degenerate stub state). Real adapters will produce variance."
            ),
        )
    return AuditCheck(
        name,
        STATUS_OK,
        f"random seeds produced {distinct} distinct reparse-decision sequences",
    )


def _check_accuracy_status(frozen: FrozenBudgets) -> AuditCheck:
    """Phase 7 always warns here on stubs; this is the explicit honest gate."""

    name = "accuracy_status"
    stub_adapters = [
        b.adapter_kind
        for b in (frozen.b_low, frozen.b_high, frozen.b_fix_2b, frozen.b_fix_8b)
        if b.adapter_kind == "stub"
    ]
    if stub_adapters:
        return AuditCheck(
            name,
            STATUS_WARN,
            (
                "accuracy_status=not_applicable_stub_adapters — frozen artifact "
                "uses stub adapters, so no TEDS / edit-distance is computed. "
                "Phase 7 ships cost / reparse / verifier summaries only. "
                "Real adapters and a real scorer are required before any "
                "accuracy claim."
            ),
        )
    return AuditCheck(
        name,
        STATUS_OK,
        "real adapters configured — but Phase 7 still does not bundle a TEDS scorer; ensure one is wired downstream.",
    )


def to_dict(report: AuditReport) -> Mapping[str, object]:
    return {
        "any_failed": report.any_failed,
        "any_warned": report.any_warned,
        "checks": [
            {"name": c.name, "status": c.status, "detail": c.detail}
            for c in report.checks
        ],
    }
