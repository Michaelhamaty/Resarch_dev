"""Integration audit tests.

Happy-path tests confirm every check returns the right status on a
healthy fixture. Tampering tests confirm the audit fails loudly when
specific invariants are violated — the project's only defense against
silent drift between phases.
"""

from __future__ import annotations

import json

from adaptive_inference.analysis.audit import (
    STATUS_FAIL,
    STATUS_OK,
    STATUS_WARN,
    run_audit,
)
from adaptive_inference.analysis.loaders import (
    load_loaded_systems,
    load_phase6_manifest,
)
from adaptive_inference.calibration.artifact import load_frozen_budgets


def _audit(fix, *, splits_are_identical_acknowledged: bool = False):
    manifest = load_phase6_manifest(fix.phase6_manifest_path)
    systems = load_loaded_systems(manifest, repo_root=fix.root)
    frozen = load_frozen_budgets(fix.frozen_budgets_path)
    return run_audit(
        manifest=manifest,
        systems=systems,
        frozen=frozen,
        repo_root=fix.root,
        calibration_split_path=fix.calibration_split_path,
        held_out_split_path=fix.held_out_split_path,
        frozen_budgets_path=fix.frozen_budgets_path,
        splits_are_identical_acknowledged=splits_are_identical_acknowledged,
    )


def _by_name(report, name):
    for c in report.checks:
        if c.name == name:
            return c
    raise KeyError(name)


def test_happy_path_status_distribution(phase7_fixture):
    report = _audit(phase7_fixture)
    assert not report.any_failed
    # Two warns: stub-adapter accuracy gate + degenerate seed variance
    statuses = {c.name: c.status for c in report.checks}
    assert statuses["accuracy_status"] == STATUS_WARN
    assert statuses["random_baseline_seeds_distinct"] == STATUS_WARN
    # Everything else is ok
    for name, status in statuses.items():
        if name in {"accuracy_status", "random_baseline_seeds_distinct"}:
            continue
        assert status == STATUS_OK, f"{name} unexpectedly returned {status}"


def test_splits_disjoint_fails_on_overlap(phase7_fixture):
    # Inject the held-out's first page into the calibration split.
    cal_path = phase7_fixture.calibration_split_path
    raw = json.loads(cal_path.read_text())
    raw["page_ids"].append(phase7_fixture.held_out_page_ids[0])
    cal_path.write_text(json.dumps(raw, indent=2, sort_keys=True))

    report = _audit(phase7_fixture)
    assert _by_name(report, "phase1_splits_disjoint").status == STATUS_FAIL


def test_splits_identical_acknowledged_downgrades_to_warn(phase7_fixture):
    # Overwrite the calibration split to match held_out exactly.
    cal_path = phase7_fixture.calibration_split_path
    cal_raw = json.loads(cal_path.read_text())
    cal_raw["page_ids"] = list(phase7_fixture.held_out_page_ids)
    cal_path.write_text(json.dumps(cal_raw, indent=2, sort_keys=True))

    # Without the acknowledgment, the audit still fails.
    report = _audit(phase7_fixture, splits_are_identical_acknowledged=False)
    assert _by_name(report, "phase1_splits_disjoint").status == STATUS_FAIL

    # With the acknowledgment, it becomes a warn — and the report no longer fails.
    report = _audit(phase7_fixture, splits_are_identical_acknowledged=True)
    chk = _by_name(report, "phase1_splits_disjoint")
    assert chk.status == STATUS_WARN
    assert "MVP shortcut" in chk.detail
    assert not report.any_failed


def test_held_out_sha_mismatch_fails(phase7_fixture):
    # Mutate the held-out file *after* the manifest header was written.
    held = phase7_fixture.held_out_split_path
    raw = json.loads(held.read_text())
    raw["page_ids"].append("extra_page")
    held.write_text(json.dumps(raw, indent=2, sort_keys=True))

    report = _audit(phase7_fixture)
    assert _by_name(report, "phase6_manifest_sha_matches_held_out").status == STATUS_FAIL


def test_frozen_artifact_modified_after_phase6_fails(phase7_fixture):
    frozen = phase7_fixture.frozen_budgets_path
    raw = json.loads(frozen.read_text())
    raw["budgets"]["B_low"]["max_tiles"] = 99  # tamper
    frozen.write_text(json.dumps(raw, indent=2, sort_keys=True))

    report = _audit(phase7_fixture)
    # Both SHA checks catch this; we assert the dedicated read-only contract:
    assert _by_name(report, "frozen_artifact_unchanged_by_phase6").status == STATUS_FAIL


def test_missing_required_family_fails(phase7_fixture):
    # Drop the adaptive_2b entry from the manifest.
    raw = json.loads(phase7_fixture.phase6_manifest_path.read_text())
    raw["entries"] = [e for e in raw["entries"] if e["system_id"] != "adaptive_2b"]
    phase7_fixture.phase6_manifest_path.write_text(
        json.dumps(raw, indent=2, sort_keys=True)
    )

    report = _audit(phase7_fixture)
    assert _by_name(report, "phase6_entries_complete").status == STATUS_FAIL


def test_log_page_id_leak_from_calibration_fails(phase7_fixture):
    """A held-out log emitting a calibration page_id must trip the audit."""

    log = phase7_fixture.phase6_dir / "fixed_2b_low" / "run.log.jsonl"
    text = log.read_text(encoding="utf-8")
    rec = json.loads(text.splitlines()[0])
    rec["page_id"] = "page_c"  # the calibration-only page id from conftest
    log.write_text(
        json.dumps(rec, sort_keys=True) + "\n" + "\n".join(text.splitlines()[1:]) + "\n",
        encoding="utf-8",
    )

    report = _audit(phase7_fixture)
    assert _by_name(report, "phase6_log_pages_match_held_out").status == STATUS_FAIL


def test_prompt_id_drift_fails(phase7_fixture):
    log = phase7_fixture.phase6_dir / "adaptive_2b" / "run.log.jsonl"
    text = log.read_text(encoding="utf-8")
    rec = json.loads(text.splitlines()[0])
    rec["prompt_id"] = "drifted_v2"
    log.write_text(
        json.dumps(rec, sort_keys=True) + "\n" + "\n".join(text.splitlines()[1:]) + "\n",
        encoding="utf-8",
    )

    report = _audit(phase7_fixture)
    assert _by_name(report, "prompt_id_pinned").status == STATUS_FAIL


def test_unknown_verifier_code_fails(phase7_fixture):
    log = phase7_fixture.phase6_dir / "adaptive_2b" / "run.log.jsonl"
    text = log.read_text(encoding="utf-8")
    rec = json.loads(text.splitlines()[0])
    rec["verifier_failure_codes"] = ["NOT_A_REAL_CODE"]
    log.write_text(
        json.dumps(rec, sort_keys=True) + "\n" + "\n".join(text.splitlines()[1:]) + "\n",
        encoding="utf-8",
    )

    report = _audit(phase7_fixture)
    assert _by_name(report, "verifier_decision_codes_known").status == STATUS_FAIL


def test_pages_processed_mismatch_fails(phase7_fixture):
    raw = json.loads(phase7_fixture.phase6_manifest_path.read_text())
    for entry in raw["entries"]:
        if entry["system_id"] == "adaptive_2b":
            entry["pages_processed"] = 99
            break
    phase7_fixture.phase6_manifest_path.write_text(
        json.dumps(raw, indent=2, sort_keys=True)
    )
    # Recompute the SHAs the manifest is about — this only tampers with
    # the entries block, not with referenced files. Rewrite the SHA block
    # to keep the SHA checks green so we isolate the disk-mismatch check.
    raw2 = json.loads(phase7_fixture.phase6_manifest_path.read_text())
    # SHAs of held_out / frozen are derived elsewhere; re-write to current.
    from adaptive_inference.experiment.manifest import sha256_of_file
    raw2["header"]["held_out_manifest_sha256"] = sha256_of_file(
        phase7_fixture.held_out_split_path
    )
    raw2["header"]["frozen_budgets_sha256"] = sha256_of_file(
        phase7_fixture.frozen_budgets_path
    )
    phase7_fixture.phase6_manifest_path.write_text(
        json.dumps(raw2, indent=2, sort_keys=True)
    )

    report = _audit(phase7_fixture)
    assert _by_name(report, "phase6_entries_match_disk").status == STATUS_FAIL
