"""End-to-end Phase 7 smoke test.

Build a complete on-disk Phase 6 fixture, run the orchestrator, assert
all expected output files land under the analysis directory and that
the audit report is healthy modulo the deliberate stub-adapter warns.
"""

from __future__ import annotations

import json

from adaptive_inference.analysis.config import Phase7Config
from adaptive_inference.analysis.runner import (
    ANALYSIS_MANIFEST,
    AUDIT_REPORT,
    COST_SUMMARY,
    QUALITATIVE,
    REPARSE_SUMMARY,
    RESULTS_TABLE,
    run_phase7,
)


def test_run_phase7_emits_all_artifacts(phase7_fixture):
    cfg = Phase7Config(
        phase6_manifest_path=phase7_fixture.phase6_manifest_path,
        frozen_budgets_path=phase7_fixture.frozen_budgets_path,
        held_out_split_path=phase7_fixture.held_out_split_path,
        calibration_split_path=phase7_fixture.calibration_split_path,
        output_root=phase7_fixture.output_root,
    )
    result = run_phase7(cfg, repo_root=phase7_fixture.root)

    out = result.output_dir
    for name in (
        ANALYSIS_MANIFEST,
        RESULTS_TABLE,
        COST_SUMMARY,
        REPARSE_SUMMARY,
        QUALITATIVE,
        AUDIT_REPORT,
    ):
        assert (out / name).exists(), f"missing {name}"

    assert not result.audit.any_failed
    # Stub adapters always produce the explicit accuracy warn.
    statuses = {c.name: c.status for c in result.audit.checks}
    assert statuses["accuracy_status"] == "warn"


def test_phase7_results_table_has_expected_systems(phase7_fixture):
    cfg = Phase7Config(
        phase6_manifest_path=phase7_fixture.phase6_manifest_path,
        frozen_budgets_path=phase7_fixture.frozen_budgets_path,
        held_out_split_path=phase7_fixture.held_out_split_path,
        calibration_split_path=phase7_fixture.calibration_split_path,
        output_root=phase7_fixture.output_root,
    )
    result = run_phase7(cfg, repo_root=phase7_fixture.root)

    payload = json.loads((result.output_dir / RESULTS_TABLE).read_text())
    assert payload["accuracy_status"] == "not_applicable_stub_adapters"
    ids = {s["system_id"] for s in payload["systems"]}
    assert ids == {
        "adaptive_2b",
        "fixed_2b_low",
        "fixed_2b_matched",
        "random_2b_seed0",
        "random_2b_seed1",
        "fixed_8b_matched",
    }


def test_phase7_qualitative_join_matches_held_out_pages(phase7_fixture):
    cfg = Phase7Config(
        phase6_manifest_path=phase7_fixture.phase6_manifest_path,
        frozen_budgets_path=phase7_fixture.frozen_budgets_path,
        held_out_split_path=phase7_fixture.held_out_split_path,
        calibration_split_path=phase7_fixture.calibration_split_path,
        output_root=phase7_fixture.output_root,
    )
    result = run_phase7(cfg, repo_root=phase7_fixture.root)

    lines = (result.output_dir / QUALITATIVE).read_text().splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    assert [r["page_id"] for r in rows] == list(phase7_fixture.held_out_page_ids)
    # Every row joins all 6 systems
    for row in rows:
        assert {s["system_id"] for s in row["systems"]} == {
            "adaptive_2b",
            "fixed_2b_low",
            "fixed_2b_matched",
            "random_2b_seed0",
            "random_2b_seed1",
            "fixed_8b_matched",
        }
