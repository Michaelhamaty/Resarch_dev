"""Tests for the 8B stub gate: Phase 6 must not silently fake the 8B baseline."""

from __future__ import annotations

import json

from adaptive_inference.experiment.runner import Phase6Config, run_phase6


def _cfg(fixture) -> Phase6Config:
    return Phase6Config(
        run_set_id="phase6_gate_test",
        frozen_budgets_path=fixture.frozen_budgets_path,
        split_name="held_out_eval_split",
        held_out_manifest_path=fixture.manifest_path,
        records_path=fixture.records_path,
        image_root=fixture.image_root,
        model_config_path=fixture.model_config_path,
        prompt_config_path=fixture.prompt_config_path,
        output_root=fixture.output_root,
        random_seeds=(0,),
    )


def test_8b_is_skipped_without_flag(phase6_fixture) -> None:
    cfg = _cfg(phase6_fixture)
    result = run_phase6(cfg, allow_stubbed_8b=False)
    by_id = {e.system_id: e for e in result.entries}
    entry = by_id["fixed_8b_matched"]
    assert entry.status == "skipped_stub_8b"
    assert entry.reason is not None
    assert "stub" in entry.reason.lower()
    # Skip does NOT flip the global failure flag.
    assert result.any_failed is False
    # No output directory was written for the skipped system.
    assert not (phase6_fixture.output_root / "fixed_8b_matched").exists()


def test_8b_runs_with_flag(phase6_fixture) -> None:
    cfg = _cfg(phase6_fixture)
    result = run_phase6(cfg, allow_stubbed_8b=True)
    by_id = {e.system_id: e for e in result.entries}
    entry = by_id["fixed_8b_matched"]
    assert entry.status == "ok"
    assert entry.output_dir is not None
    assert (phase6_fixture.output_root / "fixed_8b_matched" / "run.log.jsonl").exists()


def test_other_systems_complete_when_8b_is_skipped(phase6_fixture) -> None:
    cfg = _cfg(phase6_fixture)
    result = run_phase6(cfg, allow_stubbed_8b=False)
    for e in result.entries:
        if e.system_id == "fixed_8b_matched":
            continue
        assert e.status == "ok", f"{e.system_id} did not complete (status={e.status})"


def test_manifest_header_records_stub_adapter_for_skipped_8b(phase6_fixture) -> None:
    cfg = _cfg(phase6_fixture)
    result = run_phase6(cfg, allow_stubbed_8b=False)
    data = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    skipped = next(
        e for e in data["entries"] if e["system_id"] == "fixed_8b_matched"
    )
    assert skipped["adapter_kind"] == "stub"
    assert skipped["model_name"] == "internvl2-8b"
