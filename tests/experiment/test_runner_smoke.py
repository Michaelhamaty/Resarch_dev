"""End-to-end smoke tests for the Phase 6 experiment runner."""

from __future__ import annotations

import json
from pathlib import Path

from adaptive_inference.experiment.runner import Phase6Config, run_phase6


def _cfg_from_fixture(fixture, seeds=(0, 1, 2)) -> Phase6Config:
    return Phase6Config(
        run_set_id="phase6_smoke",
        frozen_budgets_path=fixture.frozen_budgets_path,
        split_name="held_out_eval_split",
        held_out_manifest_path=fixture.manifest_path,
        records_path=fixture.records_path,
        image_root=fixture.image_root,
        model_config_path=fixture.model_config_path,
        prompt_config_path=fixture.prompt_config_path,
        output_root=fixture.output_root,
        random_seeds=seeds,
    )


def test_all_seven_systems_run_with_stub_flag(phase6_fixture) -> None:
    cfg = _cfg_from_fixture(phase6_fixture)
    result = run_phase6(cfg, allow_stubbed_8b=True)

    assert result.manifest_path.exists()
    assert not result.any_failed

    system_ids = [e.system_id for e in result.entries]
    assert system_ids == [
        "adaptive_2b",
        "fixed_2b_low",
        "fixed_2b_matched",
        "random_2b_seed0",
        "random_2b_seed1",
        "random_2b_seed2",
        "fixed_8b_matched",
    ]
    for e in result.entries:
        assert e.status == "ok", f"{e.system_id} status={e.status} err={e.error}"

    # Every system produced outputs.
    root = phase6_fixture.output_root
    for sid in system_ids:
        assert (root / sid).exists(), f"output dir missing for {sid}"
        assert (root / sid / "run.log.jsonl").exists(), f"log missing for {sid}"


def test_adaptive_and_random_have_adaptive_layout(phase6_fixture) -> None:
    cfg = _cfg_from_fixture(phase6_fixture)
    run_phase6(cfg, allow_stubbed_8b=True)
    root = phase6_fixture.output_root
    for sid in ("adaptive_2b", "random_2b_seed0"):
        assert (root / sid / "first_pass" / "raw").exists()
        assert (root / sid / "final" / "raw").exists()
        assert (root / sid / "final" / "pages").exists()


def test_fixed_systems_have_single_pass_layout(phase6_fixture) -> None:
    cfg = _cfg_from_fixture(phase6_fixture)
    run_phase6(cfg, allow_stubbed_8b=True)
    root = phase6_fixture.output_root
    for sid in ("fixed_2b_low", "fixed_2b_matched", "fixed_8b_matched"):
        assert (root / sid / "raw").exists(), f"raw/ missing for {sid}"
        assert (root / sid / "pages").exists(), f"pages/ missing for {sid}"
        assert not (root / sid / "first_pass").exists(), f"{sid} has adaptive layout"


def test_manifest_header_includes_pinning(phase6_fixture) -> None:
    cfg = _cfg_from_fixture(phase6_fixture)
    result = run_phase6(cfg, allow_stubbed_8b=True)
    data = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    header = data["header"]
    assert header["held_out_manifest_path"] == str(phase6_fixture.manifest_path)
    assert header["frozen_budgets_path"] == str(phase6_fixture.frozen_budgets_path)
    assert len(header["held_out_manifest_sha256"]) == 64
    assert len(header["frozen_budgets_sha256"]) == 64
    assert header["random_seeds"] == [0, 1, 2]
    assert header["prompt_id"] == "table_parse_v1"


def test_manifest_records_budgets_from_frozen(phase6_fixture) -> None:
    cfg = _cfg_from_fixture(phase6_fixture)
    result = run_phase6(cfg, allow_stubbed_8b=True)
    by_id = {e.system_id: e for e in result.entries}
    # Budgets in the manifest match the frozen artifact values (6 / 8 / 6 / 6).
    assert by_id["adaptive_2b"].budget_low_max_tiles == 6
    assert by_id["adaptive_2b"].budget_high_max_tiles == 8
    assert by_id["fixed_2b_low"].budget_max_tiles == 6
    assert by_id["fixed_2b_matched"].budget_max_tiles == 6
    assert by_id["fixed_8b_matched"].budget_max_tiles == 6


def test_random_entries_record_seed_and_probability(phase6_fixture_reparse_rate_half) -> None:
    cfg = _cfg_from_fixture(phase6_fixture_reparse_rate_half)
    result = run_phase6(cfg, allow_stubbed_8b=True)
    by_id = {e.system_id: e for e in result.entries}
    for seed in (0, 1, 2):
        entry = by_id[f"random_2b_seed{seed}"]
        assert entry.seed == seed
        assert entry.random_probability == 0.5
        assert entry.random_probability_source.startswith("phase5_frozen_artifact")


def test_degenerate_reparse_rate_is_reported_in_notes(phase6_fixture) -> None:
    # phase6_fixture uses measured=6.0 → p=0.0 → degenerate.
    cfg = _cfg_from_fixture(phase6_fixture)
    result = run_phase6(cfg, allow_stubbed_8b=True)
    data = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert data["header"]["calibration_reparse_rate"] == 0.0
    assert data["header"]["calibration_reparse_rate_degenerate"] is True
    random_entries = [
        e for e in data["entries"] if e["family"] == "random_2b"
    ]
    for e in random_entries:
        assert any("0.0" in n or "never escalate" in n for n in e["notes"])


def test_systems_filter_reduces_runs(phase6_fixture) -> None:
    cfg = _cfg_from_fixture(phase6_fixture)
    result = run_phase6(
        cfg, allow_stubbed_8b=True, systems_filter=("adaptive_2b", "fixed_2b_low")
    )
    ids = [e.system_id for e in result.entries]
    assert ids == ["adaptive_2b", "fixed_2b_low"]


def test_filter_unknown_system_raises(phase6_fixture) -> None:
    import pytest

    cfg = _cfg_from_fixture(phase6_fixture)
    with pytest.raises(ValueError, match="Unknown"):
        run_phase6(cfg, systems_filter=("not_a_real_system",))


def test_per_system_runs_are_deterministic(phase6_fixture_reparse_rate_half) -> None:
    cfg = _cfg_from_fixture(phase6_fixture_reparse_rate_half)
    first = run_phase6(
        cfg, allow_stubbed_8b=True, systems_filter=("random_2b",)
    )
    # Re-run same seeds — final raw bytes should match.
    root = phase6_fixture_reparse_rate_half.output_root
    baseline: dict[str, str] = {}
    for seed in (0, 1, 2):
        for pid in ("page_0001", "page_0002", "page_0003"):
            p = root / f"random_2b_seed{seed}" / "final" / "raw" / f"{pid}.md"
            baseline[f"{seed}:{pid}"] = p.read_text(encoding="utf-8")

    second = run_phase6(
        cfg, allow_stubbed_8b=True, systems_filter=("random_2b",)
    )
    for seed in (0, 1, 2):
        for pid in ("page_0001", "page_0002", "page_0003"):
            p = root / f"random_2b_seed{seed}" / "final" / "raw" / f"{pid}.md"
            assert p.read_text(encoding="utf-8") == baseline[f"{seed}:{pid}"]

    assert not first.any_failed
    assert not second.any_failed
