"""Unit tests for the Phase 6 random-escalation adaptive runner."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from adaptive_inference.config.budgets import Budget
from adaptive_inference.config.models import ModelConfig
from adaptive_inference.config.prompts import PromptTemplate
from adaptive_inference.runner import adaptive_random as ar_module
from adaptive_inference.runner.adaptive_random import (
    AdaptiveRandomRunConfig,
    run_adaptive_random,
)


def _build_cfg(
    tmp_path: Path, page_ids: list[str], *, seed: int, probability: float
) -> AdaptiveRandomRunConfig:
    image_root = tmp_path / "data"
    images_dir = image_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for pid in page_ids:
        Image.new("L", (4, 4), color=0).save(images_dir / f"{pid}.png", format="PNG")

    records = tmp_path / "records.json"
    records.write_text(
        json.dumps(
            [
                {
                    "page_id": pid,
                    "image_path": f"images/{pid}.png",
                    "language": "en",
                    "contains_table": True,
                    "is_english_table_page": True,
                }
                for pid in page_ids
            ]
        ),
        encoding="utf-8",
    )

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"page_ids": page_ids}),
        encoding="utf-8",
    )

    return AdaptiveRandomRunConfig(
        run_id="test_random",
        split_name="held_out_eval_split",
        manifest_path=manifest,
        records_path=records,
        image_root=image_root,
        model_cfg=ModelConfig(
            name="internvl2-2b", adapter_kind="stub", model_id="x", notes=""
        ),
        budget_low=Budget(name="low", max_tiles=4),
        budget_high=Budget(name="high", max_tiles=12),
        prompt=PromptTemplate(
            id="table_parse_v1", version=1, description="d", template="t"
        ),
        output_dir=tmp_path / "out",
        seed=seed,
        probability=probability,
    )


def test_probability_zero_never_reparses(tmp_path: Path) -> None:
    cfg = _build_cfg(tmp_path, ["page_0001", "page_0002"], seed=0, probability=0.0)
    summary = run_adaptive_random(cfg)
    assert summary.reparse_count == 0
    for pid in ("page_0001", "page_0002"):
        assert not (cfg.output_dir / "reparse" / "raw" / f"{pid}.md").exists()
        assert (cfg.output_dir / "final" / "raw" / f"{pid}.md").exists()


def test_probability_one_always_reparses(tmp_path: Path) -> None:
    cfg = _build_cfg(tmp_path, ["page_0001", "page_0002"], seed=0, probability=1.0)
    summary = run_adaptive_random(cfg)
    assert summary.reparse_count == 2
    for pid in ("page_0001", "page_0002"):
        assert (cfg.output_dir / "reparse" / "raw" / f"{pid}.md").exists()


def test_log_contains_random_policy_fields(tmp_path: Path) -> None:
    cfg = _build_cfg(tmp_path, ["page_0001"], seed=42, probability=1.0)
    summary = run_adaptive_random(cfg)
    record = json.loads(
        summary.log_path.read_text(encoding="utf-8").strip().splitlines()[0]
    )
    assert record["random_seed"] == 42
    assert record["random_probability"] == 1.0
    assert record["policy"] == "random_escalation"
    assert record["reparse_triggered"] is True
    # Verifier decision is still recorded even though it's not used for routing.
    assert "verifier_decision" in record


def test_different_seeds_can_produce_different_reparse_counts(tmp_path: Path) -> None:
    # Use a medium probability and enough pages to detect divergence.
    page_ids = [f"page_{i:04d}" for i in range(12)]
    cfg_a = _build_cfg(tmp_path / "a", page_ids, seed=0, probability=0.5)
    cfg_b = _build_cfg(tmp_path / "b", page_ids, seed=1, probability=0.5)
    a = run_adaptive_random(cfg_a)
    b = run_adaptive_random(cfg_b)
    # At minimum, *some* page should disagree across seeds. The strict
    # assertion is that per-page decisions differ somewhere — we read logs.
    a_lines = [
        json.loads(line) for line in a.log_path.read_text(encoding="utf-8").splitlines()
    ]
    b_lines = [
        json.loads(line) for line in b.log_path.read_text(encoding="utf-8").splitlines()
    ]
    a_map = {r["page_id"]: r["reparse_triggered"] for r in a_lines}
    b_map = {r["page_id"]: r["reparse_triggered"] for r in b_lines}
    assert a_map != b_map


def test_rerun_produces_identical_log_lines_modulo_timing(tmp_path: Path) -> None:
    # Same seed + same probability + deterministic stub ⇒ same per-page
    # routing decisions on every run. Runtime fields are dropped before diff.
    cfg = _build_cfg(tmp_path, ["page_0001", "page_0002"], seed=0, probability=1.0)
    s1 = run_adaptive_random(cfg)
    first = [
        _strip_timing(json.loads(line))
        for line in s1.log_path.read_text(encoding="utf-8").splitlines()
    ]
    s2 = run_adaptive_random(cfg)
    second = [
        _strip_timing(json.loads(line))
        for line in s2.log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert first == second


def test_empty_manifest_materializes_log(tmp_path: Path) -> None:
    cfg = _build_cfg(tmp_path, [], seed=0, probability=0.5)
    summary = run_adaptive_random(cfg)
    assert summary.pages_processed == 0
    assert summary.log_path.exists()
    assert summary.log_path.read_text(encoding="utf-8") == ""


def test_adapter_called_at_most_twice_per_page(tmp_path: Path, monkeypatch) -> None:
    cfg = _build_cfg(tmp_path, ["page_0001", "page_0002"], seed=0, probability=1.0)

    calls: list[tuple[str, str]] = []
    real_build = ar_module.build_adapter

    def counting_build_adapter(model_cfg):
        adapter = real_build(model_cfg)
        real_run = adapter.run

        def wrapped_run(*, page_id, image, budget, prompt):
            calls.append((page_id, budget.name))
            return real_run(page_id=page_id, image=image, budget=budget, prompt=prompt)

        adapter.run = wrapped_run  # type: ignore[method-assign]
        return adapter

    monkeypatch.setattr(ar_module, "build_adapter", counting_build_adapter)
    run_adaptive_random(cfg)

    from collections import Counter

    per_page = Counter(page_id for page_id, _ in calls)
    assert per_page == {"page_0001": 2, "page_0002": 2}


def _strip_timing(rec: dict) -> dict:
    for k in (
        "first_pass_runtime_ms",
        "reparse_runtime_ms",
        "total_runtime_ms",
        "verifier_runtime_ms",
    ):
        rec.pop(k, None)
    # Also drop timestamps from inference — they are wall-clock.
    for rec_ts_key in ("started_at", "finished_at"):
        rec.pop(rec_ts_key, None)
    return rec
