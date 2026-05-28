"""Smoke tests for the Stage 9 figure generator.

Renders all three figures from a synthetic results_v2.json into a tmp
dir and asserts the PDFs land. Catches import / API breakage without a
real sweep.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "make_figures",
    Path(__file__).resolve().parents[2] / "scripts" / "scaleup" / "make_figures.py",
)
mf = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = mf
_SPEC.loader.exec_module(mf)  # type: ignore[union-attr]


def _synthetic_results() -> dict:
    def sysblk(mean, cost):
        return {
            "cost_tiles": cost,
            "cell_f1": {"mean": mean, "lo": mean - 0.05, "hi": mean + 0.05, "n": 100},
            "teds": {"mean": mean + 0.1, "lo": mean + 0.05, "hi": mean + 0.15, "n": 100},
            "cell_f1_by_bucket": {
                "simple": {"mean": mean + 0.1, "lo": mean + 0.05, "hi": mean + 0.15, "n": 40},
                "complex": {"mean": mean, "lo": mean - 0.05, "hi": mean + 0.05, "n": 40},
                "very_complex": {"mean": mean - 0.1, "lo": mean - 0.15, "hi": mean - 0.05, "n": 20},
            },
        }

    return {
        "datasets": {
            "fintabnet": {
                "status": "ok",
                "n_held_out": 100,
                "systems": {
                    "fixed_2b_low": sysblk(0.20, 2.0),
                    "fixed_2b_matched": sysblk(0.32, 10.0),
                    "adaptive_2b": sysblk(0.28, 9.6),
                },
                "comparisons": [],
            }
        }
    }


def test_all_three_figures_render(tmp_path):
    results_path = tmp_path / "results_v2.json"
    results_path.write_text(json.dumps(_synthetic_results()), encoding="utf-8")
    # A diagnostic file so figure 2 also renders.
    (tmp_path / "diagnostic_fintabnet.jsonl").write_text(
        "\n".join(
            json.dumps({
                "first_pass_cell_f1": 0.2 + 0.01 * i,
                "final_cell_f1": 0.2 + 0.01 * i + (0.1 if i % 2 else 0.0),
                "reparsed": bool(i % 2),
            })
            for i in range(10)
        ) + "\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "figures"
    rc = mf.main(["--results", str(results_path), "--out-dir", str(out_dir)])
    assert rc == 0
    for stem in ("cost_accuracy", "diagnostic_scatter", "stratified_bars"):
        assert (out_dir / f"{stem}.pdf").exists(), f"{stem}.pdf missing"
        assert (out_dir / f"{stem}.png").exists(), f"{stem}.png missing"


def test_diagnostic_skipped_when_absent(tmp_path, capsys):
    results_path = tmp_path / "results_v2.json"
    results_path.write_text(json.dumps(_synthetic_results()), encoding="utf-8")
    out_dir = tmp_path / "figures"
    rc = mf.main(["--results", str(results_path), "--out-dir", str(out_dir)])
    assert rc == 0
    # No diagnostic JSONL -> figure 2 skipped, others still produced.
    assert not (out_dir / "diagnostic_scatter.pdf").exists()
    assert (out_dir / "cost_accuracy.pdf").exists()
    assert (out_dir / "stratified_bars.pdf").exists()
