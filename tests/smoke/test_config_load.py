from pathlib import Path

from adaptive_inference.config.loader import (
    load_budgets_file,
    load_model_config,
    load_prompt_config,
    load_run_config,
)

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def test_load_sample_configs():
    model_2b = load_model_config(CONFIGS / "models" / "internvl2_2b.yaml")
    model_8b = load_model_config(CONFIGS / "models" / "internvl2_8b.yaml")
    prompt = load_prompt_config(CONFIGS / "prompts" / "table_parse_v1.yaml")
    budgets = load_budgets_file(CONFIGS / "budgets" / "default.yaml")
    run = load_run_config(CONFIGS / "runs" / "smoke_adaptive_2b.yaml")

    assert model_2b.name == "internvl2_2b"
    assert model_8b.name == "internvl2_8b"
    assert prompt.version == 1

    assert run.system == "adaptive"
    assert run.first_pass_budget in budgets
    assert run.reparse_budget in budgets
    assert budgets[run.first_pass_budget].model == model_2b.name
    assert budgets[run.reparse_budget].model == model_2b.name
