"""Typed config schemas. Each dataclass corresponds to one YAML file under ``configs/``."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ModelConfig:
    name: str
    hf_id: str
    dtype: str
    description: str = ""


@dataclass(frozen=True)
class PromptConfig:
    name: str
    version: int
    template: str
    description: str = ""


@dataclass(frozen=True)
class BudgetConfig:
    name: str
    model: str
    tile_budget: int
    max_new_tokens: int
    description: str = ""


SystemKind = Literal["adaptive", "fixed", "random_escalation"]


@dataclass(frozen=True)
class RunConfig:
    name: str
    system: SystemKind
    prompt: str
    first_pass_budget: str
    split: str
    seed: int
    reparse_budget: str | None = None
    description: str = ""
