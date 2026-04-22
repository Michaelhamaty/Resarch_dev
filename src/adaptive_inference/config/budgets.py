"""Named compute budgets backed by a YAML file.

A ``Budget`` is the single compute knob the adaptive system tunes (Phase 5
calibration freezes ``low``/``high`` values; Phase 4 reparse picks between
them). Phase 2 only logs the chosen budget — the stub adapter does not yet
honor the tile count — but the contract is in place so later phases can
plug in without rewriting callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class Budget:
    name: str
    max_tiles: int


def load_budgets(path: str | Path) -> dict[str, Budget]:
    """Load every named budget defined in ``path`` keyed by its name."""

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "budgets" not in raw:
        raise ValueError(
            f"Budget config at {path} must be a YAML mapping with a 'budgets' key"
        )
    budgets_section = raw["budgets"]
    if not isinstance(budgets_section, dict) or not budgets_section:
        raise ValueError(f"Budget config at {path} has empty or invalid 'budgets'")

    out: dict[str, Budget] = {}
    for name, body in budgets_section.items():
        out[str(name)] = _budget_from_mapping(str(name), body)
    return out


def load_budget(path: str | Path, name: str) -> Budget:
    """Load a single named budget from ``path`` or raise ``KeyError``."""

    budgets = load_budgets(path)
    if name not in budgets:
        raise KeyError(
            f"Budget '{name}' not found in {path}; available={sorted(budgets)}"
        )
    return budgets[name]


def _budget_from_mapping(name: str, data: Mapping[str, Any]) -> Budget:
    if not isinstance(data, Mapping) or "max_tiles" not in data:
        raise ValueError(
            f"Budget '{name}' must be a mapping with a 'max_tiles' field"
        )
    max_tiles = int(data["max_tiles"])
    if max_tiles <= 0:
        raise ValueError(f"Budget '{name}' has non-positive max_tiles={max_tiles}")
    return Budget(name=name, max_tiles=max_tiles)
