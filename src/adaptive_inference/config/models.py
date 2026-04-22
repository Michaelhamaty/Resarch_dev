"""Model registry config (InternVL2-2B / -8B descriptors).

Each entry names which ``adapter_kind`` to instantiate. Phase 2 only ships
the ``stub`` kind; a future phase will add ``internvl2`` (real HF model
loader). Keeping the kind in YAML means swapping in the real model is one
config edit, not a code change.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class ModelConfig:
    name: str
    adapter_kind: str
    model_id: str
    notes: str = ""


def load_model_configs(path: str | Path) -> dict[str, ModelConfig]:
    """Load every model entry defined in ``path`` keyed by name."""

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "models" not in raw:
        raise ValueError(
            f"Model config at {path} must be a YAML mapping with a 'models' key"
        )
    models_section = raw["models"]
    if not isinstance(models_section, dict) or not models_section:
        raise ValueError(f"Model config at {path} has empty or invalid 'models'")

    out: dict[str, ModelConfig] = {}
    for name, body in models_section.items():
        out[str(name)] = _model_from_mapping(str(name), body)
    return out


def load_model_config(path: str | Path, name: str) -> ModelConfig:
    """Load a single model entry from ``path`` or raise ``KeyError``."""

    models = load_model_configs(path)
    if name not in models:
        raise KeyError(
            f"Model '{name}' not found in {path}; available={sorted(models)}"
        )
    return models[name]


def _model_from_mapping(name: str, data: Mapping[str, Any]) -> ModelConfig:
    if not isinstance(data, Mapping):
        raise ValueError(f"Model '{name}' must be a mapping")
    missing = [f for f in ("adapter_kind", "model_id") if f not in data]
    if missing:
        raise ValueError(f"Model '{name}' missing required fields {missing}")
    return ModelConfig(
        name=name,
        adapter_kind=str(data["adapter_kind"]),
        model_id=str(data["model_id"]),
        notes=str(data.get("notes", "")),
    )
