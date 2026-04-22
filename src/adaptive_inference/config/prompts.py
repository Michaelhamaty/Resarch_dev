"""Versioned prompt template dataclass + YAML loader.

The build brief's Contract 3 ("Reproducible prompts") says the prompt text
must be versioned and fixed across compared systems. Storing it in YAML
with an ``id`` and ``version`` makes that explicit: the loader rejects any
file missing those fields so a prompt can never be silently rewritten.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class PromptTemplate:
    id: str
    version: int
    description: str
    template: str

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "PromptTemplate":
        missing = [f for f in _REQUIRED_FIELDS if f not in data]
        if missing:
            raise ValueError(
                f"PromptTemplate missing required fields {missing} "
                f"(got keys={sorted(data.keys())})"
            )
        return cls(
            id=str(data["id"]),
            version=int(data["version"]),
            description=str(data["description"]),
            template=str(data["template"]),
        )


def load_prompt_template(path: str | Path) -> PromptTemplate:
    """Load a prompt YAML file into a ``PromptTemplate``."""

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Prompt config at {path} must be a YAML mapping")
    return PromptTemplate.from_mapping(raw)


_REQUIRED_FIELDS = ("id", "version", "description", "template")
