"""Manifest dataclass + stable JSON writer.

Each manifest is a self-describing JSON file with a pinning header and a
sorted list of page IDs. Serialization is deterministic (sorted keys, 2-space
indent, trailing newline) so manifests diff cleanly in version control.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .pinning import DatasetPinning


ManifestKind = Literal[
    "eval_universe",
    "hard_subset",
    "calibration_split",
    "held_out_eval_split",
]

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Manifest:
    manifest_kind: ManifestKind
    pinning: DatasetPinning
    page_ids: tuple[str, ...]
    generated_with_seed: int | None = None
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {
            "manifest_kind": self.manifest_kind,
            "schema_version": self.schema_version,
            "generated_with_seed": self.generated_with_seed,
            "pinning": self.pinning.to_dict(),
            "page_ids": sorted(self.page_ids),
        }


def write_manifest(manifest: Manifest, path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest.to_dict(), sort_keys=True, indent=2)
    out_path.write_text(payload + "\n", encoding="utf-8")
    return out_path
