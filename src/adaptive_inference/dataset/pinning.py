"""Dataset pinning metadata.

Every Phase 1 manifest embeds a ``DatasetPinning`` header so that a stray
manifest is still self-describing: what dataset it came from, which snapshot
version, which evaluation code commit, where the source lived, and any free-
form notes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class DatasetPinning:
    dataset_name: str
    snapshot_version: str
    eval_code_commit: str
    source: str
    notes: str

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "DatasetPinning":
        missing = [f for f in _REQUIRED_FIELDS if f not in data]
        if missing:
            raise ValueError(f"DatasetPinning missing required fields: {missing}")
        return cls(
            dataset_name=str(data["dataset_name"]),
            snapshot_version=str(data["snapshot_version"]),
            eval_code_commit=str(data["eval_code_commit"]),
            source=str(data["source"]),
            notes=str(data.get("notes", "")),
        )

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


_REQUIRED_FIELDS = (
    "dataset_name",
    "snapshot_version",
    "eval_code_commit",
    "source",
)
