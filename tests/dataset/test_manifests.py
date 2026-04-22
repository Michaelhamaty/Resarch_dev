from __future__ import annotations

import json
from pathlib import Path

from adaptive_inference.dataset import (
    DatasetPinning,
    Manifest,
    SCHEMA_VERSION,
    write_manifest,
)


PINNING = DatasetPinning(
    dataset_name="omnidocbench",
    snapshot_version="v1-test",
    eval_code_commit="TBD",
    source="fixture.json",
    notes="unit test",
)


def test_manifest_json_is_byte_stable(tmp_path: Path) -> None:
    manifest = Manifest(
        manifest_kind="eval_universe",
        pinning=PINNING,
        page_ids=("page_0003", "page_0001", "page_0002"),
    )
    path_a = write_manifest(manifest, tmp_path / "a.json")
    path_b = write_manifest(manifest, tmp_path / "b.json")
    assert path_a.read_bytes() == path_b.read_bytes()


def test_manifest_embeds_pinning_header(tmp_path: Path) -> None:
    manifest = Manifest(
        manifest_kind="calibration_split",
        pinning=PINNING,
        page_ids=("page_0001",),
        generated_with_seed=1337,
    )
    path = write_manifest(manifest, tmp_path / "m.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["manifest_kind"] == "calibration_split"
    assert data["generated_with_seed"] == 1337
    assert data["schema_version"] == SCHEMA_VERSION
    assert data["pinning"]["dataset_name"] == "omnidocbench"
    assert data["pinning"]["snapshot_version"] == "v1-test"


def test_manifest_page_ids_are_sorted(tmp_path: Path) -> None:
    manifest = Manifest(
        manifest_kind="hard_subset",
        pinning=PINNING,
        page_ids=("page_0009", "page_0001", "page_0005"),
    )
    path = write_manifest(manifest, tmp_path / "h.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["page_ids"] == ["page_0001", "page_0005", "page_0009"]


def test_manifest_ends_with_newline(tmp_path: Path) -> None:
    manifest = Manifest(
        manifest_kind="eval_universe",
        pinning=PINNING,
        page_ids=("page_0001",),
    )
    path = write_manifest(manifest, tmp_path / "m.json")
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")


def test_manifest_creates_missing_parent_dirs(tmp_path: Path) -> None:
    manifest = Manifest(
        manifest_kind="eval_universe",
        pinning=PINNING,
        page_ids=(),
    )
    nested = tmp_path / "a" / "b" / "c" / "m.json"
    path = write_manifest(manifest, nested)
    assert path.exists()
