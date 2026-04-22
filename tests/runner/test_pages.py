"""Tests for the manifest -> records -> images page loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from adaptive_inference.runner.pages import (
    load_manifest_page_ids,
    load_pages_for_manifest,
)


def _write_records(path: Path, entries: list[dict]) -> None:
    path.write_text(json.dumps(entries), encoding="utf-8")


def _write_manifest(path: Path, page_ids: list[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "manifest_kind": "calibration_split",
                "schema_version": 1,
                "generated_with_seed": 1,
                "pinning": {},
                "page_ids": page_ids,
            }
        ),
        encoding="utf-8",
    )


def _make_img(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (4, 4), color=0).save(path, format="PNG")


def test_load_manifest_page_ids_roundtrip(tmp_path: Path) -> None:
    mpath = tmp_path / "m.json"
    _write_manifest(mpath, ["page_0002", "page_0001"])
    # We preserve manifest order; callers may sort if they need alphabetical.
    assert load_manifest_page_ids(mpath) == ["page_0002", "page_0001"]


def test_load_pages_returns_records_and_images(tmp_path: Path) -> None:
    img_rel = "images/page_0001.png"
    img_abs = tmp_path / img_rel
    _make_img(img_abs)

    records = tmp_path / "records.json"
    _write_records(
        records,
        [
            {
                "page_id": "page_0001",
                "image_path": img_rel,
                "language": "en",
                "contains_table": True,
                "is_english_table_page": True,
            }
        ],
    )

    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, ["page_0001"])

    loaded = load_pages_for_manifest(manifest, records, tmp_path)
    assert len(loaded) == 1
    assert loaded[0].record.page_id == "page_0001"
    assert loaded[0].image.size == (4, 4)


def test_missing_image_raises(tmp_path: Path) -> None:
    records = tmp_path / "records.json"
    _write_records(
        records,
        [
            {
                "page_id": "page_0001",
                "image_path": "images/nope.png",
                "language": "en",
                "contains_table": True,
                "is_english_table_page": True,
            }
        ],
    )
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, ["page_0001"])
    with pytest.raises(FileNotFoundError):
        load_pages_for_manifest(manifest, records, tmp_path)


def test_manifest_refers_unknown_id(tmp_path: Path) -> None:
    records = tmp_path / "records.json"
    _write_records(records, [])
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, ["ghost"])
    with pytest.raises(KeyError, match="not in"):
        load_pages_for_manifest(manifest, records, tmp_path)
