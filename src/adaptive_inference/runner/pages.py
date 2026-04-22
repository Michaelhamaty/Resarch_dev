"""Page loader: resolve a manifest's page IDs into ``(PageRecord, Image)`` pairs.

Reads page IDs from a Phase 1 manifest, looks up each ID in the records
JSON, and opens the referenced image with Pillow. Keeping this logic
isolated means swapping real OmniDocBench images in later only touches
the ``image_root`` argument — the adapter boundary stays unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from ..dataset.records import PageRecord, load_page_records


@dataclass(frozen=True)
class LoadedPage:
    record: PageRecord
    image: Image.Image


def load_manifest_page_ids(manifest_path: str | Path) -> list[str]:
    """Return the list of page IDs declared by a manifest JSON file."""

    raw = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "page_ids" not in raw:
        raise ValueError(
            f"Manifest at {manifest_path} missing 'page_ids' mapping"
        )
    page_ids = raw["page_ids"]
    if not isinstance(page_ids, list):
        raise ValueError(f"Manifest at {manifest_path} has non-list page_ids")
    return [str(pid) for pid in page_ids]


def load_pages_for_manifest(
    manifest_path: str | Path,
    records_path: str | Path,
    image_root: str | Path,
) -> list[LoadedPage]:
    """Load every page listed in ``manifest_path`` as ``LoadedPage`` objects.

    Raises ``KeyError`` if the manifest references a page ID missing from
    ``records_path``. Raises ``FileNotFoundError`` if any image file is
    absent — the smoke path should never silently skip pages.
    """

    page_ids = load_manifest_page_ids(manifest_path)
    records_by_id = {r.page_id: r for r in load_page_records(records_path)}
    image_root_path = Path(image_root)

    missing = [pid for pid in page_ids if pid not in records_by_id]
    if missing:
        raise KeyError(
            f"Manifest references page IDs not in {records_path}: {missing}"
        )

    loaded: list[LoadedPage] = []
    for pid in page_ids:
        record = records_by_id[pid]
        image_path = image_root_path / record.image_path
        if not image_path.exists():
            raise FileNotFoundError(
                f"Image for page {pid} not found at {image_path}"
            )
        image = Image.open(image_path)
        image.load()  # force decode now; closes the file handle
        loaded.append(LoadedPage(record=record, image=image))
    return loaded
