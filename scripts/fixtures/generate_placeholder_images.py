"""Generate tiny placeholder PNGs for the fixture page records.

The Phase 1 fixture (`data/fixtures/sample_pages.json`) declares
`image_path` fields like `fixtures/images/page_0001.png`, but those PNGs
do not exist. Phase 2's page loader needs real files on disk, so this
script creates a tiny 8x8 grayscale PNG for every page referenced by the
fixture. The images are intentionally trivial — the stub adapter ignores
their content — but they exist, so the loader contract is exercised.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECORDS = REPO_ROOT / "data" / "fixtures" / "sample_pages.json"
# PageRecord.image_path is stored as "fixtures/images/{page_id}.png", so the
# image root is `data/` — image_root / image_path resolves under `data/fixtures/images/`.
DEFAULT_IMAGE_ROOT = REPO_ROOT / "data"


def generate(records_path: Path, image_root: Path) -> list[Path]:
    raw = json.loads(records_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON list at {records_path}")

    written: list[Path] = []
    for item in raw:
        image_path = image_root / item["image_path"]
        image_path.parent.mkdir(parents=True, exist_ok=True)
        # Tiny 8x8 grayscale image; deterministic content so re-runs are stable.
        img = Image.new("L", (8, 8), color=128)
        img.save(image_path, format="PNG")
        written.append(image_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        type=Path,
        default=DEFAULT_RECORDS,
        help="Path to sample_pages.json (default: %(default)s)",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=DEFAULT_IMAGE_ROOT,
        help="Root directory under which PageRecord.image_path is resolved",
    )
    args = parser.parse_args()

    written = generate(args.records, args.image_root)
    print(f"Wrote {len(written)} placeholder images under {args.image_root}")


if __name__ == "__main__":
    main()
