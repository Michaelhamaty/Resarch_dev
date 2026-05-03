"""Download an OmniDocBench subset and emit project-shaped fixture files.

Run on Colab (or any host with internet + huggingface_hub installed):

    python scripts/data/build_omnidocbench_fixture.py --limit 14

What it does:
1. Pulls the OmniDocBench annotations JSON from
   ``opendatalab/OmniDocBench`` on HuggingFace.
2. Selects English pages with at least one HTML table (deterministic
   sort by image filename, truncated to ``--limit``).
3. Downloads only the selected page images.
4. Writes:
   - ``data/omnidocbench/images/<page_id>.<ext>`` — raw images.
   - ``data/omnidocbench/records.json`` — list of ``PageRecord`` dicts
     compatible with the existing Phase 1 loader.
   - ``data/omnidocbench/ground_truth.json`` — ``{page_id: html_str}``
     mapping the scorer will consume in a later step.

The selection logic itself lives in
``adaptive_inference.dataset.omnidocbench`` and is unit-tested without
any network access. This script is the thin I/O wrapper.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ID = "opendatalab/OmniDocBench"
REPO_TYPE = "dataset"
ANNOTATION_CANDIDATES = (
    "OmniDocBench.json",
    "annotations/OmniDocBench.json",
    "annotation/OmniDocBench.json",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=14,
        help="Maximum number of English-table pages to keep (default: 14).",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("data/omnidocbench"),
        help="Where to write images, records.json, and ground_truth.json.",
    )
    parser.add_argument(
        "--annotation-path", type=str, default=None,
        help="Override the in-repo annotation file path if upstream renames it.",
    )
    parser.add_argument(
        "--inspect", type=int, default=0, metavar="N",
        help=(
            "Diagnostic mode: download annotations and print a structural "
            "summary of the first N entries, then exit. Use this when the "
            "selector finds 0 pages so we can see what changed upstream."
        ),
    )
    args = parser.parse_args(argv)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(
            "huggingface_hub is not installed. Install it with:\n"
            "    pip install huggingface_hub",
            file=sys.stderr,
        )
        return 2

    # Lazy import so the unit tests do not need this module's deps.
    from adaptive_inference.dataset.omnidocbench import (
        SelectedPage,
        describe_entry,
        select_english_table_pages,
    )

    annotations_path = _download_annotations(
        hf_hub_download, override=args.annotation_path
    )
    print(f"loaded annotations: {annotations_path}")
    raw = json.loads(annotations_path.read_text(encoding="utf-8"))
    entries = raw if isinstance(raw, list) else raw.get("data") or raw.get("annotations")
    if not isinstance(entries, list):
        print(
            "Could not find a list of page entries in the annotations JSON. "
            "Top-level keys: " + ", ".join(sorted(raw)),
            file=sys.stderr,
        )
        return 3
    print(f"total annotation entries: {len(entries)}")

    if args.inspect > 0:
        print(f"\n--- inspecting first {args.inspect} entries ---\n")
        for i, entry in enumerate(entries[: args.inspect]):
            print(f"=== entry {i} ===")
            print(json.dumps(describe_entry(entry), indent=2, ensure_ascii=False))
            print()
        return 0

    selected = select_english_table_pages(entries, limit=args.limit)
    if not selected:
        print(
            "No English table pages found. Re-run with --inspect 2 to dump "
            "the first two entries' structure and report back.",
            file=sys.stderr,
        )
        return 4
    print(f"selected {len(selected)} English-table pages")

    images_dir = args.out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    ground_truth: dict[str, str] = {}
    for page in selected:
        local_path = _download_image(hf_hub_download, page)
        image_ext = local_path.suffix.lower() or ".jpg"
        target = images_dir / f"{page.page_id}{image_ext}"
        shutil.copy2(local_path, target)
        rel_path = target.relative_to(args.out_dir.parent).as_posix()
        records.append(_to_record(page, rel_path))
        ground_truth[page.page_id] = page.table_html
        print(f"  {page.page_id} <- {page.image_filename} -> {target}")

    (args.out_dir / "records.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    (args.out_dir / "ground_truth.json").write_text(
        json.dumps(ground_truth, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {len(records)} records to {args.out_dir / 'records.json'}")
    print(f"wrote {len(ground_truth)} GT entries to {args.out_dir / 'ground_truth.json'}")
    return 0


def _download_annotations(hf_hub_download, *, override: str | None):
    paths = (override,) if override else ANNOTATION_CANDIDATES
    last_error: Exception | None = None
    for path in paths:
        if not path:
            continue
        try:
            local = hf_hub_download(repo_id=REPO_ID, filename=path, repo_type=REPO_TYPE)
            return Path(local)
        except Exception as exc:  # pragma: no cover - depends on HF responses
            last_error = exc
            continue
    msg = (
        "Could not locate OmniDocBench annotations JSON. Tried: "
        + ", ".join(p for p in paths if p)
    )
    if last_error:
        msg += f"\nlast error: {last_error}"
    raise RuntimeError(msg)


def _download_image(hf_hub_download, page) -> Path:
    local = hf_hub_download(
        repo_id=REPO_ID,
        filename=page.image_filename,
        repo_type=REPO_TYPE,
    )
    return Path(local)


def _to_record(page, image_path: str) -> dict:
    return {
        "page_id": page.page_id,
        "image_path": image_path,
        "language": "en",
        "contains_table": True,
        "is_english_table_page": True,
        "row_count": page.row_count,
        "col_count": page.col_count,
        "has_merged_cells": page.has_merged_cells,
        "has_nested_headers": page.has_nested_headers,
    }


if __name__ == "__main__":
    raise SystemExit(main())
