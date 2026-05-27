"""Download a FinTabNet subset and emit project-shaped fixture files.

Target HF repo (default): ``bsmock25/FinTabNet.c`` — Brandon Smock's
cleaned FinTabNet variant used in TATR. See ``configs/dataset/fintabnet.yaml``.

Two-phase usage on the VM:

1. **Discovery**:

       python scripts/data/build_fintabnet_fixture.py --list-files

   Prints every file in the HF repo. Use the output to identify the
   annotation JSONL path and the image (or PDF) folder prefix.

2. **Build** (once paths are known):

       python scripts/data/build_fintabnet_fixture.py \
           --annotations-jsonl FinTabNet.c_PDF_Annotations_JSON.jsonl \
           --images-prefix images/ \
           --limit 250

   Writes:
   - ``data/fintabnet/images/<page_id>.png`` — page images.
   - ``data/fintabnet/records.json`` — list of ``PageRecord`` dicts.
   - ``data/fintabnet/ground_truth.json`` — ``{page_id: html_str}``.
   - ``data/fintabnet/manifest.jsonl`` — one line per page with full
     metadata (row_count, col_count, has_merged_cells, image_path).

The selection logic itself is in
``adaptive_inference.dataset.fintabnet`` and is unit-tested with no
network access. This script is the thin I/O wrapper.

For PDF-shipped variants, pass ``--pdfs-prefix pdf/`` and the script
will render each referenced PDF page to PNG via PyMuPDF.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable


DEFAULT_REPO_ID = "bsmock25/FinTabNet.c"
DEFAULT_REPO_TYPE = "dataset"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--repo-type", default=DEFAULT_REPO_TYPE)
    parser.add_argument(
        "--list-files", action="store_true",
        help="Print every file in the repo and exit. Use first to discover paths.",
    )
    parser.add_argument(
        "--annotations-jsonl", default=None,
        help="Repo-relative path to the annotation JSONL (one entry per line).",
    )
    parser.add_argument(
        "--images-prefix", default=None,
        help="Repo-relative folder containing page images (e.g. 'images/').",
    )
    parser.add_argument(
        "--pdfs-prefix", default=None,
        help="Alternative to --images-prefix: render PDFs at --dpi via PyMuPDF.",
    )
    parser.add_argument("--dpi", type=int, default=144, help="PDF render DPI.")
    parser.add_argument("--limit", type=int, default=250)
    parser.add_argument(
        "--splits", nargs="+", default=["test"],
        help="FinTabNet splits to include (default: test).",
    )
    parser.add_argument(
        "--min-non-empty-cells", type=int, default=4,
        help="Drop pages with fewer than N text-filled gold cells (default 4).",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/fintabnet"))
    parser.add_argument(
        "--inspect", type=int, default=0, metavar="N",
        help="Print first N annotation entries (after JSONL load) and exit.",
    )
    args = parser.parse_args(argv)

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        print("Install huggingface_hub:  pip install huggingface_hub", file=sys.stderr)
        return 2

    if args.list_files:
        api = HfApi()
        files = api.list_repo_files(repo_id=args.repo_id, repo_type=args.repo_type)
        for f in files:
            print(f)
        return 0

    if not args.annotations_jsonl:
        print(
            "--annotations-jsonl is required (run with --list-files first to discover).",
            file=sys.stderr,
        )
        return 2

    # Lazy import: keeps unit tests independent of FinTabNet deps.
    from adaptive_inference.dataset.fintabnet import (
        select_english_table_pages,
    )

    local_jsonl = Path(
        hf_hub_download(
            repo_id=args.repo_id,
            filename=args.annotations_jsonl,
            repo_type=args.repo_type,
        )
    )
    print(f"loaded annotations: {local_jsonl}")
    entries = list(_iter_jsonl(local_jsonl))
    print(f"total annotation entries: {len(entries)}")

    if args.inspect > 0:
        for i, entry in enumerate(entries[: args.inspect]):
            print(f"=== entry {i} ===")
            print(json.dumps(_describe(entry), indent=2, ensure_ascii=False))
        return 0

    selected = select_english_table_pages(
        entries,
        limit=args.limit,
        splits=set(args.splits),
        min_non_empty_cells=args.min_non_empty_cells,
    )
    if not selected:
        print(
            "No pages selected. Re-run with --inspect 3 to dump entry shape, "
            "or check --splits values.",
            file=sys.stderr,
        )
        return 4
    print(f"selected {len(selected)} FinTabNet pages")

    images_dir = args.out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    ground_truth: dict[str, str] = {}
    manifest_lines: list[str] = []
    skipped: list[tuple[str, str]] = []

    for page in selected:
        try:
            local_img = _materialize_page_image(
                hf_hub_download,
                page.image_filename,
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                images_prefix=args.images_prefix,
                pdfs_prefix=args.pdfs_prefix,
                dpi=args.dpi,
            )
        except RuntimeError as exc:
            skipped.append((page.page_id, str(exc)))
            print(f"  SKIP {page.page_id}: {exc}", file=sys.stderr)
            continue

        ext = ".png" if args.pdfs_prefix else (local_img.suffix.lower() or ".png")
        target = images_dir / f"{page.page_id}{ext}"
        if local_img.resolve() != target.resolve():
            shutil.copy2(local_img, target)
        rel_path = target.relative_to(args.out_dir.parent).as_posix()

        records.append(_to_record(page, rel_path))
        ground_truth[page.page_id] = page.table_html
        manifest_lines.append(json.dumps(_to_manifest_row(page, rel_path)))
        print(f"  {page.page_id} <- {page.image_filename}")

    if not records:
        print(
            "Every image materialization failed. Re-run with --list-files "
            "to verify --images-prefix or --pdfs-prefix.",
            file=sys.stderr,
        )
        return 5
    if skipped:
        print(f"\nWARNING: skipped {len(skipped)} pages")

    (args.out_dir / "records.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    (args.out_dir / "ground_truth.json").write_text(
        json.dumps(ground_truth, indent=2), encoding="utf-8"
    )
    (args.out_dir / "manifest.jsonl").write_text(
        "\n".join(manifest_lines) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {len(records)} records to {args.out_dir / 'records.json'}")
    print(f"wrote {len(ground_truth)} GT entries to {args.out_dir / 'ground_truth.json'}")
    print(f"wrote {len(manifest_lines)} manifest rows to {args.out_dir / 'manifest.jsonl'}")
    return 0


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _materialize_page_image(
    hf_hub_download,
    rel_name: str,
    *,
    repo_id: str,
    repo_type: str,
    images_prefix: str | None,
    pdfs_prefix: str | None,
    dpi: int,
) -> Path:
    if pdfs_prefix:
        # rel_name like "X/Y.pdf-2" (some FinTabNet entries encode page suffix)
        pdf_name, page_num = _split_pdf_page(rel_name)
        local_pdf = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=f"{pdfs_prefix}{pdf_name}",
                repo_type=repo_type,
            )
        )
        return _render_pdf_page(local_pdf, page_num=page_num, dpi=dpi)

    if not images_prefix:
        raise RuntimeError("must pass --images-prefix or --pdfs-prefix")

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=f"{images_prefix}{rel_name}",
            repo_type=repo_type,
        )
    )


def _split_pdf_page(rel: str) -> tuple[str, int]:
    """FinTabNet PDF entries sometimes encode page as 'Doc.pdf-3'. Default page 0."""

    if ".pdf-" in rel:
        base, _, page_str = rel.rpartition("-")
        try:
            return base, int(page_str)
        except ValueError:
            pass
    return rel, 0


def _render_pdf_page(pdf_path: Path, *, page_num: int, dpi: int) -> Path:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF not installed; pip install PyMuPDF"
        ) from exc

    doc = fitz.open(pdf_path)
    try:
        page = doc[page_num]
        pix = page.get_pixmap(dpi=dpi)
        out_path = pdf_path.with_suffix(f".p{page_num}.png")
        pix.save(out_path)
        return out_path
    finally:
        doc.close()


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


def _to_manifest_row(page, image_path: str) -> dict:
    return {
        "page_id": page.page_id,
        "image_path": image_path,
        "row_count": page.row_count,
        "col_count": page.col_count,
        "has_merged_cells": page.has_merged_cells,
        "has_nested_headers": page.has_nested_headers,
    }


def _describe(node: Any, max_str: int = 120) -> Any:
    if isinstance(node, dict):
        return {k: _describe(v, max_str) for k, v in node.items()}
    if isinstance(node, list):
        first = _describe(node[0], max_str) if node else None
        return {"_list_len": len(node), "_first": first}
    if isinstance(node, str):
        return node if len(node) <= max_str else node[: max_str - 1] + "…"
    return node


if __name__ == "__main__":
    raise SystemExit(main())
