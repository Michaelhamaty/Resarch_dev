"""Download a FinTabNet subset and emit project-shaped fixture files.

Target HF repo (default): ``apoidea/fintabnet-html`` — scale-up v2 LOCKED
source. CDLA-Permissive 1.0 by inheritance from the original FinTabNet
(Zheng et al. 2021). See ``configs/dataset/fintabnet.yaml`` and
``docs/runbooks/fintabnet_provenance.md``.

The apoidea mirror ships parquet shards under ``en/{train,validation}-*``
with two columns: ``image`` (struct of ``{bytes, path}``) and
``html_table`` (raw HTML string). Use ``--parquet-glob``:

    python scripts/data/build_fintabnet_fixture.py \\
        --parquet-glob 'en/validation-*.parquet' \\
        --splits validation \\
        --limit 250

Writes:
    - ``data/fintabnet/images/<page_id>.png`` — page images
    - ``data/fintabnet/records.json`` — list of ``PageRecord`` dicts
    - ``data/fintabnet/ground_truth.json`` — ``{page_id: html_str}``
    - ``data/fintabnet/manifest.jsonl`` — one line per page

Legacy token-shape paths (``--annotations-jsonl`` / ``--images-prefix``
/ ``--pdfs-prefix``) remain supported for PubTabNet-token mirrors.

The selection logic lives in
``adaptive_inference.dataset.fintabnet_html`` (HTML rows) and
``adaptive_inference.dataset.fintabnet`` (token streams); both are
unit-testable without network. This script is the thin I/O wrapper.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable


DEFAULT_REPO_ID = "apoidea/fintabnet-html"
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
    parser.add_argument(
        "--parquet-glob", default=None,
        help=(
            "Repo-relative glob for HTML-shape parquet shards "
            "(e.g. 'en/validation-*.parquet'). Triggers the apoidea/"
            "fintabnet-html ingest path; mutually exclusive with "
            "--annotations-jsonl."
        ),
    )
    parser.add_argument("--limit", type=int, default=250)
    parser.add_argument(
        "--splits", nargs="+", default=["validation"],
        help=(
            "Splits to include. For --parquet-glob this matches the "
            "parquet folder name (validation/train). For "
            "--annotations-jsonl this matches the entry's 'split' field."
        ),
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

    if args.parquet_glob and args.annotations_jsonl:
        print(
            "Pass only one of --parquet-glob or --annotations-jsonl.",
            file=sys.stderr,
        )
        return 2

    if args.parquet_glob:
        return _run_parquet_mode(args, hf_hub_download_fn=hf_hub_download,
                                 hf_api_fn=HfApi)

    if not args.annotations_jsonl:
        print(
            "Pass --parquet-glob (apoidea HTML mirror) or --annotations-jsonl "
            "(legacy token-shape mirror). Run with --list-files first to "
            "discover repo paths.",
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


def _run_parquet_mode(args, *, hf_hub_download_fn, hf_api_fn) -> int:
    """Apoidea/fintabnet-html parquet ingest.

    Resolves ``args.parquet_glob`` against the repo file list, downloads
    matching shards via ``hf_hub_download``, streams each row through
    the HTML-shape selector, writes embedded PNG bytes to disk, and
    emits the same ``records.json`` / ``ground_truth.json`` /
    ``manifest.jsonl`` triad as the legacy path.
    """

    import fnmatch

    try:
        import pyarrow.parquet as pq
    except ImportError:
        print(
            "Install pyarrow for parquet ingest:  uv pip install pyarrow",
            file=sys.stderr,
        )
        return 2

    # Lazy import so unit tests don't need pyarrow.
    from adaptive_inference.dataset.fintabnet_html import (
        HtmlRow,
        select_html_pages,
    )

    api = hf_api_fn()
    all_files = api.list_repo_files(
        repo_id=args.repo_id, repo_type=args.repo_type
    )
    shard_paths = sorted(fnmatch.filter(all_files, args.parquet_glob))
    if not shard_paths:
        print(
            f"No parquet shards matched glob {args.parquet_glob!r} in "
            f"{args.repo_id!r}. Run --list-files to verify.",
            file=sys.stderr,
        )
        return 4
    print(f"matched {len(shard_paths)} parquet shard(s):")
    for sp in shard_paths:
        print(f"  {sp}")

    def _row_stream():
        for shard in shard_paths:
            local = Path(
                hf_hub_download_fn(
                    repo_id=args.repo_id,
                    filename=shard,
                    repo_type=args.repo_type,
                )
            )
            split = _split_name_from_shard_path(shard)
            print(f"  reading {shard} (split={split})")
            pf = pq.ParquetFile(local)
            for batch in pf.iter_batches(batch_size=128):
                cols = {n: batch.column(n) for n in batch.schema.names}
                for i in range(batch.num_rows):
                    image_cell = cols["image"][i].as_py()
                    image_bytes = image_cell.get("bytes") if isinstance(image_cell, dict) else None
                    if not isinstance(image_bytes, (bytes, bytearray)):
                        continue
                    html_val = cols["html_table"][i].as_py()
                    if not isinstance(html_val, str):
                        continue
                    yield HtmlRow(
                        html_table=html_val,
                        image_bytes=bytes(image_bytes),
                        split=split,
                    )

    if args.inspect > 0:
        for i, row in zip(range(args.inspect), _row_stream()):
            print(f"=== row {i} (split={row.split}) ===")
            print(f"image_bytes: {len(row.image_bytes)} bytes")
            html_preview = row.html_table[:300].replace("\n", " ")
            print(f"html_table[:300]: {html_preview!r}")
        return 0

    # Materialize rows so we can iterate twice: once for selection, once
    # for image-bytes lookup. For 250-page caps this is fine in memory.
    rows = list(_row_stream())
    print(f"streamed {len(rows)} candidate rows from parquet shards")

    selected = select_html_pages(
        rows,
        limit=args.limit,
        splits=set(args.splits),
        min_non_empty_cells=args.min_non_empty_cells,
    )
    if not selected:
        print(
            "No pages selected. Check --splits and --min-non-empty-cells.",
            file=sys.stderr,
        )
        return 4
    print(f"selected {len(selected)} FinTabNet pages")

    # Build a content-hash → image_bytes map so we can write the right
    # PNG for each SelectedPage. The selector hashes html_table for the
    # page_id, so look up by html_table.
    from adaptive_inference.dataset.fintabnet_html import _page_id_from_html
    image_bytes_by_id: dict[str, bytes] = {}
    for row in rows:
        image_bytes_by_id.setdefault(
            _page_id_from_html(row.html_table), row.image_bytes
        )

    images_dir = args.out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    ground_truth: dict[str, str] = {}
    manifest_lines: list[str] = []

    for page in selected:
        img_bytes = image_bytes_by_id.get(page.page_id)
        if img_bytes is None:
            print(f"  SKIP {page.page_id}: no image bytes", file=sys.stderr)
            continue
        target = images_dir / f"{page.page_id}.png"
        target.write_bytes(img_bytes)
        rel_path = target.relative_to(args.out_dir.parent).as_posix()
        records.append(_to_record(page, rel_path))
        ground_truth[page.page_id] = page.table_html
        manifest_lines.append(json.dumps(_to_manifest_row(page, rel_path)))
        print(f"  {page.page_id} <- {len(img_bytes)} bytes")

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


def _split_name_from_shard_path(shard_path: str) -> str:
    """Infer the split label from a parquet path like 'en/validation-00000-of-00003.parquet'."""

    stem = shard_path.rsplit("/", 1)[-1]
    # e.g. "validation-00000-of-00003.parquet" → "validation"
    return stem.split("-", 1)[0]


if __name__ == "__main__":
    raise SystemExit(main())
