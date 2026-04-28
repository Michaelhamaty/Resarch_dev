"""Per-page qualitative join across systems.

Each row in the output JSONL covers one page on the held-out split and
records, for every system, what happened: which budget was used, whether
a reparse fired, what the verifier decided, and the relative artifact
path so a human can spot-check the on-disk markdown.

We deliberately do NOT inline the raw model output here. Full output
strings can be large and noisy, and the analysis layer is read-only.
The path pointer is enough for any reader who wants to open the
markdown file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Mapping

from .loaders import LoadedSystem
from .results import ADAPTIVE_RUNNERS, SINGLE_PASS_RUNNER


@dataclass(frozen=True)
class PerSystemPage:
    """One system's view of a single page."""

    system_id: str
    runner: str
    reparse_triggered: bool | None
    verifier_decision: str | None
    verifier_failure_codes: tuple[str, ...]
    final_output_source: str | None
    final_raw_path: str | None
    output_token_count: int
    runtime_ms: float


@dataclass(frozen=True)
class PageRow:
    """All systems' views of one page on the held-out split."""

    page_id: str
    systems: tuple[PerSystemPage, ...]


def build_page_rows(
    systems: tuple[LoadedSystem, ...], held_out_page_ids: tuple[str, ...]
) -> tuple[PageRow, ...]:
    """Join records by page_id; preserves held-out manifest order."""

    by_system: dict[str, dict[str, Mapping[str, object]]] = {}
    for sys in systems:
        if sys.entry.status != "ok":
            continue
        by_system[sys.entry.system_id] = _index_by_page_id(sys.records)

    rows: list[PageRow] = []
    for page_id in held_out_page_ids:
        per_sys: list[PerSystemPage] = []
        for sys in systems:
            if sys.entry.status != "ok":
                continue
            rec = by_system.get(sys.entry.system_id, {}).get(page_id)
            if rec is None:
                continue
            per_sys.append(_extract(sys.entry.runner, sys.entry.system_id, rec))
        rows.append(PageRow(page_id=page_id, systems=tuple(per_sys)))
    return tuple(rows)


def _index_by_page_id(
    records: tuple[Mapping[str, object], ...],
) -> dict[str, Mapping[str, object]]:
    return {str(r["page_id"]): r for r in records if "page_id" in r}


def _extract(runner: str, system_id: str, rec: Mapping[str, object]) -> PerSystemPage:
    if runner in ADAPTIVE_RUNNERS:
        codes = rec.get("verifier_failure_codes") or []
        return PerSystemPage(
            system_id=system_id,
            runner=runner,
            reparse_triggered=bool(rec.get("reparse_triggered", False)),
            verifier_decision=_opt_str(rec.get("verifier_decision")),
            verifier_failure_codes=tuple(str(c) for c in codes if isinstance(codes, list)),
            final_output_source=_opt_str(rec.get("final_output_source")),
            final_raw_path=_opt_str(rec.get("final_raw_path")),
            output_token_count=_token_count_adaptive(rec),
            runtime_ms=float(rec.get("total_runtime_ms") or 0.0),
        )
    if runner == SINGLE_PASS_RUNNER:
        return PerSystemPage(
            system_id=system_id,
            runner=runner,
            reparse_triggered=None,
            verifier_decision=None,
            verifier_failure_codes=(),
            final_output_source=None,
            final_raw_path=_opt_str(rec.get("raw_output_path")),
            output_token_count=int(rec.get("output_token_count") or 0),
            runtime_ms=float(rec.get("runtime_ms") or 0.0),
        )
    raise ValueError(f"{system_id}: unknown runner {runner!r}")


def _token_count_adaptive(rec: Mapping[str, object]) -> int:
    """Tokens of the kept pass — reparse if it fired, otherwise first pass."""

    if rec.get("reparse_triggered") and rec.get("reparse_output_tokens") is not None:
        return int(rec["reparse_output_tokens"])  # type: ignore[arg-type]
    return int(rec.get("first_pass_output_tokens") or 0)


def _opt_str(v: object) -> str | None:
    return str(v) if v is not None else None


def iter_jsonl_rows(rows: tuple[PageRow, ...]) -> Iterator[Mapping[str, object]]:
    """Stream qualitative rows as JSONL-shaped dicts (sorted-keys friendly)."""

    for row in rows:
        yield {
            "page_id": row.page_id,
            "systems": [
                {
                    "system_id": s.system_id,
                    "runner": s.runner,
                    "reparse_triggered": s.reparse_triggered,
                    "verifier_decision": s.verifier_decision,
                    "verifier_failure_codes": list(s.verifier_failure_codes),
                    "final_output_source": s.final_output_source,
                    "final_raw_path": s.final_raw_path,
                    "output_token_count": s.output_token_count,
                    "runtime_ms": s.runtime_ms,
                }
                for s in row.systems
            ],
        }
