"""Typed result objects for the structural verifier.

The verifier is a decision surface: Phase 4 reads these records to choose
whether to trigger a reparse, and the analysis scripts (Phase 7) read
them to slice runs by failure mode. Both result types are frozen so a
VerifierResult is safe to pass around and log verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TableSummary:
    index: int
    rows: int
    cols: int
    failure_codes: tuple[str, ...]


@dataclass(frozen=True)
class VerifierResult:
    decision: str  # "PASS" or "REPARSE"
    failure_codes: tuple[str, ...]
    predicted_table_count: int
    html_parse_ok: bool
    span_normalization_ok: bool
    tables: tuple[TableSummary, ...]
