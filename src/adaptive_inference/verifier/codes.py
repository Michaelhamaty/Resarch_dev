"""Stable failure-code identifiers for the structural verifier.

These strings are written into run logs and consumed by Phase 4+ analysis
code, so they are part of the project's persistent output contract and
must not be renamed casually.
"""

from __future__ import annotations

NO_TABLE_FOUND = "NO_TABLE_FOUND"
HTML_PARSE_ERROR = "HTML_PARSE_ERROR"
SPAN_EXPANSION_FAILED = "SPAN_EXPANSION_FAILED"
RECTANGULAR_INCONSISTENCY = "RECTANGULAR_INCONSISTENCY"
DEGENERATE_TABLE = "DEGENERATE_TABLE"

DECISION_PASS = "PASS"
DECISION_REPARSE = "REPARSE"
