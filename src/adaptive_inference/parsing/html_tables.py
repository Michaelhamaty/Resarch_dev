"""Extract top-level HTML ``<table>`` blocks from model page output.

Phase 3 keeps extraction syntactic: page markdown can mix prose with HTML,
and we don't want to feed surrounding text into an HTML parser. A regex
scan finds ``<table ...>...</table>`` substrings and returns them as raw
strings so the verifier can decide how to parse each one. Nested tables
are out of scope for the MVP — only the outermost ``<table>`` at each
position is returned.
"""

from __future__ import annotations

import re

# Opening tag: `<table` optionally followed by attributes (anything that
# isn't `>`), then `>`. Closing tag: `</table>`. Both case-insensitive,
# DOTALL so the body may span newlines.
_TABLE_BLOCK_RE = re.compile(
    r"<table\b[^>]*>.*?</table\s*>",
    re.IGNORECASE | re.DOTALL,
)


def extract_table_blocks(raw_text: str) -> tuple[str, ...]:
    """Return every outermost ``<table>...</table>`` substring in order."""

    if not raw_text:
        return ()
    return tuple(match.group(0) for match in _TABLE_BLOCK_RE.finditer(raw_text))
