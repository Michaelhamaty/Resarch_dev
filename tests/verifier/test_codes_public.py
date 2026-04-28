"""Drift guard: every code the verifier emits must be exported by codes.py.

If someone adds a new failure path to ``structural.py`` without adding a
named constant in ``codes.py``, downstream readers (Phase 7 audit, run
logs) will see an unknown string and break. This test pins the contract.
"""

from __future__ import annotations

import re
from pathlib import Path

from adaptive_inference.verifier import codes as codes_module


SOURCE_FILES = ("structural.py", "spans.py")


def _public_code_constants() -> set[str]:
    return {
        getattr(codes_module, name)
        for name in dir(codes_module)
        if name.isupper() and isinstance(getattr(codes_module, name), str)
    }


def test_all_emitted_codes_are_public_constants(repo_root: Path):
    """Every quoted-string code referenced in the verifier source must be
    a constant in ``verifier.codes`` — otherwise either the code is dead
    or the public surface is incomplete."""

    public = _public_code_constants()
    package_root = repo_root / "src" / "adaptive_inference" / "verifier"

    for filename in SOURCE_FILES:
        text = (package_root / filename).read_text(encoding="utf-8")
        for match in re.finditer(r'"([A-Z_][A-Z0-9_]+)"', text):
            literal = match.group(1)
            # Heuristic: skip generic identifiers that aren't codes.
            if literal in {"PASS", "REPARSE"} or literal.endswith(("_TABLE", "_FOUND", "_ERROR", "_FAILED", "_INCONSISTENCY")):
                assert literal in public, (
                    f"verifier/{filename} emits {literal!r} but it is not "
                    f"exported by verifier.codes. Add it as a constant."
                )
