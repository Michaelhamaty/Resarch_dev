"""Runtime data types for pages flowing through the adaptive inference pipeline.

These are produced at runtime by the dataset loader, inference adapter, and
verifier. They are kept separate from config schemas (see ``config.schemas``),
which describe the *experiment setup*.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class PageInput:
    page_id: str
    image_path: Path
    split: str
    contains_table: bool = True


@dataclass(frozen=True)
class PageOutput:
    page_id: str
    raw_text: str
    html_blocks: tuple[str, ...]
    model_name: str
    budget_name: str
    stage_name: str
    output_token_count: int
    runtime_ms: float


VerifierDecision = Literal["PASS", "REPARSE"]


@dataclass(frozen=True)
class VerifierResult:
    page_id: str
    decision: VerifierDecision
    failure_codes: tuple[str, ...]
    predicted_table_count: int
    html_parse_ok: bool
    span_normalization_ok: bool
