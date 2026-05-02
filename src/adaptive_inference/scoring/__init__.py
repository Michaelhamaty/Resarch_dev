"""Quality scoring for predicted vs. ground-truth HTML table pages.

This package is intentionally narrow: it computes per-page quality metrics
from a pair of HTML strings. It is *not* wired into the run pipeline yet —
real ground truth must arrive first. The scorer is unit-testable in
isolation so we can land it before real adapters and real data.
"""

from .cell_f1 import ScoreResult, score_page

__all__ = ["ScoreResult", "score_page"]
