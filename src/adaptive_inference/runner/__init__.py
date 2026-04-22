"""Public exports for the runner package (orchestration + IO).

Phase 2 single-pass and Phase 4 adaptive orchestrators both live here.
Each has its own writer + logger module so their artifact layouts stay
independent; single-pass was not rewritten to accommodate Phase 4.
"""

from .adaptive import AdaptiveRunSummary, run_adaptive
from .adaptive_logger import (
    AdaptivePageLog,
    append_adaptive_page_log,
    reset_adaptive_log,
)
from .adaptive_writer import (
    WrittenFinalArtifacts,
    WrittenPassArtifacts,
    write_final_artifacts,
    write_pass_artifacts,
)
from .output_writer import WrittenPage, write_page_result
from .pages import LoadedPage, load_manifest_page_ids, load_pages_for_manifest
from .runtime_logger import LOG_FILENAME, append_page_log, reset_log
from .single_pass import SinglePassSummary, run_single_pass

__all__ = [
    "LOG_FILENAME",
    "AdaptivePageLog",
    "AdaptiveRunSummary",
    "LoadedPage",
    "SinglePassSummary",
    "WrittenFinalArtifacts",
    "WrittenPage",
    "WrittenPassArtifacts",
    "append_adaptive_page_log",
    "append_page_log",
    "load_manifest_page_ids",
    "load_pages_for_manifest",
    "reset_adaptive_log",
    "reset_log",
    "run_adaptive",
    "run_single_pass",
    "write_final_artifacts",
    "write_page_result",
    "write_pass_artifacts",
]
