"""Phase 7 analysis: consume Phase 1-6 artifacts and produce final summaries.

This package is read-only with respect to every earlier phase. It loads
the Phase 6 manifest, per-system run logs, and frozen calibration
artifact, then writes structured analysis files plus an integration
audit.

The analysis intentionally does NOT compute table-level accuracy
(TEDS / edit-distance). The MVP repo runs entirely on stub adapters,
so any "accuracy" number would be fictional. Phase 7 reports cost,
runtime, reparse-rate, and verifier failure-code distributions, and
gates accuracy with ``accuracy_status="not_applicable_stub_adapters"``.

Public surface:

- :class:`Phase7Config` / :func:`load_phase7_config`
- :func:`run_phase7` orchestrator + :class:`Phase7Result`
- :class:`AuditReport` / :class:`AuditCheck`
"""

from .audit import AuditCheck, AuditReport, run_audit
from .config import Phase7Config, load_phase7_config
from .runner import Phase7Result, run_phase7

__all__ = [
    "AuditCheck",
    "AuditReport",
    "Phase7Config",
    "Phase7Result",
    "load_phase7_config",
    "run_audit",
    "run_phase7",
]
