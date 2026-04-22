"""Escalation policy package — the one-shot REPARSE rule.

Phase 4's policy is deliberately trivial: translate a ``VerifierResult``
into a boolean ``should_reparse`` decision. Keeping it in its own
package makes it obvious where future ablations (e.g. random escalation
baseline) would hook in without leaking into the orchestrator.
"""

from .escalation import should_reparse

__all__ = ["should_reparse"]
