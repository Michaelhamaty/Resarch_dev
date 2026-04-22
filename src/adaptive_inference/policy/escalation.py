"""One-shot escalation policy: verifier decision -> reparse yes/no.

Intentionally dumb per the build brief: the policy is a pure function of
the verifier decision string. No hidden heuristics, no history, no
thresholds. Future ablations (random escalation, confidence-based
routing) would live as sibling functions in this module; the
orchestrator only consumes the boolean.
"""

from __future__ import annotations

from ..verifier.codes import DECISION_REPARSE
from ..verifier.types import VerifierResult


def should_reparse(verifier_result: VerifierResult) -> bool:
    """Return True iff the verifier asked for a reparse."""

    return verifier_result.decision == DECISION_REPARSE
