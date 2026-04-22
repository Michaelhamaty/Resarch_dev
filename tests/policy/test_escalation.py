"""Unit tests for the one-shot escalation policy."""

from __future__ import annotations

from adaptive_inference.policy.escalation import should_reparse
from adaptive_inference.verifier.codes import DECISION_PASS, DECISION_REPARSE
from adaptive_inference.verifier.types import VerifierResult


def _verifier(decision: str) -> VerifierResult:
    return VerifierResult(
        decision=decision,
        failure_codes=(),
        predicted_table_count=1,
        html_parse_ok=True,
        span_normalization_ok=True,
        tables=(),
    )


def test_should_reparse_true_for_reparse_decision() -> None:
    assert should_reparse(_verifier(DECISION_REPARSE)) is True


def test_should_reparse_false_for_pass_decision() -> None:
    assert should_reparse(_verifier(DECISION_PASS)) is False
