"""Tests for deterministic confidence routing logic."""

import unittest

from src.routing import make_routing_decision
from src.schema import ClassificationResult, ValidationOutcome


class TestRouting(unittest.TestCase):

    def test_high_confidence_routes_to_auto_triage(self):
        parsed = ClassificationResult(
            category="billing",
            urgency="high",
            confidence=0.92,
            reasoning_summary="Clear withdrawal question",
            needs_human_review=False,
        )
        outcome = ValidationOutcome(is_valid=True, parsed=parsed)
        decision = make_routing_decision("T1", outcome, threshold=0.65)

        self.assertEqual(decision.route, "auto_triage")
        self.assertEqual(decision.confidence, 0.92)
        self.assertIn("satisfies threshold", decision.routing_reason)

    def test_low_confidence_routes_to_human_review(self):
        parsed = ClassificationResult(
            category="technical_issue",
            urgency="low",
            confidence=0.58,  # Below 0.65 threshold
            reasoning_summary="Vague error description",
            needs_human_review=False,
        )
        outcome = ValidationOutcome(is_valid=True, parsed=parsed)
        decision = make_routing_decision("T2", outcome, threshold=0.65)

        self.assertEqual(decision.route, "human_review")
        self.assertEqual(decision.confidence, 0.58)
        self.assertIn("below required threshold", decision.routing_reason)

    def test_model_explicitly_flagged_human_review(self):
        parsed = ClassificationResult(
            category="login_access",
            urgency="high",
            confidence=0.90,  # High confidence, but model flagged human review
            reasoning_summary="Account security alert suspected",
            needs_human_review=True,
        )
        outcome = ValidationOutcome(is_valid=True, parsed=parsed)
        decision = make_routing_decision("T3", outcome, threshold=0.65)

        self.assertEqual(decision.route, "human_review")
        self.assertIn("Model explicitly flagged", decision.routing_reason)

    def test_invalid_unparsable_routes_to_human_review(self):
        outcome = ValidationOutcome(
            is_valid=False,
            error_reason="JSONDecodeError: Unterminated string",
        )
        decision = make_routing_decision("T4", outcome, threshold=0.65)

        self.assertEqual(decision.route, "human_review")
        self.assertEqual(decision.confidence, 0.0)
        self.assertIn("Escalated due to parsing/validation failure", decision.routing_reason)


if __name__ == "__main__":
    unittest.main()
