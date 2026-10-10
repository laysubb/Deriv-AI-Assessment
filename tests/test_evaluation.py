"""Tests for evaluation metrics and confusion summary computation."""

import unittest

from src.evaluation import compute_evaluation
from src.reply_generator import TriageResult
from src.routing import RoutingDecision


class TestEvaluation(unittest.TestCase):

    def test_compute_evaluation_metrics(self):
        raw_tickets = [
            {
                "ticket_id": "T1",
                "customer_message": "Reset password issue",
                "expected_category": "login_access",
                "expected_urgency": "high",
            },
            {
                "ticket_id": "T2",
                "customer_message": "Withdrawal delay",
                "expected_category": "billing",
                "expected_urgency": "high",
            },
            {
                "ticket_id": "T3",
                "customer_message": "Dark mode feature",
                "expected_category": "feature_request",
                "expected_urgency": "low",
            },
        ]

        triage_results = [
            TriageResult(
                ticket_id="T1",
                predicted_category="login_access",  # match
                predicted_urgency="high",  # match
                confidence=0.9,
                route="auto_triage",
                customer_reply="We can help reset your password.",
                internal_note=None,
            ),
            TriageResult(
                ticket_id="T2",
                predicted_category="technical_issue",  # mismatch (expected billing)
                predicted_urgency="high",  # match
                confidence=0.8,
                route="auto_triage",
                customer_reply="Checking technical issue.",
                internal_note=None,
            ),
            TriageResult(
                ticket_id="T3",
                predicted_category="feature_request",  # match
                predicted_urgency="low",  # match
                confidence=0.5,
                route="human_review",
                customer_reply=None,
                internal_note="Escalated due to low confidence.",
            ),
        ]

        routing_decisions = [
            RoutingDecision(ticket_id="T1", route="auto_triage", confidence=0.9, routing_reason="OK"),
            RoutingDecision(ticket_id="T2", route="auto_triage", confidence=0.8, routing_reason="OK"),
            RoutingDecision(ticket_id="T3", route="human_review", confidence=0.5, routing_reason="Low conf"),
        ]

        report, comparisons, confusion = compute_evaluation(
            raw_tickets=raw_tickets,
            triage_results=triage_results,
            routing_decisions=routing_decisions,
            parse_failures_count=0,
        )

        self.assertEqual(report["total_tickets"], 3)
        self.assertEqual(report["evaluated_tickets"], 3)
        self.assertAlmostEqual(report["category_accuracy"], 2 / 3, places=2)
        self.assertEqual(report["urgency_accuracy"], 1.0)
        self.assertEqual(report["auto_triage_count"], 2)
        self.assertEqual(report["human_review_count"], 1)

        # Check comparisons
        self.assertEqual(len(comparisons), 3)
        t2_comp = next(c for c in comparisons if c["ticket_id"] == "T2")
        self.assertFalse(t2_comp["category_match"])
        self.assertTrue(t2_comp["urgency_match"])

        # Check confusion summary
        self.assertEqual(len(confusion["misclassifications"]), 1)
        self.assertEqual(confusion["misclassifications"][0]["expected_category"], "billing")
        self.assertEqual(confusion["misclassifications"][0]["predicted_category"], "technical_issue")


if __name__ == "__main__":
    unittest.main()
