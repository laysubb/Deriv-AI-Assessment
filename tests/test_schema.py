"""Tests for label schema validation and structured classification output parsing."""

import unittest

from src.schema import LabelSchema, validate_classification_data


class TestSchemaValidation(unittest.TestCase):

    def setUp(self):
        self.schema = LabelSchema(
            categories=["billing", "verification", "login_access", "technical_issue"],
            urgency_levels=["low", "medium", "high"],
        )

    def test_valid_classification_data(self):
        valid_data = {
            "category": "billing",
            "urgency": "high",
            "confidence": 0.95,
            "reasoning_summary": "Issue with fund withdrawal.",
            "needs_human_review": False,
        }
        outcome = validate_classification_data(valid_data, self.schema)
        self.assertTrue(outcome.is_valid)
        self.assertIsNotNone(outcome.parsed)
        self.assertEqual(outcome.parsed.category, "billing")
        self.assertEqual(outcome.parsed.urgency, "high")
        self.assertEqual(outcome.parsed.confidence, 0.95)
        self.assertFalse(outcome.parsed.needs_human_review)

    def test_missing_required_fields(self):
        invalid_data = {
            "category": "billing",
            "urgency": "high",
            # missing confidence, reasoning_summary, needs_human_review
        }
        outcome = validate_classification_data(invalid_data, self.schema)
        self.assertFalse(outcome.is_valid)
        self.assertIn("Missing required fields", outcome.error_reason)

    def test_invalid_category_rejected(self):
        invalid_data = {
            "category": "random_unknown_category",
            "urgency": "medium",
            "confidence": 0.8,
            "reasoning_summary": "Test",
            "needs_human_review": False,
        }
        outcome = validate_classification_data(invalid_data, self.schema)
        self.assertFalse(outcome.is_valid)
        self.assertIn("Invalid category", outcome.error_reason)

    def test_invalid_urgency_rejected(self):
        invalid_data = {
            "category": "billing",
            "urgency": "critical_extreme",  # not in allowed urgency levels
            "confidence": 0.8,
            "reasoning_summary": "Test",
            "needs_human_review": False,
        }
        outcome = validate_classification_data(invalid_data, self.schema)
        self.assertFalse(outcome.is_valid)
        self.assertIn("Invalid urgency", outcome.error_reason)

    def test_percentage_confidence_normalization(self):
        # Models sometimes return 95.0 instead of 0.95
        data = {
            "category": "billing",
            "urgency": "medium",
            "confidence": 95.0,
            "reasoning_summary": "Normalized test",
            "needs_human_review": False,
        }
        outcome = validate_classification_data(data, self.schema)
        self.assertTrue(outcome.is_valid)
        self.assertAlmostEqual(outcome.parsed.confidence, 0.95)

    def test_out_of_bounds_confidence_rejected(self):
        data = {
            "category": "billing",
            "urgency": "medium",
            "confidence": -0.5,
            "reasoning_summary": "Out of bounds test",
            "needs_human_review": False,
        }
        outcome = validate_classification_data(data, self.schema)
        self.assertFalse(outcome.is_valid)
        self.assertIn("outside valid", outcome.error_reason)


if __name__ == "__main__":
    unittest.main()
