"""Tests for prompt/output robustness, JSON extraction, and noise recovery."""

import unittest

from src.llm_client import extract_json_from_text


class TestRobustness(unittest.TestCase):

    def test_clean_json(self):
        text = '{"category": "billing", "urgency": "high"}'
        res = extract_json_from_text(text)
        self.assertEqual(res["category"], "billing")
        self.assertEqual(res["urgency"], "high")

    def test_markdown_code_fence_json(self):
        text = """Here is your prediction:
```json
{
  "category": "technical_issue",
  "urgency": "medium",
  "confidence": 0.88
}
```
Hope this helps!"""
        res = extract_json_from_text(text)
        self.assertEqual(res["category"], "technical_issue")
        self.assertEqual(res["urgency"], "medium")
        self.assertEqual(res["confidence"], 0.88)

    def test_noisy_text_with_embedded_json(self):
        text = """Sure! The classification output is:
{"category": "verification", "urgency": "medium", "confidence": 0.91}
Let me know if you need more info."""
        res = extract_json_from_text(text)
        self.assertEqual(res["category"], "verification")
        self.assertEqual(res["urgency"], "medium")

    def test_trailing_commas_recovery(self):
        text = """{
  "category": "feature_request",
  "urgency": "low",
}"""
        res = extract_json_from_text(text)
        self.assertEqual(res["category"], "feature_request")
        self.assertEqual(res["urgency"], "low")

    def test_completely_invalid_text_raises_error(self):
        text = "This is not JSON at all and contains no braces."
        with self.assertRaises(ValueError):
            extract_json_from_text(text)


if __name__ == "__main__":
    unittest.main()
