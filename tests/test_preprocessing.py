"""Tests for text preprocessing and length statistics."""

import unittest

from src.preprocessing import clean_ticket_text, compute_text_stats, preprocess_ticket


class TestPreprocessing(unittest.TestCase):

    def test_whitespace_and_newline_cleanup(self):
        raw = "  Hello   world! \n\n\n  How are you?   "
        cleaned = clean_ticket_text(raw)
        self.assertEqual(cleaned, "Hello world! How are you?")

    def test_repeated_punctuation_normalization(self):
        raw = "Is my withdrawal ready???? Please help!!!!!"
        cleaned = clean_ticket_text(raw)
        self.assertEqual(cleaned, "Is my withdrawal ready? Please help!")

    def test_text_stats_computation(self):
        text = "Four distinct words here."
        stats = compute_text_stats(text)
        self.assertEqual(stats["word_count"], 4)
        self.assertEqual(stats["char_count"], len(text))

    def test_preprocess_ticket_record(self):
        raw_ticket = {
            "ticket_id": "T100",
            "customer_message": "  My password reset   fails.  ",
            "expected_category": "login_access",
            "expected_urgency": "high",
        }
        res = preprocess_ticket(raw_ticket)
        self.assertEqual(res["ticket_id"], "T100")
        self.assertEqual(res["original_text"], "My password reset   fails.")
        self.assertEqual(res["cleaned_text"], "My password reset fails.")
        self.assertEqual(res["word_count"], 4)
        self.assertEqual(res["char_count"], len("My password reset fails."))
        self.assertEqual(res["expected_category"], "login_access")


if __name__ == "__main__":
    unittest.main()
