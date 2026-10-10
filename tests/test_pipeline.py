"""End-to-end integration tests for the full ticket triage pipeline."""

import json
from pathlib import Path
import tempfile
import unittest

from src.pipeline import PipelineConfig, TicketTriagePipeline


class TestPipelineIntegration(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

        # Create schema
        self.schema_path = self.test_dir / "label_schema.json"
        schema_content = {
            "categories": [
                "billing",
                "verification",
                "login_access",
                "technical_issue",
                "account_closure",
                "feature_request",
                "other",
            ],
            "urgency_levels": ["low", "medium", "high"],
        }
        with open(self.schema_path, "w", encoding="utf-8") as f:
            json.dump(schema_content, f, indent=2)

        # Create test tickets including low-confidence edge case
        self.tickets_path = self.test_dir / "tickets.json"
        tickets_content = [
            {
                "ticket_id": "T1",
                "customer_message": "Cannot log in, password reset failed.",
                "expected_category": "login_access",
                "expected_urgency": "high",
            },
            {
                "ticket_id": "T2",
                "customer_message": "low_confidence_test ambiguous text",
                "expected_category": "other",
                "expected_urgency": "low",
            },
        ]
        with open(self.tickets_path, "w", encoding="utf-8") as f:
            json.dump(tickets_content, f, indent=2)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_pipeline_execution_and_routing(self):
        output_dir = self.test_dir / "output"
        config = PipelineConfig(
            tickets_path=self.tickets_path,
            schema_path=self.schema_path,
            output_dir=output_dir,
            confidence_threshold=0.65,
            provider="mock",
        )

        pipeline = TicketTriagePipeline(config)
        summary = pipeline.run()

        # Check stage progression history
        expected_stages = [
            "INIT",
            "INPUTS_LOADED",
            "TEXT_PREPROCESSED",
            "MODEL_PROMPTED",
            "STRUCTURED_OUTPUT_PARSED",
            "CONFIDENCE_CHECKED",
            "ROUTED",
            "RESPONSE_GENERATED",
            "RESULTS_SAVED",
            "EVALUATION_COMPUTED",
            "VALIDATION_COMPLETED",
        ]
        self.assertEqual(summary["stages"], expected_stages)

        # Check routing decisions
        with open(output_dir / "routing_decisions.json", "r", encoding="utf-8") as f:
            routing = json.load(f)

        self.assertEqual(len(routing), 2)
        t1_route = next(r for r in routing if r["ticket_id"] == "T1")
        t2_route = next(r for r in routing if r["ticket_id"] == "T2")

        # T1 has high confidence -> auto_triage
        self.assertEqual(t1_route["route"], "auto_triage")
        self.assertGreaterEqual(t1_route["confidence"], 0.65)

        # T2 has low confidence (0.45) -> human_review
        self.assertEqual(t2_route["route"], "human_review")
        self.assertLess(t2_route["confidence"], 0.65)

        # Check triage results
        with open(output_dir / "triage_results.json", "r", encoding="utf-8") as f:
            triage = json.load(f)

        t1_res = next(r for r in triage if r["ticket_id"] == "T1")
        t2_res = next(r for r in triage if r["ticket_id"] == "T2")

        # Auto-triaged ticket MUST have customer reply and null internal note
        self.assertIsNotNone(t1_res["customer_reply"])
        self.assertIsNone(t1_res["internal_note"])

        # Human-review ticket MUST have null customer reply and non-empty internal note
        self.assertIsNone(t2_res["customer_reply"])
        self.assertIsNotNone(t2_res["internal_note"])
        self.assertTrue(
            "below required threshold" in t2_res["internal_note"]
            or "flagged ticket for human review" in t2_res["internal_note"]
        )


if __name__ == "__main__":
    unittest.main()
