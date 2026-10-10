"""Master pipeline orchestrator enforcing strict stage transitions and artifact generation."""

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import shutil
from typing import Any, Optional

from src.evaluation import compute_evaluation
from src.llm_client import (
    LLMClient,
    build_classification_prompt,
    build_stricter_retry_prompt,
    extract_json_from_text,
    get_llm_client,
)
from src.logger import LLMCallLogger
from src.preprocessing import preprocess_tickets
from src.reply_generator import TriageResult, generate_triage_result
from src.routing import RoutingDecision, make_routing_decision
from src.schema import (
    ClassificationResult,
    LabelSchema,
    ValidationOutcome,
    validate_classification_data,
)
from src.stages import PipelineStage, PipelineStageTracker

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration options for pipeline execution."""

    tickets_path: Path
    schema_path: Path
    output_dir: Path
    confidence_threshold: float = 0.65
    provider: str = "auto"
    model: Optional[str] = None
    api_key: Optional[str] = None
    copy_inputs_to_output: bool = True


class TicketTriagePipeline:
    """Executes the ticket triage pipeline with enforced sequential stages."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.tracker = PipelineStageTracker()
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger_util = LLMCallLogger(
            log_file_path=self.output_dir / "llm_calls.jsonl",
            raw_artifacts_dir=self.output_dir / "raw_llm_outputs",
        )

        self.llm_client: LLMClient = get_llm_client(
            provider=config.provider,
            model=config.model,
            api_key=config.api_key,
        )

    def run(self) -> dict[str, Any]:
        """Runs the entire triage pipeline through all required stages."""
        # 1. INPUTS_LOADED
        self.tracker.transition_to(
            PipelineStage.INPUTS_LOADED,
            f"Loading inputs from {self.config.tickets_path} and {self.config.schema_path}",
        )
        with open(self.config.tickets_path, "r", encoding="utf-8") as f:
            raw_tickets = json.load(f)
        schema = LabelSchema.load_from_file(self.config.schema_path)

        # 2. TEXT_PREPROCESSED
        self.tracker.transition_to(
            PipelineStage.TEXT_PREPROCESSED,
            f"Preprocessing {len(raw_tickets)} tickets",
        )
        preprocessed = preprocess_tickets(raw_tickets)

        # 3. MODEL_PROMPTED & 4. STRUCTURED_OUTPUT_PARSED & 5. CONFIDENCE_CHECKED & 6. ROUTED
        # We process each ticket through prompt -> parse -> retry (if needed) -> confidence check -> route
        self.tracker.transition_to(
            PipelineStage.MODEL_PROMPTED,
            "Prompting model for structured classification",
        )

        classification_outcomes: list[ValidationOutcome] = []
        parse_failures_count = 0

        for ticket in preprocessed:
            tid = ticket["ticket_id"]
            text = ticket["cleaned_text"]

            # Initial classification prompt
            prompt = build_classification_prompt(text, schema)
            raw_resp = self.llm_client.generate(prompt)

            self.logger_util.log_call(
                stage="classification",
                ticket_id=tid,
                provider=self.llm_client.provider_name,
                model=self.llm_client.model_name,
                prompt=prompt,
                raw_response=raw_resp,
                attempt=1,
            )

            # Attempt JSON extraction
            outcome: ValidationOutcome
            try:
                parsed_json = extract_json_from_text(raw_resp)
                outcome = validate_classification_data(parsed_json, schema, raw_response=raw_resp)
            except Exception as e:
                outcome = ValidationOutcome(
                    is_valid=False,
                    error_reason=f"Failed to parse JSON: {str(e)}",
                )

            # Robustness Recovery Path: Retry once with stricter prompt if initial parse failed
            if not outcome.is_valid:
                logger.warning(
                    f"Ticket {tid} failed initial validation ({outcome.error_reason}). "
                    f"Executing recovery retry prompt..."
                )
                retry_prompt = build_stricter_retry_prompt(
                    ticket_text=text,
                    schema=schema,
                    previous_raw=raw_resp,
                    error_reason=outcome.error_reason or "Unknown parse error",
                )
                retry_resp = self.llm_client.generate(retry_prompt)

                self.logger_util.log_call(
                    stage="classification",
                    ticket_id=tid,
                    provider=self.llm_client.provider_name,
                    model=self.llm_client.model_name,
                    prompt=retry_prompt,
                    raw_response=retry_resp,
                    attempt=2,
                )

                try:
                    retry_json = extract_json_from_text(retry_resp)
                    retry_outcome = validate_classification_data(
                        retry_json, schema, raw_response=retry_resp
                    )
                    if retry_outcome.is_valid:
                        outcome = retry_outcome
                        logger.info(f"Ticket {tid} successfully recovered on retry.")
                    else:
                        parse_failures_count += 1
                        logger.error(f"Ticket {tid} retry also failed validation.")
                except Exception as retry_err:
                    parse_failures_count += 1
                    logger.error(f"Ticket {tid} retry failed JSON extraction: {retry_err}")
                    outcome = ValidationOutcome(
                        is_valid=False,
                        error_reason=f"Recovery retry failed JSON extraction: {str(retry_err)}",
                    )

            classification_outcomes.append(outcome)

        # 4. STRUCTURED_OUTPUT_PARSED
        self.tracker.transition_to(
            PipelineStage.STRUCTURED_OUTPUT_PARSED,
            f"Parsed {len(classification_outcomes)} outputs with {parse_failures_count} parse failures",
        )

        # 5. CONFIDENCE_CHECKED
        self.tracker.transition_to(
            PipelineStage.CONFIDENCE_CHECKED,
            f"Checking confidence against threshold {self.config.confidence_threshold}",
        )

        # 6. ROUTED
        self.tracker.transition_to(
            PipelineStage.ROUTED,
            "Applying deterministic routing rules",
        )
        routing_decisions: list[RoutingDecision] = []
        for ticket, outcome in zip(preprocessed, classification_outcomes):
            decision = make_routing_decision(
                ticket_id=ticket["ticket_id"],
                validation_outcome=outcome,
                threshold=self.config.confidence_threshold,
            )
            routing_decisions.append(decision)

        # 7. RESPONSE_GENERATED
        self.tracker.transition_to(
            PipelineStage.RESPONSE_GENERATED,
            "Generating customer replies for auto_triage and internal notes for human_review",
        )
        triage_results: list[TriageResult] = []
        for ticket, outcome, decision in zip(preprocessed, classification_outcomes, routing_decisions):
            res = generate_triage_result(
                ticket_id=ticket["ticket_id"],
                ticket_text=ticket["cleaned_text"],
                routing_decision=decision,
                classification_result=outcome.parsed,
                llm_client=self.llm_client,
                logger=self.logger_util,
            )
            triage_results.append(res)

        # 8. RESULTS_SAVED
        self.tracker.transition_to(
            PipelineStage.RESULTS_SAVED,
            "Saving required machine-readable JSON artifacts",
        )
        self._save_artifacts(
            preprocessed=preprocessed,
            routing_decisions=routing_decisions,
            triage_results=triage_results,
        )

        # 9. EVALUATION_COMPUTED
        self.tracker.transition_to(
            PipelineStage.EVALUATION_COMPUTED,
            "Computing evaluation metrics, prediction comparisons, and confusion matrix",
        )
        eval_report, comparisons, confusion_summary = compute_evaluation(
            raw_tickets=raw_tickets,
            triage_results=triage_results,
            routing_decisions=routing_decisions,
            parse_failures_count=parse_failures_count,
        )

        with open(self.output_dir / "evaluation_report.json", "w", encoding="utf-8") as f:
            json.dump(eval_report, f, indent=2)

        with open(self.output_dir / "prediction_comparison.json", "w", encoding="utf-8") as f:
            json.dump(comparisons, f, indent=2)

        with open(self.output_dir / "confusion_summary.json", "w", encoding="utf-8") as f:
            json.dump(confusion_summary, f, indent=2)

        # 10. VALIDATION_COMPLETED
        self.tracker.transition_to(
            PipelineStage.VALIDATION_COMPLETED,
            "Pipeline finished execution and internal state validated",
        )

        return {
            "evaluation_report": eval_report,
            "total_tickets": len(raw_tickets),
            "stages": [s.value for s, _ in self.tracker.history],
        }

    def _save_artifacts(
        self,
        preprocessed: list[dict[str, Any]],
        routing_decisions: list[RoutingDecision],
        triage_results: list[TriageResult],
    ) -> None:
        """Saves machine-readable JSON artifacts."""
        # 1. preprocessed_tickets.json
        prep_path = self.output_dir / "preprocessed_tickets.json"
        with open(prep_path, "w", encoding="utf-8") as f:
            # Only keep the required fields in preprocessed_tickets.json
            clean_records = [
                {
                    "ticket_id": p["ticket_id"],
                    "original_text": p["original_text"],
                    "cleaned_text": p["cleaned_text"],
                    "char_count": p["char_count"],
                    "word_count": p["word_count"],
                }
                for p in preprocessed
            ]
            json.dump(clean_records, f, indent=2)

        # 2. routing_decisions.json
        rout_path = self.output_dir / "routing_decisions.json"
        with open(rout_path, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in routing_decisions], f, indent=2)

        # 3. triage_results.json
        res_path = self.output_dir / "triage_results.json"
        with open(res_path, "w", encoding="utf-8") as f:
            json.dump([t.to_dict() for t in triage_results], f, indent=2)

        # 4. Copy input files to output directory if configured to ensure all REQUIRED ARTIFACTS exist together
        if self.config.copy_inputs_to_output:
            target_tickets = self.output_dir / "tickets.json"
            target_schema = self.output_dir / "label_schema.json"
            if self.config.tickets_path.resolve() != target_tickets.resolve():
                shutil.copyfile(self.config.tickets_path, target_tickets)
            if self.config.schema_path.resolve() != target_schema.resolve():
                shutil.copyfile(self.config.schema_path, target_schema)
