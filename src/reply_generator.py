"""Generation of customer support draft replies and escalation internal notes."""

from dataclasses import dataclass
from typing import Any, Optional

from src.llm_client import LLMClient, build_reply_prompt
from src.logger import LLMCallLogger
from src.routing import RoutingDecision
from src.schema import ClassificationResult


@dataclass
class TriageResult:
    """Triage result record according to requirement specification."""

    ticket_id: str
    predicted_category: str
    predicted_urgency: str
    confidence: float
    route: str
    customer_reply: Optional[str]
    internal_note: Optional[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticket_id": self.ticket_id,
            "predicted_category": self.predicted_category,
            "predicted_urgency": self.predicted_urgency,
            "confidence": round(self.confidence, 4),
            "route": self.route,
            "customer_reply": self.customer_reply,
            "internal_note": self.internal_note,
        }


def generate_triage_result(
    ticket_id: str,
    ticket_text: str,
    routing_decision: RoutingDecision,
    classification_result: Optional[ClassificationResult],
    llm_client: LLMClient,
    logger: LLMCallLogger,
) -> TriageResult:
    """Produces the final triage result for a ticket based on routing decision.

    - For 'auto_triage': makes an LLM call to generate a 2-4 sentence customer reply, internal_note is null.
    - For 'human_review': customer_reply is null, internal_note explains escalation reason.
    """
    if routing_decision.route == "auto_triage":
        if classification_result is None:
            raise ValueError(f"Cannot auto_triage ticket {ticket_id} without valid classification.")

        category = classification_result.category
        urgency = classification_result.urgency
        confidence = classification_result.confidence

        prompt = build_reply_prompt(
            ticket_text=ticket_text,
            category=category,
            urgency=urgency,
        )

        raw_reply = llm_client.generate(prompt)

        # Log LLM call for reply generation
        logger.log_call(
            stage="reply_generation",
            ticket_id=ticket_id,
            provider=llm_client.provider_name,
            model=llm_client.model_name,
            prompt=prompt,
            raw_response=raw_reply,
        )

        customer_reply = raw_reply.strip()
        internal_note = None

    else:  # human_review
        category = classification_result.category if classification_result else "unclassified"
        urgency = classification_result.urgency if classification_result else "unclassified"
        confidence = classification_result.confidence if classification_result else 0.0

        customer_reply = None
        internal_note = (
            f"Escalated to human agent review. Reason: {routing_decision.routing_reason}"
        )

    return TriageResult(
        ticket_id=ticket_id,
        predicted_category=category,
        predicted_urgency=urgency,
        confidence=confidence,
        route=routing_decision.route,
        customer_reply=customer_reply,
        internal_note=internal_note,
    )
