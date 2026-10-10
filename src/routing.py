"""Deterministic confidence routing logic."""

from dataclasses import dataclass
from typing import Any, Optional

from src.schema import ClassificationResult, ValidationOutcome

DEFAULT_CONFIDENCE_THRESHOLD = 0.65


@dataclass
class RoutingDecision:
    """Routing record according to requirement specification."""

    ticket_id: str
    route: str  # "auto_triage" | "human_review"
    confidence: float
    routing_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticket_id": self.ticket_id,
            "route": self.route,
            "confidence": round(self.confidence, 4),
            "routing_reason": self.routing_reason,
        }


def make_routing_decision(
    ticket_id: str,
    validation_outcome: ValidationOutcome,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> RoutingDecision:
    """Determines routing strictly in deterministic code.

    Rules:
    - If model output is invalid or unparsable: route to 'human_review' with error reason.
    - If model explicitly flagged 'needs_human_review': route to 'human_review'.
    - If confidence < threshold: route to 'human_review'.
    - Otherwise: route to 'auto_triage'.
    """
    if not validation_outcome.is_valid or validation_outcome.parsed is None:
        err = validation_outcome.error_reason or "Unknown validation failure."
        return RoutingDecision(
            ticket_id=ticket_id,
            route="human_review",
            confidence=0.0,
            routing_reason=f"Escalated due to parsing/validation failure: {err}",
        )

    parsed = validation_outcome.parsed

    if parsed.needs_human_review:
        return RoutingDecision(
            ticket_id=ticket_id,
            route="human_review",
            confidence=parsed.confidence,
            routing_reason=(
                f"Model explicitly flagged ticket for human review: {parsed.reasoning_summary}"
            ),
        )

    if parsed.confidence < threshold:
        return RoutingDecision(
            ticket_id=ticket_id,
            route="human_review",
            confidence=parsed.confidence,
            routing_reason=(
                f"Confidence score {parsed.confidence:.2f} is below required threshold {threshold:.2f}"
            ),
        )

    return RoutingDecision(
        ticket_id=ticket_id,
        route="auto_triage",
        confidence=parsed.confidence,
        routing_reason=(
            f"Confidence score {parsed.confidence:.2f} satisfies threshold {threshold:.2f}"
        ),
    )
