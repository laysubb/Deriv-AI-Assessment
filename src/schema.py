"""Schema models and strict validation for labels and structured classification."""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Optional


@dataclass
class LabelSchema:
    """Allowed label categories and urgency levels loaded from schema file."""

    categories: list[str]
    urgency_levels: list[str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LabelSchema":
        categories = [str(c).strip() for c in data.get("categories", []) if str(c).strip()]
        urgency_levels = [str(u).strip() for u in data.get("urgency_levels", []) if str(u).strip()]
        if not categories:
            raise ValueError("Label schema must provide at least one category.")
        if not urgency_levels:
            raise ValueError("Label schema must provide at least one urgency level.")
        return cls(categories=categories, urgency_levels=urgency_levels)

    @classmethod
    def load_from_file(cls, path: str | Path) -> "LabelSchema":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def is_valid_category(self, category: str) -> bool:
        return category.strip() in self.categories

    def is_valid_urgency(self, urgency: str) -> bool:
        return urgency.strip() in self.urgency_levels


@dataclass
class ClassificationResult:
    """Validated structured classification record."""

    category: str
    urgency: str
    confidence: float
    reasoning_summary: str
    needs_human_review: bool
    raw_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "urgency": self.urgency,
            "confidence": round(self.confidence, 4),
            "reasoning_summary": self.reasoning_summary,
            "needs_human_review": self.needs_human_review,
        }


@dataclass
class ValidationOutcome:
    """Result of schema validation on model output."""

    is_valid: bool
    parsed: Optional[ClassificationResult] = None
    error_reason: Optional[str] = None


def validate_classification_data(
    data: Any,
    schema: LabelSchema,
    raw_response: str = "",
) -> ValidationOutcome:
    """Validates structured data against label schema constraints."""
    if not isinstance(data, dict):
        return ValidationOutcome(
            is_valid=False,
            error_reason=f"Model output is not a JSON object: received type {type(data).__name__}",
        )

    # Check required fields
    required_fields = ["category", "urgency", "confidence", "reasoning_summary", "needs_human_review"]
    missing = [f for f in required_fields if f not in data]
    if missing:
        return ValidationOutcome(
            is_valid=False,
            error_reason=f"Missing required fields: {', '.join(missing)}",
        )

    category = str(data["category"]).strip()
    urgency = str(data["urgency"]).strip()

    # Validate category against schema
    if not schema.is_valid_category(category):
        return ValidationOutcome(
            is_valid=False,
            error_reason=(
                f"Invalid category '{category}'. Allowed categories: {schema.categories}"
            ),
        )

    # Validate urgency against schema
    if not schema.is_valid_urgency(urgency):
        return ValidationOutcome(
            is_valid=False,
            error_reason=(
                f"Invalid urgency '{urgency}'. Allowed urgency levels: {schema.urgency_levels}"
            ),
        )

    # Validate confidence as float between 0.0 and 1.0
    try:
        confidence = float(data["confidence"])
    except (TypeError, ValueError):
        return ValidationOutcome(
            is_valid=False,
            error_reason=f"Invalid confidence value '{data['confidence']}': must be float.",
        )

    if not (0.0 <= confidence <= 1.0):
        # Clip or flag if outside range
        if 0.0 <= confidence <= 100.0 and confidence > 1.0:
            # Common model artifact: percentage returned instead of 0.0-1.0
            confidence = confidence / 100.0
        else:
            return ValidationOutcome(
                is_valid=False,
                error_reason=f"Confidence {confidence} is outside valid [0.0, 1.0] interval.",
            )

    # Validate needs_human_review
    needs_review_val = data["needs_human_review"]
    if isinstance(needs_review_val, bool):
        needs_human_review = needs_review_val
    elif isinstance(needs_review_val, str):
        needs_human_review = needs_review_val.lower() in ("true", "1", "yes")
    else:
        needs_human_review = bool(needs_review_val)

    reasoning_summary = str(data.get("reasoning_summary", "")).strip()
    if not reasoning_summary:
        reasoning_summary = "No reasoning summary provided."

    parsed = ClassificationResult(
        category=category,
        urgency=urgency,
        confidence=confidence,
        reasoning_summary=reasoning_summary,
        needs_human_review=needs_human_review,
        raw_response=raw_response,
    )
    return ValidationOutcome(is_valid=True, parsed=parsed)
