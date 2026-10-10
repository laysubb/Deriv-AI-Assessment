"""Deterministic text preprocessing and statistics calculation."""

import re
from typing import Any


def clean_ticket_text(text: str) -> str:
    """Cleans raw ticket text using deterministic normalization rules.

    Rules applied:
    - Normalizes non-standard whitespaces (tabs, non-breaking spaces) to single space.
    - Strips leading and trailing whitespaces.
    - Compresses runs of multiple spaces into a single space.
    - Normalizes repeated punctuation (e.g. '????' -> '?', '!!!!' -> '!').
    - Normalizes excessive newlines into a single newline.
    """
    if not text:
        return ""

    # Replace carriage returns and tabs
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")

    # Collapse multiple consecutive newlines into single newline
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)

    # Normalize repeated punctuation: ??? -> ?, !!! -> !
    cleaned = re.sub(r"\?{2,}", "?", cleaned)
    cleaned = re.sub(r"!{2,}", "!", cleaned)

    # Collapse multiple consecutive horizontal whitespace characters
    cleaned = re.sub(r"[^\S\n]+", " ", cleaned)

    # Clean lines and strip
    lines = [line.strip() for line in cleaned.split("\n") if line.strip()]
    cleaned = " ".join(lines).strip()

    return cleaned


def compute_text_stats(text: str) -> dict[str, int]:
    """Computes basic length stats: character count and word count."""
    char_count = len(text)
    word_count = len(text.split()) if text else 0
    return {
        "char_count": char_count,
        "word_count": word_count,
    }


def preprocess_ticket(raw_ticket: dict[str, Any]) -> dict[str, Any]:
    """Preprocesses a single ticket record into the required format."""
    ticket_id = str(raw_ticket.get("ticket_id", "")).strip()
    original_text = str(raw_ticket.get("customer_message", "")).strip()

    cleaned_text = clean_ticket_text(original_text)
    stats = compute_text_stats(cleaned_text)

    record: dict[str, Any] = {
        "ticket_id": ticket_id,
        "original_text": original_text,
        "cleaned_text": cleaned_text,
        "char_count": stats["char_count"],
        "word_count": stats["word_count"],
    }

    # Retain ground truth annotations if available for downstream evaluation
    if "expected_category" in raw_ticket:
        record["expected_category"] = raw_ticket["expected_category"]
    if "expected_urgency" in raw_ticket:
        record["expected_urgency"] = raw_ticket["expected_urgency"]

    return record


def preprocess_tickets(raw_tickets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Preprocesses all tickets in a batch."""
    return [preprocess_ticket(t) for t in raw_tickets]
