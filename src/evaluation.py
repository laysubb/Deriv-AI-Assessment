"""Evaluation metrics calculation, comparison reporting, and confusion summary."""

from collections import defaultdict
from typing import Any, Optional

from src.reply_generator import TriageResult
from src.routing import RoutingDecision


def compute_evaluation(
    raw_tickets: list[dict[str, Any]],
    triage_results: list[TriageResult],
    routing_decisions: list[RoutingDecision],
    parse_failures_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Computes evaluation report, per-ticket comparisons, and category confusion summary.

    Returns:
    - evaluation_report: dict containing overall metrics
    - prediction_comparisons: list of per-ticket comparisons
    - confusion_summary: summary of category predictions vs expected labels
    """
    triage_by_id = {r.ticket_id: r for r in triage_results}
    routing_by_id = {r.ticket_id: r for r in routing_decisions}

    total_tickets = len(raw_tickets)
    human_review_count = sum(1 for r in routing_decisions if r.route == "human_review")
    auto_triage_count = sum(1 for r in routing_decisions if r.route == "auto_triage")

    evaluated_count = 0
    cat_correct = 0
    urg_correct = 0

    comparisons: list[dict[str, Any]] = []
    confusion_matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    misclassifications: list[dict[str, Any]] = []

    for ticket in raw_tickets:
        tid = str(ticket.get("ticket_id", ""))
        exp_cat = ticket.get("expected_category")
        exp_urg = ticket.get("expected_urgency")

        triage = triage_by_id.get(tid)
        pred_cat = triage.predicted_category if triage else "unknown"
        pred_urg = triage.predicted_urgency if triage else "unknown"
        conf = triage.confidence if triage else 0.0
        route = triage.route if triage else "unknown"

        has_ground_truth = exp_cat is not None and exp_urg is not None

        cat_match = (pred_cat == exp_cat) if exp_cat is not None else None
        urg_match = (pred_urg == exp_urg) if exp_urg is not None else None

        if has_ground_truth:
            evaluated_count += 1
            if cat_match:
                cat_correct += 1
            else:
                misclassifications.append({
                    "ticket_id": tid,
                    "expected_category": exp_cat,
                    "predicted_category": pred_cat,
                })
            if urg_match:
                urg_correct += 1

            confusion_matrix[str(exp_cat)][str(pred_cat)] += 1

        comparisons.append({
            "ticket_id": tid,
            "expected_category": exp_cat,
            "predicted_category": pred_cat,
            "category_match": cat_match,
            "expected_urgency": exp_urg,
            "predicted_urgency": pred_urg,
            "urgency_match": urg_match,
            "confidence": round(conf, 4),
            "route": route,
        })

    cat_accuracy = round(cat_correct / evaluated_count, 4) if evaluated_count > 0 else None
    urg_accuracy = round(urg_correct / evaluated_count, 4) if evaluated_count > 0 else None

    evaluation_report = {
        "total_tickets": total_tickets,
        "evaluated_tickets": evaluated_count,
        "category_accuracy": cat_accuracy,
        "urgency_accuracy": urg_accuracy,
        "category_correct_count": cat_correct,
        "urgency_correct_count": urg_correct,
        "human_review_count": human_review_count,
        "auto_triage_count": auto_triage_count,
        "parse_validation_failures": parse_failures_count,
    }

    # Format confusion pairs
    confused_pairs = []
    for exp_c, pred_dict in confusion_matrix.items():
        for pred_c, count in pred_dict.items():
            if exp_c != pred_c and count > 0:
                confused_pairs.append({
                    "expected": exp_c,
                    "predicted": pred_c,
                    "count": count,
                })
    confused_pairs.sort(key=lambda x: x["count"], reverse=True)

    # Convert defaultdict to normal dict
    matrix_clean = {k: dict(v) for k, v in confusion_matrix.items()}

    confusion_summary = {
        "confusion_matrix": matrix_clean,
        "misclassifications": misclassifications,
        "most_confused_pairs": confused_pairs,
    }

    return evaluation_report, comparisons, confusion_summary
