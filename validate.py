"""Validation script verifying pipeline outputs against all evaluation requirements."""

import argparse
import json
from pathlib import Path
import sys
from typing import Any


def check_artifact_exists(path: Path, name: str) -> bool:
    if not path.exists():
        print(f"  [FAIL] Missing required artifact: {name} at {path}")
        return False
    print(f"  [PASS] Artifact exists: {name}")
    return True


def load_json(path: Path, name: str) -> tuple[bool, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  [PASS] Valid JSON format: {name}")
        return True, data
    except Exception as e:
        print(f"  [FAIL] Invalid JSON in {name}: {e}")
        return False, None


def validate_pipeline_artifacts(artifacts_dir: Path) -> bool:
    print("=" * 60)
    print("Starting Pipeline Artifact Validation")
    print(f"Inspecting directory: {artifacts_dir.resolve()}")
    print("=" * 60)

    all_passed = True

    required_artifacts = [
        "tickets.json",
        "label_schema.json",
        "preprocessed_tickets.json",
        "routing_decisions.json",
        "triage_results.json",
        "prediction_comparison.json",
        "evaluation_report.json",
        "llm_calls.jsonl",
    ]

    print("\n1. Checking Required Artifact Existence:")
    for art in required_artifacts:
        p = artifacts_dir / art
        if not check_artifact_exists(p, art):
            all_passed = False

    if not all_passed:
        print("\nAborting detailed checks: Missing one or more required artifact files.")
        return False

    print("\n2. Validating JSON Files Integrity:")
    json_artifacts = [
        "tickets.json",
        "label_schema.json",
        "preprocessed_tickets.json",
        "routing_decisions.json",
        "triage_results.json",
        "prediction_comparison.json",
        "evaluation_report.json",
    ]
    loaded_data: dict[str, Any] = {}
    for ja in json_artifacts:
        ok, data = load_json(artifacts_dir / ja, ja)
        if not ok:
            all_passed = False
        else:
            loaded_data[ja] = data

    if not all_passed:
        return False

    tickets = loaded_data["tickets.json"]
    schema = loaded_data["label_schema.json"]
    preprocessed = loaded_data["preprocessed_tickets.json"]
    routing = loaded_data["routing_decisions.json"]
    triage = loaded_data["triage_results.json"]
    eval_rep = loaded_data["evaluation_report.json"]

    ticket_ids = [str(t.get("ticket_id", t.get("id", ""))) for t in tickets]
    categories = set(schema.get("categories", []))
    urgency_levels = set(schema.get("urgency_levels", []))

    print("\n3. Validating Schema Labels:")
    if not categories or not urgency_levels:
        print("  [FAIL] Schema must define non-empty categories and urgency levels.")
        all_passed = False
    else:
        print(f"  [PASS] Schema defines {len(categories)} categories and {len(urgency_levels)} urgency levels.")

    print("\n4. Validating Preprocessed Tickets:")
    for p in preprocessed:
        if not all(k in p for k in ["ticket_id", "original_text", "cleaned_text", "char_count", "word_count"]):
            print(f"  [FAIL] Preprocessed record missing required keys: {p}")
            all_passed = False
            break
    else:
        print(f"  [PASS] All {len(preprocessed)} preprocessed tickets have complete length stats and cleaned text.")

    print("\n5. Validating Routing Decisions:")
    routing_by_id = {str(r.get("ticket_id")): r for r in routing}
    for tid in ticket_ids:
        if tid not in routing_by_id:
            print(f"  [FAIL] Missing routing decision for ticket {tid}.")
            all_passed = False
        else:
            decision = routing_by_id[tid]
            route = decision.get("route")
            if route not in ("auto_triage", "human_review"):
                print(f"  [FAIL] Invalid route '{route}' for ticket {tid}.")
                all_passed = False
            if not isinstance(decision.get("confidence"), (int, float)):
                print(f"  [FAIL] Invalid confidence '{decision.get('confidence')}' for ticket {tid}.")
                all_passed = False
            if not decision.get("routing_reason"):
                print(f"  [FAIL] Missing routing reason for ticket {tid}.")
                all_passed = False
    if all_passed:
        print(f"  [PASS] All {len(ticket_ids)} tickets received valid deterministic routing decisions.")

    print("\n6. Validating Triage Results:")
    triage_by_id = {str(r.get("ticket_id")): r for r in triage}
    for tid in ticket_ids:
        if tid not in triage_by_id:
            print(f"  [FAIL] Missing triage result for ticket {tid}.")
            all_passed = False
            continue

        res = triage_by_id[tid]
        route = res.get("route")
        cat = res.get("predicted_category")
        urg = res.get("predicted_urgency")
        cust_reply = res.get("customer_reply")
        internal_note = res.get("internal_note")

        # Category/urgency validation
        if route == "auto_triage":
            if cat not in categories:
                print(f"  [FAIL] Ticket {tid} predicted category '{cat}' not in schema categories.")
                all_passed = False
            if urg not in urgency_levels:
                print(f"  [FAIL] Ticket {tid} predicted urgency '{urg}' not in schema urgency levels.")
                all_passed = False

            if not cust_reply or not cust_reply.strip():
                print(f"  [FAIL] auto_triage ticket {tid} MUST have non-empty customer_reply.")
                all_passed = False
            if internal_note is not None:
                print(f"  [FAIL] auto_triage ticket {tid} internal_note should be null/None.")
                all_passed = False

        elif route == "human_review":
            if cust_reply is not None:
                print(f"  [FAIL] human_review ticket {tid} customer_reply MUST be null/None.")
                all_passed = False
            if not internal_note or not internal_note.strip():
                print(f"  [FAIL] human_review ticket {tid} MUST have non-empty internal_note explaining escalation.")
                all_passed = False

    if all_passed:
        print(f"  [PASS] All {len(ticket_ids)} triage results adhere strictly to reply/note routing rules.")

    print("\n7. Validating Evaluation Report:")
    req_metrics = [
        "total_tickets",
        "human_review_count",
        "auto_triage_count",
        "parse_validation_failures",
    ]
    for rm in req_metrics:
        if rm not in eval_rep:
            print(f"  [FAIL] Evaluation report missing required metric: {rm}")
            all_passed = False
    if all_passed:
        print(f"  [PASS] Evaluation report contains all required metrics: total={eval_rep.get('total_tickets')}, "
              f"auto={eval_rep.get('auto_triage_count')}, human={eval_rep.get('human_review_count')}, "
              f"failures={eval_rep.get('parse_validation_failures')}.")

    print("\n8. Validating LLM Call Logging (llm_calls.jsonl):")
    jsonl_path = artifacts_dir / "llm_calls.jsonl"
    call_records = []
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if line.strip():
                    rec = json.loads(line)
                    call_records.append(rec)
                    for key in ["stage", "ticket_id", "timestamp", "provider", "model", "prompt_hash", "output_artifact"]:
                        if key not in rec:
                            print(f"  [FAIL] Line {line_no} of llm_calls.jsonl missing key '{key}'.")
                            all_passed = False
        print(f"  [PASS] Logged {len(call_records)} valid LLM calls in {jsonl_path.name}.")
    except Exception as e:
        print(f"  [FAIL] Error reading {jsonl_path.name}: {e}")
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("RESULT: ALL VALIDATION CHECKS PASSED SUCCESSFULLY (100% COMPLIANT)!")
        print("=" * 60)
        return True
    else:
        print("RESULT: VALIDATION FAILED. Please review the failed checks above.")
        print("=" * 60)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate pipeline output artifacts.")
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default=".",
        help="Directory containing output artifacts to validate (default: current directory)",
    )
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    success = validate_pipeline_artifacts(artifacts_dir)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
