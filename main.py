"""CLI entrypoint for running the AI-powered ticket triage pipeline."""

import argparse
import logging
from pathlib import Path
import sys

from src.pipeline import PipelineConfig, TicketTriagePipeline


def resolve_path(specified: str | None, candidates: list[str]) -> Path:
    """Resolves a file path using specified argument or candidates with resilient fallbacks."""
    if specified:
        p = Path(specified)
        if p.exists():
            return p
        # Check if basename matches any existing candidate
        for c in candidates:
            cand_p = Path(c)
            if cand_p.name == p.name and cand_p.exists():
                return cand_p
        return p
    for c in candidates:
        cand_p = Path(c)
        if cand_p.exists():
            return cand_p
    return Path(candidates[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the AI-powered customer support ticket triage pipeline."
    )
    parser.add_argument(
        "--tickets",
        type=str,
        default=None,
        help="Path to tickets JSON file (default: input/tickets.json or tickets.json)",
    )
    parser.add_argument(
        "--schema",
        type=str,
        default=None,
        help="Path to label schema JSON file (default: input/label_schema.json or label_schema.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory where pipeline artifacts will be saved (default: current directory)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="auto",
        choices=["auto", "mock", "openai", "anthropic", "gemini"],
        help="LLM provider to use (default: auto)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific model name to use (e.g., gpt-4o-mini, claude-3-5-sonnet, gemini-1.5-flash)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional API key for external LLM provider",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="Confidence threshold for auto-triage vs human review (default: 0.65)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed debug logging",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Resolve default paths with resilient fallbacks
    tickets_path = resolve_path(args.tickets, ["tickets.json", "input/tickets.json"])
    schema_path = resolve_path(args.schema, ["label_schema.json", "input/label_schema.json"])
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("AI-Powered Ticket Triage Pipeline")
    print("=" * 60)
    print(f"Tickets File   : {tickets_path}")
    print(f"Label Schema   : {schema_path}")
    print(f"Output Dir     : {output_dir.resolve()}")
    print(f"LLM Provider   : {args.provider}")
    print(f"Confidence Cut : {args.threshold}")
    print("-" * 60)

    if not tickets_path.exists():
        print(f"Error: Tickets file not found at {tickets_path}", file=sys.stderr)
        return 1
    if not schema_path.exists():
        print(f"Error: Label schema file not found at {schema_path}", file=sys.stderr)
        return 1

    config = PipelineConfig(
        tickets_path=tickets_path,
        schema_path=schema_path,
        output_dir=output_dir,
        confidence_threshold=args.threshold,
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        copy_inputs_to_output=True,
    )

    try:
        pipeline = TicketTriagePipeline(config)
        summary = pipeline.run()

        report = summary["evaluation_report"]
        print("\n" + "=" * 60)
        print("Pipeline Execution Completed Successfully!")
        print("=" * 60)
        print(f"Total Tickets Processed : {report['total_tickets']}")
        print(f"Auto-Triaged Tickets    : {report['auto_triage_count']}")
        print(f"Human-Review Escalated  : {report['human_review_count']}")
        if report["category_accuracy"] is not None:
            print(f"Category Accuracy       : {report['category_accuracy'] * 100:.1f}%")
        if report["urgency_accuracy"] is not None:
            print(f"Urgency Accuracy        : {report['urgency_accuracy'] * 100:.1f}%")
        print(f"Parse Failures          : {report['parse_validation_failures']}")
        print("-" * 60)
        print("Generated Artifacts:")
        artifacts = [
            "tickets.json",
            "label_schema.json",
            "preprocessed_tickets.json",
            "routing_decisions.json",
            "triage_results.json",
            "prediction_comparison.json",
            "evaluation_report.json",
            "confusion_summary.json",
            "llm_calls.jsonl",
        ]
        for art in artifacts:
            art_file = output_dir / art
            exists = "OK" if art_file.exists() else "FAIL"
            print(f"  [{exists}] {art}")
        print("=" * 60)
        return 0

    except Exception as e:
        logging.exception(f"Pipeline execution failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
