# AI-Powered Customer Support Ticket Triage Pipeline

A modular, replayable AI-powered ticket triage pipeline built in Python that loads customer support tickets from local disk files, applies deterministic preprocessing, classifies each ticket into category and urgency using structured model outputs, enforces confidence-based deterministic routing to human review, generates short draft replies for auto-triaged cases, and computes evaluation metrics with a category confusion summary.

---

## Key Features & Compliance

1. **Strict Pipeline Lifecycle Enforcement**: Enforces an explicit 11-stage pipeline state machine:
   ```text
   INIT -> INPUTS_LOADED -> TEXT_PREPROCESSED -> MODEL_PROMPTED ->
   STRUCTURED_OUTPUT_PARSED -> CONFIDENCE_CHECKED -> ROUTED ->
   RESPONSE_GENERATED -> RESULTS_SAVED -> EVALUATION_COMPUTED ->
   VALIDATION_COMPLETED
   ```
2. **Deterministic Preprocessing**: Whitespace normalization, repeated punctuation compression (e.g. `???` -> `?`, `!!!` -> `!`), and length statistics (`char_count`, `word_count`). Saved to `preprocessed_tickets.json`.
3. **Structured Classification with Schema Constraints**: Validates against categories and urgency levels in `label_schema.json`. Invalid labels or malformed payloads are safely rejected.
4. **Prompt & Output Robustness (Recovery Path)**: Automatically extracts JSON wrapped in noisy text or markdown code fences (` ```json `), fixes trailing commas, and executes a stricter retry prompt if initial validation fails.
5. **Deterministic Routing Logic**:
   - Tickets with `confidence < 0.65` or explicit model review flags are routed to `human_review`.
   - Invalid or unparsable model outputs are routed to `human_review` with an explanatory error reason.
   - Valid classifications with `confidence >= 0.65` are routed to `auto_triage`.
   - Outputs saved to `routing_decisions.json`.
6. **Suggested Replies & Escalation Notes**:
   - `auto_triage`: Generates a concise (2–4 sentences), polite draft response acknowledging the customer's issue without making up account-specific facts.
   - `human_review`: Produces an internal escalation note (`customer_reply` is `null`).
   - Saved to `triage_results.json`.
7. **Basic Evaluation & Confusion Summary (Stretch Goal)**:
   - Computes category accuracy, urgency accuracy, review counts, and parse failures.
   - Saves per-ticket comparison to `prediction_comparison.json`.
   - Saves full metrics to `evaluation_report.json`.
   - Saves confusion matrix and misclassification pairs to `confusion_summary.json`.
8. **LLM Call Audit Logging**: Appends every LLM interaction to `llm_calls.jsonl` with ISO-8601 timestamps, provider, model name, SHA-256 prompt hash, and references to raw saved output artifacts.
9. **Zero-Dependency Core**: Runs using standard Python 3 (3.10+) with standard library modules. Can run completely offline using the built-in deterministic `mock` provider or connect to live LLM providers (`openai`, `anthropic`, `gemini`).

---

## Project Structure

```text
Deriv-AI-Assessment/
├── input/
│   ├── tickets.json               # Input tickets fixture
│   └── label_schema.json          # Input category and urgency schema
├── src/
│   ├── __init__.py
│   ├── stages.py                  # 11-stage pipeline lifecycle & state tracker
│   ├── preprocessing.py           # Deterministic text cleaning & stats computation
│   ├── schema.py                  # Schema models & strict output validation
│   ├── routing.py                 # Confidence threshold & deterministic routing
│   ├── llm_client.py              # Multi-provider LLM interface (Mock, OpenAI, Anthropic, Gemini)
│   ├── reply_generator.py         # Customer draft replies & internal escalation notes
│   ├── evaluation.py              # Accuracy metrics, comparisons, & confusion summary
│   ├── logger.py                  # JSONL audit logger & artifact persistence
│   └── pipeline.py                # Pipeline orchestrator
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py      # Unit tests for text cleaning and stats
│   ├── test_schema.py             # Unit tests for schema validation and label constraints
│   ├── test_routing.py            # Unit tests for threshold routing (< 0.65 vs >= 0.65)
│   ├── test_evaluation.py         # Unit tests for evaluation metrics and confusion analysis
│   ├── test_robustness.py         # Unit tests for noisy JSON extraction & error recovery
│   └── test_pipeline.py          # End-to-end integration test of the full pipeline
├── main.py                        # CLI entrypoint for running the pipeline
├── validate.py                    # Verification script checking all requirements
├── Makefile                       # Optional helper commands (run, validate, test)
├── requirements.txt               # Optional dependencies
├── requirements.md                # Assessment specification
└── README.md                      # Documentation
```

---

## Quickstart Guide

### 1. Run the Pipeline

Run the pipeline from the workspace root:

```bash
python main.py
```

By default, this loads `input/tickets.json` and `input/label_schema.json`, automatically uses the built-in deterministic LLM provider (or detects active API keys), and generates all required artifacts.

#### CLI Options

You can specify custom input files, output directories, confidence thresholds, or live LLM providers:

```bash
# Custom input files and threshold
python main.py --tickets input/tickets.json --schema input/label_schema.json --threshold 0.70

# Using live OpenAI provider
python main.py --provider openai --model gpt-4o-mini --api-key sk-...

# Using live Anthropic provider
python main.py --provider anthropic --model claude-3-5-sonnet-20241022

# Using live Gemini provider
python main.py --provider gemini --model gemini-1.5-flash
```

---

### 2. Validate Artifacts

Run the validation suite to ensure that all generated artifacts adhere to all project requirements:

```bash
python validate.py
```

Validation verifies:
- All required artifacts exist on disk.
- All JSON files are well-formed.
- Schema defines valid categories and urgency levels.
- Preprocessed tickets contain complete length statistics (`char_count`, `word_count`) and cleaned text.
- Every ticket has a valid deterministic routing decision (`auto_triage` or `human_review`).
- All `auto_triage` tickets have a valid draft customer reply and `internal_note` is null.
- All `human_review` tickets have a valid internal escalation note and `customer_reply` is null.
- Evaluation metrics can be computed.
- `llm_calls.jsonl` contains valid audit logs with prompt hashes and linked raw output files.

---

### 3. Run Automated Tests

Run the unit and integration test suite:

```bash
python -m unittest discover tests
```

Or using `pytest` if installed:

```bash
pytest
```

---

## Generated Artifacts Specification

| File | Description |
| :--- | :--- |
| `tickets.json` | Input tickets fixture. |
| `label_schema.json` | Allowed category and urgency schema. |
| `preprocessed_tickets.json` | Tickets with cleaned text and character/word statistics. |
| `routing_decisions.json` | Deterministic routing decision (`auto_triage` / `human_review`) with reasons. |
| `triage_results.json` | Final triage predictions, customer reply (if auto-triaged), or internal escalation note (if human review). |
| `prediction_comparison.json` | Per-ticket comparison between ground truth and predicted labels. |
| `evaluation_report.json` | Overall accuracy metrics, review counts, and parse failure counts. |
| `confusion_summary.json` | Category confusion matrix and misclassification pairs. |
| `llm_calls.jsonl` | Structured log of every LLM call with prompt hash and raw response artifact path. |