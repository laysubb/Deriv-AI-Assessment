.PHONY: run validate test clean help

help:
	@echo "Available commands:"
	@echo "  make run       - Run the ticket triage pipeline"
	@echo "  make validate  - Validate generated output artifacts"
	@echo "  make test      - Run automated unit and integration tests"
	@echo "  make clean     - Clean generated output artifacts"

run:
	python main.py --tickets input/tickets.json --schema input/label_schema.json

validate:
	python validate.py

test:
	python -m unittest discover tests

clean:
	python -c "import os, glob; [os.remove(f) for f in glob.glob('*.json') if f not in ['tickets.json', 'label_schema.json'] and os.path.exists(f)]"
	python -c "import os; os.remove('llm_calls.jsonl') if os.path.exists('llm_calls.jsonl') else None"
