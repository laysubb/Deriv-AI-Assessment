"""Logging utility for tracking LLM calls and artifact persistence."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


def compute_prompt_hash(prompt: str) -> str:
    """Computes a SHA-256 hash for a prompt string."""
    return hashlib.sha256(prompt.strip().encode("utf-8")).hexdigest()


class LLMCallLogger:
    """Logs LLM calls into a JSONL file and records raw response artifacts."""

    def __init__(
        self,
        log_file_path: Path | str,
        raw_artifacts_dir: Path | str,
        reset_existing: bool = True,
    ):
        self.log_file_path = Path(log_file_path)
        self.raw_artifacts_dir = Path(raw_artifacts_dir)
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.raw_artifacts_dir.mkdir(parents=True, exist_ok=True)
        if reset_existing and self.log_file_path.exists():
            self.log_file_path.unlink()

    def log_call(
        self,
        stage: str,  # "classification" | "reply_generation"
        ticket_id: str,
        provider: str,
        model: str,
        prompt: str,
        raw_response: str,
        attempt: int = 1,
    ) -> dict[str, Any]:
        """Saves raw output artifact and appends a structured log entry to JSONL."""
        timestamp = datetime.now(timezone.utc).isoformat()
        prompt_hash = compute_prompt_hash(prompt)

        # Save raw output artifact to disk
        safe_ticket_id = re_safe_id = "".join(c for c in ticket_id if c.isalnum() or c in ("-", "_"))
        artifact_filename = f"{safe_ticket_id}_{stage}_attempt{attempt}.txt"
        artifact_path = self.raw_artifacts_dir / artifact_filename
        with open(artifact_path, "w", encoding="utf-8") as f:
            f.write(raw_response)

        log_entry = {
            "stage": stage,
            "ticket_id": ticket_id,
            "timestamp": timestamp,
            "provider": provider,
            "model": model,
            "prompt_hash": prompt_hash,
            "output_artifact": str(artifact_path).replace("\\", "/"),
        }

        # Append to JSONL file
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        return log_entry
