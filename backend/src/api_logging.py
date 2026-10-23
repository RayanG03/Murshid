"""Append-only JSONL logging for the API endpoints (data/logs/, gitignored).

One JSON object per line, appended under a lock so concurrent FastAPI
threadpool requests never interleave. Only caller-provided dicts are
serialized — API keys or other secrets can never reach these files, and
the logger never touches anything else.
"""

import json
import threading
from pathlib import Path


class JsonlLogger:
    """Thread-safe JSONL appender plus an answer->question lookup."""

    def __init__(self, logs_dir: Path, name: str) -> None:
        self.path = Path(logs_dir) / f"{name}.jsonl"
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict) -> None:
        """Append one record as a single UTF-8 JSON line."""
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def last_assembly(self, university: str, answer_text: str) -> str | None:
        """Question text of the LAST ask.jsonl entry matching this answer.

        POST /feedback identifies the request by the FULL answer text (API
        contract); this lookup reads ask.jsonl backwards and returns the
        question of the first matching entry (same university + exact
        answer text) or None when there is nothing to match.
        """
        with self._lock:
            if not self.path.exists():
                return None
            lines = self.path.read_text(encoding="utf-8").splitlines()
        for line in reversed(lines):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                record.get("university") == university
                and record.get("answer") == answer_text
            ):
                return record.get("question")
        return None
