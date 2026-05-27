"""Human-readable conversation logging helpers."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any

class ConversationLog:
    """Store one readable JSON conversation log."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = Lock()

    def append_round(self, *, user: str, assistant: str | None) -> None:
        """Append a user/assistant round to the conversation log."""
        with self._lock:
            log = self._read()
            log["rounds"].append({"user": user, "assistant": assistant or ""})
            self._write(log)

    def _read(self) -> dict[str, list[dict[str, str]]]:
        if not self.path.exists() or not self.path.read_text(encoding="utf-8").strip():
            return {"rounds": []}

        data: Any = json.loads(self.path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("rounds"), list):
            return {"rounds": data["rounds"]}
        raise ValueError(f"{self.path} is not a conversation log")

    def _write(self, log: dict[str, list[dict[str, str]]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(log, ensure_ascii=False, indent=2)
        self.path.write_text(f"{payload}\n", encoding="utf-8")


def append_round(
    log: ConversationLog,
    *,
    user_input: str,
    final_answer: str | None,
) -> None:
    """Append a simple user/assistant round."""
    log.append_round(user=user_input, assistant=final_answer)
