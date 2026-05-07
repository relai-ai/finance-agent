"""JSONL trace and turn logging helpers."""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

from agents.tracing import TracingProcessor


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(UTC).isoformat()


def to_jsonable(value: Any) -> Any:
    """Best-effort conversion of SDK objects into JSON-serializable values."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [to_jsonable(item) for item in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(dataclasses.asdict(value))
    if hasattr(value, "model_dump"):
        return to_jsonable(value.model_dump())
    if hasattr(value, "__dict__"):
        return to_jsonable(vars(value))
    return repr(value)


class JsonlWriter:
    """Append JSON records to a file, creating parent directories as needed."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = Lock()

    def append(self, record: dict[str, Any]) -> None:
        payload = to_jsonable(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        with self._lock, self.path.open("a", encoding="utf-8") as file:
            file.write(f"{line}\n")


class JsonlTraceProcessor(TracingProcessor):
    """Trace processor that mirrors SDK trace/span exports into JSONL."""

    def __init__(self, writer: JsonlWriter) -> None:
        self.writer = writer

    def _append_export(self, event: str, item: Any) -> None:
        try:
            exported = item.export() if hasattr(item, "export") else to_jsonable(item)
            self.writer.append(
                {
                    "timestamp": utc_now_iso(),
                    "record_type": event,
                    "data": exported,
                }
            )
        except Exception as exc:  # pragma: no cover - tracing must not break runs
            self.writer.append(
                {
                    "timestamp": utc_now_iso(),
                    "record_type": "trace_logger_error",
                    "error": repr(exc),
                }
            )

    def on_trace_start(self, trace: Any) -> None:
        self._append_export("trace_start", trace)

    def on_trace_end(self, trace: Any) -> None:
        self._append_export("trace_end", trace)

    def on_span_start(self, span: Any) -> None:
        self._append_export("span_start", span)

    def on_span_end(self, span: Any) -> None:
        self._append_export("span_end", span)

    def shutdown(self) -> None:
        return None

    def force_flush(self) -> None:
        return None


def append_turn_record(
    writer: JsonlWriter,
    *,
    session_id: str,
    turn_number: int,
    trace_id: str | None,
    user_input: str,
    final_answer: str | None,
    status: str,
    usage: Any = None,
    error: str | None = None,
) -> None:
    """Append a per-turn summary record."""
    writer.append(
        {
            "timestamp": utc_now_iso(),
            "record_type": "turn",
            "session_id": session_id,
            "turn_number": turn_number,
            "trace_id": trace_id,
            "user_input": user_input,
            "final_answer": final_answer,
            "usage": to_jsonable(usage),
            "status": status,
            "error": error,
        }
    )
