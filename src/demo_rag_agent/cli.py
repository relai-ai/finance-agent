"""Terminal chat entry point."""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import uuid
from pathlib import Path

from agents import (
    RawResponsesStreamEvent,
    Runner,
    SQLiteSession,
    add_trace_processor,
    flush_traces,
    trace,
)
from dotenv import load_dotenv
from openai.types.responses import ResponseTextDeltaEvent

from demo_rag_agent.agent import build_agent
from demo_rag_agent.logging import JsonlTraceProcessor, JsonlWriter, append_turn_record


DEFAULT_LOG_DIR = Path("logs")
LOG_FILE_PATTERN = re.compile(r"^traces-(\d+)\.jsonl$")
DEFAULT_SESSION_DIR = ".agent_sessions"
EXIT_COMMANDS = {"/exit", "/quit"}


def parse_command(text: str) -> str | None:
    """Return a normalized slash command, or None for normal user input."""
    stripped = text.strip().lower()
    if stripped in EXIT_COMMANDS:
        return "exit"
    if stripped == "/new":
        return "new"
    if stripped == "/help":
        return "help"
    return None


def help_text() -> str:
    return "\n".join(
        [
            "Commands:",
            "  /help  Show this help",
            "  /new   Start a fresh conversation session",
            "  /quit  Exit",
            "  /exit  Exit",
        ]
    )


def new_session_id() -> str:
    return f"session-{uuid.uuid4().hex[:12]}"


def build_session(session_id: str, session_dir: Path) -> SQLiteSession:
    session_dir.mkdir(parents=True, exist_ok=True)
    return SQLiteSession(session_id, str(session_dir / "conversations.db"))


def next_numbered_log_file(log_dir: Path = DEFAULT_LOG_DIR) -> Path:
    """Reserve and return the next traces-NNN.jsonl file path."""
    log_dir.mkdir(parents=True, exist_ok=True)
    existing_numbers: list[int] = []
    for path in log_dir.glob("traces-*.jsonl"):
        match = LOG_FILE_PATTERN.match(path.name)
        if match:
            existing_numbers.append(int(match.group(1)))

    next_number = max(existing_numbers, default=0) + 1
    while True:
        candidate = log_dir / f"traces-{next_number:03d}.jsonl"
        try:
            with candidate.open("x", encoding="utf-8"):
                pass
            return candidate
        except FileExistsError:
            next_number += 1


def explicit_log_file(args: argparse.Namespace) -> Path | None:
    """Return the explicit log path, if the user configured one."""
    configured_log_file = args.log_file or os.getenv("AGENT_LOG_FILE")
    if configured_log_file:
        return Path(configured_log_file)
    return None


def resolve_log_file(args: argparse.Namespace) -> Path:
    """Resolve the log path, using numbered files unless explicitly overridden."""
    if configured_log_file := explicit_log_file(args):
        return configured_log_file
    return next_numbered_log_file()


def trace_metadata(turn_number: int) -> dict[str, str]:
    """Build SDK trace metadata. Values must be strings for trace export."""
    return {"turn_number": str(turn_number)}


async def stream_agent_response(agent, user_input: str, session: SQLiteSession) -> tuple[str, object]:
    """Run the agent in streaming mode and print text deltas as they arrive."""
    result = Runner.run_streamed(agent, user_input, session=session)
    print("\nAssistant: ", end="", flush=True)

    async for event in result.stream_events():
        if isinstance(event, RawResponsesStreamEvent) and isinstance(
            event.data, ResponseTextDeltaEvent
        ):
            print(event.data.delta, end="", flush=True)

    print()
    return str(result.final_output), getattr(getattr(result, "context_wrapper", None), "usage", None)


async def chat(args: argparse.Namespace) -> int:
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is required. See .env.example.")
        return 2

    uses_numbered_logs = explicit_log_file(args) is None
    log_file = resolve_log_file(args)
    writer = JsonlWriter(log_file)
    trace_processor = JsonlTraceProcessor(writer)
    add_trace_processor(trace_processor)

    agent = build_agent()
    session_dir = Path(args.session_dir)
    session_id = args.session_id or os.getenv("AGENT_SESSION_ID") or new_session_id()
    session = build_session(session_id, session_dir)
    turn_number = 0

    print("OpenAI Agents SDK web retrieval demo")
    print(f"Session: {session_id}")
    print(f"Log file: {log_file}")
    print("Type /help for commands.")

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return 0

        if not user_input:
            continue

        command = parse_command(user_input)
        if command == "exit":
            print("Exiting.")
            return 0
        if command == "help":
            print(help_text())
            continue
        if command == "new":
            session_id = new_session_id()
            session = build_session(session_id, session_dir)
            turn_number = 0
            if uses_numbered_logs:
                log_file = next_numbered_log_file()
                writer = JsonlWriter(log_file)
                trace_processor.writer = writer
            print(f"Started new session: {session_id}")
            print(f"Log file: {log_file}")
            continue

        turn_number += 1
        final_answer: str | None = None
        trace_id: str | None = None
        usage = None

        try:
            with trace(
                "demo-rag-agent-turn",
                group_id=session_id,
                metadata=trace_metadata(turn_number),
            ) as turn_trace:
                active_trace = getattr(turn_trace, "trace", None)
                trace_id = getattr(active_trace, "trace_id", None)
                final_answer, usage = await stream_agent_response(agent, user_input, session)

            flush_traces()
            append_turn_record(
                writer,
                session_id=session_id,
                turn_number=turn_number,
                trace_id=trace_id,
                user_input=user_input,
                final_answer=final_answer,
                usage=usage,
                status="ok",
            )
        except Exception as exc:
            flush_traces()
            append_turn_record(
                writer,
                session_id=session_id,
                turn_number=turn_number,
                trace_id=trace_id,
                user_input=user_input,
                final_answer=final_answer,
                usage=usage,
                status="error",
                error=repr(exc),
            )
            print(f"\nError: {exc}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the OpenAI Agents SDK web retrieval demo.")
    parser.add_argument("--session-id", help="Conversation session ID. Defaults to AGENT_SESSION_ID or a new ID.")
    parser.add_argument("--session-dir", default=DEFAULT_SESSION_DIR, help="Directory for SQLite session storage.")
    parser.add_argument(
        "--log-file",
        help="JSONL log file. Defaults to AGENT_LOG_FILE or the next logs/traces-NNN.jsonl file.",
    )
    return parser


def main() -> int:
    return asyncio.run(chat(build_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
