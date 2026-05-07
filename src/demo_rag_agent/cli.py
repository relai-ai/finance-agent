"""Terminal chat entry point."""

from __future__ import annotations

import argparse
import asyncio
import os
import uuid
from pathlib import Path

from agents import RawResponsesStreamEvent, Runner, SQLiteSession, add_trace_processor, flush_traces, trace
from dotenv import load_dotenv
from openai.types.responses import ResponseTextDeltaEvent

from demo_rag_agent.agent import build_agent
from demo_rag_agent.logging import JsonlTraceProcessor, JsonlWriter, append_turn_record


DEFAULT_LOG_FILE = "logs/traces.jsonl"
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

    log_file = Path(args.log_file or os.getenv("AGENT_LOG_FILE", DEFAULT_LOG_FILE))
    writer = JsonlWriter(log_file)
    add_trace_processor(JsonlTraceProcessor(writer))

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
            print(f"Started new session: {session_id}")
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
    parser.add_argument("--log-file", help=f"JSONL log file. Defaults to AGENT_LOG_FILE or {DEFAULT_LOG_FILE}.")
    return parser


def main() -> int:
    return asyncio.run(chat(build_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
