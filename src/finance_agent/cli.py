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
)
from dotenv import load_dotenv
from openai.types.responses import ResponseTextDeltaEvent

from finance_agent.agent import build_agent
from finance_agent.logging import ConversationLog, append_round


DEFAULT_LOG_DIR = Path("logs")
LOG_FILE_PATTERN = re.compile(r"^conversation-(\d+)\.json$")
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
    """Reserve and return the next conversation-NNN.json file path."""
    log_dir.mkdir(parents=True, exist_ok=True)
    existing_numbers: list[int] = []
    for path in log_dir.glob("conversation-*.json"):
        match = LOG_FILE_PATTERN.match(path.name)
        if match:
            existing_numbers.append(int(match.group(1)))

    next_number = max(existing_numbers, default=0) + 1
    while True:
        candidate = log_dir / f"conversation-{next_number:03d}.json"
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
    conversation_log = ConversationLog(log_file)

    agent = build_agent()
    session_dir = Path(args.session_dir)
    session_id = args.session_id or os.getenv("AGENT_SESSION_ID") or new_session_id()
    session = build_session(session_id, session_dir)

    print("OpenAI Agents SDK financial research demo")
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
            if uses_numbered_logs:
                log_file = next_numbered_log_file()
                conversation_log = ConversationLog(log_file)
            print(f"Started new session: {session_id}")
            print(f"Log file: {log_file}")
            continue

        final_answer: str | None = None

        try:
            final_answer, _usage = await stream_agent_response(agent, user_input, session)
            append_round(
                conversation_log,
                user_input=user_input,
                final_answer=final_answer,
            )
        except Exception as exc:
            append_round(
                conversation_log,
                user_input=user_input,
                final_answer=final_answer,
            )
            print(f"\nError: {exc}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the OpenAI Agents SDK financial research demo."
    )
    parser.add_argument("--session-id", help="Conversation session ID. Defaults to AGENT_SESSION_ID or a new ID.")
    parser.add_argument("--session-dir", default=DEFAULT_SESSION_DIR, help="Directory for SQLite session storage.")
    parser.add_argument(
        "--log-file",
        help="Conversation JSON log file. Defaults to AGENT_LOG_FILE or the next logs/conversation-NNN.json file.",
    )
    return parser


def main() -> int:
    return asyncio.run(chat(build_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
