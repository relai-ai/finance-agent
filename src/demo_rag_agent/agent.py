"""Agent construction for the terminal demo."""

from __future__ import annotations

import os

from agents import Agent, WebSearchTool


DEFAULT_MODEL = "gpt-5.4"

AGENT_INSTRUCTIONS = """
You are a concise research assistant in a terminal chat.

Use web search when the user asks about current, recent, external, or factual
information that may benefit from retrieval. Use the retrieved information to
answer directly. If web search does not provide enough support, say what is
missing instead of guessing.

Always end final answers with a Markdown "# References" section. Put every
source URL used to support the answer in that final section, and keep source
links out of the main answer body.

For follow-up questions, use the conversation context and search again when the
answer depends on facts outside the conversation or may have changed.
""".strip()


def build_agent() -> Agent:
    """Build the web-retrieval agent."""
    return Agent(
        name="Web Retrieval Assistant",
        instructions=AGENT_INSTRUCTIONS,
        model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        tools=[WebSearchTool(search_context_size="medium")],
    )
