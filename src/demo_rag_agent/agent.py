"""Agent construction for the terminal financial research demo."""

from __future__ import annotations

import os

from agents import Agent, WebSearchTool


DEFAULT_MODEL = "gpt-5.4"

AGENT_INSTRUCTIONS = """
You are a financial research agent.

Your job is to answer finance-related questions by combining reasoning with web research.

Rules:
- Use web search when the answer depends on current or recent information.
- Prefer primary sources when available, such as company investor relations pages, earnings releases, SEC filings.
- Surface concrete figures, dates, units, and context when they matter.
- If multiple sources disagree, say so and explain the most likely reason.
- Do not invent numbers. If the data is unavailable or unclear, say that explicitly.
- Keep the answer concise but decision-useful.
- Include source citations in the final answer.
""".strip()


def build_agent() -> Agent:
    """Build the financial research agent."""
    return Agent(
        name="Financial Research Agent",
        instructions=AGENT_INSTRUCTIONS,
        model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        tools=[WebSearchTool(search_context_size="medium")],
    )
