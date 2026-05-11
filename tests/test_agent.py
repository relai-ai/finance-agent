from agents import WebSearchTool

from demo_rag_agent.agent import AGENT_INSTRUCTIONS, build_agent


def test_build_agent_uses_web_search_tool() -> None:
    agent = build_agent()

    assert agent.name == "Financial Research Agent"
    assert any(isinstance(tool, WebSearchTool) for tool in agent.tools)


def test_agent_instructions_are_finance_focused() -> None:
    assert "financial research agent" in AGENT_INSTRUCTIONS
    assert "primary sources" in AGENT_INSTRUCTIONS
    assert "SEC filings" in AGENT_INSTRUCTIONS
    assert "concrete figures, dates, units" in AGENT_INSTRUCTIONS
    assert "source citations" in AGENT_INSTRUCTIONS
