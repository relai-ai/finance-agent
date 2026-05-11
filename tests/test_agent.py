from agents import WebSearchTool

from finance_agent.agent import AGENT_INSTRUCTIONS, build_agent


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


def test_agent_instructions_require_claim_level_primary_source_citations() -> None:
    assert "material public-company metrics" in AGENT_INSTRUCTIONS
    assert "investor relations pages" in AGENT_INSTRUCTIONS
    assert "earnings releases" in AGENT_INSTRUCTIONS
    assert "Tie every concrete figure, date, and unit" in AGENT_INSTRUCTIONS
    assert "secondary sources" in AGENT_INSTRUCTIONS
