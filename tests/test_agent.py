from agents import WebSearchTool

from finance_agent.agent import AGENT_INSTRUCTIONS, build_agent


def test_build_agent_uses_web_search_tool() -> None:
    agent = build_agent()

    assert agent.name == "Financial Research Agent"
    assert any(isinstance(tool, WebSearchTool) for tool in agent.tools)
