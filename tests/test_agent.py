from agents import WebSearchTool

from demo_rag_agent.agent import build_agent


def test_build_agent_uses_web_search_tool() -> None:
    agent = build_agent()

    assert agent.name == "Web Retrieval Assistant"
    assert any(isinstance(tool, WebSearchTool) for tool in agent.tools)
