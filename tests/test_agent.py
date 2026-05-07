from agents import WebSearchTool

from demo_rag_agent.agent import AGENT_INSTRUCTIONS, build_agent


def test_build_agent_uses_web_search_tool() -> None:
    agent = build_agent()

    assert agent.name == "Web Retrieval Assistant"
    assert any(isinstance(tool, WebSearchTool) for tool in agent.tools)


def test_agent_instructions_require_detailed_weather_answers() -> None:
    assert "weather requests" in AGENT_INSTRUCTIONS
    assert "each requested location explicitly" in AGENT_INSTRUCTIONS
    assert "current weather or near-term forecast details" in AGENT_INSTRUCTIONS


def test_agent_instructions_preserve_fahrenheit_only_requests() -> None:
    assert "only in Fahrenheit" in AGENT_INSTRUCTIONS
    assert "unit constraint" in AGENT_INSTRUCTIONS
    assert "do not include Celsius" in AGENT_INSTRUCTIONS
