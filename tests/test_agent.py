from agents import WebSearchTool

from demo_rag_agent.agent import AGENT_INSTRUCTIONS, build_agent


def test_build_agent_uses_web_search_tool() -> None:
    agent = build_agent()

    assert agent.name == "Web Retrieval Assistant"
    assert any(isinstance(tool, WebSearchTool) for tool in agent.tools)


def test_agent_instructions_require_final_references_section() -> None:
    normalized_instructions = " ".join(AGENT_INSTRUCTIONS.split())

    assert 'Markdown "# References" section' in AGENT_INSTRUCTIONS
    assert "keep source links out of the main answer body" in normalized_instructions
