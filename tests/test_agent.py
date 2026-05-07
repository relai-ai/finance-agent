from agents import WebSearchTool

from demo_rag_agent.agent import AGENT_INSTRUCTIONS, build_agent


def test_build_agent_uses_web_search_tool() -> None:
    agent = build_agent()

    assert agent.name == "Web Retrieval Assistant"
    assert any(isinstance(tool, WebSearchTool) for tool in agent.tools)


def test_agent_instructions_require_terminal_references_section() -> None:
    assert "Never put citations, URLs, source names, or markdown links in the answer body." in (
        AGENT_INSTRUCTIONS
    )
    assert 'exactly "# References"' in AGENT_INSTRUCTIONS
    assert "Nothing may appear after that section." in AGENT_INSTRUCTIONS


def test_agent_instructions_require_light_comic_tone() -> None:
    assert "Keep the final answer lightly comic" in AGENT_INSTRUCTIONS
    assert "witty phrase or playful aside" in AGENT_INSTRUCTIONS
