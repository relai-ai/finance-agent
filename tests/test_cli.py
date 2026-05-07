import asyncio

from agents import RawResponsesStreamEvent
from openai.types.responses import ResponseTextDeltaEvent

from demo_rag_agent import cli
from demo_rag_agent.cli import help_text, parse_command, trace_metadata


def test_parse_command_normalizes_exit_commands() -> None:
    assert parse_command(" /EXIT ") == "exit"
    assert parse_command("/quit") == "exit"


def test_parse_command_handles_known_commands() -> None:
    assert parse_command("/new") == "new"
    assert parse_command("/help") == "help"


def test_parse_command_returns_none_for_user_input() -> None:
    assert parse_command("what happened today?") is None


def test_help_text_lists_commands() -> None:
    text = help_text()

    assert "/help" in text
    assert "/new" in text
    assert "/quit" in text


def test_trace_metadata_values_are_strings() -> None:
    assert trace_metadata(3) == {"turn_number": "3"}


def test_stream_agent_response_prints_text_deltas(monkeypatch, capsys) -> None:
    class FakeResult:
        final_output = "Hello world"
        context_wrapper = None

        async def stream_events(self):
            yield RawResponsesStreamEvent(
                data=ResponseTextDeltaEvent(
                    content_index=0,
                    delta="Hello",
                    item_id="item_1",
                    logprobs=[],
                    output_index=0,
                    sequence_number=0,
                    type="response.output_text.delta",
                )
            )
            yield RawResponsesStreamEvent(
                data=ResponseTextDeltaEvent(
                    content_index=0,
                    delta=" world",
                    item_id="item_1",
                    logprobs=[],
                    output_index=0,
                    sequence_number=1,
                    type="response.output_text.delta",
                )
            )

    monkeypatch.setattr(cli.Runner, "run_streamed", lambda *args, **kwargs: FakeResult())

    final_answer, usage = asyncio.run(cli.stream_agent_response(object(), "hi", object()))

    assert final_answer == "Hello world"
    assert usage is None
    assert capsys.readouterr().out == "\nAssistant: Hello world\n"
