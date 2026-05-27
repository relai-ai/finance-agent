import json

from finance_agent.logging import ConversationLog, append_round


def test_conversation_log_appends_rounds(tmp_path) -> None:
    path = tmp_path / "logs" / "conversation.json"
    log = ConversationLog(path)

    log.append_round(user="first question", assistant="first answer")
    log.append_round(user="second question", assistant="second answer")

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "rounds": [
            {"user": "first question", "assistant": "first answer"},
            {"user": "second question", "assistant": "second answer"},
        ]
    }


def test_append_round_writes_expected_shape(tmp_path) -> None:
    path = tmp_path / "conversation.json"
    log = ConversationLog(path)

    append_round(
        log,
        user_input="question",
        final_answer="answer",
    )

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "rounds": [{"user": "question", "assistant": "answer"}]
    }
