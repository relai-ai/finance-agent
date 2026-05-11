import json

from finance_agent.logging import JsonlWriter, append_turn_record


def test_jsonl_writer_appends_records(tmp_path) -> None:
    path = tmp_path / "logs" / "traces.jsonl"
    writer = JsonlWriter(path)

    writer.append({"record_type": "example", "value": {"nested": True}})
    writer.append({"record_type": "second", "value": 2})

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert records == [
        {"record_type": "example", "value": {"nested": True}},
        {"record_type": "second", "value": 2},
    ]


def test_append_turn_record_writes_expected_shape(tmp_path) -> None:
    path = tmp_path / "traces.jsonl"
    writer = JsonlWriter(path)

    append_turn_record(
        writer,
        session_id="session-1",
        turn_number=3,
        trace_id="trace_123",
        user_input="question",
        final_answer="answer",
        usage={"total_tokens": 10},
        status="ok",
    )

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["record_type"] == "turn"
    assert record["session_id"] == "session-1"
    assert record["turn_number"] == 3
    assert record["trace_id"] == "trace_123"
    assert record["user_input"] == "question"
    assert record["final_answer"] == "answer"
    assert record["usage"] == {"total_tokens": 10}
    assert record["status"] == "ok"
