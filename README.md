# finance-agent

A small terminal chat demo using the OpenAI Agents SDK for financial research.

The agent uses the SDK's hosted `WebSearchTool` to answer finance-related
questions with current web research, prefers primary sources such as investor
relations pages and SEC filings, keeps multi-turn conversation state in a local
SQLite session, and mirrors SDK trace/span exports plus turn summaries into a
JSONL log file.

## Setup

Install dependencies with `uv`:

```bash
uv sync
```

Set your API key:

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY.
```

Optional environment variables:

- `OPENAI_MODEL`: model used by the agent, default `gpt-5.4`
- `AGENT_SESSION_ID`: reuse a specific terminal conversation session
- `AGENT_LOG_FILE`: explicit JSONL log path override. By default each run
  reserves the next numbered file, such as `logs/traces-001.jsonl`.

## Run

```bash
uv run finance-agent
```

Useful commands inside the chat:

- `/help`: show commands
- `/new`: start a fresh conversation session
- `/quit` or `/exit`: stop the program

Manual smoke test:

1. Ask a current finance question that requires web research.
2. Ask a follow-up that depends on the previous answer.
3. Confirm the printed `logs/traces-NNN.jsonl` file exists and contains trace/span records plus
   `turn` records.

## Development

```bash
uv run pytest
```

Runtime artifacts are ignored by git:

- `.agent_sessions/`
- `logs/`
