from __future__ import annotations

import asyncio
import inspect
import os
import uuid
from pathlib import Path

from agents import Runner, SQLiteSession

from finance_agent.agent import build_agent
from relai_simulator.adapter_contract import AgentAdapter, AgentTurnResult

_SIM_DIR = Path(__file__).resolve().parents[2]
_DEFAULT_SESSION_DIR = _SIM_DIR / ".cache" / "sessions"


class ProjectAgentAdapter:
    def __init__(self) -> None:
        self._agent = build_agent()
        self.agent_or_tools = self._agent
        self._session = SQLiteSession(
            os.getenv("RELAI_SIMULATOR_SESSION_ID", f"relai-sim-{uuid.uuid4().hex[:12]}"),
            str(self._session_db_path()),
        )

    async def run_turn(self, user_message: str) -> AgentTurnResult:
        result = await self._run_agent(user_message)
        final_output = getattr(result, "final_output", None)
        return AgentTurnResult(
            assistant_message=None if final_output is None else str(final_output)
        )

    async def _run_agent(self, user_message: str):
        runner_run = getattr(Runner, "run", None)
        if callable(runner_run):
            result = runner_run(self._agent, user_message, session=self._session)
            if inspect.isawaitable(result):
                return await result
            return result
        return await asyncio.to_thread(
            Runner.run_sync,
            self._agent,
            user_message,
            session=self._session,
        )

    def _session_db_path(self) -> Path:
        session_dir = Path(
            os.getenv("RELAI_SIMULATOR_SESSION_DIR", str(_DEFAULT_SESSION_DIR))
        )
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir / "conversations.db"


def build_agent_adapter() -> AgentAdapter:
    return ProjectAgentAdapter()
