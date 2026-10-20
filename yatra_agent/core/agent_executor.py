"""LangChain ReAct Agent Executor assembly for Yatra AI."""

from typing import Any, List


class YatraAgentExecutor:
    """Wrapper around LangChain ReAct agent execution."""

    def __init__(self, tools: List[Any] | None = None):
        self.tools = tools or []

    async def execute_query(self, query: str) -> str:
        """Stub execution for Phase 0."""
        return "Yatra AI Agent initialized."
