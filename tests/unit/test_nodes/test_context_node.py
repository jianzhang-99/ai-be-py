from __future__ import annotations

import pytest

from backend.graph.nodes.context_node import ContextNode, MEMORY_HINT


class StubChatLogRepo:
    async def list_by_session_id(self, session_id: str, limit: int = 10) -> list[dict]:
        return [
            {"user_input": "上一轮用户问题", "ai_response": "上一轮助手回复"},
        ]


class StubChatSummaryRepo:
    async def list_by_session_id(self, session_id: str, limit: int = 5) -> list[dict]:
        return [
            {"summary_content": "用户最近在查南京天气"},
            {"summary_content": "用户偏好简洁回答"},
        ]


class StubUserMemoryRepo:
    async def list_active(
        self,
        user_id: int | None = None,
        phone: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        return [
            {"memory_content": "用户常问长江沿线天气"},
        ]


class StubSessionStore:
    async def get_session_data(self, session_id: str) -> dict:
        return {
            "current_scene": "query_weather",
            "state": "WAITING_USER",
        }


class TestContextNode:
    @pytest.mark.asyncio
    async def test_entrypoint_loads_context_and_merges_history(self) -> None:
        node = ContextNode(
            chat_log_repo=StubChatLogRepo(),
            chat_summary_repo=StubChatSummaryRepo(),
            user_memory_repo=StubUserMemoryRepo(),
            session_store=StubSessionStore(),
        )

        result = await node.entrypoint(
            {
                "session_id": "sess-001",
                "user_id": "1",
                "history": [{"role": "user", "content": "这一轮的新问题"}],
            }
        )

        assert result["working_memory"]["current_scene"] == "query_weather"
        assert result["history"] == [
            {"role": "user", "content": "上一轮用户问题"},
            {"role": "assistant", "content": "上一轮助手回复"},
            {"role": "user", "content": "这一轮的新问题"},
        ]
        assert result["chat_summaries"] == "- 用户偏好简洁回答\n- 用户最近在查南京天气"
        assert result["user_profile"] == "- 用户常问长江沿线天气"
        assert result["memory_hint"] == MEMORY_HINT

    @pytest.mark.asyncio
    async def test_entrypoint_keeps_request_history_when_same_as_persisted(self) -> None:
        node = ContextNode(
            chat_log_repo=StubChatLogRepo(),
            chat_summary_repo=StubChatSummaryRepo(),
            user_memory_repo=StubUserMemoryRepo(),
            session_store=StubSessionStore(),
        )

        same_history = [
            {"role": "user", "content": "上一轮用户问题"},
            {"role": "assistant", "content": "上一轮助手回复"},
        ]
        result = await node.entrypoint(
            {
                "session_id": "sess-001",
                "user_id": "1",
                "history": same_history,
            }
        )

        assert result["history"] == same_history
