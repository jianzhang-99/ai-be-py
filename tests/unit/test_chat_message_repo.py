from __future__ import annotations

"""Unit tests for ChatMessageRepository"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock

from backend.infra.database.repositories.chat_message_repo import ChatMessageRepository


class TestChatMessageRepositorySaveMessage:
    """save_message 测试"""

    @pytest.mark.asyncio
    async def test_saves_user_message_with_all_fields(self) -> None:
        repo = ChatMessageRepository()
        repo.client.execute_async = AsyncMock()

        await repo.save_message(
            session_id="sess-123",
            role="user",
            content="我想查天气",
            intent="QUERY_WEATHER",
            tool_name=None,
            tool_result=None,
            latency_ms=None,
        )

        repo.client.execute_async.assert_called_once()
        call_args = repo.client.execute_async.call_args
        sql = call_args[0][0]
        params = call_args[0][1]
        assert "INSERT INTO chat_message" in sql
        assert params[0] == "sess-123"
        assert params[1] == "user"
        assert params[2] == "我想查天气"
        assert params[3] == "QUERY_WEATHER"

    @pytest.mark.asyncio
    async def test_saves_assistant_message_with_tool_result(self) -> None:
        repo = ChatMessageRepository()
        repo.client.execute_async = AsyncMock()

        await repo.save_message(
            session_id="sess-456",
            role="assistant",
            content="天气怎么样",
            intent="QUERY_WEATHER",
            tool_name="weather_tool",
            tool_result='{"city": "上海", "temp": 25}',
            latency_ms=120,
        )

        call_args = repo.client.execute_async.call_args
        params = call_args[0][1]
        assert params[0] == "sess-456"
        assert params[1] == "assistant"
        assert params[4] == "weather_tool"
        assert params[5] == '{"city": "上海", "temp": 25}'
        assert params[6] == 120

    @pytest.mark.asyncio
    async def test_saves_minimal_message(self) -> None:
        repo = ChatMessageRepository()
        repo.client.execute_async = AsyncMock()

        await repo.save_message(
            session_id="sess-min",
            role="user",
            content="hello",
        )

        call_args = repo.client.execute_async.call_args
        params = call_args[0][1]
        assert params[3] is None
        assert params[4] is None
        assert params[5] is None
        assert params[6] is None


class TestChatMessageRepositoryListSessionMessages:
    """list_session_messages 测试"""

    @pytest.mark.asyncio
    async def test_returns_ordered_messages(self) -> None:
        rows = [
            {
                "id": 1,
                "session_id": "sess-order",
                "role": "user",
                "content": "第一条",
                "intent": None,
                "tool_name": None,
                "tool_result": None,
                "latency_ms": None,
                "create_time": datetime(2026, 5, 11, 10, 0, 0),
            },
            {
                "id": 2,
                "session_id": "sess-order",
                "role": "assistant",
                "content": "第二条",
                "intent": "QUERY_WEATHER",
                "tool_name": "weather",
                "tool_result": "{}",
                "latency_ms": 50,
                "create_time": datetime(2026, 5, 11, 10, 0, 1),
            },
        ]
        repo = ChatMessageRepository()
        repo.client.fetch_all_async = AsyncMock(return_value=rows)

        result = await repo.list_session_messages("sess-order")

        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_empty_when_no_messages(self) -> None:
        repo = ChatMessageRepository()
        repo.client.fetch_all_async = AsyncMock(return_value=[])

        result = await repo.list_session_messages("sess-empty")

        assert result == []

    @pytest.mark.asyncio
    async def test_respects_pagination_params(self) -> None:
        repo = ChatMessageRepository()
        repo.client.fetch_all_async = AsyncMock(return_value=[])

        await repo.list_session_messages("sess-page", limit=10, offset=20)

        call_args = repo.client.fetch_all_async.call_args
        params = call_args[0][1]
        assert params[1] == 10
        assert params[2] == 20

    @pytest.mark.asyncio
    async def test_queries_by_session_id(self) -> None:
        repo = ChatMessageRepository()
        repo.client.fetch_all_async = AsyncMock(return_value=[])

        await repo.list_session_messages("sess-specific")

        call_args = repo.client.fetch_all_async.call_args
        sql = call_args[0][0]
        assert "sess-specific" in call_args[0][1]
        assert "ORDER BY create_time ASC" in sql