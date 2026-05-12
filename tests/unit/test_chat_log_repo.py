from __future__ import annotations

"""Unit tests for ChatLogRepository (ai_chat_log table)"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock

from backend.infra.database.repositories.chat_log_repo import ChatLogRepository


class TestChatLogRepositorySaveLog:
    """save_log 测试"""

    @pytest.mark.asyncio
    async def test_saves_user_and_ai_message(self) -> None:
        repo = ChatLogRepository()
        repo.client.fetch_one_async = AsyncMock(return_value={"max_seq": 0})
        repo.client.execute_async = AsyncMock()

        await repo.save_log(
            session_id="sess-123",
            phone="17327756086",
            user_id=1,
            user_input="南京明天天气",
            ai_response="南京明天晴，温度15-25度",
            intent_code="QUERY_WEATHER",
            intent_name="天气查询",
        )

        repo.client.execute_async.assert_called_once()
        call_args = repo.client.execute_async.call_args
        sql = call_args[0][0]
        params = call_args[0][1]
        assert "INSERT INTO ai_chat_log" in sql
        assert params[0] == "sess-123"
        assert params[2] == "17327756086"
        assert params[3] == 1
        assert params[4] == "南京明天天气"
        assert params[5] == "南京明天晴，温度15-25度"
        assert params[6] == "QUERY_WEATHER"

    @pytest.mark.asyncio
    async def test_increments_seq_per_session(self) -> None:
        repo = ChatLogRepository()
        repo.client.fetch_one_async = AsyncMock(return_value={"max_seq": 5})
        repo.client.execute_async = AsyncMock()

        await repo.save_log(
            session_id="sess-456",
            phone="17327756086",
            user_id=1,
            user_input="test",
            ai_response="response",
        )

        call_args = repo.client.execute_async.call_args
        params = call_args[0][1]
        assert params[1] == 6  # max_seq(5) + 1

    @pytest.mark.asyncio
    async def test_first_message_seq_is_one(self) -> None:
        repo = ChatLogRepository()
        repo.client.fetch_one_async = AsyncMock(return_value={"max_seq": 0})
        repo.client.execute_async = AsyncMock()

        await repo.save_log(
            session_id="sess-new",
            phone="17327756086",
            user_id=1,
            user_input="hello",
            ai_response="hi",
        )

        call_args = repo.client.execute_async.call_args
        params = call_args[0][1]
        assert params[1] == 1

    @pytest.mark.asyncio
    async def test_save_log_returns_seq(self) -> None:
        repo = ChatLogRepository()
        repo.client.fetch_one_async = AsyncMock(return_value={"max_seq": 3})
        repo.client.execute_async = AsyncMock()

        result = await repo.save_log(
            session_id="sess-ret",
            phone="17327756086",
            user_id=1,
            user_input="msg",
            ai_response="resp",
        )

        assert result == 4

    @pytest.mark.asyncio
    async def test_handles_null_max_seq(self) -> None:
        repo = ChatLogRepository()
        repo.client.fetch_one_async = AsyncMock(return_value=None)
        repo.client.execute_async = AsyncMock()

        await repo.save_log(
            session_id="sess-null",
            phone="17327756086",
            user_id=1,
            user_input="msg",
            ai_response="resp",
        )

        call_args = repo.client.execute_async.call_args
        params = call_args[0][1]
        assert params[1] == 1


class TestChatLogRepositoryListBySessionId:
    """list_by_session_id 测试"""

    @pytest.mark.asyncio
    async def test_returns_messages_ordered_by_time(self) -> None:
        rows = [
            {
                "id": 1,
                "session_id": "sess-123",
                "seq": 1,
                "phone": "17327756086",
                "user_input": "你好",
                "ai_response": "你好啊",
                "intent_code": None,
                "intent_name": None,
                "scene_code": "DEFAULT",
                "scene_name": "默认场景",
                "model_name": None,
                "create_time": datetime(2026, 5, 11, 10, 0, 0),
            },
            {
                "id": 2,
                "session_id": "sess-123",
                "seq": 2,
                "phone": "17327756086",
                "user_input": "南京天气",
                "ai_response": "晴天",
                "intent_code": "QUERY_WEATHER",
                "intent_name": "天气查询",
                "scene_code": "DEFAULT",
                "scene_name": "默认场景",
                "model_name": "qwen-turbo",
                "create_time": datetime(2026, 5, 11, 10, 1, 0),
            },
        ]
        repo = ChatLogRepository()
        repo.client.fetch_all_async = AsyncMock(return_value=rows)

        result = await repo.list_by_session_id("sess-123")

        assert len(result) == 2
        assert result[0]["user_input"] == "你好"
        assert result[1]["user_input"] == "南京天气"

    @pytest.mark.asyncio
    async def test_empty_when_no_messages(self) -> None:
        repo = ChatLogRepository()
        repo.client.fetch_all_async = AsyncMock(return_value=[])

        result = await repo.list_by_session_id("sess-nonexist")

        assert result == []

    @pytest.mark.asyncio
    async def test_respects_limit_and_offset(self) -> None:
        repo = ChatLogRepository()
        repo.client.fetch_all_async = AsyncMock(return_value=[])

        await repo.list_by_session_id("sess-page", limit=10, offset=20)

        call_args = repo.client.fetch_all_async.call_args
        params = call_args[0][1]
        assert params[0] == "sess-page"
        assert params[1] == 10
        assert params[2] == 20


class TestChatLogRepositoryListByPhone:
    """list_by_phone 测试"""

    @pytest.mark.asyncio
    async def test_returns_sessions_grouped_by_session_id(self) -> None:
        rows = [
            {
                "session_id": "sess-001",
                "scene_code": "DEFAULT",
                "scene_name": "默认场景",
                "last_time": datetime(2026, 5, 11, 12, 0, 0),
                "msg_count": 5,
            },
            {
                "session_id": "sess-002",
                "scene_code": "QUERY_SHIP",
                "scene_name": "船舶查询",
                "last_time": datetime(2026, 5, 10, 10, 0, 0),
                "msg_count": 3,
            },
        ]
        repo = ChatLogRepository()
        repo.client.fetch_all_async = AsyncMock(return_value=rows)

        result = await repo.list_by_phone("17327756086")

        assert len(result) == 2
        assert result[0]["session_id"] == "sess-001"
        assert result[0]["msg_count"] == 5

    @pytest.mark.asyncio
    async def test_empty_when_no_sessions(self) -> None:
        repo = ChatLogRepository()
        repo.client.fetch_all_async = AsyncMock(return_value=[])

        result = await repo.list_by_phone("13900000000")

        assert result == []

    @pytest.mark.asyncio
    async def test_respects_limit_and_offset(self) -> None:
        repo = ChatLogRepository()
        repo.client.fetch_all_async = AsyncMock(return_value=[])

        await repo.list_by_phone("17327756086", limit=5, offset=10)

        call_args = repo.client.fetch_all_async.call_args
        params = call_args[0][1]
        assert params[0] == "17327756086"
        assert params[1] == 5
        assert params[2] == 10