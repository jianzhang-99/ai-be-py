from __future__ import annotations

"""Unit tests for ChatSessionRepository"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, call

from backend.infra.database.repositories.chat_session_repo import ChatSessionRepository


class TestChatSessionRepositoryCreateSession:
    """create_session 测试"""

    @pytest.mark.asyncio
    async def test_calls_execute_async_with_correct_params(self) -> None:
        repo = ChatSessionRepository()
        repo.client.execute_async = AsyncMock()

        await repo.create_session(user_id=1, session_id="sess-abc-123", scene="DEFAULT")

        repo.client.execute_async.assert_called_once()
        call_args = repo.client.execute_async.call_args
        sql = call_args[0][0]
        params = call_args[0][1]
        assert "INSERT INTO chat_session" in sql
        assert params[0] == "sess-abc-123"
        assert params[1] == 1
        assert params[2] == "DEFAULT"

    @pytest.mark.asyncio
    async def test_create_session_custom_scene(self) -> None:
        repo = ChatSessionRepository()
        repo.client.execute_async = AsyncMock()

        await repo.create_session(user_id=5, session_id="sess-xyz", scene="QUERY_SHIP")

        call_args = repo.client.execute_async.call_args
        params = call_args[0][1]
        assert params[2] == "QUERY_SHIP"


class TestChatSessionRepositoryGetSession:
    """get_session 测试"""

    @pytest.mark.asyncio
    async def test_found_returns_session_dict(self) -> None:
        row_data = {
            "id": 10,
            "session_id": "sess-found",
            "user_id": 1,
            "scene": "DEFAULT",
            "status": 1,
            "create_time": datetime(2026, 5, 11, 10, 0, 0),
            "update_time": datetime(2026, 5, 11, 10, 0, 0),
        }
        repo = ChatSessionRepository()
        repo.client.fetch_one_async = AsyncMock(return_value=row_data)

        result = await repo.get_session("sess-found")

        assert result is not None
        assert result["session_id"] == "sess-found"
        assert result["user_id"] == 1
        assert result["status"] == 1

    @pytest.mark.asyncio
    async def test_not_found_returns_none(self) -> None:
        repo = ChatSessionRepository()
        repo.client.fetch_one_async = AsyncMock(return_value=None)

        result = await repo.get_session("sess-nonexist")

        assert result is None

    @pytest.mark.asyncio
    async def test_calls_fetch_one_async_with_session_id(self) -> None:
        repo = ChatSessionRepository()
        repo.client.fetch_one_async = AsyncMock(return_value=None)

        await repo.get_session("sess-target")

        repo.client.fetch_one_async.assert_called_once()
        call_args = repo.client.fetch_one_async.call_args
        assert "sess-target" in call_args[0][1]


class TestChatSessionRepositoryListUserSessions:
    """list_user_sessions 测试"""

    @pytest.mark.asyncio
    async def test_returns_list_of_sessions(self) -> None:
        rows = [
            {
                "id": 1,
                "session_id": "sess-1",
                "user_id": 1,
                "scene": "DEFAULT",
                "status": 1,
                "create_time": datetime(2026, 5, 11, 10, 0, 0),
                "update_time": datetime(2026, 5, 11, 10, 0, 0),
            },
            {
                "id": 2,
                "session_id": "sess-2",
                "user_id": 1,
                "scene": "QUERY_WEATHER",
                "status": 0,
                "create_time": datetime(2026, 5, 10, 10, 0, 0),
                "update_time": datetime(2026, 5, 10, 10, 0, 0),
            },
        ]
        repo = ChatSessionRepository()
        repo.client.fetch_all_async = AsyncMock(return_value=rows)

        result = await repo.list_user_sessions(user_id=1, limit=20, offset=0)

        assert len(result) == 2
        assert result[0]["session_id"] == "sess-1"
        assert result[1]["status"] == 0

    @pytest.mark.asyncio
    async def test_empty_list_when_no_sessions(self) -> None:
        repo = ChatSessionRepository()
        repo.client.fetch_all_async = AsyncMock(return_value=[])

        result = await repo.list_user_sessions(user_id=9999)

        assert result == []

    @pytest.mark.asyncio
    async def test_respects_limit_and_offset(self) -> None:
        repo = ChatSessionRepository()
        repo.client.fetch_all_async = AsyncMock(return_value=[])

        await repo.list_user_sessions(user_id=1, limit=10, offset=5)

        call_args = repo.client.fetch_all_async.call_args
        params = call_args[0][1]
        assert params[1] == 10
        assert params[2] == 5


class TestChatSessionRepositoryUpdateSessionStatus:
    """update_session_status 测试"""

    @pytest.mark.asyncio
    async def test_calls_execute_async_to_set_status(self) -> None:
        repo = ChatSessionRepository()
        repo.client.execute_async = AsyncMock()

        await repo.update_session_status("sess-abc", status=0)

        repo.client.execute_async.assert_called_once()
        call_args = repo.client.execute_async.call_args
        sql = call_args[0][0]
        params = call_args[0][1]
        assert "UPDATE chat_session" in sql
        assert params[0] == 0
        # params[1] 是 update_time (datetime)
        assert params[2] == "sess-abc"