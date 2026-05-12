"""Unit tests for SessionStore (Redis cache layer)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from backend.infra.cache.session_store import (
    TOKEN_PREFIX,
    TOKEN_TTL_DAYS,
    SessionStore,
    get_session_store,
)


class StubRedis:
    """内存 stub，模拟 Redis 行为。"""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    async def setex(self, key: str, ttl: int, value: str) -> None:
        self._store[key] = value

    async def get(self, key: str) -> str | None:
        return self._store.get(key)

    async def delete(self, key: str) -> int:
        if key in self._store:
            del self._store[key]
            return 1
        return 0

    async def exists(self, key: str) -> int:
        return 1 if key in self._store else 0

    async def ping(self) -> bool:
        return True

    async def aclose(self) -> None:
        pass


@pytest.fixture
def store() -> SessionStore:
    """返回 SessionStore 实例（注入 stub redis）。"""
    store_instance = SessionStore()
    store_instance._client = StubRedis()
    return store_instance


class TestSessionStoreToken:
    """Token 缓存测试。"""

    @pytest.mark.asyncio
    async def test_set_and_get_token(self, store: SessionStore) -> None:
        """正常缓存 token -> user_id，再查回来。"""
        await store.set_token("token-abc", user_id=123)
        user_id = await store.get_user_id_by_token("token-abc")
        assert user_id == 123

    @pytest.mark.asyncio
    async def test_get_nonexistent_token(self, store: SessionStore) -> None:
        """不存在的 token -> None。"""
        user_id = await store.get_user_id_by_token("no-such-token")
        assert user_id is None

    @pytest.mark.asyncio
    async def test_delete_token(self, store: SessionStore) -> None:
        """删除 token 后再查返回 None。"""
        await store.set_token("token-del", user_id=456)
        result = await store.delete_token("token-del")
        assert result is True
        assert await store.get_user_id_by_token("token-del") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_token(self, store: SessionStore) -> None:
        """删除不存在的 token 返回 False。"""
        result = await store.delete_token("no-such-token")
        assert result is False

    @pytest.mark.asyncio
    async def test_exists_token(self, store: SessionStore) -> None:
        """exists_token 正确判断。"""
        await store.set_token("token-exist", user_id=789)
        assert await store.exists_token("token-exist") is True
        assert await store.exists_token("no-exist") is False


class TestSessionStoreSessionData:
    """会话数据缓存测试。"""

    @pytest.mark.asyncio
    async def test_set_and_get_session_data(self, store: SessionStore) -> None:
        """正常缓存会话数据。"""
        session_data = {"scene": "DEFAULT", "user_id": 1}
        await store.set_session_data("sess-001", session_data, ttl_hours=1)
        result = await store.get_session_data("sess-001")
        assert result == session_data

    @pytest.mark.asyncio
    async def test_get_nonexistent_session_data(self, store: SessionStore) -> None:
        """不存在的会话返回 None。"""
        result = await store.get_session_data("no-such-session")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_session_data(self, store: SessionStore) -> None:
        """删除会话数据。"""
        await store.set_session_data("sess-del", {"key": "value"})
        result = await store.delete_session_data("sess-del")
        assert result is True
        assert await store.get_session_data("sess-del") is None


class TestSessionStoreHealth:
    """健康检查测试。"""

    @pytest.mark.asyncio
    async def test_health_check_ok(self, store: SessionStore) -> None:
        """Redis 正常时 health_check 返回 True。"""
        assert await store.health_check() is True


class TestSessionStoreSingleton:
    """单例模式测试。"""

    def test_get_session_store_returns_singleton(self) -> None:
        """多次调用返回同一实例。"""
        s1 = get_session_store()
        s2 = get_session_store()
        assert s1 is s2