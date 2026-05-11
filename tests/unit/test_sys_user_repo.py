from __future__ import annotations

"""Unit tests for SysUserRepository"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock

from backend.auth.schemas import AuthUser
from backend.infra.database.repositories.sys_user_repository import SysUserRepository

class TestSysUserRepositoryFindByPhone:
    """find_by_phone 测试"""

    @pytest.fixture
    def alice(self) -> AuthUser:
        return AuthUser(
            id=1,
            phone="15888888888",
            password_hash="$2b$12$hash",
            nick_name="梁嘉健",
            status=0,
            is_delete=False,
            last_login_time=datetime(2026, 4, 17, 8, 42, 5),
        )

    @pytest.mark.asyncio
    async def test_found_returns_auth_user(self, alice: AuthUser) -> None:
        row_data = {
            "id": 1,
            "phone": "15888888888",
            "password": "$2b$12$hash",
            "nick_name": "梁嘉健",
            "status": 0,
            "is_delete": 0,
            "last_login_time": datetime(2026, 4, 17, 8, 42, 5),
        }
        repo = SysUserRepository()
        repo.client.fetch_one_async = AsyncMock(return_value=row_data)
        result = await repo.find_by_phone("15888888888")

        assert result is not None
        assert result.id == 1
        assert result.phone == "15888888888"
        assert result.password_hash == "$2b$12$hash"
        assert result.nick_name == "梁嘉健"
        assert result.status == 0
        assert result.is_delete is False

    @pytest.mark.asyncio
    async def test_not_found_returns_none(self) -> None:
        repo = SysUserRepository()
        repo.client.fetch_one_async = AsyncMock(return_value=None)
        result = await repo.find_by_phone("13900000000")
        assert result is None

    @pytest.mark.asyncio
    async def test_calls_async_find_by_phone(self, alice: AuthUser) -> None:
        row_data = {
            "id": 1, "phone": "15888888888", "password": "h",
            "nick_name": "n", "status": 0, "is_delete": 0, "last_login_time": None,
        }
        repo = SysUserRepository()
        repo.client.fetch_one_async = AsyncMock(return_value=row_data)
        await repo.find_by_phone("15888888888")
        repo.client.fetch_one_async.assert_called_once()


class TestSysUserRepositoryFindById:
    """find_by_id 测试"""

    @pytest.fixture
    def bob(self) -> AuthUser:
        return AuthUser(
            id=42,
            phone="15900000000",
            password_hash="$2b$12$xyz",
            nick_name="其他用户",
            status=0,
            is_delete=False,
            last_login_time=None,
        )

    @pytest.mark.asyncio
    async def test_found_returns_auth_user(self, bob: AuthUser) -> None:
        row_data = {
            "id": 42, "phone": "15900000000", "password": "$2b$12$xyz",
            "nick_name": "其他用户", "status": 0, "is_delete": 0, "last_login_time": None,
        }
        repo = SysUserRepository()
        repo.client.fetch_one_async = AsyncMock(return_value=row_data)
        result = await repo.find_by_id(42)

        assert result is not None
        assert result.id == 42
        assert result.phone == "15900000000"

    @pytest.mark.asyncio
    async def test_not_found_returns_none(self) -> None:
        repo = SysUserRepository()
        repo.client.fetch_one_async = AsyncMock(return_value=None)
        result = await repo.find_by_id(9999)
        assert result is None

    @pytest.mark.asyncio
    async def test_calls_async_find_by_id(self) -> None:
        row_data = {
            "id": 5, "phone": "p", "password": "h",
            "nick_name": "n", "status": 0, "is_delete": 0, "last_login_time": None,
        }
        repo = SysUserRepository()
        repo.client.fetch_one_async = AsyncMock(return_value=row_data)
        await repo.find_by_id(5)
        repo.client.fetch_one_async.assert_called_once()


class TestSysUserRepositoryUpdateLastLogin:
    """update_last_login 测试"""

    @pytest.mark.asyncio
    async def test_calls_async_update_last_login(self) -> None:
        repo = SysUserRepository()
        repo.client.update_last_login_async = AsyncMock()
        await repo.update_last_login(user_id=1, login_ip="127.0.0.1")
        repo.client.update_last_login_async.assert_called_once_with(1)
