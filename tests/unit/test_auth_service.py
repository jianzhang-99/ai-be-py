from __future__ import annotations

"""Unit tests for the async authentication service using stub repository."""

import pytest
from unittest.mock import AsyncMock

from backend.auth.schemas import AuthUser, LoginRequest
from backend.auth.service import AuthError, AuthService, STATUS_NORMAL, STATUS_DISABLE


class StubSysUserRepository:
    """内存 stub，不走真实数据库。"""

    def __init__(self, users: dict[str, AuthUser]) -> None:
        self._users = users
        self._id_index: dict[int, AuthUser] = {u.id: u for u in users.values()}
        self._update_last_login_calls: list[int] = []

    async def find_by_phone(self, phone: str) -> AuthUser | None:
        return self._users.get(phone)

    async def find_by_id(self, user_id: int) -> AuthUser | None:
        return self._id_index.get(user_id)

    async def update_last_login(self, user_id: int, login_ip: str = "") -> None:
        self._update_last_login_calls.append(user_id)


@pytest.fixture
def service() -> AuthService:
    return AuthService()


@pytest.fixture
def alice() -> AuthUser:
    import bcrypt
    return AuthUser(
        id=1,
        phone="13800138000",
        password_hash=bcrypt.hashpw("123456".encode(), bcrypt.gensalt()).decode(),
        nick_name="测试用户",
        status=STATUS_NORMAL,
        is_delete=False,
        last_login_time=None,
    )


@pytest.fixture
def disabled_user() -> AuthUser:
    import bcrypt
    return AuthUser(
        id=99,
        phone="13800138001",
        password_hash=bcrypt.hashpw("123456".encode(), bcrypt.gensalt()).decode(),
        nick_name="已禁用用户",
        status=STATUS_DISABLE,
        is_delete=False,
        last_login_time=None,
    )


@pytest.fixture
def alice_repo(alice: AuthUser) -> StubSysUserRepository:
    return StubSysUserRepository({alice.phone: alice})


@pytest.fixture
def mixed_repo(alice: AuthUser, disabled_user: AuthUser) -> StubSysUserRepository:
    return StubSysUserRepository({alice.phone: alice, disabled_user.phone: disabled_user})


# ---- login -------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_login_success(
    service: AuthService, alice: AuthUser, alice_repo: StubSysUserRepository
) -> None:
    """正确手机号+密码 -> 返回 token 和用户信息。"""
    result = await service.login(LoginRequest(phone="13800138000", password="123456"), alice_repo)
    assert result.userId == 1
    assert result.phone == "13800138000"
    assert result.token


@pytest.mark.asyncio
async def test_login_bad_phone_format(service: AuthService, alice_repo: StubSysUserRepository) -> None:
    """格式错误的手机号立即拒绝，不查库。"""
    with pytest.raises(AuthError) as error:
        await service.login(LoginRequest(phone="admin", password="123456"), alice_repo)
    assert error.value.code == 1001


@pytest.mark.asyncio
async def test_login_user_not_found(service: AuthService, alice_repo: StubSysUserRepository) -> None:
    """不存在的用户 -> 1002。"""
    with pytest.raises(AuthError) as error:
        await service.login(LoginRequest(phone="13900000000", password="123456"), alice_repo)
    assert error.value.code == 1002


@pytest.mark.asyncio
async def test_login_wrong_password(service: AuthService, alice: AuthUser, alice_repo: StubSysUserRepository) -> None:
    """密码错误 -> 1002。"""
    with pytest.raises(AuthError) as error:
        await service.login(LoginRequest(phone="13800138000", password="wrong"), alice_repo)
    assert error.value.code == 1002


@pytest.mark.asyncio
async def test_login_updates_last_login_time(
    service: AuthService, alice: AuthUser, alice_repo: StubSysUserRepository
) -> None:
    """登录成功后 update_last_login 被调用。"""
    await service.login(LoginRequest(phone="13800138000", password="123456"), alice_repo)
    assert alice_repo._update_last_login_calls == [1]


@pytest.mark.asyncio
async def test_login_disabled_user(
    service: AuthService, disabled_user: AuthUser, mixed_repo: StubSysUserRepository
) -> None:
    """已禁用的用户 -> 1002。"""
    with pytest.raises(AuthError) as error:
        await service.login(LoginRequest(phone="13800138001", password="123456"), mixed_repo)
    assert error.value.code == 1002


@pytest.mark.asyncio
async def test_login_deleted_user(
    service: AuthService, alice: AuthUser, alice_repo: StubSysUserRepository
) -> None:
    """已删除用户 -> 1002。"""
    alice.is_delete = True
    with pytest.raises(AuthError) as error:
        await service.login(LoginRequest(phone="13800138000", password="123456"), alice_repo)
    assert error.value.code == 1002
    alice.is_delete = False


# ---- logout ------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_logout_revokes_token(service: AuthService, alice_repo: StubSysUserRepository) -> None:
    """登出后 token 失效。"""
    result = await service.login(LoginRequest(phone="13800138000", password="123456"), alice_repo)
    await service.logout(result.token)
    with pytest.raises(AuthError) as error:
        await service.get_current_user(result.token, alice_repo)
    assert error.value.code == 401


# ---- get_current_user --------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_current_user_ok(
    service: AuthService, alice_repo: StubSysUserRepository
) -> None:
    """有效 token -> 返回用户信息。"""
    login_result = await service.login(LoginRequest(phone="13800138000", password="123456"), alice_repo)
    user = await service.get_current_user(login_result.token, alice_repo)
    assert user.userId == 1
    assert user.phone == "13800138000"


@pytest.mark.asyncio
async def test_get_current_user_fake_token(service: AuthService, alice_repo: StubSysUserRepository) -> None:
    """假 token -> 401。"""
    with pytest.raises(AuthError) as error:
        await service.get_current_user("fake-token", alice_repo)
    assert error.value.code == 401


@pytest.mark.asyncio
async def test_get_current_user_deleted_after_login(
    service: AuthService, alice: AuthUser, alice_repo: StubSysUserRepository
) -> None:
    """登录后用户被删除 -> 401 并清除 session。"""
    login_result = await service.login(LoginRequest(phone="13800138000", password="123456"), alice_repo)
    alice.is_delete = True
    with pytest.raises(AuthError) as error:
        await service.get_current_user(login_result.token, alice_repo)
    assert error.value.code == 401
    # 下次再拿应该还是 401（session 已清除）
    with pytest.raises(AuthError) as error2:
        await service.get_current_user(login_result.token, alice_repo)
    assert error2.value.code == 401


# ---- authenticate_token ------------------------------------------------------------

@pytest.mark.asyncio
async def test_authenticate_token_alias(
    service: AuthService, alice_repo: StubSysUserRepository
) -> None:
    """authenticate_token 是 get_current_user 的别名。"""
    login_result = await service.login(LoginRequest(phone="13800138000", password="123456"), alice_repo)
    user = await service.authenticate_token(login_result.token, alice_repo)
    assert user.userId == 1