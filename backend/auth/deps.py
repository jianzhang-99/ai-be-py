from __future__ import annotations

"""Dependency helpers for the authentication module."""

from functools import lru_cache

from backend.auth.service import AuthService
from backend.infra.database.repositories.sys_user_repository import SysUserRepository


@lru_cache(maxsize=1)
def get_auth_service() -> AuthService:
    """Reuse one auth service instance so issued tokens stay valid in-process."""

    return AuthService()


def get_sys_user_repository() -> SysUserRepository:
    """返回 sys_user 仓储实例"""

    return SysUserRepository()
