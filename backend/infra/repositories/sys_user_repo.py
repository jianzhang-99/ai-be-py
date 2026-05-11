from __future__ import annotations

"""兼容旧导入路径的 sys_user 仓储"""

from backend.infra.database.repositories.sys_user_repository import SysUserRepository

__all__ = ["SysUserRepository"]
