from __future__ import annotations

"""数据库仓储统一出口"""

from backend.infra.database.repositories.base import BaseRepository
from backend.infra.database.repositories.chat_log_repo import ChatLogRepository
from backend.infra.database.repositories.sys_user_repository import SysUserRepository

__all__ = [
    "BaseRepository",
    "ChatLogRepository",
    "SysUserRepository",
]
