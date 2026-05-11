from __future__ import annotations

"""sys_user 仓储"""

from backend.auth.schemas import AuthUser
from backend.infra.database.repositories.base import BaseRepository


class SysUserRepository(BaseRepository):
    """sys_user 最小读写仓储"""

    async def find_by_phone(self, phone: str) -> AuthUser | None:
        """按手机号查询用户"""

        row = await self.client.fetch_one_async(
            """
            SELECT id, phone, password, nick_name, status, is_delete, last_login_time
              FROM sys_user
             WHERE phone = %s AND is_delete = 0
            """,
            (phone,),
        )
        return self._to_auth_user(row)

    async def find_by_id(self, user_id: int) -> AuthUser | None:
        """按用户 id 查询用户"""

        row = await self.client.fetch_one_async(
            """
            SELECT id, phone, password, nick_name, status, is_delete, last_login_time
              FROM sys_user
             WHERE id = %s AND is_delete = 0
            """,
            (user_id,),
        )
        return self._to_auth_user(row)

    async def update_last_login(self, user_id: int, login_ip: str = "") -> None:
        """更新最近登录时间"""

        await self.client.update_last_login_async(user_id)

    def _to_auth_user(self, row: dict[str, object] | None) -> AuthUser | None:
        """转换数据库行到认证用户对象"""

        if row is None:
            return None

        return AuthUser(
            id=int(row["id"]),
            phone=str(row["phone"] or ""),
            password_hash=str(row["password"] or ""),
            nick_name=str(row["nick_name"]) if row["nick_name"] is not None else None,
            status=int(row["status"] or 0),
            is_delete=bool(row["is_delete"]),
            last_login_time=row["last_login_time"],
        )
