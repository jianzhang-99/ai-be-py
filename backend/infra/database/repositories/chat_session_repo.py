from __future__ import annotations

"""chat_session 仓储 - 会话管理"""

from datetime import datetime

from backend.infra.database.repositories.base import BaseRepository


class ChatSessionRepository(BaseRepository):
    """chat_session 会话读写仓储"""

    async def create_session(
        self,
        user_id: int,
        session_id: str,
        scene: str = "DEFAULT",
    ) -> None:
        """创建新会话"""

        now = datetime.now()
        await self.client.execute_async(
            """
            INSERT INTO chat_session (session_id, user_id, scene, status, create_time, update_time)
            VALUES (%s, %s, %s, 1, %s, %s)
            """,
            (session_id, user_id, scene, now, now),
        )

    async def get_session(self, session_id: str) -> dict | None:
        """按 session_id 查询会话"""

        return await self.client.fetch_one_async(
            """
            SELECT id, session_id, user_id, scene, status, create_time, update_time
              FROM chat_session
             WHERE session_id = %s
            """,
            (session_id,),
        )

    async def list_user_sessions(
        self,
        user_id: int,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict]:
        """查询用户会话列表（按创建时间倒序）"""

        rows = await self.client.fetch_all_async(
            """
            SELECT id, session_id, user_id, scene, status, create_time, update_time
              FROM chat_session
             WHERE user_id = %s
             ORDER BY create_time DESC
             LIMIT %s OFFSET %s
            """,
            (user_id, limit, offset),
        )
        return rows

    async def update_session_status(self, session_id: str, status: int) -> None:
        """更新会话状态"""

        now = datetime.now()
        await self.client.execute_async(
            """
            UPDATE chat_session
               SET status = %s, update_time = %s
             WHERE session_id = %s
            """,
            (status, now, session_id),
        )