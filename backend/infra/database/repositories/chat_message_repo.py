from __future__ import annotations

"""chat_message 仓储 - 消息持久化"""

from datetime import datetime

from backend.infra.database.repositories.base import BaseRepository


class ChatMessageRepository(BaseRepository):
    """chat_message 消息读写仓储"""

    async def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        intent: str | None = None,
        tool_name: str | None = None,
        tool_result: str | None = None,
        latency_ms: int | None = None,
    ) -> None:
        """保存单条消息到数据库"""

        now = datetime.now()
        await self.client.execute_async(
            """
            INSERT INTO chat_message (session_id, role, content, intent, tool_name, tool_result, latency_ms, create_time)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (session_id, role, content, intent, tool_name, tool_result, latency_ms, now),
        )

    async def list_session_messages(
        self,
        session_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        """查询会话的消息历史（按时间正序）"""

        rows = await self.client.fetch_all_async(
            """
            SELECT id, session_id, role, content, intent, tool_name, tool_result, latency_ms, create_time
              FROM chat_message
             WHERE session_id = %s
             ORDER BY create_time ASC
             LIMIT %s OFFSET %s
            """,
            (session_id, limit, offset),
        )
        return rows