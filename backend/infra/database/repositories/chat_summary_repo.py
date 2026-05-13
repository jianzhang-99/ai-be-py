"""ai_chat_summary 仓储 - 会话摘要读取"""

from __future__ import annotations

from backend.infra.database.repositories.base import BaseRepository


class ChatSummaryRepository(BaseRepository):
    """ai_chat_summary 表读取仓储。"""

    async def list_by_session_id(
        self,
        session_id: str,
        limit: int = 5,
    ) -> list[dict]:
        """按会话查询最近摘要。"""
        rows = await self.client.fetch_all_async(
            """
            SELECT id, session_id, user_id, phone,
                   summary_content, original_count, summary_seq, create_time
              FROM ai_chat_summary
             WHERE session_id = %s
               AND (is_delete = 0 OR is_delete IS NULL)
             ORDER BY summary_seq DESC, create_time DESC
             LIMIT %s
            """,
            (session_id, limit),
        )
        return rows
