"""ai_user_memory_profile 仓储 - 用户画像记忆读取"""

from __future__ import annotations

from backend.infra.database.repositories.base import BaseRepository


class UserMemoryProfileRepository(BaseRepository):
    """ai_user_memory_profile 表读取仓储。"""

    async def list_active(
        self,
        user_id: int | None = None,
        phone: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """查询激活中的用户画像。"""
        if user_id is None and not phone:
            return []

        where_parts = [
            "(is_delete = 0 OR is_delete IS NULL)",
            "(is_active = 1 OR is_active IS NULL)",
        ]
        params: list[object] = []

        if user_id is not None and phone:
            where_parts.append("(user_id = %s OR phone = %s)")
            params.extend([user_id, phone])
        elif user_id is not None:
            where_parts.append("user_id = %s")
            params.append(user_id)
        else:
            where_parts.append("phone = %s")
            params.append(phone)

        params.append(limit)
        rows = await self.client.fetch_all_async(
            f"""
            SELECT id, user_id, phone, memory_content, memory_category,
                   importance, context_snippet, create_time
              FROM ai_user_memory_profile
             WHERE {" AND ".join(where_parts)}
             ORDER BY importance DESC, create_time DESC
             LIMIT %s
            """,
            tuple(params),
        )
        return rows
