"""AI 提示词仓储 - 从数据库读取提示词，与 Java 版 ai_prompt 表兼容。"""

from __future__ import annotations

from backend.infra.database.repositories.base import BaseRepository


class PromptRepository(BaseRepository):
    """ai_prompt 表读写仓储。"""

    async def get_enabled_by_code(self, code: str) -> dict | None:
        """根据编码查询启用状态的提示词。"""
        row = await self.client.fetch_one_async(
            """
            SELECT id, code, name, version,
                   system_prompt, user_prompt,
                   status, remark,
                   create_time, update_time
              FROM ai_prompt
             WHERE code = %s
               AND (is_delete = 0 OR is_delete IS NULL)
               AND status = 1
            """,
            (code,),
        )
        return row

    async def list_all_enabled(self) -> list[dict]:
        """查询所有启用状态的提示词。"""
        rows = await self.client.fetch_all_async(
            """
            SELECT id, code, name, version,
                   system_prompt, user_prompt,
                   status, remark
              FROM ai_prompt
             WHERE (is_delete = 0 OR is_delete IS NULL)
               AND status = 1
            """,
            (),
        )
        return rows