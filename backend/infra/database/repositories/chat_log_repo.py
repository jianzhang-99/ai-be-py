"""ai_chat_log 仓储 - 消息日志持久化（兼容 Java 版本）"""

from __future__ import annotations

from datetime import datetime

from backend.infra.database.repositories.base import BaseRepository


class ChatLogRepository(BaseRepository):
    """ai_chat_log 消息读写仓储"""

    async def save_log(
        self,
        session_id: str,
        phone: str,
        user_id: int,
        user_input: str,
        ai_response: str,
        intent_code: str | None = None,
        intent_name: str | None = None,
        intent_desc: str | None = None,
        scene_code: str = "DEFAULT",
        scene_name: str = "默认场景",
        model_name: str | None = None,
        model_version: str | None = None,
        original_request: str | None = None,
        app_source: int = 1,
    ) -> int:
        """保存单条聊天日志到数据库，返回自增主键id"""
        now = datetime.now()

        # 先查询当前 session 下最大的 seq
        row = await self.client.fetch_one_async(
            "SELECT COALESCE(MAX(seq), 0) as max_seq FROM ai_chat_log WHERE session_id = %s",
            (session_id,),
        )
        next_seq = (row["max_seq"] + 1) if row else 1

        await self.client.execute_async(
            """
            INSERT INTO ai_chat_log (
                session_id, seq, phone, user_id, user_input, ai_response,
                intent_code, intent_name, intent_desc,
                scene_code, scene_name, model_name, model_version,
                original_request, app_source, create_time, is_delete
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 0)
            """,
            (
                session_id, next_seq, phone, user_id, user_input, ai_response,
                intent_code, intent_name, intent_desc,
                scene_code, scene_name, model_name, model_version,
                original_request, app_source, now,
            ),
        )
        return next_seq

    async def list_by_session_id(
        self,
        session_id: str,
        phone: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        """按会话ID查询聊天历史（按时间正序）"""
        where_sql = "WHERE session_id = %s AND is_delete = 0"
        params: list[object] = [session_id]
        if phone:
            where_sql += " AND phone = %s"
            params.append(phone)
        params.extend([limit, offset])
        rows = await self.client.fetch_all_async(
            f"""
            SELECT id, session_id, seq, phone, user_input, ai_response,
                   intent_code, intent_name, scene_code, scene_name,
                   model_name, original_request, user_id, create_time
              FROM ai_chat_log
             {where_sql}
             ORDER BY create_time ASC
             LIMIT %s OFFSET %s
            """,
            tuple(params),
        )
        return rows

    async def list_by_phone(
        self,
        phone: str,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict]:
        """按手机号查询聊天会话列表（按最后一条消息倒序）"""
        rows = await self.client.fetch_all_async(
            """
            SELECT t.session_id,
                   t.id AS last_id,
                   t.seq AS last_seq,
                   t.user_id,
                   t.scene_code,
                   t.scene_name,
                   t.user_input,
                   t.original_request,
                   t.ai_response,
                   COALESCE(t.update_time, t.create_time) AS last_time
              FROM ai_chat_log t
              JOIN (
                    SELECT session_id, MAX(id) AS max_id
                      FROM ai_chat_log
                     WHERE phone = %s AND is_delete = 0
                     GROUP BY session_id
                   ) latest ON latest.max_id = t.id
             ORDER BY last_time DESC
             LIMIT %s OFFSET %s
            """,
            (phone, limit, offset),
        )
        return rows

    async def count_sessions_by_phone(self, phone: str) -> int:
        """统计手机号对应的有效会话数。"""
        row = await self.client.fetch_one_async(
            """
            SELECT COUNT(DISTINCT session_id) AS total
              FROM ai_chat_log
             WHERE phone = %s AND is_delete = 0
            """,
            (phone,),
        )
        return int(row["total"]) if row and row.get("total") is not None else 0

    async def list_by_session_id_for_share(
        self,
        session_id: str,
        limit: int = 200,
        offset: int = 0,
    ) -> list[dict]:
        """按会话ID查询分享态历史。"""
        return await self.client.fetch_all_async(
            """
            SELECT id, seq, user_input, original_request, ai_response, create_time
              FROM ai_chat_log
             WHERE session_id = %s AND is_delete = 0
             ORDER BY create_time ASC
             LIMIT %s OFFSET %s
            """,
            (session_id, limit, offset),
        )

    async def soft_delete_by_session_id(self, session_id: str, phone: str) -> int:
        """按会话ID逻辑删除当前手机号下的消息。"""
        return await self.client.execute_async(
            """
            UPDATE ai_chat_log
               SET is_delete = 1
             WHERE session_id = %s AND phone = %s AND is_delete = 0
            """,
            (session_id, phone),
        )
