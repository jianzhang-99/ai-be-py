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

    async def page_with_user_phone(
        self,
        current: int = 1,
        size: int = 10,
        session_id: str | None = None,
        phone: str | None = None,
        scene_code: str | None = None,
        intent_code: str | None = None,
        user_input: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        include_internal_users: bool = False,
    ) -> tuple[list[dict], int]:
        """分页查询聊天日志（JOIN sys_user 获取用户姓名），返回 (记录列表, 总数)。"""

        # 基础 JOIN 和 WHERE
        base_sql = """
            FROM ai_chat_log l
            JOIN sys_user u ON l.user_id = u.id AND u.is_delete = 0
            WHERE l.is_delete = 0
        """
        params: list[object] = []

        if session_id:
            base_sql += " AND l.session_id = %s"
            params.append(session_id)

        if phone:
            base_sql += " AND l.phone = %s"
            params.append(phone)

        if scene_code:
            base_sql += " AND l.scene_code = %s"
            params.append(scene_code)

        if intent_code:
            base_sql += " AND l.intent_code = %s"
            params.append(intent_code)

        if user_input:
            base_sql += " AND l.user_input LIKE %s"
            params.append(f"%{user_input}%")

        if start_time:
            base_sql += " AND l.create_time >= %s"
            params.append(start_time)

        if end_time:
            base_sql += " AND l.create_time <= %s"
            params.append(end_time)

        if not include_internal_users:
            # 排除内部用户：手机号以 "ydd" 开头视为内部测试账号
            base_sql += " AND l.phone NOT LIKE %s"
            params.append('ydd%')

        # 查询总数
        count_sql = f"SELECT COUNT(*) AS total {base_sql}"
        count_row = await self.client.fetch_one_async(count_sql, tuple(params))
        total = int(count_row["total"]) if count_row and count_row.get("total") is not None else 0

        # 查询分页记录
        offset = (current - 1) * size
        params.extend([size, offset])
        select_sql = f"""
            SELECT l.id, l.session_id, l.seq, l.phone,
                   COALESCE(u.nick_name, '') AS user_name,
                   l.scene_code, l.scene_name,
                   l.intent_code, l.intent_name, l.intent_desc,
                   l.user_input, l.ai_response,
                   l.model_name, l.model_version,
                   l.input_tokens, l.output_tokens,
                   l.app_source,
                   l.create_time
            {base_sql}
            ORDER BY l.create_time DESC
            LIMIT %s OFFSET %s
        """
        rows = await self.client.fetch_all_async(select_sql, tuple(params))
        return rows, total
