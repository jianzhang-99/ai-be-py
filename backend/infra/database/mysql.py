from __future__ import annotations

"""MySQL 客户端封装"""

import asyncio
import threading
from datetime import datetime
from typing import Any

import pymysql
from pymysql.cursors import DictCursor

from backend.config import get_settings


class MySQLClient:
    """轻量 MySQL 客户端"""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._local = threading.local()
        self._lock = threading.Lock()

    def build_dsn(self) -> dict[str, Any]:
        """构建连接参数"""

        return {
            "host": self._settings.mysql_host,
            "port": self._settings.mysql_port,
            "user": self._settings.mysql_user,
            "password": self._settings.mysql_password,
            "database": self._settings.mysql_database,
            "charset": "utf8mb4",
            "connect_timeout": 15,
            "read_timeout": 30,
            "write_timeout": 30,
        }

    def get_connection(self) -> pymysql.connections.Connection:
        """获取线程内复用连接"""

        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = pymysql.connect(**self.build_dsn())
        return self._local.conn

    def close_connection(self) -> None:
        """关闭当前线程连接"""

        conn = getattr(self._local, "conn", None)
        if conn is None:
            return

        try:
            conn.close()
        except Exception:
            pass
        self._local.conn = None

    def fetch_one(self, sql: str, params: tuple[Any, ...]) -> dict[str, Any] | None:
        """执行单条查询"""

        conn = self.get_connection()
        with self._lock:
            try:
                with conn.cursor(DictCursor) as cursor:
                    cursor.execute(sql, params)
                    return cursor.fetchone()
            except pymysql.err.OperationalError:
                self._local.conn = None
                conn = self.get_connection()
                with conn.cursor(DictCursor) as cursor:
                    cursor.execute(sql, params)
                    return cursor.fetchone()

    def execute(self, sql: str, params: tuple[Any, ...]) -> None:
        """执行写操作"""

        conn = self.get_connection()
        with self._lock:
            try:
                with conn.cursor() as cursor:
                    cursor.execute(sql, params)
                    conn.commit()
            except pymysql.err.OperationalError:
                self._local.conn = None
                conn = self.get_connection()
                with conn.cursor() as cursor:
                    cursor.execute(sql, params)
                    conn.commit()

    async def fetch_one_async(self, sql: str, params: tuple[Any, ...]) -> dict[str, Any] | None:
        """异步执行单条查询"""

        return await asyncio.to_thread(self.fetch_one, sql, params)

    def fetch_all(self, sql: str, params: tuple[Any, ...]) -> list[dict]:
        """执行多条查询"""

        conn = self.get_connection()
        with self._lock:
            try:
                with conn.cursor(DictCursor) as cursor:
                    cursor.execute(sql, params)
                    return cursor.fetchall()
            except pymysql.err.OperationalError:
                self._local.conn = None
                conn = self.get_connection()
                with conn.cursor(DictCursor) as cursor:
                    cursor.execute(sql, params)
                    return cursor.fetchall()

    async def fetch_all_async(self, sql: str, params: tuple[Any, ...]) -> list[dict]:
        """异步执行多条查询"""

        return await asyncio.to_thread(self.fetch_all, sql, params)

    async def execute_async(self, sql: str, params: tuple[Any, ...]) -> None:
        """异步执行写操作"""

        await asyncio.to_thread(self.execute, sql, params)

    async def update_last_login_async(self, user_id: int) -> None:
        """更新用户最近登录时间"""

        now = datetime.now()
        await self.execute_async(
            """
            UPDATE sys_user
               SET last_login_time = %s, update_time = %s
             WHERE id = %s
            """,
            (now, now, user_id),
        )

    def check_connectivity(self) -> tuple[bool, str]:
        """检查数据库连通性"""

        try:
            conn = pymysql.connect(**self.build_dsn())
            conn.close()
            return True, ""
        except Exception as exc:
            return False, str(exc)


_mysql_client = MySQLClient()


def get_mysql_client() -> MySQLClient:
    """返回全局 MySQL 客户端"""

    return _mysql_client
