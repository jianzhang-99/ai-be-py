from __future__ import annotations

"""兼容旧导入路径的数据库转发层"""

from datetime import datetime

from backend.infra.database.health import check_mysql_connectivity as check_db_connectivity
from backend.infra.database.mysql import get_mysql_client


def sync_find_by_phone(phone: str) -> dict | None:
    """兼容旧同步手机号查询"""

    return get_mysql_client().fetch_one(
        """
        SELECT id, phone, password, nick_name, status, is_delete, last_login_time
          FROM sys_user
         WHERE phone = %s AND is_delete = 0
        """,
        (phone,),
    )


def sync_find_by_id(user_id: int) -> dict | None:
    """兼容旧同步用户查询"""

    return get_mysql_client().fetch_one(
        """
        SELECT id, phone, password, nick_name, status, is_delete, last_login_time
          FROM sys_user
         WHERE id = %s AND is_delete = 0
        """,
        (user_id,),
    )


def sync_update_last_login(user_id: int) -> None:
    """兼容旧同步登录时间更新"""

    now = datetime.now()
    get_mysql_client().execute(
        """
        UPDATE sys_user
           SET last_login_time = %s, update_time = %s
         WHERE id = %s
        """,
        (now, now, user_id),
    )


async def async_find_by_phone(phone: str) -> dict | None:
    """兼容旧异步手机号查询"""

    return await get_mysql_client().fetch_one_async(
        """
        SELECT id, phone, password, nick_name, status, is_delete, last_login_time
          FROM sys_user
         WHERE phone = %s AND is_delete = 0
        """,
        (phone,),
    )


async def async_find_by_id(user_id: int) -> dict | None:
    """兼容旧异步用户查询"""

    return await get_mysql_client().fetch_one_async(
        """
        SELECT id, phone, password, nick_name, status, is_delete, last_login_time
          FROM sys_user
         WHERE id = %s AND is_delete = 0
        """,
        (user_id,),
    )


async def async_update_last_login(user_id: int) -> None:
    """兼容旧异步登录时间更新"""

    await get_mysql_client().update_last_login_async(user_id)
