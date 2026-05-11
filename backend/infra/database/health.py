from __future__ import annotations

"""数据库健康检查入口"""

from backend.infra.database.mysql import get_mysql_client


def check_mysql_connectivity() -> tuple[bool, str]:
    """检查 MySQL 可连接性"""

    return get_mysql_client().check_connectivity()
