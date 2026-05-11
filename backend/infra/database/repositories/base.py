from __future__ import annotations

"""仓储基类"""

from backend.infra.database.mysql import MySQLClient, get_mysql_client


class BaseRepository:
    """数据库仓储基类"""

    def __init__(self, client: MySQLClient | None = None) -> None:
        self.client = client or get_mysql_client()
