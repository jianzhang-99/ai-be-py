from __future__ import annotations

"""sys_oss 仓储。"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from backend.infra.database.repositories.base import BaseRepository


@dataclass
class OssRecord:
    """sys_oss 记录。"""

    id: int
    file_name: str
    original_name: str
    file_suffix: str
    url: str
    service: str
    create_time: Optional[datetime]
    create_by: Optional[int]
    update_time: Optional[datetime]
    update_by: Optional[int]
    is_delete: bool


class OssRepository(BaseRepository):
    """sys_oss 读写仓储。"""

    async def find_by_id(self, oss_id: int) -> OssRecord | None:
        row = await self.client.fetch_one_async(
            """
            SELECT id, file_name, original_name, file_suffix, url, service,
                   create_time, create_by, update_time, update_by, is_delete
              FROM sys_oss
             WHERE id = %s AND is_delete = 0
            """,
            (oss_id,),
        )
        return self._to_record(row)

    async def create(
        self,
        *,
        file_name: str,
        original_name: str,
        file_suffix: str,
        url: str,
        service: str,
        create_by: int | None = None,
    ) -> OssRecord:
        now = datetime.now()
        oss_id = await self.client.insert_async(
            """
            INSERT INTO sys_oss (
                file_name, original_name, file_suffix, url, service,
                create_time, create_by, update_time, update_by, is_delete
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 0)
            """,
            (
                file_name,
                original_name,
                file_suffix,
                url,
                service,
                now,
                create_by,
                now,
                create_by,
            ),
        )
        record = await self.find_by_id(oss_id)
        if record is None:
            raise RuntimeError("sys_oss 写入成功但读取失败")
        return record

    def _to_record(self, row: dict[str, object] | None) -> OssRecord | None:
        if row is None:
            return None

        return OssRecord(
            id=int(row["id"]),
            file_name=str(row["file_name"] or ""),
            original_name=str(row["original_name"] or ""),
            file_suffix=str(row["file_suffix"] or ""),
            url=str(row["url"] or ""),
            service=str(row["service"] or ""),
            create_time=row["create_time"],
            create_by=int(row["create_by"]) if row["create_by"] is not None else None,
            update_time=row["update_time"],
            update_by=int(row["update_by"]) if row["update_by"] is not None else None,
            is_delete=bool(row["is_delete"]),
        )
