from __future__ import annotations

"""OSS 兼容服务。"""

import mimetypes
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from backend.config import get_settings
from backend.infra.database.repositories.oss_repository import OssRecord, OssRepository


class OssService:
    """提供与 Java `/oss` 接口兼容的最小能力。"""

    def __init__(self, repository: OssRepository | None = None) -> None:
        self.settings = get_settings()
        self.repository = repository or OssRepository()
        self.storage_root = Path(self.settings.local_upload_dir).expanduser()
        self.storage_root.mkdir(parents=True, exist_ok=True)

    async def upload(self, file: UploadFile, create_by: int | None = None) -> OssRecord:
        content = await file.read()
        if not content:
            raise ValueError("上传文件不能为空")

        suffix = self._normalize_suffix(file.filename or "")
        object_key = self._build_object_key(suffix)
        file_path = self.storage_root / object_key
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(content)

        return await self.repository.create(
            file_name=object_key,
            original_name=file.filename or object_key,
            file_suffix=suffix,
            url=f"/oss/file/{object_key}",
            service="local",
            create_by=create_by,
        )

    async def get_by_id(self, oss_id: int) -> OssRecord | None:
        return await self.repository.find_by_id(oss_id)

    def resolve_file_path(self, record: OssRecord) -> Path:
        path = (self.storage_root / record.file_name).resolve()
        root = self.storage_root.resolve()
        if root not in path.parents and path != root:
            raise ValueError("非法文件路径")
        return path

    def detect_media_type(self, filename: str) -> str:
        media_type, _ = mimetypes.guess_type(filename)
        return media_type or "application/octet-stream"

    def _build_object_key(self, suffix: str) -> str:
        from datetime import datetime

        date_path = datetime.now().strftime("%Y/%m/%d")
        return f"{date_path}/{uuid4().hex}{suffix}"

    def _normalize_suffix(self, filename: str) -> str:
        suffix = Path(filename).suffix.strip()
        return suffix if suffix else ".bin"
