from __future__ import annotations

"""OSS 兼容路由。"""

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from backend.auth.schemas import ResultResponse
from backend.services.oss_service import OssService

router = APIRouter(prefix="/oss", tags=["oss"])


def _record_to_payload(record: object) -> dict[str, object]:
    return {
        "id": getattr(record, "id"),
        "fileName": getattr(record, "file_name"),
        "originalName": getattr(record, "original_name"),
        "fileSuffix": getattr(record, "file_suffix"),
        "url": getattr(record, "url"),
        "service": getattr(record, "service"),
        "createTime": getattr(record, "create_time"),
        "createBy": getattr(record, "create_by"),
        "updateTime": getattr(record, "update_time"),
        "updateBy": getattr(record, "update_by"),
    }


@router.post("/upload", response_model=ResultResponse)
async def upload(request: Request, file: UploadFile = File(...)) -> ResultResponse:
    service = OssService()
    current_user = getattr(request.state, "current_user", None)
    try:
        record = await service.upload(file, create_by=getattr(current_user, "userId", None))
    except ValueError as exc:
        return ResultResponse.failure(1001, str(exc))
    return ResultResponse.success(data=_record_to_payload(record))


@router.get("/{oss_id}", response_model=ResultResponse)
async def get_by_id(oss_id: int) -> ResultResponse:
    service = OssService()
    record = await service.get_by_id(oss_id)
    if record is None:
        return ResultResponse.success()
    return ResultResponse.success(data=_record_to_payload(record))


@router.get("/file/{object_key:path}")
async def get_file(object_key: str):
    service = OssService()
    pseudo_record = type("OssFile", (), {"file_name": object_key})()
    try:
        path = service.resolve_file_path(pseudo_record)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")

    return FileResponse(
        path,
        media_type=service.detect_media_type(path.name),
        filename=path.name,
    )
