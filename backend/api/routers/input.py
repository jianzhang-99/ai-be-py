"""AI-BE 输入兼容接口。"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, File, Form, Request, UploadFile

from backend.auth.schemas import ResultResponse
from backend.services.input_service import InputService, InputServiceError

router = APIRouter(tags=["ai-input-compatible"])


async def _extract_raw_text(request: Request, raw_text_form: Optional[str]) -> str:
    """兼容 JSON 和表单两种订单参数解析请求。"""

    if raw_text_form:
        return raw_text_form

    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
        return str(body.get("rawText") or body.get("text") or "")

    if "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        form = await request.form()
        return str(form.get("rawText") or form.get("text") or "")

    return ""


@router.post("/ai/input/image", response_model=ResultResponse)
async def image(file: UploadFile = File(...)) -> ResultResponse:
    """兼容图片 OCR 接口。"""

    content = await file.read()
    if not content:
        return ResultResponse.failure(1001, "上传文件不能为空")

    try:
        text = await InputService().recognize_image(content, file.filename or "image", file.content_type or "image/jpeg")
    except InputServiceError as exc:
        return ResultResponse.failure(1002, str(exc))
    return ResultResponse.success(data=text)


@router.post("/ai/input/file", response_model=ResultResponse)
async def file_understanding(file: UploadFile = File(...)) -> ResultResponse:
    """兼容附件理解接口。"""

    content = await file.read()
    if not content:
        return ResultResponse.failure(1001, "上传文件不能为空")
    return ResultResponse.success(data="附件处理功能正在开发中，敬请期待！")


@router.post("/ai/input/speech", response_model=ResultResponse)
async def speech(file: UploadFile = File(...)) -> ResultResponse:
    """兼容语音识别接口。"""

    content = await file.read()
    if not content:
        return ResultResponse.failure(1001, "上传文件不能为空")

    try:
        text = await InputService().recognize_speech(
            content,
            file.filename or "audio.bin",
            file.content_type or "application/octet-stream",
        )
    except InputServiceError as exc:
        return ResultResponse.failure(1002, str(exc))
    return ResultResponse.success(data=text)


@router.post("/ai/input/shippingRoute", response_model=ResultResponse)
async def shipping_route(file: UploadFile = File(...)) -> ResultResponse:
    """兼容语音航线解析接口。"""

    content = await file.read()
    if not content:
        return ResultResponse.failure(1001, "上传文件不能为空")

    try:
        data = await InputService().shipping_route_by_audio(
            content,
            file.filename or "audio.bin",
            file.content_type or "application/octet-stream",
        )
    except InputServiceError as exc:
        return ResultResponse.failure(1002, str(exc))
    return ResultResponse.success(data=data)


@router.post("/ai/input/recognizeByOssId", response_model=ResultResponse)
async def recognize_by_oss_id(ossId: int) -> ResultResponse:
    """兼容根据 ossId 识别文本的接口。"""

    try:
        text = await InputService().recognize_by_oss_id(ossId)
    except InputServiceError as exc:
        return ResultResponse.failure(1002, str(exc))
    return ResultResponse.success(data=text)


@router.post("/ai/input/analysisOrderParams", response_model=ResultResponse)
async def analysis_order_params(
    request: Request,
    rawText: Optional[str] = Form(default=None),
) -> ResultResponse:
    """兼容运单参数解析接口。"""

    raw_text = (await _extract_raw_text(request, rawText)).strip()
    if not raw_text:
        return ResultResponse.failure(1001, "rawText 参数不能为空")

    try:
        data = await InputService().analysis_order_params(raw_text)
    except InputServiceError as exc:
        return ResultResponse.failure(1002, str(exc))
    return ResultResponse.success(data=data)
