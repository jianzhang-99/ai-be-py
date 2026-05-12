"""AI-BE 兼容层 API 路由。"""

from __future__ import annotations

import json
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from sse_starlette.sse import EventSourceResponse

from backend.api.deps import get_chat_service
from backend.api.schemas import ChatRequest
from backend.auth.constants import PHONE, USER_ID
from backend.auth.deps import get_auth_service
from backend.auth.schemas import ResultResponse
from backend.auth.service import AuthError, AuthService
from backend.infra.database.repositories.chat_log_repo import ChatLogRepository
from backend.infra.database.repositories.sys_user_repository import SysUserRepository
from backend.services.chat_service import ChatService

router = APIRouter(tags=["ai-be-compatible"])


def get_chat_log_repo() -> ChatLogRepository:
    """返回聊天日志仓储实例。"""

    return ChatLogRepository()


async def _resolve_current_user_context(
    request: Request,
    auth_service: AuthService,
) -> tuple[int | None, str | None]:
    """从 middleware、cookie 或 query token 恢复用户上下文。"""

    user_id = getattr(request.state, USER_ID, None)
    phone = getattr(request.state, PHONE, None)
    if user_id is not None:
        return int(user_id), phone

    token = request.query_params.get("token") or request.cookies.get("satoken")
    if not token:
        return None, None

    user_repo = SysUserRepository()
    try:
        current_user = await auth_service.authenticate_token(token, user_repo)
    except AuthError:
        return None, None
    return current_user.userId, current_user.phone


@router.post("/ai/chat")
async def ai_chat_compat(
    request: Request,
    chat_service: Annotated[ChatService, Depends(get_chat_service)],
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
):
    """兼容原有 `/ai/chat` 的 SSE 聊天接口。"""

    user_id, phone = await _resolve_current_user_context(request, auth_service)
    if user_id is None:
        return {"code": 401, "msg": "未登录或登录已过期，请重新登录", "data": None}

    body = await request.json()
    chat_request = ChatRequest(
        message=body.get("input", ""),
        session_id=body.get("sessionId"),
        user_id=str(user_id),
        scene=body.get("scene"),
        model=body.get("model"),
        app_source=body.get("appSource"),
        attachments=body.get("attachments") or [],
    )

    async def event_generator():
        async for payload in chat_service.chat_ai_be_stream(chat_request, phone=phone):
            yield {"data": json.dumps(payload, ensure_ascii=False)}

    return EventSourceResponse(event_generator())


@router.get("/ai/chat/history/page", response_model=ResultResponse)
async def get_chat_history_page(
    request: Request,
    current: int = 1,
    size: int = 10,
    repo: ChatLogRepository = Depends(get_chat_log_repo),
) -> ResultResponse:
    """兼容 `/ai/chat/history/page` 的会话分页接口。"""

    if current < 1:
        return ResultResponse.failure(1001, "current 参数无效")
    if size < 1 or size > 100:
        return ResultResponse.failure(1001, "size 参数需在 1-100 之间")

    phone = getattr(request.state, PHONE, None)
    if not phone:
        return ResultResponse.failure(2001, "用户未登录")

    offset = (current - 1) * size
    sessions = await repo.list_by_phone(phone, limit=size, offset=offset)
    total = await repo.count_sessions_by_phone(phone)

    records = [
        {
            "sessionId": row.get("session_id"),
            "lastId": row.get("last_id"),
            "lastSeq": row.get("last_seq"),
            "userId": row.get("user_id"),
            "sceneCode": row.get("scene_code"),
            "sceneName": row.get("scene_name"),
            "userInput": row.get("user_input"),
            "originalRequest": row.get("original_request"),
            "aiResponse": row.get("ai_response"),
            "lastTime": str(row.get("last_time")) if row.get("last_time") else None,
        }
        for row in sessions
    ]
    return ResultResponse.success(
        data={
            "records": records,
            "total": total,
            "current": current,
            "size": size,
        }
    )


@router.get("/ai/chat/history/listBySessionId/{session_id}", response_model=ResultResponse)
async def list_by_session_id(
    session_id: str,
    request: Request,
    repo: ChatLogRepository = Depends(get_chat_log_repo),
) -> ResultResponse:
    """兼容 `/ai/chat/history/listBySessionId/{sessionId}`。"""

    phone = getattr(request.state, PHONE, None)
    if not phone:
        return ResultResponse.failure(2001, "用户未登录")

    records = await repo.list_by_session_id(session_id, phone=phone, limit=200, offset=0)
    data = [
        {
            "id": row.get("id"),
            "sessionId": row.get("session_id"),
            "seq": row.get("seq"),
            "userId": row.get("user_id"),
            "userInput": row.get("user_input"),
            "originalRequest": row.get("original_request"),
            "aiResponse": row.get("ai_response"),
            "intentCode": row.get("intent_code"),
            "intentName": row.get("intent_name"),
            "sceneCode": row.get("scene_code"),
            "sceneName": row.get("scene_name"),
            "createTime": str(row.get("create_time")) if row.get("create_time") else None,
        }
        for row in records
    ]
    return ResultResponse.success(data=data)


@router.get("/ai/chat/history/share/listBySessionId/{session_id}", response_model=ResultResponse)
async def list_share_by_session_id(
    session_id: str,
    repo: ChatLogRepository = Depends(get_chat_log_repo),
) -> ResultResponse:
    """兼容公开分享态历史查询接口。"""

    records = await repo.list_by_session_id_for_share(session_id, limit=200, offset=0)
    data = [
        {
            "id": row.get("id"),
            "seq": row.get("seq"),
            "userInput": row.get("user_input"),
            "originalRequest": row.get("original_request"),
            "aiResponse": row.get("ai_response"),
            "createTime": str(row.get("create_time")) if row.get("create_time") else None,
        }
        for row in records
    ]
    return ResultResponse.success(data=data)


@router.delete("/ai/chat/history/deleteBySessionId/{session_id}", response_model=ResultResponse)
async def delete_by_session_id(
    session_id: str,
    request: Request,
    repo: ChatLogRepository = Depends(get_chat_log_repo),
) -> ResultResponse:
    """兼容 `/ai/chat/history/deleteBySessionId/{sessionId}`。"""

    phone = getattr(request.state, PHONE, None)
    if not phone:
        return ResultResponse.failure(2001, "用户未登录")

    await repo.soft_delete_by_session_id(session_id, phone)
    return ResultResponse.success()
