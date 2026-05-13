"""AI 聊天日志管理端 API 路由。"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query, Request

from backend.auth.constants import USER_ID
from backend.auth.schemas import ResultResponse
from backend.infra.database.repositories.chat_log_repo import ChatLogRepository

router = APIRouter(prefix="/admin/ai/chatLog", tags=["admin-chatlog"])


def get_chat_log_repo() -> ChatLogRepository:
    """返回聊天日志仓储实例。"""
    return ChatLogRepository()


@router.get("/page", response_model=ResultResponse)
async def page_chat_log(
    request: Request,
    current: int = Query(default=1, ge=1, description="当前页"),
    size: int = Query(default=10, ge=1, le=100, description="每页条数"),
    session_id: Optional[str] = Query(default=None, description="会话ID"),
    phone: Optional[str] = Query(default=None, description="手机号"),
    scene_code: Optional[str] = Query(default=None, description="场景编码"),
    intent_code: Optional[str] = Query(default=None, description="意图编码"),
    user_input: Optional[str] = Query(default=None, description="用户输入关键词"),
    start_time: Optional[str] = Query(default=None, description="开始时间"),
    end_time: Optional[str] = Query(default=None, description="结束时间"),
    include_internal_users: bool = Query(default=False, description="是否包含内部用户"),
    repo: ChatLogRepository = Depends(get_chat_log_repo),
) -> ResultResponse:
    """分页查询聊天日志（管理员接口，需要登录认证）。"""

    user_id = getattr(request.state, USER_ID, None)
    if user_id is None:
        return ResultResponse.failure(2001, "用户未登录")

    rows, total = await repo.page_with_user_phone(
        current=current,
        size=size,
        session_id=session_id,
        phone=phone,
        scene_code=scene_code,
        intent_code=intent_code,
        user_input=user_input,
        start_time=start_time,
        end_time=end_time,
        include_internal_users=include_internal_users,
    )

    records = [
        {
            "id": row.get("id"),
            "sessionId": row.get("session_id"),
            "seq": row.get("seq"),
            "phone": row.get("phone"),
            "userName": row.get("user_name"),
            "sceneCode": row.get("scene_code"),
            "sceneName": row.get("scene_name"),
            "intentCode": row.get("intent_code"),
            "intentName": row.get("intent_name"),
            "intentDesc": row.get("intent_desc"),
            "userInput": row.get("user_input"),
            "aiResponse": row.get("ai_response"),
            "modelName": row.get("model_name"),
            "modelVersion": row.get("model_version"),
            "inputTokens": row.get("input_tokens"),
            "outputTokens": row.get("output_tokens"),
            "appSource": row.get("app_source"),
            "createTime": str(row.get("create_time")) if row.get("create_time") else None,
        }
        for row in rows
    ]

    return ResultResponse.success(
        data={
            "records": records,
            "total": total,
            "current": current,
            "size": size,
        }
    )