from __future__ import annotations

"""会话管理 API 路由"""

from fastapi import APIRouter, Depends, Request

from backend.auth.constants import USER_ID
from backend.auth.schemas import ResultResponse
from backend.infra.database.repositories.chat_message_repo import ChatMessageRepository
from backend.infra.database.repositories.chat_session_repo import ChatSessionRepository

router = APIRouter(prefix="/api/session", tags=["session"])


def get_session_repo() -> ChatSessionRepository:
    """返回会话仓储实例"""

    return ChatSessionRepository()


def get_message_repo() -> ChatMessageRepository:
    """返回消息仓储实例"""

    return ChatMessageRepository()


@router.get("/list")
async def list_sessions(
    request: Request,
    limit: int = 20,
    offset: int = 0,
    repo: ChatSessionRepository = Depends(get_session_repo),
) -> ResultResponse:
    """获取当前用户的会话列表"""

    user_id = getattr(request.state, USER_ID, None)
    if user_id is None:
        return ResultResponse.failure(2001, "用户未登录")

    if limit <= 0 or limit > 100:
        return ResultResponse.failure(1001, "limit 参数需在 1-100 之间")
    if offset < 0:
        return ResultResponse.failure(1001, "offset 参数不能为负数")

    sessions = await repo.list_user_sessions(user_id, limit, offset)

    # 转换 datetime 为字符串
    data = []
    for s in sessions:
        data.append({
            "session_id": s["session_id"],
            "scene": s["scene"],
            "status": s["status"],
            "create_time": str(s["create_time"]) if s["create_time"] else None,
            "update_time": str(s["update_time"]) if s["update_time"] else None,
        })

    return ResultResponse.success(data=data)


@router.get("/{session_id}/history")
async def get_session_history(
    session_id: str,
    request: Request,
    limit: int = 50,
    offset: int = 0,
    session_repo: ChatSessionRepository = Depends(get_session_repo),
    message_repo: ChatMessageRepository = Depends(get_message_repo),
) -> ResultResponse:
    """获取指定会话的历史消息"""

    user_id = getattr(request.state, USER_ID, None)
    if user_id is None:
        return ResultResponse.failure(2001, "用户未登录")

    if limit <= 0 or limit > 200:
        return ResultResponse.failure(1001, "limit 参数需在 1-200 之间")
    if offset < 0:
        return ResultResponse.failure(1001, "offset 参数不能为负数")

    # 校验会话归属
    session = await session_repo.get_session(session_id)
    if session is None:
        return ResultResponse.failure(2002, "会话不存在")

    if session["user_id"] != user_id:
        return ResultResponse.failure(2003, "无权访问此会话")

    messages = await message_repo.list_session_messages(session_id, limit, offset)

    # 转换 datetime 为字符串
    data = []
    for m in messages:
        data.append({
            "role": m["role"],
            "content": m["content"],
            "intent": m["intent"],
            "tool_name": m["tool_name"],
            "tool_result": m["tool_result"],
            "latency_ms": m["latency_ms"],
            "create_time": str(m["create_time"]) if m["create_time"] else None,
        })

    return ResultResponse.success(data=data)