from __future__ import annotations

"""会话管理 API 路由"""

from fastapi import APIRouter, Depends, Request

from backend.auth.constants import USER_ID
from backend.auth.schemas import ResultResponse
from backend.infra.database.repositories.chat_log_repo import ChatLogRepository
from backend.infra.database.repositories.sys_user_repository import SysUserRepository

router = APIRouter(prefix="/api/session", tags=["session"])


def get_chat_log_repo() -> ChatLogRepository:
    """返回聊天日志仓储实例"""
    return ChatLogRepository()


@router.get("/list")
async def list_sessions(
    request: Request,
    limit: int = 20,
    offset: int = 0,
    repo: ChatLogRepository = Depends(get_chat_log_repo),
) -> ResultResponse:
    """获取当前用户的会话列表"""

    user_id = getattr(request.state, USER_ID, None)
    if user_id is None:
        return ResultResponse.failure(2001, "用户未登录")

    if limit <= 0 or limit > 100:
        return ResultResponse.failure(1001, "limit 参数需在 1-100 之间")
    if offset < 0:
        return ResultResponse.failure(1001, "offset 参数不能为负数")

    # 通过 user_id 查手机号
    user_repo = SysUserRepository()
    user = await user_repo.find_by_id(user_id)
    if user is None:
        return ResultResponse.failure(2001, "用户未登录")

    sessions = await repo.list_by_phone(user.phone, limit=limit, offset=offset)

    # 转换 datetime 为字符串
    data = []
    for s in sessions:
        data.append({
            "session_id": s.get("session_id"),
            "scene_code": s.get("scene_code"),
            "scene_name": s.get("scene_name"),
            "last_time": str(s.get("last_time")) if s.get("last_time") else None,
            "msg_count": s.get("msg_count"),
        })

    return ResultResponse.success(data=data)


@router.get("/{session_id}/history")
async def get_session_history(
    session_id: str,
    request: Request,
    limit: int = 50,
    offset: int = 0,
    repo: ChatLogRepository = Depends(get_chat_log_repo),
) -> ResultResponse:
    """获取指定会话的历史消息"""

    user_id = getattr(request.state, USER_ID, None)
    if user_id is None:
        return ResultResponse.failure(2001, "用户未登录")

    if limit <= 0 or limit > 200:
        return ResultResponse.failure(1001, "limit 参数需在 1-200 之间")
    if offset < 0:
        return ResultResponse.failure(1001, "offset 参数不能为负数")

    messages = await repo.list_by_session_id(session_id, limit=limit, offset=offset)

    # 转换 datetime 为字符串
    data = []
    for m in messages:
        data.append({
            "user_input": m.get("user_input"),
            "ai_response": m.get("ai_response"),
            "intent_code": m.get("intent_code"),
            "intent_name": m.get("intent_name"),
            "scene_code": m.get("scene_code"),
            "create_time": str(m.get("create_time")) if m.get("create_time") else None,
        })

    return ResultResponse.success(data=data)