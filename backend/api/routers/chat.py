from __future__ import annotations

"""阶段一 MVP 的聊天 API 路由。"""

import json

from fastapi import APIRouter, Depends
from sse_starlette.sse import EventSourceResponse

from backend.api.deps import get_chat_service
from backend.api.schemas import ChatRequest, ChatResponse
from backend.services.chat_service import ChatService

router = APIRouter(prefix="/api/chat", tags=["chat"])

@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service),
):
    """根据约定的 SSE 事件契约返回聊天事件。"""

    async def event_generator():
        async for event in service.chat_stream(request):
            yield {
                "event": event.event,
                "data": json.dumps(event.data, ensure_ascii=False),
            }

    return EventSourceResponse(event_generator())


@router.post("", response_model=ChatResponse)
async def chat_simple(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service),
):
    """返回聚合的聊天结果。"""

    return await service.chat_simple(request)
