from __future__ import annotations

"""规范化请求/响应处理的聊天服务。"""

import uuid
from typing import AsyncGenerator

from backend.api.schemas import ChatRequest, ChatResponse, ChatStreamResponse
from backend.graph.workflows.chat_workflow import ChatWorkflow


class ChatService:
    """聊天端点的应用服务。"""

    def __init__(self):
        self.workflow = ChatWorkflow()

    async def chat_stream(
        self, request: ChatRequest
    ) -> AsyncGenerator[ChatStreamResponse, None]:
        """流式传输规范化的第一阶段聊天事件。"""

        session_id = request.session_id or str(uuid.uuid4())
        history = [message.model_dump() for message in request.history]

        async for event in self.workflow.run_stream(
            user_input=request.message,
            user_id=request.user_id or "anonymous",
            session_id=session_id,
            history=history,
        ):
            yield event

    async def chat_simple(
        self, request: ChatRequest
    ) -> ChatResponse:
        """返回非流式 API 的聚合响应。"""

        session_id = request.session_id or str(uuid.uuid4())
        history = [message.model_dump() for message in request.history]

        result = await self.workflow.run_simple(
            user_input=request.message,
            user_id=request.user_id or "anonymous",
            session_id=session_id,
            history=history,
        )

        return ChatResponse(
            message=result.get("response_text", ""),
            session_id=session_id,
            intent=result["intent"].intent if result.get("intent") else None,
        )
