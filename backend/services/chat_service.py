from __future__ import annotations

"""规范化请求/响应处理的聊天服务。"""

import asyncio
import uuid
from typing import AsyncGenerator

from backend.api.schemas import ChatEvent, ChatRequest, ChatResponse, ChatStreamResponse
from backend.graph.workflows.chat_workflow import ChatWorkflow
from backend.infra.database.repositories.chat_message_repo import ChatMessageRepository
from backend.infra.database.repositories.chat_session_repo import ChatSessionRepository


class ChatService:
    """聊天端点的应用服务。"""

    def __init__(self):
        self.workflow = ChatWorkflow()
        self._session_repo = ChatSessionRepository()
        self._message_repo = ChatMessageRepository()

    async def chat_stream(
        self, request: ChatRequest
    ) -> AsyncGenerator[ChatStreamResponse, None]:
        """流式传输规范化的第一阶段聊天事件，并持久化消息。"""

        session_id = request.session_id or str(uuid.uuid4())
        user_id = request.user_id or "anonymous"
        history = [message.model_dump() for message in request.history]

        # 创建会话（若不存在）
        user_id_int = self._parse_user_id(user_id)
        if user_id_int is not None:
            existing = await self._session_repo.get_session(session_id)
            if existing is None:
                await self._session_repo.create_session(
                    user_id=user_id_int,
                    session_id=session_id,
                    scene="DEFAULT",
                )

        # 记录消息开始时间用于计算延迟
        msg_start = asyncio.get_event_loop().time()
        user_content = request.message

        # 保存用户消息
        if user_id_int is not None:
            await self._message_repo.save_message(
                session_id=session_id,
                role="user",
                content=user_content,
            )

        # 执行工作流
        assistant_content = ""
        intent_name = None
        tool_name_used = None
        tool_result_str = None
        assistant_start = asyncio.get_event_loop().time()

        async for event in self.workflow.run_stream(
            user_input=request.message,
            user_id=user_id,
            session_id=session_id,
            history=history,
        ):
            yield {
                "event": event.event,
                "data": event.data,
            }

            # 收集完整响应和工具信息
            if event.event == "intent" and event.data:
                intent_name = event.data.get("intent")
            elif event.event == "tool_start" and event.data:
                tool_name_used = event.data.get("tool")
            elif event.event == "tool_result" and event.data:
                tool_result_str = str(event.data.get("result", ""))
            elif event.event == "response" and event.data:
                assistant_content = event.data.get("text", "")
            elif event.event == "done":
                # 流结束，计算延迟并保存助手消息
                assistant_end = asyncio.get_event_loop().time()
                latency_ms = int((assistant_end - assistant_start) * 1000)
                if user_id_int is not None and assistant_content:
                    await self._message_repo.save_message(
                        session_id=session_id,
                        role="assistant",
                        content=assistant_content,
                        intent=intent_name,
                        tool_name=tool_name_used,
                        tool_result=tool_result_str,
                        latency_ms=latency_ms,
                    )

    def _parse_user_id(self, user_id: str) -> int | None:
        """尝试将 user_id 解析为整数"""

        try:
            return int(user_id)
        except (ValueError, TypeError):
            return None

    async def chat_simple(
        self, request: ChatRequest
    ) -> ChatResponse:
        """返回非流式 API 的聚合响应。"""

        session_id = request.session_id or str(uuid.uuid4())
        user_id = request.user_id or "anonymous"
        history = [message.model_dump() for message in request.history]

        # 创建会话
        user_id_int = self._parse_user_id(user_id)
        if user_id_int is not None:
            existing = await self._session_repo.get_session(session_id)
            if existing is None:
                await self._session_repo.create_session(
                    user_id=user_id_int,
                    session_id=session_id,
                    scene="DEFAULT",
                )

        # 保存用户消息
        if user_id_int is not None:
            await self._message_repo.save_message(
                session_id=session_id,
                role="user",
                content=request.message,
            )

        result = await self.workflow.run_simple(
            user_input=request.message,
            user_id=user_id,
            session_id=session_id,
            history=history,
        )

        # 保存助手消息
        if user_id_int is not None and result.get("response_text"):
            intent_obj = result.get("intent")
            await self._message_repo.save_message(
                session_id=session_id,
                role="assistant",
                content=result["response_text"],
                intent=intent_obj.intent if intent_obj else None,
                tool_name=result.get("tool_name"),
                tool_result=str(result.get("tool_result", "")) if result.get("tool_result") else None,
                latency_ms=None,
            )

        return ChatResponse(
            message=result.get("response_text", ""),
            session_id=session_id,
            intent=result["intent"].intent if result.get("intent") else None,
        )
