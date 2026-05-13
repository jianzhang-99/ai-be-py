from __future__ import annotations

"""规范化请求/响应处理的聊天服务。"""

import json
import uuid
from typing import AsyncGenerator

from backend.api.schemas import ChatEvent, ChatRequest, ChatResponse, ChatStreamResponse
from backend.graph.state.agent_state import SceneEnum
from backend.graph.workflows.chat_workflow import ChatWorkflow
from backend.infra.database.repositories.chat_log_repo import ChatLogRepository


class ChatService:
    """聊天端点的应用服务。"""

    def __init__(self):
        self.workflow = ChatWorkflow()
        self._log_repo = ChatLogRepository()

    async def chat_stream(
        self, request: ChatRequest
    ) -> AsyncGenerator[ChatStreamResponse, None]:
        """流式传输规范化的第一阶段聊天事件，并持久化消息。"""

        session_id = request.session_id or str(uuid.uuid4())
        user_id = request.user_id or "anonymous"
        history = [message.model_dump() for message in request.history]
        final_response = ""

        async for event in self.workflow.run_stream(
            user_input=request.message,
            user_id=user_id,
            session_id=session_id,
            history=history,
        ):
            if event.event == "response":
                final_response = str(event.data.get("text", ""))
            yield {
                "event": event.event,
                "data": event.data,
            }

        await self._persist_chat_log(request, session_id, final_response)

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

        result = await self.workflow.run_simple(
            user_input=request.message,
            user_id=user_id,
            session_id=session_id,
            history=history,
        )

        await self._persist_chat_log(request, session_id, result.get("response_text", ""), result=result)

        return ChatResponse(
            message=result.get("response_text", ""),
            session_id=session_id,
            intent=result["intent"].intent if result.get("intent") else None,
        )

    async def chat_ai_be_stream(
        self,
        request: ChatRequest,
        phone: str | None = None,
    ) -> AsyncGenerator[dict[str, str | dict], None]:
        """兼容 Java `/ai/chat` 的 SSE 负载结构。"""

        session_id = request.session_id or str(uuid.uuid4())
        user_id = request.user_id or "anonymous"
        history = [message.model_dump() for message in request.history]
        final_response = ""
        intent_code: str | None = None
        scene_code = request.scene or "TALK"

        async for event in self.workflow.run_stream(
            user_input=request.message,
            user_id=user_id,
            session_id=session_id,
            history=history,
        ):
            if event.event == "intent":
                intent = str(event.data.get("intent") or "")
                intent_code = intent or None
                if intent:
                    scene_code = intent
                yield self._build_ai_be_event(
                    "THINKING",
                    self._build_thinking_message(intent),
                )
                continue

            if event.event == "tool_start":
                yield self._build_ai_be_event(
                    "THINKING",
                    f"正在调用{event.data.get('tool', '工具')}能力，请稍候",
                )
                continue

            if event.event == "response":
                final_response = str(event.data.get("text", ""))
                yield self._build_ai_be_event("TEXT", final_response)
                continue

        await self._persist_chat_log(
            request,
            session_id,
            final_response,
            phone=phone,
            intent_code=intent_code,
            scene_code=scene_code,
        )
        yield self._build_ai_be_event("DONE", None)

    async def _persist_chat_log(
        self,
        request: ChatRequest,
        session_id: str,
        response_text: str,
        phone: str | None = None,
        result: dict | None = None,
        intent_code: str | None = None,
        scene_code: str | None = None,
    ) -> None:
        """保存兼容 Java 版的聊天日志。"""

        user_id_int = self._parse_user_id(request.user_id)
        if user_id_int is None:
            return

        chat_result = result or {}
        intent_obj = chat_result.get("intent")
        resolved_intent_code = intent_code or (intent_obj.intent if intent_obj else None)
        resolved_scene = SceneEnum.normalize(str(scene_code or chat_result.get("scene") or request.scene or "talk"))
        resolved_scene_code = str(SceneEnum.to_model_intent(resolved_scene) or "TALK")
        original_request = {
            "input": request.message,
            "sessionId": session_id,
            "scene": request.scene,
            "model": request.model,
            "appSource": request.app_source,
            "attachments": request.attachments,
        }
        await self._log_repo.save_log(
            session_id=session_id,
            phone=phone or str(user_id_int),
            user_id=user_id_int,
            user_input=request.message,
            ai_response=response_text,
            intent_code=resolved_intent_code,
            intent_name=resolved_intent_code,
            scene_code=resolved_scene_code,
            scene_name=self._scene_name(resolved_scene_code),
            model_name=request.model,
            original_request=json.dumps(original_request, ensure_ascii=False),
        )

    def _scene_name(self, scene_code: str) -> str:
        """将场景编码映射为展示名称。"""

        mapping = {
            "TALK": "闲聊问答",
            "QUERY_WEATHER": "天气查询",
            "QUERY_SHIP": "船舶查询",
            "SAVE_ORDER": "运单录入",
        }
        return mapping.get(scene_code, scene_code)

    def _build_thinking_message(self, intent: str | None) -> str:
        """返回较贴近前端预期的思考提示。"""

        mapping = {
            "QUERY_WEATHER": "正在识别为天气查询并准备检索天气信息",
            "QUERY_SHIP": "正在识别为船舶查询并准备检索船舶信息",
            "SAVE_ORDER": "正在识别为运单场景并提取运单要素",
            "TALK": "正在理解你的问题",
        }
        normalized_intent = SceneEnum.to_model_intent(intent) or intent or ""
        return mapping.get(normalized_intent, "正在处理你的问题")

    def _build_ai_be_event(self, event_type: str, content: object) -> dict[str, str]:
        """构造 AI-BE 兼容的 SSE 数据负载。"""

        return {"type": event_type, "content": content}
