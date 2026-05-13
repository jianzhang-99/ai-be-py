from __future__ import annotations

"""阶段一 MVP 的 API 请求和响应模式。"""

from typing import Any, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    """对话历史中的单个回合。"""

    role: str = Field(..., description="Message role: user / assistant / system")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """同步和流式聊天 API 共享的统一请求体。"""

    message: str = Field(..., min_length=1, description="Current user message")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    history: list[Message] = Field(default_factory=list, description="Conversation history")
    scene: Optional[str] = Field(default=None, description="Requested scene code")
    model: Optional[str] = Field(default=None, description="Requested model name")
    app_source: Optional[str] = Field(default=None, description="Client app source")
    attachments: list[dict[str, Any]] = Field(default_factory=list, description="Uploaded attachments")


class ChatResponse(BaseModel):
    """非流式聊天响应。"""

    message: str
    session_id: str
    intent: Optional[str] = None


class ChatEvent(BaseModel):
    """单个 SSE 负载。"""

    event: str
    data: dict[str, Any]


ChatStreamRequest = ChatRequest
ChatStreamResponse = ChatEvent


class PipelineAnalysisResponse(BaseModel):
    """意图判定流水线调试结果。"""

    preprocessed_input: str
    intent_candidates: list[dict[str, Any]]
    slots: dict[str, dict[str, Any]]
    entity_candidates: list[dict[str, Any]]
    clarify: dict[str, Any]
    final_intent: Optional[str] = None
    final_slots: dict[str, list[str]]
    can_route: bool
    route_tool: Optional[str] = None
