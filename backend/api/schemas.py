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
