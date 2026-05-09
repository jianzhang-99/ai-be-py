from __future__ import annotations

"""聊天工作流使用的共享状态定义。"""

from typing import Any, Optional, TypedDict

from pydantic import BaseModel


class IntentInfo(BaseModel):
    """意图识别结果。"""

    intent: str
    confidence: float
    method: str


class ToolCall(BaseModel):
    """存储在工作流状态中的工具执行记录。"""

    tool_name: str
    args: dict[str, Any]
    result: Optional[Any] = None
    error: Optional[str] = None


class AgentState(TypedDict, total=False):
    """所有节点共享的最小阶段一工作流状态。"""

    request_id: str
    user_input: str
    user_id: str
    session_id: str
    history: list[dict[str, str]]
    working_memory: dict[str, Any]
    intent: Optional[IntentInfo]
    scene: Optional[str]
    tool_name: Optional[str]
    tool_result: Optional[Any]
    tool_calls: list[ToolCall]
    response_text: Optional[str]
    error: Optional[str]


class SceneEnum:
    """支持的阶段一场景。"""

    TALK = "TALK"
    QUERY_WEATHER = "QUERY_WEATHER"
    QUERY_SHIP = "QUERY_SHIP"
    SAVE_ORDER = "SAVE_ORDER"
    WAITING_USER = "WAITING_USER"

    @classmethod
    def values(cls) -> list[str]:
        return [
            cls.TALK,
            cls.QUERY_WEATHER,
            cls.QUERY_SHIP,
            cls.SAVE_ORDER,
            cls.WAITING_USER,
        ]
