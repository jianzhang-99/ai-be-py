from __future__ import annotations

"""聊天工作流使用的共享状态定义。"""

from typing import Any, Optional, TypedDict

from pydantic import BaseModel


class IntentInfo(BaseModel):
    """意图识别结果。"""

    intent: str
    confidence: float
    method: str


class IntentCandidate(BaseModel):
    """意图候选，支持 TopK 展示与裁决。"""

    intent: str
    score: float
    source: str


class SlotInfo(BaseModel):
    """标准化后的槽位值。"""

    name: str
    values: list[str]
    source: str
    confidence: float = 0.0


class EntityCandidate(BaseModel):
    """槽位对应的实体候选。"""

    slot_name: str
    value: str
    entity_type: str
    score: float


class ClarifyInfo(BaseModel):
    """澄清裁决结果。"""

    need_clarify: bool
    question: Optional[str] = None
    reason: Optional[str] = None


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
    chat_summaries: str
    user_profile: str
    memory_hint: str
    intent: Optional[IntentInfo]
    scene: Optional[str]
    intent_candidates: list[IntentCandidate]
    slots: dict[str, SlotInfo]
    entity_candidates: list[EntityCandidate]
    clarify: Optional[ClarifyInfo]
    final_intent: Optional[str]
    final_slots: dict[str, list[str]]
    tool_name: Optional[str]
    tool_result: Optional[Any]
    tool_calls: list[ToolCall]
    response_text: Optional[str]
    error: Optional[str]


class SceneEnum:
    """支持的场景枚举，与 Java 版 ai-be 保持一致。

    共 14 种意图标签。
    """

    # 运单相关
    SAVE_ORDER = "save_order"
    DISPATCH_MONITOR = "dispatch_monitor"
    QUERY_ORDER = "query_order"

    # 船舶相关
    QUERY_SHIP = "query_ship"
    FIND_SHIP = "find_ship"
    QUERY_SHIP_INFO = "query_ship_info"

    # 天气水位
    QUERY_WEATHER = "query_weather"
    QUERY_WATER_LEVEL = "query_water_level"

    # 运价相关
    QUERY_FREIGHT = "query_freight"
    QUERY_OIL_STATION = "query_oil_station"

    # 其他
    IMAGE_OCR = "image_ocr"
    FEEDBACK = "feedback"
    DOC_QA = "doc_qa"
    TALK = "talk"

    # 内部状态
    WAITING_USER = "WAITING_USER"

    @classmethod
    def values(cls) -> list[str]:
        return [
            cls.SAVE_ORDER,
            cls.DISPATCH_MONITOR,
            cls.QUERY_ORDER,
            cls.QUERY_SHIP,
            cls.FIND_SHIP,
            cls.QUERY_SHIP_INFO,
            cls.QUERY_WEATHER,
            cls.QUERY_WATER_LEVEL,
            cls.QUERY_FREIGHT,
            cls.QUERY_OIL_STATION,
            cls.IMAGE_OCR,
            cls.FEEDBACK,
            cls.DOC_QA,
            cls.TALK,
            cls.WAITING_USER,
        ]

    @classmethod
    def model_values(cls) -> list[str]:
        """训练模型使用的大写标签。"""

        return [value.upper() for value in cls.values() if value != cls.WAITING_USER]

    @classmethod
    def normalize(cls, value: str | None) -> str | None:
        """统一将大小写混杂的场景值转换为运行时小写标签。"""

        if not value:
            return value
        lowered = value.strip().lower()
        if lowered == cls.WAITING_USER.lower():
            return cls.WAITING_USER
        valid_values = set(cls.values())
        return lowered if lowered in valid_values else value

    @classmethod
    def to_model_intent(cls, value: str | None) -> str | None:
        """将运行时小写标签转换为训练模型的大写标签。"""

        normalized = cls.normalize(value)
        if not normalized or normalized == cls.WAITING_USER:
            return normalized
        return normalized.upper()

    @classmethod
    def from_model_intent(cls, value: str | None) -> str | None:
        """将训练模型的大写标签转换为运行时小写标签。"""

        return cls.normalize(value)
