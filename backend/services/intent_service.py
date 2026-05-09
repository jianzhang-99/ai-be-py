from __future__ import annotations

"""基于规则优先的意图识别，用于第一阶段。"""

from typing import Any, Optional

from backend.infra.llm.client import LLMClient
from backend.graph.state.agent_state import SceneEnum


class IntentService:
    """意图识别服务。"""

    KEYWORD_RULES = (
        ("天气", SceneEnum.QUERY_WEATHER),
        ("查船", SceneEnum.QUERY_SHIP),
        ("船舶", SceneEnum.QUERY_SHIP),
        ("运单", SceneEnum.SAVE_ORDER),
        ("录单", SceneEnum.SAVE_ORDER),
    )

    def __init__(self):
        self.llm = LLMClient()

    async def recognize_by_rule(
        self,
        user_input: str,
        working_memory: dict[str, Any],
    ) -> Optional[dict[str, str]]:
        """当第一阶段规则匹配时，返回确定性的意图。"""

        for keyword, intent in self.KEYWORD_RULES:
            if keyword in user_input:
                return {"intent": intent}

        previous_scene = working_memory.get("current_scene")
        if previous_scene in {SceneEnum.QUERY_SHIP, SceneEnum.SAVE_ORDER}:
            if working_memory.get("state") == SceneEnum.WAITING_USER:
                return {"intent": previous_scene}

        return None

    async def recognize_by_llm(
        self,
        user_input: str,
        working_memory: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        """当规则匹配失败时，使用 LLM 作为后备分类器。"""

        prompt = f"""用户输入: {user_input}
当前场景: {working_memory.get('current_scene', '未知')}
状态: {working_memory.get('state', '正常')}

请判断用户意图，可选场景: TALK, QUERY_WEATHER, QUERY_SHIP, SAVE_ORDER

只输出意图名称，不要其他内容。"""

        try:
            result = await self.llm.chat(
                system_prompt="你是一个意图识别助手，根据用户输入判断意图。",
                user_message=prompt,
            )
            intent = result.strip()
            if intent in SceneEnum.values():
                return {"intent": intent, "confidence": 0.8}
        except Exception:
            return None

        return None
