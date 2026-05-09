from __future__ import annotations

"""意图识别节点。"""

from backend.graph.state.agent_state import AgentState, IntentInfo
from backend.services.intent_service import IntentService


class IntentNode:
    """在工作流分支之前解析场景。"""

    def __init__(self):
        self.service = IntentService()

    async def entrypoint(self, state: AgentState) -> AgentState:
        """优先使用规则识别，LLM 作为备选方案。"""
        user_input = state["user_input"]
        working_memory = state.get("working_memory", {})

        # Layer 1: 规则网关
        intent_result = await self.service.recognize_by_rule(
            user_input, working_memory
        )
        if intent_result:
            return {
                "intent": IntentInfo(
                    intent=intent_result["intent"],
                    confidence=1.0,
                    method="rule",
                ),
                "scene": intent_result["intent"],
            }

        # Layer 2: LLM 分类
        intent_result = await self.service.recognize_by_llm(
            user_input, working_memory
        )
        if intent_result:
            return {
                "intent": IntentInfo(
                    intent=intent_result["intent"],
                    confidence=intent_result.get("confidence", 0.8),
                    method="llm",
                ),
                "scene": intent_result["intent"],
            }

        # Layer 3: 默认 fallback
        return {
            "intent": IntentInfo(
                intent="TALK",
                confidence=0.5,
                method="guard",
            ),
            "scene": "TALK",
        }
