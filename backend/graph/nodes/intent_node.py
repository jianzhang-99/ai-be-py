from __future__ import annotations

"""意图识别节点。"""

from backend.graph.state.agent_state import AgentState, IntentInfo, SceneEnum
from backend.services.intent_service import IntentService


class IntentNode:
    """在工作流分支之前解析场景。"""

    def __init__(self):
        self.service = IntentService()

    async def entrypoint(self, state: AgentState) -> AgentState:
        """按最小链路执行：规则预处理 -> 专属意图模型。"""
        user_input = state["user_input"]
        working_memory = state.get("working_memory", {})
        history = state.get("history", [])

        intent_result = await self.service.recognize_by_rule(user_input, working_memory)
        if intent_result:
            return {
                "intent": IntentInfo(
                    intent=intent_result["intent"],
                    confidence=float(intent_result.get("confidence", 1.0)),
                    method=str(intent_result.get("method", "rule")),
                ),
                "scene": intent_result["intent"],
            }

        intent_result = await self.service.recognize_by_model(user_input, history)
        if intent_result:
            return {
                "intent": IntentInfo(
                    intent=intent_result["intent"],
                    confidence=float(intent_result.get("confidence", 0.8)),
                    method=str(intent_result.get("method", "custom_model")),
                ),
                "scene": intent_result["intent"],
            }

        return {
            "intent": IntentInfo(
                intent=SceneEnum.TALK,
                confidence=0.5,
                method="guard",
            ),
            "scene": SceneEnum.TALK,
        }
