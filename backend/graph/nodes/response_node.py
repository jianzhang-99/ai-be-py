from __future__ import annotations

"""最终响应生成节点。"""

from typing import Optional

from backend.graph.state.agent_state import AgentState
from backend.infra.llm.client import LLMClient


class ResponseNode:
    """构建面向用户的最终答案。"""

    def __init__(self):
        self.llm = LLMClient()

    async def entrypoint(self, state: AgentState) -> AgentState:
        """返回基于工具的摘要或纯 LLM 回答。"""
        user_input = state.get("user_input", "")
        history = state.get("history", [])
        scene = state.get("scene")
        tool_result = state.get("tool_result")

        if tool_result:
            return {"response_text": self._build_tool_response(scene, tool_result)}

        response = await self.llm.chat(
            system_prompt=self._build_system_prompt(scene),
            user_message=user_input,
            messages=history,
        )
        return {"response_text": response}

    def _build_system_prompt(self, scene: Optional[str]) -> str:
        """为第一阶段 TALK 流程构建轻量级系统提示词。"""

        base_prompt = """你是航运领域的智能助手小吨，擅长回答航运相关问题。
请用友好、专业的语气回答用户问题。"""

        scene_prompts = {
            "TALK": "你是一个友好的航运助手，可以闲聊也可以回答专业问题。",
            "QUERY_WEATHER": "你是一个天气预报助手，根据查询结果告诉用户天气信息。",
            "QUERY_SHIP": "你是一个船舶查询助手，请简明转述工具返回的船舶信息。",
            "SAVE_ORDER": "你是一个运单助手，请清晰展示抽取出的运单预览。",
        }

        return base_prompt + "\n\n" + scene_prompts.get(scene, "")

    def _build_tool_response(self, scene: Optional[str], tool_result: dict) -> str:
        """将标准化的工具输出渲染为最终消息。"""

        if scene == "QUERY_WEATHER":
            return tool_result["summary"]
        if scene == "QUERY_SHIP":
            return tool_result["summary"]
        if scene == "SAVE_ORDER":
            return tool_result["summary"]
        return str(tool_result)
