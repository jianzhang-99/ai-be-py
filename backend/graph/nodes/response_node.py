from __future__ import annotations

"""最终响应生成节点。"""

from typing import Optional

from backend.graph.state.agent_state import AgentState, SceneEnum
from backend.infra.llm.client import LLMClient
from backend.infra.llm.prompt_loader import talk_system_prompt


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

        # 使用 Java 版迁移过来的 talk_system 提示词
        system_prompt, user_prompt = await talk_system_prompt(
            user_profile=state.get("user_profile", ""),
            chat_summaries=state.get("chat_summaries", ""),
            memory_hint=state.get("memory_hint", ""),
        )

        # user_prompt 包含用户画像、历史摘要等信息
        full_prompt = f"{user_prompt}\n\n用户输入：{user_input}"

        response = await self.llm.chat(
            system_prompt=system_prompt,
            user_message=full_prompt,
            messages=history,
        )
        return {"response_text": response}

    def _build_tool_response(self, scene: Optional[str], tool_result: dict) -> str:
        """将标准化的工具输出渲染为最终消息。"""

        if scene == SceneEnum.QUERY_WEATHER:
            return tool_result["summary"]
        if scene == SceneEnum.QUERY_SHIP:
            return tool_result["summary"]
        if scene == SceneEnum.SAVE_ORDER:
            return tool_result["summary"]
        return str(tool_result)
