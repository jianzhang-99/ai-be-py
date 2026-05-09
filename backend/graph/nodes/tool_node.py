from __future__ import annotations

"""工具执行节点。"""

from backend.graph.state.agent_state import AgentState, ToolCall
from backend.tools.registry import ToolRegistry


class ToolNode:
    """调用与当前场景对应的工具。"""

    def __init__(self):
        self.registry = ToolRegistry()

    async def entrypoint(self, state: AgentState) -> AgentState:
        """为支持的场景执行第一阶段工具流程。"""

        scene = state.get("scene")
        user_input = state.get("user_input", "")
        tool_name = self.registry.get_tool_name(scene)

        if tool_name is None:
            return {"tool_calls": [], "tool_name": None, "tool_result": None}

        payload = self._build_payload(tool_name, user_input)
        result = await self.registry.call_tool(tool_name, payload)
        tool_call = ToolCall(tool_name=tool_name, args=payload, result=result)

        return {"tool_calls": [tool_call], "tool_name": tool_name, "tool_result": result}

    def _build_payload(self, tool_name: str, user_input: str) -> dict[str, str]:
        """为每种工具类型构建最小化的标准化负载。"""

        if tool_name == "weather":
            return {"city": self._extract_city(user_input)}
        if tool_name == "ship":
            return {"query": user_input}
        if tool_name == "order":
            return {"text": user_input}
        return {"query": user_input}

    def _extract_city(self, text: str) -> str:
        """从天气问题中提取城市标识。"""

        city = text.replace("帮我查一下", "").replace("天气", "").strip(" ，。")
        return city or "武汉"
