from __future__ import annotations

"""基于LangGraph的第一阶段聊天工作流。"""

import uuid
from typing import AsyncGenerator

try:
    from langgraph.graph import END, StateGraph
except ModuleNotFoundError:  # pragma: no cover - exercised only in lightweight envs
    END = "__end__"
    StateGraph = None

from backend.api.schemas import ChatEvent
from backend.graph.nodes.intent_node import IntentNode
from backend.graph.nodes.memory_node import MemoryNode
from backend.graph.nodes.response_node import ResponseNode
from backend.graph.nodes.routing_node import RoutingNode
from backend.graph.nodes.tool_node import ToolNode
from backend.graph.state.agent_state import AgentState, SceneEnum


class ChatWorkflow:
    """聊天服务使用的工作流包装器。"""

    def __init__(self):
        self.intent_node = IntentNode()
        self.routing_node = RoutingNode()
        self.tool_node = ToolNode()
        self.response_node = ResponseNode()
        self.memory_node = MemoryNode()
        self.graph = self._build_graph()

    def _build_graph(self):
        """构建最小的第一阶段图。"""

        if StateGraph is None:
            return None

        workflow = StateGraph(AgentState)

        workflow.add_node("intent", self.intent_node.entrypoint)
        workflow.add_node("routing", self.routing_node.entrypoint)
        workflow.add_node("tool", self.tool_node.entrypoint)
        workflow.add_node("response", self.response_node.entrypoint)
        workflow.add_node("memory", self.memory_node.entrypoint)
        workflow.set_entry_point("intent")
        workflow.add_edge("intent", "routing")
        workflow.add_conditional_edges(
            "routing",
            self._route_decision,
            {
                "tool": "tool",
                "response_direct": "response",
            },
        )
        workflow.add_edge("tool", "response")
        workflow.add_edge("response", "memory")
        workflow.add_edge("memory", END)

        return workflow.compile()

    def _route_decision(self, state: AgentState) -> str:
        """决定下一步是否需要调用工具。"""

        scene = state.get("scene")
        if scene in {SceneEnum.QUERY_WEATHER, SceneEnum.QUERY_SHIP, SceneEnum.SAVE_ORDER}:
            return "tool"
        return "response_direct"

    def build_initial_state(
        self,
        user_input: str,
        user_id: str,
        session_id: str,
        history: list[dict[str, str]],
    ) -> AgentState:
        """从入站请求创建共享的工作流状态。"""

        return {
            "request_id": str(uuid.uuid4()),
            "user_input": user_input,
            "user_id": user_id,
            "session_id": session_id,
            "history": history,
            "working_memory": {},
            "tool_calls": [],
            "tool_name": None,
            "tool_result": None,
            "response_text": None,
            "error": None,
        }

    async def run_stream(
        self,
        user_input: str,
        user_id: str,
        session_id: str,
        history: list,
    ) -> AsyncGenerator[ChatEvent, None]:
        """按文档顺序生成第一阶段的SSE事件。"""

        state = self.build_initial_state(user_input, user_id, session_id, history)
        state.update(await self.intent_node.entrypoint(state))

        intent = state.get("intent")
        if intent is not None:
            yield ChatEvent(
                event="intent",
                data={
                    "intent": intent.intent,
                    "confidence": intent.confidence,
                    "method": intent.method,
                },
            )

        state.update(await self.routing_node.entrypoint(state))

        if self._route_decision(state) == "tool":
            tool_name = self.tool_node.registry.get_tool_name(state.get("scene"))
            if tool_name is not None:
                yield ChatEvent(event="tool_start", data={"tool": tool_name})
            state.update(await self.tool_node.entrypoint(state))
            if state.get("tool_result") is not None:
                yield ChatEvent(
                    event="tool_result",
                    data={
                        "tool": state.get("tool_name"),
                        "result": state.get("tool_result"),
                    },
                )

        state.update(await self.response_node.entrypoint(state))
        if state.get("response_text"):
            yield ChatEvent(event="response", data={"text": state["response_text"]})

        state.update(await self.memory_node.entrypoint(state))
        yield ChatEvent(event="done", data={"session_id": session_id})

    async def run_simple(
        self,
        user_input: str,
        user_id: str,
        session_id: str,
        history: list,
    ) -> AgentState:
        """运行LangGraph路径并返回最终状态。"""

        initial_state = self.build_initial_state(user_input, user_id, session_id, history)
        if self.graph is None:
            state = initial_state
            state.update(await self.intent_node.entrypoint(state))
            state.update(await self.routing_node.entrypoint(state))
            if self._route_decision(state) == "tool":
                state.update(await self.tool_node.entrypoint(state))
            state.update(await self.response_node.entrypoint(state))
            state.update(await self.memory_node.entrypoint(state))
            return state

        return await self.graph.ainvoke(initial_state)
