from __future__ import annotations

"""占位符 RAG 节点，保留用于后续阶段。"""

from backend.graph.state.agent_state import AgentState


class RAGNode:
    """为未来工作流扩展保留的空操作占位符。"""

    def __init__(self):
        pass

    async def entrypoint(self, state: AgentState) -> AgentState:
        """在后续阶段引入 RAG 之前，返回空的更新。"""

        return {}
