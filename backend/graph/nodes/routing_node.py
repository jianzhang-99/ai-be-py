from __future__ import annotations

"""路由节点，用于显式化工作流分支。"""

from backend.graph.state.agent_state import AgentState


class RoutingNode:
    """在分支前标准化场景信息。"""

    def __init__(self):
        pass

    async def entrypoint(self, state: AgentState) -> AgentState:
        """将当前场景持久化，以便下游节点使用。"""

        scene = state.get("scene", "TALK")
        return {"scene": scene}
