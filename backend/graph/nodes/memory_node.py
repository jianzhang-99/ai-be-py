from __future__ import annotations

"""记忆节点，将最新一轮对话追加到本地历史记录中。"""

from backend.graph.state.agent_state import AgentState


class MemoryNode:
    """将已完成的对话轮次追加到内存历史记录中。"""

    def __init__(self):
        pass

    async def entrypoint(self, state: AgentState) -> AgentState:
        """更新返回给调用者的历史记录列表。"""

        history = state.get("history", [])
        user_input = state.get("user_input", "")
        response = state.get("response_text", "")

        history = history + [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": response},
        ]

        return {"history": history}
