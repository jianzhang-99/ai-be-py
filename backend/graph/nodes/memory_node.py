from __future__ import annotations

"""记忆节点，将最新一轮对话追加到本地历史记录中。"""

from backend.graph.state.agent_state import AgentState
from backend.graph.state.agent_state import SceneEnum
from backend.infra.cache.session_store import SessionStore, get_session_store


class MemoryNode:
    """将已完成的对话轮次追加到内存历史记录中。"""

    def __init__(self, session_store: SessionStore | None = None):
        self.session_store = session_store or get_session_store()

    async def entrypoint(self, state: AgentState) -> AgentState:
        """更新返回给调用者的历史记录列表。"""

        history = state.get("history", [])
        user_input = state.get("user_input", "")
        response = state.get("response_text", "")

        additions = []
        if user_input:
            additions.append({"role": "user", "content": user_input})
        if response:
            additions.append({"role": "assistant", "content": response})
        history = history + additions

        await self._write_working_memory(state)
        return {"history": history}

    async def _write_working_memory(self, state: AgentState) -> None:
        session_id = state.get("session_id")
        if not session_id:
            return

        working_memory = dict(state.get("working_memory", {}))
        scene = state.get("scene")
        tool_result = state.get("tool_result")
        response = state.get("response_text", "")

        if scene:
            working_memory["current_scene"] = scene

        waiting_for_user = False
        if scene == SceneEnum.SAVE_ORDER and isinstance(tool_result, dict):
            waiting_for_user = "_detail" in tool_result and "submit_result" not in tool_result
            if waiting_for_user:
                working_memory["pending_order"] = tool_result.get("_detail", {})

        working_memory["state"] = SceneEnum.WAITING_USER if waiting_for_user else "DONE"
        if response:
            working_memory["last_response"] = response
        if state.get("user_input"):
            working_memory["last_user_input"] = state["user_input"]

        try:
            await self.session_store.set_session_data(session_id, working_memory)
        except Exception:
            return
