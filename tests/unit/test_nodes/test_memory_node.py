from __future__ import annotations

import pytest

from backend.graph.nodes.memory_node import MemoryNode
from backend.graph.state.agent_state import SceneEnum


class StubSessionStore:
    def __init__(self) -> None:
        self.saved: dict[str, dict] = {}

    async def set_session_data(self, session_id: str, data: dict, ttl_hours: int = 24) -> None:
        self.saved[session_id] = data


class TestMemoryNode:
    @pytest.mark.asyncio
    async def test_memory_node_appends_history_and_writes_done_state(self) -> None:
        store = StubSessionStore()
        node = MemoryNode(session_store=store)

        result = await node.entrypoint(
            {
                "session_id": "sess-001",
                "scene": SceneEnum.QUERY_WEATHER,
                "user_input": "南京天气",
                "response_text": "南京今天多云",
                "history": [],
                "working_memory": {},
            }
        )

        assert result["history"] == [
            {"role": "user", "content": "南京天气"},
            {"role": "assistant", "content": "南京今天多云"},
        ]
        assert store.saved["sess-001"]["current_scene"] == SceneEnum.QUERY_WEATHER
        assert store.saved["sess-001"]["state"] == "DONE"

    @pytest.mark.asyncio
    async def test_memory_node_marks_save_order_preview_as_waiting(self) -> None:
        store = StubSessionStore()
        node = MemoryNode(session_store=store)

        await node.entrypoint(
            {
                "session_id": "sess-002",
                "scene": SceneEnum.SAVE_ORDER,
                "user_input": "武汉装煤5000吨到南京",
                "response_text": "已提取运单预览，如需提交请说确认提交",
                "history": [],
                "working_memory": {},
                "tool_result": {
                    "_detail": {
                        "loading_port": "武汉",
                        "discharge_port": "南京",
                    },
                    "summary": "已提取运单预览",
                },
            }
        )

        assert store.saved["sess-002"]["current_scene"] == SceneEnum.SAVE_ORDER
        assert store.saved["sess-002"]["state"] == SceneEnum.WAITING_USER
        assert store.saved["sess-002"]["pending_order"] == {
            "loading_port": "武汉",
            "discharge_port": "南京",
        }
