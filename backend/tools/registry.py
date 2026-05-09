from __future__ import annotations

"""第一阶段工作流节点使用的静态工具注册表。"""

from collections.abc import Awaitable, Callable
from typing import Any, Optional

from backend.tools.order import OrderTool
from backend.tools.ship import ShipTool
from backend.tools.weather import WeatherTool

ToolHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


class ToolRegistry:
    """通过简单的内存注册表解析和执行工具。"""

    SCENE_TO_TOOL = {
        "QUERY_WEATHER": "weather",
        "QUERY_SHIP": "ship",
        "SAVE_ORDER": "order",
    }

    def __init__(self) -> None:
        self._tools: dict[str, ToolHandler] = {
            "weather": WeatherTool().run,
            "ship": ShipTool().run,
            "order": OrderTool().run,
        }

    def get_tool_name(self, scene: Optional[str]) -> Optional[str]:
        """返回映射到场景的工具名称。"""

        if scene is None:
            return None
        return self.SCENE_TO_TOOL.get(scene)

    def has_tool(self, tool_name: str) -> bool:
        """检查注册表是否包含给定工具。"""

        return tool_name in self._tools

    async def call_tool(self, tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        """执行工具并返回其标准化结果。"""

        if tool_name not in self._tools:
            raise ValueError(f"Unknown tool: {tool_name}")
        return await self._tools[tool_name](payload)
